"""
Continuous batching scheduler for LLM inference (Phase 4).

Implements iteration-level scheduling with waiting and running queues, FCFS admission,
watermark-based capacity checks, and prefill vs. decode batch separation.
Includes LIFO preemption with re-prefill for memory exhaustion.

Block-ownership contract
------------------------
* step()      - SOLE caller of allocator.allocate().  For prefill: allocates all
                prompt blocks immediately and sets bt.num_tokens = prompt_len.
                For decode: allocates one block at each kv_len % block_size == 0.
* post_step() - SOLE writer of generated_ids / last_token_id / total_generated_tokens.
                Mirrors cur_len from bt.num_tokens but does NOT increment num_tokens.
* model runner:
    prefill()     - writes K/V into already-allocated blocks; must NOT call
                    allocator.allocate() or change block_table.num_tokens.
    decode_step() - bumps bt.num_tokens += 1 before the kernel call (so the
                    kernel knows the write position); must NOT call allocate().

Single source of truth for KV occupancy: block_table.num_tokens.
seq.cur_len is a mirror kept in sync by post_step().
"""
from __future__ import annotations

import enum
import itertools
import logging
import math
from dataclasses import dataclass, field
from typing import Any, List, Literal, Optional

from engine.memory.block_allocator import BlockAllocator, BlockTable

logger = logging.getLogger(__name__)


class SeqStatus(enum.Enum):
    WAITING = "waiting"
    RUNNING = "running"
    SWAPPED = "swapped"
    FINISHED = "finished"


@dataclass
class Sequence:
    seq_id: int
    prompt: str
    max_new_tokens: int
    status: SeqStatus = SeqStatus.WAITING
    generated_ids: list[int] = field(default_factory=list)
    prompt_token_ids: list[int] = field(default_factory=list)
    block_table: Optional[BlockTable] = None
    cur_len: int = 0
    num_cached_tokens: int = 0
    prefix_stats_recorded: bool = False
    last_token_id: Optional[int] = None
    num_preemptions: int = 0
    total_generated_tokens: int = 0
    all_generated_ids: list[int] = field(default_factory=list)
    last_logits: Optional[Any] = None
    last_logit_margin: Optional[float] = None
    ignore_eos: bool = False

    @property
    def kv_len(self) -> int:
        """Authoritative KV-cache token count, derived solely from block_table."""
        if self.block_table is not None:
            return self.block_table.num_tokens
        return self.cur_len

    @property
    def num_tokens(self) -> int:
        """Compat alias for kv_len."""
        return self.kv_len

    def is_finished(self, eos_token_id: Optional[int] = None) -> bool:
        if self.total_generated_tokens >= self.max_new_tokens:
            return True
        if not self.ignore_eos and eos_token_id is not None and self.generated_ids and self.generated_ids[-1] == eos_token_id:
            return True
        return False


class BatchType(enum.Enum):
    PREFILL = "prefill"
    DECODE = "decode"


@dataclass
class ScheduledBatch:
    batch_type: BatchType
    sequences: list[Sequence]

    @property
    def is_empty(self) -> bool:
        return len(self.sequences) == 0


class Scheduler:
    """
    Iteration-level continuous batching scheduler with LIFO preemption.

    Queues:
      - waiting: sequences awaiting initial prefill or re-prefill after preemption
      - running: sequences currently undergoing autoregressive decode
      - swapped: placeholder for preempted sequences

    Admission & Scheduling Invariants:
      1. Each step() returns a ScheduledBatch: either a PREFILL (one sequence)
         or a DECODE (all running sequences).
      2. Priority order:
         - Prefill admission: If waiting queue has requests, admit the next sequence (FCFS)
           only if the allocator can allocate (blocks_for_prompt + watermark).
           Returns ScheduledBatch(BatchType.PREFILL, [admitted_seq]).
         - Decode step: If no sequence is admitted (or waiting queue is empty) and running
           sequences exist, schedule a decode step for all running sequences.
           Before the decode step, any running sequence whose seq_len % block_size == 0
           needs one new physical block allocated.
         - Preemption: If a running sequence needs a new block and the allocator has none,
           preempt the most recently admitted running sequence (LIFO): free all its blocks,
           and move it to the front of waiting with its prompt plus generated-so-far tokens
           as the new prompt, so it re-prefills later. Keep retrying until the needy sequence
           gets its block or it is itself the only running sequence. If a single sequence
           can never fit in the pool, raise a MemoryError.
      3. Finished sequences (reached max_new_tokens or generated EOS) are freed immediately:
         all their physical blocks are returned to the allocator and status set to FINISHED.
    """

    def __init__(
        self,
        block_allocator: BlockAllocator,
        block_size: int = 16,
        watermark_blocks: int = 1,
        max_batch_size: int = 32,
        eos_token_id: Optional[int] = None,
        debug_invariants: bool = False,
        prefix_mode: Optional[Literal["off", "share", "full"]] = None,
    ):
        if prefix_mode not in (None, "off", "share", "full"):
            raise ValueError(
                f"Unsupported prefix_mode {prefix_mode!r}; expected 'off', 'share', or 'full'."
            )
        if prefix_mode in ("share", "full") and not block_allocator.enable_prefix_cache:
            raise ValueError(
                f"prefix_mode={prefix_mode!r} requires an allocator with prefix caching enabled."
            )
        self.block_allocator = block_allocator
        self.block_size = block_size
        self.watermark_blocks = watermark_blocks
        self.max_batch_size = max_batch_size
        self.eos_token_id = eos_token_id
        self.debug_invariants = debug_invariants
        self.prefix_mode = prefix_mode

        self.waiting: list[Sequence] = []
        self.running: list[Sequence] = []
        self.swapped: list[Sequence] = []
        self._id_counter = itertools.count()
        self.preemption_count: int = 0
        self.admission_blocked_count: int = 0
        self.prefix_prompt_tokens: int = 0
        self.prefix_cached_tokens: int = 0
        self.prefix_blocks_saved: int = 0

    @property
    def _prefix_cache_enabled(self) -> bool:
        return self.block_allocator.enable_prefix_cache and self.prefix_mode != "off"

    def add_request(
        self,
        prompt: str,
        max_new_tokens: int,
        prompt_token_ids: Optional[list[int]] = None,
        ignore_eos: bool = False,
    ) -> Sequence:
        """Enqueues a new generation request into the waiting queue."""
        seq = Sequence(
            seq_id=next(self._id_counter),
            prompt=prompt,
            max_new_tokens=max_new_tokens,
            prompt_token_ids=prompt_token_ids if prompt_token_ids is not None else [],
            ignore_eos=ignore_eos,
        )
        self.waiting.append(seq)
        return seq

    def has_unfinished_requests(self) -> bool:
        """Returns True if there are requests currently waiting, running, or swapped."""
        return bool(self.waiting or self.running or self.swapped)

    def _calc_blocks_needed(self, num_tokens: int) -> int:
        if num_tokens <= 0:
            return 1
        return (num_tokens + self.block_size - 1) // self.block_size

    def _register_completed_blocks(self, seq: Sequence, token_ids: list[int]) -> None:
        if not self._prefix_cache_enabled or seq.block_table is None:
            return

        full_blocks = min(
            seq.block_table.num_tokens // self.block_size,
            len(token_ids) // self.block_size,
        )
        parent_key = None
        for block_idx in range(full_blocks):
            token_start = block_idx * self.block_size
            block_tokens = tuple(token_ids[token_start : token_start + self.block_size])
            key = (parent_key, block_tokens)
            block_id = seq.block_table.physical_block_ids[block_idx]
            existing_block = self.block_allocator.cache.get(key)
            if existing_block is None:
                self.block_allocator.register(block_id, key)
            elif existing_block == block_id:
                self.block_allocator.register(block_id, key)
            parent_key = key

    def free_sequence(self, seq: Sequence) -> None:
        """Frees all physical blocks held by sequence and updates its status."""
        if seq.block_table is not None:
            seq.block_table.free(self.block_allocator)
            seq.block_table = None
        seq.num_cached_tokens = 0
        seq.status = SeqStatus.FINISHED

    def abort_sequence(self, seq_id: int) -> bool:
        """
        Aborts a sequence by seq_id from waiting or running queue,
        freeing all its allocated blocks immediately.
        """
        # Check waiting queue
        for i, seq in enumerate(self.waiting):
            if seq.seq_id == seq_id:
                self.waiting.pop(i)
                self.free_sequence(seq)
                return True

        # Check running queue
        for i, seq in enumerate(self.running):
            if seq.seq_id == seq_id:
                self.running.pop(i)
                self.free_sequence(seq)
                return True

        return False

    def _preempt_sequence(self, victim: Sequence) -> None:
        """
        Preempts a running sequence:
        1. Frees all its physical blocks.
        2. Absorbs its generated tokens so far into prompt / prompt_token_ids for re-prefill.
        3. Moves victim to the front of the waiting queue.
        """
        self.preemption_count += 1
        victim.num_preemptions += 1
        pool_size = self.block_allocator.num_blocks
        allocated_blocks = pool_size - self.block_allocator.num_available()
        logger.info(
            f"Preempting sequence {victim.seq_id} (preemption #{self.preemption_count}). "
            f"Peak allocated blocks: {allocated_blocks}/{pool_size} (pool size: {pool_size}). "
            f"Generated {len(victim.generated_ids)} tokens so far."
        )

        # Free all allocated blocks back to allocator
        if victim.block_table is not None:
            victim.block_table.free(self.block_allocator)
            victim.block_table = None
        victim.num_cached_tokens = 0

        # Fold generated tokens into prompt_token_ids for re-prefill
        if victim.generated_ids:
            victim.prompt_token_ids.extend(victim.generated_ids)
            victim.generated_ids = []

        victim.cur_len = len(victim.prompt_token_ids)
        victim.last_token_id = None
        victim.status = SeqStatus.WAITING

        # Prepend victim to the front of waiting queue
        self.waiting.insert(0, victim)

    def post_step(self, batch: ScheduledBatch, next_tokens: list[int]) -> None:
        """
        Updates sequence state after engine execution of the scheduled batch.

        - If prefill: transitions sequence from WAITING to RUNNING.
        - If decode: updates running sequences with newly generated tokens.
        - Detects finished sequences (max_new_tokens or EOS) and frees their blocks immediately.

        Invariant: seq.cur_len == bt.num_tokens == tokens written into KV cache.
          Prefill: KV cache holds prompt_len tokens. cur_len was already set to
                   prompt_len in scheduler.step() admission — do NOT increment it.
                   The first generated token is the input_ids for the first decode
                   step; it is not in the KV cache yet.
          Decode:  decode_step wrote the new token into the KV cache (bumped
                   bt.num_tokens). Reflect that by incrementing cur_len.
        """
        if batch.is_empty:
            return

        for seq, tok in zip(batch.sequences, next_tokens):
            if seq.block_table is not None:
                if batch.batch_type == BatchType.PREFILL:
                    kv_token_ids = seq.prompt_token_ids
                else:
                    kv_token_ids = seq.prompt_token_ids + seq.generated_ids
                self._register_completed_blocks(seq, kv_token_ids)

            seq.last_token_id = tok
            seq.generated_ids.append(tok)
            seq.all_generated_ids.append(tok)
            seq.total_generated_tokens += 1

            if batch.batch_type == BatchType.PREFILL:
                # step() set bt.num_tokens = prompt_len.  The first generated token
                # is the decode input, not yet in the KV cache — do NOT increment.
                # Mirror cur_len for any backward-compat code that reads it.
                seq.cur_len = seq.block_table.num_tokens if seq.block_table else 0
                seq.status = SeqStatus.RUNNING
                self.running.append(seq)
            else:
                # decode_step() already bumped bt.num_tokens += 1 before the kernel
                # call.  post_step must NOT increment again — just mirror cur_len.
                if seq.block_table is not None:
                    seq.cur_len = seq.block_table.num_tokens
                else:
                    seq.cur_len += 1

            # Check if sequence finished
            if seq.is_finished(self.eos_token_id):
                if seq in self.running:
                    self.running.remove(seq)
                self.free_sequence(seq)

        if self.debug_invariants:
            self._check_invariants()

    def step(self) -> Optional[ScheduledBatch]:
        """
        Executes one scheduler iteration.

        1. Cleanup any finished sequences in running queue (defensive check).
        2. FCFS Admission: check head of waiting queue.
           If allocator.can_allocate(blocks_for_prompt + watermark) and len(running) < max_batch_size:
             - Pop sequence from waiting
             - Allocate physical blocks for its prompt
             - Returns ScheduledBatch(PREFILL, [seq])
        3. Decode Scheduling with Preemption:
           If running queue is non-empty:
             - For any running sequence whose cur_len % block_size == 0, allocate 1 block.
             - If allocator is exhausted, preempt most recently admitted sequence (LIFO)
               until block is available or the needy sequence is the only running sequence.
             - If single sequence cannot fit in pool, raise MemoryError.
             - Returns ScheduledBatch(DECODE, running)
        4. If both waiting and running are idle, returns None.
        """
        # 1. Purge any running sequences that are finished
        still_running: list[Sequence] = []
        for seq in self.running:
            if seq.is_finished(self.eos_token_id):
                self.free_sequence(seq)
            else:
                still_running.append(seq)
        self.running = still_running

        # 2. Try FCFS admission from waiting queue
        if self.waiting and len(self.running) < self.max_batch_size:
            candidate = self.waiting[0]
            prompt_len = candidate.cur_len or len(candidate.prompt_token_ids)
            blocks_for_prompt = self._calc_blocks_needed(prompt_len)
            cached_block_ids: list[int] = []
            if self._prefix_cache_enabled:
                cached_block_ids = self.block_allocator.lookup_prefix(candidate.prompt_token_ids)
                max_cached_blocks = max(0, (prompt_len - 1) // self.block_size)
                cached_block_ids = cached_block_ids[:max_cached_blocks]
            blocks_to_allocate = blocks_for_prompt - len(cached_block_ids)
            cached_free_hits = sum(
                block_id in self.block_allocator.cached_free for block_id in cached_block_ids
            )
            available_after_hits = self.block_allocator.num_available() - cached_free_hits

            # Check if candidate can never fit in pool
            if blocks_to_allocate > self.block_allocator.num_blocks:
                self.waiting.pop(0)
                candidate.status = SeqStatus.FINISHED
                raise MemoryError(
                    f"Sequence {candidate.seq_id} requires {blocks_to_allocate} blocks for prompt, "
                    f"exceeding total pool capacity of {self.block_allocator.num_blocks} blocks."
                )

            if available_after_hits >= blocks_to_allocate + self.watermark_blocks:
                admitted = self.waiting.pop(0)

                if admitted.block_table is None:
                    admitted.block_table = BlockTable(seq_id=admitted.seq_id, block_size=self.block_size)

                admitted.num_cached_tokens = len(cached_block_ids) * self.block_size
                if not admitted.prefix_stats_recorded:
                    self.prefix_prompt_tokens += prompt_len
                    self.prefix_cached_tokens += admitted.num_cached_tokens
                    self.prefix_blocks_saved += len(cached_block_ids)
                    admitted.prefix_stats_recorded = True
                for blk in cached_block_ids:
                    self.block_allocator.acquire(blk)
                    admitted.block_table.physical_block_ids.append(blk)

                # Allocate only the uncached prompt blocks before model execution.
                # This ensures watermark checks and preemption triggers work properly.
                for _ in range(blocks_to_allocate):
                    blk = self.block_allocator.allocate()
                    admitted.block_table.physical_block_ids.append(blk)
                admitted.block_table.num_tokens = prompt_len
                admitted.cur_len = prompt_len

                return ScheduledBatch(batch_type=BatchType.PREFILL, sequences=[admitted])
            else:
                self.admission_blocked_count += 1
        elif self.waiting:
            # Waiting queue has requests but running queue reached max_batch_size
            self.admission_blocked_count += 1

        # 3. Decode step for running sequences
        if self.running:
            # Index-based loop so preemptions can be handled dynamically.
            idx = 0
            while idx < len(self.running):
                seq = self.running[idx]
                if seq.block_table is None:
                    seq.block_table = BlockTable(seq_id=seq.seq_id, block_size=self.block_size)

                bt = seq.block_table
                # Use bt.num_tokens (authoritative) not seq.cur_len (mirror).
                kv_len = bt.num_tokens
                needs_block = (kv_len > 0) and (kv_len % self.block_size == 0)

                if needs_block:
                    while not self.block_allocator.can_allocate(1):
                        # Find victim to preempt using LIFO (most recently admitted running sequence != seq)
                        victim = None
                        for cand in reversed(self.running):
                            if cand is not seq:
                                victim = cand
                                break

                        if victim is not None:
                            self.running.remove(victim)
                            self._preempt_sequence(victim)
                            # If victim was before or at our current index, adjust idx accordingly
                            # Re-evaluating seq at current index in running
                            idx = self.running.index(seq)
                        else:
                            # seq is the only running sequence and no blocks free!
                            # Verify if seq exceeds total pool capacity
                            num_held = len(bt.physical_block_ids)
                            if num_held + 1 > self.block_allocator.num_blocks:
                                self.running.remove(seq)
                                self.free_sequence(seq)
                                raise MemoryError(
                                    f"Sequence {seq.seq_id} requires {num_held + 1} blocks, "
                                    f"which exceeds total pool capacity of {self.block_allocator.num_blocks} blocks."
                                )
                            # Otherwise cannot allocate even though it is the only sequence
                            raise MemoryError(
                                f"Cannot allocate block for seq {seq.seq_id} despite being only running sequence."
                            )

                    # Now we have capacity to allocate 1 block for seq
                    new_blk = self.block_allocator.allocate()
                    bt.physical_block_ids.append(new_blk)

                idx += 1

            if self.running:
                return ScheduledBatch(batch_type=BatchType.DECODE, sequences=list(self.running))

        return None

    def _check_invariants(self) -> None:
        """
        Debug invariant checker (runs when debug_invariants=True after each post_step).

        Invariant 1 — Block coverage:
            ceil(bt.num_tokens / block_size) == len(bt.physical_block_ids)
        Invariant 2 — Shared references:
            allocator.ref_counts[block_id] equals the number of running block
            tables containing that physical block.
        Invariant 3 — Conservation:
            referenced + plain free + cached-free == total blocks
        """
        table_ref_counts: dict[int, int] = {}

        for seq in self.running:
            bt = seq.block_table
            if bt is None:
                continue

            expected = math.ceil(bt.num_tokens / self.block_size) if bt.num_tokens > 0 else 0
            actual = len(bt.physical_block_ids)
            if expected != actual:
                raise AssertionError(
                    f"[Invariant 1] seq {seq.seq_id}: bt.num_tokens={bt.num_tokens} -> "
                    f"expected {expected} blocks, found {actual}."
                )

            for blk_id in bt.physical_block_ids:
                table_ref_counts[blk_id] = table_ref_counts.get(blk_id, 0) + 1

        if table_ref_counts != self.block_allocator.ref_counts:
            raise AssertionError(
                f"[Invariant 2] Block table references {table_ref_counts} do not match "
                f"allocator references {self.block_allocator.ref_counts}."
            )

        referenced_blocks = len(self.block_allocator.ref_counts)
        plain_free_blocks = len(self.block_allocator.free_block_ids)
        cached_free_blocks = len(self.block_allocator.cached_free)
        total = self.block_allocator.num_blocks
        if referenced_blocks + plain_free_blocks + cached_free_blocks != total:
            raise AssertionError(
                f"[Invariant 3] Block conservation: referenced={referenced_blocks}, "
                f"plain_free={plain_free_blocks}, cached_free={cached_free_blocks}, "
                f"total={total}."
            )
