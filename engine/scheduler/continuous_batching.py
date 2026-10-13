"""
Continuous batching scheduler for LLM inference (Phase 4).

Implements iteration-level scheduling with waiting and running queues, FCFS admission,
watermark-based capacity checks, and prefill vs. decode batch separation.
Includes LIFO preemption with re-prefill for memory exhaustion.
"""
from __future__ import annotations

import enum
import itertools
import logging
from dataclasses import dataclass, field
from typing import Any, List, Optional

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
    last_token_id: Optional[int] = None
    num_preemptions: int = 0
    total_generated_tokens: int = 0
    last_logits: Optional[Any] = None

    @property
    def num_tokens(self) -> int:
        if self.cur_len > 0:
            return self.cur_len
        return len(self.prompt_token_ids) + len(self.generated_ids)

    def is_finished(self, eos_token_id: Optional[int] = None) -> bool:
        if self.total_generated_tokens >= self.max_new_tokens:
            return True
        if eos_token_id is not None and self.generated_ids and self.generated_ids[-1] == eos_token_id:
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
    ):
        self.block_allocator = block_allocator
        self.block_size = block_size
        self.watermark_blocks = watermark_blocks
        self.max_batch_size = max_batch_size
        self.eos_token_id = eos_token_id

        self.waiting: list[Sequence] = []
        self.running: list[Sequence] = []
        self.swapped: list[Sequence] = []
        self._id_counter = itertools.count()
        self.preemption_count: int = 0

    def add_request(
        self,
        prompt: str,
        max_new_tokens: int,
        prompt_token_ids: Optional[list[int]] = None,
    ) -> Sequence:
        """Enqueues a new generation request into the waiting queue."""
        seq = Sequence(
            seq_id=next(self._id_counter),
            prompt=prompt,
            max_new_tokens=max_new_tokens,
            prompt_token_ids=prompt_token_ids if prompt_token_ids is not None else [],
        )
        self.waiting.append(seq)
        return seq

    def _calc_blocks_needed(self, num_tokens: int) -> int:
        if num_tokens <= 0:
            return 1
        return (num_tokens + self.block_size - 1) // self.block_size

    def free_sequence(self, seq: Sequence) -> None:
        """Frees all physical blocks held by sequence and updates its status."""
        if seq.block_table is not None:
            seq.block_table.free(self.block_allocator)
            seq.block_table = None
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
        logger.info(
            f"Preempting sequence {victim.seq_id} (preemption #{self.preemption_count}). "
            f"Generated {len(victim.generated_ids)} tokens so far."
        )

        # Free all allocated blocks back to allocator
        if victim.block_table is not None:
            victim.block_table.free(self.block_allocator)
            victim.block_table = None

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
            seq.last_token_id = tok
            seq.generated_ids.append(tok)
            seq.total_generated_tokens += 1

            if batch.batch_type == BatchType.PREFILL:
                # cur_len was set to prompt_len by prefill(). KV cache matches. No change.
                seq.status = SeqStatus.RUNNING
                self.running.append(seq)
            else:
                # Sync cur_len to the ground-truth KV count kept in the block table.
                # Using bt.num_tokens avoids drift if any path increments differently.
                if seq.block_table is not None:
                    seq.cur_len = seq.block_table.num_tokens
                else:
                    seq.cur_len += 1

            # Check if sequence finished
            if seq.is_finished(self.eos_token_id):
                if seq in self.running:
                    self.running.remove(seq)
                self.free_sequence(seq)

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

            # Check if candidate can never fit in pool
            if blocks_for_prompt > self.block_allocator.num_blocks:
                self.waiting.pop(0)
                candidate.status = SeqStatus.FINISHED
                raise MemoryError(
                    f"Sequence {candidate.seq_id} requires {blocks_for_prompt} blocks for prompt, "
                    f"exceeding total pool capacity of {self.block_allocator.num_blocks} blocks."
                )

            if self.block_allocator.can_allocate(blocks_for_prompt + self.watermark_blocks):
                admitted = self.waiting.pop(0)

                if admitted.block_table is None:
                    admitted.block_table = BlockTable(seq_id=admitted.seq_id, block_size=self.block_size)

                admitted.cur_len = prompt_len

                return ScheduledBatch(batch_type=BatchType.PREFILL, sequences=[admitted])

        # 3. Decode step for running sequences
        if self.running:
            # We iterate through running sequences to ensure each has enough blocks.
            # Using an index-based loop allows handling preemptions dynamically.
            idx = 0
            while idx < len(self.running):
                seq = self.running[idx]
                if seq.block_table is None:
                    seq.block_table = BlockTable(seq_id=seq.seq_id, block_size=self.block_size)

                bt = seq.block_table
                needs_block = (seq.cur_len > 0) and (seq.cur_len % self.block_size == 0)

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
