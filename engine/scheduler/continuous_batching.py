"""
Phase 4 — Continuous batching scheduler. STUB.

Implement using the Phase 4 prompt. Interface contract below.

Design reference: vLLM's `Scheduler` (waiting / running / swapped queues,
called once per engine `step()`).
"""
from __future__ import annotations

import enum
import itertools
from dataclasses import dataclass, field


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


class Scheduler:
    """
    TODO(Phase 4) implement:

      - add_request(seq: Sequence): append to `waiting`
      - step() -> ScheduledBatch:
          1. Move finished sequences out of `running`.
          2. Try to admit sequences from `waiting` into `running`, but only
             if BlockAllocator.can_allocate(seq.num_blocks_needed) — i.e.
             don't admit if there isn't VRAM capacity.
          3. If `running` needs more blocks than available (a running
             sequence needs a new block mid-decode and none are free),
             PREEMPT: move the lowest-priority running sequence to
             `swapped` (or just drop-and-recompute, simplest first), free
             its blocks, and retry admission.
          4. Return the batch to run this iteration: a mix of prefill
             sequences (just admitted) and decode-step sequences (already
             running) — these have different sequence lengths and need to
             be batched together carefully (ragrged batch / padding mask).

    Key invariant: `waiting` + `running` + `swapped` should never require
    more physical blocks than BlockAllocator has, and preemption should
    always free real capacity before this method returns.
    """

    def __init__(self, block_allocator, max_batch_size: int = 32):
        self.block_allocator = block_allocator
        self.max_batch_size = max_batch_size
        self.waiting: list[Sequence] = []
        self.running: list[Sequence] = []
        self.swapped: list[Sequence] = []
        self._id_counter = itertools.count()

    def add_request(self, prompt: str, max_new_tokens: int) -> Sequence:
        seq = Sequence(seq_id=next(self._id_counter), prompt=prompt, max_new_tokens=max_new_tokens)
        self.waiting.append(seq)
        return seq

    def step(self):
        raise NotImplementedError("Implement iteration-level scheduling — see Phase 4 docstring above.")
