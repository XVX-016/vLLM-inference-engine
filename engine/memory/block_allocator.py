"""
Phase 2 — Paged KV-cache memory manager. STUB.

This is the core systems-engineering deliverable of the project. Implement
this using the prompt in the project plan (Phase 2). Below is the interface
contract every later phase (kernel, scheduler) will depend on — build to
this shape so Phases 3-4 don't need rework.

Design reference: vLLM's `BlockManager` / `BlockSpaceManager`.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class PhysicalBlockPool:
    """
    One big pre-allocated tensor pair (K and V) per layer, divided into
    fixed-size blocks. Shape per layer: [num_blocks, block_size, num_kv_heads, head_dim].

    TODO(Phase 2): allocate this once at engine startup, sized to fill a
    target fraction of free VRAM (mirrors vLLM's `gpu_memory_utilization`).
    """
    num_blocks: int
    block_size: int
    num_layers: int
    num_kv_heads: int
    head_dim: int
    # k_cache: list[torch.Tensor]  # one per layer, shape [num_blocks, block_size, num_kv_heads, head_dim]
    # v_cache: list[torch.Tensor]


class BlockAllocator:
    """
    Free-list based allocator over physical block indices.

    TODO(Phase 2) implement:
      - allocate() -> int                  # pop a free physical block index
      - free(block_idx: int) -> None       # return a block to the free list
      - fork(block_idx: int) -> int        # increment ref count for copy-on-write sharing
      - can_allocate(n: int) -> bool       # check capacity before scheduler admits a request

    Must support:
      - ref counting per physical block (needed for prefix caching / COW in Phase 6)
      - O(1) allocate/free (free list, not a linear scan)
    """

    def __init__(self, num_blocks: int):
        self.num_blocks = num_blocks
        self.free_block_ids: list[int] = list(range(num_blocks))
        self.ref_counts: dict[int, int] = {}

    def can_allocate(self, n: int) -> bool:
        return len(self.free_block_ids) >= n

    def allocate(self) -> int:
        if not self.free_block_ids:
            raise MemoryError("No free physical blocks — scheduler should have preempted before calling this.")
        block_id = self.free_block_ids.pop()
        self.ref_counts[block_id] = 1
        return block_id

    def free(self, block_id: int) -> None:
        self.ref_counts[block_id] -= 1
        if self.ref_counts[block_id] <= 0:
            del self.ref_counts[block_id]
            self.free_block_ids.append(block_id)

    def fork(self, block_id: int) -> int:
        """Increment ref count for copy-on-write sharing (Phase 6: prefix caching)."""
        self.ref_counts[block_id] = self.ref_counts.get(block_id, 1) + 1
        return block_id


@dataclass
class BlockTable:
    """
    Per-sequence logical -> physical block index mapping.

    TODO(Phase 2) implement:
      - append_token(): if current last block is full, allocate a new
        physical block via BlockAllocator and append its index to
        `physical_block_ids`.
      - free(allocator): release every physical block this sequence holds.
    """
    seq_id: int
    block_size: int
    physical_block_ids: list[int] = field(default_factory=list)
    num_tokens: int = 0

    def num_blocks_needed_for(self, num_tokens: int) -> int:
        return (num_tokens + self.block_size - 1) // self.block_size

    def last_block_has_space(self) -> bool:
        return self.num_tokens % self.block_size != 0 or self.num_tokens == 0
