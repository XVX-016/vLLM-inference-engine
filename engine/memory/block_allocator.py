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

import torch


def compute_num_blocks(
    model_config: any,
    block_size: int,
    gpu_memory_utilization: float = 0.9,
    device: Optional[torch.device | str] = None,
    dtype: torch.dtype = torch.float16,
) -> int:
    """
    Computes num_blocks by taking a target fraction of free VRAM divided by
    the per-block byte cost across all transformer layers (K + V).

    per_block_bytes = 2 (K+V) * num_layers * block_size * num_kv_heads * head_dim * dtype_bytes
    """
    device_obj = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    if device_obj.type != "cuda":
        raise ValueError(f"compute_num_blocks requires a CUDA device, got {device_obj}")

    # Extract model dimensions
    num_layers = getattr(model_config, "num_hidden_layers", getattr(model_config, "n_layer", None))
    num_kv_heads = getattr(
        model_config,
        "num_key_value_heads",
        getattr(model_config, "num_attention_heads", getattr(model_config, "n_head", None)),
    )
    head_dim = getattr(
        model_config,
        "head_dim",
        getattr(model_config, "hidden_size", 0) // getattr(model_config, "num_attention_heads", 1),
    )

    if not (num_layers and num_kv_heads and head_dim):
        raise ValueError(
            f"Unable to infer all required KV dimensions from model_config: "
            f"num_layers={num_layers}, num_kv_heads={num_kv_heads}, head_dim={head_dim}"
        )

    element_size = torch.tensor([], dtype=dtype).element_size()
    # 2 for both K and V cache per block across all layers
    block_bytes = 2 * num_layers * block_size * num_kv_heads * head_dim * element_size

    # Check available free memory on device
    free_bytes, total_bytes = torch.cuda.mem_get_info(device_obj)
    target_bytes = int(free_bytes * gpu_memory_utilization)

    num_blocks = target_bytes // block_bytes
    return int(num_blocks)


@dataclass
class PhysicalBlockPool:
    """
    One big pre-allocated tensor pair (K and V) per layer, divided into
    fixed-size blocks. Shape per layer: [num_blocks, block_size, num_kv_heads, head_dim].

    Allocated once at engine startup, sized to fill a target fraction of free VRAM.
    """
    num_blocks: int
    block_size: int
    num_layers: int
    num_kv_heads: int
    head_dim: int
    dtype: torch.dtype = torch.float16
    device: Optional[torch.device | str] = None
    k_cache: list[torch.Tensor] = field(init=False)
    v_cache: list[torch.Tensor] = field(init=False)

    def __post_init__(self):
        self.device = torch.device(self.device or ("cuda" if torch.cuda.is_available() else "cpu"))
        shape = (self.num_blocks, self.block_size, self.num_kv_heads, self.head_dim)
        self.k_cache = [
            torch.empty(shape, dtype=self.dtype, device=self.device)
            for _ in range(self.num_layers)
        ]
        self.v_cache = [
            torch.empty(shape, dtype=self.dtype, device=self.device)
            for _ in range(self.num_layers)
        ]

    @property
    def total_vram_bytes(self) -> int:
        element_size = torch.tensor([], dtype=self.dtype).element_size()
        return 2 * self.num_layers * self.num_blocks * self.block_size * self.num_kv_heads * self.head_dim * element_size

    @classmethod
    def from_model_config(
        cls,
        model_config: any,
        block_size: int = 16,
        gpu_memory_utilization: float = 0.9,
        device: Optional[torch.device | str] = None,
        dtype: torch.dtype = torch.float16,
    ) -> PhysicalBlockPool:
        num_layers = getattr(model_config, "num_hidden_layers", getattr(model_config, "n_layer", None))
        num_kv_heads = getattr(
            model_config,
            "num_key_value_heads",
            getattr(model_config, "num_attention_heads", getattr(model_config, "n_head", None)),
        )
        head_dim = getattr(
            model_config,
            "head_dim",
            getattr(model_config, "hidden_size", 0) // getattr(model_config, "num_attention_heads", 1),
        )

        num_blocks = compute_num_blocks(
            model_config=model_config,
            block_size=block_size,
            gpu_memory_utilization=gpu_memory_utilization,
            device=device,
            dtype=dtype,
        )

        return cls(
            num_blocks=num_blocks,
            block_size=block_size,
            num_layers=num_layers,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            dtype=dtype,
            device=device,
        )


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

    def get_free_block_count(self) -> int:
        """Returns the number of unallocated blocks currently in the free list."""
        return len(self.free_block_ids)

    @property
    def num_free_blocks(self) -> int:
        return len(self.free_block_ids)

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
        if not self.physical_block_ids:
            return False
        return self.num_tokens % self.block_size != 0

    def free(self, allocator: BlockAllocator) -> None:
        """Release all physical blocks held by this sequence back to the allocator."""
        for b_id in self.physical_block_ids:
            allocator.free(b_id)
        self.physical_block_ids.clear()
