"""
Phase 3 — Paged-attention Triton kernel. STUB.

Implement this using the Phase 3 prompt. Interface contract below —
`paged_attention_decode` is what the engine's forward pass will call once
this is real; keep the signature stable so Phase 4's scheduler and the
serving layer don't need to change when the kernel implementation matures.

Build order (do not skip steps):
  1. Naive PyTorch reference implementation (gather blocks with indexing,
     plain softmax attention). This is your ground truth — write it first.
  2. Triton kernel for SINGLE-sequence decode attention against paged KV.
  3. Randomized correctness test: kernel output vs. reference, across
     random block-table layouts (non-contiguous physical blocks).
  4. Extend to BATCHED multi-sequence decode (this is the actual payoff —
     one kernel launch serving many sequences with different block tables).
  5. Online (flash-attention-style) softmax so you never materialize the
     full [seq_len] score row when sequences are long.

import triton
import triton.language as tl
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl


def naive_reference_paged_attention(
    q: torch.Tensor,            # [num_seqs, num_heads, head_dim]  (single decode step)
    k_cache: torch.Tensor,      # [num_blocks, block_size, num_kv_heads, head_dim]
    v_cache: torch.Tensor,      # [num_blocks, block_size, num_kv_heads, head_dim]
    block_tables: list[list[int]],  # per-sequence physical block indices, in logical order
    seq_lens: list[int],
    scale: float,
) -> torch.Tensor:
    """
    Ground-truth reference: gather K/V per sequence via plain indexing,
    run standard softmax attention. O(seq_len) memory per sequence — fine
    for correctness testing, not for production. Every kernel change in
    this file must be validated against this function before it's trusted.
    """
    num_seqs, num_q_heads, head_dim = q.shape
    num_kv_heads = k_cache.shape[2]
    assert num_q_heads % num_kv_heads == 0, (
        f"num_q_heads ({num_q_heads}) must be divisible by num_kv_heads ({num_kv_heads})"
    )
    group_size = num_q_heads // num_kv_heads

    outputs = torch.zeros_like(q)

    for s in range(num_seqs):
        block_size = k_cache.shape[1]
        seq_len = seq_lens[s]
        block_ids = block_tables[s]

        k_parts, v_parts = [], []
        remaining = seq_len
        for b in block_ids:
            take = min(block_size, remaining)
            k_parts.append(k_cache[b, :take])   # [take, num_kv_heads, head_dim]
            v_parts.append(v_cache[b, :take])
            remaining -= take
            if remaining <= 0:
                break

        k = torch.cat(k_parts, dim=0)  # [seq_len, num_kv_heads, head_dim]
        v = torch.cat(v_parts, dim=0)  # [seq_len, num_kv_heads, head_dim]

        q_s = q[s]  # [num_q_heads, head_dim]

        # For GQA, query head h reads from KV head h // group_size
        kv_head_indices = torch.arange(num_q_heads, device=q.device) // group_size
        k_expanded = k[:, kv_head_indices, :]  # [seq_len, num_q_heads, head_dim]
        v_expanded = v[:, kv_head_indices, :]  # [seq_len, num_q_heads, head_dim]

        scores = torch.einsum("hd,shd->hs", q_s, k_expanded) * scale  # [num_q_heads, seq_len]
        probs = torch.softmax(scores, dim=-1)
        out = torch.einsum("hs,shd->hd", probs, v_expanded)  # [num_q_heads, head_dim]
        outputs[s] = out

    return outputs


def paged_attention_decode(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    block_tables: list[list[int]],
    seq_lens: list[int],
    scale: float,
) -> torch.Tensor:
    """
    TODO(Phase 3): Triton kernel implementation. Until implemented, falls
    back to the (slow, correctness-only) naive reference so the rest of the
    engine can be wired up and tested end-to-end before the kernel exists.
    """
    return naive_reference_paged_attention(q, k_cache, v_cache, block_tables, seq_lens, scale)


@triton.jit
def _paged_attention_decode_v1_kernel(
    # Pointers
    q_ptr,              # [num_q_heads, head_dim]
    k_cache_ptr,        # [num_blocks, block_size, num_kv_heads, head_dim]
    v_cache_ptr,        # [num_blocks, block_size, num_kv_heads, head_dim]
    block_table_ptr,    # [max_blocks_per_seq] (1D int32)
    out_ptr,            # [num_q_heads, head_dim]
    # Strides for Q: shape [num_q_heads, head_dim]
    stride_qh, stride_qd,
    # Strides for K cache: shape [num_blocks, block_size, num_kv_heads, head_dim]
    stride_kb, stride_kt, stride_kh, stride_kd,
    # Strides for V cache: shape [num_blocks, block_size, num_kv_heads, head_dim]
    stride_vb, stride_vt, stride_vh, stride_vd,
    # Strides for Out: shape [num_q_heads, head_dim]
    stride_oh, stride_od,
    # Meta scalars
    scale,
    seq_len,
    num_blocks_for_seq,
    group_size: tl.constexpr,   # num_q_heads // num_kv_heads
    BLOCK_SIZE: tl.constexpr,   # block size (tokens per block)
    HEAD_DIM: tl.constexpr,     # head dimension
):
    """
    V1 Kernel: Single sequence, single decode step.
    Grid: (num_q_heads,) -> 1 program per query head.
    Two-pass softmax over the logical blocks for simplicity & correctness.
    """
    # 1. Query head ID and its corresponding KV head ID (GQA mapping)
    q_head_idx = tl.program_id(0)
    kv_head_idx = q_head_idx // group_size

    d_offsets = tl.arange(0, HEAD_DIM)

    # 2. Load the query vector for this head: q[q_head_idx, :]
    q_ptrs = q_ptr + q_head_idx * stride_qh + d_offsets * stride_qd
    q = tl.load(q_ptrs)  # [HEAD_DIM], float16

    tok_offsets = tl.arange(0, BLOCK_SIZE)

    # =========================================================================
    # PASS 1: Loop over blocks to find max_score and compute sum(exp(score - max))
    # =========================================================================
    max_score = -float("inf")
    for b_idx in range(num_blocks_for_seq):
        physical_block_id = tl.load(block_table_ptr + b_idx)

        # Base pointer to K for this physical block and KV head
        k_base_ptrs = (
            k_cache_ptr
            + physical_block_id * stride_kb
            + kv_head_idx * stride_kh
        )
        # 2D pointers [BLOCK_SIZE, HEAD_DIM]: across tokens in block and feature dims
        k_ptrs = (
            k_base_ptrs
            + tok_offsets[:, None] * stride_kt
            + d_offsets[None, :] * stride_kd
        )

        token_indices = b_idx * BLOCK_SIZE + tok_offsets
        token_mask = token_indices < seq_len

        # Load K block: shape [BLOCK_SIZE, HEAD_DIM]
        k_block = tl.load(k_ptrs, mask=token_mask[:, None], other=0.0)

        # Dot product q . k: q is [HEAD_DIM], k_block is [BLOCK_SIZE, HEAD_DIM]
        # fp32 accumulation
        scores = tl.sum(q[None, :] * k_block, axis=1) * scale  # [BLOCK_SIZE]
        scores = tl.where(token_mask, scores, -float("inf"))

        block_max = tl.max(scores, axis=0)
        max_score = tl.maximum(max_score, block_max)

    # Compute sum of exponentials with stabilized max_score
    denom = 0.0
    for b_idx in range(num_blocks_for_seq):
        physical_block_id = tl.load(block_table_ptr + b_idx)
        k_base_ptrs = (
            k_cache_ptr
            + physical_block_id * stride_kb
            + kv_head_idx * stride_kh
        )
        k_ptrs = (
            k_base_ptrs
            + tok_offsets[:, None] * stride_kt
            + d_offsets[None, :] * stride_kd
        )
        token_indices = b_idx * BLOCK_SIZE + tok_offsets
        token_mask = token_indices < seq_len

        k_block = tl.load(k_ptrs, mask=token_mask[:, None], other=0.0)
        scores = tl.sum(q[None, :] * k_block, axis=1) * scale
        scores = tl.where(token_mask, scores, -float("inf"))

        p = tl.exp(scores - max_score)
        p = tl.where(token_mask, p, 0.0)
        denom += tl.sum(p, axis=0)

    # =========================================================================
    # PASS 2: Compute weighted sum of values: acc = sum(probs * V)
    # =========================================================================
    acc = tl.zeros([HEAD_DIM], dtype=tl.float32)

    for b_idx in range(num_blocks_for_seq):
        physical_block_id = tl.load(block_table_ptr + b_idx)

        # Recompute probabilities for this block
        k_base_ptrs = (
            k_cache_ptr
            + physical_block_id * stride_kb
            + kv_head_idx * stride_kh
        )
        k_ptrs = (
            k_base_ptrs
            + tok_offsets[:, None] * stride_kt
            + d_offsets[None, :] * stride_kd
        )
        token_indices = b_idx * BLOCK_SIZE + tok_offsets
        token_mask = token_indices < seq_len

        k_block = tl.load(k_ptrs, mask=token_mask[:, None], other=0.0)
        scores = tl.sum(q[None, :] * k_block, axis=1) * scale
        scores = tl.where(token_mask, scores, -float("inf"))

        p = tl.exp(scores - max_score)
        p = tl.where(token_mask, p, 0.0)
        probs = p / denom  # [BLOCK_SIZE]

        # Base pointer to V for this physical block and KV head
        v_base_ptrs = (
            v_cache_ptr
            + physical_block_id * stride_vb
            + kv_head_idx * stride_vh
        )
        v_ptrs = (
            v_base_ptrs
            + tok_offsets[:, None] * stride_vt
            + d_offsets[None, :] * stride_vd
        )
        # Load V block: shape [BLOCK_SIZE, HEAD_DIM]
        v_block = tl.load(v_ptrs, mask=token_mask[:, None], other=0.0)

        # Weighted sum: probs [BLOCK_SIZE, 1] * v_block [BLOCK_SIZE, HEAD_DIM]
        acc += tl.sum(probs[:, None] * v_block, axis=0)

    # 3. Store result to out[q_head_idx, :]
    out_ptrs = out_ptr + q_head_idx * stride_oh + d_offsets * stride_od
    tl.store(out_ptrs, acc.to(tl.float16))


def paged_attention_decode_v1(
    q: torch.Tensor,            # [num_q_heads, head_dim] (fp16, cuda)
    k_cache: torch.Tensor,      # [num_blocks, block_size, num_kv_heads, head_dim] (fp16, cuda)
    v_cache: torch.Tensor,      # [num_blocks, block_size, num_kv_heads, head_dim] (fp16, cuda)
    block_table: torch.Tensor,  # 1D int32 cuda tensor of physical block IDs
    seq_len: int,
    scale: float,
) -> torch.Tensor:
    """
    Triton v1 decode paged-attention for a single sequence and single decode step.
    Supports GQA (num_q_heads % num_kv_heads == 0).
    """
    assert q.is_cuda and k_cache.is_cuda and v_cache.is_cuda and block_table.is_cuda
    assert q.dtype == torch.float16 and k_cache.dtype == torch.float16 and v_cache.dtype == torch.float16
    assert block_table.dtype == torch.int32

    num_q_heads, head_dim = q.shape
    num_blocks, block_size, num_kv_heads, head_dim_k = k_cache.shape
    assert head_dim == head_dim_k, f"Head dimensions must match: {head_dim} vs {head_dim_k}"
    assert num_q_heads % num_kv_heads == 0, (
        f"num_q_heads ({num_q_heads}) must be divisible by num_kv_heads ({num_kv_heads})"
    )
    group_size = num_q_heads // num_kv_heads

    num_blocks_for_seq = (seq_len + block_size - 1) // block_size
    assert block_table.shape[0] >= num_blocks_for_seq, (
        f"block_table size ({block_table.shape[0]}) smaller than needed ({num_blocks_for_seq})"
    )

    out = torch.empty_like(q)

    grid = (num_q_heads,)
    _paged_attention_decode_v1_kernel[grid](
        q,
        k_cache,
        v_cache,
        block_table,
        out,
        # Strides
        q.stride(0), q.stride(1),
        k_cache.stride(0), k_cache.stride(1), k_cache.stride(2), k_cache.stride(3),
        v_cache.stride(0), v_cache.stride(1), v_cache.stride(2), v_cache.stride(3),
        out.stride(0), out.stride(1),
        # Meta
        scale,
        seq_len,
        num_blocks_for_seq,
        group_size=group_size,
        BLOCK_SIZE=block_size,
        HEAD_DIM=head_dim,
    )
    return out


@triton.jit
def _paged_attention_decode_v2_kernel(
    # Pointers
    q_ptr,              # [num_q_heads, head_dim]
    k_cache_ptr,        # [num_blocks, block_size, num_kv_heads, head_dim]
    v_cache_ptr,        # [num_blocks, block_size, num_kv_heads, head_dim]
    block_table_ptr,    # [max_blocks_per_seq] (1D int32)
    out_ptr,            # [num_q_heads, head_dim]
    # Strides for Q: shape [num_q_heads, head_dim]
    stride_qh, stride_qd,
    # Strides for K cache: shape [num_blocks, block_size, num_kv_heads, head_dim]
    stride_kb, stride_kt, stride_kh, stride_kd,
    # Strides for V cache: shape [num_blocks, block_size, num_kv_heads, head_dim]
    stride_vb, stride_vt, stride_vh, stride_vd,
    # Strides for Out: shape [num_q_heads, head_dim]
    stride_oh, stride_od,
    # Meta scalars
    scale,
    seq_len,
    num_blocks_for_seq,
    group_size: tl.constexpr,   # num_q_heads // num_kv_heads
    BLOCK_SIZE: tl.constexpr,   # block size (tokens per block)
    HEAD_DIM: tl.constexpr,     # head dimension
):
    """
    V2 Kernel: Single-pass online softmax (FlashAttention style).
    Grid: (num_q_heads,) -> 1 program per query head.
    Iterates through logical blocks once, maintaining:
      m_i: running maximum of unnormalized scores
      l_i: running sum of exp(scores - m_i)
      acc: running unnormalized weighted sum sum(exp(score - m_i) * V)
    When m_new > m_prev, previous accumulators are rescaled by exp(m_prev - m_new).
    """
    q_head_idx = tl.program_id(0)
    kv_head_idx = q_head_idx // group_size

    d_offsets = tl.arange(0, HEAD_DIM)
    tok_offsets = tl.arange(0, BLOCK_SIZE)

    # Load query vector for this head: q[q_head_idx, :]
    q_ptrs = q_ptr + q_head_idx * stride_qh + d_offsets * stride_qd
    q = tl.load(q_ptrs)  # [HEAD_DIM]

    # Running online softmax state
    m_prev = -float("inf")
    l_prev = 0.0
    acc = tl.zeros([HEAD_DIM], dtype=tl.float32)

    for b_idx in range(num_blocks_for_seq):
        physical_block_id = tl.load(block_table_ptr + b_idx)

        # 1. Load K block [BLOCK_SIZE, HEAD_DIM]
        k_base_ptrs = (
            k_cache_ptr
            + physical_block_id * stride_kb
            + kv_head_idx * stride_kh
        )
        k_ptrs = (
            k_base_ptrs
            + tok_offsets[:, None] * stride_kt
            + d_offsets[None, :] * stride_kd
        )
        token_indices = b_idx * BLOCK_SIZE + tok_offsets
        token_mask = token_indices < seq_len

        k_block = tl.load(k_ptrs, mask=token_mask[:, None], other=0.0)

        # 2. Compute unnormalized scores for this block
        scores = tl.sum(q[None, :] * k_block, axis=1) * scale  # [BLOCK_SIZE]
        scores = tl.where(token_mask, scores, -float("inf"))

        # Block max
        m_curr = tl.max(scores, axis=0)
        m_new = tl.maximum(m_prev, m_curr)

        # 3. Rescaling factor for previous accumulator: exp(m_prev - m_new)
        alpha = tl.exp(m_prev - m_new)

        # 4. Numerators for current block: exp(scores - m_new)
        p = tl.exp(scores - m_new)
        p = tl.where(token_mask, p, 0.0)

        # 5. Update running denominator l_new = l_prev * alpha + sum(p)
        l_new = l_prev * alpha + tl.sum(p, axis=0)

        # 6. Load V block [BLOCK_SIZE, HEAD_DIM]
        v_base_ptrs = (
            v_cache_ptr
            + physical_block_id * stride_vb
            + kv_head_idx * stride_vh
        )
        v_ptrs = (
            v_base_ptrs
            + tok_offsets[:, None] * stride_vt
            + d_offsets[None, :] * stride_vd
        )
        v_block = tl.load(v_ptrs, mask=token_mask[:, None], other=0.0)

        # 7. Update running accumulator: acc = acc * alpha + sum(p * V)
        acc = acc * alpha + tl.sum(p[:, None] * v_block, axis=0)

        # Update running state for next iteration
        m_prev = m_new
        l_prev = l_new

    # Final normalization: acc / l_prev
    out = acc / l_prev
    out_ptrs = out_ptr + q_head_idx * stride_oh + d_offsets * stride_od
    tl.store(out_ptrs, out.to(tl.float16))


def paged_attention_decode_v2(
    q: torch.Tensor,            # [num_q_heads, head_dim] (fp16, cuda)
    k_cache: torch.Tensor,      # [num_blocks, block_size, num_kv_heads, head_dim] (fp16, cuda)
    v_cache: torch.Tensor,      # [num_blocks, block_size, num_kv_heads, head_dim] (fp16, cuda)
    block_table: torch.Tensor,  # 1D int32 cuda tensor of physical block IDs
    seq_len: int,
    scale: float,
) -> torch.Tensor:
    """
    Triton v2 decode paged-attention using single-pass online softmax.
    Supports GQA (num_q_heads % num_kv_heads == 0).
    """
    assert q.is_cuda and k_cache.is_cuda and v_cache.is_cuda and block_table.is_cuda
    assert q.dtype == torch.float16 and k_cache.dtype == torch.float16 and v_cache.dtype == torch.float16
    assert block_table.dtype == torch.int32

    num_q_heads, head_dim = q.shape
    num_blocks, block_size, num_kv_heads, head_dim_k = k_cache.shape
    assert head_dim == head_dim_k, f"Head dimensions must match: {head_dim} vs {head_dim_k}"
    assert num_q_heads % num_kv_heads == 0, (
        f"num_q_heads ({num_q_heads}) must be divisible by num_kv_heads ({num_kv_heads})"
    )
    group_size = num_q_heads // num_kv_heads

    num_blocks_for_seq = (seq_len + block_size - 1) // block_size
    assert block_table.shape[0] >= num_blocks_for_seq, (
        f"block_table size ({block_table.shape[0]}) smaller than needed ({num_blocks_for_seq})"
    )

    out = torch.empty_like(q)

    grid = (num_q_heads,)
    _paged_attention_decode_v2_kernel[grid](
        q,
        k_cache,
        v_cache,
        block_table,
        out,
        # Strides
        q.stride(0), q.stride(1),
        k_cache.stride(0), k_cache.stride(1), k_cache.stride(2), k_cache.stride(3),
        v_cache.stride(0), v_cache.stride(1), v_cache.stride(2), v_cache.stride(3),
        out.stride(0), out.stride(1),
        # Meta
        scale,
        seq_len,
        num_blocks_for_seq,
        group_size=group_size,
        BLOCK_SIZE=block_size,
        HEAD_DIM=head_dim,
    )
    return out


def pad_block_tables(
    block_tables: list[list[int]] | list[any],
    device: torch.device | str = "cuda",
) -> torch.Tensor:
    """
    Converts a list-of-lists (or list of BlockTable objects with .physical_block_ids)
    into a 2D padded int32 CUDA tensor of shape [num_seqs, max_blocks_per_seq].
    Empty or unused positions are padded with 0 (or any valid dummy block ID,
    since the kernel masks out-of-bounds blocks via seq_lens).
    """
    num_seqs = len(block_tables)
    if num_seqs == 0:
        return torch.empty((0, 0), dtype=torch.int32, device=device)

    # Extract integer lists
    raw_lists: list[list[int]] = []
    for bt in block_tables:
        if hasattr(bt, "physical_block_ids"):
            raw_lists.append(list(bt.physical_block_ids))
        elif isinstance(bt, (list, tuple)):
            raw_lists.append(list(bt))
        else:
            raise TypeError(f"Unsupported block table type: {type(bt)}")

    max_len = max((len(l) for l in raw_lists), default=0)
    max_len = max(max_len, 1)  # Ensure at least 1 column for valid 2D tensor

    padded = torch.zeros((num_seqs, max_len), dtype=torch.int32, device=device)
    for i, row in enumerate(raw_lists):
        if row:
            padded[i, :len(row)] = torch.tensor(row, dtype=torch.int32, device=device)

    return padded


@triton.jit
def _paged_attention_decode_batched_kernel(
    # Pointers
    q_ptr,                  # [num_seqs, num_q_heads, head_dim]
    k_cache_ptr,            # [num_blocks, block_size, num_kv_heads, head_dim]
    v_cache_ptr,            # [num_blocks, block_size, num_kv_heads, head_dim]
    block_tables_ptr,       # [num_seqs, max_blocks_per_seq] (2D int32)
    seq_lens_ptr,           # [num_seqs] (1D int32)
    out_ptr,                # [num_seqs, num_q_heads, head_dim]
    # Strides for Q: [num_seqs, num_q_heads, head_dim]
    stride_qs, stride_qh, stride_qd,
    # Strides for K cache: [num_blocks, block_size, num_kv_heads, head_dim]
    stride_kb, stride_kt, stride_kh, stride_kd,
    # Strides for V cache: [num_blocks, block_size, num_kv_heads, head_dim]
    stride_vb, stride_vt, stride_vh, stride_vd,
    # Strides for Block Tables: [num_seqs, max_blocks_per_seq]
    stride_bts, stride_btb,
    # Strides for Out: [num_seqs, num_q_heads, head_dim]
    stride_os, stride_oh, stride_od,
    # Meta scalars
    scale,
    group_size: tl.constexpr,   # num_q_heads // num_kv_heads
    BLOCK_SIZE: tl.constexpr,   # cache block size in physical pool (e.g. 16)
    HEAD_DIM: tl.constexpr,     # channels per head (e.g. 64)
    TILE_SIZE: tl.constexpr,    # tokens processed per iteration (e.g. 16 or 64)
):
    """
    Batched Kernel: 2D Grid = (num_seqs, num_q_heads).
    Each program computes attention for one (seq_idx, q_head_idx) pair.
    Online single-pass softmax across that sequence's logical blocks.
    Supports TILE_SIZE multiples of BLOCK_SIZE (e.g. TILE_SIZE=64 across 16-token blocks)
    for reduced loop trip overhead on large sequences.
    """
    seq_idx = tl.program_id(0)
    q_head_idx = tl.program_id(1)
    kv_head_idx = q_head_idx // group_size

    # Load this sequence's actual token count
    seq_len = tl.load(seq_lens_ptr + seq_idx)
    num_tiles = (seq_len + TILE_SIZE - 1) // TILE_SIZE

    d_offsets = tl.arange(0, HEAD_DIM)
    tile_tok_offsets = tl.arange(0, TILE_SIZE)

    # Pointer to Q for this sequence and query head: q[seq_idx, q_head_idx, :]
    q_ptrs = (
        q_ptr
        + seq_idx * stride_qs
        + q_head_idx * stride_qh
        + d_offsets * stride_qd
    )
    q = tl.load(q_ptrs)  # [HEAD_DIM]

    # Running online softmax state
    m_prev = -float("inf")
    l_prev = 0.0
    acc = tl.zeros([HEAD_DIM], dtype=tl.float32)

    # Base pointer for this sequence's row in the block table
    seq_bt_ptr = block_tables_ptr + seq_idx * stride_bts

    for tile_idx in range(num_tiles):
        token_indices = tile_idx * TILE_SIZE + tile_tok_offsets
        token_mask = token_indices < seq_len

        logical_block_indices = token_indices // BLOCK_SIZE
        offsets_in_block = token_indices % BLOCK_SIZE

        phys_ptrs = seq_bt_ptr + logical_block_indices * stride_btb
        max_blocks = (seq_len + BLOCK_SIZE - 1) // BLOCK_SIZE
        valid_block_mask = (logical_block_indices < max_blocks) & token_mask
        phys_ids = tl.load(phys_ptrs, mask=valid_block_mask, other=0)

        # 1. Pointer arithmetic for K cache [TILE_SIZE, HEAD_DIM]
        k_ptrs = (
            k_cache_ptr
            + phys_ids[:, None] * stride_kb
            + offsets_in_block[:, None] * stride_kt
            + kv_head_idx * stride_kh
            + d_offsets[None, :] * stride_kd
        )
        k_tile = tl.load(k_ptrs, mask=token_mask[:, None], other=0.0)

        # 2. Compute unnormalized scores
        scores = tl.sum(q[None, :] * k_tile, axis=1) * scale
        scores = tl.where(token_mask, scores, -float("inf"))

        # 3. New running maximum and rescaling factor
        m_curr = tl.max(scores, axis=0)
        m_new = tl.maximum(m_prev, m_curr)
        alpha = tl.exp(m_prev - m_new)

        # 4. Numerators for current block
        p = tl.exp(scores - m_new)
        p = tl.where(token_mask, p, 0.0)

        # 5. Update running denominator
        l_new = l_prev * alpha + tl.sum(p, axis=0)

        # 6. Pointer arithmetic for V cache [TILE_SIZE, HEAD_DIM]
        v_ptrs = (
            v_cache_ptr
            + phys_ids[:, None] * stride_vb
            + offsets_in_block[:, None] * stride_vt
            + kv_head_idx * stride_vh
            + d_offsets[None, :] * stride_vd
        )
        v_tile = tl.load(v_ptrs, mask=token_mask[:, None], other=0.0)

        # 7. Rescale and accumulate value vectors
        acc = acc * alpha + tl.sum(p[:, None] * v_tile, axis=0)

        m_prev = m_new
        l_prev = l_new

    # Final normalization
    out = acc / l_prev
    out_ptrs = (
        out_ptr
        + seq_idx * stride_os
        + q_head_idx * stride_oh
        + d_offsets * stride_od
    )
    tl.store(out_ptrs, out.to(tl.float16))


def paged_attention_decode_batched(
    q: torch.Tensor,            # [num_seqs, num_q_heads, head_dim] (fp16, cuda)
    k_cache: torch.Tensor,      # [num_blocks, block_size, num_kv_heads, head_dim] (fp16, cuda)
    v_cache: torch.Tensor,      # [num_blocks, block_size, num_kv_heads, head_dim] (fp16, cuda)
    block_tables: torch.Tensor, # [num_seqs, max_blocks_per_seq] (int32, cuda)
    seq_lens: torch.Tensor,     # [num_seqs] (int32, cuda)
    scale: float,
) -> torch.Tensor:
    """
    Batched Triton decode paged-attention for multiple sequences simultaneously.
    Grid = (num_seqs, num_q_heads).
    Supports GQA and variable sequence lengths.
    Selects optimal tile size and warp count based on batch size and sequence length.
    """
    assert q.is_cuda and k_cache.is_cuda and v_cache.is_cuda
    assert block_tables.is_cuda and seq_lens.is_cuda
    assert q.dtype == torch.float16 and k_cache.dtype == torch.float16 and v_cache.dtype == torch.float16
    assert block_tables.dtype == torch.int32 and seq_lens.dtype == torch.int32

    num_seqs, num_q_heads, head_dim = q.shape
    num_blocks, block_size, num_kv_heads, head_dim_k = k_cache.shape
    assert head_dim == head_dim_k, f"Head dimensions must match: {head_dim} vs {head_dim_k}"
    assert num_q_heads % num_kv_heads == 0, (
        f"num_q_heads ({num_q_heads}) must be divisible by num_kv_heads ({num_kv_heads})"
    )
    assert block_tables.shape[0] == num_seqs
    assert seq_lens.shape[0] == num_seqs

    group_size = num_q_heads // num_kv_heads
    out = torch.empty_like(q)

    # Adaptive config promotion:
    # Large workloads (num_seqs >= 16) benefit from TILE_SIZE=64 and 8 warps.
    # Small workloads (num_seqs < 16) minimize launch floor overhead with TILE_SIZE=16 and 4 warps.
    if num_seqs >= 16:
        tile_size = 64
        num_warps = 8
    else:
        tile_size = 16
        num_warps = 4

    grid = (num_seqs, num_q_heads)
    _paged_attention_decode_batched_kernel[grid](
        q,
        k_cache,
        v_cache,
        block_tables,
        seq_lens,
        out,
        # Strides
        q.stride(0), q.stride(1), q.stride(2),
        k_cache.stride(0), k_cache.stride(1), k_cache.stride(2), k_cache.stride(3),
        v_cache.stride(0), v_cache.stride(1), v_cache.stride(2), v_cache.stride(3),
        block_tables.stride(0), block_tables.stride(1),
        out.stride(0), out.stride(1), out.stride(2),
        # Meta
        scale,
        group_size=group_size,
        BLOCK_SIZE=block_size,
        HEAD_DIM=head_dim,
        TILE_SIZE=tile_size,
        num_warps=num_warps,
        num_stages=2,
    )
    return out
