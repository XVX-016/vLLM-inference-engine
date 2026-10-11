"""
Triton Paged Attention Autotuning Experiment.
Profiles configurations of (num_warps, num_stages, tile_size) across batch sizes and sequence lengths.
"""
import os
import sys
import math
import torch
import triton
import triton.language as tl

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from engine.kernels.paged_attention import pad_block_tables

@triton.jit
def _paged_attention_decode_batched_tune_kernel(
    q_ptr,                  # [num_seqs, num_q_heads, head_dim]
    k_cache_ptr,            # [num_blocks, block_size, num_kv_heads, head_dim]
    v_cache_ptr,            # [num_blocks, block_size, num_kv_heads, head_dim]
    block_tables_ptr,       # [num_seqs, max_blocks_per_seq] (2D int32)
    seq_lens_ptr,           # [num_seqs] (1D int32)
    out_ptr,                # [num_seqs, num_q_heads, head_dim]
    # Strides
    stride_qs, stride_qh, stride_qd,
    stride_kb, stride_kt, stride_kh, stride_kd,
    stride_vb, stride_vt, stride_vh, stride_vd,
    stride_bts, stride_btb,
    stride_os, stride_oh, stride_od,
    scale,
    group_size: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,   # cache block size in physical pool (e.g. 16)
    HEAD_DIM: tl.constexpr,
    TILE_SIZE: tl.constexpr,    # tokens processed per iteration: 16, 32, or 64
):
    seq_idx = tl.program_id(0)
    q_head_idx = tl.program_id(1)
    kv_head_idx = q_head_idx // group_size

    seq_len = tl.load(seq_lens_ptr + seq_idx)
    num_tiles = (seq_len + TILE_SIZE - 1) // TILE_SIZE

    d_offsets = tl.arange(0, HEAD_DIM)
    tile_tok_offsets = tl.arange(0, TILE_SIZE)

    # Load Q
    q_ptrs = (
        q_ptr
        + seq_idx * stride_qs
        + q_head_idx * stride_qh
        + d_offsets * stride_qd
    )
    q = tl.load(q_ptrs)

    m_prev = -float("inf")
    l_prev = 0.0
    acc = tl.zeros([HEAD_DIM], dtype=tl.float32)

    seq_bt_ptr = block_tables_ptr + seq_idx * stride_bts

    for tile_idx in range(num_tiles):
        token_indices = tile_idx * TILE_SIZE + tile_tok_offsets
        token_mask = token_indices < seq_len

        # Map token index to physical block and offset inside block
        # logical_block_id = token_index // BLOCK_SIZE
        # offset_in_block  = token_index % BLOCK_SIZE
        logical_block_indices = token_indices // BLOCK_SIZE
        offsets_in_block = token_indices % BLOCK_SIZE

        # Load physical block id from block table for each token in tile
        phys_ptrs = seq_bt_ptr + logical_block_indices * stride_btb
        # Mask out-of-bound logical block table reads
        max_blocks = (seq_len + BLOCK_SIZE - 1) // BLOCK_SIZE
        valid_block_mask = (logical_block_indices < max_blocks) & token_mask
        phys_ids = tl.load(phys_ptrs, mask=valid_block_mask, other=0)

        # K pointers: [TILE_SIZE, HEAD_DIM]
        k_ptrs = (
            k_cache_ptr
            + phys_ids[:, None] * stride_kb
            + offsets_in_block[:, None] * stride_kt
            + kv_head_idx * stride_kh
            + d_offsets[None, :] * stride_kd
        )
        k_tile = tl.load(k_ptrs, mask=token_mask[:, None], other=0.0)

        # Scores
        scores = tl.sum(q[None, :] * k_tile, axis=1) * scale
        scores = tl.where(token_mask, scores, -float("inf"))

        m_curr = tl.max(scores, axis=0)
        m_new = tl.maximum(m_prev, m_curr)
        alpha = tl.exp(m_prev - m_new)

        p = tl.exp(scores - m_new)
        p = tl.where(token_mask, p, 0.0)

        l_new = l_prev * alpha + tl.sum(p, axis=0)

        # V pointers: [TILE_SIZE, HEAD_DIM]
        v_ptrs = (
            v_cache_ptr
            + phys_ids[:, None] * stride_vb
            + offsets_in_block[:, None] * stride_vt
            + kv_head_idx * stride_vh
            + d_offsets[None, :] * stride_vd
        )
        v_tile = tl.load(v_ptrs, mask=token_mask[:, None], other=0.0)

        acc = acc * alpha + tl.sum(p[:, None] * v_tile, axis=0)

        m_prev = m_new
        l_prev = l_new

    out = acc / l_prev
    out_ptrs = (
        out_ptr
        + seq_idx * stride_os
        + q_head_idx * stride_oh
        + d_offsets * stride_od
    )
    tl.store(out_ptrs, out.to(tl.float16))


def benchmark_config(q, k_cache, v_cache, block_tables, seq_lens, scale, num_warps, num_stages, tile_size):
    num_seqs, num_q_heads, head_dim = q.shape
    _, block_size, num_kv_heads, _ = k_cache.shape
    group_size = num_q_heads // num_kv_heads
    out = torch.empty_like(q)

    grid = (num_seqs, num_q_heads)

    def run():
        _paged_attention_decode_batched_tune_kernel[grid](
            q, k_cache, v_cache, block_tables, seq_lens, out,
            q.stride(0), q.stride(1), q.stride(2),
            k_cache.stride(0), k_cache.stride(1), k_cache.stride(2), k_cache.stride(3),
            v_cache.stride(0), v_cache.stride(1), v_cache.stride(2), v_cache.stride(3),
            block_tables.stride(0), block_tables.stride(1),
            out.stride(0), out.stride(1), out.stride(2),
            scale,
            group_size=group_size,
            BLOCK_SIZE=block_size,
            HEAD_DIM=head_dim,
            TILE_SIZE=tile_size,
            num_warps=num_warps,
            num_stages=num_stages,
        )

    # Warmup
    for _ in range(10):
        run()
    torch.cuda.synchronize()

    # Time with CUDA events
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    iters = 50
    start_event.record()
    for _ in range(iters):
        run()
    end_event.record()
    torch.cuda.synchronize()

    return start_event.elapsed_time(end_event) / iters

if __name__ == "__main__":
    device = "cuda"
    batch_size = 16
    seq_len = 512
    num_q_heads = 14
    num_kv_heads = 2
    head_dim = 64
    block_size = 16

    q = torch.randn(batch_size, num_q_heads, head_dim, dtype=torch.float16, device=device)
    max_blocks = (seq_len + block_size - 1) // block_size
    num_blocks = batch_size * max_blocks + 64
    k_cache = torch.randn(num_blocks, block_size, num_kv_heads, head_dim, dtype=torch.float16, device=device)
    v_cache = torch.randn(num_blocks, block_size, num_kv_heads, head_dim, dtype=torch.float16, device=device)

    # Simple block tables
    bt_list = [list(range(i * max_blocks, (i + 1) * max_blocks)) for i in range(batch_size)]
    block_tables = torch.tensor(bt_list, dtype=torch.int32, device=device)
    seq_lens = torch.tensor([seq_len] * batch_size, dtype=torch.int32, device=device)
    scale = 1.0 / math.sqrt(head_dim)

    print("Testing autotune configs...")
    configs = [
        (2, 2, 16),
        (4, 2, 16),
        (4, 3, 16),
        (4, 2, 32),
        (4, 2, 64),
        (8, 2, 32),
        (8, 2, 64),
    ]

    for nw, ns, ts in configs:
        ms = benchmark_config(q, k_cache, v_cache, block_tables, seq_lens, scale, nw, ns, ts)
        print(f"num_warps={nw}, num_stages={ns}, tile_size={ts}: {ms:.4f} ms")
