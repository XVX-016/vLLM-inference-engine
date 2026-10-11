"""
Comprehensive Benchmarking and Profiling Suite for Paged Attention.
Compares:
1. Batched Triton Kernel (Default v1/v2 configuration)
2. Autotuned Batched Triton Kernel (tuned warps, stages, tile_size)
3. Naive Python Reference (per-sequence gather + attention)
4. torch.nn.functional.scaled_dot_product_attention (PyTorch native SDPA on contiguous K/V)

Sweeps:
- Batch sizes: (1, 4, 16, 32)
- Sequence lengths: (128, 512, 2048)

Measures:
- Latency (ms) with CUDA events and warmup
- Effective memory bandwidth (GB/s and % of RTX 4060 peak ~256 GB/s)
- Speedup vs PyTorch baseline and Naive reference
- Generates summary tables and comparison plots.
"""
import os
import sys
import math
import time
import random
from typing import Dict, List, Tuple

# Ensure project root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import torch.nn.functional as F
import triton
import triton.language as tl
import matplotlib.pyplot as plt

from engine.kernels.paged_attention import (
    naive_reference_paged_attention,
    paged_attention_decode_batched,
    pad_block_tables,
)

# RTX 4060 Laptop GPU: 128-bit memory bus @ 16 Gbps GDDR6 = ~256 GB/s peak
PEAK_BANDWIDTH_GB_S = 256.0


# ---------------------------------------------------------------------------
# Autotuned Triton Paged Attention Kernel
# ---------------------------------------------------------------------------
@triton.autotune(
    configs=[
        triton.Config({"TILE_SIZE": 16}, num_warps=2, num_stages=2),
        triton.Config({"TILE_SIZE": 16}, num_warps=4, num_stages=2),
        triton.Config({"TILE_SIZE": 32}, num_warps=4, num_stages=2),
        triton.Config({"TILE_SIZE": 32}, num_warps=8, num_stages=2),
        triton.Config({"TILE_SIZE": 64}, num_warps=4, num_stages=2),
        triton.Config({"TILE_SIZE": 64}, num_warps=8, num_stages=2),
    ],
    key=["HEAD_DIM", "BLOCK_SIZE"],
)
@triton.jit
def _paged_attention_decode_batched_autotuned_kernel(
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
    HEAD_DIM: tl.constexpr,     # channels per head (e.g. 64)
    TILE_SIZE: tl.constexpr,    # tokens per loop iteration (16, 32, 64)
):
    seq_idx = tl.program_id(0)
    q_head_idx = tl.program_id(1)
    kv_head_idx = q_head_idx // group_size

    seq_len = tl.load(seq_lens_ptr + seq_idx)
    num_tiles = (seq_len + TILE_SIZE - 1) // TILE_SIZE

    d_offsets = tl.arange(0, HEAD_DIM)
    tile_tok_offsets = tl.arange(0, TILE_SIZE)

    # 1. Load Q: [HEAD_DIM]
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

        logical_block_indices = token_indices // BLOCK_SIZE
        offsets_in_block = token_indices % BLOCK_SIZE

        phys_ptrs = seq_bt_ptr + logical_block_indices * stride_btb
        max_blocks = (seq_len + BLOCK_SIZE - 1) // BLOCK_SIZE
        valid_block_mask = (logical_block_indices < max_blocks) & token_mask
        phys_ids = tl.load(phys_ptrs, mask=valid_block_mask, other=0)

        # K load: [TILE_SIZE, HEAD_DIM]
        k_ptrs = (
            k_cache_ptr
            + phys_ids[:, None] * stride_kb
            + offsets_in_block[:, None] * stride_kt
            + kv_head_idx * stride_kh
            + d_offsets[None, :] * stride_kd
        )
        k_tile = tl.load(k_ptrs, mask=token_mask[:, None], other=0.0)

        # Attention scores: Q * K^T
        scores = tl.sum(q[None, :] * k_tile, axis=1) * scale
        scores = tl.where(token_mask, scores, -float("inf"))

        m_curr = tl.max(scores, axis=0)
        m_new = tl.maximum(m_prev, m_curr)
        alpha = tl.exp(m_prev - m_new)

        p = tl.exp(scores - m_new)
        p = tl.where(token_mask, p, 0.0)

        l_new = l_prev * alpha + tl.sum(p, axis=0)

        # V load: [TILE_SIZE, HEAD_DIM]
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


def paged_attention_decode_batched_autotuned(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    block_tables: torch.Tensor,
    seq_lens: torch.Tensor,
    scale: float,
) -> torch.Tensor:
    num_seqs, num_q_heads, head_dim = q.shape
    _, block_size, num_kv_heads, _ = k_cache.shape
    group_size = num_q_heads // num_kv_heads
    out = torch.empty_like(q)

    grid = (num_seqs, num_q_heads)
    _paged_attention_decode_batched_autotuned_kernel[grid](
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
    )
    return out


# ---------------------------------------------------------------------------
# Benchmarking Helpers
# ---------------------------------------------------------------------------
def time_fn_cuda(fn, warmup: int = 15, iters: int = 40) -> float:
    """Measure median CUDA execution time in milliseconds using CUDA events."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    start_events = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    end_events = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]

    for i in range(iters):
        start_events[i].record()
        fn()
        end_events[i].record()

    torch.cuda.synchronize()
    times = [start_events[i].elapsed_time(end_events[i]) for i in range(iters)]
    times.sort()
    return times[len(times) // 2]  # median


def run_benchmark():
    device = "cuda"
    assert torch.cuda.is_available(), "CUDA device required"

    batch_sizes = [1, 4, 16, 32]
    seq_lengths = [128, 512, 2048]

    num_q_heads = 14
    num_kv_heads = 2
    head_dim = 64
    block_size = 16
    scale = 1.0 / math.sqrt(head_dim)

    results = []

    print("=" * 80)
    print("  PAGED ATTENTION DECODE BENCHMARK: BATCHED TRITON VS SDPA VS NAIVE")
    print(f"  Device: {torch.cuda.get_device_name(0)}")
    print(f"  Peak Bandwidth: {PEAK_BANDWIDTH_GB_S:.1f} GB/s")
    print("=" * 80)

    for B in batch_sizes:
        for S in seq_lengths:
            print(f"\n---> Evaluating Batch Size = {B}, Sequence Length = {S} ...")

            q = torch.randn(B, num_q_heads, head_dim, dtype=torch.float16, device=device)

            # Block tables with randomized physical block layout
            max_blocks_per_seq = (S + block_size - 1) // block_size
            total_blocks_needed = B * max_blocks_per_seq
            total_pool_blocks = total_blocks_needed + 128

            all_phys_ids = list(range(total_pool_blocks))
            random.seed(42 + B * 100 + S)
            random.shuffle(all_phys_ids)

            block_tables_list = []
            for b in range(B):
                start = b * max_blocks_per_seq
                block_tables_list.append(all_phys_ids[start : start + max_blocks_per_seq])

            k_cache = torch.randn(total_pool_blocks, block_size, num_kv_heads, head_dim, dtype=torch.float16, device=device)
            v_cache = torch.randn(total_pool_blocks, block_size, num_kv_heads, head_dim, dtype=torch.float16, device=device)

            padded_bt = pad_block_tables(block_tables_list, device=device)
            seq_lens_t = torch.tensor([S] * B, dtype=torch.int32, device=device)

            # Contiguous K/V for PyTorch SDPA comparison
            # Extract gathered K and V for SDPA
            # shape: [B, num_kv_heads, S, head_dim] -> repeated to [B, num_q_heads, S, head_dim]
            gathered_k = torch.zeros(B, num_kv_heads, S, head_dim, dtype=torch.float16, device=device)
            gathered_v = torch.zeros(B, num_kv_heads, S, head_dim, dtype=torch.float16, device=device)
            for b in range(B):
                for l_idx, p_id in enumerate(block_tables_list[b]):
                    tok_start = l_idx * block_size
                    tok_end = min(tok_start + block_size, S)
                    n_tok = tok_end - tok_start
                    if n_tok <= 0:
                        break
                    gathered_k[b, :, tok_start:tok_end, :] = k_cache[p_id, :n_tok].permute(1, 0, 2)
                    gathered_v[b, :, tok_start:tok_end, :] = v_cache[p_id, :n_tok].permute(1, 0, 2)

            # GQA repeat for SDPA: [B, num_q_heads, S, head_dim]
            gqa_ratio = num_q_heads // num_kv_heads
            sdpa_k = gathered_k.repeat_interleave(gqa_ratio, dim=1)
            sdpa_v = gathered_v.repeat_interleave(gqa_ratio, dim=1)
            sdpa_q = q.unsqueeze(2).permute(0, 1, 2, 3) # [B, num_q_heads, 1, head_dim]

            # 1. Batched Triton default
            t_triton_default = time_fn_cuda(
                lambda: paged_attention_decode_batched(q, k_cache, v_cache, padded_bt, seq_lens_t, scale)
            )

            # 2. Batched Triton autotuned
            t_triton_tuned = time_fn_cuda(
                lambda: paged_attention_decode_batched_autotuned(q, k_cache, v_cache, padded_bt, seq_lens_t, scale)
            )

            # 3. PyTorch SDPA on contiguous memory
            t_sdpa = time_fn_cuda(
                lambda: F.scaled_dot_product_attention(sdpa_q, sdpa_k, sdpa_v, is_causal=False)
            )

            # 4. Naive Python Reference (run only for moderate sizes to avoid slow runtimes)
            if B * S <= 16 * 512:
                t_naive = time_fn_cuda(
                    lambda: naive_reference_paged_attention(q, k_cache, v_cache, block_tables_list, [S] * B, scale),
                    warmup=3,
                    iters=10,
                )
            else:
                t_naive = float("nan")

            # Effective Memory Bandwidth Calculation
            # Total K/V read per sequence = S tokens * num_kv_heads * head_dim * 2 bytes (fp16) * 2 (K and V)
            # Total across batch B = B * S * num_kv_heads * head_dim * 2 * 2 bytes
            # Plus query and output write: B * num_q_heads * head_dim * 2 * 2
            total_kv_bytes = B * S * num_kv_heads * head_dim * 2 * 2
            total_qo_bytes = B * num_q_heads * head_dim * 2 * 2
            total_bytes = total_kv_bytes + total_qo_bytes

            bw_default_gb_s = (total_bytes / (t_triton_default * 1e-3)) / 1e9
            bw_tuned_gb_s = (total_bytes / (t_triton_tuned * 1e-3)) / 1e9
            bw_sdpa_gb_s = (total_bytes / (t_sdpa * 1e-3)) / 1e9

            pct_peak_default = (bw_default_gb_s / PEAK_BANDWIDTH_GB_S) * 100.0
            pct_peak_tuned = (bw_tuned_gb_s / PEAK_BANDWIDTH_GB_S) * 100.0

            row = {
                "batch_size": B,
                "seq_len": S,
                "t_default_ms": t_triton_default,
                "t_tuned_ms": t_triton_tuned,
                "t_sdpa_ms": t_sdpa,
                "t_naive_ms": t_naive,
                "bw_default_gb_s": bw_default_gb_s,
                "bw_tuned_gb_s": bw_tuned_gb_s,
                "pct_peak_default": pct_peak_default,
                "pct_peak_tuned": pct_peak_tuned,
                "speedup_vs_sdpa": t_sdpa / t_triton_tuned,
                "speedup_vs_naive": t_naive / t_triton_tuned if not math.isnan(t_naive) else float("nan"),
            }
            results.append(row)

            print(f"  Triton Default:   {t_triton_default:.4f} ms | {bw_default_gb_s:.1f} GB/s ({pct_peak_default:.1f}% peak)")
            print(f"  Triton Tuned:     {t_triton_tuned:.4f} ms | {bw_tuned_gb_s:.1f} GB/s ({pct_peak_tuned:.1f}% peak)")
            print(f"  PyTorch SDPA:     {t_sdpa:.4f} ms | {bw_sdpa_gb_s:.1f} GB/s")
            if not math.isnan(t_naive):
                print(f"  Naive Reference:  {t_naive:.4f} ms")

    # ---------------------------------------------------------------------------
    # Print Formatted Markdown Table
    # ---------------------------------------------------------------------------
    print("\n" + "=" * 115)
    print("                                    SUMMARY RESULTS TABLE")
    print("=" * 115)
    header = "| Batch | SeqLen | Triton Def (ms) | Triton Tuned (ms) | SDPA Contig (ms) | Naive (ms) | Tuned BW (GB/s) | % Peak BW | Speedup vs SDPA |"
    sep    = "|-------|--------|-----------------|-------------------|------------------|------------|-----------------|-----------|-----------------|"
    print(header)
    print(sep)
    for r in results:
        naive_str = f"{r['t_naive_ms']:.3f}" if not math.isnan(r['t_naive_ms']) else "N/A"
        line = (
            f"| {r['batch_size']:<5} "
            f"| {r['seq_len']:<6} "
            f"| {r['t_default_ms']:<15.4f} "
            f"| {r['t_tuned_ms']:<17.4f} "
            f"| {r['t_sdpa_ms']:<16.4f} "
            f"| {naive_str:<10} "
            f"| {r['bw_tuned_gb_s']:<15.1f} "
            f"| {r['pct_peak_tuned']:<9.1f}% "
            f"| {r['speedup_vs_sdpa']:<15.2f}x |"
        )
        print(line)

    # ---------------------------------------------------------------------------
    # Generate Plots
    # ---------------------------------------------------------------------------
    generate_plots(results)

    return results


def generate_plots(results: List[Dict]):
    """Generate latency and memory bandwidth scaling figures."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), dpi=150)
    
    # 1. Latency vs Batch Size for each sequence length
    ax1 = axes[0]
    colors = {128: "#2ecc71", 512: "#3498db", 2048: "#e74c3c"}
    
    for s in [128, 512, 2048]:
        sub = [r for r in results if r["seq_len"] == s]
        b_vals = [r["batch_size"] for r in sub]
        t_tuned = [r["t_tuned_ms"] for r in sub]
        t_sdpa = [r["t_sdpa_ms"] for r in sub]
        
        ax1.plot(b_vals, t_tuned, marker="o", linewidth=2, color=colors[s], label=f"Triton Tuned (S={s})")
        ax1.plot(b_vals, t_sdpa, marker="s", linestyle="--", linewidth=1.5, color=colors[s], alpha=0.7, label=f"SDPA Contig (S={s})")

    ax1.set_xscale("log", base=2)
    ax1.set_yscale("log")
    ax1.set_xlabel("Batch Size", fontsize=11, fontweight="bold")
    ax1.set_ylabel("Latency (ms, log scale)", fontsize=11, fontweight="bold")
    ax1.set_title("Decode Latency: Paged Triton vs Contiguous SDPA", fontsize=12, fontweight="bold")
    ax1.set_xticks([1, 4, 16, 32])
    ax1.get_xaxis().set_major_formatter(plt.ScalarFormatter())
    ax1.grid(True, which="both", linestyle=":", alpha=0.6)
    ax1.legend(fontsize=8, loc="upper left")

    # 2. Bandwidth Utilization (% of Peak)
    ax2 = axes[1]
    width = 0.25
    x_indices = range(len(results))
    
    bw_pcts = [r["pct_peak_tuned"] for r in results]
    labels = [f"B={r['batch_size']}\nS={r['seq_len']}" for r in results]
    
    bars = ax2.bar(range(len(results)), bw_pcts, color="#8e44ad", width=0.6, edgecolor="black", alpha=0.85)
    ax2.axhline(100.0, color="red", linestyle="--", linewidth=1.5, label="Peak Bandwidth (256 GB/s)")
    
    # Add values on top of bars
    for bar in bars:
        height = bar.get_height()
        ax2.annotate(f"{height:.1f}%",
                     xy=(bar.get_x() + bar.get_width() / 2, height),
                     xytext=(0, 3),  # 3 points vertical offset
                     textcoords="offset points",
                     ha="center", va="bottom", fontsize=7.5, fontweight="bold")

    ax2.set_ylabel("% of Peak Bandwidth (~256 GB/s)", fontsize=11, fontweight="bold")
    ax2.set_xlabel("Workload Configuration (Batch & SeqLen)", fontsize=11, fontweight="bold")
    ax2.set_title("Effective Memory Bandwidth Utilization (% Peak)", fontsize=12, fontweight="bold")
    ax2.set_xticks(range(len(results)))
    ax2.set_xticklabels(labels, fontsize=8)
    ax2.set_ylim(0, max(bw_pcts) * 1.25)
    ax2.grid(axis="y", linestyle=":", alpha=0.6)
    ax2.legend(fontsize=9, loc="upper left")

    plt.tight_layout()
    plot_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "benchmark_paged_attention_profile.png"))
    plt.savefig(plot_path)
    print(f"\nSaved profile plot to: {plot_path}")


if __name__ == "__main__":
    run_benchmark()
