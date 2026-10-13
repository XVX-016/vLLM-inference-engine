"""
Benchmark with L2 Cache Flushes and Fair Native GQA SDPA Comparison.
Fixes:
1. Flushes L2 cache between iterations via a 128 MB buffer write to measure DRAM bandwidth.
2. Fair SDPA comparison using enable_gqa=True (no repeat_interleave overhead).
3. Evaluates promoted batched Triton kernel across (B=1,4,16,32) and (S=128,512,2048).
"""
import os
import sys
import math
import random
from typing import Dict, List

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

from engine.kernels.paged_attention import (
    paged_attention_decode_batched,
    pad_block_tables,
)

PEAK_BANDWIDTH_GB_S = 256.0  # RTX 4060 Laptop peak DRAM bandwidth

# Preallocate a 128 MB buffer on GPU for L2 flushing (RTX 4060 has 32 MB L2 cache)
FLUSH_BUFFER = torch.empty(32 * 1024 * 1024, dtype=torch.int32, device="cuda")

def flush_l2_cache():
    """Flushes GPU L2 cache by writing to a large scratch buffer."""
    FLUSH_BUFFER.zero_()


def time_fn_with_l2_flush(fn, warmup: int = 15, iters: int = 35) -> float:
    """Measures execution time with L2 cache cleared before each timed launch."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    start_events = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    end_events = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]

    for i in range(iters):
        flush_l2_cache()
        start_events[i].record()
        fn()
        end_events[i].record()

    torch.cuda.synchronize()
    times = [start_events[i].elapsed_time(end_events[i]) for i in range(iters)]
    times.sort()
    return times[len(times) // 2]  # median


def run_clean_benchmark():
    device = "cuda"
    batch_sizes = [1, 4, 16, 32]
    seq_lengths = [128, 512, 2048]

    num_q_heads = 14
    num_kv_heads = 2
    head_dim = 64
    block_size = 16
    scale = 1.0 / math.sqrt(head_dim)

    results = []

    print("=" * 90)
    print("  RIGOROUS BENCHMARK: L2 CACHE FLUSHED + FAIR NATIVE GQA SDPA")
    print(f"  Device: {torch.cuda.get_device_name(0)}")
    print(f"  Peak DRAM Bandwidth: {PEAK_BANDWIDTH_GB_S:.1f} GB/s")
    print("=" * 90)

    for B in batch_sizes:
        for S in seq_lengths:
            q = torch.randn(B, num_q_heads, head_dim, dtype=torch.float16, device=device)

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

            # Contiguous K/V for fair SDPA comparison using native enable_gqa=True
            # Shape for SDPA: [B, num_kv_heads, S, head_dim]
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

            # q shape for SDPA: [B, num_q_heads, 1, head_dim]
            sdpa_q = q.unsqueeze(2)

            # 1. Paged Triton batched kernel (with L2 cache flush)
            t_triton = time_fn_with_l2_flush(
                lambda: paged_attention_decode_batched(q, k_cache, v_cache, padded_bt, seq_lens_t, scale)
            )

            # 2. Fair Native SDPA with enable_gqa=True (with L2 cache flush)
            t_sdpa = time_fn_with_l2_flush(
                lambda: F.scaled_dot_product_attention(sdpa_q, gathered_k, gathered_v, enable_gqa=True)
            )

            # Effective DRAM Bandwidth calculation
            total_kv_bytes = B * S * num_kv_heads * head_dim * 2 * 2  # fp16 (2 bytes) * 2 (K + V)
            total_qo_bytes = B * num_q_heads * head_dim * 2 * 2       # fp16 * 2 (Q read + Out write)
            total_bytes = total_kv_bytes + total_qo_bytes

            bw_triton_gb_s = (total_bytes / (t_triton * 1e-3)) / 1e9
            bw_sdpa_gb_s = (total_bytes / (t_sdpa * 1e-3)) / 1e9
            pct_peak_triton = (bw_triton_gb_s / PEAK_BANDWIDTH_GB_S) * 100.0

            row = {
                "batch_size": B,
                "seq_len": S,
                "t_triton_ms": t_triton,
                "t_sdpa_ms": t_sdpa,
                "bw_triton_gb_s": bw_triton_gb_s,
                "bw_sdpa_gb_s": bw_sdpa_gb_s,
                "pct_peak_triton": pct_peak_triton,
                "ratio_vs_sdpa": t_sdpa / t_triton,
            }
            results.append(row)
            print(f"B={B:<2} S={S:<4} | Triton: {t_triton:.4f} ms ({bw_triton_gb_s:.1f} GB/s, {pct_peak_triton:.1f}% peak) | SDPA (GQA): {t_sdpa:.4f} ms | Ratio: {row['ratio_vs_sdpa']:.2f}x")

    print("\n" + "=" * 105)
    print("                     RIGOROUS BENCHMARK RESULTS (DRAM L2-FLUSHED)")
    print("=" * 105)
    header = "| Batch | SeqLen | Total Data | Triton Paged (ms) | SDPA Contig (ms) | Triton DRAM BW | % Peak DRAM | Ratio vs SDPA |"
    sep    = "|-------|--------|------------|-------------------|------------------|----------------|-------------|---------------|"
    print(header)
    print(sep)
    for r in results:
        mb = (r['batch_size'] * r['seq_len'] * num_kv_heads * head_dim * 4) / (1024 * 1024)
        print(
            f"| {r['batch_size']:<5} "
            f"| {r['seq_len']:<6} "
            f"| {mb:<7.2f} MB "
            f"| {r['t_triton_ms']:<17.4f} "
            f"| {r['t_sdpa_ms']:<16.4f} "
            f"| {r['bw_triton_gb_s']:<11.1f} GB/s "
            f"| {r['pct_peak_triton']:<8.1f}% "
            f"| {r['ratio_vs_sdpa']:<10.2f}x |"
        )

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.2), dpi=150)
    colors = {128: "#2ecc71", 512: "#3498db", 2048: "#e74c3c"}

    # Latency comparison
    ax1 = axes[0]
    for s in [128, 512, 2048]:
        sub = [r for r in results if r["seq_len"] == s]
        b_vals = [r["batch_size"] for r in sub]
        t_tri = [r["t_triton_ms"] for r in sub]
        t_sdp = [r["t_sdpa_ms"] for r in sub]
        ax1.plot(b_vals, t_tri, marker="o", linewidth=2, color=colors[s], label=f"Triton Paged (S={s})")
        ax1.plot(b_vals, t_sdp, marker="s", linestyle="--", linewidth=1.5, color=colors[s], alpha=0.7, label=f"SDPA GQA (S={s})")

    ax1.set_xscale("log", base=2)
    ax1.set_yscale("log")
    ax1.set_xlabel("Batch Size", fontsize=11, fontweight="bold")
    ax1.set_ylabel("Latency (ms, log scale)", fontsize=11, fontweight="bold")
    ax1.set_title("Decode Latency (L2 Flushed): Paged vs Native SDPA", fontsize=11, fontweight="bold")
    ax1.set_xticks([1, 4, 16, 32])
    ax1.get_xaxis().set_major_formatter(plt.ScalarFormatter())
    ax1.grid(True, which="both", linestyle=":", alpha=0.6)
    ax1.legend(fontsize=8, loc="upper left")

    # True DRAM Bandwidth % Peak
    ax2 = axes[1]
    bw_pcts = [r["pct_peak_triton"] for r in results]
    labels = [f"B={r['batch_size']}\nS={r['seq_len']}" for r in results]
    bars = ax2.bar(range(len(results)), bw_pcts, color="#2980b9", width=0.6, edgecolor="black", alpha=0.85)
    for bar in bars:
        h = bar.get_height()
        ax2.annotate(f"{h:.1f}%", xy=(bar.get_x() + bar.get_width() / 2, h), xytext=(0, 3),
                     textcoords="offset points", ha="center", va="bottom", fontsize=7.5, fontweight="bold")
    ax2.set_ylabel("% of Peak DRAM Bandwidth (256 GB/s)", fontsize=11, fontweight="bold")
    ax2.set_xlabel("Workload (Batch & SeqLen)", fontsize=11, fontweight="bold")
    ax2.set_title("True DRAM Bandwidth Utilization (L2 Cache Flushed)", fontsize=11, fontweight="bold")
    ax2.set_xticks(range(len(results)))
    ax2.set_xticklabels(labels, fontsize=8)
    ax2.set_ylim(0, max(bw_pcts) * 1.3)
    ax2.grid(axis="y", linestyle=":", alpha=0.6)

    plt.tight_layout()
    out_img = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "benchmark_l2_flushed_sdpa_fair.png"))
    plt.savefig(out_img)
    print(f"\nSaved plot to: {out_img}")

if __name__ == "__main__":
    run_clean_benchmark()
