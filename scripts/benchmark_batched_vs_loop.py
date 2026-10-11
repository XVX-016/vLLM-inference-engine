"""
Benchmark: paged_attention_decode_batched (one 2D Triton launch)
vs. per-sequence paged_attention_decode_v2 called in a Python loop.

Run with:
    python scripts/benchmark_batched_vs_loop.py

Outputs median latency and speedup across four scenarios.
"""
import os
import random
import sys
import time

# Ensure project root is on sys.path when run directly
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch

from engine.kernels.paged_attention import (
    pad_block_tables,
    paged_attention_decode_batched,
    paged_attention_decode_v2,
)

WARMUP_ITERS = 10
BENCH_ITERS = 100


def build_scenario(
    seq_lens_list: list,
    num_q_heads: int = 14,
    num_kv_heads: int = 2,
    head_dim: int = 64,
    block_size: int = 16,
    seed: int = 42,
):
    torch.manual_seed(seed)
    random.seed(seed)

    num_seqs = len(seq_lens_list)
    total_logical = sum((s + block_size - 1) // block_size for s in seq_lens_list)
    num_physical_blocks = total_logical + 32

    k_cache = torch.randn(
        num_physical_blocks, block_size, num_kv_heads, head_dim,
        device="cuda", dtype=torch.float16
    )
    v_cache = torch.randn(
        num_physical_blocks, block_size, num_kv_heads, head_dim,
        device="cuda", dtype=torch.float16
    )
    q = torch.randn(num_seqs, num_q_heads, head_dim, device="cuda", dtype=torch.float16)

    block_tables_list = []
    used_physical = set()

    for seq_len in seq_lens_list:
        n_blocks = (seq_len + block_size - 1) // block_size
        available = [i for i in range(num_physical_blocks) if i not in used_physical]
        physical_ids = random.sample(available, n_blocks)
        used_physical.update(physical_ids)
        block_tables_list.append(physical_ids)

    padded_bt = pad_block_tables(block_tables_list, device="cuda")
    seq_lens_tensor = torch.tensor(seq_lens_list, device="cuda", dtype=torch.int32)
    scale = head_dim ** -0.5

    return {
        "q": q,
        "k_cache": k_cache,
        "v_cache": v_cache,
        "block_tables_list": block_tables_list,
        "padded_bt": padded_bt,
        "seq_lens_list": seq_lens_list,
        "seq_lens_tensor": seq_lens_tensor,
        "scale": scale,
    }


def bench_batched(scenario, n_iters):
    q = scenario["q"]
    k_cache = scenario["k_cache"]
    v_cache = scenario["v_cache"]
    padded_bt = scenario["padded_bt"]
    seq_lens_tensor = scenario["seq_lens_tensor"]
    scale = scenario["scale"]

    times = []
    for _ in range(n_iters):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        paged_attention_decode_batched(q, k_cache, v_cache, padded_bt, seq_lens_tensor, scale)
        torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)
    return times


def bench_loop(scenario, n_iters):
    q = scenario["q"]
    k_cache = scenario["k_cache"]
    v_cache = scenario["v_cache"]
    block_tables_list = scenario["block_tables_list"]
    seq_lens_list = scenario["seq_lens_list"]
    scale = scenario["scale"]
    num_seqs = len(seq_lens_list)

    # Pre-convert block tables to GPU tensors once (outside timing)
    block_table_tensors = [
        torch.tensor(block_tables_list[i], device="cuda", dtype=torch.int32)
        for i in range(num_seqs)
    ]

    times = []
    for _ in range(n_iters):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for i in range(num_seqs):
            paged_attention_decode_v2(
                q=q[i],
                k_cache=k_cache,
                v_cache=v_cache,
                block_table=block_table_tensors[i],
                seq_len=seq_lens_list[i],
                scale=scale,
            )
        torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)
    return times


def median_ms(times):
    s = sorted(times)
    n = len(s)
    mid = (s[n // 2 - 1] + s[n // 2]) / 2 * 1000 if n % 2 == 0 else s[n // 2] * 1000
    return mid


def run_benchmark(label, seq_lens_list):
    num_seqs = len(seq_lens_list)
    print(f"\n{'=' * 62}")
    print(f"  {label}")
    print(f"  seqs={num_seqs}  lengths={seq_lens_list}")
    print(f"  total_tokens={sum(seq_lens_list)}")
    print(f"{'=' * 62}")

    scenario = build_scenario(seq_lens_list)

    print("  [warmup] batched ...", end="", flush=True)
    bench_batched(scenario, WARMUP_ITERS)
    print(" done")

    print("  [warmup] loop ...", end="", flush=True)
    bench_loop(scenario, WARMUP_ITERS)
    print(" done")

    print(f"  [bench ] {BENCH_ITERS} iters ...", end="", flush=True)
    loop_times = bench_loop(scenario, BENCH_ITERS)
    batched_times = bench_batched(scenario, BENCH_ITERS)
    print(" done")

    loop_med = median_ms(loop_times)
    batched_med = median_ms(batched_times)
    speedup = loop_med / batched_med if batched_med > 0 else float("inf")

    print(f"\n  Batched  (1 launch):         {batched_med:8.3f} ms")
    print(f"  Loop     ({num_seqs:2d} launches):      {loop_med:8.3f} ms")
    print(f"  Speedup:                     {speedup:8.2f}x")


def main():
    if not torch.cuda.is_available():
        print("No CUDA device found, skipping benchmark.")
        return

    device_name = torch.cuda.get_device_name(0)
    print(f"\nBenchmark: paged_attention_decode_batched vs per-sequence loop")
    print(f"Device:  {device_name}")
    print(f"Warmup:  {WARMUP_ITERS} iters    Bench: {BENCH_ITERS} iters")

    # 1. Mixed extreme lengths — severe load imbalance
    #    seq0 processes 1 block, seq3 processes 125 blocks.
    #    SMs assigned to seq0 finish instantly and sit idle while seq3
    #    completes, wasting hardware resources in both modes.
    run_benchmark(
        label="Mixed lengths (1, 17, 500, 2000 tokens)",
        seq_lens_list=[1, 17, 500, 2000],
    )

    # 2. Uniform lengths — balanced, SMs do equal work, best case
    run_benchmark(
        label="Uniform lengths (4 x 512 tokens)",
        seq_lens_list=[512, 512, 512, 512],
    )

    # 3. Larger batch, short sequences
    run_benchmark(
        label="Large batch, short seqs (8 x 64 tokens)",
        seq_lens_list=[64] * 8,
    )

    # 4. Larger batch, long sequences
    run_benchmark(
        label="Large batch, long seqs (8 x 512 tokens)",
        seq_lens_list=[512] * 8,
    )


if __name__ == "__main__":
    main()
