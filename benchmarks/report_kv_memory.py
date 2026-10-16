"""
KV-memory evaluation and comparative report generator for Paged vs. Naive Serving.

Evaluates both systems under the exact same memory budget (e.g. 16 MB or 32 MB):
1. Naive server:
   - Reserved KV bytes (running_slots * max_seq_len * bytes_per_token)
   - Bytes actually holding tokens (tokens_stored * bytes_per_token)
   - Sampled every 100 ms during a run.
2. Paged engine:
   - Allocated blocks * block_bytes
   - Tokens stored * bytes_per_token
   - Separates internal fragmentation (last partial block) from truly free blocks.
3. Reports:
   - Average KV utilization (used / reserved) for each system.
   - Max concurrent running requests each achieved under the same budget.
   - Analytic check of the paged system's worst-case waste per sequence (< 1 block).
4. Runs the same Poisson trace at a load that saturates the naive system.
5. Produces a markdown results table and a comparative bar chart.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import subprocess
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np

from benchmarks.bench_serving import MetricSample, RunSummary, run_trace_replay
from benchmarks.workload_trace import (
    TraceRequest,
    generate_workload_trace,
    load_trace_from_json,
    save_trace_to_json,
)
from engine.serving.api_server import calculate_paged_blocks_from_budget
from engine.serving.naive_server import calculate_naive_slots_from_budget


def compute_kv_memory_stats(
    samples: List[MetricSample],
    system_type: str,
    block_size: int = 16,
    bytes_per_token: int = 12288,
) -> Dict[str, Any]:
    """
    Computes detailed KV memory breakdown from 100ms periodic samples:
    - active_samples: samples where server was processing requests
    - total_kv_bytes: total memory pool budget
    - avg_reserved_bytes: average reserved/allocated memory
    - avg_used_bytes: average memory actively storing tokens
    - avg_internal_frag_bytes: (allocated - used) for paged engine, or (reserved - used) for naive
    - avg_utilization: used / reserved
    - max_concurrent: peak running requests observed
    """
    active_samples = [s for s in samples if s.running > 0 or s.allocated_kv_bytes > 0 or s.reserved_kv_bytes > 0]
    if not active_samples:
        return {
            "system_type": system_type,
            "avg_utilization": 0.0,
            "max_concurrent": 0,
            "avg_reserved_mb": 0.0,
            "avg_used_mb": 0.0,
            "avg_internal_frag_mb": 0.0,
            "avg_free_mb": 0.0,
        }

    max_concurrent = max(s.running for s in active_samples)

    if system_type.lower() == "naive":
        reserved_bytes_list = [
            (s.reserved_kv_bytes if s.reserved_kv_bytes > 0 else s.running * s.total_blocks * bytes_per_token)
            for s in active_samples
        ]
        used_bytes_list = [s.active_token_kv_bytes for s in active_samples]
        utilizations = [
            min(1.0, float(u / r)) if r > 0 else 0.0
            for u, r in zip(used_bytes_list, reserved_bytes_list)
        ]
        avg_reserved = float(np.mean(reserved_bytes_list))
        avg_used = float(np.mean(used_bytes_list))
        avg_internal_frag = max(0.0, avg_reserved - avg_used)
        total_pool_bytes = active_samples[0].total_kv_bytes if active_samples[0].total_kv_bytes > 0 else avg_reserved
        avg_free = max(0.0, total_pool_bytes - avg_reserved)
    else:
        # Paged engine
        bytes_per_block = bytes_per_token * block_size
        allocated_bytes_list = [
            (s.allocated_kv_bytes if s.allocated_kv_bytes > 0 else (s.total_blocks - s.free_blocks) * bytes_per_block)
            for s in active_samples
        ]
        used_bytes_list = [s.active_token_kv_bytes for s in active_samples]
        utilizations = [
            min(1.0, float(u / a)) if a > 0 else 0.0
            for u, a in zip(used_bytes_list, allocated_bytes_list)
        ]
        avg_reserved = float(np.mean(allocated_bytes_list))
        avg_used = float(np.mean(used_bytes_list))
        avg_internal_frag = max(0.0, avg_reserved - avg_used)
        total_pool_bytes = active_samples[0].total_kv_bytes if active_samples[0].total_kv_bytes > 0 else (active_samples[0].total_blocks * bytes_per_block)
        avg_free = max(0.0, total_pool_bytes - avg_reserved)

    avg_util = float(np.mean(utilizations)) if utilizations else 0.0

    return {
        "system_type": system_type,
        "avg_utilization": avg_util,
        "max_concurrent": max_concurrent,
        "avg_reserved_mb": avg_reserved / (1024 * 1024),
        "avg_used_mb": avg_used / (1024 * 1024),
        "avg_internal_frag_mb": avg_internal_frag / (1024 * 1024),
        "avg_free_mb": avg_free / (1024 * 1024),
        "total_pool_mb": total_pool_bytes / (1024 * 1024),
    }


def analytic_paged_worst_case_check(
    block_size: int,
    num_layers: int = 24,
    num_kv_heads: int = 2,
    head_dim: int = 64,
    dtype_bytes: int = 2,
) -> Dict[str, Any]:
    """
    Performs analytic check of the paged system's worst-case internal fragmentation.
    In PagedAttention, each sequence is allocated physical blocks of size `block_size`.
    The only unused memory inside an allocated sequence is the tail of its final block.
    Thus, worst-case waste per sequence is strictly less than 1 block:
    waste_tokens <= block_size - 1
    waste_bytes < block_size * bytes_per_token
    """
    bytes_per_token = 2 * num_layers * num_kv_heads * head_dim * dtype_bytes
    bytes_per_block = block_size * bytes_per_token
    max_wasted_tokens = block_size - 1
    max_wasted_bytes = max_wasted_tokens * bytes_per_token
    fraction_of_block = max_wasted_tokens / block_size

    return {
        "block_size": block_size,
        "bytes_per_token": bytes_per_token,
        "bytes_per_block": bytes_per_block,
        "max_wasted_tokens_per_seq": max_wasted_tokens,
        "max_wasted_bytes_per_seq": max_wasted_bytes,
        "max_wasted_kb_per_seq": max_wasted_bytes / 1024,
        "fraction_of_block": fraction_of_block,
        "property_holds": max_wasted_bytes < bytes_per_block,
    }


def generate_kv_comparison_chart(
    naive_stats: Dict[str, Any],
    paged_stats: Dict[str, Any],
    output_path: Path | str,
    budget_mb: float,
) -> None:
    """Produces comparative bar charts for KV memory utilization and breakdown."""
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), dpi=150)

    # 1. Left Chart: Average KV Utilization & Max Concurrent Running Slots
    systems = ["Naive Static Batching", "Paged Continuous Batching"]
    x = np.arange(len(systems))
    width = 0.35

    util_pcts = [naive_stats["avg_utilization"] * 100.0, paged_stats["avg_utilization"] * 100.0]
    concurrency = [naive_stats["max_concurrent"], paged_stats["max_concurrent"]]

    color_util = "#2b5c8f"
    color_conc = "#d95f02"

    rects1 = ax1.bar(x - width / 2, util_pcts, width, label="Avg KV Utilization (%)", color=color_util)
    ax1.set_ylabel("KV Cache Utilization (%)", color=color_util, fontweight="bold")
    ax1.set_ylim(0, 105)
    ax1.tick_params(axis="y", labelcolor=color_util)

    ax1_twin = ax1.twinx()
    rects2 = ax1_twin.bar(x + width / 2, concurrency, width, label="Max Concurrent Requests", color=color_conc)
    ax1_twin.set_ylabel("Concurrent Requests", color=color_conc, fontweight="bold")
    ax1_twin.set_ylim(0, max(concurrency + [5]) * 1.3)
    ax1_twin.tick_params(axis="y", labelcolor=color_conc)

    ax1.set_xticks(x)
    ax1.set_xticklabels(systems, fontweight="bold")
    ax1.set_title(f"KV Utilization & Max Concurrency\n(Budget = {budget_mb} MB)", fontsize=11, fontweight="bold")

    # Add data labels
    for r in rects1:
        h = r.get_height()
        ax1.annotate(f"{h:.1f}%", xy=(r.get_x() + r.get_width() / 2, h), xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontweight="bold")
    for r in rects2:
        h = r.get_height()
        ax1_twin.annotate(f"{int(h)}", xy=(r.get_x() + r.get_width() / 2, h), xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontweight="bold")

    # 2. Right Chart: Memory Breakdown (Used Tokens vs. Internal Fragmentation vs. Free)
    categories = systems
    used_mb = [naive_stats["avg_used_mb"], paged_stats["avg_used_mb"]]
    frag_mb = [naive_stats["avg_internal_frag_mb"], paged_stats["avg_internal_frag_mb"]]
    free_mb = [naive_stats["avg_free_mb"], paged_stats["avg_free_mb"]]

    b_width = 0.45
    p1 = ax2.bar(categories, used_mb, b_width, label="Tokens Stored (Useful KV)", color="#2ca02c")
    p2 = ax2.bar(categories, frag_mb, b_width, bottom=used_mb, label="Internal Fragmentation (Unused Reserved)", color="#d62728")
    bottom_for_free = [u + f for u, f in zip(used_mb, frag_mb)]
    p3 = ax2.bar(categories, free_mb, b_width, bottom=bottom_for_free, label="Truly Free Unallocated Pool", color="#bcbd22")

    ax2.set_ylabel("Memory (MB)", fontweight="bold")
    ax2.set_title(f"KV Memory Allocation Breakdown\n(Budget = {budget_mb} MB)", fontsize=11, fontweight="bold")
    ax2.legend(loc="upper right", frameon=True, fontsize=9)
    ax2.set_ylim(0, budget_mb * 1.15)

    fig.tight_layout()
    plt.savefig(output_path)
    plt.close()


async def run_comparative_experiment(
    budget_mb: float = 16.0,
    max_seq_len: int = 512,
    block_size: int = 16,
    num_requests: int = 24,
    arrival_rate: float = 10.0,
    seed: int = 42,
    out_dir: str = "benchmarks/results",
) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    """
    Executes fair comparison run:
    1. Spawns NaiveBatchServer with budget_mb.
    2. Replays trace and collects memory samples.
    3. Spawns Paged AsyncEngine with identical budget_mb.
    4. Replays identical trace and collects memory samples.
    """
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # Generate reproducible trace at saturating arrival rate
    trace = generate_workload_trace(
        num_requests=num_requests,
        arrival_rate=arrival_rate,
        min_prompt_len=16,
        max_prompt_len=128,
        min_output_len=8,
        max_output_len=64,
        seed=seed,
        ignore_eos=True,
    )
    trace_file = out_path / "saturation_trace.json"
    save_trace_to_json(trace, trace_file)

    port_naive = 8141
    port_paged = 8142

    # --- 1. Run NaiveBatchServer ---
    env_naive = os.environ.copy()
    env_naive["NAIVE_KV_MEMORY_MB"] = str(budget_mb)
    env_naive["VLLM_BLOCK_SIZE"] = str(block_size)

    server_naive = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "engine.serving.naive_server:app", "--port", str(port_naive)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=env_naive,
    )
    time.sleep(2.5)

    try:
        summary_naive, results_naive, samples_naive = await run_trace_replay(
            base_url=f"http://127.0.0.1:{port_naive}",
            trace=trace,
            offered_rate=arrival_rate,
            warmup_count=2,
            stream=True,
            slo_threshold_s=2.0,
        )
    finally:
        server_naive.terminate()
        server_naive.wait(timeout=5)

    # --- 2. Run Paged AsyncEngine ---
    env_paged = os.environ.copy()
    env_paged["VLLM_KV_MEMORY_MB"] = str(budget_mb)
    env_paged["VLLM_BLOCK_SIZE"] = str(block_size)

    server_paged = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "engine.serving.api_server:app", "--port", str(port_paged)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=env_paged,
    )
    time.sleep(2.5)

    try:
        summary_paged, results_paged, samples_paged = await run_trace_replay(
            base_url=f"http://127.0.0.1:{port_paged}",
            trace=trace,
            offered_rate=arrival_rate,
            warmup_count=2,
            stream=True,
            slo_threshold_s=2.0,
        )
    finally:
        server_paged.terminate()
        server_paged.wait(timeout=5)

    # Compute metrics breakdown
    naive_stats = compute_kv_memory_stats(samples_naive, system_type="Naive", block_size=block_size)
    paged_stats = compute_kv_memory_stats(samples_paged, system_type="Paged", block_size=block_size)
    analytic_check = analytic_paged_worst_case_check(block_size=block_size)

    # Generate comparative chart
    chart_path = out_path / "kv_memory_comparison.png"
    generate_kv_comparison_chart(naive_stats, paged_stats, chart_path, budget_mb=budget_mb)

    # Save raw report data
    report_data = {
        "budget_mb": budget_mb,
        "max_seq_len": max_seq_len,
        "block_size": block_size,
        "offered_rate_rps": arrival_rate,
        "naive_stats": naive_stats,
        "paged_stats": paged_stats,
        "analytic_check": analytic_check,
        "naive_summary": asdict(summary_naive),
        "paged_summary": asdict(summary_paged),
    }
    with open(out_path / "kv_memory_report.json", "w", encoding="utf-8") as f:
        json.dump(report_data, f, indent=2)

    return naive_stats, paged_stats, analytic_check


def generate_markdown_report_table(
    naive_stats: Dict[str, Any],
    paged_stats: Dict[str, Any],
    analytic_check: Dict[str, Any],
    budget_mb: float,
) -> str:
    """Formats experimental results into a clean markdown table."""
    table = f"""
### KV Memory Utilization & Concurrency Report (Budget: {budget_mb:.1f} MB)

| Metric | Naive Static Batching | Paged Continuous Batching | Impact / Advantage |
| :--- | :---: | :---: | :---: |
| **Max Concurrent Requests** | **{naive_stats['max_concurrent']}** | **{paged_stats['max_concurrent']}** | **{paged_stats['max_concurrent'] / max(naive_stats['max_concurrent'], 1):.1f}x higher concurrency** |
| **Average KV Utilization** | **{naive_stats['avg_utilization'] * 100:.1f}%** | **{paged_stats['avg_utilization'] * 100:.1f}%** | **+{ (paged_stats['avg_utilization'] - naive_stats['avg_utilization']) * 100:.1f}% effective memory** |
| **Tokens Stored (Active KV)** | {naive_stats['avg_used_mb']:.2f} MB | {paged_stats['avg_used_mb']:.2f} MB | Useful token representations in VRAM |
| **Internal Fragmentation** | {naive_stats['avg_internal_frag_mb']:.2f} MB | {paged_stats['avg_internal_frag_mb']:.2f} MB | Static slots vs. tail block overhead |
| **Truly Free Pool Capacity** | {naive_stats['avg_free_mb']:.2f} MB | {paged_stats['avg_free_mb']:.2f} MB | Available to admit new waiting requests |

#### Analytic Check: Worst-Case Waste per Sequence
- **Configured Block Size**: {analytic_check['block_size']} tokens ({analytic_check['bytes_per_block'] / 1024:.1f} KB per block)
- **Max Wasted Tokens per Sequence**: $\le {analytic_check['max_wasted_tokens_per_seq']}$ tokens ($< 1$ full physical block)
- **Max Wasted Bytes per Sequence**: ${analytic_check['max_wasted_kb_per_seq']:.1f}$ KB ($<{analytic_check['bytes_per_block'] / 1024:.1f}$ KB)
- **Analytical Bound Invariant**: `worst_case_waste < block_size` holds: **{analytic_check['property_holds']}**
"""
    return table


def main():
    parser = argparse.ArgumentParser(description="Run fair KV-memory comparison report")
    parser.add_argument("--budget-mb", type=float, default=16.0, help="KV memory budget in MB")
    parser.add_argument("--arrival-rate", type=float, default=12.0, help="Saturating arrival rate (req/s)")
    parser.add_argument("--num-requests", type=int, default=24)
    parser.add_argument("--output-dir", type=str, default="benchmarks/results")
    args = parser.parse_args()

    naive_stats, paged_stats, analytic_check = asyncio.run(
        run_comparative_experiment(
            budget_mb=args.budget_mb,
            arrival_rate=args.arrival_rate,
            num_requests=args.num_requests,
            out_dir=args.output_dir,
        )
    )

    report_md = generate_markdown_report_table(naive_stats, paged_stats, analytic_check, budget_mb=args.budget_mb)
    print(report_md)


if __name__ == "__main__":
    main()
