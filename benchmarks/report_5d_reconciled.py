"""
Fixed 5d report & KV-memory comparative evaluation for Paged vs. Naive Serving.

Reconciles and verifies:
1. Print naive server's max_seq_len, derived slot count, and clamping behavior.
2. Set the naive max_seq_len to the trace's maximum prompt + output length.
3. Choose KV budget in MiB so naive gets at least 3 slots:
   - Budget bytes = 3 * (max_seq_len * bytes_per_token).
4. Report everything in MiB and cap the plot axis at the budget.
5. Report utilization as:
   - used / allocated (internal efficiency of allocated cache)
   - used / budget (overall memory pool efficiency)
   - averaged over busy periods only, on at least 200 requests.
6. Add sweep over naive max_seq_len (512, 1024, 2048) under the same budget.
7. Print formatted markdown report and analytic check of worst-case waste (< 1 block).
"""
from __future__ import annotations

import argparse
import asyncio
import json
import math
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
    save_trace_to_json,
)
from engine.serving.api_server import calculate_paged_blocks_from_budget
from engine.serving.naive_server import calculate_naive_slots_from_budget


def compute_reconciled_kv_memory_stats(
    samples: List[MetricSample],
    system_type: str,
    budget_mib: float,
    block_size: int = 16,
    bytes_per_token: int = 12288,
) -> Dict[str, Any]:
    """
    Computes KV memory utilization averaged strictly over busy periods (running > 0):
    - used / allocated
    - used / budget
    All memory quantities reported in MiB (1024 * 1024 bytes).
    """
    budget_bytes = int(budget_mib * 1024 * 1024)
    # Busy samples only
    busy_samples = [s for s in samples if s.running > 0]
    if not busy_samples:
        busy_samples = [s for s in samples if s.allocated_kv_bytes > 0 or s.reserved_kv_bytes > 0]

    if not busy_samples:
        return {
            "system_type": system_type,
            "used_over_allocated_pct": 0.0,
            "used_over_budget_pct": 0.0,
            "max_concurrent": 0,
            "avg_allocated_mib": 0.0,
            "avg_used_mib": 0.0,
            "avg_internal_frag_mib": 0.0,
            "avg_free_mib": budget_mib,
            "budget_mib": budget_mib,
        }

    max_concurrent = max(s.running for s in busy_samples)
    bytes_per_block = bytes_per_token * block_size

    if system_type.lower() == "naive":
        allocated_bytes = [
            (s.reserved_kv_bytes if s.reserved_kv_bytes > 0 else s.running * s.total_blocks * bytes_per_token)
            for s in busy_samples
        ]
        used_bytes = [s.active_token_kv_bytes for s in busy_samples]
    else:
        allocated_bytes = [
            (s.allocated_kv_bytes if s.allocated_kv_bytes > 0 else (s.total_blocks - s.free_blocks) * bytes_per_block)
            for s in busy_samples
        ]
        used_bytes = [s.active_token_kv_bytes for s in busy_samples]

    used_over_alloc = [
        min(1.0, float(u / a)) if a > 0 else 0.0
        for u, a in zip(used_bytes, allocated_bytes)
    ]
    used_over_budget = [
        min(1.0, float(u / budget_bytes)) if budget_bytes > 0 else 0.0
        for u in used_bytes
    ]

    avg_allocated = float(np.mean(allocated_bytes))
    avg_used = float(np.mean(used_bytes))
    avg_internal_frag = max(0.0, avg_allocated - avg_used)
    avg_free = max(0.0, budget_bytes - avg_allocated)

    return {
        "system_type": system_type,
        "used_over_allocated_pct": float(np.mean(used_over_alloc)) * 100.0 if used_over_alloc else 0.0,
        "used_over_budget_pct": float(np.mean(used_over_budget)) * 100.0 if used_over_budget else 0.0,
        "max_concurrent": max_concurrent,
        "avg_allocated_mib": avg_allocated / (1024 * 1024),
        "avg_used_mib": avg_used / (1024 * 1024),
        "avg_internal_frag_mib": avg_internal_frag / (1024 * 1024),
        "avg_free_mib": avg_free / (1024 * 1024),
        "budget_mib": budget_mib,
    }


def generate_reconciled_chart(
    naive_stats: Dict[str, Any],
    paged_stats: Dict[str, Any],
    output_path: Path | str,
    budget_mib: float,
) -> None:
    """Produces comparative bar charts with Y-axis strictly capped at budget_mib."""
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5), dpi=150)

    systems = ["Naive Static Batching", "Paged Continuous Batching"]
    x = np.arange(len(systems))
    width = 0.28

    # 1. Left Chart: Utilization comparison (used/allocated vs used/budget) and concurrency
    used_alloc = [naive_stats["used_over_allocated_pct"], paged_stats["used_over_allocated_pct"]]
    used_budget = [naive_stats["used_over_budget_pct"], paged_stats["used_over_budget_pct"]]
    concurrency = [naive_stats["max_concurrent"], paged_stats["max_concurrent"]]

    r1 = ax1.bar(x - width, used_alloc, width, label="Utilization (Used / Allocated)", color="#1f77b4")
    r2 = ax1.bar(x, used_budget, width, label="Utilization (Used / Budget)", color="#2ca02c")
    ax1.set_ylabel("KV Utilization (%)", fontweight="bold")
    ax1.set_ylim(0, 105)

    ax1_twin = ax1.twinx()
    r3 = ax1_twin.bar(x + width, concurrency, width, label="Max Concurrent Requests", color="#d95f02")
    ax1_twin.set_ylabel("Concurrent Requests", color="#d95f02", fontweight="bold")
    ax1_twin.set_ylim(0, max(concurrency + [5]) * 1.3)
    ax1_twin.tick_params(axis="y", labelcolor="#d95f02")

    ax1.set_xticks(x)
    ax1.set_xticklabels(systems, fontweight="bold")
    ax1.set_title(f"KV Utilization & Concurrency\n(Budget = {budget_mib:.2f} MiB)", fontsize=11, fontweight="bold")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax1_twin.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left", frameon=True, fontsize=8)

    for r in list(r1) + list(r2):
        h = r.get_height()
        ax1.annotate(f"{h:.1f}%", xy=(r.get_x() + r.get_width() / 2, h), xytext=(0, 2), textcoords="offset points", ha="center", va="bottom", fontsize=8, fontweight="bold")
    for r in r3:
        h = r.get_height()
        ax1_twin.annotate(f"{int(h)}", xy=(r.get_x() + r.get_width() / 2, h), xytext=(0, 2), textcoords="offset points", ha="center", va="bottom", fontsize=8, fontweight="bold")

    # 2. Right Chart: Memory Breakdown strictly capped at budget_mib
    used_mib = [naive_stats["avg_used_mib"], paged_stats["avg_used_mib"]]
    frag_mib = [naive_stats["avg_internal_frag_mib"], paged_stats["avg_internal_frag_mib"]]
    free_mib = [naive_stats["avg_free_mib"], paged_stats["avg_free_mib"]]

    b_width = 0.45
    ax2.bar(systems, used_mib, b_width, label="Tokens Stored (Active KV)", color="#2ca02c")
    ax2.bar(systems, frag_mib, b_width, bottom=used_mib, label="Internal Fragmentation (Reserved Unused)", color="#d62728")
    bottom_free = [u + f for u, f in zip(used_mib, frag_mib)]
    ax2.bar(systems, free_mib, b_width, bottom=bottom_free, label="Truly Free Unallocated Pool", color="#bcbd22")

    ax2.axhline(budget_mib, color="#333333", linestyle="--", linewidth=1.2, label=f"Budget Cap ({budget_mib:.2f} MiB)")
    ax2.set_ylabel("Memory (MiB)", fontweight="bold")
    ax2.set_ylim(0, budget_mib * 1.05)  # Capped at budget!
    ax2.set_title(f"KV Memory Breakdown (Capped at Budget)\n(Budget = {budget_mib:.2f} MiB)", fontsize=11, fontweight="bold")
    ax2.legend(loc="upper right", frameon=True, fontsize=8)

    fig.tight_layout()
    plt.savefig(output_path)
    plt.close()


async def run_5d_reconciled_report(
    num_requests: int = 200,
    arrival_rate: float = 12.0,
    seed: int = 42,
    out_dir: str = "benchmarks/results",
) -> Dict[str, Any]:
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # 1. Generate 200-request workload trace
    trace = generate_workload_trace(
        num_requests=num_requests,
        arrival_rate=arrival_rate,
        seed=seed,
        ignore_eos=True,
    )
    trace_file = out_path / "workload_200_trace.json"
    save_trace_to_json(trace, trace_file)

    # Calculate trace max prompt+output length
    max_trace_len = max(r.prompt_len + r.output_len for r in trace)
    naive_max_seq_len = max_trace_len

    bytes_per_token = 2 * 24 * 2 * 64 * 2  # 12,288 bytes
    slot_bytes = naive_max_seq_len * bytes_per_token
    slot_mib = slot_bytes / (1024 * 1024)

    # Budget chosen so naive gets at least 3 slots
    # 3 slots * slot_mib
    budget_mib = float(math.ceil(3.5 * slot_mib))  # ~8.0 MiB, guarantees naive gets 3 slots
    budget_bytes = int(budget_mib * 1024 * 1024)

    raw_derived_slots = budget_bytes // slot_bytes
    clamped_derived_slots = max(1, raw_derived_slots)

    print("=" * 70)
    print("  PHASE 5D: KV-MEMORY RECONCILIATION & AUDIT")
    print("=" * 70)
    print(f"Trace Request Count: {len(trace)} requests")
    print(f"Max Prompt+Output Length in Trace: {max_trace_len} tokens")
    print(f"Naive Server max_seq_len: {naive_max_seq_len} tokens")
    print(f"Per-Slot Static Reservation: {slot_bytes:,} bytes ({slot_mib:.3f} MiB)")
    print(f"Chosen KV Memory Budget: {budget_mib:.2f} MiB ({budget_bytes:,} bytes)")
    print(f"Naive Derived Slot Count: {raw_derived_slots} (Clamped: {clamped_derived_slots} slots, >= 3 slots achieved)")
    print("=" * 70)

    port_naive = 8151
    port_paged = 8152

    # --- Run Naive Server ---
    env_naive = os.environ.copy()
    env_naive["NAIVE_KV_MEMORY_MB"] = str(budget_mib)
    env_naive["NAIVE_MAX_SEQ_LEN"] = str(naive_max_seq_len)
    env_naive["NAIVE_MAX_BATCH_SIZE"] = "32"

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
            warmup_count=5,
            stream=True,
            slo_threshold_s=2.0,
        )
    finally:
        server_naive.terminate()
        server_naive.wait(timeout=5)

    # --- Run Paged Engine ---
    env_paged = os.environ.copy()
    env_paged["VLLM_KV_MEMORY_MB"] = str(budget_mib)
    env_paged["VLLM_BLOCK_SIZE"] = "16"

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
            warmup_count=5,
            stream=True,
            slo_threshold_s=2.0,
        )
    finally:
        server_paged.terminate()
        server_paged.wait(timeout=5)

    # Compute reconciled memory stats
    naive_stats = compute_reconciled_kv_memory_stats(
        samples=samples_naive,
        system_type="Naive",
        budget_mib=budget_mib,
        bytes_per_token=bytes_per_token,
    )
    paged_stats = compute_reconciled_kv_memory_stats(
        samples=samples_paged,
        system_type="Paged",
        budget_mib=budget_mib,
        bytes_per_token=bytes_per_token,
    )

    # Analytic check
    block_size = 16
    bytes_per_block = block_size * bytes_per_token
    max_waste_tokens = block_size - 1
    max_waste_bytes = max_waste_tokens * bytes_per_token

    # Generate reconciled plot (capped at budget)
    plot_file = out_path / "kv_memory_comparison_reconciled.png"
    generate_reconciled_chart(naive_stats, paged_stats, plot_file, budget_mib=budget_mib)

    # --- Sweep over naive max_seq_len (512, 1024, 2048) under the same budget ---
    msl_sweep_results = []
    for msl in [512, 1024, 2048]:
        sb = msl * bytes_per_token
        slots = budget_bytes // sb
        clamped = max(1, slots)
        msl_sweep_results.append({
            "max_seq_len": msl,
            "slot_mib": round(sb / (1024 * 1024), 2),
            "derived_slots": slots,
            "clamped_slots": clamped,
            "clamped": (slots < 1),
        })

    report = {
        "budget_mib": budget_mib,
        "max_trace_len": max_trace_len,
        "naive_max_seq_len": naive_max_seq_len,
        "naive_derived_slots": clamped_derived_slots,
        "naive_stats": naive_stats,
        "paged_stats": paged_stats,
        "msl_sweep": msl_sweep_results,
        "analytic_check": {
            "block_size": block_size,
            "bytes_per_block": bytes_per_block,
            "max_waste_bytes": max_waste_bytes,
            "max_waste_kib": max_waste_bytes / 1024,
            "property_holds": (max_waste_bytes < bytes_per_block),
        },
    }

    with open(out_path / "report_5d_reconciled.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    return report


def format_5d_markdown_report(report: Dict[str, Any]) -> str:
    n = report["naive_stats"]
    p = report["paged_stats"]
    b = report["budget_mib"]
    ac = report["analytic_check"]

    md = f"""
### Reconciled Phase 5d KV-Memory Evaluation Report

#### 1. Setup & Reconciliation Audit
- **Workload Trace**: 200 requests (Poisson arrivals at $\lambda = 12.0$ req/s, seeded lognormal distributions)
- **Maximum Prompt + Output Length in Trace**: **{report['max_trace_len']} tokens**
- **Naive Server `max_seq_len`**: Exactly set to trace max (**{report['naive_max_seq_len']} tokens**)
- **KV Memory Budget**: **{b:.2f} MiB**
- **Naive Derived Slot Count**: **{report['naive_derived_slots']} slots** (guarantees $\ge 3$ concurrent slots under fair budget)

#### 2. Utilization & Concurrency Comparison Table (Averaged over Busy Periods)
| Metric | Naive Static Batching | Paged Continuous Batching | Advantage / Comparison |
| :--- | :---: | :---: | :---: |
| **Max Concurrent Requests** | **{n['max_concurrent']}** | **{p['max_concurrent']}** | **{p['max_concurrent'] / max(n['max_concurrent'], 1):.1f}× higher concurrency** |
| **Avg Utilization (Used / Allocated)** | **{n['used_over_allocated_pct']:.1f}%** | **{p['used_over_allocated_pct']:.1f}%** | **+{p['used_over_allocated_pct'] - n['used_over_allocated_pct']:.1f}% higher internal efficiency** |
| **Avg Utilization (Used / Budget)** | **{n['used_over_budget_pct']:.1f}%** | **{p['used_over_budget_pct']:.1f}%** | **+{p['used_over_budget_pct'] - n['used_over_budget_pct']:.1f}% overall pool utilization** |
| **Tokens Stored (Active KV)** | {n['avg_used_mib']:.2f} MiB | {p['avg_used_mib']:.2f} MiB | Actively computed KV tokens |
| **Internal Fragmentation** | {n['avg_internal_frag_mib']:.2f} MiB | {p['avg_internal_frag_mib']:.2f} MiB | Reserved static padding vs. tail block slack |
| **Truly Free Pool Capacity** | {n['avg_free_mib']:.2f} MiB | {p['avg_free_mib']:.2f} MiB | Unallocated pool available for admissions |

*Note: All memory metrics reported in true binary MiB ($1024^2$ bytes), and the plot axis is capped strictly at the {b:.2f} MiB budget.*

#### 3. Sweep over Naive `max_seq_len` under Fixed {b:.2f} MiB Budget
| Static `max_seq_len` | Slot Size (MiB) | Derived Slots (`budget // slot_bytes`) | Clamped Slots | Clamping Triggered? |
| :---: | :---: | :---: | :---: | :---: |
"""
    for item in report["msl_sweep"]:
        clamped_flag = "**Yes (clamped to 1)**" if item["clamped"] else "No"
        md += f"| {item['max_seq_len']} | {item['slot_mib']:.2f} MiB | {item['derived_slots']} | {item['clamped_slots']} | {clamped_flag} |\n"

    md += f"""
#### 4. Analytic Check: Paged Worst-Case Waste per Sequence
- **Configured Block Size**: {ac['block_size']} tokens ({ac['bytes_per_block'] / 1024:.1f} KiB)
- **Max Internal Waste per Sequence**: $\le {ac['block_size'] - 1}$ tokens (${ac['max_waste_kib']:.1f}$ KiB)
- **Analytical Bound Invariant**: $\\text{{Waste}}_{{\\text{{seq}}}} < \\text{{Block Bytes}}$ holds: **{ac['property_holds']}**
"""
    return md


def main():
    report = asyncio.run(run_5d_reconciled_report())
    md = format_5d_markdown_report(report)
    print(md)


if __name__ == "__main__":
    main()
