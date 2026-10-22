"""
Benchmark serving client using asyncio + aiohttp.

Replays workload traces against an OpenAI-compatible serving URL (e.g. FastAPI AsyncEngine or NaiveBatchServer).
Measures:
- Per-request: TTFT (Time To First Token), per-token timestamps (for TPOT / inter-token latency), E2E latency, output token count.
- Overall: Total throughput (output tokens/s), request throughput (req/s), P50/P90/P99 for TTFT and E2E, fraction of requests completed within an SLO.
- Background metric poller: polls /metrics every 100 ms and logs KV blocks in use, running, and waiting counts.
- Features: Warmup phase (excluded from stats), fixed-duration run option, lambda sweep mode to detect saturation.
- Generates raw JSON and three plots:
  1. Throughput vs. offered load
  2. TTFT P99 vs. offered load (highlighting saturation)
  3. Timeline of running/waiting counts and memory utilization
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import math
import os
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import aiohttp
import matplotlib.pyplot as plt
import numpy as np

from benchmarks.workload_trace import (
    TraceRequest,
    generate_workload_trace,
    load_trace_from_json,
    save_trace_to_json,
)

logger = logging.getLogger(__name__)


@dataclass
class RequestBenchmarkResult:
    request_id: int
    prompt_len: int
    target_output_len: int
    actual_output_len: int
    start_time: float
    first_token_time: Optional[float] = None
    finish_time: Optional[float] = None
    token_timestamps: List[float] = field(default_factory=list)
    token_ids: List[int] = field(default_factory=list)
    token_margins: List[Optional[float]] = field(default_factory=list)
    success: bool = False
    error: Optional[str] = None
    is_warmup: bool = False

    @property
    def ttft_s(self) -> Optional[float]:
        if self.first_token_time is not None:
            return self.first_token_time - self.start_time
        return None

    @property
    def e2e_latency_s(self) -> Optional[float]:
        if self.finish_time is not None:
            return self.finish_time - self.start_time
        return None

    @property
    def inter_token_latencies_s(self) -> List[float]:
        if len(self.token_timestamps) < 2:
            return []
        return [
            self.token_timestamps[i] - self.token_timestamps[i - 1]
            for i in range(1, len(self.token_timestamps))
        ]

    @property
    def mean_tpot_s(self) -> Optional[float]:
        itls = self.inter_token_latencies_s
        if itls:
            return float(np.mean(itls))
        return None


@dataclass
class MetricSample:
    timestamp: float
    running: int
    waiting: int
    free_blocks: int
    total_blocks: int
    tokens_generated: int
    preemptions: int
    tokens_stored: int = 0
    allocated_kv_bytes: int = 0
    reserved_kv_bytes: int = 0
    active_token_kv_bytes: int = 0
    total_kv_bytes: int = 0
    admission_blocked: int = 0
    cached_free_blocks: int = 0
    prefix_hit_rate: float = 0.0
    prefix_cached_tokens: int = 0
    prefix_prompt_tokens: int = 0
    blocks_saved: int = 0
    cache_evictions: int = 0

    @property
    def blocks_in_use(self) -> int:
        return max(0, self.total_blocks - self.free_blocks)

    @property
    def memory_utilization(self) -> float:
        # Utilization = active token bytes / reserved (or allocated) bytes
        denominator = self.allocated_kv_bytes or self.reserved_kv_bytes
        if denominator > 0:
            return min(1.0, float(self.active_token_kv_bytes / denominator))
        return 0.0


@dataclass
class RunSummary:
    offered_load_rps: float
    actual_request_throughput_rps: float
    token_throughput_tps: float
    total_requests: int
    completed_requests: int
    failed_requests: int
    warmup_requests: int
    duration_s: float
    slo_threshold_s: float
    slo_attainment: float
    ttft_p50_s: Optional[float]
    ttft_p90_s: Optional[float]
    ttft_p99_s: Optional[float]
    ttft_mean_s: Optional[float]
    e2e_p50_s: Optional[float]
    e2e_p90_s: Optional[float]
    e2e_p99_s: Optional[float]
    e2e_mean_s: Optional[float]
    tpot_mean_s: Optional[float]
    peak_running: int
    peak_waiting: int
    peak_blocks_in_use: int
    avg_kv_utilization: float = 0.0
    avg_allocated_kv_mb: float = 0.0
    avg_active_token_kv_mb: float = 0.0


async def poll_metrics_loop(
    session: aiohttp.ClientSession,
    base_url: str,
    samples: List[MetricSample],
    stop_event: asyncio.Event,
    interval_s: float = 0.1,
) -> None:
    """Polls /metrics endpoint every interval_s (100ms) until stop_event is set."""
    metrics_url = f"{base_url.rstrip('/')}/metrics"
    t0 = time.perf_counter()

    while not stop_event.is_set():
        try:
            async with session.get(metrics_url, timeout=aiohttp.ClientTimeout(total=2.0)) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    now = time.perf_counter() - t0
                    samples.append(
                        MetricSample(
                            timestamp=now,
                            running=int(data.get("running", 0)),
                            waiting=int(data.get("waiting", 0)),
                            free_blocks=int(data.get("free_blocks", 0)),
                            total_blocks=int(data.get("total_blocks", 0)),
                            tokens_generated=int(data.get("tokens_generated", 0)),
                            preemptions=int(data.get("preemptions", 0)),
                            tokens_stored=int(data.get("tokens_stored", 0)),
                            allocated_kv_bytes=int(data.get("allocated_kv_bytes", 0)),
                            reserved_kv_bytes=int(data.get("reserved_kv_bytes", 0)),
                            active_token_kv_bytes=int(data.get("active_token_kv_bytes", 0)),
                            total_kv_bytes=int(data.get("total_kv_bytes", data.get("total_reserved_kv_bytes", 0))),
                            admission_blocked=int(data.get("admission_blocked", 0)),
                            cached_free_blocks=int(data.get("cached_free_blocks", 0)),
                            prefix_hit_rate=float(data.get("prefix_hit_rate", 0.0)),
                            prefix_cached_tokens=int(data.get("prefix_cached_tokens", 0)),
                            prefix_prompt_tokens=int(data.get("prefix_prompt_tokens", 0)),
                            blocks_saved=int(data.get("blocks_saved", 0)),
                            cache_evictions=int(data.get("cache_evictions", 0)),
                        )
                    )
        except Exception:
            pass

        try:
            await asyncio.wait_for(stop_event.wait(), timeout=interval_s)
        except asyncio.TimeoutError:
            pass


async def send_single_request(
    session: aiohttp.ClientSession,
    base_url: str,
    req: TraceRequest,
    is_warmup: bool,
    stream: bool = True,
    include_token_ids: bool = False,
) -> RequestBenchmarkResult:
    """Dispatches a single completion request (streaming SSE or non-streaming) and instruments timings."""
    url = f"{base_url.rstrip('/')}/v1/completions"
    payload = {
        "model": "default",
        "prompt": req.prompt,
        "max_tokens": req.output_len,
        "temperature": 0.0,
        "stream": stream,
        "ignore_eos": req.ignore_eos,
        "include_token_ids": include_token_ids,
    }

    result = RequestBenchmarkResult(
        request_id=req.request_id,
        prompt_len=req.prompt_len,
        target_output_len=req.output_len,
        actual_output_len=0,
        start_time=time.perf_counter(),
        is_warmup=is_warmup,
    )

    try:
        timeout = aiohttp.ClientTimeout(total=180.0)
        if stream:
            async with session.post(url, json=payload, timeout=timeout) as resp:
                if resp.status != 200:
                    text = await resp.text()
                    result.error = f"HTTP {resp.status}: {text}"
                    return result

                async for raw_line in resp.content:
                    line = raw_line.decode("utf-8").strip()
                    if not line or not line.startswith("data: "):
                        continue
                    if line == "data: [DONE]":
                        break

                    now = time.perf_counter()
                    if result.first_token_time is None:
                        result.first_token_time = now

                    data_str = line[6:]
                    try:
                        chunk = json.loads(data_str)
                        choices = chunk.get("choices", [])
                        if choices and (choices[0].get("text") or "token_id" in chunk):
                            result.actual_output_len += 1
                            result.token_timestamps.append(now)
                            if "token_id" in chunk:
                                result.token_ids.append(int(chunk["token_id"]))
                                margin = chunk.get("top1_top2_margin")
                                result.token_margins.append(
                                    float(margin) if margin is not None else None
                                )
                    except json.JSONDecodeError:
                        pass
        else:
            async with session.post(url, json=payload, timeout=timeout) as resp:
                if resp.status != 200:
                    text = await resp.text()
                    result.error = f"HTTP {resp.status}: {text}"
                    return result

                data = await resp.json()
                now = time.perf_counter()
                result.first_token_time = now
                usage = data.get("usage", {})
                result.actual_output_len = usage.get("completion_tokens", req.output_len)
                result.token_timestamps.append(now)
                if include_token_ids:
                    result.token_ids = [int(token) for token in data.get("token_ids", [])]

        result.finish_time = time.perf_counter()
        result.success = True
    except Exception as e:
        result.error = str(e)
        result.finish_time = time.perf_counter()

    return result


def compute_run_summary(
    results: List[RequestBenchmarkResult],
    metric_samples: List[MetricSample],
    offered_load_rps: float,
    slo_threshold_s: float = 2.0,
    elapsed_duration_s: Optional[float] = None,
) -> RunSummary:
    """Computes aggregate throughput, latency percentiles, and SLO attainment."""
    eval_results = [r for r in results if not r.is_warmup]
    successful = [r for r in eval_results if r.success and r.e2e_latency_s is not None]

    total_requests = len(eval_results)
    completed_requests = len(successful)
    failed_requests = total_requests - completed_requests
    warmup_requests = sum(1 for r in results if r.is_warmup)

    if elapsed_duration_s is not None and elapsed_duration_s > 0:
        duration_s = elapsed_duration_s
    elif successful:
        t_min = min(r.start_time for r in successful)
        t_max = max(r.finish_time for r in successful if r.finish_time is not None)
        duration_s = max(t_max - t_min, 0.001)
    else:
        duration_s = 1.0

    total_tokens = sum(r.actual_output_len for r in successful)
    token_throughput_tps = float(total_tokens / duration_s) if duration_s > 0 else 0.0
    request_throughput_rps = float(completed_requests / duration_s) if duration_s > 0 else 0.0

    ttfts = [r.ttft_s for r in successful if r.ttft_s is not None]
    e2es = [r.e2e_latency_s for r in successful if r.e2e_latency_s is not None]
    tpots = [r.mean_tpot_s for r in successful if r.mean_tpot_s is not None]

    def calc_percentiles(vals: List[float]) -> Tuple[Optional[float], Optional[float], Optional[float], Optional[float]]:
        if not vals:
            return None, None, None, None
        return (
            float(np.percentile(vals, 50)),
            float(np.percentile(vals, 90)),
            float(np.percentile(vals, 99)),
            float(np.mean(vals)),
        )

    ttft_p50, ttft_p90, ttft_p99, ttft_mean = calc_percentiles(ttfts)
    e2e_p50, e2e_p90, e2e_p99, e2e_mean = calc_percentiles(e2es)
    tpot_mean = float(np.mean(tpots)) if tpots else None

    # SLO fraction
    within_slo = sum(1 for r in successful if r.e2e_latency_s is not None and r.e2e_latency_s <= slo_threshold_s)
    slo_attainment = float(within_slo / total_requests) if total_requests > 0 else 0.0

    peak_running = max((s.running for s in metric_samples), default=0)
    peak_waiting = max((s.waiting for s in metric_samples), default=0)
    peak_blocks = max((s.blocks_in_use for s in metric_samples), default=0)

    # Compute average KV utilization across samples with active load
    active_samples = [s for s in metric_samples if s.running > 0 or s.allocated_kv_bytes > 0 or s.reserved_kv_bytes > 0]
    if active_samples:
        avg_kv_util = float(np.mean([s.memory_utilization for s in active_samples]))
        avg_alloc_mb = float(np.mean([(s.allocated_kv_bytes or s.reserved_kv_bytes) / (1024 * 1024) for s in active_samples]))
        avg_active_mb = float(np.mean([s.active_token_kv_bytes / (1024 * 1024) for s in active_samples]))
    else:
        avg_kv_util = 0.0
        avg_alloc_mb = 0.0
        avg_active_mb = 0.0

    return RunSummary(
        offered_load_rps=offered_load_rps,
        actual_request_throughput_rps=round(request_throughput_rps, 3),
        token_throughput_tps=round(token_throughput_tps, 2),
        total_requests=total_requests,
        completed_requests=completed_requests,
        failed_requests=failed_requests,
        warmup_requests=warmup_requests,
        duration_s=round(duration_s, 3),
        slo_threshold_s=slo_threshold_s,
        slo_attainment=round(slo_attainment, 4),
        ttft_p50_s=round(ttft_p50, 4) if ttft_p50 else None,
        ttft_p90_s=round(ttft_p90, 4) if ttft_p90 else None,
        ttft_p99_s=round(ttft_p99, 4) if ttft_p99 else None,
        ttft_mean_s=round(ttft_mean, 4) if ttft_mean else None,
        e2e_p50_s=round(e2e_p50, 4) if e2e_p50 else None,
        e2e_p90_s=round(e2e_p90, 4) if e2e_p90 else None,
        e2e_p99_s=round(e2e_p99, 4) if e2e_p99 else None,
        e2e_mean_s=round(e2e_mean, 4) if e2e_mean else None,
        tpot_mean_s=round(tpot_mean, 4) if tpot_mean else None,
        peak_running=peak_running,
        peak_waiting=peak_waiting,
        peak_blocks_in_use=peak_blocks,
        avg_kv_utilization=round(avg_kv_util, 4),
        avg_allocated_kv_mb=round(avg_alloc_mb, 3),
        avg_active_token_kv_mb=round(avg_active_mb, 3),
    )


async def run_trace_replay(
    base_url: str,
    trace: List[TraceRequest],
    offered_rate: float,
    warmup_count: int = 5,
    max_duration_s: Optional[float] = None,
    stream: bool = True,
    slo_threshold_s: float = 2.0,
    include_token_ids: bool = False,
) -> Tuple[RunSummary, List[RequestBenchmarkResult], List[MetricSample]]:
    """Replays a trace at the specified offered rate, polling metrics concurrently."""
    metric_samples: List[MetricSample] = []
    stop_metrics = asyncio.Event()

    conn = aiohttp.TCPConnector(limit=500, force_close=False)
    async with aiohttp.ClientSession(connector=conn) as session:
        # Start background metrics poller (every 100ms)
        metrics_task = asyncio.create_task(
            poll_metrics_loop(
                session=session,
                base_url=base_url,
                samples=metric_samples,
                stop_event=stop_metrics,
                interval_s=0.1,
            )
        )

        results: List[RequestBenchmarkResult] = []
        active_tasks: List[asyncio.Task] = []

        # 1. Warmup phase (excluded from benchmark stats)
        if warmup_count > 0:
            logger.info(f"Executing {warmup_count} warmup requests...")
            for i in range(min(warmup_count, len(trace))):
                warm_res = await send_single_request(
                    session=session,
                    base_url=base_url,
                    req=trace[i],
                    is_warmup=True,
                    stream=stream,
                    include_token_ids=include_token_ids,
                )
                results.append(warm_res)

        # 2. Main replay benchmark phase
        t_benchmark_start = time.perf_counter()
        remaining_trace = trace[warmup_count:] if warmup_count < len(trace) else trace

        for req in remaining_trace:
            # Check fixed-duration cut-off
            now = time.perf_counter()
            if max_duration_s is not None and (now - t_benchmark_start) >= max_duration_s:
                break

            # Schedule request according to trace arrival offset
            target_arrival = req.arrival_time_s
            elapsed = now - t_benchmark_start
            wait_s = target_arrival - elapsed
            if wait_s > 0:
                await asyncio.sleep(wait_s)

            # Spawn request task concurrently
            task = asyncio.create_task(
                send_single_request(
                    session=session,
                    base_url=base_url,
                    req=req,
                    is_warmup=False,
                    stream=stream,
                    include_token_ids=include_token_ids,
                )
            )
            active_tasks.append(task)

        # Await all dispatched requests
        if active_tasks:
            completed_results = await asyncio.gather(*active_tasks)
            results.extend(completed_results)

        total_benchmark_time = time.perf_counter() - t_benchmark_start

        # Stop background metrics poller
        stop_metrics.set()
        await metrics_task

    summary = compute_run_summary(
        results=results,
        metric_samples=metric_samples,
        offered_load_rps=offered_rate,
        slo_threshold_s=slo_threshold_s,
        elapsed_duration_s=total_benchmark_time,
    )

    return summary, results, metric_samples


def generate_benchmark_plots(
    summaries: List[RunSummary],
    metric_samples: List[MetricSample],
    output_dir: Path | str,
    system_label: str = "Paged AsyncEngine",
) -> List[str]:
    """Generates the three required benchmark plots and saves them to disk."""
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    generated_plots = []

    # Modern visual styling
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.size"] = 10

    offered_loads = [s.offered_load_rps for s in summaries]
    token_throughputs = [s.token_throughput_tps for s in summaries]
    req_throughputs = [s.actual_request_throughput_rps for s in summaries]
    ttft_p99s = [s.ttft_p99_s * 1000.0 if s.ttft_p99_s else 0.0 for s in summaries]
    ttft_p50s = [s.ttft_p50_s * 1000.0 if s.ttft_p50_s else 0.0 for s in summaries]

    # --- Plot 1: Throughput vs. Offered Load ---
    p1_path = out_dir / "throughput_vs_offered_load.png"
    fig, ax1 = plt.subplots(figsize=(8, 5), dpi=150)
    color1 = "#1f77b4"
    color2 = "#2ca02c"

    ax1.set_xlabel("Offered Load (req/s)", fontweight="bold")
    ax1.set_ylabel("Token Throughput (tokens/s)", color=color1, fontweight="bold")
    line1 = ax1.plot(offered_loads, token_throughputs, marker="o", color=color1, linewidth=2, label="Tokens/s")
    ax1.tick_params(axis="y", labelcolor=color1)

    ax2 = ax1.twinx()
    ax2.set_ylabel("Request Throughput (req/s)", color=color2, fontweight="bold")
    line2 = ax2.plot(offered_loads, req_throughputs, marker="s", linestyle="--", color=color2, linewidth=2, label="Req/s")
    ax2.tick_params(axis="y", labelcolor=color2)

    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc="upper left", frameon=True)

    plt.title(f"Throughput vs. Offered Load ({system_label})", fontsize=12, fontweight="bold", pad=12)
    fig.tight_layout()
    plt.savefig(p1_path)
    plt.close()
    generated_plots.append(str(p1_path))

    # --- Plot 2: TTFT P99 vs. Offered Load (Saturation Point) ---
    p2_path = out_dir / "ttft_p99_vs_offered_load.png"
    fig, ax = plt.subplots(figsize=(8, 5), dpi=150)
    ax.plot(offered_loads, ttft_p99s, marker="^", color="#d62728", linewidth=2.5, label="TTFT P99 (ms)")
    ax.plot(offered_loads, ttft_p50s, marker="o", color="#ff7f0e", linestyle=":", linewidth=1.5, label="TTFT P50 (ms)")

    # Identify potential saturation knee
    if len(ttft_p99s) > 2:
        diffs = np.diff(ttft_p99s)
        knee_idx = int(np.argmax(diffs))
        if diffs[knee_idx] > 20.0:
            sat_load = offered_loads[knee_idx + 1]
            sat_ttft = ttft_p99s[knee_idx + 1]
            ax.annotate(
                f"Saturation Point: ~{sat_load} req/s",
                xy=(sat_load, sat_ttft),
                xytext=(sat_load * 0.7, sat_ttft * 1.15),
                arrowprops=dict(facecolor="#d62728", shrink=0.08, width=1.5, headwidth=7),
                fontweight="bold",
                color="#b2182b",
                bbox=dict(boxstyle="round,pad=0.3", fc="#ffebee", ec="#d62728", lw=1),
            )

    ax.set_xlabel("Offered Load (req/s)", fontweight="bold")
    ax.set_ylabel("TTFT Latency (ms)", fontweight="bold")
    ax.set_title(f"TTFT Latency vs. Offered Load - Saturation Point ({system_label})", fontsize=12, fontweight="bold", pad=12)
    ax.legend(loc="upper left", frameon=True)
    fig.tight_layout()
    plt.savefig(p2_path)
    plt.close()
    generated_plots.append(str(p2_path))

    # --- Plot 3: Timeline of Running/Waiting Counts & KV Blocks ---
    p3_path = out_dir / "timeline_running_waiting.png"
    fig, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=(10, 6), sharex=True, dpi=150)

    if metric_samples:
        timestamps = [s.timestamp for s in metric_samples]
        running_counts = [s.running for s in metric_samples]
        waiting_counts = [s.waiting for s in metric_samples]
        blocks_used = [s.blocks_in_use for s in metric_samples]
        total_blocks = [s.total_blocks for s in metric_samples]

        # Top: Queue Concurrency
        ax_top.plot(timestamps, running_counts, color="#1f77b4", label="Running Requests", linewidth=1.8)
        ax_top.plot(timestamps, waiting_counts, color="#e377c2", label="Waiting Requests", linewidth=1.8, linestyle="--")
        ax_top.set_ylabel("Request Count", fontweight="bold")
        ax_top.set_title(f"Serving Concurrency & Memory Timeline ({system_label})", fontsize=12, fontweight="bold", pad=10)
        ax_top.legend(loc="upper right", frameon=True)

        # Bottom: KV Blocks in use
        ax_bot.plot(timestamps, blocks_used, color="#2ca02c", label="KV Blocks In Use", linewidth=2.0)
        if total_blocks and total_blocks[0] > 0:
            ax_bot.axhline(total_blocks[0], color="#7f7f7f", linestyle=":", label=f"Total Capacity ({total_blocks[0]} blocks)")
        ax_bot.set_xlabel("Time Elapsed (s)", fontweight="bold")
        ax_bot.set_ylabel("KV Blocks", fontweight="bold")
        ax_bot.legend(loc="upper right", frameon=True)
    else:
        ax_top.text(0.5, 0.5, "No metric samples logged", ha="center", va="center")
        ax_bot.text(0.5, 0.5, "No metric samples logged", ha="center", va="center")

    fig.tight_layout()
    plt.savefig(p3_path)
    plt.close()
    generated_plots.append(str(p3_path))

    return generated_plots


async def main_async(args):
    # Load or generate trace
    if args.trace_file:
        trace = load_trace_from_json(args.trace_file)
        logger.info(f"Loaded {len(trace)} requests from {args.trace_file}")
    else:
        logger.info(f"Generating synthetic trace with {args.num_requests} requests at lambda={args.arrival_rate}...")
        trace = generate_workload_trace(
            num_requests=args.num_requests,
            arrival_rate=args.arrival_rate,
            seed=args.seed,
            ignore_eos=True,
        )

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_summaries: List[RunSummary] = []
    all_raw_results: Dict[str, Any] = {}
    last_metric_samples: List[MetricSample] = []

    # Sweep mode vs single run
    rates_to_run = args.sweep_rates if args.sweep_rates else [args.arrival_rate]

    for rate in rates_to_run:
        print(f"\n=======================================================")
        print(f"  Benchmarking Offered Rate: {rate} req/s")
        print(f"=======================================================")

        # Regenerate arrival timestamps for this rate if sweeping
        current_trace = generate_workload_trace(
            num_requests=len(trace),
            arrival_rate=rate,
            seed=args.seed,
            ignore_eos=True,
        )

        summary, results, metrics = await run_trace_replay(
            base_url=args.server_url,
            trace=current_trace,
            offered_rate=rate,
            warmup_count=args.warmup,
            max_duration_s=args.duration,
            stream=args.stream,
            slo_threshold_s=args.slo,
        )

        all_summaries.append(summary)
        last_metric_samples = metrics
        all_raw_results[f"rate_{rate}"] = {
            "summary": asdict(summary),
            "requests": [asdict(r) for r in results],
            "metrics": [asdict(m) for m in metrics],
        }

        print(f"Completed: {summary.completed_requests}/{summary.total_requests} reqs")
        print(f"Throughput: {summary.token_throughput_tps} tokens/s, {summary.actual_request_throughput_rps} req/s")
        mean_ms_per_tok = (1000.0 / summary.token_throughput_tps) if summary.token_throughput_tps > 0 else 0.0
        print(f"Mean ms per output token: {mean_ms_per_tok:.2f} ms/token")
        print(f"TTFT P50: {summary.ttft_p50_s}s, P90: {summary.ttft_p90_s}s, P99: {summary.ttft_p99_s}s")
        print(f"E2E  P50: {summary.e2e_p50_s}s, P90: {summary.e2e_p90_s}s, P99: {summary.e2e_p99_s}s")
        print(f"SLO Attainment (<={args.slo}s): {summary.slo_attainment * 100:.1f}%")

    # Save raw JSON results
    json_path = out_dir / "serving_benchmark_results.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(all_raw_results, f, indent=2)
    print(f"\nSaved raw JSON benchmark data to {json_path}")

    # Generate the 3 requested plots
    plot_paths = generate_benchmark_plots(
        summaries=all_summaries,
        metric_samples=last_metric_samples,
        output_dir=out_dir,
        system_label=args.label,
    )
    for p in plot_paths:
        print(f"Saved plot: {p}")


def main():
    parser = argparse.ArgumentParser(description="AsyncIO serving benchmark client for LLM engines")
    parser.add_argument("--server-url", type=str, default="http://127.0.0.1:8000")
    parser.add_argument("--trace-file", type=str, default=None, help="Path to pre-generated trace JSON")
    parser.add_argument("--num-requests", type=int, default=50)
    parser.add_argument("--arrival-rate", type=float, default=4.0, help="Poisson arrival lambda (req/s)")
    parser.add_argument("--sweep-rates", type=float, nargs="+", default=None, help="List of lambdas for sweep mode (e.g. 0.5 1 2 4 8)")
    parser.add_argument("--warmup", type=int, default=5, help="Number of warmup requests excluded from stats")
    parser.add_argument("--duration", type=float, default=None, help="Fixed max run duration in seconds")
    parser.add_argument("--stream", action="store_true", default=True, help="Use streaming SSE /v1/completions")
    parser.add_argument("--slo", type=float, default=2.0, help="SLO latency threshold in seconds")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=str, default="benchmarks/results")
    parser.add_argument("--label", type=str, default="Continuous Batching Engine")
    parser.add_argument(
        "--runner",
        type=str,
        default="real",
        choices=["real"],
        help="Model runner mode for benchmark runs (only 'real' HF Qwen is allowed)",
    )
    args = parser.parse_args()

    if args.runner != "real":
        parser.error(f"Invalid runner '{args.runner}'. Only '--runner real' is allowed for benchmark runs.")

    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
