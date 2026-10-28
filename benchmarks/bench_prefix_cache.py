"""Run the shared-prefix paged-serving benchmark matrix."""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import socket
import subprocess
import sys
import time
import urllib.request
from dataclasses import asdict
from pathlib import Path
from typing import Any

import aiohttp
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from benchmarks.bench_serving import RequestBenchmarkResult, run_trace_replay
from benchmarks.workload_trace import (
    SharedPrefixTrace,
    generate_shared_prefix_trace,
    save_shared_prefix_trace,
)

BLOCK_SIZE = 16
BYTES_PER_TOKEN = 12_288
PREFIX_MODES = ("off", "share", "full")
DEFAULT_RATES = (1.0, 2.0, 3.0, 4.0)
DEFAULT_SEEDS = (42, 43)
PORT = 8268


def gpu_clocks() -> dict[str, Any]:
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,clocks.current.graphics,clocks.current.memory,"
                "temperature.gpu,power.draw",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
        return {"sampled_at": time.time(), "value": result.stdout.strip()}
    except (OSError, subprocess.SubprocessError) as error:
        return {"sampled_at": time.time(), "error": str(error)}


async def poll_gpu_clocks(samples: list[dict[str, Any]], stop: asyncio.Event) -> None:
    while not stop.is_set():
        samples.append(await asyncio.to_thread(gpu_clocks))
        try:
            await asyncio.wait_for(stop.wait(), timeout=5.0)
        except asyncio.TimeoutError:
            pass


def wait_server_ready(
    port: int,
    process: subprocess.Popen,
    timeout_s: float = 180.0,
    expected_blocks: int | None = None,
    expected_prefix_mode: str | None = None,
    expected_block_size: int | None = None,
    expected_budget_tokens: int | None = None,
) -> None:
    url = f"http://127.0.0.1:{port}/health"
    deadline = time.monotonic() + timeout_s
    last_error: Exception | None = None
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"Paged server exited during startup with code {process.returncode}.")
        try:
            with urllib.request.urlopen(url, timeout=2.0) as response:
                health = json.loads(response.read().decode("utf-8"))
                if response.status == 200 and health.get("runner") == "hf":
                    # Verify configuration via /metrics
                    if any(x is not None for x in (expected_blocks, expected_prefix_mode, expected_block_size, expected_budget_tokens)):
                        metrics_url = f"http://127.0.0.1:{port}/metrics"
                        with urllib.request.urlopen(metrics_url, timeout=2.0) as m_resp:
                            m_data = json.loads(m_resp.read().decode("utf-8"))
                            actual_blocks = m_data.get("total_blocks")
                            actual_block_size = m_data.get("block_size")
                            actual_mode = m_data.get("prefix_mode")
                            actual_budget = (actual_blocks or 0) * (actual_block_size or 0)

                            if expected_blocks is not None and actual_blocks != expected_blocks:
                                raise RuntimeError(
                                    f"Startup block count mismatch: server reported {actual_blocks} blocks, "
                                    f"expected capacity {expected_blocks} blocks."
                                )
                            if expected_prefix_mode is not None and actual_mode != expected_prefix_mode:
                                raise RuntimeError(
                                    f"Startup prefix_mode mismatch: server reported {actual_mode!r}, "
                                    f"expected {expected_prefix_mode!r}."
                                )
                            if expected_block_size is not None and actual_block_size != expected_block_size:
                                raise RuntimeError(
                                    f"Startup block_size mismatch: server reported {actual_block_size}, "
                                    f"expected {expected_block_size}."
                                )
                            if expected_budget_tokens is not None and actual_budget != expected_budget_tokens:
                                raise RuntimeError(
                                    f"Startup budget tokens mismatch: server reported {actual_budget} tokens, "
                                    f"expected {expected_budget_tokens} tokens."
                                )
                    return
                raise RuntimeError(f"Unexpected paged server health response: {health!r}")
        except RuntimeError:
            raise
        except Exception as error:
            last_error = error
            time.sleep(0.5)
    raise RuntimeError(f"Paged server failed to become ready: {last_error}")


def start_paged_server(
    prefix_mode: str,
    kv_budget_tokens: int,
    port: int,
    log_path: Path | None = None,
) -> tuple["subprocess.Popen", Any, Path]:
    """Start the paged server subprocess.

    Redirects server stderr/stdout to log_path.
    Asserts server block count matches expected capacity.
    Returns (process, log_file_handle, log_path).
    """
    if log_path is None:
        log_dir = Path("benchmarks/results/prefix_cache/server_logs")
        log_dir.mkdir(parents=True, exist_ok=True)
        log_path = log_dir / f"server_port_{port}_{time.time_ns()}.log"
    else:
        log_path.parent.mkdir(parents=True, exist_ok=True)

    log_file = open(log_path, "w", encoding="utf-8")

    env = os.environ.copy()
    env["VLLM_KV_MEMORY_MB"] = str(kv_budget_tokens * BYTES_PER_TOKEN / (1024 * 1024))
    env["VLLM_BLOCK_SIZE"] = str(BLOCK_SIZE)
    env["VLLM_PREFIX_MODE"] = prefix_mode
    process = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "uvicorn",
            "engine.serving.api_server:app",
            "--port",
            str(port),
            "--log-level",
            "warning",
        ],
        env=env,
        stdout=log_file,
        stderr=subprocess.STDOUT,
        text=True,
    )
    try:
        expected_blocks = kv_budget_tokens // BLOCK_SIZE
        wait_server_ready(
            port,
            process,
            expected_blocks=expected_blocks,
            expected_prefix_mode=prefix_mode,
            expected_block_size=BLOCK_SIZE,
            expected_budget_tokens=kv_budget_tokens,
        )
    except Exception:
        process.terminate()
        try:
            process.wait(timeout=10)
        except Exception:
            process.kill()
        log_file.flush()
        log_file.close()
        raise
    return process, log_file, log_path


def stop_server(process: subprocess.Popen, log_file: Any, log_path: Path, port: int = PORT) -> tuple[int, str]:
    """Terminate server; return (exit_code, last_stderr_lines)."""
    process.terminate()
    try:
        process.wait(timeout=20)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=10)
    log_file.flush()
    log_file.close()

    # On Windows, wait briefly for the port to release and avoid connecting to lingering process
    deadline = time.monotonic() + 10.0
    while time.monotonic() < deadline:
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=0.2):
                time.sleep(0.2)
        except OSError:
            break

    log_tail = ""
    try:
        if log_path.exists():
            content = log_path.read_text(encoding="utf-8", errors="replace")
            log_tail = content[-8000:]
    except Exception:
        pass
    return process.returncode or 0, log_tail


async def read_metrics(port: int) -> dict[str, Any]:
    async with aiohttp.ClientSession() as session:
        async with session.get(
            f"http://127.0.0.1:{port}/metrics",
            timeout=aiohttp.ClientTimeout(total=10),
        ) as response:
            response.raise_for_status()
            return await response.json()


def summarize_run(
    mode: str,
    profile: str,
    prefix_metadata: dict[str, Any],
    offered_rate: float,
    seed: int,
    budget_tokens: int,
    summary: Any,
    requests: list[RequestBenchmarkResult],
    metrics: list[Any],
    final_metrics: dict[str, Any],
    clocks: list[dict[str, Any]],
) -> dict[str, Any]:
    completed = [
        request
        for request in requests
        if request.success and request.finish_time is not None
    ]
    if completed:
        elapsed = max(request.finish_time for request in completed) - min(
            request.start_time for request in completed
        )
        completed_tokens = sum(request.actual_output_len for request in completed)
        throughput = completed_tokens / max(elapsed, 1e-9)
        ttfts = [request.ttft_s for request in completed if request.ttft_s is not None]
        ttft_p50 = float(np.percentile(ttfts, 50)) if ttfts else None
        ttft_p99 = float(np.percentile(ttfts, 99)) if ttfts else None
    else:
        elapsed = 0.0
        completed_tokens = 0
        throughput = 0.0
        ttft_p50 = None
        ttft_p99 = None

    mean_running = float(np.mean([sample.running for sample in metrics])) if metrics else 0.0
    cached_tokens = int(final_metrics.get("prefix_cached_tokens", 0))
    prompt_tokens = int(final_metrics.get("prefix_prompt_tokens", 0))
    preemption_count = int(final_metrics.get("preemptions", 0))
    # Promote key fields to the top level so downstream analysis doesn't have to
    # reach into run_summary for them.
    peak_blocks_in_use = getattr(summary, "peak_blocks_in_use", 0)
    # prompt_tokens_computed = tokens that were NOT served from cache (actual GPU work)
    prompt_tokens_computed = max(0, prompt_tokens - cached_tokens)
    prompt_tokens_skipped = cached_tokens  # served from shared cache blocks
    theoretical_shareable_fraction = prefix_metadata.get("theoretical_shareable_fraction", None)
    return {
        "prefix_mode": mode,
        "profile": profile,
        "prefix_metadata": prefix_metadata,
        "offered_load_rps": offered_rate,
        "seed": seed,
        "kv_budget_tokens": budget_tokens,
        "completed_requests": len(completed),
        "failed_requests": len(requests) - len(completed),
        "completed_tokens": completed_tokens,
        "completion_window_s": elapsed,
        "throughput_tokens_per_s": throughput,
        # TTFT promoted to top-level (also present inside run_summary)
        "ttft_p50_s": ttft_p50,
        "ttft_p99_s": ttft_p99,
        "mean_running_concurrency": mean_running,
        "prefix_hit_rate": cached_tokens / prompt_tokens if prompt_tokens else 0.0,
        "prefix_cached_tokens": cached_tokens,
        "prefix_prompt_tokens": prompt_tokens,
        # Derived compute load: how many prompt tokens were actually forwarded
        "prompt_tokens_computed": prompt_tokens_computed,
        "prompt_tokens_skipped": prompt_tokens_skipped,
        # Promoted from run_summary for fast per-run indexing
        "peak_blocks_in_use": peak_blocks_in_use,
        "preemption_count": preemption_count,
        "theoretical_shareable_fraction": theoretical_shareable_fraction,
        "blocks_saved": int(final_metrics.get("blocks_saved", 0)),
        "cached_free_blocks": int(final_metrics.get("cached_free_blocks", 0)),
        "cache_evictions": int(final_metrics.get("cache_evictions", 0)),
        "run_summary": asdict(summary),
        "requests": [asdict(request) for request in requests],
        "metric_samples": [asdict(sample) for sample in metrics],
        "gpu_clocks": clocks,
    }


async def run_one(
    trace: SharedPrefixTrace,
    mode: str,
    profile: str,
    offered_rate: float,
    seed: int,
    budget_tokens: int,
    port: int,
    log_dir: Path | None = None,
) -> dict[str, Any]:
    if log_dir is None:
        log_dir = Path("benchmarks/results/prefix_cache/server_logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{profile}_budget{budget_tokens}_rate{offered_rate:g}_seed{seed}_{mode}.log"

    process, log_file, log_path = start_paged_server(mode, budget_tokens, port, log_path=log_path)
    clock_samples: list[dict[str, Any]] = [gpu_clocks()]
    stop_clocks = asyncio.Event()
    clock_task = asyncio.create_task(poll_gpu_clocks(clock_samples, stop_clocks))
    server_stderr = ""
    try:
        summary, requests, metrics = await run_trace_replay(
            base_url=f"http://127.0.0.1:{port}",
            trace=trace.requests,
            offered_rate=offered_rate,
            warmup_count=0,
            stream=True,
            include_token_ids=True,
        )
        final_metrics = await read_metrics(port)
    finally:
        stop_clocks.set()
        await clock_task
        clock_samples.append(gpu_clocks())
        _exit_code, server_stderr = stop_server(process, log_file, log_path, port=port)

    record = summarize_run(
        mode,
        profile,
        trace.metadata,
        offered_rate,
        seed,
        budget_tokens,
        summary,
        requests,
        metrics,
        final_metrics,
        clock_samples,
    )
    # Attach last 200 lines of server stderr for post-mortem if needed
    if server_stderr:
        record["server_stderr_tail"] = server_stderr[-8000:]  # ~200 lines
    record["server_log_path"] = str(log_path)
    return record


def compare_token_outputs(results: dict[str, dict[str, Any]], subset_stride: int = 10) -> list[dict[str, Any]]:
    baseline_requests = {
        request["request_id"]: request
        for request in results["off"]["requests"]
    }
    divergences = []
    for mode in ("share", "full"):
        candidate_requests = {
            request["request_id"]: request
            for request in results[mode]["requests"]
        }
        selected_ids = sorted(baseline_requests)[::subset_stride]
        for request_id in selected_ids:
            baseline = baseline_requests[request_id]
            candidate = candidate_requests.get(request_id)
            if candidate is None:
                divergences.append(
                    {"mode": mode, "request_id": request_id, "error": "No candidate result"}
                )
                continue
            baseline_ids = baseline["token_ids"]
            candidate_ids = candidate["token_ids"]
            first_divergent = next(
                (
                    step
                    for step, (left, right) in enumerate(zip(baseline_ids, candidate_ids))
                    if left != right
                ),
                min(len(baseline_ids), len(candidate_ids)),
            )
            if baseline_ids != candidate_ids:
                margins = baseline.get("token_margins", [])
                divergences.append(
                    {
                        "mode": mode,
                        "request_id": request_id,
                        "first_divergent_step": first_divergent,
                        "off_token_id": (
                            baseline_ids[first_divergent]
                            if first_divergent < len(baseline_ids)
                            else None
                        ),
                        "candidate_token_id": (
                            candidate_ids[first_divergent]
                            if first_divergent < len(candidate_ids)
                            else None
                        ),
                        "off_top1_top2_margin": (
                            margins[first_divergent]
                            if first_divergent < len(margins)
                            else None
                        ),
                    }
                )
    return divergences


def build_profiles(args: argparse.Namespace) -> list[tuple[str, int, int]]:
    """Return list of (profile_name, prefix_count, system_tokens) tuples.

    With --trimmed-grid (default True for new runs), emit only the 5 informative
    profiles at the trimmed lambda points instead of the full 384-cell matrix.
    The compute-skip profile (shared_p1_s512) runs at a separate budget and rate
    set via the COMPUTE_SKIP_ constants when --compute-skip is set.
    """
    if getattr(args, "trimmed_grid", False):
        # Trimmed grid: P=1/4/16 x S=64 + two controls
        profiles: list[tuple[str, int, int]] = [
            (f"shared_p{p}_s64", p, 64)
            for p in (1, 4, 16)
        ]
        profiles.extend(
            [
                ("control_no_share_p200_s64", args.no_share_prefixes, 64),
                ("control_eviction_p64_s128", args.eviction_prefixes, 128),
            ]
        )
    else:
        # Full matrix (kept for backward compatibility)
        profiles = [
            (f"shared_p{prefix_count}_s{system_tokens}", prefix_count, system_tokens)
            for prefix_count in (1, 4, 16)
            for system_tokens in (64, 128)
        ]
        profiles.extend(
            [
                ("control_no_share_p200_s64", args.no_share_prefixes, 64),
                ("control_eviction_p64_s128", args.eviction_prefixes, 128),
            ]
        )
    if getattr(args, "profiles", None):
        selected = set(args.profiles)
        profiles = [p for p in profiles if p[0] in selected]
    return profiles


def save_checkpoint(results_path: Path, results: dict[str, Any]) -> None:
    temporary = results_path.with_suffix(".tmp")
    temporary.write_text(json.dumps(results), encoding="utf-8")
    temporary.replace(results_path)


def generate_plots(results: dict[str, Any], output_dir: Path) -> list[Path]:
    entries = [
        entry
        for entry in results["runs"].values()
        if entry.get("profile") == "shared_p4_s64"
        and entry.get("kv_budget_tokens") == 768
        and entry.get("prefix_mode") in PREFIX_MODES
    ]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for mode in PREFIX_MODES:
        mode_entries = [entry for entry in entries if entry["prefix_mode"] == mode]
        rate_means = []
        ttft_means = []
        rates = []
        for rate in DEFAULT_RATES:
            points = [entry for entry in mode_entries if entry["offered_load_rps"] == rate]
            if points:
                rates.append(rate)
                rate_means.append(float(np.mean([point["throughput_tokens_per_s"] for point in points])))
                valid_ttft = [point["ttft_p99_s"] for point in points if point["ttft_p99_s"] is not None]
                ttft_means.append(float(np.mean(valid_ttft)) if valid_ttft else float("nan"))
        axes[0].plot(rates, rate_means, marker="o", label=mode)
        axes[1].plot(rates, ttft_means, marker="o", label=mode)
    axes[0].set(title="Throughput vs offered load (P=4, S=64, 768-token KV)", xlabel="Offered load (req/s)", ylabel="Completed tokens/s")
    axes[1].set(title="TTFT P99 vs offered load (P=4, S=64, 768-token KV)", xlabel="Offered load (req/s)", ylabel="TTFT P99 (s)")
    for axis in axes:
        axis.grid(True, alpha=0.25)
        axis.legend()
    fig.tight_layout()
    curve_path = output_dir / "prefix_cache_curves.png"
    fig.savefig(curve_path, dpi=160)
    plt.close(fig)

    bar_entries = [
        entry
        for entry in results["runs"].values()
        if "error" not in entry and "prefix_hit_rate" in entry and entry.get("profile", "").startswith("shared_p")
        and "_s64" in entry["profile"]
        and entry.get("kv_budget_tokens") == 768
    ]
    prefixes = (1, 4, 16)
    x = np.arange(len(prefixes))
    width = 0.24
    fig, axis = plt.subplots(figsize=(8, 5))
    for index, mode in enumerate(PREFIX_MODES):
        heights = []
        for prefix_count in prefixes:
            points = [
                entry
                for entry in bar_entries
                if entry["prefix_mode"] == mode
                and entry["profile"].startswith(f"shared_p{prefix_count}_")
            ]
            heights.append(float(np.mean([entry["prefix_hit_rate"] for entry in points])) if points else 0.0)
        axis.bar(x + (index - 1) * width, heights, width, label=mode)
    axis.set_xticks(x, [str(prefix_count) for prefix_count in prefixes])
    axis.set(title="Prefix hit rate vs prefix pool size (S=64, 768-token KV)", xlabel="System-prompt pool P", ylabel="Cached prompt tokens / prompt tokens")
    axis.grid(axis="y", alpha=0.25)
    axis.legend()
    fig.tight_layout()
    hit_path = output_dir / "prefix_cache_hit_rate_by_p.png"
    fig.savefig(hit_path, dpi=160)
    plt.close(fig)
    return [curve_path, hit_path]


async def run_compute_skip_profile(
    args: argparse.Namespace,
    tokenizer: Any,
    results: dict[str, Any],
    results_path: Path,
    trace_dir: Path,
) -> None:
    """Item (5): P=1, S=512, budget=4096 tokens, lambda in {0.5, 1.0}, modes share/full.

    At low load memory is not binding, so 'share' and 'off' should be identical
    in throughput while 'full' wins on TTFT only if compute-skipping saves
    real prefill work.  We report mean prefill_time_s and TTFT P50.
    """
    COMPUTE_SKIP_RATES = getattr(args, "compute_skip_rates", None) or [0.5, 1.0]
    COMPUTE_SKIP_BUDGET = 4096  # tokens
    COMPUTE_SKIP_SYSTEM_TOKENS = 512
    # Include "off" as a baseline so we can confirm share≈off in throughput
    # while full wins on TTFT when compute-skipping is effective.
    COMPUTE_SKIP_MODES = ("off", "share", "full")

    profile = "compute_skip_p1_s512"
    prefix_count = 1
    system_tokens = COMPUTE_SKIP_SYSTEM_TOKENS
    budget_tokens = COMPUTE_SKIP_BUDGET

    # Same 3-row Latin square as the main grid so mode order is
    # properly counterbalanced across seed x rate cells.
    cs_latin_square = (
        ("off", "share", "full"),
        ("share", "full", "off"),
        ("full", "off", "share"),
    )
    for rate in COMPUTE_SKIP_RATES:
        for seed in args.seeds:
            trace = generate_shared_prefix_trace(
                tokenizer=tokenizer,
                num_requests=getattr(args, "compute_skip_requests", None) or args.num_requests,
                arrival_rate=rate,
                prefix_count=prefix_count,
                system_prompt_tokens=system_tokens,
                seed=seed,
                block_size=BLOCK_SIZE,
            )
            trace_path = trace_dir / f"{profile}_lambda{rate:g}_seed{seed}.json"
            save_shared_prefix_trace(trace, trace_path)
            results["trace_metadata"][trace_path.name] = trace.metadata

            scenario_key = f"{profile}|budget={budget_tokens}|lambda={rate:g}|seed={seed}"
            mode_results: dict[str, dict[str, Any]] = {}
            order_idx = (int(rate * 2) + seed + (budget_tokens // 256)) % 3
            order = cs_latin_square[order_idx]
            for mode in order:
                run_key = f"{scenario_key}|mode={mode}"
                if args.resume and run_key in results["runs"]:
                    cached_run = results["runs"][run_key]
                    if "error" not in cached_run and "requests" in cached_run:
                        mode_results[mode] = cached_run
                        continue
                print(
                    f"COMPUTE-SKIP RUN {run_key} theoretical_hit="
                    f"{trace.metadata['theoretical_shareable_fraction']:.4f}",
                    flush=True,
                )
                try:
                    record = await run_one(
                        trace, mode, profile, rate, seed, budget_tokens, args.port,
                    )
                    record["trace_path"] = str(trace_path)
                    results["runs"][run_key] = record
                    mode_results[mode] = record
                except Exception as error:
                    server_log = Path("benchmarks/results/prefix_cache/server_logs") / f"{profile}_budget{budget_tokens}_rate{rate:g}_seed{seed}_{mode}.log"
                    stderr_tail = ""
                    if server_log.exists():
                        try:
                            stderr_tail = server_log.read_text(encoding="utf-8", errors="replace")[-8000:]
                        except Exception:
                            pass
                    failure = {
                        "prefix_mode": mode,
                        "profile": profile,
                        "offered_load_rps": rate,
                        "seed": seed,
                        "kv_budget_tokens": budget_tokens,
                        "error": repr(error),
                        "gpu_clocks": [gpu_clocks()],
                        "server_stderr_tail": stderr_tail,
                        "server_log_path": str(server_log) if server_log.exists() else None,
                    }
                    results["runs"][run_key] = failure
                    print(f"FAILED COMPUTE-SKIP {run_key}: {error!r}", flush=True)
                finally:
                    save_checkpoint(results_path, results)


async def run_matrix(args: argparse.Namespace) -> None:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    output_dir = Path(args.output_dir)
    trace_dir = output_dir / "traces"
    output_dir.mkdir(parents=True, exist_ok=True)
    trace_dir.mkdir(parents=True, exist_ok=True)
    results_path = output_dir / "prefix_cache_benchmark.json"
    if args.resume and results_path.exists():
        results = json.loads(results_path.read_text(encoding="utf-8"))
    else:
        results = {
            "config": {
                "modes": list(PREFIX_MODES),
                "budgets_tokens": args.budget_tokens,
                "rates": args.rates,
                "seeds": args.seeds,
                "num_requests": args.num_requests,
                "block_size": BLOCK_SIZE,
                "tokenizer": args.model_name,
                "trimmed_grid": getattr(args, "trimmed_grid", False),
            },
            "runs": {},
            "comparisons": {},
            "trace_metadata": {},
        }

    traces: dict[tuple[str, float, int], SharedPrefixTrace] = {}
    for profile, prefix_count, system_tokens in build_profiles(args):
        for rate in args.rates:
            for seed in args.seeds:
                trace = generate_shared_prefix_trace(
                    tokenizer=tokenizer,
                    num_requests=args.num_requests,
                    arrival_rate=rate,
                    prefix_count=prefix_count,
                    system_prompt_tokens=system_tokens,
                    seed=seed,
                    block_size=BLOCK_SIZE,
                )
                trace_path = trace_dir / f"{profile}_lambda{rate:g}_seed{seed}.json"
                save_shared_prefix_trace(trace, trace_path)
                traces[(profile, rate, seed)] = trace
                results["trace_metadata"][trace_path.name] = trace.metadata
    save_checkpoint(results_path, results)

    latin_square_orders = (
        ("off", "share", "full"),
        ("share", "full", "off"),
        ("full", "off", "share"),
    )
    for profile, _, _ in build_profiles(args):
        for budget_tokens in args.budget_tokens:
            for rate in args.rates:
                for seed in args.seeds:
                    trace = traces[(profile, rate, seed)]
                    scenario_key = f"{profile}|budget={budget_tokens}|lambda={rate:g}|seed={seed}"
                    mode_results: dict[str, dict[str, Any]] = {}
                    order_idx = (int(rate * 2) + seed + (budget_tokens // 256)) % 3
                    order = latin_square_orders[order_idx]
                    for mode in order:
                        run_key = f"{scenario_key}|mode={mode}"
                        if args.resume and run_key in results["runs"]:
                            cached_run = results["runs"][run_key]
                            if "error" not in cached_run and "requests" in cached_run:
                                mode_results[mode] = cached_run
                                continue
                        print(
                            f"RUN {run_key} theoretical_hit="
                            f"{trace.metadata['theoretical_shareable_fraction']:.4f}",
                            flush=True,
                        )
                        try:
                            record = await run_one(
                                trace,
                                mode,
                                profile,
                                rate,
                                seed,
                                budget_tokens,
                                args.port,
                            )
                            record["trace_path"] = str(
                                trace_dir / f"{profile}_lambda{rate:g}_seed{seed}.json"
                            )
                            results["runs"][run_key] = record
                            mode_results[mode] = record
                        except Exception as error:
                            # IMPORTANT: Preserve the failure record with diagnosis info.
                            # Never delete or skip failed records without named cause.
                            server_log = Path("benchmarks/results/prefix_cache/server_logs") / f"{profile}_budget{budget_tokens}_rate{rate:g}_seed{seed}_{mode}.log"
                            stderr_tail = ""
                            if server_log.exists():
                                try:
                                    stderr_tail = server_log.read_text(encoding="utf-8", errors="replace")[-8000:]
                                except Exception:
                                    pass
                            failure = {
                                "prefix_mode": mode,
                                "profile": profile,
                                "offered_load_rps": rate,
                                "seed": seed,
                                "kv_budget_tokens": budget_tokens,
                                "error": repr(error),
                                "error_type": type(error).__name__,
                                "gpu_clocks": [gpu_clocks()],
                                "server_stderr_tail": stderr_tail,
                                "server_log_path": str(server_log) if server_log.exists() else None,
                            }
                            results["runs"][run_key] = failure
                            print(f"FAILED {run_key}: {error!r}", flush=True)
                        finally:
                            save_checkpoint(results_path, results)

                    if set(mode_results) == set(PREFIX_MODES) and all("requests" in mode_results[m] for m in PREFIX_MODES):
                        divergences = compare_token_outputs(mode_results)
                        results["comparisons"][scenario_key] = {
                            "checked_every_nth_request": 10,
                            "divergences": divergences,
                        }
                        save_checkpoint(results_path, results)

    # Item (5): Compute-skip profile
    if getattr(args, "compute_skip", False):
        await run_compute_skip_profile(args, tokenizer, results, results_path, trace_dir)
        save_checkpoint(results_path, results)

    plots = generate_plots(results, output_dir)
    print(f"Saved results: {results_path}")
    for plot in plots:
        print(f"Saved plot: {plot}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the shared-prefix KV-cache benchmark matrix.")
    parser.add_argument("--model-name", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--num-requests", type=int, default=200)
    # Trimmed grid defaults: lambda in {3,4}, budget 768 only.
    # Pass --rates 1 2 3 4 --budget-tokens 768 1536 to restore the full matrix.
    parser.add_argument("--rates", nargs="+", type=float, default=[3.0, 4.0])
    parser.add_argument("--seeds", nargs="+", type=int, default=list(DEFAULT_SEEDS))
    parser.add_argument("--budget-tokens", nargs="+", type=int, default=[768],
                        help="KV budget(s) in tokens. Default: [768] (trimmed grid).")
    parser.add_argument("--no-share-prefixes", type=int, default=200)
    parser.add_argument("--eviction-prefixes", type=int, default=64)
    parser.add_argument("--output-dir", default="benchmarks/results/prefix_cache")
    parser.add_argument("--port", type=int, default=PORT)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--trimmed-grid", action="store_true", default=True,
        help="Only run P=1/4/16 x S=64 profiles plus controls (default: True).",
    )
    parser.add_argument(
        "--full-grid", dest="trimmed_grid", action="store_false",
        help="Run the full PxS matrix (6 profiles x 2 system-token sizes).",
    )
    parser.add_argument(
        "--profiles", nargs="+", type=str, default=None,
        help="Filter specific profiles to run (e.g. shared_p16_s64).",
    )
    parser.add_argument(
        "--compute-skip", action="store_true", default=False,
        help="Run the compute-skipping profile: P=1, S=512, budget=4096, lambda in {0.5,1.0}, modes share/full.",
    )
    parser.add_argument(
        "--compute-skip-rates", nargs="+", type=float, default=None,
        help="Override lambda values for the compute-skip profile (default: 0.5 1.0).",
    )
    parser.add_argument(
        "--compute-skip-requests", type=int, default=None,
        help="Override num-requests for compute-skip profile (default: same as --num-requests).",
    )
    args = parser.parse_args()
    asyncio.run(run_matrix(args))


if __name__ == "__main__":
    main()
