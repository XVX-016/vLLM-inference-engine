"""Run the shared-prefix paged-serving benchmark matrix."""
from __future__ import annotations

import argparse
import asyncio
import json
import os
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


def wait_server_ready(port: int, process: subprocess.Popen, timeout_s: float = 180.0) -> None:
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
                    return
                raise RuntimeError(f"Unexpected paged server health response: {health!r}")
        except RuntimeError:
            raise
        except Exception as error:
            last_error = error
            time.sleep(0.5)
    raise RuntimeError(f"Paged server failed to become ready: {last_error}")


def start_paged_server(prefix_mode: str, kv_budget_tokens: int, port: int) -> subprocess.Popen:
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
            "error",
        ],
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        wait_server_ready(port, process)
    except Exception:
        process.terminate()
        process.wait(timeout=15)
        raise
    return process


def stop_server(process: subprocess.Popen) -> None:
    process.terminate()
    try:
        process.wait(timeout=20)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=10)


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
        "ttft_p50_s": ttft_p50,
        "ttft_p99_s": ttft_p99,
        "mean_running_concurrency": mean_running,
        "prefix_hit_rate": cached_tokens / prompt_tokens if prompt_tokens else 0.0,
        "prefix_cached_tokens": cached_tokens,
        "prefix_prompt_tokens": prompt_tokens,
        "blocks_saved": int(final_metrics.get("blocks_saved", 0)),
        "cached_free_blocks": int(final_metrics.get("cached_free_blocks", 0)),
        "cache_evictions": int(final_metrics.get("cache_evictions", 0)),
        "preemption_count": int(final_metrics.get("preemptions", 0)),
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
) -> dict[str, Any]:
    process = start_paged_server(mode, budget_tokens, port)
    clock_samples: list[dict[str, Any]] = [gpu_clocks()]
    stop_clocks = asyncio.Event()
    clock_task = asyncio.create_task(poll_gpu_clocks(clock_samples, stop_clocks))
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
        stop_server(process)

    return summarize_run(
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
        if entry["profile"].startswith("shared_p")
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
                "modes": PREFIX_MODES,
                "budgets_tokens": [768, args.larger_budget],
                "rates": args.rates,
                "seeds": args.seeds,
                "num_requests": args.num_requests,
                "block_size": BLOCK_SIZE,
                "tokenizer": args.model_name,
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

    run_order = ("off", "share", "full")
    for profile, _, _ in build_profiles(args):
        for budget_tokens in (768, args.larger_budget):
            for rate in args.rates:
                for seed in args.seeds:
                    trace = traces[(profile, rate, seed)]
                    scenario_key = f"{profile}|budget={budget_tokens}|lambda={rate:g}|seed={seed}"
                    mode_results: dict[str, dict[str, Any]] = {}
                    order = run_order if (seed + int(rate) + budget_tokens) % 2 == 0 else tuple(reversed(run_order))
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
                            failure = {
                                "prefix_mode": mode,
                                "profile": profile,
                                "offered_load_rps": rate,
                                "seed": seed,
                                "kv_budget_tokens": budget_tokens,
                                "error": repr(error),
                                "gpu_clocks": [gpu_clocks()],
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

    plots = generate_plots(results, output_dir)
    print(f"Saved results: {results_path}")
    for plot in plots:
        print(f"Saved plot: {plot}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the shared-prefix KV-cache benchmark matrix.")
    parser.add_argument("--model-name", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--num-requests", type=int, default=200)
    parser.add_argument("--rates", nargs="+", type=float, default=list(DEFAULT_RATES))
    parser.add_argument("--seeds", nargs="+", type=int, default=list(DEFAULT_SEEDS))
    parser.add_argument("--larger-budget", type=int, default=1536)
    parser.add_argument("--no-share-prefixes", type=int, default=200)
    parser.add_argument("--eviction-prefixes", type=int, default=64)
    parser.add_argument("--output-dir", default="benchmarks/results/prefix_cache")
    parser.add_argument("--port", type=int, default=PORT)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    asyncio.run(run_matrix(args))


if __name__ == "__main__":
    main()
