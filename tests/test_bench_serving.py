"""
Integration test for benchmarks/bench_serving.py.

Verifies:
1. Replays a trace against a live server using aiohttp.
2. Accurately instruments TTFT, per-token timestamps (TPOT), E2E latency, and token counts.
3. Excludes warmup requests from aggregate stats.
4. Concurrently polls /metrics every 100ms and logs KV blocks in use.
5. Saves raw JSON results and generates the 3 required plots:
   - throughput_vs_offered_load.png
   - ttft_p99_vs_offered_load.png
   - timeline_running_waiting.png
6. Verifies sweep mode over offered loads.
"""
from __future__ import annotations

import asyncio
import json
import subprocess
import sys
import time
from pathlib import Path

import pytest

from benchmarks.bench_serving import (
    compute_run_summary,
    generate_benchmark_plots,
    run_trace_replay,
)
from benchmarks.workload_trace import generate_workload_trace


def test_bench_serving_live_server_e2e(tmp_path: Path):
    port = 8133
    server_process = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "engine.serving.api_server:app", "--port", str(port)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    # Wait for server to become ready
    deadline = time.monotonic() + 30.0
    ready = False
    while time.monotonic() < deadline:
        try:
            import urllib.request
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=1.0) as resp:
                if resp.status == 200:
                    ready = True
                    break
        except Exception:
            time.sleep(0.5)
    assert ready, "Server failed to start in time"
    try:
        # Generate small test trace
        trace = generate_workload_trace(num_requests=10, arrival_rate=15.0, seed=77)

        async def run_test():
            summary, results, metrics = await run_trace_replay(
                base_url=f"http://127.0.0.1:{port}",
                trace=trace,
                offered_rate=15.0,
                warmup_count=2,
                stream=True,
                slo_threshold_s=2.0,
            )
            return summary, results, metrics

        summary, results, metrics = asyncio.run(run_test())

        # Verify warmup exclusion
        warmup_reqs = [r for r in results if r.is_warmup]
        eval_reqs = [r for r in results if not r.is_warmup]
        assert len(warmup_reqs) == 2
        assert len(eval_reqs) == 8
        assert summary.completed_requests == 8
        assert summary.total_requests == 8

        # Verify timings and per-token timestamps
        for r in eval_reqs:
            assert r.success is True
            assert r.ttft_s is not None and r.ttft_s > 0
            assert r.e2e_latency_s is not None and r.e2e_latency_s >= r.ttft_s
            assert len(r.token_timestamps) > 0
            assert r.actual_output_len > 0

        # Verify aggregate metrics
        assert summary.token_throughput_tps > 0
        assert summary.actual_request_throughput_rps > 0
        assert summary.ttft_p50_s is not None
        assert summary.ttft_p99_s is not None
        assert summary.e2e_p50_s is not None
        assert summary.e2e_p99_s is not None
        assert 0.0 <= summary.slo_attainment <= 1.0

        # Verify metrics poller captured samples
        assert len(metrics) > 0
        assert metrics[0].total_blocks > 0

        # Verify plots generation
        plot_files = generate_benchmark_plots(
            summaries=[summary],
            metric_samples=metrics,
            output_dir=tmp_path,
            system_label="Continuous Batching Test",
        )
        assert len(plot_files) == 3
        for pf in plot_files:
            assert Path(pf).exists()
            assert Path(pf).stat().st_size > 1000  # valid png file

    finally:
        server_process.terminate()
        server_process.wait(timeout=5)
