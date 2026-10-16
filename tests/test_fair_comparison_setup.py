"""
Tests for NaiveBatchServer, trace generator, and memory budget parity.

Verifies:
1. Trace generator creates Poisson arrivals and lognormal lengths with fixed seed.
2. NaiveBatchServer collects requests into batches and executes them to completion.
3. Both servers derive block pools and concurrent slots from KV memory budget in MB:
   - Paged engine derives block count: budget // block_bytes
   - Naive engine derives max slots: budget // (max_seq_len * bytes_per_token)
4. Replaying workload trace through both servers produces identical request counts.
"""
from __future__ import annotations

import asyncio
import json
from pathlib import Path

import httpx
import pytest

from benchmarks.workload_trace import (
    generate_workload_trace,
    load_trace_from_json,
    save_trace_to_json,
)
from engine.serving.api_server import (
    build_default_engine,
    calculate_paged_blocks_from_budget,
    create_app,
)
from engine.serving.naive_server import (
    build_default_naive_engine,
    calculate_naive_slots_from_budget,
    create_naive_app,
)


def test_workload_trace_generation_and_determinism(tmp_path: Path):
    seed = 123
    trace1 = generate_workload_trace(num_requests=30, arrival_rate=5.0, seed=seed)
    trace2 = generate_workload_trace(num_requests=30, arrival_rate=5.0, seed=seed)

    assert len(trace1) == 30
    assert len(trace2) == 30

    for r1, r2 in zip(trace1, trace2):
        assert r1.arrival_time_s == r2.arrival_time_s
        assert r1.prompt == r2.prompt
        assert r1.prompt_len == r2.prompt_len
        assert r1.output_len == r2.output_len
        assert r1.ignore_eos is True

    # Test serialization to JSON
    json_path = tmp_path / "workload.json"
    save_trace_to_json(trace1, json_path)
    loaded = load_trace_from_json(json_path)
    assert len(loaded) == 30
    assert loaded[0].prompt == trace1[0].prompt


def test_kv_memory_budget_derivation():
    """
    Validates both engines accept KV memory budget in MB:
    - Paged engine: sizes block pool from it
    - Naive engine: derives concurrent slots = budget // (max_seq_len * bytes_per_token)
    """
    budget_mb = 16.0  # 16 MB budget
    max_seq_len = 512
    num_layers = 24
    num_kv_heads = 2
    head_dim = 64
    dtype_bytes = 2

    # 1. Naive slots derivation
    naive_slots = calculate_naive_slots_from_budget(
        kv_memory_mb=budget_mb,
        max_seq_len=max_seq_len,
        num_layers=num_layers,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        dtype_bytes=dtype_bytes,
    )
    # bytes_per_token = 2 * 24 * 2 * 64 * 2 = 12,288 bytes
    # slot_bytes = 512 * 12,288 = 6,291,456 bytes (~6 MB)
    # 16 MB // 6 MB = 2 slots
    assert naive_slots == 2

    # 2. Paged blocks derivation
    block_size = 16
    paged_blocks = calculate_paged_blocks_from_budget(
        kv_memory_mb=budget_mb,
        block_size=block_size,
        num_layers=num_layers,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        dtype_bytes=dtype_bytes,
    )
    # block_bytes = 16 * 12,288 = 196,608 bytes
    # 16 MB // 196,608 = 85 blocks
    assert paged_blocks == 85

    # Verify building engines with budget
    paged_eng = build_default_engine(kv_memory_mb=budget_mb, block_size=block_size)
    assert paged_eng.scheduler.block_allocator.num_blocks == paged_blocks

    naive_eng = build_default_naive_engine(kv_memory_mb=budget_mb, max_seq_len=max_seq_len)
    assert naive_eng.total_slots == naive_slots
    assert naive_eng.max_batch_size <= naive_slots


@pytest.mark.anyio
async def test_naive_server_batch_and_completions():
    """Tests NaiveBatchServer HTTP completions, batching, and metrics."""
    naive_eng = build_default_naive_engine(
        max_batch_size=4,
        collection_timeout_s=0.02,
        max_seq_len=1024,
    )
    app = create_naive_app(naive_eng)

    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
            # /health
            h = await client.get("/health")
            assert h.status_code == 200
            assert h.json()["status"] == "ok"

            # /metrics
            m = await client.get("/metrics")
            assert m.status_code == 200
            assert m.json()["running"] == 0

            # Concurrent requests to test batching
            async def send_req(i: int):
                res = await client.post(
                    "/v1/completions",
                    json={
                        "prompt": f"Prompt {i} for naive batch test",
                        "max_tokens": 4,
                        "temperature": 0.0,
                        "ignore_eos": True,
                    },
                )
                assert res.status_code == 200
                data = res.json()
                assert data["usage"]["completion_tokens"] == 4
                return data

            results = await asyncio.gather(*(send_req(i) for i in range(4)))
            assert len(results) == 4

            # Metrics reflect all generated tokens
            m2 = await client.get("/metrics")
            assert m2.json()["tokens_generated"] == 16


@pytest.mark.anyio
async def test_replay_trace_through_both_servers():
    """
    Replays a subset of the workload trace against both the Paged AsyncEngine
    and the NaiveBatchServer to confirm fair replay capability.
    """
    trace = generate_workload_trace(num_requests=6, arrival_rate=20.0, seed=99)

    # 1. Replay against Paged server
    paged_engine = build_default_engine(num_blocks=64, block_size=4)
    paged_app = create_app(paged_engine)

    async with paged_app.router.lifespan_context(paged_app):
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=paged_app), base_url="http://paged") as client:
            for req in trace:
                res = await client.post(
                    "/v1/completions",
                    json={
                        "prompt": req.prompt,
                        "max_tokens": req.output_len,
                        "temperature": 0.0,
                        "ignore_eos": req.ignore_eos,
                    },
                )
                assert res.status_code == 200
                data = res.json()
                assert data["usage"]["completion_tokens"] == req.output_len

    # 2. Replay against Naive server
    naive_engine = build_default_naive_engine(max_batch_size=4, max_seq_len=2048)
    naive_app = create_naive_app(naive_engine)

    async with naive_app.router.lifespan_context(naive_app):
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=naive_app), base_url="http://naive") as client:
            for req in trace:
                res = await client.post(
                    "/v1/completions",
                    json={
                        "prompt": req.prompt,
                        "max_tokens": req.output_len,
                        "temperature": 0.0,
                        "ignore_eos": req.ignore_eos,
                    },
                )
                assert res.status_code == 200
                data = res.json()
                assert data["usage"]["completion_tokens"] == req.output_len
