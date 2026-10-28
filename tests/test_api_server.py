"""
Comprehensive tests for FastAPI serving layer (engine.serving.api_server).

Verifies:
1. Engine start and clean shutdown via app lifespan.
2. GET /health and GET /metrics (running, waiting, free blocks, total blocks, preemptions, tokens generated).
3. POST /v1/completions non-streaming returns correct structure with usage token counts.
4. Input validation: temperature != 0 returns 400.
5. Input validation: prompt tokens + max_tokens > pool capacity returns 400.
6. POST /v1/completions streaming returns Server-Sent Events ending with data: [DONE].
7. Official openai Python client pointed at local base_url (stream and non-stream).
8. Curl-like raw HTTP requests.
9. Mid-stream client disconnect aborts sequence and returns blocks to the pool.
10. High concurrency test with 20 concurrent httpx clients, asserting free blocks equal total blocks afterward.
"""
from __future__ import annotations

import asyncio
import json
import subprocess
import sys
import time
import urllib.request
from typing import List

import httpx
import pytest
from openai import OpenAI

from engine.memory.block_allocator import BlockAllocator
from engine.scheduler.continuous_batching import Scheduler
from engine.serving.api_server import create_app
from engine.serving.async_engine import AsyncEngine


def _wait_for_server(port: int, process: subprocess.Popen, timeout_s: float = 60.0) -> None:
    """Poll /health until the server is ready or the process dies.

    Raises RuntimeError with stderr tail if startup fails.
    """
    url = f"http://127.0.0.1:{port}/health"
    deadline = time.monotonic() + timeout_s
    last_error: Exception | None = None
    while time.monotonic() < deadline:
        if process.poll() is not None:
            # Process exited — drain stderr
            stderr_text = ""
            try:
                _, stderr_text = process.communicate(timeout=2)
            except Exception:
                pass
            raise RuntimeError(
                f"Server process exited early (code={process.returncode}) on port {port}.\n"
                f"STDERR:\n{stderr_text[-3000:]}"
            )
        try:
            with urllib.request.urlopen(url, timeout=1.5) as response:
                if response.status == 200:
                    return
        except Exception as e:
            last_error = e
        time.sleep(0.5)
    raise RuntimeError(
        f"Server on port {port} not ready after {timeout_s}s. Last error: {last_error}"
    )


class FastMockModelRunner:
    """Mock model runner that returns predictable tokens with minimal latency."""
    def __init__(self, step_delay_s: float = 0.002):
        self.step_delay_s = step_delay_s

    def step(self, batch) -> List[int]:
        if self.step_delay_s > 0:
            time.sleep(self.step_delay_s)
        from engine.scheduler.continuous_batching import BatchType
        if batch.batch_type == BatchType.DECODE:
            for seq in batch.sequences:
                if seq.block_table is not None:
                    seq.block_table.num_tokens += 1
        return [200 + seq.seq_id for seq in batch.sequences]


def build_test_engine(num_blocks: int = 64, block_size: int = 4) -> AsyncEngine:
    allocator = BlockAllocator(num_blocks=num_blocks)
    scheduler = Scheduler(
        block_allocator=allocator,
        block_size=block_size,
        watermark_blocks=1,
    )
    runner = FastMockModelRunner()
    return AsyncEngine(
        model_runner=runner,
        scheduler=scheduler,
        idle_sleep_s=0.001,
    )


@pytest.mark.anyio
async def test_lifespan_health_and_metrics():
    total_blocks = 32
    engine = build_test_engine(num_blocks=total_blocks, block_size=4)
    app = create_app(engine)

    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
            # /health
            res = await client.get("/health")
            assert res.status_code == 200
            data = res.json()
            assert data["status"] == "ok"
            assert data["running"] is True

            # /metrics initial
            res = await client.get("/metrics")
            assert res.status_code == 200
            metrics = res.json()
            assert metrics["running"] == 0
            assert metrics["waiting"] == 0
            assert metrics["free_blocks"] == total_blocks
            assert metrics["total_blocks"] == total_blocks
            assert metrics["preemptions"] == 0
            assert metrics["tokens_generated"] == 0
            assert metrics["cached_free_blocks"] == 0
            assert metrics["prefix_hit_rate"] == 0.0
            assert metrics["cache_evictions"] == 0


@pytest.mark.anyio
async def test_completions_non_streaming():
    total_blocks = 32
    engine = build_test_engine(num_blocks=total_blocks, block_size=4)
    app = create_app(engine)

    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
            req_payload = {
                "prompt": "Hello world test prompt",
                "max_tokens": 5,
                "temperature": 0.0,
                "stream": False,
            }
            res = await client.post("/v1/completions", json=req_payload)
            assert res.status_code == 200
            body = res.json()
            assert body["object"] == "text_completion"
            assert len(body["choices"]) == 1
            assert body["choices"][0]["finish_reason"] == "length"
            assert "usage" in body
            assert body["usage"]["completion_tokens"] == 5
            assert body["usage"]["prompt_tokens"] == 4  # 4 words in prompt
            assert body["usage"]["total_tokens"] == 9

            # Verify blocks are freed back to pool
            allocator = engine.scheduler.block_allocator
            assert len(allocator.free_block_ids) == total_blocks

            # Check metrics updated
            res_m = await client.get("/metrics")
            m = res_m.json()
            assert m["tokens_generated"] == 5
            assert m["free_blocks"] == total_blocks


@pytest.mark.anyio
async def test_input_validations():
    total_blocks = 8  # 8 blocks * 4 tokens = 32 tokens max capacity
    engine = build_test_engine(num_blocks=total_blocks, block_size=4)
    app = create_app(engine)

    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
            # 1. Non-zero temperature should fail
            res_temp = await client.post("/v1/completions", json={"prompt": "hi", "temperature": 0.7})
            assert res_temp.status_code == 400
            assert "Only temperature=0 is supported" in res_temp.json()["detail"]

            # 2. Exceeding usable capacity (accounting for watermark) should fail with 400
            # 8 blocks of size 4 can hold at most 32 tokens, watermark=1 means 7 blocks usable = 28 tokens.
            res_mem = await client.post("/v1/completions", json={"prompt": "short prompt", "max_tokens": 30})
            assert res_mem.status_code == 400
            assert "exceeding usable pool capacity" in res_mem.json()["detail"]

            # Assert no memory leak
            allocator = engine.scheduler.block_allocator
            assert len(allocator.free_block_ids) == total_blocks


@pytest.mark.anyio
async def test_completions_streaming_sse():
    total_blocks = 32
    engine = build_test_engine(num_blocks=total_blocks, block_size=4)
    app = create_app(engine)

    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
            req_payload = {
                "prompt": "Test streaming SSE prompt",
                "max_tokens": 4,
                "temperature": 0.0,
                "stream": True,
                "include_token_ids": True,
            }
            async with client.stream("POST", "/v1/completions", json=req_payload) as response:
                assert response.status_code == 200
                assert "text/event-stream" in response.headers["content-type"]

                lines = []
                async for line in response.aiter_lines():
                    if line:
                        lines.append(line)

                # Check that it ends with data: [DONE]
                assert lines[-1] == "data: [DONE]"

                data_lines = [l for l in lines if l.startswith("data: ") and l != "data: [DONE]"]
                assert len(data_lines) >= 4  # tokens + final chunk

                # Parse first data chunk
                first_chunk = json.loads(data_lines[0][6:])
                assert first_chunk["object"] == "text_completion"
                assert "text" in first_chunk["choices"][0]
                assert isinstance(first_chunk["token_id"], int)
                assert first_chunk["top1_top2_margin"] is None

            # Ensure all blocks freed
            allocator = engine.scheduler.block_allocator
            assert len(allocator.free_block_ids) == total_blocks


@pytest.mark.anyio
async def test_midstream_client_disconnect_reclaims_blocks():
    """
    Spawns live server and connects with httpx client, breaking mid-stream.
    Verifies that the sequence is aborted and all its blocks are returned to the pool.
    """
    port = 8121
    server_process = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "engine.serving.api_server:app", "--port", str(port)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        _wait_for_server(port, server_process, timeout_s=60.0)
    except RuntimeError as exc:
        server_process.kill()
        pytest.fail(str(exc))
    try:
        async with httpx.AsyncClient(base_url=f"http://127.0.0.1:{port}", timeout=10.0) as client:
            req_payload = {
                "prompt": "Midstream cancellation test prompt",
                "max_tokens": 30,
                "stream": True,
            }
            async with client.stream("POST", "/v1/completions", json=req_payload) as response:
                assert response.status_code == 200
                count = 0
                async for line in response.aiter_lines():
                    if line.startswith("data: ") and line != "data: [DONE]":
                        count += 1
                        if count == 2:
                            # Break out prematurely to disconnect socket
                            break

            # Allow engine background loop to process abort at step boundary
            reclaimed = False
            for _ in range(50):
                await asyncio.sleep(0.1)
                metrics_resp = await client.get("/metrics")
                assert metrics_resp.status_code == 200
                metrics = metrics_resp.json()
                if metrics["running"] == 0 and metrics["waiting"] == 0 and metrics["free_blocks"] == metrics["total_blocks"]:
                    reclaimed = True
                    break

            assert reclaimed, f"Engine did not reclaim blocks in time: {metrics}"
    finally:
        server_process.terminate()
        try:
            _, stderr_out = server_process.communicate(timeout=5)
        except subprocess.TimeoutExpired:
            server_process.kill()
            _, stderr_out = server_process.communicate(timeout=3)
        if stderr_out and stderr_out.strip():
            print(f"\n[test_midstream] Server stderr:\n{stderr_out[-2000:]}")


@pytest.mark.anyio
async def test_concurrency_20_clients_and_reclamation():
    """
    Launches 20 concurrent httpx clients performing completions (mix of stream and non-stream).
    Asserts free blocks equal total blocks afterward.
    """
    total_blocks = 128
    engine = build_test_engine(num_blocks=total_blocks, block_size=4)
    app = create_app(engine)

    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app),
            base_url="http://test",
            timeout=30.0,
        ) as client:
            num_clients = 20

            async def worker(client_id: int):
                is_stream = (client_id % 2 == 0)
                payload = {
                    "prompt": f"Client {client_id} concurrent request prompt text",
                    "max_tokens": 5,
                    "temperature": 0.0,
                    "stream": is_stream,
                    "ignore_eos": True,
                }
                if is_stream:
                    chunks = []
                    async with client.stream("POST", "/v1/completions", json=payload) as resp:
                        assert resp.status_code == 200
                        async for line in resp.aiter_lines():
                            if line:
                                chunks.append(line)
                    assert chunks[-1] == "data: [DONE]"
                else:
                    resp = await client.post("/v1/completions", json=payload)
                    assert resp.status_code == 200
                    data = resp.json()
                    assert data["usage"]["completion_tokens"] == 5

            # Execute all 20 requests concurrently
            await asyncio.gather(*(worker(i) for i in range(num_clients)))

            # Allow any final scheduler cleanup step to settle
            await asyncio.sleep(0.05)

            # Assert free blocks equal total blocks afterward
            allocator = engine.scheduler.block_allocator
            assert len(allocator.free_block_ids) == total_blocks, (
                f"Expected {total_blocks} free blocks, but found {len(allocator.free_block_ids)}. "
                f"Leaked: {total_blocks - len(allocator.free_block_ids)} blocks!"
            )
            assert len(engine.scheduler.running) == 0
            assert len(engine.scheduler.waiting) == 0

            # Assert metrics reflect all generated tokens
            res_m = await client.get("/metrics")
            m = res_m.json()
            assert m["free_blocks"] == total_blocks
            assert m["total_blocks"] == total_blocks
            assert m["tokens_generated"] == num_clients * 5


def test_curl_and_openai_client_live_server():
    """
    Spawns uvicorn on an ephemeral port and tests:
    1. curl against POST /v1/completions
    2. official openai Python client pointed at base_url
    """
    port = 8117
    server_process = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "engine.serving.api_server:app", "--port", str(port)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        _wait_for_server(port, server_process, timeout_s=60.0)
    except RuntimeError as exc:
        server_process.kill()
        pytest.fail(str(exc))

    stderr_tail = ""
    try:
        # 1. Test curl
        curl_payload = json.dumps({"prompt": "curl live test", "max_tokens": 3, "temperature": 0.0})
        curl_res = subprocess.run(
            ["curl.exe", "-s", "-X", "POST", f"http://127.0.0.1:{port}/v1/completions",
             "-H", "Content-Type: application/json", "-d", curl_payload],
            capture_output=True,
            text=True,
        )
        assert curl_res.returncode == 0
        curl_data = json.loads(curl_res.stdout)
        assert curl_data["object"] == "text_completion"
        assert curl_data["usage"]["completion_tokens"] == 3

        # 2. Test official OpenAI client
        openai_client = OpenAI(base_url=f"http://127.0.0.1:{port}/v1", api_key="dummy")
        # Non-streaming
        comp = openai_client.completions.create(
            model="default",
            prompt="openai python test",
            max_tokens=4,
            temperature=0.0,
        )
        assert comp.object == "text_completion"
        assert comp.usage.completion_tokens == 4

        # Streaming
        stream_comp = openai_client.completions.create(
            model="default",
            prompt="openai stream test",
            max_tokens=4,
            temperature=0.0,
            stream=True,
        )
        collected_tokens = []
        for chunk in stream_comp:
            if chunk.choices and chunk.choices[0].text:
                collected_tokens.append(chunk.choices[0].text)
        assert len(collected_tokens) == 4

    finally:
        server_process.terminate()
        try:
            _, stderr_out = server_process.communicate(timeout=5)
            stderr_tail = stderr_out or ""
        except subprocess.TimeoutExpired:
            server_process.kill()
            _, stderr_out = server_process.communicate(timeout=3)
            stderr_tail = stderr_out or ""
        if stderr_tail.strip():
            print(f"\n[test_curl_openai] Server stderr:\n{stderr_tail[-2000:]}")


@pytest.mark.anyio
async def test_oversized_then_normal_request():
    """Item (1) & Boundary tests:
    - Exactly fits usable capacity (num_blocks - watermark) -> succeeds.
    - Exactly one block over usable capacity -> rejected with 400.
    - Large oversized request -> rejected with 400.
    - Follow-on normal request succeeds, all blocks are free, and check_invariants() passes.
    """
    total_blocks = 8  # 8 blocks * 4 tokens = 32 tokens max capacity
    # build_test_engine uses watermark_blocks=1 -> usable blocks = 7 -> 28 tokens
    engine = build_test_engine(num_blocks=total_blocks, block_size=4)
    app = create_app(engine)

    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
            allocator = engine.scheduler.block_allocator

            # 1. Boundary: exactly fits usable capacity (7 blocks = 28 tokens)
            # prompt = 4 words (4 tokens), max_tokens = 24 -> total 28 tokens -> 7 blocks.
            res_exact = await client.post(
                "/v1/completions",
                json={"prompt": "one two three four", "max_tokens": 24, "temperature": 0.0},
            )
            assert res_exact.status_code == 200
            assert allocator.num_available() == total_blocks
            engine.scheduler.check_invariants()

            # 2. Boundary: exactly one block over usable capacity (8 blocks = 32 tokens, leaves 0 for watermark)
            # prompt = 4 words, max_tokens = 25 -> total 29 tokens -> 8 blocks needed > 7 usable.
            res_one_over = await client.post(
                "/v1/completions",
                json={"prompt": "one two three four", "max_tokens": 25, "temperature": 0.0},
            )
            assert res_one_over.status_code == 400
            assert "exceeding usable pool capacity" in res_one_over.json()["detail"]
            assert allocator.num_available() == total_blocks
            engine.scheduler.check_invariants()

            # 3. Oversized request: prompt 10 words + max_tokens 30 = 40 tokens > 32 tokens
            oversized_payload = {
                "prompt": "one two three four five six seven eight nine ten",
                "max_tokens": 30,
                "temperature": 0.0,
            }
            res_bad = await client.post("/v1/completions", json=oversized_payload)
            assert res_bad.status_code == 400
            assert "exceeding usable pool capacity" in res_bad.json()["detail"]
            assert allocator.num_available() == total_blocks
            engine.scheduler.check_invariants()

            # 4. Normal request to the same server must succeed cleanly
            normal_payload = {
                "prompt": "normal prompt",
                "max_tokens": 5,
                "temperature": 0.0,
            }
            res_ok = await client.post("/v1/completions", json=normal_payload)
            assert res_ok.status_code == 200
            data = res_ok.json()
            assert data["object"] == "text_completion"
            assert data["usage"]["completion_tokens"] == 5

            # Blocks should be freed back and invariants must hold
            assert allocator.num_available() == total_blocks
            engine.scheduler.check_invariants()
