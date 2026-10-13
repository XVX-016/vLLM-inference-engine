"""
Unit and integration tests for AsyncEngine (Phase 5).

Tests:
1. Iteration-level continuous batching: submits requests at staggered times and verifies
   that a late-arriving short request finishes BEFORE an earlier long request.
2. Client cancellation: cancelling an active async generator aborts the sequence and
   reclaims its physical blocks at the next step boundary without leaking memory.
3. Concurrent requests: multiple simultaneous async streams receive all expected tokens.
"""
from __future__ import annotations

import asyncio
import time
from typing import Dict, List, Optional

import pytest

from engine.memory.block_allocator import BlockAllocator
from engine.scheduler.continuous_batching import ScheduledBatch, Scheduler
from engine.serving.async_engine import AsyncEngine


class ScriptedAsyncModelRunner:
    """
    Simulates model forward passes with controllable latency to accurately test
    staggered arrivals, asynchronous loop non-blocking behavior, and preemption.
    """

    def __init__(self, step_delay_s: float = 0.01, default_token: int = 100):
        self.step_delay_s = step_delay_s
        self.default_token = default_token

    def step(self, batch: ScheduledBatch) -> List[int]:
        if self.step_delay_s > 0:
            time.sleep(self.step_delay_s)
        # Yield unique identifiable tokens or sequence-specific tokens
        return [self.default_token + seq.seq_id for seq in batch.sequences]


@pytest.mark.anyio
async def test_late_arriving_short_request_finishes_before_early_long_request():
    """
    Submits a long request at t=0, and a short request after a brief delay.
    Verifies that the short request finishes BEFORE the earlier long request.
    This directly proves iteration-level continuous batching is functioning.
    """
    total_blocks = 32
    block_size = 4
    allocator = BlockAllocator(num_blocks=total_blocks)
    scheduler = Scheduler(
        block_allocator=allocator,
        block_size=block_size,
        watermark_blocks=1,
    )
    runner = ScriptedAsyncModelRunner(step_delay_s=0.015)
    engine = AsyncEngine(
        model_runner=runner,
        scheduler=scheduler,
        idle_sleep_s=0.002,
    )

    completion_times: Dict[str, float] = {}
    tokens_received: Dict[str, List[int]] = {"long": [], "short": []}

    async def run_long_request():
        # Long request: 16 new tokens
        t0 = time.perf_counter()
        async for tok in engine.add_request("long prompt", max_tokens=16, prompt_token_ids=[1, 2]):
            tokens_received["long"].append(tok)
        completion_times["long"] = time.perf_counter() - t0

    async def run_short_request():
        # Stagger arrival: wait 0.05s so long request is already in decode phase
        await asyncio.sleep(0.05)
        t0 = time.perf_counter()
        # Short request: only 3 new tokens
        async for tok in engine.add_request("short prompt", max_tokens=3, prompt_token_ids=[3, 4]):
            tokens_received["short"].append(tok)
        completion_times["short_duration"] = time.perf_counter() - t0
        completion_times["short_finish_timestamp"] = time.perf_counter()

    t_start = time.perf_counter()
    task_long = asyncio.create_task(run_long_request())
    task_short = asyncio.create_task(run_short_request())

    # Wait for short request to finish first
    await task_short
    short_finished_at = time.perf_counter() - t_start

    # Verify short request has already completed all 3 tokens
    assert len(tokens_received["short"]) == 3
    # Verify long request is NOT yet completed
    assert not task_long.done(), "Long request should NOT have finished before short request!"
    assert len(tokens_received["long"]) < 16, "Long request still running decode steps!"

    # Now wait for long request
    await task_long
    long_finished_at = time.perf_counter() - t_start

    assert len(tokens_received["long"]) == 16
    assert short_finished_at < long_finished_at, (
        f"Short request finished at {short_finished_at:.3f}s, "
        f"long request finished at {long_finished_at:.3f}s. "
        f"Expected short to finish earlier!"
    )

    await engine.stop()

    # Verify no leaked memory
    assert len(allocator.free_block_ids) == total_blocks


@pytest.mark.anyio
async def test_client_cancellation_aborts_and_frees_blocks():
    """
    Submits a request, consumes a few tokens, and cancels the consumer task.
    Verifies that the sequence is aborted and its allocated physical blocks
    are cleanly returned to the allocator at the next step boundary.
    """
    total_blocks = 16
    block_size = 4
    allocator = BlockAllocator(num_blocks=total_blocks)
    scheduler = Scheduler(
        block_allocator=allocator,
        block_size=block_size,
        watermark_blocks=1,
    )
    runner = ScriptedAsyncModelRunner(step_delay_s=0.01)
    engine = AsyncEngine(
        model_runner=runner,
        scheduler=scheduler,
        idle_sleep_s=0.002,
    )

    consumed_tokens = []

    async def client_worker():
        async for tok in engine.add_request("cancellation prompt", max_tokens=20, prompt_token_ids=[1, 2, 3]):
            consumed_tokens.append(tok)
            if len(consumed_tokens) == 3:
                # Early cancellation after 3 tokens
                break

    task = asyncio.create_task(client_worker())
    await task

    # Give engine background loop 2 steps to process abort at step boundary
    await asyncio.sleep(0.05)

    # All queues should be empty and all blocks freed back to allocator
    assert len(scheduler.running) == 0
    assert len(scheduler.waiting) == 0
    assert len(allocator.free_block_ids) == total_blocks
    assert len(allocator.ref_counts) == 0

    await engine.stop()


@pytest.mark.anyio
async def test_concurrent_streaming_requests():
    """
    Submits 8 concurrent requests simultaneously and verifies all complete
    with their exact requested token counts.
    """
    total_blocks = 64
    block_size = 4
    allocator = BlockAllocator(num_blocks=total_blocks)
    scheduler = Scheduler(
        block_allocator=allocator,
        block_size=block_size,
        watermark_blocks=1,
        max_batch_size=8,
    )
    runner = ScriptedAsyncModelRunner(step_delay_s=0.005)
    engine = AsyncEngine(
        model_runner=runner,
        scheduler=scheduler,
        idle_sleep_s=0.002,
    )

    num_reqs = 8
    target_tokens = 6
    results = [[] for _ in range(num_reqs)]

    async def stream_one(i: int):
        async for tok in engine.add_request(f"prompt_{i}", max_tokens=target_tokens, prompt_token_ids=[i, i + 1]):
            results[i].append(tok)

    await asyncio.gather(*(stream_one(i) for i in range(num_reqs)))

    for i in range(num_reqs):
        assert len(results[i]) == target_tokens, f"Request {i} expected {target_tokens} tokens, got {len(results[i])}"

    await engine.stop()
    assert len(allocator.free_block_ids) == total_blocks
