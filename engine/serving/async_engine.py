"""
AsyncEngine for Continuous Batching Inference (Phase 5).

Wraps the Scheduler and Model execution in an asyncio-based serving engine.
Supports:
1. add_request(prompt, max_tokens, ...) returning an async generator of token IDs or text.
2. Single background asyncio task driving iteration-level continuous batching.
3. run_in_executor offloading of model forward execution so event loop remains responsive.
4. Client cancellation handling by aborting sequence and reclaiming physical blocks at step boundary.
"""
from __future__ import annotations

import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor
from typing import Any, AsyncGenerator, Dict, List, Optional, Set, Union

from engine.scheduler.continuous_batching import (
    BatchType,
    ScheduledBatch,
    Scheduler,
    SeqStatus,
    Sequence,
)

logger = logging.getLogger(__name__)


class _StopTokenSentinel:
    """Sentinel indicating completion of sequence generation."""
    pass


class AsyncEngine:
    """
    Asynchronous Continuous Batching Inference Engine.

    Manages concurrent request streams and drives the continuous batching
    scheduler and model forward execution asynchronously.
    """

    def __init__(
        self,
        model_runner: Any,
        scheduler: Scheduler,
        tokenizer: Optional[Any] = None,
        idle_sleep_s: float = 0.005,
        executor: Optional[ThreadPoolExecutor] = None,
        run_logger: Optional[Any] = None,
    ):
        """
        Args:
            model_runner: Model execution engine. Can be KernelPagedManualDecode,
                          FakeModelEngine, or any object implementing step(batch) -> List[int]
                          or prefill(seq) and decode_step(seqs).
            scheduler: Continuous batching Scheduler instance.
            tokenizer: Optional Hugging Face tokenizer for token string decoding.
            idle_sleep_s: Sleep duration (in seconds) when scheduler has no work.
            executor: ThreadPoolExecutor for offloading synchronous model compute.
            run_logger: Optional RunLogger instance to record per-iteration and per-request metrics.
        """
        self.model_runner = model_runner
        self.scheduler = scheduler
        self.tokenizer = tokenizer
        self.idle_sleep_s = idle_sleep_s
        self.executor = executor or ThreadPoolExecutor(max_workers=1)
        self.run_logger = run_logger
        self._iteration_count: int = 0
        self.total_tokens_generated: int = 0

        # Per-request queues mapping seq_id -> asyncio.Queue[Union[int, _StopTokenSentinel]]
        self._output_queues: Dict[int, asyncio.Queue] = {}
        # Set of sequence IDs aborted by client disconnect / cancellation
        self._aborted_seq_ids: Set[int] = set()

        # Engine background lifecycle
        self._loop_task: Optional[asyncio.Task] = None
        self._is_running: bool = False
        self._lock = asyncio.Lock()

    def start(self) -> None:
        """Starts the background engine loop if not already running."""
        if not self._is_running:
            self._is_running = True
            loop = asyncio.get_running_loop()
            self._loop_task = loop.create_task(self._run_engine_loop())

    async def stop(self) -> None:
        """Stops the background engine loop and shuts down worker thread pool."""
        self._is_running = False
        if self._loop_task is not None:
            self._loop_task.cancel()
            try:
                await self._loop_task
            except asyncio.CancelledError:
                pass
            self._loop_task = None
        self.executor.shutdown(wait=False)

    def _execute_model_batch_sync(self, batch: ScheduledBatch) -> List[int]:
        """
        Synchronous model forward computation intended to run in a worker thread.
        Supports both model_runner.step(batch) (FakeModelEngine) and
        model_runner.prefill / decode_step (KernelPagedManualDecode).
        """
        if hasattr(self.model_runner, "step"):
            return self.model_runner.step(batch)

        # KernelPagedManualDecode interface
        if batch.batch_type == BatchType.PREFILL:
            seq = batch.sequences[0]
            tok = self.model_runner.prefill(seq, allocator=self.scheduler.block_allocator)
            return [tok]
        else:
            return self.model_runner.decode_step(batch.sequences, allocator=self.scheduler.block_allocator)

    async def _run_engine_loop(self) -> None:
        """
        Single background asyncio task driving iteration-level continuous batching.
        1. Checks for aborted sequences and purges them.
        2. Calls scheduler.step() to retrieve ScheduledBatch.
        3. If idle, sleeps for idle_sleep_s without blocking.
        4. Offloads GPU/model forward pass to thread pool via run_in_executor.
        5. Updates scheduler state and routes tokens to corresponding asyncio.Queues.
        """
        loop = asyncio.get_running_loop()

        while self._is_running:
            # 1. Process client cancellations / aborts at step boundary
            if self._aborted_seq_ids:
                async with self._lock:
                    to_remove = list(self._aborted_seq_ids)
                    for seq_id in to_remove:
                        self.scheduler.abort_sequence(seq_id)
                        queue = self._output_queues.pop(seq_id, None)
                        if queue is not None:
                            await queue.put(_StopTokenSentinel())
                        self._aborted_seq_ids.remove(seq_id)

            # 2. Schedule next batch
            try:
                batch = self.scheduler.step()
            except MemoryError as mem_err:
                # A single sequence exceeded the block pool capacity.
                # Log it, signal the sequence's output queue with the error,
                # and continue — do NOT crash the entire engine loop.
                logger.warning(f"MemoryError in scheduler.step(): {mem_err}")
                # Find any sequence without an output queue entry and clean up
                # The scheduler already removed/finished the problematic sequence.
                # Signal all queues that have no corresponding running/waiting seq.
                known_seq_ids = (
                    {s.seq_id for s in self.scheduler.running}
                    | {s.seq_id for s in self.scheduler.waiting}
                )
                orphaned = [sid for sid in list(self._output_queues) if sid not in known_seq_ids]
                for sid in orphaned:
                    q = self._output_queues.pop(sid, None)
                    if q is not None:
                        await q.put(mem_err)
                await asyncio.sleep(self.idle_sleep_s)
                continue

            # 3. If idle, sleep briefly
            if batch is None or batch.is_empty:
                await asyncio.sleep(self.idle_sleep_s)
                continue

            # 4. Offload model forward computation to thread pool
            t_step_start = asyncio.get_running_loop().time()
            try:
                next_tokens: List[int] = await loop.run_in_executor(
                    self.executor,
                    self._execute_model_batch_sync,
                    batch,
                )
            except Exception as e:
                logger.error(f"Error during model forward execution: {e}")
                for seq in batch.sequences:
                    queue = self._output_queues.pop(seq.seq_id, None)
                    if queue is not None:
                        await queue.put(e)
                continue
            step_duration_s = asyncio.get_running_loop().time() - t_step_start

            # 5. Route new tokens to client queues and advance scheduler
            self.scheduler.post_step(batch, next_tokens)
            self.total_tokens_generated += len(next_tokens)

            # Log per-iteration metrics
            if self.run_logger is not None:
                self._iteration_count += 1
                free_blocks = len(self.scheduler.block_allocator.free_block_ids)
                self.run_logger.log_iteration(
                    iteration=self._iteration_count,
                    iteration_type=batch.batch_type.value,
                    running_count=len(self.scheduler.running),
                    waiting_count=len(self.scheduler.waiting),
                    free_blocks=free_blocks,
                    preemptions=self.scheduler.preemption_count,
                    step_time_s=step_duration_s,
                    tokens_generated=len(next_tokens),
                )

            for seq, token_id in zip(batch.sequences, next_tokens):
                # Don't push to aborted sequences
                if seq.seq_id in self._aborted_seq_ids:
                    continue

                # Record time-to-first-token
                if self.run_logger is not None and seq.total_generated_tokens == 1:
                    self.run_logger.record_request_first_token(seq.seq_id)

                queue = self._output_queues.get(seq.seq_id)
                if queue is not None:
                    await queue.put(token_id)
                    # If sequence is finished, signal end of stream
                    if seq.is_finished(self.scheduler.eos_token_id):
                        if self.run_logger is not None:
                            self.run_logger.record_request_finish(seq.seq_id, seq.total_generated_tokens)
                        await queue.put(_StopTokenSentinel())
                        self._output_queues.pop(seq.seq_id, None)

    async def add_request(
        self,
        prompt: str,
        max_tokens: int = 16,
        prompt_token_ids: Optional[List[int]] = None,
        ignore_eos: bool = False,
    ) -> AsyncGenerator[int, None]:
        """
        Submits a new generation request and returns an async generator yielding
        token IDs as they are produced by the engine.

        Handles client cancellation by aborting the sequence and freeing its
        blocks at the next step boundary.
        """
        self.start()

        if prompt_token_ids is None and self.tokenizer is not None:
            prompt_token_ids = self.tokenizer.encode(prompt)

        prompt_len = len(prompt_token_ids) if prompt_token_ids else len(prompt.split())

        seq = self.scheduler.add_request(
            prompt=prompt,
            max_new_tokens=max_tokens,
            prompt_token_ids=prompt_token_ids,
            ignore_eos=ignore_eos,
        )
        seq_id = seq.seq_id

        if self.run_logger is not None:
            self.run_logger.record_request_arrival(seq_id, prompt_len)

        queue: asyncio.Queue = asyncio.Queue()
        self._output_queues[seq_id] = queue

        finished = False
        try:
            while True:
                item = await queue.get()
                if isinstance(item, _StopTokenSentinel):
                    finished = True
                    break
                if isinstance(item, Exception):
                    raise item
                yield item
        except (asyncio.CancelledError, GeneratorExit):
            # Client cancelled or generator closed
            async with self._lock:
                self._aborted_seq_ids.add(seq_id)
            raise
        finally:
            if not finished:
                async with self._lock:
                    self._aborted_seq_ids.add(seq_id)
            self._output_queues.pop(seq_id, None)
