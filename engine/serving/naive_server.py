"""
NaiveBatchServer: Phase 0 static-cache engine wrapped behind the OpenAI-compatible HTTP interface.

Features:
1. Implements request-level batching (collects requests up to max_batch_size B or collection_timeout_s).
2. Reserves a static KV-cache of max_seq_len per slot.
3. Derives max concurrent slots from KV memory budget:
   slots = budget_bytes // (max_seq_len * bytes_per_token).
4. Runs each batch to completion before starting the next batch.
5. Exposes POST /v1/completions (streaming SSE & non-streaming), GET /health, GET /metrics.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import time
import traceback
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Any, AsyncGenerator, Callable, Dict, List, Optional, Union

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from transformers.cache_utils import StaticCache

logger = logging.getLogger(__name__)


class NaiveHFModelRunner:
    """
    Real Phase 0 static-cache model runner using HuggingFace Qwen forward pass.
    Allocates slots * max_seq_len static KV cache upfront.
    Executes prefill with left padding and lockstep decode.
    """

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-0.5B-Instruct",
        max_slots: int = 3,
        max_seq_len: int = 173,
        device: str = "cuda",
        dtype: torch.dtype = torch.float16,
    ):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.dtype = dtype
        self.max_slots = max_slots
        self.max_seq_len = max_seq_len

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"

        self.pad_token_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else 0

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            dtype=self.dtype,
        ).to(self.device)
        self.model.eval()

        config = self.model.config
        self.num_layers = getattr(config, "num_hidden_layers", 24)
        self.num_kv_heads = getattr(config, "num_key_value_heads", getattr(config, "num_attention_heads", 2))
        self.head_dim = getattr(config, "head_dim", config.hidden_size // getattr(config, "num_attention_heads", 16))

        # Static pre-allocation upfront for max_slots * max_seq_len
        self.cache = StaticCache(
            config=config,
            max_batch_size=self.max_slots,
            max_cache_len=self.max_seq_len,
            device=self.device,
            dtype=self.dtype,
        )
        self.cache.early_initialization(
            batch_size=self.max_slots,
            num_heads=self.num_kv_heads,
            head_dim=self.head_dim,
            dtype=self.dtype,
            device=self.device,
        )
        self.decode_times: List[float] = []

    @torch.inference_mode()
    def execute_batch(
        self,
        batch: List[PendingNaiveRequest],
        token_callback,
    ) -> None:
        """
        Executes a batch of requests to completion using left-padded prefill
        and lockstep decode, invoking token_callback(req, token_id, is_last) for each output.
        Builds padded batches directly from prompt_token_ids without invoking tokenizer.
        """
        if not batch:
            return

        B = len(batch)
        pad_count = self.max_slots - B
        
        # Build left-padded tensors directly from prompt_token_ids in the runner
        batch_ids = [r.prompt_token_ids for r in batch]
        max_prompt_len = max(len(ids) for ids in batch_ids)

        input_ids = torch.full((self.max_slots, max_prompt_len), self.pad_token_id, dtype=torch.long, device=self.device)
        attention_mask = torch.zeros((self.max_slots, max_prompt_len), dtype=torch.long, device=self.device)

        for i, ids in enumerate(batch_ids):
            cur_len = len(ids)
            offset = max_prompt_len - cur_len
            input_ids[i, offset:] = torch.tensor(ids, dtype=torch.long, device=self.device)
            attention_mask[i, offset:] = 1

        # Reset static cache buffers for the new batch
        self.cache.reset()

        # ── runtime grad probe (printed once per process) ───────────────────
        if not getattr(self, "_grad_probed", False):
            self._grad_probed = True
            grad_on = torch.is_grad_enabled()
            logger.info(
                "[NaiveRunner] GRAD PROBE: torch.is_grad_enabled()=%s  "
                "(expected False inside @inference_mode)", grad_on
            )
            assert not grad_on, (
                "execute_batch must run under torch.inference_mode(); "
                "is_grad_enabled()=True means the decorator is missing!"
            )
        # ────────────────────────────────────────────────────────────────────

        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        out = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=self.cache,
            use_cache=True,
        )
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

        # ── logits.requires_grad probe (printed once) ────────────────────────
        if not getattr(self, "_logits_probed", False):
            self._logits_probed = True
            rg = out.logits.requires_grad
            logger.info(
                "[NaiveRunner] LOGITS PROBE: out.logits.requires_grad=%s  "
                "(expected False inside @inference_mode)", rg
            )
            assert not rg, (
                "out.logits.requires_grad=True — inference_mode not active!"
            )
        # ─────────────────────────────────────────────────────────────────────

        next_tok = torch.argmax(out.logits[:, -1, :], dim=-1, keepdim=True)
        cur_mask = torch.cat(
            [attention_mask, torch.ones((self.max_slots, 1), dtype=attention_mask.dtype, device=self.device)],
            dim=-1,
        )
        if pad_count > 0:
            cur_mask[-pad_count:, -1] = 0

        gen_counts = [1] * B
        for i, req in enumerate(batch):
            tok_id = next_tok[i].item()
            token_callback(req, tok_id)

        max_tokens = max(r.max_tokens for r in batch)

        # 2. Lockstep decode loop to batch completion
        for step in range(1, max_tokens):
            active = [gen_counts[i] < batch[i].max_tokens and not batch[i].aborted for i in range(B)]
            if not any(active):
                break

            if self.device.type == "cuda":
                torch.cuda.synchronize(self.device)
            t0 = time.perf_counter()
            dec_out = self.model(
                input_ids=next_tok,
                attention_mask=cur_mask,
                past_key_values=self.cache,
                use_cache=True,
            )
            if self.device.type == "cuda":
                torch.cuda.synchronize(self.device)
            self.decode_times.append(time.perf_counter() - t0)

            next_tok = torch.argmax(dec_out.logits[:, -1, :], dim=-1, keepdim=True)
            for i, req in enumerate(batch):
                if gen_counts[i] < req.max_tokens and not req.aborted:
                    tok_id = next_tok[i].item()
                    gen_counts[i] += 1
                    token_callback(req, tok_id)

            cur_mask = torch.cat(
                [cur_mask, torch.ones((self.max_slots, 1), dtype=cur_mask.dtype, device=self.device)],
                dim=-1,
            )
            if pad_count > 0:
                cur_mask[-pad_count:, -1] = 0


def calculate_naive_slots_from_budget(
    kv_memory_mb: float,
    max_seq_len: int = 2048,
    num_layers: int = 24,
    num_kv_heads: int = 2,
    head_dim: int = 64,
    dtype_bytes: int = 2,
) -> int:
    """
    Computes max concurrent slots that can be allocated for static KV cache:
    slots = budget_bytes // (max_seq_len * bytes_per_token)
    where bytes_per_token = 2 (K+V) * num_layers * num_kv_heads * head_dim * dtype_bytes
    """
    bytes_per_token = 2 * num_layers * num_kv_heads * head_dim * dtype_bytes
    budget_bytes = int(kv_memory_mb * 1024 * 1024)
    slot_bytes = max_seq_len * bytes_per_token
    return max(1, budget_bytes // slot_bytes)


@dataclass
class PendingNaiveRequest:
    request_id: str
    prompt: str
    prompt_token_ids: List[int]
    max_tokens: int
    ignore_eos: bool
    queue: asyncio.Queue
    created_ts: int = field(default_factory=lambda: int(time.time()))
    aborted: bool = False


class _StopTokenSentinel:
    pass


class NaiveBatchEngine:
    """
    Orchestrates request-level batching.
    Collects requests into batches of size up to max_batch_size (or collection_timeout_s),
    reserves static cache slots, and executes the batch to completion before starting the next.
    """

    def __init__(
        self,
        model_runner: Any,
        tokenizer: Optional[Any] = None,
        max_batch_size: int = 8,
        collection_timeout_s: float = 0.05,
        max_seq_len: int = 2048,
        kv_memory_mb: Optional[float] = None,
        num_layers: int = 24,
        num_kv_heads: int = 2,
        head_dim: int = 64,
        dtype_bytes: int = 2,
    ):
        self.model_runner = model_runner
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.collection_timeout_s = collection_timeout_s

        # Compute max concurrent slots from KV budget
        if kv_memory_mb is not None:
            derived_slots = calculate_naive_slots_from_budget(
                kv_memory_mb=kv_memory_mb,
                max_seq_len=max_seq_len,
                num_layers=num_layers,
                num_kv_heads=num_kv_heads,
                head_dim=head_dim,
                dtype_bytes=dtype_bytes,
            )
            self.max_batch_size = min(max_batch_size, derived_slots)
            self.total_slots = derived_slots
        else:
            self.max_batch_size = max_batch_size
            self.total_slots = max_batch_size

        self.num_layers = num_layers
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.dtype_bytes = dtype_bytes
        self.bytes_per_token = 2 * num_layers * num_kv_heads * head_dim * dtype_bytes
        self.slot_bytes = max_seq_len * self.bytes_per_token

        self._pending_queue: asyncio.Queue[PendingNaiveRequest] = asyncio.Queue()
        self._loop_task: Optional[asyncio.Task] = None
        self._is_running: bool = False

        # Metrics tracking
        self.running_count: int = 0
        self.waiting_count: int = 0
        self.total_tokens_generated: int = 0
        self.active_tokens_stored: int = 0
        self.aborted_requests_count: int = 0
        self.preemption_count: int = 0  # Static batching does not preempt mid-batch
        self.start_time: float = time.time()

    def start(self) -> None:
        if not self._is_running:
            self._is_running = True
            loop = asyncio.get_running_loop()
            self._loop_task = loop.create_task(self._batching_loop())

    async def stop(self) -> None:
        self._is_running = False
        if self._loop_task is not None:
            self._loop_task.cancel()
            try:
                await self._loop_task
            except asyncio.CancelledError:
                pass
            self._loop_task = None

    async def submit_request(
        self,
        prompt: str,
        max_tokens: int,
        prompt_token_ids: Optional[List[int]] = None,
        ignore_eos: bool = False,
        disconnect_check: Optional[Callable[[], Any]] = None,
    ) -> AsyncGenerator[int, None]:
        self.start()
        req_id = f"naive-{int(time.time()*1000)}-{id(prompt)}"
        queue: asyncio.Queue = asyncio.Queue()
        pending = PendingNaiveRequest(
            request_id=req_id,
            prompt=prompt,
            prompt_token_ids=prompt_token_ids or [hash(w) % 10000 for w in prompt.split()] or [1],
            max_tokens=max_tokens,
            ignore_eos=ignore_eos,
            queue=queue,
        )
        if disconnect_check is not None:
            pending.disconnect_check = disconnect_check
        await self._pending_queue.put(pending)

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
            pending.aborted = True
            self.aborted_requests_count += 1
            raise
        finally:
            if not finished:
                if not pending.aborted:
                    pending.aborted = True
                    self.aborted_requests_count += 1

    async def _batching_loop(self) -> None:
        """
        Request-level batching loop:
        1. Waits for at least one request.
        2. Gathers up to max_batch_size requests within collection_timeout_s.
        3. Runs the entire batch to completion (all sequences finish) before taking the next batch.
        """
        loop = asyncio.get_running_loop()

        while self._is_running:
            self.waiting_count = self._pending_queue.qsize()

            try:
                first_req = await asyncio.wait_for(self._pending_queue.get(), timeout=0.01)
            except asyncio.TimeoutError:
                continue

            current_batch: List[PendingNaiveRequest] = [first_req]
            deadline = loop.time() + self.collection_timeout_s

            # Collect requests up to max_batch_size or until timeout expires
            while len(current_batch) < self.max_batch_size:
                rem_time = deadline - loop.time()
                if rem_time <= 0:
                    break
                try:
                    req = await asyncio.wait_for(self._pending_queue.get(), timeout=rem_time)
                    current_batch.append(req)
                except asyncio.TimeoutError:
                    break

            # Filter out requests aborted or disconnected while waiting in queue
            active_batch: List[PendingNaiveRequest] = []
            for r in current_batch:
                if r.aborted:
                    continue
                # Check if client disconnected while waiting in queue
                if hasattr(r, "disconnect_check") and callable(r.disconnect_check):
                    try:
                        disc = r.disconnect_check()
                        if asyncio.iscoroutine(disc):
                            disc = await asyncio.wait_for(disc, timeout=0.001)
                        if disc:
                            r.aborted = True
                            self.aborted_requests_count += 1
                            logger.info(f"Dropped queued request {r.request_id}: client disconnected.")
                            continue
                    except Exception:
                        pass
                active_batch.append(r)

            if not active_batch:
                continue

            self.running_count = len(active_batch)
            self.waiting_count = self._pending_queue.qsize()

            # Execute batch to completion (request-level batching)
            try:
                await self._run_batch_to_completion(active_batch)
            except Exception as e:
                logger.error(f"Error executing naive batch: {e}")
                for r in active_batch:
                    await r.queue.put(e)
            finally:
                self.running_count = 0
                self.waiting_count = self._pending_queue.qsize()

    async def _run_batch_to_completion(self, batch: List[PendingNaiveRequest]) -> None:
        """
        Executes static batch decode loop until every sequence finishes.
        """
        loop = asyncio.get_running_loop()

        # Initialize active tokens stored (prompts)
        self.active_tokens_stored = sum(len(r.prompt_token_ids) for r in batch if not r.aborted)

        if hasattr(self.model_runner, "execute_batch"):
            def _cb(req: PendingNaiveRequest, tok_id: int):
                self.total_tokens_generated += 1
                self.active_tokens_stored += 1
                if not req.aborted:
                    loop.call_soon_threadsafe(req.queue.put_nowait, tok_id)

            try:
                await loop.run_in_executor(None, self.model_runner.execute_batch, batch, _cb)
            finally:
                self.active_tokens_stored = 0
                for req in batch:
                    await req.queue.put(_StopTokenSentinel())
            return

        max_steps = max(r.max_tokens for r in batch)
        seq_tokens_gen: Dict[str, int] = {r.request_id: 0 for r in batch}
        seq_finished: Dict[str, bool] = {r.request_id: False for r in batch}

        try:
            for step_idx in range(max_steps):
                active_reqs = [r for r in batch if not seq_finished[r.request_id] and not r.aborted]
                if not active_reqs:
                    break

                # Call model runner step
                if hasattr(self.model_runner, "step_naive"):
                    next_tokens = await loop.run_in_executor(None, self.model_runner.step_naive, active_reqs, step_idx)
                elif hasattr(self.model_runner, "step"):
                    class _BatchShim:
                        def __init__(self, sequences):
                            self.sequences = sequences
                    next_tokens = await loop.run_in_executor(
                        None,
                        self.model_runner.step,
                        _BatchShim([type("SeqShim", (), {"seq_id": i})() for i, _ in enumerate(active_reqs)])
                    )
                else:
                    await asyncio.sleep(0.005)
                    next_tokens = [200 + i for i in range(len(active_reqs))]

                self.total_tokens_generated += len(next_tokens)
                self.active_tokens_stored += len(next_tokens)

                for req, tok in zip(active_reqs, next_tokens):
                    seq_tokens_gen[req.request_id] += 1
                    if not req.aborted:
                        await req.queue.put(tok)

                    # Termination check
                    if seq_tokens_gen[req.request_id] >= req.max_tokens:
                        seq_finished[req.request_id] = True

                    if seq_finished[req.request_id] and not req.aborted:
                        await req.queue.put(_StopTokenSentinel())
        finally:
            self.active_tokens_stored = 0

        # Cleanly signal completion for all
        for req in batch:
            if not seq_finished[req.request_id] and not req.aborted:
                await req.queue.put(_StopTokenSentinel())


# Global naive engine holder
_naive_engine: Optional[NaiveBatchEngine] = None


def set_global_naive_engine(engine: Optional[NaiveBatchEngine]) -> None:
    global _naive_engine
    _naive_engine = engine


def get_global_naive_engine() -> NaiveBatchEngine:
    global _naive_engine
    if _naive_engine is None:
        raise RuntimeError("NaiveBatchEngine is not initialized.")
    return _naive_engine


def build_default_naive_engine(
    kv_memory_mb: Optional[float] = None,
    max_batch_size: int = 8,
    max_seq_len: int = 2048,
    collection_timeout_s: float = 0.05,
    runner: Optional[Any] = None,
    tokenizer: Optional[Any] = None,
) -> NaiveBatchEngine:
    env_budget = os.environ.get("NAIVE_KV_MEMORY_MB")
    if kv_memory_mb is not None:
        budget_mb = float(kv_memory_mb)
    elif env_budget is not None:
        budget_mb = float(env_budget)
    else:
        budget_mb = None

    if "NAIVE_MAX_SEQ_LEN" in os.environ:
        max_seq_len = int(os.environ["NAIVE_MAX_SEQ_LEN"])
    if "NAIVE_MAX_BATCH_SIZE" in os.environ:
        max_batch_size = int(os.environ["NAIVE_MAX_BATCH_SIZE"])

    # Compute slots
    if budget_mb is not None:
        derived_slots = calculate_naive_slots_from_budget(
            kv_memory_mb=budget_mb,
            max_seq_len=max_seq_len,
        )
        slots = min(max_batch_size, derived_slots)
    else:
        slots = max_batch_size

    if runner is None:
        logger.info(f"Initializing NaiveHFModelRunner (slots={slots}, max_seq_len={max_seq_len})...")
        runner = NaiveHFModelRunner(
            model_name="Qwen/Qwen2.5-0.5B-Instruct",
            max_slots=slots,
            max_seq_len=max_seq_len,
        )
        if tokenizer is None:
            tokenizer = runner.tokenizer

    return NaiveBatchEngine(
        model_runner=runner,
        tokenizer=tokenizer,
        max_batch_size=max_batch_size,
        collection_timeout_s=collection_timeout_s,
        max_seq_len=max_seq_len,
        kv_memory_mb=budget_mb,
    )


@asynccontextmanager
async def naive_lifespan(app: FastAPI):
    global _naive_engine
    if _naive_engine is None:
        _naive_engine = build_default_naive_engine()
    _naive_engine.start()
    try:
        yield
    finally:
        if _naive_engine is not None:
            await _naive_engine.stop()


class CompletionRequest(BaseModel):
    model: Optional[str] = "default"
    prompt: Union[str, List[str]]
    max_tokens: Optional[int] = Field(default=16, ge=1)
    temperature: Optional[float] = 0.0
    stream: Optional[bool] = False
    stop: Optional[Union[str, List[str]]] = None
    ignore_eos: Optional[bool] = False


def create_naive_app(engine: Optional[NaiveBatchEngine] = None) -> FastAPI:
    if engine is not None:
        set_global_naive_engine(engine)

    app = FastAPI(title="Naive Static-Cache Serving Server", lifespan=naive_lifespan)

    @app.get("/health")
    def health():
        eng = get_global_naive_engine()
        runner_type = "hf" if hasattr(eng.model_runner, "model") or isinstance(eng.model_runner, NaiveHFModelRunner) else "mock"
        return {"status": "ok", "running": eng._is_running, "runner": runner_type}

    @app.get("/metrics")
    def metrics():
        eng = get_global_naive_engine()
        free_slots = max(0, eng.total_slots - eng.running_count)
        reserved_kv_bytes = eng.running_count * eng.slot_bytes
        total_reserved_kv_bytes = eng.total_slots * eng.slot_bytes
        active_token_kv_bytes = eng.active_tokens_stored * eng.bytes_per_token

        allocated_kv_bytes = eng.total_slots * eng.slot_bytes  # slots * max_seq_len * bytes_per_token
        preallocation_mode = "preallocated_slots"  # static preallocation: derived slots x max_seq_len

        # GPU device memory stats
        mem_alloc_mb = 0.0
        mem_res_mb = 0.0
        if torch.cuda.is_available():
            mem_alloc_mb = torch.cuda.memory_allocated() / (1024 * 1024)
            mem_res_mb = torch.cuda.memory_reserved() / (1024 * 1024)

        elapsed = max(0.001, time.time() - getattr(eng, "start_time", time.time()))
        tokens_per_sec = eng.total_tokens_generated / elapsed

        return {
            "running": eng.running_count,
            "waiting": eng.waiting_count,
            "free_blocks": free_slots,
            "total_blocks": eng.total_slots,
            "max_seq_len": eng.max_seq_len,
            "bytes_per_token": eng.bytes_per_token,
            "slot_bytes": eng.slot_bytes,
            "reserved_kv_bytes": reserved_kv_bytes,
            "total_reserved_kv_bytes": total_reserved_kv_bytes,
            "allocated_kv_bytes": allocated_kv_bytes,
            "preallocation_mode": preallocation_mode,
            "active_token_kv_bytes": active_token_kv_bytes,
            "tokens_stored": eng.active_tokens_stored,
            "preemptions": 0,
            "tokens_generated": eng.total_tokens_generated,
            "tokens_per_second": tokens_per_sec,
            "aborted_requests": getattr(eng, "aborted_requests_count", 0),
            "memory_allocated": mem_alloc_mb,
            "memory_reserved": mem_res_mb,
            "memory_allocated_mb": mem_alloc_mb,
            "memory_reserved_mb": mem_res_mb,
        }

    @app.middleware("http")
    async def catch_exceptions_middleware(request: Request, call_next):
        try:
            return await call_next(request)
        except Exception as exc:
            req_id = request.headers.get("x-request-id", f"req-{int(time.time()*1000)}-{id(request)}")
            tb = traceback.format_exc()
            logger.error(
                f"[500 CATCH-ALL] Request ID: {req_id} | Path: {request.url.path} | "
                f"Exception: {type(exc).__name__}: {exc}\nFull Traceback:\n{tb}"
            )
            return JSONResponse(
                status_code=500,
                content={
                    "error": {
                        "message": f"Internal server error: {type(exc).__name__}: {exc}",
                        "type": "internal_server_error",
                        "request_id": req_id,
                        "traceback": tb,
                    }
                },
            )

    @app.post("/v1/completions")
    async def completions(req: CompletionRequest, raw_request: Request):
        eng = get_global_naive_engine()

        if req.temperature is not None and req.temperature != 0.0:
            raise HTTPException(
                status_code=400,
                detail=f"Only temperature=0 is supported currently, got {req.temperature}",
            )

        prompt_str = req.prompt if isinstance(req.prompt, str) else req.prompt[0]
        max_tokens = req.max_tokens or 16

        if eng.tokenizer is not None:
            prompt_token_ids = eng.tokenizer.encode(prompt_str)
        else:
            prompt_token_ids = [hash(w) % 10000 for w in prompt_str.split()] or [1]

        prompt_len = len(prompt_token_ids)
        total_tokens = prompt_len + max_tokens

        # Check against static max_seq_len capacity
        if total_tokens > eng.max_seq_len:
            raise HTTPException(
                status_code=400,
                detail=f"Prompt len ({prompt_len}) + max_tokens ({max_tokens}) exceeds max_seq_len ({eng.max_seq_len})",
            )

        stop_list = [req.stop] if isinstance(req.stop, str) else (req.stop or [])
        req_id = f"naive-cmpl-{int(time.time()*1000)}-{id(raw_request)}"
        created_ts = int(time.time())

        token_gen = eng.submit_request(
            prompt=prompt_str,
            max_tokens=max_tokens,
            prompt_token_ids=prompt_token_ids,
            ignore_eos=bool(req.ignore_eos),
            disconnect_check=raw_request.is_disconnected,
        )

        def decode_token(tok_id: int) -> str:
            if eng.tokenizer is not None:
                return eng.tokenizer.decode([tok_id])
            return f" token_{tok_id}"

        # Streaming SSE
        if req.stream:
            async def event_generator() -> AsyncGenerator[str, None]:
                accumulated_text = ""
                finish_reason: Optional[str] = None
                try:
                    async for tok in token_gen:
                        try:
                            if await asyncio.wait_for(raw_request.is_disconnected(), timeout=0.0001):
                                break
                        except (asyncio.TimeoutError, TimeoutError):
                            pass

                        tok_str = decode_token(tok)
                        accumulated_text += tok_str

                        hit_stop = False
                        for stop_seq in stop_list:
                            if stop_seq and stop_seq in accumulated_text:
                                hit_stop = True
                                finish_reason = "stop"
                                break

                        chunk = {
                            "id": req_id,
                            "object": "text_completion",
                            "created": created_ts,
                            "model": req.model or "default",
                            "choices": [
                                {
                                    "text": tok_str,
                                    "index": 0,
                                    "logprobs": None,
                                    "finish_reason": finish_reason if hit_stop else None,
                                }
                            ],
                        }
                        yield f"data: {json.dumps(chunk)}\n\n"
                        if hit_stop:
                            break

                    final_chunk = {
                        "id": req_id,
                        "object": "text_completion",
                        "created": created_ts,
                        "model": req.model or "default",
                        "choices": [
                            {
                                "text": "",
                                "index": 0,
                                "logprobs": None,
                                "finish_reason": finish_reason or "length",
                            }
                        ],
                    }
                    yield f"data: {json.dumps(final_chunk)}\n\n"
                    yield "data: [DONE]\n\n"
                finally:
                    await token_gen.aclose()

            return StreamingResponse(
                event_generator(),
                media_type="text/event-stream",
                headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
            )

        # Non-streaming
        accumulated_text = ""
        output_tokens = 0
        finish_reason = "length"

        try:
            async for tok in token_gen:
                try:
                    if await asyncio.wait_for(raw_request.is_disconnected(), timeout=0.0001):
                        break
                except (asyncio.TimeoutError, TimeoutError):
                    pass

                output_tokens += 1
                tok_str = decode_token(tok)
                accumulated_text += tok_str

                hit_stop = False
                for stop_seq in stop_list:
                    if stop_seq and stop_seq in accumulated_text:
                        hit_stop = True
                        finish_reason = "stop"
                        accumulated_text = accumulated_text[:accumulated_text.find(stop_seq)]
                        break
                if hit_stop:
                    break
        finally:
            await token_gen.aclose()

        return {
            "id": req_id,
            "object": "text_completion",
            "created": created_ts,
            "model": req.model or "default",
            "choices": [
                {
                    "text": accumulated_text,
                    "index": 0,
                    "logprobs": None,
                    "finish_reason": finish_reason,
                }
            ],
            "usage": {
                "prompt_tokens": prompt_len,
                "completion_tokens": output_tokens,
                "total_tokens": prompt_len + output_tokens,
            },
        }

    return app


app = create_naive_app()
