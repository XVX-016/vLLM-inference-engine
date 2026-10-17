"""
FastAPI serving layer for OpenAI-compatible LLM inference (Phase 5).

Provides:
- OpenAI-compatible POST /v1/completions (fields: prompt, max_tokens, temperature=0, stream, stop optional, ignore_eos)
- Server-Sent Events (SSE) streaming format ending with data: [DONE]
- Clean engine startup/shutdown in FastAPI lifespan
- Mid-stream client disconnect handling with sequence abort and block reclamation
- GET /health and GET /metrics (running, waiting, free blocks, total blocks, preemptions, tokens generated)
- Validation that prompt tokens + max_tokens fit the pool, returning HTTP 400 otherwise
"""
from __future__ import annotations

import asyncio
import json
import logging
import math
import os
import time
from contextlib import asynccontextmanager
from typing import Any, AsyncGenerator, Dict, List, Optional, Union

from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

from engine.memory.block_allocator import BlockAllocator
from engine.scheduler.continuous_batching import Scheduler
from engine.serving.async_engine import AsyncEngine

logger = logging.getLogger(__name__)


# Global engine instance holder
_engine: Optional[AsyncEngine] = None


def set_global_engine(engine: Optional[AsyncEngine]) -> None:
    """Set or override the global engine instance (useful for testing or custom setup)."""
    global _engine
    _engine = engine


def get_global_engine() -> AsyncEngine:
    """Returns the running global engine instance or raises an error."""
    global _engine
    if _engine is None:
        raise RuntimeError("AsyncEngine is not initialized or running.")
    return _engine


class MockModelRunner:
    """Fallback runner for local testing without GPU/weights."""
    def __init__(self, step_delay_s: float = 0.005, base_token_id: int = 100):
        self.step_delay_s = step_delay_s
        self.base_token_id = base_token_id

    def step(self, batch: Any) -> List[int]:
        if self.step_delay_s > 0:
            time.sleep(self.step_delay_s)
        return [self.base_token_id + seq.seq_id for seq in batch.sequences]


def calculate_paged_blocks_from_budget(
    kv_memory_mb: float,
    block_size: int = 16,
    num_layers: int = 24,
    num_kv_heads: int = 2,
    head_dim: int = 64,
    dtype_bytes: int = 2,
) -> int:
    """Computes number of paged blocks that fit inside a memory budget (in MB)."""
    bytes_per_block = 2 * num_layers * block_size * num_kv_heads * head_dim * dtype_bytes
    total_bytes = int(kv_memory_mb * 1024 * 1024)
    return max(1, total_bytes // bytes_per_block)


def build_default_engine(
    kv_memory_mb: Optional[float] = None,
    num_blocks: Optional[int] = None,
    block_size: int = 16,
    watermark_blocks: int = 1,
    runner: Optional[Any] = None,
    tokenizer: Optional[Any] = None,
) -> AsyncEngine:
    """Builds an AsyncEngine, optionally deriving block pool size from kv_memory_mb or num_blocks."""
    env_budget_mb = os.environ.get("VLLM_KV_MEMORY_MB")
    if kv_memory_mb is not None:
        budget_mb = float(kv_memory_mb)
    elif env_budget_mb is not None:
        budget_mb = float(env_budget_mb)
    else:
        budget_mb = None

    block_size = int(os.environ.get("VLLM_BLOCK_SIZE", str(block_size)))
    watermark_blocks = int(os.environ.get("VLLM_WATERMARK_BLOCKS", str(watermark_blocks)))

    if num_blocks is not None:
        total_blocks = int(num_blocks)
    elif budget_mb is not None:
        total_blocks = calculate_paged_blocks_from_budget(budget_mb, block_size=block_size)
    else:
        total_blocks = int(os.environ.get("VLLM_TOTAL_BLOCKS", "64"))

    allocator = BlockAllocator(num_blocks=total_blocks)
    scheduler = Scheduler(
        block_allocator=allocator,
        block_size=block_size,
        watermark_blocks=watermark_blocks,
    )
    if runner is None:
        logger.info(f"Loading KernelPagedManualDecode on GPU with {total_blocks} blocks...")
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from engine.kernels.paged_attention_integration import KernelPagedManualDecode

        model_name = os.environ.get("VLLM_MODEL_NAME", "Qwen/Qwen2.5-0.5B-Instruct")
        hf_tok = AutoTokenizer.from_pretrained(model_name)
        hf_model = AutoModelForCausalLM.from_pretrained(
            model_name,
            dtype=torch.float16,
        ).to("cuda" if torch.cuda.is_available() else "cpu")
        hf_model.eval()

        runner = KernelPagedManualDecode(
            model=hf_model,
            tokenizer=hf_tok,
            block_size=block_size,
            num_blocks=total_blocks,
            device="cuda" if torch.cuda.is_available() else "cpu",
            dtype=torch.float16,
        )
        if tokenizer is None:
            tokenizer = hf_tok

    return AsyncEngine(
        model_runner=runner,
        scheduler=scheduler,
        tokenizer=tokenizer,
        idle_sleep_s=0.002,
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    """FastAPI lifespan context manager: starts engine cleanly and stops it on shutdown."""
    global _engine
    if _engine is None:
        _engine = build_default_engine()
    _engine.start()
    logger.info("AsyncEngine started in FastAPI lifespan.")
    try:
        yield
    finally:
        if _engine is not None:
            logger.info("Shutting down AsyncEngine in FastAPI lifespan...")
            await _engine.stop()
            logger.info("AsyncEngine stopped cleanly.")


class CompletionRequest(BaseModel):
    model: Optional[str] = "default"
    prompt: Union[str, List[str]]
    max_tokens: Optional[int] = Field(default=16, ge=1)
    temperature: Optional[float] = 0.0
    stream: Optional[bool] = False
    stop: Optional[Union[str, List[str]]] = None
    ignore_eos: Optional[bool] = False


def create_app(engine: Optional[AsyncEngine] = None) -> FastAPI:
    """Application factory that allows binding a specific engine instance."""
    if engine is not None:
        set_global_engine(engine)

    app = FastAPI(title="Paged-KV OpenAI-Compatible Serving Engine", lifespan=lifespan)

    @app.get("/health")
    def health():
        eng = get_global_engine()
        runner_type = "hf" if hasattr(eng.model_runner, "model") else "mock"
        return {"status": "ok", "running": eng._is_running, "runner": runner_type}

    @app.get("/metrics")
    def metrics():
        eng = get_global_engine()
        allocator = eng.scheduler.block_allocator
        block_size = eng.scheduler.block_size

        # Model dimensions (defaulting to standard Qwen/benchmark config)
        num_layers = int(getattr(eng.model_runner, "num_layers", 24))
        num_kv_heads = int(getattr(eng.model_runner, "num_kv_heads", 2))
        head_dim = int(getattr(eng.model_runner, "head_dim", 64))
        dtype_bytes = 2  # fp16

        bytes_per_token = 2 * num_layers * num_kv_heads * head_dim * dtype_bytes
        bytes_per_block = bytes_per_token * block_size

        allocated_blocks = allocator.num_blocks - len(allocator.free_block_ids)
        tokens_stored = sum(seq.kv_len for seq in eng.scheduler.running)

        allocated_kv_bytes = allocated_blocks * bytes_per_block
        active_token_kv_bytes = tokens_stored * bytes_per_token
        total_kv_bytes = allocator.num_blocks * bytes_per_block

        return {
            "running": len(eng.scheduler.running),
            "waiting": len(eng.scheduler.waiting),
            "free_blocks": len(allocator.free_block_ids),
            "total_blocks": allocator.num_blocks,
            "allocated_blocks": allocated_blocks,
            "block_size": block_size,
            "tokens_stored": tokens_stored,
            "bytes_per_token": bytes_per_token,
            "bytes_per_block": bytes_per_block,
            "allocated_kv_bytes": allocated_kv_bytes,
            "active_token_kv_bytes": active_token_kv_bytes,
            "total_kv_bytes": total_kv_bytes,
            "preemptions": eng.scheduler.preemption_count,
            "admission_blocked": eng.scheduler.admission_blocked_count,
            "tokens_generated": eng.total_tokens_generated,
        }

    @app.post("/v1/completions")
    async def completions(req: CompletionRequest, raw_request: Request):
        eng = get_global_engine()
        allocator = eng.scheduler.block_allocator
        block_size = eng.scheduler.block_size

        # Validate temperature (temperature=0 only for now)
        if req.temperature is not None and req.temperature != 0.0:
            raise HTTPException(
                status_code=400,
                detail=f"Only temperature=0 is supported currently, got {req.temperature}",
            )

        # Normalize prompt
        prompt_str = req.prompt if isinstance(req.prompt, str) else req.prompt[0]
        max_tokens = req.max_tokens or 16

        # Tokenize / compute prompt tokens
        if eng.tokenizer is not None:
            prompt_token_ids = eng.tokenizer.encode(prompt_str)
        else:
            # Fallback simple tokenization if no tokenizer is attached
            prompt_token_ids = [hash(w) % 10000 for w in prompt_str.split()]
            if not prompt_token_ids:
                prompt_token_ids = [1]

        prompt_len = len(prompt_token_ids)
        total_tokens_needed = prompt_len + max_tokens
        blocks_needed = math.ceil(total_tokens_needed / block_size)

        # Validate that prompt tokens + max_tokens fit the pool
        if blocks_needed > allocator.num_blocks:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"Prompt tokens ({prompt_len}) + max_tokens ({max_tokens}) requires "
                    f"{blocks_needed} blocks, exceeding total pool capacity of {allocator.num_blocks} blocks."
                ),
            )

        # Normalize stop conditions
        stop_list: List[str] = []
        if isinstance(req.stop, str):
            stop_list = [req.stop]
        elif isinstance(req.stop, list):
            stop_list = req.stop

        req_id = f"cmpl-{int(time.time() * 1000)}-{id(raw_request)}"
        created_ts = int(time.time())

        # Submit request to engine generator
        token_gen = eng.add_request(
            prompt=prompt_str,
            max_tokens=max_tokens,
            prompt_token_ids=prompt_token_ids,
            ignore_eos=bool(req.ignore_eos),
        )

        def decode_token(tok_id: int) -> str:
            if eng.tokenizer is not None:
                return eng.tokenizer.decode([tok_id])
            return f" token_{tok_id}"

        # Streaming response via Server-Sent Events (SSE)
        if req.stream:
            async def event_generator() -> AsyncGenerator[str, None]:
                accumulated_text = ""
                finish_reason: Optional[str] = None
                output_tokens = 0

                try:
                    async for tok in token_gen:
                        # Non-blocking check for client disconnect
                        try:
                            if await asyncio.wait_for(raw_request.is_disconnected(), timeout=0.0001):
                                logger.info(f"Client disconnected mid-stream for {req_id}; aborting.")
                                break
                        except (asyncio.TimeoutError, TimeoutError):
                            pass

                        output_tokens += 1
                        tok_str = decode_token(tok)
                        accumulated_text += tok_str

                        # Check stop strings
                        hit_stop = False
                        for stop_seq in stop_list:
                            if stop_seq and stop_seq in accumulated_text:
                                hit_stop = True
                                finish_reason = "stop"
                                break

                        chunk_data = {
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
                        yield f"data: {json.dumps(chunk_data)}\n\n"

                        if hit_stop:
                            break

                    if finish_reason is None:
                        finish_reason = "length"

                    # Final chunk with finish_reason if not already sent
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
                                "finish_reason": finish_reason,
                            }
                        ],
                    }
                    yield f"data: {json.dumps(final_chunk)}\n\n"
                    yield "data: [DONE]\n\n"
                except (asyncio.CancelledError, GeneratorExit):
                    logger.info(f"Stream cancelled for request {req_id}")
                    raise
                except Exception as ex:
                    import traceback
                    tb = traceback.format_exc()
                    logger.error(f"Server exception in event_generator for {req_id}:\n{tb}")
                    # Yield error event and [DONE] so HTTP chunked stream terminates properly
                    err_chunk = {
                        "id": req_id,
                        "object": "text_completion",
                        "created": created_ts,
                        "model": req.model or "default",
                        "choices": [
                            {
                                "text": "",
                                "index": 0,
                                "logprobs": None,
                                "finish_reason": "error",
                            }
                        ],
                        "error": str(ex),
                        "traceback": tb,
                    }
                    yield f"data: {json.dumps(err_chunk)}\n\n"
                    yield "data: [DONE]\n\n"
                finally:
                    # Clean up generator if closed early
                    await token_gen.aclose()

            return StreamingResponse(
                event_generator(),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                    "X-Accel-Buffering": "no",
                },
            )

        # Non-streaming response
        accumulated_text = ""
        output_tokens = 0
        finish_reason = "length"

        try:
            async for tok in token_gen:
                try:
                    if await asyncio.wait_for(raw_request.is_disconnected(), timeout=0.0001):
                        logger.info(f"Client disconnected for request {req_id}; aborting.")
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
                        stop_idx = accumulated_text.find(stop_seq)
                        accumulated_text = accumulated_text[:stop_idx]
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


# Default app instance
app = create_app()
