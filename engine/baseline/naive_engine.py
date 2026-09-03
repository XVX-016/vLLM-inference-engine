"""
Phase 0 — Naive baseline engine.

This is deliberately "the wrong way to do it": every sequence gets a
statically pre-allocated KV-cache sized to `max_seq_len`, regardless of how
long the sequence actually turns out to be. This is what most from-scratch
inference loops do, and it's exactly the VRAM-fragmentation problem that
Phases 2-3 (paged allocator + paged-attention kernel) fix.

Keep this engine working and unmodified once Phase 2+ lands — it's the
control group your benchmark numbers get compared against.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from engine.baseline.instrumentation import MemoryTracker, RunLogger


@dataclass
class GenerationRequest:
    prompt: str
    max_new_tokens: int = 64
    input_ids: torch.Tensor = field(default=None, repr=False)
    generated_ids: list[int] = field(default_factory=list)
    finished: bool = False


class NaiveStaticCacheEngine:
    """
    Batched decode loop using HF's built-in DynamicCache / static cache,
    with every sequence's cache pre-sized to `max_seq_len` up front.

    This intentionally does NOT implement continuous batching: the whole
    batch runs prefill together and decodes together until every sequence
    in the batch is finished (or hits max_new_tokens). New requests cannot
    join mid-batch. That limitation is exactly what Phase 4 removes.
    """

    def __init__(self, model_name: str, max_seq_len: int = 2048, device: str | None = None):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if self.device.type == "cuda" else torch.float32,
        ).to(self.device)
        self.model.eval()
        self.max_seq_len = max_seq_len

    @torch.inference_mode()
    def run_batch(self, requests: list[GenerationRequest], run_name: str) -> RunLogger:
        mem = MemoryTracker(self.device)
        logger = RunLogger(run_name)

        # Tokenize + left-pad so every sequence starts aligned for a shared batch.
        enc = self.tokenizer(
            [r.prompt for r in requests],
            return_tensors="pt",
            padding=True,
        ).to(self.device)
        input_ids = enc["input_ids"]
        attention_mask = enc["attention_mask"]

        # --- Prefill: this is where the naive engine "wastes" memory ---
        # HF's default cache grows dynamically per call, but a *true* naive
        # engine pre-allocates for max_seq_len. We simulate that VRAM cost
        # explicitly here so the instrumentation reflects the real bottleneck
        # described in the project brief, rather than relying on HF's
        # (already somewhat smarter) DynamicCache.
        batch_size = input_ids.shape[0]
        self._simulate_static_preallocation(batch_size)

        t0 = time.perf_counter()
        out = self.model(input_ids=input_ids, attention_mask=attention_mask, use_cache=True)
        past_key_values = out.past_key_values
        step_time = time.perf_counter() - t0
        logger.log_step(
            step=0,
            num_active_seqs=batch_size,
            tokens_generated_this_step=0,
            step_time_s=step_time,
            mem=mem,
        )

        next_token_logits = out.logits[:, -1, :]
        next_tokens = torch.argmax(next_token_logits, dim=-1)
        for r, tok in zip(requests, next_tokens.tolist()):
            r.generated_ids.append(tok)

        # --- Decode loop: static batch, no continuous batching ---
        max_steps = max(r.max_new_tokens for r in requests) - 1
        for step in range(1, max_steps + 1):
            active = [not r.finished for r in requests]
            if not any(active):
                break

            t0 = time.perf_counter()
            input_ids_step = next_tokens.unsqueeze(-1)
            attention_mask = torch.cat(
                [attention_mask, torch.ones((batch_size, 1), device=self.device, dtype=attention_mask.dtype)],
                dim=1,
            )
            out = self.model(
                input_ids=input_ids_step,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                use_cache=True,
            )
            past_key_values = out.past_key_values
            step_time = time.perf_counter() - t0

            next_token_logits = out.logits[:, -1, :]
            next_tokens = torch.argmax(next_token_logits, dim=-1)

            tokens_this_step = 0
            for i, r in enumerate(requests):
                if r.finished:
                    continue
                tok = next_tokens[i].item()
                r.generated_ids.append(tok)
                tokens_this_step += 1
                if tok == self.tokenizer.eos_token_id or len(r.generated_ids) >= r.max_new_tokens:
                    r.finished = True

            logger.log_step(
                step=step,
                num_active_seqs=sum(active),
                tokens_generated_this_step=tokens_this_step,
                step_time_s=step_time,
                mem=mem,
            )

        logger.save()
        return logger

    def _simulate_static_preallocation(self, batch_size: int) -> None:
        """
        Allocates (and immediately frees) a tensor the size of a worst-case
        static KV-cache, purely so peak-VRAM instrumentation captures the
        cost a real static-allocation engine would pay. This makes the
        Phase 0 vs Phase 2+ comparison honest even though HF's own cache
        under the hood is smarter than a naive engine would be.
        """
        if self.device.type != "cuda":
            return
        cfg = self.model.config
        num_layers = getattr(cfg, "num_hidden_layers", 24)
        num_kv_heads = getattr(cfg, "num_key_value_heads", getattr(cfg, "num_attention_heads", 16))
        head_dim = getattr(cfg, "head_dim", cfg.hidden_size // getattr(cfg, "num_attention_heads", 16))
        dtype_bytes = 2  # fp16

        bytes_needed = (
            2  # K and V
            * num_layers
            * batch_size
            * num_kv_heads
            * self.max_seq_len
            * head_dim
            * dtype_bytes
        )
        try:
            scratch = torch.empty(bytes_needed // dtype_bytes, dtype=torch.float16, device=self.device)
            del scratch
        except torch.cuda.OutOfMemoryError:
            print(
                f"[naive_engine] WARNING: static pre-allocation of "
                f"{bytes_needed / 1e9:.2f} GB for max_seq_len={self.max_seq_len} "
                f"OOM'd — this is the exact fragmentation problem Phase 2 fixes."
            )
        torch.cuda.empty_cache()
