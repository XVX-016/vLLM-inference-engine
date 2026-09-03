"""
Phase 0 — Naive baseline engine with true static pre-allocated KV Cache.

This engine serves as the fundamental control-group baseline for demonstrating
why vLLM / PagedAttention was invented:
1. Static pre-allocation: Allocates [batch_size, num_kv_heads, max_seq_len, head_dim]
   per layer upfront via early_initialization, causing massive internal memory
   fragmentation for short requests.
2. Naive batching: All requests run together; sequences decode in lockstep.
3. Unified Instrumentation: Integrates with MemoryTracker and RunLogger so metrics
   directly plug into Phase 5 benchmarks and JSON output.
"""
from __future__ import annotations

import argparse
import time
from dataclasses import dataclass, field
from typing import List, Optional

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from transformers.cache_utils import StaticCache

from engine.baseline.instrumentation import MemoryTracker, RunLogger


@dataclass
class GenerationRequest:
    prompt: str
    max_new_tokens: int = 64
    input_ids: Optional[torch.Tensor] = field(default=None, repr=False)
    generated_ids: list[int] = field(default_factory=list)
    finished: bool = False


class NaiveStaticCacheEngine:
    """
    Batched decode engine using real pre-allocated StaticCache.
    """

    def __init__(
        self,
        model_name: str,
        max_seq_len: int = 2048,
        device: Optional[str] = None,
        torch_dtype: Optional[torch.dtype] = None,
    ):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.torch_dtype = torch_dtype or (torch.float16 if self.device.type == "cuda" else torch.float32)
        self.max_seq_len = max_seq_len

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            dtype=self.torch_dtype,
        ).to(self.device)
        self.model.eval()

        self.config = AutoConfig.from_pretrained(model_name)
        self.num_kv_heads = getattr(self.config, "num_key_value_heads", getattr(self.config, "num_attention_heads", 16))
        self.head_dim = getattr(self.config, "head_dim", self.config.hidden_size // getattr(self.config, "num_attention_heads", 16))

    @torch.inference_mode()
    def run_batch(self, requests: list[GenerationRequest], run_name: str = "baseline_naive") -> RunLogger:
        batch_size = len(requests)
        if batch_size == 0:
            raise ValueError("Cannot run empty batch.")

        mem = MemoryTracker(self.device)
        logger = RunLogger(run_name)

        # 1. Tokenize inputs with left padding
        encodings = self.tokenizer(
            [r.prompt for r in requests],
            return_tensors="pt",
            padding=True,
        ).to(self.device)
        input_ids = encodings.input_ids
        attention_mask = encodings.attention_mask
        prompt_len = input_ids.shape[1]

        # 2. Pre-allocate Static KV Cache for max_seq_len upfront
        static_kv_cache = StaticCache(
            config=self.config,
            max_batch_size=batch_size,
            max_cache_len=self.max_seq_len,
            device=self.device,
            dtype=self.torch_dtype,
        )
        static_kv_cache.early_initialization(
            batch_size=batch_size,
            num_heads=self.num_kv_heads,
            head_dim=self.head_dim,
            dtype=self.torch_dtype,
            device=self.device,
        )

        # 3. Prefill Step
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        t0_prefill = time.perf_counter()

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=static_kv_cache,
            use_cache=True,
        )

        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        prefill_time = time.perf_counter() - t0_prefill

        logger.log_step(
            step=0,
            num_active_seqs=batch_size,
            tokens_generated_this_step=batch_size,
            step_time_s=prefill_time,
            mem=mem,
        )

        next_token_logits = outputs.logits[:, -1, :]
        next_tokens = torch.argmax(next_token_logits, dim=-1, keepdim=True)
        for i, r in enumerate(requests):
            tok = next_tokens[i].item()
            r.generated_ids.append(tok)
            if tok == self.tokenizer.eos_token_id or len(r.generated_ids) >= r.max_new_tokens:
                r.finished = True

        # Attention mask expands for autoregressive steps
        cur_attention_mask = torch.cat(
            [attention_mask, torch.ones((batch_size, 1), dtype=attention_mask.dtype, device=self.device)],
            dim=-1,
        )
        cur_token = next_tokens

        # 4. Decode Loop
        max_steps = max(r.max_new_tokens for r in requests) - 1
        for step in range(1, max_steps + 1):
            active = [not r.finished for r in requests]
            if not any(active):
                break

            if self.device.type == "cuda":
                torch.cuda.synchronize(self.device)
            t0_step = time.perf_counter()

            outputs = self.model(
                input_ids=cur_token,
                attention_mask=cur_attention_mask,
                past_key_values=static_kv_cache,
                use_cache=True,
            )

            if self.device.type == "cuda":
                torch.cuda.synchronize(self.device)
            step_duration = time.perf_counter() - t0_step

            next_token_logits = outputs.logits[:, -1, :]
            cur_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)

            tokens_this_step = 0
            for i, r in enumerate(requests):
                if r.finished:
                    continue
                tok = cur_token[i].item()
                r.generated_ids.append(tok)
                tokens_this_step += 1
                if tok == self.tokenizer.eos_token_id or len(r.generated_ids) >= r.max_new_tokens:
                    r.finished = True

            cur_attention_mask = torch.cat(
                [cur_attention_mask, torch.ones((batch_size, 1), dtype=cur_attention_mask.dtype, device=self.device)],
                dim=-1,
            )

            logger.log_step(
                step=step,
                num_active_seqs=sum(active),
                tokens_generated_this_step=tokens_this_step,
                step_time_s=step_duration,
                mem=mem,
            )

        logger.save()
        return logger


def run_baseline(
    model_name: str,
    prompts: List[str],
    max_new_tokens: int = 64,
    max_seq_len: int = 512,
    device_str: Optional[str] = None,
    run_name: str = "baseline_run",
) -> dict:
    """
    Convenience function that sets up requests, executes NaiveStaticCacheEngine,
    and returns a summary dictionary with metrics and decoded outputs.
    """
    engine = NaiveStaticCacheEngine(
        model_name=model_name,
        max_seq_len=max_seq_len,
        device=device_str,
    )
    requests = [GenerationRequest(prompt=p, max_new_tokens=max_new_tokens) for p in prompts]
    logger = engine.run_batch(requests, run_name=run_name)
    summary = logger.summary()

    decoded_texts = [
        engine.tokenizer.decode(r.generated_ids, skip_special_tokens=True)
        for r in requests
    ]
    summary["decoded_texts"] = decoded_texts
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Naive LLM Baseline with Static KV Cache")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-0.5B-Instruct", help="Hugging Face model ID")
    parser.add_argument("--max-seq-len", type=int, default=512, help="Max sequence length for static KV pre-allocation")
    parser.add_argument("--max-new-tokens", type=int, default=32, help="Tokens to generate")
    args = parser.parse_args()

    sample_prompts = [
        "Explain virtual memory paging in operating systems:",
        "Write a Python function to perform binary search:",
        "What is the capital of France and why is it famous?",
        "How does key-value caching accelerate transformer inference?",
    ]

    res = run_baseline(
        model_name=args.model,
        prompts=sample_prompts,
        max_new_tokens=args.max_new_tokens,
        max_seq_len=args.max_seq_len,
    )
    import json
    print("\n--- Summary ---")
    print(json.dumps({k: v for k, v in res.items() if k != "decoded_texts"}, indent=2))
    print("\n--- Generated Samples ---")
    for i, t in enumerate(res["decoded_texts"]):
        print(f"[{i+1}] {repr(t[:120])}...")
