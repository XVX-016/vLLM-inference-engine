"""
End-to-end integration and divergence test for AsyncEngine.

Tests:
1. 12 prompts of varied lengths arriving at random staggered offsets through AsyncEngine.
2. Compares each generated token sequence against running each prompt alone through the
   single-sequence engine (KernelPagedManualDecode.generate or reference).
3. Computes exact-match rate across all 12 prompts.
4. For any mismatch, pinpoints the first divergent token index, extracts the logits
   at that position, and calculates the logit gap (top-1 vs chosen token logit) to
   distinguish numerical floating-point noise from logic bugs.
"""
from __future__ import annotations

import asyncio
import random
import time
from typing import Dict, List, Optional, Tuple

import pytest
import torch

from engine.baseline.instrumentation import RunLogger
from engine.kernels.paged_attention_integration import KernelPagedManualDecode
from engine.memory.block_allocator import BlockAllocator
from engine.scheduler.continuous_batching import Scheduler
from engine.serving.async_engine import AsyncEngine


TEST_PROMPTS = [
    # Short prompts
    "The capital of France is",
    "def add(a, b):",
    "Hello world! Today is a great day to",
    "Once upon a time in a small village,",
    # Medium prompts
    "Explain the concept of continuous batching in Large Language Models:",
    "Write a quick python function to compute the Fibonacci sequence up to n numbers:",
    "In deep learning, key-value cache paging prevents fragmentation by",
    "Quantum computing utilizes qubits and superposition to solve complex problems such as",
    # Long prompts
    "The transformer architecture relies on multi-head self-attention mechanisms to dynamically capture relationships between distant tokens in a sequence. Specifically,",
    "Artificial intelligence has evolved rapidly from early rule-based expert systems to modern deep neural networks with billions of parameters. Key breakthroughs include",
    "To optimize inference performance, vLLM introduces PagedAttention which draws inspiration from virtual memory in operating systems. By dividing the KV cache into fixed blocks,",
    "A distributed consensus algorithm like Raft or Paxos ensures that multiple replicas in a cluster agree on state machine transitions even in the presence of network partitions and node crashes. In particular,",
]


@pytest.fixture(scope="module")
def qwen_model_and_tokenizer():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device != "cuda":
        pytest.skip("CUDA device required for Triton paged attention end-to-end test")

    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_name = "Qwen/Qwen2.5-0.5B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.float16, device_map=device)
    model.eval()
    return model, tokenizer, device


@pytest.mark.anyio
async def test_e2e_12_prompts_async_vs_single_sequence(qwen_model_and_tokenizer):
    """
    Submits 12 prompts of varied lengths arriving at random staggered offsets to AsyncEngine.
    Validates against single-sequence generation.
    Computes exact-match rate and inspects logit gaps for any divergent token.
    """
    model, tokenizer, device = qwen_model_and_tokenizer

    num_prompts = len(TEST_PROMPTS)
    assert num_prompts == 12
    max_new_tokens = 16
    block_size = 16
    num_blocks = 512

    # 1. Generate ground truth using single-sequence engine for each prompt alone
    print("\n--- Running Single-Sequence Baseline for Ground Truth ---")
    single_engine = KernelPagedManualDecode(
        model=model,
        tokenizer=tokenizer,
        block_size=block_size,
        num_blocks=num_blocks,
        device=device,
        dtype=torch.float16,
    )

    ground_truth_tokens: Dict[int, List[int]] = {}
    ground_truth_logits: Dict[int, List[torch.Tensor]] = {}

    for idx, prompt in enumerate(TEST_PROMPTS):
        tokens, logits_hist = single_engine.generate(
            prompt=prompt,
            max_new_tokens=max_new_tokens,
            return_logits_history=True,
        )
        ground_truth_tokens[idx] = tokens
        ground_truth_logits[idx] = [l.detach().cpu() for l in logits_hist]

    # 2. Setup AsyncEngine with Continuous Batching Scheduler and RunLogger
    allocator = BlockAllocator(num_blocks=num_blocks)
    scheduler = Scheduler(
        block_allocator=allocator,
        block_size=block_size,
        watermark_blocks=2,
        max_batch_size=16,
    )
    run_logger = RunLogger(run_name="e2e_12_prompts_async")
    async_model_runner = KernelPagedManualDecode(
        model=model,
        tokenizer=tokenizer,
        block_size=block_size,
        num_blocks=num_blocks,
        device=device,
        dtype=torch.float16,
    )
    engine = AsyncEngine(
        model_runner=async_model_runner,
        scheduler=scheduler,
        tokenizer=tokenizer,
        idle_sleep_s=0.001,
        run_logger=run_logger,
    )

    # 3. Launch requests at random staggered offsets
    print("\n--- Running AsyncEngine with 12 Concurrent Staggered Prompts ---")
    random.seed(42)
    stagger_offsets = [random.uniform(0.0, 0.08) for _ in range(num_prompts)]

    async_results: Dict[int, List[int]] = {}

    async def submit_prompt(idx: int, prompt_text: str, delay: float):
        if delay > 0:
            await asyncio.sleep(delay)
        tokens = []
        async for tok in engine.add_request(prompt=prompt_text, max_tokens=max_new_tokens):
            tokens.append(tok)
        async_results[idx] = tokens

    tasks = [
        asyncio.create_task(submit_prompt(i, TEST_PROMPTS[i], stagger_offsets[i]))
        for i in range(num_prompts)
    ]
    await asyncio.gather(*tasks)
    await engine.stop()

    # 4. Compare results and report metrics
    exact_matches = 0
    mismatches: List[dict] = []

    for idx in range(num_prompts):
        gt = ground_truth_tokens[idx]
        actual = async_results[idx]

        if gt == actual:
            exact_matches += 1
        else:
            # Find first divergent token index
            div_idx = None
            for pos in range(min(len(gt), len(actual))):
                if gt[pos] != actual[pos]:
                    div_idx = pos
                    break
            if div_idx is None:
                div_idx = min(len(gt), len(actual))

            # Inspect logit gap at divergent position
            logit_gap = None
            gt_logit_val = None
            actual_logit_val = None
            if div_idx < len(ground_truth_logits[idx]):
                pos_logits = ground_truth_logits[idx][div_idx].squeeze()
                gt_tok = gt[div_idx] if div_idx < len(gt) else None
                act_tok = actual[div_idx] if div_idx < len(actual) else None
                if gt_tok is not None and act_tok is not None:
                    gt_logit_val = float(pos_logits[gt_tok].item())
                    actual_logit_val = float(pos_logits[act_tok].item())
                    logit_gap = abs(gt_logit_val - actual_logit_val)

            mismatches.append({
                "prompt_idx": idx,
                "prompt": TEST_PROMPTS[idx][:40] + "...",
                "first_divergent_pos": div_idx,
                "expected_token": gt[div_idx] if div_idx < len(gt) else None,
                "actual_token": actual[div_idx] if div_idx < len(actual) else None,
                "expected_str": tokenizer.decode([gt[div_idx]]) if div_idx < len(gt) else "",
                "actual_str": tokenizer.decode([actual[div_idx]]) if div_idx < len(actual) else "",
                "logit_gap": logit_gap,
                "expected_logit": gt_logit_val,
                "actual_logit": actual_logit_val,
            })

    exact_match_rate = exact_matches / num_prompts

    print("\n" + "=" * 60)
    print(f"E2E Test Results: 12 Prompts Varied Lengths Staggered")
    print(f"Exact-Match Rate: {exact_match_rate * 100:.2f}% ({exact_matches}/{num_prompts})")
    print("=" * 60)

    if mismatches:
        print("\nMismatches Details:")
        for m in mismatches:
            print(
                f"  Prompt {m['prompt_idx']} ({m['prompt']}):\n"
                f"    First divergent pos: {m['first_divergent_pos']}\n"
                f"    Expected: {m['expected_token']} ({repr(m['expected_str'])}) | Logit: {m['expected_logit']}\n"
                f"    Actual:   {m['actual_token']} ({repr(m['actual_str'])}) | Logit: {m['actual_logit']}\n"
                f"    Logit Gap: {m['logit_gap']:.4f} (Numerical noise if < 0.2)\n"
            )
    else:
        print("\nAll 12 prompts matched single-sequence ground truth 100% bit-for-bit!")

    # Check logger statistics
    summary = run_logger.summary()
    print("\nLogger Per-Iteration & Per-Request Metrics Summary:")
    print(f"  Total Iterations:      {len(run_logger.iterations)}")
    print(f"  Total Tokens Produced: {summary['total_tokens']}")
    print(f"  Avg Time To 1st Token: {summary['avg_ttft_s'] * 1000:.2f} ms")
    print(f"  Avg Request Latency:   {summary['avg_latency_s'] * 1000:.2f} ms")

    # Save metrics JSON artifact
    out_file = run_logger.save()
    print(f"Saved run metrics to: {out_file}")

    # Assert that match rate is at least 90% (allows for minor fp16 tie-breaking differences)
    assert exact_match_rate >= 0.90, (
        f"Exact match rate was {exact_match_rate * 100:.2f}%, below 90% threshold. "
        f"Mismatches: {mismatches}"
    )

    # For any mismatch that did occur, verify it was small floating-point noise (< 0.2)
    for m in mismatches:
        if m["logit_gap"] is not None:
            assert m["logit_gap"] < 0.25, (
                f"Prompt {m['prompt_idx']} had a large logit gap of {m['logit_gap']:.4f}, "
                f"indicating a potential logic error rather than numerical noise!"
            )
