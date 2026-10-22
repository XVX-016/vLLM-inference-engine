"""
Real-GPU Integration Tests for Qwen2.5-0.5B-Instruct.

Contains three tests:
(1) test_real_gpu_four_prompts_concurrent_decode:
    Four prompts of different lengths decoded together for 48 tokens,
    each compared to its own single-sequence run, reporting first divergent step
    and the reference's top-1/top-2 logit margin there.
(2) test_real_gpu_preemption_stress_tiny_pool:
    A deliberately small pool (e.g. 80 blocks) with 10 concurrent requests
    so preemption fires: assert preemptions > 0, exact token counts,
    free blocks == num_blocks at the end, and compare outputs to an unconstrained run.
(3) test_real_gpu_12_prompts_staggered_divergence:
    The 12-prompt staggered-arrival e2e test, with top-1/top-2 margin at the
    first divergent step and a teacher-forced single-step forward check on any mismatched prompt.
"""
from __future__ import annotations

import asyncio
import math
import random
from typing import Dict, List, Optional, Tuple

import pytest
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from engine.baseline.instrumentation import RunLogger
from engine.kernels.paged_attention_integration import KernelPagedManualDecode
from engine.memory.block_allocator import BlockAllocator
from engine.scheduler.continuous_batching import Scheduler
from engine.serving.async_engine import AsyncEngine


@pytest.fixture(scope="module")
def qwen_model_and_tokenizer():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device != "cuda":
        pytest.skip("CUDA device required for Triton paged attention real-GPU tests")

    model_name = "Qwen/Qwen2.5-0.5B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.float16, device_map=device)
    model.eval()
    return model, tokenizer, device


def test_real_gpu_four_prompts_concurrent_decode(qwen_model_and_tokenizer):
    """
    (1) Four prompts of different lengths decoded together for 48 tokens,
    each compared to its own single-sequence run, reporting first divergent step
    and the reference's top-1/top-2 logit margin there.
    """
    model, tokenizer, device = qwen_model_and_tokenizer
    block_size = 16
    num_blocks = 512
    max_new_tokens = 48

    prompts = [
        "Hi,",
        "What is the capital of France?",
        "Continuous batching improves GPU utilization by dynamically merging requests.",
        "The quick brown fox jumps over the lazy dog in a vibrant sunny afternoon while birds sing melodiously across the green meadows.",
    ]

    # 1. Single sequence reference generation
    single_engine = KernelPagedManualDecode(
        model=model,
        tokenizer=tokenizer,
        block_size=block_size,
        num_blocks=num_blocks,
        device=device,
        dtype=torch.float16,
    )

    reference_tokens: Dict[int, List[int]] = {}
    reference_logits: Dict[int, List[torch.Tensor]] = {}

    for i, p in enumerate(prompts):
        tokens, logits_hist = single_engine.generate(
            prompt=p,
            max_new_tokens=max_new_tokens,
            return_logits_history=True,
        )
        reference_tokens[i] = tokens
        reference_logits[i] = [l.detach().cpu() for l in logits_hist]

    # 2. Batch decoded run using Scheduler with debug_invariants=True
    allocator = BlockAllocator(num_blocks=num_blocks)
    scheduler = Scheduler(
        block_allocator=allocator,
        block_size=block_size,
        watermark_blocks=2,
        max_batch_size=16,
        debug_invariants=True,
    )
    runner = KernelPagedManualDecode(
        model=model,
        tokenizer=tokenizer,
        block_size=block_size,
        num_blocks=num_blocks,
        device=device,
        dtype=torch.float16,
    )

    seq_objs = []
    for i, p in enumerate(prompts):
        inp = tokenizer(p, return_tensors="pt")
        prompt_ids = inp.input_ids[0].tolist()
        s = scheduler.add_request(prompt=p, max_new_tokens=max_new_tokens, prompt_token_ids=prompt_ids)
        seq_objs.append(s)

    # Step through scheduler until all complete
    steps = 0
    max_steps = 500
    while scheduler.has_unfinished_requests() and steps < max_steps:
        batch = scheduler.step()
        if batch is None:
            break
        if batch.batch_type.value == "prefill":
            token = runner.prefill(batch.sequences[0], allocator=allocator)
            scheduler.post_step(batch, [token])
        else:
            tokens = runner.decode_step(batch.sequences, allocator=allocator)
            scheduler.post_step(batch, tokens)
        steps += 1

    # 3. Compare outputs
    divergence_info = []
    exact_matches = 0
    for i, seq in enumerate(seq_objs):
        ref = reference_tokens[i]
        actual = seq.generated_ids
        if ref == actual:
            exact_matches += 1
            divergence_info.append({
                "prompt_idx": i,
                "status": "EXACT_MATCH",
                "divergent_step": None,
                "top1_top2_margin": None,
            })
        else:
            # Find first divergent step
            div_step = None
            for step in range(min(len(ref), len(actual))):
                if ref[step] != actual[step]:
                    div_step = step
                    break
            if div_step is None:
                div_step = min(len(ref), len(actual))

            # Compute reference top-1 / top-2 margin at divergent step
            margin = None
            if div_step < len(reference_logits[i]):
                logits_at_step = reference_logits[i][div_step].squeeze()
                top2_vals, _ = torch.topk(logits_at_step, k=2)
                margin = float((top2_vals[0] - top2_vals[1]).item())

            divergence_info.append({
                "prompt_idx": i,
                "status": "DIVERGED",
                "divergent_step": div_step,
                "top1_top2_margin": margin,
                "ref_token": ref[div_step] if div_step < len(ref) else None,
                "actual_token": actual[div_step] if div_step < len(actual) else None,
                "ref_str": tokenizer.decode([ref[div_step]]) if div_step < len(ref) else "",
                "actual_str": tokenizer.decode([actual[div_step]]) if div_step < len(actual) else "",
            })

    print("\n--- Test (1) 4 Prompts Concurrent Decode Summary ---")
    for d in divergence_info:
        print(d)

    # Invariants and block cleanup
    assert allocator.get_free_block_count() == num_blocks, "Blocks leaked after all 4 requests finished!"
    assert exact_matches >= 3, f"Expected at least 3/4 exact matches (got {exact_matches}). Info: {divergence_info}"
    for d in divergence_info:
        if d["status"] == "DIVERGED" and d["top1_top2_margin"] is not None:
            # Margin should be small if divergence is purely fp16 tie-breaking / numerical noise
            assert d["top1_top2_margin"] < 0.5, (
                f"Large top-1/top-2 margin ({d['top1_top2_margin']}) at divergent step {d['divergent_step']}!"
            )


def test_real_gpu_six_prompts_prefix_cache_equivalence(qwen_model_and_tokenizer):
    model, tokenizer, device = qwen_model_and_tokenizer
    block_size = 16
    num_blocks = 128
    max_new_tokens = 8

    shared_prefix_texts = (
        "A short introduction to astronomy, the planets, stars, and galaxies in our solar system, "
        "with an overview of the night sky.",
        "An overview of machine learning, neural networks, training data, and how modern computers "
        "learn useful patterns from examples.",
    )
    suffix_texts = (
        "First distinct continuation.",
        "Second distinct continuation.",
        "Third distinct continuation.",
    )
    prompts: list[list[int]] = []
    for prefix_text in shared_prefix_texts:
        prefix_ids = tokenizer(prefix_text, return_tensors="pt").input_ids[0].tolist()
        assert len(prefix_ids) >= block_size
        prefix_ids = prefix_ids[:block_size]
        for suffix_text in suffix_texts:
            suffix_ids = tokenizer(suffix_text, add_special_tokens=False).input_ids
            prompts.append(prefix_ids + suffix_ids)

    def run_mode(prefix_mode: str):
        allocator = BlockAllocator(
            num_blocks=num_blocks,
            enable_prefix_cache=prefix_mode != "off",
        )
        scheduler = Scheduler(
            block_allocator=allocator,
            block_size=block_size,
            watermark_blocks=1,
            max_batch_size=8,
            debug_invariants=True,
            prefix_mode=prefix_mode,
        )
        runner = KernelPagedManualDecode(
            model=model,
            tokenizer=tokenizer,
            block_size=block_size,
            num_blocks=num_blocks,
            device=device,
            dtype=torch.float16,
            debug_prefix_cache=prefix_mode == "share",
            prefix_mode=prefix_mode,
        )
        sequences = [
            scheduler.add_request(
                prompt=f"prefix-{idx}",
                max_new_tokens=max_new_tokens,
                prompt_token_ids=prompt_ids,
            )
            for idx, prompt_ids in enumerate(prompts)
        ]
        logits_history: dict[int, list[torch.Tensor]] = {seq.seq_id: [] for seq in sequences}
        cached_tokens: dict[int, list[int]] = {seq.seq_id: [] for seq in sequences}
        steps = 0
        while scheduler.has_unfinished_requests() and steps < 500:
            batch = scheduler.step()
            if batch is None:
                continue
            for seq in batch.sequences:
                if batch.batch_type.value == "prefill":
                    cached_tokens[seq.seq_id].append(seq.num_cached_tokens)
            if batch.batch_type.value == "prefill":
                next_tokens = [runner.prefill(batch.sequences[0], allocator=allocator)]
            else:
                next_tokens = runner.decode_step(batch.sequences, allocator=allocator)
            for seq in batch.sequences:
                logits_history[seq.seq_id].append(seq.last_logits.detach().cpu())
            scheduler.post_step(batch, next_tokens)
            steps += 1

        assert not scheduler.has_unfinished_requests(), "Scheduler did not finish within 500 steps."
        assert allocator.ref_counts == {}
        assert allocator.num_available() == allocator.num_blocks
        return sequences, logits_history, cached_tokens

    off_sequences, off_logits, _ = run_mode("off")
    for prefix_mode in ("share", "full"):
        cached_sequences, cached_logits, cached_tokens = run_mode(prefix_mode)
        divergences = []
        exact_matches = 0

        for prompt_idx, (off_seq, cached_seq) in enumerate(zip(off_sequences, cached_sequences)):
            if off_seq.generated_ids == cached_seq.generated_ids:
                exact_matches += 1
                continue
            divergent_step = next(
                (
                    step
                    for step, (off_token, cached_token) in enumerate(
                        zip(off_seq.generated_ids, cached_seq.generated_ids)
                    )
                    if off_token != cached_token
                ),
                min(len(off_seq.generated_ids), len(cached_seq.generated_ids)),
            )
            logits = off_logits[off_seq.seq_id][divergent_step]
            top2_values = torch.topk(logits, k=2).values
            divergences.append(
                {
                    "prefix_mode": prefix_mode,
                    "prompt_idx": prompt_idx,
                    "divergent_step": divergent_step,
                    "off_token": (
                        off_seq.generated_ids[divergent_step]
                        if divergent_step < len(off_seq.generated_ids)
                        else None
                    ),
                    "cached_token": (
                        cached_seq.generated_ids[divergent_step]
                        if divergent_step < len(cached_seq.generated_ids)
                        else None
                    ),
                    "top1_top2_margin": float((top2_values[0] - top2_values[1]).item()),
                }
            )

        print(f"Prefix-cache {prefix_mode} divergences: {divergences}")
        assert exact_matches == len(prompts), (
            f"Prefix-cache {prefix_mode} mismatch: {len(divergences)} prompts diverged. "
            f"First divergence and off-mode top-1/top-2 margins: {divergences}"
        )
        assert any(values for values in cached_tokens.values()), (
            f"{prefix_mode} mode did not reuse any prefix-cache blocks."
        )


def test_real_gpu_full_prefix_prefill_logits_and_precision_yardstick(qwen_model_and_tokenizer):
    model, tokenizer, device = qwen_model_and_tokenizer
    block_size = 16
    num_blocks = 128
    prefix_texts = (
        "A detailed overview of astronomy, including planets, stars, galaxies, telescopes, "
        "space exploration, and the history of observing the night sky.",
        "A practical explanation of machine learning, neural networks, data, optimization, "
        "training, evaluation, and useful applications of modern artificial intelligence.",
    )
    roots = []
    for text in prefix_texts:
        repeated_text = " ".join([text] * 4)
        root_ids = tokenizer(repeated_text, return_tensors="pt").input_ids[0].tolist()
        assert len(root_ids) >= 4 * block_size
        roots.append(root_ids[: 4 * block_size])

    probe_prompts: list[list[int]] = []
    expected_hits: list[int] = []
    for idx in range(20):
        hit_blocks = idx % 4 + 1
        root = roots[idx % len(roots)]
        suffix_text = f"Continuation number {idx} considers a different example."
        suffix_ids = tokenizer(suffix_text, add_special_tokens=False).input_ids
        suffix_len = idx % (block_size - 1) + 1
        suffix_ids = (suffix_ids * ((suffix_len + len(suffix_ids) - 1) // len(suffix_ids)))[:suffix_len]
        probe_prompts.append(root[: hit_blocks * block_size] + suffix_ids)
        expected_hits.append(hit_blocks * block_size)

    def run_prefills(prefix_mode: str):
        allocator = BlockAllocator(
            num_blocks=num_blocks,
            enable_prefix_cache=prefix_mode != "off",
        )
        scheduler = Scheduler(
            block_allocator=allocator,
            block_size=block_size,
            watermark_blocks=1,
            max_batch_size=4,
            debug_invariants=True,
            prefix_mode=prefix_mode,
        )
        runner = KernelPagedManualDecode(
            model=model,
            tokenizer=tokenizer,
            block_size=block_size,
            num_blocks=num_blocks,
            device=device,
            dtype=torch.float16,
            prefix_mode=prefix_mode,
        )
        [
            scheduler.add_request(
                prompt=f"seed-{idx}",
                max_new_tokens=1,
                prompt_token_ids=root,
            )
            for idx, root in enumerate(roots)
        ]
        probes = [
            scheduler.add_request(
                prompt=f"probe-{idx}",
                max_new_tokens=1,
                prompt_token_ids=prompt_ids,
            )
            for idx, prompt_ids in enumerate(probe_prompts)
        ]
        captured: dict[int, torch.Tensor] = {}
        observed_hits: dict[int, int] = {}
        probe_seq_ids = {seq.seq_id for seq in probes}

        while scheduler.has_unfinished_requests():
            batch = scheduler.step()
            assert batch is not None and batch.batch_type.value == "prefill"
            seq = batch.sequences[0]
            if seq.seq_id in probe_seq_ids:
                observed_hits[seq.seq_id] = seq.num_cached_tokens
            token = runner.prefill(seq, allocator=allocator)
            captured[seq.seq_id] = seq.last_logits.detach().cpu()
            scheduler.post_step(batch, [token])

        assert allocator.ref_counts == {}
        assert allocator.num_available() == num_blocks
        return probes, captured, observed_hits

    off_probes, off_logits, _ = run_prefills("off")
    full_probes, full_logits, full_hits = run_prefills("full")
    cache_differences = []
    for idx, (off_seq, full_seq) in enumerate(zip(off_probes, full_probes)):
        assert full_hits[full_seq.seq_id] == expected_hits[idx]
        cache_differences.append(
            (off_logits[off_seq.seq_id].float() - full_logits[full_seq.seq_id].float())
            .abs()
            .max()
            .item()
        )

    model_dtype = next(model.parameters()).dtype
    fp32_differences = []
    try:
        model.to(dtype=torch.float32)
        with torch.inference_mode():
            for prompt_ids, off_seq in zip(probe_prompts, off_probes):
                input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)
                fp32_logits = model(
                    input_ids=input_ids,
                    use_cache=False,
                    logits_to_keep=1,
                ).logits[0, -1].float().cpu()
                fp16_logits = off_logits[off_seq.seq_id].float()
                fp32_differences.append((fp16_logits - fp32_logits).abs().max().item())
    finally:
        model.to(dtype=model_dtype)

    cache_max_diff = max(cache_differences)
    precision_max_diff = max(fp32_differences)
    per_probe_differences = [
        {
            "probe_idx": idx,
            "cached_tokens": expected_hits[idx],
            "max_abs_diff": cache_differences[idx],
            "fp16_vs_fp32_max_abs_diff": fp32_differences[idx],
        }
        for idx in range(len(probe_prompts))
    ]
    by_hit_depth = {
        hit_tokens: max(
            difference
            for difference, observed_hit in zip(cache_differences, expected_hits)
            if observed_hit == hit_tokens
        )
        for hit_tokens in set(expected_hits)
    }
    print(
        "Full-prefix prefill max abs logit difference: "
        f"full-vs-off={cache_max_diff:.8g}, fp16-vs-fp32={precision_max_diff:.8g}, "
        f"by_cached_tokens={by_hit_depth}, probes={per_probe_differences}"
    )
    assert cache_max_diff <= max(2 * precision_max_diff, 1e-5), (
        f"Cached-prefix logit difference {cache_max_diff} exceeded twice the fp16-vs-fp32 "
        f"yardstick {precision_max_diff}. Probe details: {per_probe_differences}"
    )


def test_real_gpu_preemption_stress_tiny_pool(qwen_model_and_tokenizer):
    """
    (2) A deliberately small pool (e.g. 80 blocks) with 10 concurrent requests
    so preemption fires: assert preemptions > 0, exact token counts,
    free blocks == num_blocks at the end, and compare outputs to an unconstrained run.
    """
    model, tokenizer, device = qwen_model_and_tokenizer
    block_size = 16
    small_num_blocks = 25  # Deliberately tiny pool: 10 requests needing ~40 blocks total
    large_num_blocks = 512
    max_new_tokens = 48

    prompts = [
        "Explain the theory of general relativity in simple terms for a child:",
        "Write a detailed Python function implementing a red-black balanced binary search tree:",
        "What are the major economic and geopolitical causes of the Industrial Revolution?",
        "Compare and contrast supervised learning, unsupervised learning, and reinforcement learning:",
        "Describe the architectural differences between Transformer and Recurrent Neural Networks:",
        "Discuss the biochemical pathways involved in cellular respiration and ATP generation:",
        "How do modern operating systems manage virtual memory paging and address translation?",
        "Write a summary of Shakespeare's Hamlet exploring themes of revenge and existential dread:",
        "Detail the principles of quantum cryptography, including BB84 protocol and quantum key distribution:",
        "Explain how continuous batching and PagedAttention optimize GPU memory utilization in vLLM:",
    ]

    # 1. Unconstrained reference run (with plenty of blocks, no preemption)
    unconstrained_allocator = BlockAllocator(num_blocks=large_num_blocks)
    unconstrained_scheduler = Scheduler(
        block_allocator=unconstrained_allocator,
        block_size=block_size,
        watermark_blocks=2,
        max_batch_size=16,
    )
    unconstrained_runner = KernelPagedManualDecode(
        model=model,
        tokenizer=tokenizer,
        block_size=block_size,
        num_blocks=large_num_blocks,
        device=device,
        dtype=torch.float16,
    )

    unconstrained_seqs = []
    for p in prompts:
        inp = tokenizer(p, return_tensors="pt")
        prompt_ids = inp.input_ids[0].tolist()
        s = unconstrained_scheduler.add_request(prompt=p, max_new_tokens=max_new_tokens, prompt_token_ids=prompt_ids)
        unconstrained_seqs.append(s)

    steps = 0
    while unconstrained_scheduler.has_unfinished_requests() and steps < 1000:
        b = unconstrained_scheduler.step()
        if b is None:
            break
        if b.batch_type.value == "prefill":
            tok = unconstrained_runner.prefill(b.sequences[0], allocator=unconstrained_allocator)
            unconstrained_scheduler.post_step(b, [tok])
        else:
            toks = unconstrained_runner.decode_step(b.sequences, allocator=unconstrained_allocator)
            unconstrained_scheduler.post_step(b, toks)
        steps += 1

    reference_outputs = {i: seq.generated_ids for i, seq in enumerate(unconstrained_seqs)}

    # 2. Constrained tiny-pool run (80 blocks, 10 concurrent requests)
    # With 10 requests, each prompt is ~25-30 tokens (2 blocks) + 32 generated tokens (2 blocks)
    # Total needed = ~40-50 blocks across all requests, but when watermark and dynamic allocations
    # collide with high concurrency and watermark_blocks=8, preemption is forced to fire!
    constrained_allocator = BlockAllocator(num_blocks=small_num_blocks)
    constrained_scheduler = Scheduler(
        block_allocator=constrained_allocator,
        block_size=block_size,
        watermark_blocks=1,
        max_batch_size=10,
        debug_invariants=True,
    )
    constrained_runner = KernelPagedManualDecode(
        model=model,
        tokenizer=tokenizer,
        block_size=block_size,
        num_blocks=small_num_blocks,
        device=device,
        dtype=torch.float16,
    )

    constrained_seqs = []
    for p in prompts:
        inp = tokenizer(p, return_tensors="pt")
        prompt_ids = inp.input_ids[0].tolist()
        s = constrained_scheduler.add_request(prompt=p, max_new_tokens=max_new_tokens, prompt_token_ids=prompt_ids)
        constrained_seqs.append(s)

    steps = 0
    while constrained_scheduler.has_unfinished_requests() and steps < 1500:
        b = constrained_scheduler.step()
        if b is None:
            break
        if b.batch_type.value == "prefill":
            tok = constrained_runner.prefill(b.sequences[0], allocator=constrained_allocator)
            constrained_scheduler.post_step(b, [tok])
        else:
            toks = constrained_runner.decode_step(b.sequences, allocator=constrained_allocator)
            constrained_scheduler.post_step(b, toks)
        steps += 1

    print(f"\n--- Test (2) Tiny Pool Preemption Stress Summary ---")
    print(f"Preemption count: {constrained_scheduler.preemption_count}")
    print(f"Total steps: {steps}")
    print(f"Free blocks at end: {constrained_allocator.get_free_block_count()} / {small_num_blocks}")

    # Assertions
    # 1. Preemptions fired
    assert constrained_scheduler.preemption_count > 0, "Expected preemption to fire under constrained memory!"

    # 2. Exact token counts for all sequences
    for i, seq in enumerate(constrained_seqs):
        gen_tokens = seq.all_generated_ids or seq.generated_ids
        assert len(gen_tokens) == max_new_tokens, (
            f"Seq {i} generated {len(gen_tokens)} tokens, expected {max_new_tokens}!"
        )

    # 3. Block conservation: all blocks returned to pool
    assert constrained_allocator.get_free_block_count() == small_num_blocks, (
        f"Memory leak detected! Free blocks: {constrained_allocator.get_free_block_count()}, expected {small_num_blocks}"
    )

    # 4. Compare outputs to unconstrained run
    matched_outputs = 0
    for i in range(len(prompts)):
        ref_tokens = reference_outputs[i]
        const_tokens = constrained_seqs[i].all_generated_ids or constrained_seqs[i].generated_ids
        if ref_tokens == const_tokens:
            matched_outputs += 1
    print(f"Exact output match against unconstrained run: {matched_outputs} / {len(prompts)}")
    # In deterministic greedy decode with re-prefill, greedy paths should match
    assert matched_outputs >= 8, f"Expected at least 8/10 exact matches after preemption, got {matched_outputs}"


@pytest.mark.anyio
async def test_real_gpu_12_prompts_staggered_divergence(qwen_model_and_tokenizer):
    """
    (3) The existing 12-prompt staggered-arrival e2e test, with top-1/top-2 margin
    at the first divergent step and a teacher-forced check on any mismatched prompt.
    """
    model, tokenizer, device = qwen_model_and_tokenizer

    test_prompts = [
        "The capital of France is",
        "def add(a, b):",
        "Hello world! Today is a great day to",
        "Once upon a time in a small village,",
        "Explain the concept of continuous batching in Large Language Models:",
        "Write a quick python function to compute the Fibonacci sequence up to n numbers:",
        "In deep learning, key-value cache paging prevents fragmentation by",
        "Quantum computing utilizes qubits and superposition to solve complex problems such as",
        "The transformer architecture relies on multi-head self-attention mechanisms to dynamically capture relationships between distant tokens in a sequence. Specifically,",
        "Artificial intelligence has evolved rapidly from early rule-based expert systems to modern deep neural networks with billions of parameters. Key breakthroughs include",
        "To optimize inference performance, vLLM introduces PagedAttention which draws inspiration from virtual memory in operating systems. By dividing the KV cache into fixed blocks,",
        "A distributed consensus algorithm like Raft or Paxos ensures that multiple replicas in a cluster agree on state machine transitions even in the presence of network partitions and node crashes. In particular,",
    ]
    num_prompts = len(test_prompts)
    max_new_tokens = 16
    block_size = 16
    num_blocks = 512

    # 1. Single-sequence reference ground truth
    single_engine = KernelPagedManualDecode(
        model=model,
        tokenizer=tokenizer,
        block_size=block_size,
        num_blocks=num_blocks,
        device=device,
        dtype=torch.float16,
    )

    reference_tokens: Dict[int, List[int]] = {}
    reference_logits: Dict[int, List[torch.Tensor]] = {}

    for idx, prompt in enumerate(test_prompts):
        tokens, logits_hist = single_engine.generate(
            prompt=prompt,
            max_new_tokens=max_new_tokens,
            return_logits_history=True,
        )
        reference_tokens[idx] = tokens
        reference_logits[idx] = [l.detach().cpu() for l in logits_hist]

    # 2. AsyncEngine with continuous batching scheduler and debug invariants enabled
    allocator = BlockAllocator(num_blocks=num_blocks)
    scheduler = Scheduler(
        block_allocator=allocator,
        block_size=block_size,
        watermark_blocks=2,
        max_batch_size=16,
        debug_invariants=True,
    )
    run_logger = RunLogger(run_name="e2e_12_prompts_async_gpu")
    async_runner = KernelPagedManualDecode(
        model=model,
        tokenizer=tokenizer,
        block_size=block_size,
        num_blocks=num_blocks,
        device=device,
        dtype=torch.float16,
    )
    engine = AsyncEngine(
        model_runner=async_runner,
        scheduler=scheduler,
        tokenizer=tokenizer,
        idle_sleep_s=0.001,
        run_logger=run_logger,
    )

    # 3. Launch with random staggered delays
    random.seed(42)
    stagger_offsets = [random.uniform(0.0, 0.08) for _ in range(num_prompts)]
    async_results: Dict[int, List[int]] = {}

    async def submit_prompt(idx: int, prompt_text: str, delay: float):
        if delay > 0:
            await asyncio.sleep(delay)
        toks = []
        async for tok in engine.add_request(prompt=prompt_text, max_tokens=max_new_tokens):
            toks.append(tok)
        async_results[idx] = toks

    tasks = [
        asyncio.create_task(submit_prompt(i, test_prompts[i], stagger_offsets[i]))
        for i in range(num_prompts)
    ]
    await asyncio.gather(*tasks)
    await engine.stop()

    # 4. Compare results & inspect top-1/top-2 logit margin + teacher forcing
    exact_matches = 0
    mismatches = []

    for idx in range(num_prompts):
        ref = reference_tokens[idx]
        actual = async_results[idx]
        if ref == actual:
            exact_matches += 1
        else:
            div_step = None
            for pos in range(min(len(ref), len(actual))):
                if ref[pos] != actual[pos]:
                    div_step = pos
                    break
            if div_step is None:
                div_step = min(len(ref), len(actual))

            # Reference top-1 / top-2 margin at divergent position
            top1_top2_margin = None
            if div_step < len(reference_logits[idx]):
                ref_l = reference_logits[idx][div_step].squeeze()
                top2, _ = torch.topk(ref_l, k=2)
                top1_top2_margin = float((top2[0] - top2[1]).item())

            # Teacher-forced single-step verification:
            # Feed the exact prefix up to div_step into the standard model forward pass
            # and verify the logits match closely
            prefix_tokens = tokenizer(test_prompts[idx]).input_ids + ref[:div_step]
            prefix_tensor = torch.tensor([prefix_tokens], device=device)
            with torch.inference_mode():
                tf_out = model(input_ids=prefix_tensor)
                tf_logits = tf_out.logits[0, -1, :].detach().cpu()
                ref_step_logits = reference_logits[idx][div_step].squeeze()
                tf_max_diff = float((tf_logits - ref_step_logits).abs().max().item())

            mismatches.append({
                "prompt_idx": idx,
                "prompt": test_prompts[idx][:40] + "...",
                "first_divergent_pos": div_step,
                "expected_token": ref[div_step] if div_step < len(ref) else None,
                "actual_token": actual[div_step] if div_step < len(actual) else None,
                "expected_str": tokenizer.decode([ref[div_step]]) if div_step < len(ref) else "",
                "actual_str": tokenizer.decode([actual[div_step]]) if div_step < len(actual) else "",
                "top1_top2_margin": top1_top2_margin,
                "teacher_forced_max_logit_diff": tf_max_diff,
            })

    exact_match_rate = exact_matches / num_prompts
    print("\n--- Test (3) 12 Prompts Staggered Async Summary ---")
    print(f"Exact match rate: {exact_match_rate * 100:.2f}% ({exact_matches}/{num_prompts})")
    for m in mismatches:
        print(f"Mismatch detail: {m}")

    assert allocator.get_free_block_count() == num_blocks, "Blocks leaked in async engine run!"
    assert exact_match_rate >= 0.90, f"Match rate {exact_match_rate * 100:.2f}% below 90%!"
    for m in mismatches:
        if m["top1_top2_margin"] is not None:
            assert m["top1_top2_margin"] < 0.5, (
                f"Large top1/top2 margin ({m['top1_top2_margin']}) at divergent step {m['first_divergent_pos']}!"
            )
        assert m["teacher_forced_max_logit_diff"] < 0.2, (
            f"Teacher-forced forward logits diverged: {m['teacher_forced_max_logit_diff']}!"
        )
