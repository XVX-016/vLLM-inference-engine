"""
Phase 6 Item (2) — GPU K/V equivalence and logit precision diagnostics.

Run with:
    python -m scripts.diagnose_kv_equivalence

Requires: CUDA GPU, Qwen/Qwen2.5-0.5B-Instruct weights on disk (or HF cache).

Outputs:
  A. Per-layer max-abs K/V difference between cached blocks and fresh compute
     on 6 shared-prefix prompts (mode=share, debug_prefix_cache=True).
     Answers: 6c — "are K/V blocks equivalent?"

  B. |full−fp32| vs |off−fp32| last-position logits for a 20-prompt test.
     Answers: 6d — "does the suffix-only path add error beyond off vs fp32?"
"""
from __future__ import annotations

import logging
import sys
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from engine.kernels.paged_attention_integration import KernelPagedManualDecode
from engine.memory.block_allocator import BlockAllocator
from engine.scheduler.continuous_batching import Scheduler

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s %(name)s: %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger("kv_diag")


BLOCK_SIZE = 16
MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"

SHARED_PREFIX = (
    "Continuous batching operates at the iteration level, scheduling tokens "
    "immediately rather than waiting for batches to finish. Operating systems "
    "use page tables and translation lookaside buffers to map virtual addresses "
    "to physical pages. Autoregressive generation generates tokens one by one "
    "by computing the argmax or sampling from output logits."
)

SUFFIXES = [
    " The KV cache stores key and value tensors.",
    " Attention heads compute queries, keys, and values.",
    " Memory bandwidth is the primary bottleneck during decode.",
    " FlashAttention tiles the attention computation in SRAM.",
    " Prefix caching avoids redundant prefill computation.",
    " Paged allocation removes memory fragmentation in LLM serving.",
]

PROMPTS_20 = [
    "The transformer architecture processes sequences by computing attention over all positions.",
    "Memory bandwidth is the primary bottleneck during autoregressive decoding.",
    "Continuous batching improves GPU utilization by dynamically merging requests.",
    "KV-cache paging allows zero-waste copy-on-write sharing for parallel sampling.",
    "FlashAttention restructures attention into tiled SRAM blocks to reduce HBM traffic.",
    "Quantization compresses weights to int4 or int8 to reduce memory and compute costs.",
    "Speculative decoding uses a small draft model to propose tokens that the large model verifies.",
    "Tensor parallelism splits model layers across multiple GPUs to fit large models.",
    "Pipeline parallelism assigns different layers to different devices in a pipeline.",
    "The attention mechanism computes a weighted sum of value vectors using query-key dot products.",
    "Rotary position embeddings encode relative positions via complex-valued rotations.",
    "The softmax function converts raw logits into a probability distribution over the vocabulary.",
    "Layer normalization stabilizes training by normalizing hidden states to zero mean and unit variance.",
    "Feed-forward networks in transformers apply two linear projections with a nonlinearity in between.",
    "The key-value cache stores intermediate attention computations to speed up autoregressive generation.",
    "Grouped query attention reduces the KV cache size by sharing key and value heads across query heads.",
    "The watermark block mechanism ensures the allocator always keeps at least one block free.",
    "Preemption evicts the most recently admitted sequence to free blocks for higher-priority requests.",
    "Paged attention divides the KV cache into fixed-size physical blocks indexed by a block table.",
    "The prefill phase processes the entire prompt in a single forward pass to populate the KV cache.",
]


def load_model(device: str) -> tuple[Any, Any]:
    logger.info("Loading model %s on %s ...", MODEL_NAME, device)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, dtype=torch.float16, device_map=device
    )
    model.eval()
    return model, tokenizer


# ---------------------------------------------------------------------------
# Part A: K/V equivalence for cached blocks vs fresh compute
# ---------------------------------------------------------------------------

def run_kv_equivalence(model: Any, tokenizer: Any, device: str) -> None:
    print("=" * 70)
    print("PART A: K/V block equivalence (share mode, debug_prefix_cache=True)")
    print("=" * 70)

    num_blocks = 512
    runner_share = KernelPagedManualDecode(
        model=model,
        tokenizer=tokenizer,
        block_size=BLOCK_SIZE,
        num_blocks=num_blocks,
        device=device,
        dtype=torch.float16,
        prefix_mode="share",
        debug_prefix_cache=True,
    )

    allocator = BlockAllocator(num_blocks=num_blocks, enable_prefix_cache=True)
    scheduler = Scheduler(
        block_allocator=allocator,
        block_size=BLOCK_SIZE,
        watermark_blocks=2,
        prefix_mode="share",
    )

    prefix_ids = tokenizer.encode(SHARED_PREFIX, add_special_tokens=False)
    aligned_len = (len(prefix_ids) // BLOCK_SIZE) * BLOCK_SIZE
    prefix_ids = prefix_ids[:aligned_len]
    print(f"  Shared prefix: {len(prefix_ids)} tokens ({len(prefix_ids) // BLOCK_SIZE} full blocks)")

    # Seed the cache with the shared prefix alone
    seq0 = scheduler.add_request(
        prompt=SHARED_PREFIX,
        max_new_tokens=1,
        prompt_token_ids=prefix_ids,
        ignore_eos=True,
    )
    scheduler.step()
    runner_share.prefill(seq0, allocator=allocator)

    max_abs_per_layer: list[float] | None = None
    num_layers = runner_share.num_layers

    for i, suffix in enumerate(SUFFIXES):
        full_ids = prefix_ids + tokenizer.encode(suffix, add_special_tokens=False)
        seq = scheduler.add_request(
            prompt=SHARED_PREFIX + suffix,
            max_new_tokens=1,
            prompt_token_ids=full_ids,
            ignore_eos=True,
        )

        # Capture per-layer diffs logged by _compare_cached_prefill_kv
        layer_diffs: dict[int, float] = {}
        paged_logger = logging.getLogger("engine.kernels.paged_attention_integration")
        _orig_info = paged_logger.info

        def _capture(msg: str, *args: Any) -> None:
            _orig_info(msg, *args)
            if "max_abs_difference" in (msg % args if args else msg):
                try:
                    formatted = msg % args
                    parts = {p.split("=")[0]: p.split("=")[1] for p in formatted.split() if "=" in p}
                    layer_diffs[int(parts["layer"])] = float(parts["max_abs_difference"])
                except Exception:
                    pass

        paged_logger.info = _capture  # type: ignore[method-assign]
        try:
            scheduler.step()
            runner_share.prefill(seq, allocator=allocator)
        finally:
            paged_logger.info = _orig_info  # type: ignore[method-assign]

        if layer_diffs:
            all_vals = list(layer_diffs.values())
            max_val = max(all_vals)
            mean_val = sum(all_vals) / len(all_vals)
            print(
                f"  prompt {i+1:2d}  cached={aligned_len}t  "
                f"max|diff| over {len(all_vals)} layers: {max_val:.3e}  mean: {mean_val:.3e}"
            )
            if max_abs_per_layer is None:
                max_abs_per_layer = list(all_vals)
            else:
                max_abs_per_layer = [max(a, b) for a, b in zip(max_abs_per_layer, all_vals)]
        else:
            print(f"  prompt {i+1:2d}: no K/V comparison triggered (possible cache miss)")

    if max_abs_per_layer is not None:
        print("\n  Per-layer envelope (max across all prompts):")
        for li, d in enumerate(max_abs_per_layer):
            bar = "#" * int(d / 0.001) if d < 0.05 else ">" * 40
            print(f"    layer {li:2d}: {d:.3e}  {bar}")
        fp16_eps = torch.finfo(torch.float16).eps
        print(f"\n  fp16 machine epsilon: {fp16_eps:.3e}")
        print(f"  fp16 quantization step at 1.0: {fp16_eps:.3e}")
    else:
        print("  [!] No data — check that prefix hit rate > 0 above.")


# ---------------------------------------------------------------------------
# Part B: |full-fp32| vs |off-fp32| last-position logits, 20 prompts
# ---------------------------------------------------------------------------

def run_logit_precision(model_fp16: Any, tokenizer: Any, device: str) -> None:
    print("\n" + "=" * 70)
    print("PART B: |off-fp32| last-position logit differences, 20 prompts")
    print("  (run mode=full separately with a primed cache to get |full-fp32|)")
    print("=" * 70)

    logger.info("Loading fp32 reference model on CPU ...")
    model_fp32 = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, dtype=torch.float32, device_map="cpu"
    )
    model_fp32.eval()

    diffs: list[float] = []
    for i, prompt in enumerate(PROMPTS_20):
        ids_gpu = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
        ids_cpu = ids_gpu.cpu()

        with torch.inference_mode():
            out16 = model_fp16(ids_gpu, use_cache=False)
            out32 = model_fp32(ids_cpu, use_cache=False)

        logit16 = out16.logits[0, -1].float().cpu()
        logit32 = out32.logits[0, -1].float()
        diff = (logit16 - logit32).abs().max().item()
        diffs.append(diff)
        print(f"  prompt {i+1:2d}  |off-fp32| max_abs={diff:.4f}  prompt_len={ids_gpu.shape[1]}")

    mean_d = sum(diffs) / len(diffs)
    max_d = max(diffs)
    print(f"\n  mean |off-fp32|: {mean_d:.4f}   max: {max_d:.4f}")
    print("  Yardstick from 6d: measured 0.154 (off vs fp32), threshold 2x = 0.208")
    print("  |full-fp32| should be similar to |off-fp32| if suffix-only path is clean.")


def main() -> None:
    if not torch.cuda.is_available():
        print("ERROR: No CUDA device available. This script requires a GPU.")
        sys.exit(1)

    device = "cuda"
    model_fp16, tokenizer = load_model(device)
    run_kv_equivalence(model_fp16, tokenizer, device)
    run_logit_precision(model_fp16, tokenizer, device)
    print("\nDone.")


if __name__ == "__main__":
    main()
