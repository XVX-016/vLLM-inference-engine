"""
Rigorous Multi-Prompt Integration Verification (64 tokens, 5 prompts)
Compares:
1. Triton Kernel-backed Paged Engine (FP16)
2. Hugging Face Standard Decode (FP16)
3. Hugging Face Reference (FP32 ground truth)

Proves whether numerical drift between Triton and HF-FP16 is pure FP16 rounding
or algorithmic discrepancy.
"""
import os
import sys
from typing import Dict, List

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from engine.kernels.paged_attention_integration import KernelPagedManualDecode

PROMPTS = [
    "The capital of France is",
    "Once upon a time in a land far away, there lived",
    "def fibonacci(n):\n    if n <= 1:\n        return n\n    return",
    "Artificial intelligence is transforming healthcare by",
    "The recipe for a perfect chocolate cake requires",
]

def run_rigorous_validation():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    assert device == "cuda", "CUDA required"
    model_name = "Qwen/Qwen2.5-0.5B-Instruct"
    max_new_tokens = 64

    print("=" * 85)
    print(f"  PHASE 4 RIGOROUS INTEGRATION VALIDATION ({len(PROMPTS)} prompts, {max_new_tokens} tokens)")
    print("=" * 85)

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    print("\n1. Loading FP32 Ground Truth Model...")
    model_fp32 = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.float32, device_map=device)
    model_fp32.eval()

    print("2. Loading FP16 Model for Kernel & HF Baseline...")
    model_fp16 = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.float16, device_map=device)
    model_fp16.eval()

    paged_engine = KernelPagedManualDecode(
        model=model_fp16,
        tokenizer=tokenizer,
        block_size=16,
        num_blocks=1024,
        device=device,
        dtype=torch.float16,
    )

    summary_stats = []

    for p_idx, prompt in enumerate(PROMPTS):
        print(f"\n" + "-" * 75)
        print(f"Prompt {p_idx + 1}/{len(PROMPTS)}: {repr(prompt)}")
        print("-" * 75)

        inp = tokenizer(prompt, return_tensors="pt").to(device)

        # A. FP32 HF Ground Truth Step-by-Step
        fp32_tokens, fp32_logits = [], []
        pkv32 = None
        curr32 = inp.input_ids
        with torch.inference_mode():
            for s in range(max_new_tokens):
                out = model_fp32(input_ids=curr32 if s > 0 else inp.input_ids, past_key_values=pkv32, use_cache=True)
                pkv32 = out.past_key_values
                l = out.logits[:, -1, :]
                fp32_logits.append(l)
                nxt = torch.argmax(l, dim=-1, keepdim=True)
                fp32_tokens.append(nxt.item())
                curr32 = nxt

        # B. FP16 HF Baseline Step-by-Step
        fp16_hf_tokens, fp16_hf_logits = [], []
        pkv16 = None
        curr16 = inp.input_ids
        with torch.inference_mode():
            for s in range(max_new_tokens):
                out = model_fp16(input_ids=curr16 if s > 0 else inp.input_ids, past_key_values=pkv16, use_cache=True)
                pkv16 = out.past_key_values
                l = out.logits[:, -1, :]
                fp16_hf_logits.append(l)
                nxt = torch.argmax(l, dim=-1, keepdim=True)
                fp16_hf_tokens.append(nxt.item())
                curr16 = nxt

        # C. Triton Paged Engine Decode (FP16)
        with torch.inference_mode():
            kernel_tokens, kernel_logits = paged_engine.generate(
                prompt=prompt,
                max_new_tokens=max_new_tokens,
                return_logits_history=True,
            )

        # Compare Greedy Tokens
        tokens_match_hf16 = (kernel_tokens == fp16_hf_tokens)
        tokens_match_fp32 = (kernel_tokens == fp32_tokens)

        # Measure error distributions:
        # Error 1: |Kernel_FP16 - HF_FP16|
        # Error 2: |HF_FP16 - FP32|  (intrinsic FP16 precision loss)
        # Error 3: |Kernel_FP16 - FP32|
        diffs_kernel_vs_hf16 = []
        diffs_hf16_vs_fp32 = []
        diffs_kernel_vs_fp32 = []

        for s in range(max_new_tokens):
            k_l = kernel_logits[s].float()
            h16_l = fp16_hf_logits[s].float()
            f32_l = fp32_logits[s]

            diffs_kernel_vs_hf16.append((k_l - h16_l).abs().max().item())
            diffs_hf16_vs_fp32.append((h16_l - f32_l).abs().max().item())
            diffs_kernel_vs_fp32.append((k_l - f32_l).abs().max().item())

        max_err_k_vs_hf16 = max(diffs_kernel_vs_hf16)
        mean_err_k_vs_hf16 = sum(diffs_kernel_vs_hf16) / len(diffs_kernel_vs_hf16)

        max_err_hf16_vs_f32 = max(diffs_hf16_vs_fp32)
        mean_err_hf16_vs_f32 = sum(diffs_hf16_vs_fp32) / len(diffs_hf16_vs_fp32)

        max_err_k_vs_f32 = max(diffs_kernel_vs_fp32)
        mean_err_k_vs_f32 = sum(diffs_kernel_vs_fp32) / len(diffs_kernel_vs_fp32)

        # First token divergence if any
        first_div_step = None
        for s in range(max_new_tokens):
            if kernel_tokens[s] != fp16_hf_tokens[s]:
                first_div_step = s
                break

        print(f"  Greedy Match with HF-FP16:  {'100% IDENTICAL' if tokens_match_hf16 else f'Diverged at step {first_div_step}'}")
        print(f"  Greedy Match with FP32:     {'100% IDENTICAL' if tokens_match_fp32 else 'Slight FP16 divergence'}")
        print(f"  Max Diff |Kernel - HF-FP16|: {max_err_k_vs_hf16:.4f} (Mean: {mean_err_k_vs_hf16:.4f})")
        print(f"  Max Diff |HF-FP16 - FP32|:   {max_err_hf16_vs_f32:.4f} (Mean: {mean_err_hf16_vs_f32:.4f})")
        print(f"  Max Diff |Kernel - FP32|:    {max_err_k_vs_f32:.4f} (Mean: {mean_err_k_vs_f32:.4f})")

        summary_stats.append({
            "prompt": prompt,
            "match_hf16": tokens_match_hf16,
            "match_fp32": tokens_match_fp32,
            "max_k_vs_hf16": max_err_k_vs_hf16,
            "max_hf16_vs_fp32": max_err_hf16_vs_f32,
            "max_k_vs_fp32": max_err_k_vs_f32,
            "div_step": first_div_step,
        })

    print("\n" + "=" * 95)
    print("                      64-TOKEN MULTI-PROMPT VALIDATION SUMMARY")
    print("=" * 95)
    print("| Prompt Summary                          | Match HF-16 | Max |K - HF16| | Max |HF16 - FP32| | Max |K - FP32| |")
    print("|-----------------------------------------|-------------|----------------|-------------------|----------------|")
    for stat in summary_stats:
        p_short = (stat['prompt'][:37] + "...") if len(stat['prompt']) > 37 else stat['prompt'].ljust(40)
        p_short = p_short.replace("\n", " ")
        m_str = "YES (64/64)" if stat['match_hf16'] else f"Div @ step {stat['div_step']}"
        print(f"| {p_short:<40} | {m_str:<11} | {stat['max_k_vs_hf16']:<14.4f} | {stat['max_hf16_vs_fp32']:<17.4f} | {stat['max_k_vs_fp32']:<14.4f} |")

    return summary_stats

if __name__ == "__main__":
    run_rigorous_validation()
