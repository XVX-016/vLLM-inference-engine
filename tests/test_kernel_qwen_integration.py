"""
Tests for Triton Paged Attention Kernel Integration into Qwen2.5-0.5B.
Verifies:
1. Greedy generated tokens from KernelPagedManualDecode match model.generate().
2. Per-step logits between KernelPagedManualDecode and HF standard decode
   are close within fp16 tolerance across all steps.
"""
from __future__ import annotations

import pytest
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from engine.kernels.paged_attention_integration import KernelPagedManualDecode


@pytest.fixture(scope="module")
def qwen_model_and_tokenizer():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device != "cuda":
        pytest.skip("CUDA device required for Triton paged attention kernel integration tests")

    model_name = "Qwen/Qwen2.5-0.5B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.float16, device_map=device)
    model.eval()
    return model, tokenizer, device


def test_kernel_paged_decode_matches_hf_generate_greedy(qwen_model_and_tokenizer):
    """
    Tests that greedy decoded tokens from KernelPagedManualDecode match model.generate()
    bit-for-bit for the same prompt.
    """
    model, tokenizer, device = qwen_model_and_tokenizer
    prompt = "The capital of France is"
    max_new_tokens = 8

    # 1. Kernel-backed paged manual decode
    engine = KernelPagedManualDecode(
        model=model,
        tokenizer=tokenizer,
        block_size=16,
        num_blocks=512,
        device=device,
        dtype=torch.float16,
    )
    with torch.inference_mode():
        kernel_tokens = engine.generate(prompt=prompt, max_new_tokens=max_new_tokens)

    # 2. Reference Hugging Face model.generate() with pure greedy argmax
    inp = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.inference_mode():
        hf_gen_out = model.generate(
            **inp,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            repetition_penalty=1.0,
        )
    prompt_len = inp.input_ids.shape[1]
    hf_tokens = hf_gen_out[0, prompt_len:].tolist()

    assert kernel_tokens == hf_tokens, (
        f"Greedy token mismatch:\n"
        f"Kernel tokens: {kernel_tokens}\n"
        f"HF tokens:     {hf_tokens}\n"
        f"Kernel text:   {tokenizer.decode(kernel_tokens)}\n"
        f"HF text:       {tokenizer.decode(hf_tokens)}"
    )


def test_kernel_paged_decode_logits_allclose_fp16(qwen_model_and_tokenizer):
    """
    Compares per-step logits between KernelPagedManualDecode and HF standard decode.
    Verifies torch.allclose across all steps at reasonable fp16 tolerances (atol=0.1, rtol=0.1).
    """
    model, tokenizer, device = qwen_model_and_tokenizer
    prompt = "The capital of France is"
    max_new_tokens = 6

    inp = tokenizer(prompt, return_tensors="pt").to(device)

    # 1. Standard HF step-by-step decode logits
    hf_logits = []
    pkv = None
    curr_input = inp.input_ids
    with torch.inference_mode():
        for step in range(max_new_tokens):
            if step == 0:
                out = model(input_ids=curr_input, use_cache=True)
            else:
                out = model(input_ids=curr_input, past_key_values=pkv, use_cache=True)
            pkv = out.past_key_values
            step_logit = out.logits[:, -1, :]
            hf_logits.append(step_logit)
            curr_input = torch.argmax(step_logit, dim=-1, keepdim=True)

    # 2. Kernel-backed paged decode logits
    engine = KernelPagedManualDecode(
        model=model,
        tokenizer=tokenizer,
        block_size=16,
        num_blocks=512,
        device=device,
        dtype=torch.float16,
    )
    with torch.inference_mode():
        _, kernel_logits = engine.generate(
            prompt=prompt,
            max_new_tokens=max_new_tokens,
            return_logits_history=True,
        )

    # 3. Assert allclose per step
    atol = 1e-1
    rtol = 1e-1
    for step in range(max_new_tokens):
        k_log = kernel_logits[step]
        h_log = hf_logits[step]
        max_diff = (k_log - h_log).abs().max().item()
        assert torch.allclose(k_log, h_log, atol=atol, rtol=rtol), (
            f"Step {step} logits diverged beyond fp16 tolerance! Max diff: {max_diff:.6f}"
        )
