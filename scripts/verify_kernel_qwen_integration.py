"""
Verify Qwen2.5-0.5B integration with Triton Paged Attention Kernel.
Checks:
1. Greedy generated tokens match Hugging Face standard greedy generation.
2. Per-step logits match HF step-by-step logits within reasonable fp16 tolerance (torch.allclose).
"""
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from engine.kernels.paged_attention_integration import KernelPagedManualDecode

def run_verification():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    assert device == "cuda", "CUDA required for Triton kernel verification"
    
    model_name = "Qwen/Qwen2.5-0.5B-Instruct"
    print(f"Loading model {model_name} in fp16 on {device}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.float16, device_map=device)
    model.eval()

    prompt = "The capital of France is"
    max_new_tokens = 8
    print(f"\nPrompt: '{prompt}', max_new_tokens: {max_new_tokens}")

    # 1. Hugging Face standard step-by-step decode
    inp = tokenizer(prompt, return_tensors="pt").to(device)
    hf_tokens = []
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
            next_tok = torch.argmax(step_logit, dim=-1, keepdim=True)
            hf_tokens.append(next_tok.item())
            curr_input = next_tok

    hf_text = tokenizer.decode(hf_tokens)
    print(f"\n[HF Standard Decode]")
    print(f"  Tokens: {hf_tokens}")
    print(f"  Text:   {repr(hf_text)}")

    # 2. Kernel-backed Paged Engine decode
    engine = KernelPagedManualDecode(
        model=model,
        tokenizer=tokenizer,
        block_size=16,
        num_blocks=512,
        device=device,
        dtype=torch.float16,
    )

    with torch.inference_mode():
        kernel_tokens, kernel_logits = engine.generate(
            prompt=prompt,
            max_new_tokens=max_new_tokens,
            return_logits_history=True,
        )

    kernel_text = tokenizer.decode(kernel_tokens)
    print(f"\n[Triton Kernel Paged Decode]")
    print(f"  Tokens: {kernel_tokens}")
    print(f"  Text:   {repr(kernel_text)}")

    # 3. Assert Greedy Token Equality
    assert hf_tokens == kernel_tokens, (
        f"Greedy token mismatch!\n"
        f"HF:     {hf_tokens}\n"
        f"Kernel: {kernel_tokens}"
    )
    print("\n[Check 1: Greedy Tokens Match] PASS (Bit-for-bit identical)")

    # 4. Compare Per-Step Logits with torch.allclose
    print("\n[Check 2: Per-Step Logits Comparison]")
    # For fp16 through 24 transformer layers, small differences in softmax/accumulation
    # accumulate to max abs diff ~0.05-0.10 across the vocabulary (logits reach ~20-25).
    atol = 1e-1
    rtol = 1e-1

    all_close_pass = True
    for step in range(max_new_tokens):
        k_log = kernel_logits[step]
        h_log = hf_logits[step]
        max_diff = (k_log - h_log).abs().max().item()
        mean_diff = (k_log - h_log).abs().mean().item()
        is_close = torch.allclose(k_log, h_log, atol=atol, rtol=rtol)
        print(f"  Step {step}: max_abs_diff = {max_diff:.6f}, mean_abs_diff = {mean_diff:.6f}, allclose(atol={atol}, rtol={rtol}) = {is_close}")
        if not is_close:
            all_close_pass = False

    assert all_close_pass, f"Logits exceeded tolerance at atol={atol}, rtol={rtol}"
    print(f"PASS (All {max_new_tokens} steps within fp16 tolerance)")

    # 5. Also compare against model.generate() with pure greedy decoding
    # Note: Qwen2.5 generation_config defaults repetition_penalty to 1.1;
    # set repetition_penalty=1.0 to ensure standard pure greedy argmax.
    hf_gen_out = model.generate(
        **inp,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        repetition_penalty=1.0,
    )
    prompt_len = inp.input_ids.shape[1]
    hf_gen_tokens = hf_gen_out[0, prompt_len:].tolist()
    print(f"\n[Check 3: Comparison with model.generate(repetition_penalty=1.0)]")
    print(f"  model.generate() tokens: {hf_gen_tokens}")
    print(f"  kernel engine tokens:    {kernel_tokens}")
    assert hf_gen_tokens == kernel_tokens, "Mismatch with model.generate()!"
    print("PASS (Bit-for-bit identical with model.generate())")

    print("\nALL VERIFICATIONS PASSED SUCCESSFULLY!")

if __name__ == "__main__":
    run_verification()
