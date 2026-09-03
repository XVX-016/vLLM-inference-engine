"""
Phase 1 — Manual autoregressive decode loop with direct raw tensor management.

Goal: stop relying on model.generate() and HF's cache object magic. Own the
KV-cache as plain torch.Tensors of shape:
    [num_layers, batch, num_kv_heads, max_seq_len, head_dim]
so that in Phase 2 you can swap them out for paged, non-contiguous storage
without fighting HF's internal abstractions.

Features:
1. Prefill forward pass that extracts raw K and V tensors per layer.
2. Raw tensor buffer allocated on GPU [num_layers, batch, num_kv_heads, max_seq_len, head_dim].
3. Step-by-step decode loop directly slicing, updating, and indexing raw tensors.
4. Correctness test verifying bit-identical output against HF's model.generate(do_sample=False).
"""
from __future__ import annotations

from typing import List, Tuple
import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from transformers.cache_utils import DynamicCache, DynamicLayer


class RawTensorKVCache:
    """
    Plain contiguous KV-cache managed directly as raw torch.Tensors:
    One tensor for K and one for V across all layers:
        shape: [num_layers, batch_size, num_kv_heads, max_seq_len, head_dim]

    This is the exact contract that Phase 2's PagedKVCache replaces with
    block tables and physical page pools.
    """

    def __init__(
        self,
        num_layers: int,
        batch_size: int,
        num_kv_heads: int,
        max_seq_len: int,
        head_dim: int,
        device: torch.device,
        dtype: torch.dtype,
    ):
        self.num_layers = num_layers
        self.batch_size = batch_size
        self.num_kv_heads = num_kv_heads
        self.max_seq_len = max_seq_len
        self.head_dim = head_dim
        self.device = device
        self.dtype = dtype

        # Raw contiguous buffers
        self.k_buffer = torch.zeros(
            (num_layers, batch_size, num_kv_heads, max_seq_len, head_dim),
            device=device,
            dtype=dtype,
        )
        self.v_buffer = torch.zeros(
            (num_layers, batch_size, num_kv_heads, max_seq_len, head_dim),
            device=device,
            dtype=dtype,
        )
        self.seq_len = 0  # Number of valid tokens stored

    def write_prefill(self, k_tensors: List[torch.Tensor], v_tensors: List[torch.Tensor]) -> None:
        """
        Store prefill keys and values across all layers.
        Each tensor in k_tensors/v_tensors is [batch_size, num_kv_heads, prompt_len, head_dim].
        """
        prompt_len = k_tensors[0].shape[2]
        for layer_idx in range(self.num_layers):
            self.k_buffer[layer_idx, :, :, :prompt_len, :] = k_tensors[layer_idx]
            self.v_buffer[layer_idx, :, :, :prompt_len, :] = v_tensors[layer_idx]
        self.seq_len = prompt_len

    def append_step(self, layer_idx: int, new_k: torch.Tensor, new_v: torch.Tensor) -> None:
        """
        Append single-step new_k, new_v at index self.seq_len.
        new_k, new_v shape: [batch_size, num_kv_heads, 1, head_dim].
        """
        self.k_buffer[layer_idx, :, :, self.seq_len : self.seq_len + 1, :] = new_k
        self.v_buffer[layer_idx, :, :, self.seq_len : self.seq_len + 1, :] = new_v

    def advance(self) -> None:
        self.seq_len += 1

    def to_dynamic_cache(self) -> DynamicCache:
        """
        Convert the active valid slice [:, :, :, :self.seq_len, :] into a DynamicCache
        adapter to pass into HF forward passes without copying storage.
        """
        cache = DynamicCache()
        for layer_idx in range(self.num_layers):
            layer = DynamicLayer()
            layer.keys = self.k_buffer[layer_idx, :, :, : self.seq_len, :]
            layer.values = self.v_buffer[layer_idx, :, :, : self.seq_len, :]
            layer.is_initialized = True
            cache.layers.append(layer)
        return cache


@torch.inference_mode()
def manual_autoregressive_generate(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompt: str,
    max_new_tokens: int = 16,
    max_seq_len: int = 512,
    device: str = "cuda",
) -> List[int]:
    """
    1. Runs prefill and extracts raw past_key_values.
    2. Initializes RawTensorKVCache [num_layers, batch, num_kv_heads, max_seq_len, head_dim].
    3. Runs token-by-token decode loop directly appending new K, V per step.
    """
    dev = torch.device(device if torch.cuda.is_available() else "cpu")
    dtype = torch.float32  # Use fp32 for strict bit-for-bit equivalence check

    config = model.config
    num_layers = getattr(config, "num_hidden_layers", len(config.layers) if hasattr(config, "layers") else 24)
    num_kv_heads = getattr(config, "num_key_value_heads", getattr(config, "num_attention_heads", 16))
    head_dim = getattr(config, "head_dim", config.hidden_size // getattr(config, "num_attention_heads", 16))

    # 1. Tokenize prompt
    enc = tokenizer(prompt, return_tensors="pt").to(dev)
    input_ids = enc.input_ids
    batch_size, prompt_len = input_ids.shape

    # 2. Allocate Raw Tensor KV Cache
    raw_cache = RawTensorKVCache(
        num_layers=num_layers,
        batch_size=batch_size,
        num_kv_heads=num_kv_heads,
        max_seq_len=max_seq_len,
        head_dim=head_dim,
        device=dev,
        dtype=dtype,
    )

    # 3. Prefill step
    out = model(input_ids=input_ids, use_cache=True)
    pkv = out.past_key_values

    # Extract raw K and V tensors from prefill
    prefill_k = [pkv.layers[i].keys for i in range(num_layers)]
    prefill_v = [pkv.layers[i].values for i in range(num_layers)]
    raw_cache.write_prefill(prefill_k, prefill_v)

    # Greedy token from prefill
    next_token = torch.argmax(out.logits[:, -1, :], dim=-1, keepdim=True)
    generated = [next_token.item()]

    # 4. Token-by-token autoregressive decode loop
    attention_mask = torch.ones((batch_size, raw_cache.seq_len + 1), dtype=torch.long, device=dev)

    for step in range(1, max_new_tokens):
        # Package active slice into adapter
        cache_adapter = raw_cache.to_dynamic_cache()

        step_out = model(
            input_ids=next_token,
            attention_mask=attention_mask,
            past_key_values=cache_adapter,
            use_cache=True,
        )

        # 4a. Sample next token
        next_token = torch.argmax(step_out.logits[:, -1, :], dim=-1, keepdim=True)
        tok_id = next_token.item()
        generated.append(tok_id)

        # 4b. Extract new K, V for this single step and write directly to raw tensor buffer
        for layer_idx in range(num_layers):
            layer_k = step_out.past_key_values.layers[layer_idx].keys[:, :, -1:, :]
            layer_v = step_out.past_key_values.layers[layer_idx].values[:, :, -1:, :]
            raw_cache.append_step(layer_idx, layer_k, layer_v)

        raw_cache.advance()
        attention_mask = torch.cat([attention_mask, torch.ones((batch_size, 1), dtype=torch.long, device=dev)], dim=-1)

        if tok_id == tokenizer.eos_token_id:
            break

    return generated


def verify_against_hf_generate(model_name: str = "Qwen/Qwen2.5-0.5B-Instruct", prompt: str = "The capital of France is", max_new_tokens: int = 16) -> bool:
    """
    Bit-for-bit correctness test: manual raw tensor decode vs. model.generate(do_sample=False).
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.float32).to(device)
    model.eval()

    manual_ids = manual_autoregressive_generate(
        model=model,
        tokenizer=tokenizer,
        prompt=prompt,
        max_new_tokens=max_new_tokens,
        device=device,
    )

    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.inference_mode():
        hf_out = model.generate(
            inputs.input_ids,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            generation_config=None,
            repetition_penalty=1.0,
            temperature=None,
            top_p=None,
            top_k=None,
            pad_token_id=tokenizer.eos_token_id,
        )
    hf_ids = hf_out[0, inputs.input_ids.shape[1] :].tolist()

    match = manual_ids[: len(hf_ids)] == hf_ids
    print(f"\nPrompt: {repr(prompt)}")
    print(f"Manual decode tokens: {manual_ids}")
    print(f"HF generate tokens:   {hf_ids}")
    print(f"Decoded text: {repr(tokenizer.decode(manual_ids, skip_special_tokens=True))}")
    print("STATUS:", "MATCH (PASSED)" if match else "MISMATCH (FAILED)")
    return match


if __name__ == "__main__":
    verify_against_hf_generate()
