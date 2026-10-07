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

from engine.memory.block_allocator import BlockAllocator, BlockTable, PhysicalBlockPool


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


# Alias matching project plan nomenclature
manual_generate = manual_autoregressive_generate


class PagedManualDecode:
    """
    Manual decode engine using PhysicalBlockPool + BlockTable + BlockAllocator (Phase 2a).
    Mirrors manual_autoregressive_generate() structure, replacing contiguous buffer writes
    with paged block writes.
    """

    def __init__(
        self,
        model: AutoModelForCausalLM,
        tokenizer: AutoTokenizer,
        block_size: int = 16,
        num_blocks: int = 1024,
        device: str = "cuda",
        dtype: torch.dtype = torch.float32,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.block_size = block_size
        self.num_blocks = num_blocks
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.dtype = dtype

        config = model.config
        self.num_layers = getattr(config, "num_hidden_layers", len(config.layers) if hasattr(config, "layers") else 24)
        self.num_kv_heads = getattr(config, "num_key_value_heads", getattr(config, "num_attention_heads", 16))
        self.head_dim = getattr(config, "head_dim", config.hidden_size // getattr(config, "num_attention_heads", 16))

        # Allocate physical block pool and allocator
        self.pool = PhysicalBlockPool(
            num_blocks=self.num_blocks,
            block_size=self.block_size,
            num_layers=self.num_layers,
            num_kv_heads=self.num_kv_heads,
            head_dim=self.head_dim,
            dtype=self.dtype,
            device=self.device,
        )
        self.allocator = BlockAllocator(num_blocks=self.num_blocks)

    def _write_token_kv(
        self,
        block_table: BlockTable,
        token_pos: int,
        layer_idx: int,
        new_k: torch.Tensor,
        new_v: torch.Tensor,
    ) -> None:
        """
        Write new_k, new_v for a single token position into the physical block pool:
        (physical_block_id, offset) = block_table.physical_block_ids[token_pos // block_size], token_pos % block_size
        new_k, new_v shape: [1, num_kv_heads, 1, head_dim] or [num_kv_heads, head_dim]
        """
        logical_block_idx = token_pos // self.block_size
        physical_block_id = block_table.physical_block_ids[logical_block_idx]
        offset = token_pos % self.block_size

        # k_cache/v_cache shape: [num_blocks, block_size, num_kv_heads, head_dim]
        k_val = new_k.squeeze(0).squeeze(1) if new_k.ndim == 4 else new_k
        v_val = new_v.squeeze(0).squeeze(1) if new_v.ndim == 4 else new_v

        self.pool.k_cache[layer_idx][physical_block_id, offset] = k_val
        self.pool.v_cache[layer_idx][physical_block_id, offset] = v_val

    def _build_cache_adapter(self, block_table: BlockTable, total_tokens: int) -> DynamicCache:
        """
        Gather logical tokens from non-contiguous physical blocks into a DynamicCache
        adapter to pass into Hugging Face forward passes.
        """
        cache = DynamicCache()
        for layer_idx in range(self.num_layers):
            layer = DynamicLayer()
            k_pieces = []
            v_pieces = []

            for logical_idx, phys_id in enumerate(block_table.physical_block_ids):
                start = logical_idx * self.block_size
                num_in_block = min(self.block_size, total_tokens - start)
                if num_in_block <= 0:
                    break
                # shape: [num_in_block, num_kv_heads, head_dim]
                k_piece = self.pool.k_cache[layer_idx][phys_id, :num_in_block]
                v_piece = self.pool.v_cache[layer_idx][phys_id, :num_in_block]
                # transpose to HF shape: [1, num_kv_heads, num_in_block, head_dim]
                k_pieces.append(k_piece.permute(1, 0, 2).unsqueeze(0))
                v_pieces.append(v_piece.permute(1, 0, 2).unsqueeze(0))

            layer.keys = torch.cat(k_pieces, dim=2)
            layer.values = torch.cat(v_pieces, dim=2)
            layer.is_initialized = True
            cache.layers.append(layer)

        return cache

    @torch.inference_mode()
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 16,
        allocator: Optional[BlockAllocator] = None,
        block_table: Optional[BlockTable] = None,
    ) -> List[int]:
        """
        Step-by-step decode loop mirroring manual_autoregressive_generate()
        using PhysicalBlockPool and BlockTable.
        """
        alloc = allocator or self.allocator

        # 1. Tokenize prompt
        enc = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        input_ids = enc.input_ids
        batch_size, prompt_len = input_ids.shape
        if batch_size != 1:
            raise ValueError("PagedManualDecode currently supports single-sequence decode (batch_size=1).")

        if block_table is None:
            block_table = BlockTable(seq_id=0, block_size=self.block_size)

        # 2. Prefill step
        out = self.model(input_ids=input_ids, use_cache=True)
        pkv = out.past_key_values

        # Allocate blocks and store prefill tokens into physical blocks
        for t in range(prompt_len):
            if not block_table.last_block_has_space():
                new_block_id = alloc.allocate()
                block_table.physical_block_ids.append(new_block_id)
            block_table.num_tokens += 1

            for layer_idx in range(self.num_layers):
                k_tok = pkv.layers[layer_idx].keys[:, :, t : t + 1, :]
                v_tok = pkv.layers[layer_idx].values[:, :, t : t + 1, :]
                logical_block_idx = t // self.block_size
                physical_block_id = block_table.physical_block_ids[logical_block_idx]
                offset = t % self.block_size
                self.pool.k_cache[layer_idx][physical_block_id, offset] = k_tok.squeeze(0).squeeze(1)
                self.pool.v_cache[layer_idx][physical_block_id, offset] = v_tok.squeeze(0).squeeze(1)

        # Greedy token from prefill
        next_token = torch.argmax(out.logits[:, -1, :], dim=-1, keepdim=True)
        generated = [next_token.item()]

        # 3. Token-by-token autoregressive decode loop
        seq_len = prompt_len
        attention_mask = torch.ones((1, seq_len + 1), dtype=torch.long, device=self.device)

        for step in range(1, max_new_tokens):
            # Package active paged cache into DynamicCache adapter
            cache_adapter = self._build_cache_adapter(block_table, total_tokens=seq_len)

            step_out = self.model(
                input_ids=next_token,
                attention_mask=attention_mask,
                past_key_values=cache_adapter,
                use_cache=True,
            )

            # 3a. Sample next token
            next_token = torch.argmax(step_out.logits[:, -1, :], dim=-1, keepdim=True)
            tok_id = next_token.item()
            generated.append(tok_id)

            # 3b. Before writing new K/V, check if last block has space
            if not block_table.last_block_has_space():
                new_block_id = alloc.allocate()
                block_table.physical_block_ids.append(new_block_id)

            # Compute physical block index and offset for new K/V at seq_len
            physical_block_id = block_table.physical_block_ids[seq_len // self.block_size]
            offset = seq_len % self.block_size

            # Write new K/V into physical block pool
            for layer_idx in range(self.num_layers):
                layer_k = step_out.past_key_values.layers[layer_idx].keys[:, :, -1:, :]
                layer_v = step_out.past_key_values.layers[layer_idx].values[:, :, -1:, :]
                self.pool.k_cache[layer_idx][physical_block_id, offset] = layer_k.squeeze(0).squeeze(1)
                self.pool.v_cache[layer_idx][physical_block_id, offset] = layer_v.squeeze(0).squeeze(1)

            block_table.num_tokens += 1
            seq_len += 1
            attention_mask = torch.cat([attention_mask, torch.ones((1, 1), dtype=torch.long, device=self.device)], dim=-1)

            if tok_id == self.tokenizer.eos_token_id:
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
            pad_token_id=tokenizer.eos_token_id,
        )
    hf_ids = hf_out[0, inputs.input_ids.shape[1] :].tolist()

    paged_decoder = PagedManualDecode(
        model=model,
        tokenizer=tokenizer,
        block_size=16,
        num_blocks=1024,
        device=device,
        dtype=torch.float32,
    )
    paged_ids = paged_decoder.generate(prompt=prompt, max_new_tokens=max_new_tokens)

    match_raw = manual_ids[: len(hf_ids)] == hf_ids
    match_paged = paged_ids[: len(hf_ids)] == hf_ids
    print(f"\nPrompt: {repr(prompt)}")
    print(f"HF generate tokens:   {hf_ids}")
    print(f"Raw manual tokens:    {manual_ids}")
    print(f"Paged manual tokens:  {paged_ids}")
    print(f"Decoded text: {repr(tokenizer.decode(paged_ids, skip_special_tokens=True))}")
    print("STATUS (Raw vs HF):", "MATCH (PASSED)" if match_raw else "MISMATCH (FAILED)")
    print("STATUS (Paged vs HF):", "MATCH (PASSED)" if match_paged else "MISMATCH (FAILED)")
    return match_paged


if __name__ == "__main__":
    verify_against_hf_generate()
