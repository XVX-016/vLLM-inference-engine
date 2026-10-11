"""
Triton Paged Attention Forward Integration for Qwen2 / Transformers models.

Implements custom attention forward logic that plugs Triton paged attention kernels
into Hugging Face decoder layers for decode steps.
"""
from __future__ import annotations

import math
from contextlib import contextmanager
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from transformers.models.qwen2.modeling_qwen2 import apply_rotary_pos_emb

from engine.kernels.paged_attention import (
    pad_block_tables,
    paged_attention_decode_batched,
    paged_attention_decode_v2,
)
from engine.memory.block_allocator import BlockAllocator, BlockTable, PhysicalBlockPool


class PagedAttentionContext:
    """
    Context holder passed to patched attention modules during decode steps.
    """

    def __init__(
        self,
        pool: PhysicalBlockPool,
        block_tables: Union[BlockTable, List[BlockTable]],
        seq_lens: Union[int, List[int], torch.Tensor],
        block_size: int = 16,
        is_decode: bool = True,
    ):
        self.pool = pool
        self.block_size = block_size
        self.is_decode = is_decode

        if isinstance(block_tables, BlockTable):
            self.is_batched = False
            self.single_block_table = block_tables
            self.single_seq_len = int(seq_lens)
            self.block_table_tensor = torch.tensor(
                block_tables.physical_block_ids,
                dtype=torch.int32,
                device=pool.device,
            )
        else:
            self.is_batched = True
            self.block_table_list = block_tables
            padded_bt, seq_lens_t = pad_block_tables(block_tables, seq_lens, device=pool.device)
            self.padded_block_tables = padded_bt
            self.seq_lens_tensor = seq_lens_t


def paged_decode_attention_forward(
    self: nn.Module,
    hidden_states: torch.Tensor,
    position_embeddings: Tuple[torch.Tensor, torch.Tensor],
    attention_mask: Optional[torch.Tensor] = None,
    past_key_values: Any = None,
    **kwargs: Any,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """
    Drop-in replacement for Qwen2Attention.forward during decode step.
    
    1. Computes q, k, v projections using model module.
    2. Applies rotary embeddings (apply_rotary_pos_emb).
    3. Writes new k, v into physical pool at the token's (physical_block_id, offset).
    4. Computes paged attention using Triton kernel across all logical blocks up to seq_len.
    5. Applies output projection (o_proj).
    """
    ctx: Optional[PagedAttentionContext] = getattr(self, "_paged_attention_context", None)

    # If context is not active or not in decode mode, fallback to original forward
    if ctx is None or not ctx.is_decode:
        orig_forward = getattr(self, "_orig_attention_forward", None)
        if orig_forward is not None:
            return orig_forward(
                hidden_states=hidden_states,
                position_embeddings=position_embeddings,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                **kwargs,
            )
        raise RuntimeError("No fallback forward method found on attention module.")

    layer_idx = self.layer_idx
    input_shape = hidden_states.shape[:-1]  # [batch_size, q_len=1]
    batch_size = input_shape[0]
    hidden_shape = (*input_shape, -1, self.head_dim)

    # 1. Projections
    query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
    key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
    value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

    # 2. Rotary embeddings
    cos, sin = position_embeddings
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)
    # query_states: [batch_size, num_q_heads, 1, head_dim]
    # key_states:   [batch_size, num_kv_heads, 1, head_dim]
    # value_states: [batch_size, num_kv_heads, 1, head_dim]

    scale = self.scaling if hasattr(self, "scaling") else 1.0 / math.sqrt(self.head_dim)

    # 3. Write new key and value into PhysicalBlockPool
    pool = ctx.pool
    block_size = ctx.block_size

    if not ctx.is_batched:
        # Single sequence decode
        seq_len = ctx.single_seq_len  # total length including current new token
        token_pos = seq_len - 1       # 0-indexed position for current token
        logical_block_idx = token_pos // block_size
        physical_block_id = ctx.single_block_table.physical_block_ids[logical_block_idx]
        offset = token_pos % block_size

        # key_states: [1, num_kv_heads, 1, head_dim] -> [num_kv_heads, head_dim]
        k_val = key_states.squeeze(0).squeeze(1)
        v_val = value_states.squeeze(0).squeeze(1)
        pool.k_cache[layer_idx][physical_block_id, offset] = k_val
        pool.v_cache[layer_idx][physical_block_id, offset] = v_val

        # 4. Attention kernel: q is [num_q_heads, head_dim]
        q_2d = query_states.squeeze(0).squeeze(1)  # [num_q_heads, head_dim]
        attn_output = paged_attention_decode_v2(
            q=q_2d,
            k_cache=pool.k_cache[layer_idx],
            v_cache=pool.v_cache[layer_idx],
            block_table=ctx.block_table_tensor,
            seq_len=seq_len,
            scale=scale,
        )  # [num_q_heads, head_dim]

        # Reshape to [1, 1, num_q_heads * head_dim]
        attn_output = attn_output.unsqueeze(0).unsqueeze(0)  # [1, 1, num_q_heads, head_dim]
        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
    else:
        # Batched decode
        for b in range(batch_size):
            cur_seq_len = int(ctx.seq_lens_tensor[b].item())
            cur_pos = cur_seq_len - 1
            logical_block_idx = cur_pos // block_size
            phys_id = ctx.block_table_list[b].physical_block_ids[logical_block_idx]
            offset = cur_pos % block_size

            k_val = key_states[b].squeeze(1)  # [num_kv_heads, head_dim]
            v_val = value_states[b].squeeze(1)  # [num_kv_heads, head_dim]
            pool.k_cache[layer_idx][phys_id, offset] = k_val
            pool.v_cache[layer_idx][phys_id, offset] = v_val

        # q_3d: [batch_size, num_q_heads, head_dim]
        q_3d = query_states.squeeze(2)
        attn_output = paged_attention_decode_batched(
            q=q_3d,
            k_cache=pool.k_cache[layer_idx],
            v_cache=pool.v_cache[layer_idx],
            block_tables=ctx.padded_block_tables,
            seq_lens=ctx.seq_lens_tensor,
            scale=scale,
        )  # [batch_size, num_q_heads, head_dim]
        attn_output = attn_output.reshape(*input_shape, -1).contiguous()

    # 5. Output projection
    attn_output = self.o_proj(attn_output)
    return attn_output, None


@contextmanager
def patch_paged_attention(
    model: nn.Module,
    context: PagedAttentionContext,
):
    """
    Context manager that temporarily patches all decoder attention modules
    with paged_decode_attention_forward.
    """
    attn_modules: List[Tuple[nn.Module, Callable]] = []

    # Find all self_attn modules in model
    for module in model.modules():
        if hasattr(module, "q_proj") and hasattr(module, "k_proj") and hasattr(module, "v_proj") and hasattr(module, "o_proj"):
            orig_fwd = module.forward
            module._orig_attention_forward = orig_fwd
            module._paged_attention_context = context
            # Bind custom forward
            module.forward = paged_decode_attention_forward.__get__(module, module.__class__)
            attn_modules.append((module, orig_fwd))

    try:
        yield context
    finally:
        for module, orig_fwd in attn_modules:
            module.forward = orig_fwd
            if hasattr(module, "_orig_attention_forward"):
                delattr(module, "_orig_attention_forward")
            if hasattr(module, "_paged_attention_context"):
                delattr(module, "_paged_attention_context")


class KernelPagedManualDecode:
    """
    Paged manual decode engine with Triton paged attention kernel integration.
    
    Prefill: runs model forward to compute prompt representations and K/V,
    then copies prefill K/V into PhysicalBlockPool.
    
    Decode: runs decoder forward with custom paged attention kernel patched into
    each layer's self_attn, directly writing new K/V and computing paged attention.
    """

    def __init__(
        self,
        model: nn.Module,
        tokenizer: Any,
        block_size: int = 16,
        num_blocks: int = 1024,
        device: str = "cuda",
        dtype: torch.dtype = torch.float16,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.block_size = block_size
        self.num_blocks = num_blocks
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.dtype = dtype

        config = model.config
        self.num_layers = getattr(config, "num_hidden_layers", 24)
        self.num_kv_heads = getattr(config, "num_key_value_heads", 2)
        self.head_dim = getattr(config, "head_dim", 64)

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

    def _copy_prefill_kv(
        self,
        pkv: Any,
        block_table: BlockTable,
        prompt_len: int,
        alloc: BlockAllocator,
    ) -> None:
        """
        Allocates blocks for the prefill tokens and copies prefill K/V
        into the PhysicalBlockPool.
        """
        for t in range(prompt_len):
            if not block_table.last_block_has_space():
                new_block_id = alloc.allocate()
                block_table.physical_block_ids.append(new_block_id)
            block_table.num_tokens += 1

            logical_block_idx = t // self.block_size
            physical_block_id = block_table.physical_block_ids[logical_block_idx]
            offset = t % self.block_size

            for layer_idx in range(self.num_layers):
                # HF DynamicCache or tuple format
                if hasattr(pkv, "layers"):
                    k_tok = pkv.layers[layer_idx].keys[:, :, t, :]  # [1, num_kv_heads, head_dim]
                    v_tok = pkv.layers[layer_idx].values[:, :, t, :]
                else:
                    k_tok = pkv[layer_idx][0][:, :, t, :]
                    v_tok = pkv[layer_idx][1][:, :, t, :]

                self.pool.k_cache[layer_idx][physical_block_id, offset] = k_tok.squeeze(0).to(self.dtype)
                self.pool.v_cache[layer_idx][physical_block_id, offset] = v_tok.squeeze(0).to(self.dtype)

    @torch.inference_mode()
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 16,
        allocator: Optional[BlockAllocator] = None,
        block_table: Optional[BlockTable] = None,
        return_logits_history: bool = False,
    ) -> Union[List[int], Tuple[List[int], List[torch.Tensor]]]:
        """
        Autoregressive generation using Triton paged attention kernel in decode steps.
        """
        alloc = allocator or self.allocator

        # 1. Tokenize prompt
        enc = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        input_ids = enc.input_ids
        batch_size, prompt_len = input_ids.shape
        if batch_size != 1:
            raise ValueError("KernelPagedManualDecode currently supports single-sequence decode (batch_size=1).")

        if block_table is None:
            block_table = BlockTable(seq_id=0, block_size=self.block_size)

        # 2. Prefill step via HF forward
        prefill_out = self.model(input_ids=input_ids, use_cache=True)
        pkv = prefill_out.past_key_values
        self._copy_prefill_kv(pkv, block_table, prompt_len, alloc)

        next_token = torch.argmax(prefill_out.logits[:, -1, :], dim=-1, keepdim=True)
        generated = [next_token.item()]
        logits_history: List[torch.Tensor] = [prefill_out.logits[:, -1, :]]

        # 3. Decode loop
        seq_len = prompt_len

        for step in range(1, max_new_tokens):
            # Advance sequence length to include current position
            seq_len += 1
            if not block_table.last_block_has_space():
                new_block_id = alloc.allocate()
                block_table.physical_block_ids.append(new_block_id)
            block_table.num_tokens += 1

            # Prepare PagedAttentionContext for this decode step
            ctx = PagedAttentionContext(
                pool=self.pool,
                block_tables=block_table,
                seq_lens=seq_len,
                block_size=self.block_size,
                is_decode=True,
            )

            # Position id for current decode token is seq_len - 1
            cur_pos_id = torch.tensor([[seq_len - 1]], dtype=torch.long, device=self.device)

            with patch_paged_attention(self.model, ctx):
                step_out = self.model(
                    input_ids=next_token,
                    position_ids=cur_pos_id,
                    use_cache=False,
                )

            step_logits = step_out.logits[:, -1, :]
            logits_history.append(step_logits)

            next_token = torch.argmax(step_logits, dim=-1, keepdim=True)
            tok_id = next_token.item()
            generated.append(tok_id)

        if return_logits_history:
            return generated, logits_history
        return generated
