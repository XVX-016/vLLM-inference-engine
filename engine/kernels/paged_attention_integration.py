"""
Triton Paged Attention Forward Integration for Qwen2 / Transformers models.

Implements custom attention forward logic that plugs Triton paged attention kernels
into Hugging Face decoder layers for decode steps.
"""
from __future__ import annotations

import math
import logging
from contextlib import contextmanager
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple, Union

import torch
import torch.nn as nn
from transformers.models.qwen2.modeling_qwen2 import apply_rotary_pos_emb
from transformers.cache_utils import DynamicCache, DynamicLayer

from engine.kernels.paged_attention import (
    pad_block_tables,
    paged_attention_decode_batched,
    paged_attention_decode_v2,
)
from engine.memory.block_allocator import BlockAllocator, BlockTable, PhysicalBlockPool

logger = logging.getLogger(__name__)


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
            self.padded_block_tables = pad_block_tables(block_tables, device=pool.device)
            if isinstance(seq_lens, torch.Tensor):
                self.seq_lens_tensor = seq_lens.to(device=pool.device, dtype=torch.int32)
            else:
                self.seq_lens_tensor = torch.tensor(seq_lens, dtype=torch.int32, device=pool.device)


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

        # 4. Attention kernel: use promoted autotuned batched kernel
        # q_3d: [1, num_q_heads, head_dim]
        q_3d = query_states.squeeze(2)
        bt_2d = ctx.block_table_tensor.unsqueeze(0)
        seq_lens_t = torch.tensor([seq_len], dtype=torch.int32, device=pool.device)
        attn_output = paged_attention_decode_batched(
            q=q_3d,
            k_cache=pool.k_cache[layer_idx],
            v_cache=pool.v_cache[layer_idx],
            block_tables=bt_2d,
            seq_lens=seq_lens_t,
            scale=scale,
        )  # [1, num_q_heads, head_dim]
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
    then copies owned K/V into PhysicalBlockPool. Prefix mode "full" seeds HF
    prefill with the cached prefix through a DynamicCache.
    
    Decode: runs decoder forward with custom paged attention kernel patched into
    each layer's self_attn, directly writing new K/V and computing paged attention.
    """

    def __init__(
        self,
        model: nn.Module,
        tokenizer: Any,
        block_size: int = 16,
        num_blocks: Optional[int] = None,
        gpu_memory_utilization: float = 0.65,
        device: str = "cuda",
        dtype: torch.dtype = torch.float16,
        debug_prefix_cache: bool = False,
        prefix_mode: Literal["off", "share", "full"] = "off",
    ):
        if prefix_mode not in ("off", "share", "full"):
            raise ValueError(
                f"Unsupported prefix_mode {prefix_mode!r}; expected 'off', 'share', or 'full'."
            )
        self.model = model
        self.tokenizer = tokenizer
        self.block_size = block_size
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.dtype = dtype
        self.debug_prefix_cache = debug_prefix_cache
        self.prefix_mode = prefix_mode

        config = model.config
        self.num_layers = getattr(config, "num_hidden_layers", 24)
        self.num_kv_heads = getattr(config, "num_key_value_heads", 2)
        self.head_dim = getattr(config, "head_dim", 64)

        if num_blocks is None:
            if self.device.type == "cuda":
                from engine.memory.block_allocator import compute_num_blocks
                self.num_blocks = compute_num_blocks(
                    model_config=config,
                    block_size=self.block_size,
                    gpu_memory_utilization=gpu_memory_utilization,
                    device=self.device,
                    dtype=self.dtype,
                )
            else:
                self.num_blocks = 1024
        else:
            self.num_blocks = num_blocks

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
        skip_blocks: int = 0,
    ) -> None:
        """
        Copies prefill K/V tensors into the PhysicalBlockPool using vectorized index_copy_.

        The scheduler has already allocated physical blocks and set
        block_table.num_tokens = prompt_len before calling prefill().
        This method ONLY writes tensor data — it never calls allocate().
        """
        start_token = skip_blocks * self.block_size
        if start_token >= prompt_len:
            return

        total_blocks = (prompt_len + self.block_size - 1) // self.block_size
        blocks_to_write = total_blocks - skip_blocks
        if blocks_to_write <= 0:
            return

        phys_block_ids = torch.tensor(
            block_table.physical_block_ids[skip_blocks:total_blocks],
            dtype=torch.long,
            device=self.device,
        )

        tokens_to_copy = prompt_len - start_token
        pad_tokens = blocks_to_write * self.block_size - tokens_to_copy

        for layer_idx in range(self.num_layers):
            if hasattr(pkv, "layers"):
                k_src = pkv.layers[layer_idx].keys[0, :, start_token:prompt_len, :].permute(1, 0, 2)
                v_src = pkv.layers[layer_idx].values[0, :, start_token:prompt_len, :].permute(1, 0, 2)
            else:
                k_src = pkv[layer_idx][0][0, :, start_token:prompt_len, :].permute(1, 0, 2)
                v_src = pkv[layer_idx][1][0, :, start_token:prompt_len, :].permute(1, 0, 2)

            if pad_tokens > 0:
                k_src = torch.nn.functional.pad(k_src, (0, 0, 0, 0, 0, pad_tokens))
                v_src = torch.nn.functional.pad(v_src, (0, 0, 0, 0, 0, pad_tokens))

            k_blocks = k_src.view(blocks_to_write, self.block_size, self.num_kv_heads, self.head_dim).to(self.dtype)
            v_blocks = v_src.view(blocks_to_write, self.block_size, self.num_kv_heads, self.head_dim).to(self.dtype)

            self.pool.k_cache[layer_idx].index_copy_(0, phys_block_ids, k_blocks)
            self.pool.v_cache[layer_idx].index_copy_(0, phys_block_ids, v_blocks)

    def _compare_cached_prefill_kv(
        self,
        pkv: Any,
        block_table: BlockTable,
        cached_tokens: int,
    ) -> None:
        for layer_idx in range(self.num_layers):
            max_difference = 0.0
            for token_idx in range(cached_tokens):
                block_id = block_table.physical_block_ids[token_idx // self.block_size]
                offset = token_idx % self.block_size
                if hasattr(pkv, "layers"):
                    fresh_key = pkv.layers[layer_idx].keys[:, :, token_idx, :]
                    fresh_value = pkv.layers[layer_idx].values[:, :, token_idx, :]
                else:
                    fresh_key = pkv[layer_idx][0][:, :, token_idx, :]
                    fresh_value = pkv[layer_idx][1][:, :, token_idx, :]

                stored_key = self.pool.k_cache[layer_idx][block_id, offset].unsqueeze(0)
                stored_value = self.pool.v_cache[layer_idx][block_id, offset].unsqueeze(0)
                key_difference = (
                    fresh_key.squeeze(0).to(self.dtype).float() - stored_key.float()
                ).abs().max().item()
                value_difference = (
                    fresh_value.squeeze(0).to(self.dtype).float() - stored_value.float()
                ).abs().max().item()
                max_difference = max(max_difference, key_difference, value_difference)

            logger.info(
                "Prefix-cache KV comparison layer=%d max_abs_difference=%g "
                "(cached_tokens=%d)",
                layer_idx,
                max_difference,
                cached_tokens,
            )

    def _build_cached_prefix(self, block_table: BlockTable, cached_tokens: int) -> DynamicCache:
        cache = DynamicCache()
        num_blocks = cached_tokens // self.block_size
        for layer_idx in range(self.num_layers):
            key_blocks = []
            value_blocks = []
            for block_idx in range(num_blocks):
                block_id = block_table.physical_block_ids[block_idx]
                key_blocks.append(self.pool.k_cache[layer_idx][block_id])
                value_blocks.append(self.pool.v_cache[layer_idx][block_id])

            key = (
                torch.cat(key_blocks, dim=0)
                .permute(1, 0, 2)
                .unsqueeze(0)
                .contiguous()
            )
            value = (
                torch.cat(value_blocks, dim=0)
                .permute(1, 0, 2)
                .unsqueeze(0)
                .contiguous()
            )
            layer = DynamicLayer()
            layer.keys = key
            layer.values = value
            layer.is_initialized = True
            cache.layers.append(layer)
        return cache

    @torch.inference_mode()
    def prefill(
        self,
        seq: Any,
        allocator: Optional[BlockAllocator] = None,
    ) -> int:
        """
        Single-sequence Hugging Face prefill. In "full" mode, cached K/V blocks
        seed a DynamicCache and only the uncached prompt suffix is forwarded.
        Other modes forward the full prompt and copy only blocks owned by this
        sequence.

        CONTRACT: The scheduler has already allocated seq.block_table.physical_block_ids
        and set seq.block_table.num_tokens = prompt_len before calling this.
        This method MUST NOT call allocator.allocate() or mutate num_tokens.

        Args:
            seq: Sequence object with block_table already populated by the scheduler.
            allocator: Ignored (kept for API compatibility only).

        Returns:
            first_token: The first generated token ID (argmax of prefill logits).
        """
        # 1. Resolve prompt tokens
        if hasattr(seq, "prompt_token_ids") and seq.prompt_token_ids:
            input_ids = torch.tensor([seq.prompt_token_ids], dtype=torch.long, device=self.device)
        elif hasattr(seq, "prompt") and seq.prompt:
            enc = self.tokenizer(seq.prompt, return_tensors="pt").to(self.device)
            input_ids = enc.input_ids
            if hasattr(seq, "prompt_token_ids") and not seq.prompt_token_ids:
                seq.prompt_token_ids = input_ids[0].tolist()
        elif isinstance(seq, dict) and "prompt_token_ids" in seq and seq["prompt_token_ids"]:
            input_ids = torch.tensor([seq["prompt_token_ids"]], dtype=torch.long, device=self.device)
        elif isinstance(seq, dict) and "prompt" in seq and seq["prompt"]:
            enc = self.tokenizer(seq["prompt"], return_tensors="pt").to(self.device)
            input_ids = enc.input_ids
        else:
            raise ValueError(f"Unable to extract prompt or prompt_token_ids from sequence: {seq}")

        prompt_len = input_ids.shape[1]

        # 2. Resolve block table (must already be set by the scheduler)
        if hasattr(seq, "block_table") and seq.block_table is not None:
            block_table = seq.block_table
        elif isinstance(seq, dict) and seq.get("block_table") is not None:
            block_table = seq["block_table"]
        else:
            raise RuntimeError(
                "prefill() requires seq.block_table to be pre-populated by the scheduler. "
                "The scheduler must call step() before prefill()."
            )

        cached_tokens = (
            getattr(seq, "num_cached_tokens", 0)
            if not isinstance(seq, dict)
            else seq.get("num_cached_tokens", 0)
        )
        if cached_tokens < 0 or cached_tokens > prompt_len or cached_tokens % self.block_size:
            raise ValueError(
                f"num_cached_tokens={cached_tokens} must be a full-block prefix within "
                f"prompt_len={prompt_len} (block_size={self.block_size})."
            )
        if cached_tokens // self.block_size > len(block_table.physical_block_ids):
            raise ValueError("num_cached_tokens exceeds the number of blocks in the sequence table.")

        if self.prefix_mode == "full" and cached_tokens:
            cached_prefix = self._build_cached_prefix(block_table, cached_tokens)
            suffix_ids = input_ids[:, cached_tokens:]
            position_ids = torch.arange(
                cached_tokens,
                prompt_len,
                dtype=torch.long,
                device=self.device,
            ).unsqueeze(0)
            attention_mask = torch.ones(
                (input_ids.shape[0], prompt_len),
                dtype=torch.long,
                device=self.device,
            )
            prefill_out = self.model(
                input_ids=suffix_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=cached_prefix,
                use_cache=True,
                logits_to_keep=1,
            )
            pkv = prefill_out.past_key_values
            self._copy_prefill_kv(pkv, block_table, prompt_len, skip_blocks=cached_tokens // self.block_size)
        else:
            # 3. HF prefill forward pass; "share" mode recomputes the prompt and
            # avoids overwriting cache-hit blocks in the physical pool.
            prefill_out = self.model(input_ids=input_ids, use_cache=True, logits_to_keep=1)
            pkv = prefill_out.past_key_values

            # Write K/V data into the blocks owned exclusively by this sequence.
            if self.debug_prefix_cache and cached_tokens:
                self._compare_cached_prefill_kv(pkv, block_table, cached_tokens)
            self._copy_prefill_kv(
                pkv,
                block_table,
                prompt_len,
                skip_blocks=cached_tokens // self.block_size,
            )

        logits = prefill_out.logits[:, -1, :]
        next_token_id = int(torch.argmax(logits, dim=-1).item())

        # Store logits for diagnostics.
        if hasattr(seq, "last_logits"):
            seq.last_logits = logits[0].detach().cpu()
            top2 = torch.topk(logits[0], k=2).values
            seq.last_logit_margin = float((top2[0] - top2[1]).item())
        elif isinstance(seq, dict):
            seq["last_logits"] = logits[0].detach().cpu()
            top2 = torch.topk(logits[0], k=2).values
            seq["last_logit_margin"] = float((top2[0] - top2[1]).item())

        # Do NOT set last_token_id or append to generated_ids here.
        # Those are the exclusive responsibility of Scheduler.post_step().
        # Do NOT modify cur_len or block_table.num_tokens here.
        # step() already set num_tokens = prompt_len; post_step() will mirror cur_len.
        return next_token_id

    @torch.inference_mode()
    def decode_step(
        self,
        seqs: List[Any],
        allocator: Optional[BlockAllocator] = None,
    ) -> List[int]:
        """
        Batched decode step for N running sequences.

        Args:
            seqs: List of running sequences. Each sequence must provide:
                  - block_table (BlockTable)
                  - current length (cur_len or num_tokens or len(prompt) + len(generated))
                  - last_token_id (or generated_ids[-1])
            allocator: Block allocator to use (defaults to self.allocator).

        Returns:
            next_tokens: List of newly generated token IDs of length N.
        """
        N = len(seqs)
        if N == 0:
            return []
        allocator = allocator or self.allocator

        # ── runtime grad probe (printed once per process) ────────────────────
        if not getattr(self, "_grad_probed", False):
            self._grad_probed = True
            grad_on = torch.is_grad_enabled()
            import logging as _log
            _log.getLogger(__name__).info(
                "[PagedRunner] GRAD PROBE decode_step: torch.is_grad_enabled()=%s  "
                "(expected False inside @inference_mode)", grad_on
            )
            assert not grad_on, (
                "decode_step must run under torch.inference_mode(); "
                "is_grad_enabled()=True means the decorator is missing!"
            )
        # ────────────────────────────────────────────────────────────────────

        #    The Scheduler has already pre-allocated any new block needed for this
        #    step (in step() at kv_len % block_size == 0 boundaries) and has set
        #    bt.num_tokens to the current KV occupancy.
        #    We bump bt.num_tokens += 1 here so the paged attention kernel writes
        #    the new K/V at position (num_tokens - 1) = old_kv_len, which is the
        #    correct slot.  post_step() will mirror cur_len but NOT touch num_tokens.
        seq_lens_list: List[int] = []
        block_tables: List[BlockTable] = []
        last_tokens: List[int] = []

        for seq in seqs:
            # Extract block table
            if hasattr(seq, "block_table"):
                bt = seq.block_table
            elif isinstance(seq, dict):
                bt = seq["block_table"]
            else:
                raise ValueError("Sequence must provide a block_table attribute or dict key.")

            # bt.num_tokens is the authoritative KV occupancy set by step().
            # Bump it by 1 so the kernel writes at the next available slot.
            # IMPORTANT: this is the only place num_tokens is incremented for decode;
            # post_step() mirrors cur_len from it but must NOT increment again.
            write_block_idx = bt.num_tokens // self.block_size
            if write_block_idx >= len(bt.physical_block_ids):
                raise AssertionError(
                    f"Decode write for seq {getattr(seq, 'seq_id', '?')} has no allocated block "
                    f"at logical index {write_block_idx}."
                )
            write_block_id = bt.physical_block_ids[write_block_idx]
            if allocator.ref_counts.get(write_block_id, 0) > 1:
                raise AssertionError(
                    f"Decode write for seq {getattr(seq, 'seq_id', '?')} would modify shared "
                    f"block {write_block_id} with ref_count "
                    f"{allocator.ref_counts[write_block_id]}."
                )
            bt.num_tokens += 1
            new_len = bt.num_tokens  # = old_kv_len + 1

            seq_lens_list.append(new_len)
            block_tables.append(bt)

            # The input token for this decode step is the last generated token.
            if hasattr(seq, "last_token_id") and seq.last_token_id is not None:
                last_tok = seq.last_token_id
            elif hasattr(seq, "generated_ids") and seq.generated_ids:
                last_tok = seq.generated_ids[-1]
            elif isinstance(seq, dict) and "last_token_id" in seq:
                last_tok = seq["last_token_id"]
            else:
                raise ValueError("Sequence must provide last_token_id or non-empty generated_ids.")
            last_tokens.append(last_tok)

        # 2. Build input_ids [N, 1] and per-sequence rotary position_ids [N, 1]
        # Position id for each sequence is its 0-indexed position: new_len - 1
        input_ids = torch.tensor(last_tokens, dtype=torch.long, device=self.device).unsqueeze(1)  # [N, 1]
        position_ids = torch.tensor(
            [[s_len - 1] for s_len in seq_lens_list],
            dtype=torch.long,
            device=self.device,
        )  # [N, 1]

        # 3. Setup batched PagedAttentionContext
        ctx = PagedAttentionContext(
            pool=self.pool,
            block_tables=block_tables,
            seq_lens=seq_lens_list,
            block_size=self.block_size,
            is_decode=True,
        )

        # 4. Model forward pass with patched paged attention
        with patch_paged_attention(self.model, ctx):
            step_out = self.model(
                input_ids=input_ids,
                position_ids=position_ids,
                use_cache=False,
            )

        step_logits = step_out.logits[:, -1, :]  # [N, vocab_size]
        next_tokens_tensor = torch.argmax(step_logits, dim=-1)  # [N]
        next_tokens = next_tokens_tensor.tolist()

        # 5. Store per-sequence logits for diagnostics only.
        #    Do NOT mutate generated_ids / cur_len / last_token_id — that is
        #    Scheduler.post_step()'s exclusive job.
        for i, seq in enumerate(seqs):
            if hasattr(seq, "last_logits"):
                seq.last_logits = step_logits[i].detach().cpu()
                top2 = torch.topk(step_logits[i], k=2).values
                seq.last_logit_margin = float((top2[0] - top2[1]).item())
            elif isinstance(seq, dict):
                seq["last_logits"] = step_logits[i].detach().cpu()
                top2 = torch.topk(step_logits[i], k=2).values
                seq["last_logit_margin"] = float((top2[0] - top2[1]).item())

        return next_tokens

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
        Maintains backward compatibility for single-sequence generation.
        """
        alloc = allocator or self.allocator

        # 1. Tokenize prompt
        enc = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        input_ids = enc.input_ids
        batch_size, prompt_len = input_ids.shape
        if batch_size != 1:
            raise ValueError("KernelPagedManualDecode generate() currently supports single-sequence prompt.")

        if block_table is None:
            block_table = BlockTable(seq_id=0, block_size=self.block_size)

        # Allocate prompt blocks for standalone generate() run
        blocks_needed = math.ceil(prompt_len / self.block_size) if prompt_len > 0 else 0
        while len(block_table.physical_block_ids) < blocks_needed:
            block_table.physical_block_ids.append(alloc.allocate())
        block_table.num_tokens = prompt_len

        # 2. Prefill step via HF forward
        prefill_out = self.model(input_ids=input_ids, use_cache=True)
        pkv = prefill_out.past_key_values
        self._copy_prefill_kv(pkv, block_table, prompt_len)

        next_token = torch.argmax(prefill_out.logits[:, -1, :], dim=-1, keepdim=True)
        generated = [next_token.item()]
        logits_history: List[torch.Tensor] = [prefill_out.logits[:, -1, :]]

        # 3. Decode loop using batched decode step under the hood
        seq_len = prompt_len

        for step in range(1, max_new_tokens):
            seq_len += 1
            if not block_table.last_block_has_space():
                new_block_id = alloc.allocate()
                block_table.physical_block_ids.append(new_block_id)
            block_table.num_tokens += 1

            ctx = PagedAttentionContext(
                pool=self.pool,
                block_tables=[block_table],
                seq_lens=[seq_len],
                block_size=self.block_size,
                is_decode=True,
            )

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
