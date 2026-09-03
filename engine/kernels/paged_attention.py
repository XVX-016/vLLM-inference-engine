"""
Phase 3 — Paged-attention Triton kernel. STUB.

Implement this using the Phase 3 prompt. Interface contract below —
`paged_attention_decode` is what the engine's forward pass will call once
this is real; keep the signature stable so Phase 4's scheduler and the
serving layer don't need to change when the kernel implementation matures.

Build order (do not skip steps):
  1. Naive PyTorch reference implementation (gather blocks with indexing,
     plain softmax attention). This is your ground truth — write it first.
  2. Triton kernel for SINGLE-sequence decode attention against paged KV.
  3. Randomized correctness test: kernel output vs. reference, across
     random block-table layouts (non-contiguous physical blocks).
  4. Extend to BATCHED multi-sequence decode (this is the actual payoff —
     one kernel launch serving many sequences with different block tables).
  5. Online (flash-attention-style) softmax so you never materialize the
     full [seq_len] score row when sequences are long.

import triton
import triton.language as tl
"""
from __future__ import annotations

import torch


def naive_reference_paged_attention(
    q: torch.Tensor,            # [num_seqs, num_heads, head_dim]  (single decode step)
    k_cache: torch.Tensor,      # [num_blocks, block_size, num_kv_heads, head_dim]
    v_cache: torch.Tensor,      # [num_blocks, block_size, num_kv_heads, head_dim]
    block_tables: list[list[int]],  # per-sequence physical block indices, in logical order
    seq_lens: list[int],
    scale: float,
) -> torch.Tensor:
    """
    Ground-truth reference: gather K/V per sequence via plain indexing,
    run standard softmax attention. O(seq_len) memory per sequence — fine
    for correctness testing, not for production. Every kernel change in
    this file must be validated against this function before it's trusted.
    """
    num_seqs, num_heads, head_dim = q.shape
    outputs = torch.zeros_like(q)

    for s in range(num_seqs):
        block_size = k_cache.shape[1]
        seq_len = seq_lens[s]
        block_ids = block_tables[s]

        k_parts, v_parts = [], []
        remaining = seq_len
        for b in block_ids:
            take = min(block_size, remaining)
            k_parts.append(k_cache[b, :take])   # [take, num_kv_heads, head_dim]
            v_parts.append(v_cache[b, :take])
            remaining -= take
            if remaining <= 0:
                break

        k = torch.cat(k_parts, dim=0)  # [seq_len, num_kv_heads, head_dim]
        v = torch.cat(v_parts, dim=0)

        # assume num_kv_heads == num_heads for this reference (extend for GQA/MQA later)
        q_s = q[s]  # [num_heads, head_dim]
        scores = torch.einsum("hd,skd->hsk".replace("k", ""), q_s, k.transpose(0, 1)) if False else None
        # simpler + explicit:
        scores = torch.einsum("hd,shd->hs", q_s, k) * scale  # [num_heads, seq_len]
        probs = torch.softmax(scores, dim=-1)
        out = torch.einsum("hs,shd->hd", probs, v)  # [num_heads, head_dim]
        outputs[s] = out

    return outputs


def paged_attention_decode(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    block_tables: list[list[int]],
    seq_lens: list[int],
    scale: float,
) -> torch.Tensor:
    """
    TODO(Phase 3): Triton kernel implementation. Until implemented, falls
    back to the (slow, correctness-only) naive reference so the rest of the
    engine can be wired up and tested end-to-end before the kernel exists.
    """
    return naive_reference_paged_attention(q, k_cache, v_cache, block_tables, seq_lens, scale)
