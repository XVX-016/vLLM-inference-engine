"""
Correctness test for the Phase 3 naive reference implementation: gathering
K/V through a block table (even a randomly-permuted, non-contiguous one)
must produce identical output to standard attention over the equivalent
un-paged K/V tensor.

This test runs on CPU and needs no GPU/Triton — run it FIRST, before
writing any Triton code, to lock in the ground truth the real kernel will
be checked against.

    pytest tests/test_paged_attention_reference.py -v
"""
import random

import torch

from engine.kernels.paged_attention import naive_reference_paged_attention


def standard_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, scale: float) -> torch.Tensor:
    """q: [num_seqs, num_heads, head_dim], k/v: [num_seqs, seq_len, num_heads, head_dim]"""
    scores = torch.einsum("nhd,nshd->nhs", q, k) * scale
    probs = torch.softmax(scores, dim=-1)
    return torch.einsum("nhs,nshd->nhd", probs, v)


def test_paged_matches_standard_attention_contiguous_blocks():
    torch.manual_seed(0)
    num_heads, head_dim = 4, 16
    block_size = 8
    seq_len = 24  # exactly 3 blocks, contiguous case
    num_blocks = seq_len // block_size

    q = torch.randn(1, num_heads, head_dim)
    k_flat = torch.randn(1, seq_len, num_heads, head_dim)
    v_flat = torch.randn(1, seq_len, num_heads, head_dim)

    # lay k_flat/v_flat out into blocks in natural order
    k_cache = k_flat.view(num_blocks, block_size, num_heads, head_dim)
    v_cache = v_flat.view(num_blocks, block_size, num_heads, head_dim)
    block_table = [list(range(num_blocks))]

    scale = head_dim ** -0.5
    expected = standard_attention(q, k_flat, v_flat, scale)
    actual = naive_reference_paged_attention(q, k_cache, v_cache, block_table, [seq_len], scale)

    torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)


def test_paged_matches_standard_attention_shuffled_blocks():
    """
    The whole point of paging: physical blocks need NOT be contiguous or in
    logical order. Shuffle physical block placement and verify the block
    table correctly reconstructs the same attention result regardless.
    """
    torch.manual_seed(1)
    num_heads, head_dim = 2, 8
    block_size = 4
    seq_len = 20  # 5 blocks
    num_logical_blocks = seq_len // block_size
    num_physical_blocks = num_logical_blocks * 3  # extra capacity, like a real pool

    q = torch.randn(1, num_heads, head_dim)
    k_flat = torch.randn(1, seq_len, num_heads, head_dim)
    v_flat = torch.randn(1, seq_len, num_heads, head_dim)

    k_cache = torch.zeros(num_physical_blocks, block_size, num_heads, head_dim)
    v_cache = torch.zeros(num_physical_blocks, block_size, num_heads, head_dim)

    physical_ids = random.sample(range(num_physical_blocks), num_logical_blocks)
    for logical_idx, physical_idx in enumerate(physical_ids):
        start, end = logical_idx * block_size, (logical_idx + 1) * block_size
        k_cache[physical_idx] = k_flat[0, start:end]
        v_cache[physical_idx] = v_flat[0, start:end]

    block_table = [physical_ids]
    scale = head_dim ** -0.5

    expected = standard_attention(q, k_flat, v_flat, scale)
    actual = naive_reference_paged_attention(q, k_cache, v_cache, block_table, [seq_len], scale)

    torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)


def test_paged_matches_standard_attention_multi_sequence_variable_length():
    """Multiple sequences, different lengths, different (shuffled) block layouts at once."""
    torch.manual_seed(2)
    num_heads, head_dim = 2, 8
    block_size = 4
    seq_lens = [12, 20, 8]  # 3, 5, 2 blocks respectively
    num_physical_blocks = 30

    k_cache = torch.zeros(num_physical_blocks, block_size, num_heads, head_dim)
    v_cache = torch.zeros(num_physical_blocks, block_size, num_heads, head_dim)

    q_list, k_flats, v_flats, block_tables = [], [], [], []
    used_physical = set()

    for seq_len in seq_lens:
        q_list.append(torch.randn(num_heads, head_dim))
        k_flat = torch.randn(seq_len, num_heads, head_dim)
        v_flat = torch.randn(seq_len, num_heads, head_dim)
        k_flats.append(k_flat)
        v_flats.append(v_flat)

        num_blocks = seq_len // block_size
        available = [i for i in range(num_physical_blocks) if i not in used_physical]
        physical_ids = random.sample(available, num_blocks)
        used_physical.update(physical_ids)

        for logical_idx, physical_idx in enumerate(physical_ids):
            start, end = logical_idx * block_size, (logical_idx + 1) * block_size
            k_cache[physical_idx] = k_flat[start:end]
            v_cache[physical_idx] = v_flat[start:end]

        block_tables.append(physical_ids)

    q = torch.stack(q_list)  # [num_seqs, num_heads, head_dim]
    scale = head_dim ** -0.5

    actual = naive_reference_paged_attention(q, k_cache, v_cache, block_tables, seq_lens, scale)

    for i, seq_len in enumerate(seq_lens):
        expected_i = standard_attention(
            q[i : i + 1], k_flats[i].unsqueeze(0), v_flats[i].unsqueeze(0), scale
        )
        torch.testing.assert_close(actual[i : i + 1], expected_i, atol=1e-5, rtol=1e-5)
