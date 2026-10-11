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

from engine.kernels.paged_attention import (
    naive_reference_paged_attention,
    pad_block_tables,
    paged_attention_decode_batched,
    paged_attention_decode_v1,
    paged_attention_decode_v2,
)


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


def standard_gqa_attention_repeat_interleave(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    scale: float,
) -> torch.Tensor:
    """
    Independent plain PyTorch check using repeat_interleave on K/V heads.
    q: [num_seqs, num_q_heads, head_dim]
    k/v: [num_seqs, seq_len, num_kv_heads, head_dim]
    """
    num_q_heads = q.shape[1]
    num_kv_heads = k.shape[2]
    assert num_q_heads % num_kv_heads == 0
    group_size = num_q_heads // num_kv_heads

    # repeat_interleave on head dimension (dim=2)
    k_expanded = k.repeat_interleave(group_size, dim=2)  # [num_seqs, seq_len, num_q_heads, head_dim]
    v_expanded = v.repeat_interleave(group_size, dim=2)  # [num_seqs, seq_len, num_q_heads, head_dim]

    scores = torch.einsum("nhd,nshd->nhs", q, k_expanded) * scale
    probs = torch.softmax(scores, dim=-1)
    return torch.einsum("nhs,nshd->nhd", probs, v_expanded)


def _run_paged_gqa_test(
    num_q_heads: int,
    num_kv_heads: int,
    head_dim: int,
    block_size: int,
    seq_lens: list[int],
    seed: int = 42,
):
    torch.manual_seed(seed)
    random.seed(seed)
    num_seqs = len(seq_lens)
    scale = head_dim ** -0.5

    # Determine total physical blocks needed with extra capacity
    total_logical_blocks = sum((s + block_size - 1) // block_size for s in seq_lens)
    num_physical_blocks = total_logical_blocks * 2 + 5

    k_cache = torch.zeros(num_physical_blocks, block_size, num_kv_heads, head_dim)
    v_cache = torch.zeros(num_physical_blocks, block_size, num_kv_heads, head_dim)

    q_list, k_flats, v_flats, block_tables = [], [], [], []
    used_physical = set()

    for s_idx, seq_len in enumerate(seq_lens):
        q_s = torch.randn(num_q_heads, head_dim)
        k_flat = torch.randn(seq_len, num_kv_heads, head_dim)
        v_flat = torch.randn(seq_len, num_kv_heads, head_dim)

        q_list.append(q_s)
        k_flats.append(k_flat)
        v_flats.append(v_flat)

        num_blocks = (seq_len + block_size - 1) // block_size
        available = [i for i in range(num_physical_blocks) if i not in used_physical]
        physical_ids = random.sample(available, num_blocks)
        used_physical.update(physical_ids)

        for logical_idx, physical_idx in enumerate(physical_ids):
            start = logical_idx * block_size
            end = min(start + block_size, seq_len)
            take = end - start
            k_cache[physical_idx, :take] = k_flat[start:end]
            v_cache[physical_idx, :take] = v_flat[start:end]

        block_tables.append(physical_ids)

    q = torch.stack(q_list)  # [num_seqs, num_q_heads, head_dim]

    actual = naive_reference_paged_attention(
        q, k_cache, v_cache, block_tables, seq_lens, scale
    )

    for i, seq_len in enumerate(seq_lens):
        expected_i = standard_gqa_attention_repeat_interleave(
            q[i : i + 1],
            k_flats[i].unsqueeze(0),
            v_flats[i].unsqueeze(0),
            scale,
        )
        torch.testing.assert_close(
            actual[i : i + 1],
            expected_i,
            atol=1e-5,
            rtol=1e-5,
        )


def test_paged_attention_gqa_ratio_1():
    """GQA ratio 1 (MHA baseline): num_q_heads=8, num_kv_heads=8."""
    _run_paged_gqa_test(
        num_q_heads=8,
        num_kv_heads=8,
        head_dim=32,
        block_size=16,
        seq_lens=[16, 37, 8],
        seed=101,
    )


def test_paged_attention_gqa_ratio_2():
    """GQA ratio 2: num_q_heads=8, num_kv_heads=4."""
    _run_paged_gqa_test(
        num_q_heads=8,
        num_kv_heads=4,
        head_dim=32,
        block_size=16,
        seq_lens=[16, 45, 9],
        seed=102,
    )


def test_paged_attention_gqa_ratio_7_qwen2_5_0_5b():
    """
    GQA ratio 7 matching Qwen2.5-0.5B architecture:
    num_q_heads=14, num_kv_heads=2, head_dim=64 (14 // 2 = 7).
    """
    _run_paged_gqa_test(
        num_q_heads=14,
        num_kv_heads=2,
        head_dim=64,
        block_size=16,
        seq_lens=[13, 32, 67],
        seed=103,
    )


def test_triton_paged_attention_decode_v1_matches_reference():
    """
    Verifies paged_attention_decode_v1 against naive_reference_paged_attention
    over randomized, shuffled, non-contiguous block layouts with fp16 inputs.
    Tests across multiple GQA ratios (ratio 1, ratio 2, and ratio 7 for Qwen2.5-0.5B)
    and varying sequence lengths with partial blocks.
    Tolerance: atol=1e-2, rtol=1e-2 (fp16).
    """
    if not torch.cuda.is_available():
        return

    test_configs = [
        # (num_q_heads, num_kv_heads, head_dim, block_size, seq_len, seed)
        (8, 8, 32, 16, 37, 201),   # GQA ratio 1 (MHA), partial block
        (8, 4, 32, 16, 45, 202),   # GQA ratio 2, partial block
        (14, 2, 64, 16, 67, 203),  # GQA ratio 7 (Qwen2.5-0.5B), partial block
        (14, 2, 64, 16, 32, 204),  # GQA ratio 7, exact block boundary
        (4, 1, 64, 8, 19, 205),    # MQA (ratio 4), small block size
    ]

    for num_q_heads, num_kv_heads, head_dim, block_size, seq_len, seed in test_configs:
        torch.manual_seed(seed)
        random.seed(seed)

        num_logical_blocks = (seq_len + block_size - 1) // block_size
        num_physical_blocks = num_logical_blocks * 3 + 4  # Sparse pool

        # Randomized, non-contiguous physical block IDs
        physical_ids = random.sample(range(num_physical_blocks), num_logical_blocks)

        q = torch.randn(num_q_heads, head_dim, device="cuda", dtype=torch.float16)
        k_flat = torch.randn(seq_len, num_kv_heads, head_dim, device="cuda", dtype=torch.float16)
        v_flat = torch.randn(seq_len, num_kv_heads, head_dim, device="cuda", dtype=torch.float16)

        k_cache = torch.zeros(
            num_physical_blocks, block_size, num_kv_heads, head_dim,
            device="cuda", dtype=torch.float16
        )
        v_cache = torch.zeros(
            num_physical_blocks, block_size, num_kv_heads, head_dim,
            device="cuda", dtype=torch.float16
        )

        # Scatter flat KV into randomized non-contiguous physical blocks
        for logical_idx, phys_id in enumerate(physical_ids):
            start = logical_idx * block_size
            end = min(start + block_size, seq_len)
            take = end - start
            k_cache[phys_id, :take] = k_flat[start:end]
            v_cache[phys_id, :take] = v_flat[start:end]

        block_table = torch.tensor(physical_ids, device="cuda", dtype=torch.int32)
        scale = head_dim ** -0.5

        # 1. Triton kernel execution
        actual_triton = paged_attention_decode_v1(
            q=q,
            k_cache=k_cache,
            v_cache=v_cache,
            block_table=block_table,
            seq_len=seq_len,
            scale=scale,
        )

        # 2. GQA-correct naive reference execution
        # naive_reference_paged_attention takes q with shape [num_seqs, num_q_heads, head_dim]
        expected_ref = naive_reference_paged_attention(
            q=q.unsqueeze(0),
            k_cache=k_cache,
            v_cache=v_cache,
            block_tables=[physical_ids],
            seq_lens=[seq_len],
            scale=scale,
        ).squeeze(0)

        # 3. Independent repeat_interleave check
        expected_plain = standard_gqa_attention_repeat_interleave(
            q=q.unsqueeze(0),
            k=k_flat.unsqueeze(0),
            v=v_flat.unsqueeze(0),
            scale=scale,
        ).squeeze(0)

        # Verify reference matches plain repeat_interleave
        torch.testing.assert_close(expected_ref, expected_plain, atol=1e-3, rtol=1e-3)

        # Verify Triton v1 matches reference with atol=1e-2, rtol=1e-2 (fp16)
        torch.testing.assert_close(
            actual_triton,
            expected_ref,
            atol=1e-2,
            rtol=1e-2,
            msg=lambda msg: f"Mismatch for GQA ratio {num_q_heads // num_kv_heads}, seq_len={seq_len}: {msg}"
        )


def test_triton_paged_attention_decode_v2_matches_v1_and_reference():
    """
    Verifies paged_attention_decode_v2 (single-pass online softmax) matches
    both v1 (two-pass) and naive_reference_paged_attention on randomized, shuffled,
    non-contiguous block layouts with fp16 inputs across multiple GQA ratios.
    Tolerance: atol=1e-2, rtol=1e-2.
    """
    if not torch.cuda.is_available():
        return

    test_configs = [
        # (num_q_heads, num_kv_heads, head_dim, block_size, seq_len, seed)
        (8, 8, 32, 16, 37, 301),   # GQA ratio 1 (MHA), partial block
        (8, 4, 32, 16, 45, 302),   # GQA ratio 2, partial block
        (14, 2, 64, 16, 67, 303),  # GQA ratio 7 (Qwen2.5-0.5B), partial block
        (14, 2, 64, 16, 32, 304),  # GQA ratio 7, exact block boundary
        (4, 1, 64, 8, 19, 305),    # MQA (ratio 4), small block size
    ]

    for num_q_heads, num_kv_heads, head_dim, block_size, seq_len, seed in test_configs:
        torch.manual_seed(seed)
        random.seed(seed)

        num_logical_blocks = (seq_len + block_size - 1) // block_size
        num_physical_blocks = num_logical_blocks * 3 + 4

        physical_ids = random.sample(range(num_physical_blocks), num_logical_blocks)

        q = torch.randn(num_q_heads, head_dim, device="cuda", dtype=torch.float16)
        k_flat = torch.randn(seq_len, num_kv_heads, head_dim, device="cuda", dtype=torch.float16)
        v_flat = torch.randn(seq_len, num_kv_heads, head_dim, device="cuda", dtype=torch.float16)

        k_cache = torch.zeros(
            num_physical_blocks, block_size, num_kv_heads, head_dim,
            device="cuda", dtype=torch.float16
        )
        v_cache = torch.zeros(
            num_physical_blocks, block_size, num_kv_heads, head_dim,
            device="cuda", dtype=torch.float16
        )

        for logical_idx, phys_id in enumerate(physical_ids):
            start = logical_idx * block_size
            end = min(start + block_size, seq_len)
            take = end - start
            k_cache[phys_id, :take] = k_flat[start:end]
            v_cache[phys_id, :take] = v_flat[start:end]

        block_table = torch.tensor(physical_ids, device="cuda", dtype=torch.int32)
        scale = head_dim ** -0.5

        out_v1 = paged_attention_decode_v1(
            q=q, k_cache=k_cache, v_cache=v_cache,
            block_table=block_table, seq_len=seq_len, scale=scale
        )
        out_v2 = paged_attention_decode_v2(
            q=q, k_cache=k_cache, v_cache=v_cache,
            block_table=block_table, seq_len=seq_len, scale=scale
        )
        expected_ref = naive_reference_paged_attention(
            q=q.unsqueeze(0), k_cache=k_cache, v_cache=v_cache,
            block_tables=[physical_ids], seq_lens=[seq_len], scale=scale
        ).squeeze(0)

        # Assert v2 matches reference
        torch.testing.assert_close(
            out_v2, expected_ref, atol=1e-2, rtol=1e-2,
            msg=lambda msg: f"v2 vs reference mismatch for GQA ratio {num_q_heads // num_kv_heads}: {msg}"
        )
        # Assert v2 matches v1
        torch.testing.assert_close(
            out_v2, out_v1, atol=1e-2, rtol=1e-2,
            msg=lambda msg: f"v2 vs v1 mismatch for GQA ratio {num_q_heads // num_kv_heads}: {msg}"
        )


def test_triton_paged_attention_long_sequence_2048_tokens():
    """
    Stress test with seq_len=2048 tokens spanning 128 physical blocks (block_size=16)
    in shuffled order, checking numerical stability and matching against reference.
    """
    if not torch.cuda.is_available():
        return

    torch.manual_seed(401)
    random.seed(401)

    num_q_heads, num_kv_heads, head_dim = 14, 2, 64  # Qwen2.5-0.5B shape
    block_size = 16
    seq_len = 2048
    num_logical_blocks = seq_len // block_size  # 128 blocks
    num_physical_blocks = num_logical_blocks + 20

    physical_ids = random.sample(range(num_physical_blocks), num_logical_blocks)

    q = torch.randn(num_q_heads, head_dim, device="cuda", dtype=torch.float16)
    k_flat = torch.randn(seq_len, num_kv_heads, head_dim, device="cuda", dtype=torch.float16)
    v_flat = torch.randn(seq_len, num_kv_heads, head_dim, device="cuda", dtype=torch.float16)

    k_cache = torch.zeros(num_physical_blocks, block_size, num_kv_heads, head_dim, device="cuda", dtype=torch.float16)
    v_cache = torch.zeros(num_physical_blocks, block_size, num_kv_heads, head_dim, device="cuda", dtype=torch.float16)

    for logical_idx, phys_id in enumerate(physical_ids):
        start = logical_idx * block_size
        end = start + block_size
        k_cache[phys_id] = k_flat[start:end]
        v_cache[phys_id] = v_flat[start:end]

    block_table = torch.tensor(physical_ids, device="cuda", dtype=torch.int32)
    scale = head_dim ** -0.5

    out_v1 = paged_attention_decode_v1(q, k_cache, v_cache, block_table, seq_len, scale)
    out_v2 = paged_attention_decode_v2(q, k_cache, v_cache, block_table, seq_len, scale)
    expected_ref = naive_reference_paged_attention(
        q.unsqueeze(0), k_cache, v_cache, [physical_ids], [seq_len], scale
    ).squeeze(0)

    # Check neither produces NaN or Inf
    assert not torch.isnan(out_v2).any() and not torch.isinf(out_v2).any(), "v2 produced NaN or Inf!"
    assert not torch.isnan(out_v1).any() and not torch.isinf(out_v1).any(), "v1 produced NaN or Inf!"

    torch.testing.assert_close(out_v2, expected_ref, atol=1e-2, rtol=1e-2)
    torch.testing.assert_close(out_v2, out_v1, atol=1e-2, rtol=1e-2)


def test_triton_paged_attention_large_magnitude_scores_no_overflow():
    """
    Stress test with large magnitude query and keys (scale * q . k up to ~500)
    to verify numerical stability: exp(score - max) stays safely within fp32 range
    and does not overflow or produce NaNs.
    """
    if not torch.cuda.is_available():
        return

    torch.manual_seed(501)
    random.seed(501)

    num_q_heads, num_kv_heads, head_dim = 8, 2, 64
    block_size = 16
    seq_len = 250
    num_logical_blocks = (seq_len + block_size - 1) // block_size
    num_physical_blocks = num_logical_blocks + 10

    physical_ids = random.sample(range(num_physical_blocks), num_logical_blocks)

    # Multiply inputs by a factor so unnormalized scores reach hundreds (e.g. 100 to 500)
    q = (torch.randn(num_q_heads, head_dim, device="cuda", dtype=torch.float16) * 10.0)
    k_flat = (torch.randn(seq_len, num_kv_heads, head_dim, device="cuda", dtype=torch.float16) * 10.0)
    v_flat = torch.randn(seq_len, num_kv_heads, head_dim, device="cuda", dtype=torch.float16)

    k_cache = torch.zeros(num_physical_blocks, block_size, num_kv_heads, head_dim, device="cuda", dtype=torch.float16)
    v_cache = torch.zeros(num_physical_blocks, block_size, num_kv_heads, head_dim, device="cuda", dtype=torch.float16)

    for logical_idx, phys_id in enumerate(physical_ids):
        start = logical_idx * block_size
        end = min(start + block_size, seq_len)
        take = end - start
        k_cache[phys_id, :take] = k_flat[start:end]
        v_cache[phys_id, :take] = v_flat[start:end]

    block_table = torch.tensor(physical_ids, device="cuda", dtype=torch.int32)
    scale = head_dim ** -0.5

    out_v1 = paged_attention_decode_v1(q, k_cache, v_cache, block_table, seq_len, scale)
    out_v2 = paged_attention_decode_v2(q, k_cache, v_cache, block_table, seq_len, scale)
    expected_ref = naive_reference_paged_attention(
        q.unsqueeze(0), k_cache, v_cache, [physical_ids], [seq_len], scale
    ).squeeze(0)

    # Must have no NaNs / Infs
    assert not torch.isnan(out_v2).any() and not torch.isinf(out_v2).any(), "v2 produced NaN or Inf with large scores!"
    assert not torch.isnan(out_v1).any() and not torch.isinf(out_v1).any(), "v1 produced NaN or Inf with large scores!"

    torch.testing.assert_close(out_v2, expected_ref, atol=1e-2, rtol=1e-2)
    torch.testing.assert_close(out_v2, out_v1, atol=1e-2, rtol=1e-2)


def test_triton_paged_attention_decode_batched_variable_lengths():
    """
    Tests paged_attention_decode_batched across a batch of sequences with wildly
    different lengths: [1, 17, 500, 2000].
    Verifies that:
      1. pad_block_tables properly pads jagged physical block lists to a 2D GPU tensor.
      2. The batched 2D grid kernel matches the GQA naive reference for each sequence.
      3. The batched kernel matches calling paged_attention_decode_v2 in a loop.
    """
    if not torch.cuda.is_available():
        return

    torch.manual_seed(601)
    random.seed(601)

    # Qwen2.5-0.5B configuration: 14 Q heads, 2 KV heads (ratio 7), head_dim 64
    num_q_heads, num_kv_heads, head_dim = 14, 2, 64
    block_size = 16
    seq_lens_list = [1, 17, 500, 2000]
    num_seqs = len(seq_lens_list)

    total_logical_blocks = sum((s + block_size - 1) // block_size for s in seq_lens_list)
    num_physical_blocks = total_logical_blocks + 40

    k_cache = torch.zeros(
        num_physical_blocks, block_size, num_kv_heads, head_dim,
        device="cuda", dtype=torch.float16
    )
    v_cache = torch.zeros(
        num_physical_blocks, block_size, num_kv_heads, head_dim,
        device="cuda", dtype=torch.float16
    )

    q_list, k_flats, v_flats, block_tables_list = [], [], [], []
    used_physical = set()

    for seq_len in seq_lens_list:
        q_s = torch.randn(num_q_heads, head_dim, device="cuda", dtype=torch.float16)
        k_flat = torch.randn(seq_len, num_kv_heads, head_dim, device="cuda", dtype=torch.float16)
        v_flat = torch.randn(seq_len, num_kv_heads, head_dim, device="cuda", dtype=torch.float16)

        q_list.append(q_s)
        k_flats.append(k_flat)
        v_flats.append(v_flat)

        num_blocks = (seq_len + block_size - 1) // block_size
        available = [i for i in range(num_physical_blocks) if i not in used_physical]
        physical_ids = random.sample(available, num_blocks)
        used_physical.update(physical_ids)

        for logical_idx, phys_id in enumerate(physical_ids):
            start = logical_idx * block_size
            end = min(start + block_size, seq_len)
            take = end - start
            k_cache[phys_id, :take] = k_flat[start:end]
            v_cache[phys_id, :take] = v_flat[start:end]

        block_tables_list.append(physical_ids)

    q = torch.stack(q_list)  # [num_seqs, num_q_heads, head_dim]
    seq_lens_tensor = torch.tensor(seq_lens_list, device="cuda", dtype=torch.int32)
    padded_block_tables = pad_block_tables(block_tables_list, device="cuda")
    scale = head_dim ** -0.5

    # 1. Batched Triton launch
    out_batched = paged_attention_decode_batched(
        q=q,
        k_cache=k_cache,
        v_cache=v_cache,
        block_tables=padded_block_tables,
        seq_lens=seq_lens_tensor,
        scale=scale,
    )

    # 2. Reference naive paged attention
    expected_ref = naive_reference_paged_attention(
        q=q,
        k_cache=k_cache,
        v_cache=v_cache,
        block_tables=block_tables_list,
        seq_lens=seq_lens_list,
        scale=scale,
    )

    # Verify batched kernel matches reference
    torch.testing.assert_close(
        out_batched, expected_ref, atol=1e-2, rtol=1e-2,
        msg="Batched Triton kernel output does not match naive reference"
    )

    # 3. Verify batched kernel matches per-sequence v2 in a loop
    for i in range(num_seqs):
        out_single_v2 = paged_attention_decode_v2(
            q=q[i],
            k_cache=k_cache,
            v_cache=v_cache,
            block_table=torch.tensor(block_tables_list[i], device="cuda", dtype=torch.int32),
            seq_len=seq_lens_list[i],
            scale=scale,
        )
        torch.testing.assert_close(
            out_batched[i], out_single_v2, atol=1e-2, rtol=1e-2,
            msg=f"Batched kernel does not match single-sequence v2 for sequence {i}"
        )
