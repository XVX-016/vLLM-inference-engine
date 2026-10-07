"""
Phase 2 Correctness Tests:
1. test_manual_generate_vs_paged_manual_decode_identical:
   Asserts generated token IDs from manual_generate() (contiguous buffer) and
   PagedManualDecode (paged buffer) are bit-for-bit identical with greedy decoding.

2. test_paged_manual_decode_small_block_size_multi_blocks:
   Runs PagedManualDecode with small block_size (block_size=4) so that generation
   spans multiple physical blocks (e.g., prompt_len + max_new_tokens >> 4),
   verifying correct boundary crossings.

3. test_paged_manual_decode_interleaved_non_contiguous_blocks:
   Deliberately forces non-contiguous physical block allocation by interleaving
   allocations between two sequences on the same shared PhysicalBlockPool,
   proving the block table correctly reconstructs each sequence's K/V regardless
   of physical layout.
"""
from __future__ import annotations

import pytest
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from engine.baseline.manual_decode import (
    PagedManualDecode,
    manual_generate,
)
from engine.memory.block_allocator import BlockAllocator, BlockTable


@pytest.fixture(scope="module")
def model_and_tokenizer():
    model_name = "Qwen/Qwen2.5-0.5B-Instruct"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.float32).to(device)
    model.eval()
    return model, tokenizer, device


def test_manual_generate_vs_paged_manual_decode_identical(model_and_tokenizer):
    """
    Runs the same prompt through both manual_generate() (Phase 1, contiguous buffer)
    and PagedManualDecode (Phase 2, paged buffer) with greedy decoding,
    and asserts the generated token ids are identical.
    """
    model, tokenizer, device = model_and_tokenizer
    prompt = "The capital of France is"
    max_new_tokens = 16

    # 1. Contiguous buffer (Phase 1)
    contiguous_ids = manual_generate(
        model=model,
        tokenizer=tokenizer,
        prompt=prompt,
        max_new_tokens=max_new_tokens,
        device=device,
    )

    # 2. Paged buffer (Phase 2)
    paged_engine = PagedManualDecode(
        model=model,
        tokenizer=tokenizer,
        block_size=16,
        num_blocks=512,
        device=device,
        dtype=torch.float32,
    )
    paged_ids = paged_engine.generate(prompt=prompt, max_new_tokens=max_new_tokens)

    assert contiguous_ids == paged_ids, (
        f"Mismatch between contiguous and paged manual decode:\n"
        f"Contiguous: {contiguous_ids}\n"
        f"Paged:      {paged_ids}"
    )


def test_paged_manual_decode_small_block_size_multi_blocks(model_and_tokenizer):
    """
    Runs with block_size=4 so that even a short prompt + generation spans
    multiple blocks (e.g. 5 prompt tokens + 16 new tokens = 21 tokens -> 6 blocks),
    verifying block boundary crossings across multiple layers.
    """
    model, tokenizer, device = model_and_tokenizer
    prompt = "Artificial intelligence is"
    max_new_tokens = 16
    block_size = 4

    contiguous_ids = manual_generate(
        model=model,
        tokenizer=tokenizer,
        prompt=prompt,
        max_new_tokens=max_new_tokens,
        device=device,
    )

    paged_engine = PagedManualDecode(
        model=model,
        tokenizer=tokenizer,
        block_size=block_size,
        num_blocks=512,
        device=device,
        dtype=torch.float32,
    )
    paged_ids = paged_engine.generate(prompt=prompt, max_new_tokens=max_new_tokens)

    assert paged_ids == contiguous_ids, (
        f"Small block_size={block_size} produced divergent tokens:\n"
        f"Expected: {contiguous_ids}\n"
        f"Got:      {paged_ids}"
    )


def test_paged_manual_decode_interleaved_non_contiguous_blocks(model_and_tokenizer):
    """
    Deliberately forces non-contiguous physical block allocation by interleaving
    dummy block allocations from a shared BlockAllocator on the same PhysicalBlockPool.
    Verifies that the BlockTable correctly reconstructs the sequence's K/V
    regardless of physical non-contiguity.
    """
    model, tokenizer, device = model_and_tokenizer
    prompt = "Explain quantum computing in one sentence."
    max_new_tokens = 16
    block_size = 4

    # Baseline ground truth
    expected_ids = manual_generate(
        model=model,
        tokenizer=tokenizer,
        prompt=prompt,
        max_new_tokens=max_new_tokens,
        device=device,
    )

    # Setup shared engine
    num_blocks = 512
    paged_engine = PagedManualDecode(
        model=model,
        tokenizer=tokenizer,
        block_size=block_size,
        num_blocks=num_blocks,
        device=device,
        dtype=torch.float32,
    )

    # Interleaving allocator wrapper: allocates a dummy block for another sequence
    # between every real block allocated, ensuring non-contiguous physical IDs (e.g., 511, 509, 507...)
    class InterleavingBlockAllocator(BlockAllocator):
        def __init__(self, num_blocks: int):
            super().__init__(num_blocks)
            self.dummy_blocks: list[int] = []

        def allocate(self) -> int:
            real_block = super().allocate()
            if self.free_block_ids:
                dummy_block = super().allocate()
                self.dummy_blocks.append(dummy_block)
            return real_block

    interleaved_allocator = InterleavingBlockAllocator(num_blocks=num_blocks)
    paged_ids = paged_engine.generate(
        prompt=prompt,
        max_new_tokens=max_new_tokens,
        allocator=interleaved_allocator,
    )

    # Verify dummy blocks were indeed interleaved
    assert len(interleaved_allocator.dummy_blocks) > 0, "No dummy blocks were interleaved"

    assert paged_ids == expected_ids, (
        f"Interleaved non-contiguous physical allocation caused token divergence:\n"
        f"Expected: {expected_ids}\n"
        f"Got:      {paged_ids}"
    )
