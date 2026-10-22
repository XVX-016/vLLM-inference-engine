"""
Unit tests for Continuous Batching Scheduler with FakeModel (Phase 4).

Tests:
1. Blocks are never leaked: after all requests finish, allocator free blocks == total blocks.
2. A finished sequence's slot and blocks are reused by a waiting sequence on subsequent iterations.
3. Batch dynamically grows and shrinks as requests arrive and finish.
4. Watermark admission rule prevents admitting requests when free blocks < prompt + watermark.
"""
from __future__ import annotations

from typing import Dict, List, Optional

import pytest

from engine.memory.block_allocator import BlockAllocator
from engine.scheduler.continuous_batching import (
    BatchType,
    ScheduledBatch,
    Scheduler,
    SeqStatus,
    Sequence,
)


class FakeModelEngine:
    """
    Fake model that returns deterministic scripted tokens for unit testing on CPU.
    """

    def __init__(self, token_script: Optional[Dict[int, List[int]]] = None, default_token: int = 42):
        # token_script: mapping from seq_id to list of token IDs to yield
        self.token_script = token_script or {}
        self.default_token = default_token
        self.step_counts: Dict[int, int] = {}

    def step(self, batch: ScheduledBatch) -> List[int]:
        tokens = []
        for seq in batch.sequences:
            idx = len(seq.generated_ids)
            if seq.seq_id in self.token_script and idx < len(self.token_script[seq.seq_id]):
                tok = self.token_script[seq.seq_id][idx]
            else:
                tok = self.default_token
            tokens.append(tok)
        # Simulate decode_step's responsibility: bump bt.num_tokens so the
        # scheduler's boundary check (kv_len % block_size == 0) fires correctly.
        if batch.batch_type == BatchType.DECODE:
            for seq in batch.sequences:
                if seq.block_table is not None:
                    seq.block_table.num_tokens += 1
        return tokens


def test_blocks_never_leaked():
    """
    Verifies that all allocated blocks are cleanly returned to the allocator
    after sequences finish generation.
    """
    total_blocks = 8
    block_size = 4
    allocator = BlockAllocator(num_blocks=total_blocks)
    scheduler = Scheduler(
        block_allocator=allocator,
        block_size=block_size,
        watermark_blocks=1,
    )
    fake_engine = FakeModelEngine()

    # Add 3 requests with prompt lengths 4 (1 block each) and max_new_tokens = 5
    for i in range(3):
        scheduler.add_request(
            prompt=f"prompt_{i}",
            max_new_tokens=5,
            prompt_token_ids=[10, 11, 12, 13],  # 4 tokens = 1 block
        )

    # Run scheduler until all requests finish
    max_steps = 100
    steps = 0
    while steps < max_steps:
        batch = scheduler.step()
        if batch is None:
            break
        tokens = fake_engine.step(batch)
        scheduler.post_step(batch, tokens)
        steps += 1

    # Verify all requests finished and free list is 100% restored
    assert len(scheduler.waiting) == 0
    assert len(scheduler.running) == 0
    assert len(allocator.free_block_ids) == total_blocks
    assert len(allocator.ref_counts) == 0


def test_finished_sequence_slot_reused_by_waiting():
    """
    Verifies that when memory is tight and requests are waiting,
    as soon as a running sequence finishes, its physical blocks and batch slot
    are immediately reused by the waiting sequence on the next iteration.
    """
    # Total 3 blocks:
    # seq0 needs 1 block prompt + 1 block watermark = 2 blocks to admit.
    # While seq0 is running, free blocks = 2.
    # If seq1 arrives, it also needs 1 block + 1 watermark = 2 blocks -> can be admitted.
    # Total blocks = 3 means only 1 sequence can run if prompt=2 blocks + 1 watermark.
    total_blocks = 3
    block_size = 4
    allocator = BlockAllocator(num_blocks=total_blocks)
    scheduler = Scheduler(
        block_allocator=allocator,
        block_size=block_size,
        watermark_blocks=1,
        eos_token_id=999,
    )

    # Seq 0: prompt of 8 tokens (2 blocks). To admit: 2 + 1 = 3 blocks (allocator has 3).
    # After admission, free blocks = 1.
    seq0 = scheduler.add_request(
        prompt="seq0",
        max_new_tokens=3,
        prompt_token_ids=[1] * 8,  # 8 tokens -> 2 blocks
    )

    # Seq 1: prompt of 8 tokens (2 blocks). Needs 2 + 1 = 3 blocks to admit.
    # Cannot be admitted while seq0 is running because only 1 free block remains.
    seq1 = scheduler.add_request(
        prompt="seq1",
        max_new_tokens=3,
        prompt_token_ids=[2] * 8,
    )

    # Script seq0 to emit EOS (999) after 2 decode steps
    fake_engine = FakeModelEngine(token_script={seq0.seq_id: [100, 999]})

    # Step 1: Prefill seq0
    batch1 = scheduler.step()
    assert batch1 is not None
    assert batch1.batch_type == BatchType.PREFILL
    assert batch1.sequences[0].seq_id == seq0.seq_id
    assert len(allocator.free_block_ids) == 1  # 2 blocks used by seq0
    tokens1 = fake_engine.step(batch1)
    scheduler.post_step(batch1, tokens1)

    # Step 2: seq1 cannot be admitted because free_blocks (1) < 2 + 1.
    # Therefore, decode step is scheduled for seq0.
    batch2 = scheduler.step()
    assert batch2 is not None
    assert batch2.batch_type == BatchType.DECODE
    assert len(batch2.sequences) == 1
    assert batch2.sequences[0].seq_id == seq0.seq_id
    tokens2 = fake_engine.step(batch2)
    scheduler.post_step(batch2, tokens2)

    # seq0 emitted 999 (EOS) in step 2!
    assert seq0.status == SeqStatus.FINISHED
    assert len(scheduler.running) == 0
    # Blocks from seq0 must be freed immediately
    assert len(allocator.free_block_ids) == total_blocks

    # Step 3: Now seq1 can be admitted immediately on this iteration!
    batch3 = scheduler.step()
    assert batch3 is not None
    assert batch3.batch_type == BatchType.PREFILL
    assert batch3.sequences[0].seq_id == seq1.seq_id
    assert seq1.status == SeqStatus.WAITING  # admitted, will transition to running on post_step
    tokens3 = fake_engine.step(batch3)
    scheduler.post_step(batch3, tokens3)
    assert seq1.status == SeqStatus.RUNNING


def test_batch_grows_and_shrinks():
    """
    Tests that the running batch dynamically grows as requests arrive and are admitted,
    and shrinks as requests finish at different times.
    """
    total_blocks = 16
    block_size = 4
    allocator = BlockAllocator(num_blocks=total_blocks)
    scheduler = Scheduler(
        block_allocator=allocator,
        block_size=block_size,
        watermark_blocks=1,
    )

    # Add seq0 (max_new_tokens=4), seq1 (max_new_tokens=2)
    seq0 = scheduler.add_request("p0", max_new_tokens=4, prompt_token_ids=[1, 2])
    seq1 = scheduler.add_request("p1", max_new_tokens=2, prompt_token_ids=[3, 4])

    fake_engine = FakeModelEngine()

    # Step 1: Prefill seq0
    b1 = scheduler.step()
    assert b1.batch_type == BatchType.PREFILL
    assert b1.sequences[0].seq_id == seq0.seq_id
    scheduler.post_step(b1, fake_engine.step(b1))
    assert len(scheduler.running) == 1

    # Step 2: Prefill seq1 (waiting queue still has seq1)
    b2 = scheduler.step()
    assert b2.batch_type == BatchType.PREFILL
    assert b2.sequences[0].seq_id == seq1.seq_id
    scheduler.post_step(b2, fake_engine.step(b2))
    # Batch grew to 2 running sequences!
    assert len(scheduler.running) == 2

    # Step 3: Decode step with both seq0 and seq1
    b3 = scheduler.step()
    assert b3.batch_type == BatchType.DECODE
    assert len(b3.sequences) == 2
    scheduler.post_step(b3, fake_engine.step(b3))
    # seq1 generated 2 tokens total (1 in prefill, 1 in decode step) -> finished!
    assert seq1.status == SeqStatus.FINISHED
    # Batch shrank from 2 to 1!
    assert len(scheduler.running) == 1
    assert scheduler.running[0].seq_id == seq0.seq_id

    # Add a new request seq2 mid-stream
    seq2 = scheduler.add_request("p2", max_new_tokens=3, prompt_token_ids=[5, 6])

    # Step 4: seq2 is admitted and prefilled
    b4 = scheduler.step()
    assert b4.batch_type == BatchType.PREFILL
    assert b4.sequences[0].seq_id == seq2.seq_id
    scheduler.post_step(b4, fake_engine.step(b4))
    # Batch grew back to 2!
    assert len(scheduler.running) == 2

    # Step 5: Decode step with both seq0 and seq2
    b5 = scheduler.step()
    assert b5.batch_type == BatchType.DECODE
    assert len(b5.sequences) == 2
    scheduler.post_step(b5, fake_engine.step(b5))

    # Drain remaining steps
    while True:
        b = scheduler.step()
        if b is None:
            break
        scheduler.post_step(b, fake_engine.step(b))

    assert len(scheduler.running) == 0
    assert len(scheduler.waiting) == 0
    assert len(allocator.free_block_ids) == total_blocks


def test_decode_step_allocates_block_on_boundary():
    """
    Verifies that when running sequences cross block boundaries (cur_len % block_size == 0),
    the scheduler allocates exactly one new physical block before the decode step.
    """
    total_blocks = 8
    block_size = 4
    allocator = BlockAllocator(num_blocks=total_blocks)
    scheduler = Scheduler(
        block_allocator=allocator,
        block_size=block_size,
        watermark_blocks=1,
    )
    fake_engine = FakeModelEngine()

    # seq prompt has 3 tokens -> 1 block allocated initially
    seq = scheduler.add_request("p", max_new_tokens=3, prompt_token_ids=[1, 2, 3])

    # 1. Prefill: cur_len becomes 3 (prompt_len), 1 token generated.
    #    bt.num_tokens == 3 (set by step()). The first generated token is
    #    the decode input, not yet in the KV cache, so cur_len == 3.
    b_prefill = scheduler.step()
    scheduler.post_step(b_prefill, fake_engine.step(b_prefill))
    assert seq.cur_len == 3  # prompt_len; first token is decode input, not in KV cache
    assert len(seq.block_table.physical_block_ids) == 1

    # 2. First decode step: FakeModelEngine bumps bt.num_tokens to 4.
    #    post_step mirrors cur_len = 4.  Now kv_len == 4 == block_size.
    b_decode1 = scheduler.step()
    assert b_decode1.batch_type == BatchType.DECODE
    # At this point step() sees kv_len==4 for the NEXT step call. But step()
    # is called for decode1 when kv_len==3 still. The boundary fires when
    # step() is called for decode2 and sees kv_len==4.
    fake_engine.step(b_decode1)  # bumps bt.num_tokens to 4
    scheduler.post_step(b_decode1, [42])
    assert seq.cur_len == 4  # now kv_len == 4, boundary will fire on next step()

    # 3. Second decode step: step() sees kv_len==4 (boundary!) and allocates new block.
    b_decode2 = scheduler.step()
    assert b_decode2.batch_type == BatchType.DECODE
    assert len(seq.block_table.physical_block_ids) == 2  # new block allocated
    fake_engine.step(b_decode2)  # bumps bt.num_tokens to 5
    scheduler.post_step(b_decode2, [42])
    assert seq.cur_len == 5


def test_preemption_tiny_pool_many_requests():
    """
    Stress test with a deliberately tiny pool (40 blocks) and many concurrent sequences (e.g. 20).
    Verifies:
      1. Preemptions occur under memory pressure.
      2. No deadlock occurs.
      3. No physical blocks are leaked (all returned to allocator at end).
      4. Every request finishes with the exact expected token count.
      5. Preemption count is logged and non-zero.
    """
    total_blocks = 40
    block_size = 4
    allocator = BlockAllocator(num_blocks=total_blocks)
    scheduler = Scheduler(
        block_allocator=allocator,
        block_size=block_size,
        watermark_blocks=1,
        max_batch_size=16,
    )

    num_requests = 20
    # Each request has prompt length 6 (2 blocks) and generates 10 tokens (3 blocks total needed per sequence)
    # 20 sequences * 3 blocks = 60 blocks > 40 blocks pool!
    # Therefore, multiple LIFO preemptions MUST happen.
    sequences = []
    target_token_counts = {}

    for i in range(num_requests):
        prompt_len = 6
        max_tokens = 10
        seq = scheduler.add_request(
            prompt=f"prompt_{i}",
            max_new_tokens=max_tokens,
            prompt_token_ids=[100 + i] * prompt_len,
        )
        sequences.append(seq)
        target_token_counts[seq.seq_id] = max_tokens

    fake_engine = FakeModelEngine(default_token=777)

    max_steps = 1000
    steps = 0
    while steps < max_steps:
        batch = scheduler.step()
        if batch is None:
            break
        tokens = fake_engine.step(batch)
        scheduler.post_step(batch, tokens)
        steps += 1

    print(f"\n[Test Result] Completed in {steps} steps with {scheduler.preemption_count} preemptions.")

    # Invariants verification:
    # 1. Preemptions actually occurred under memory pressure
    assert scheduler.preemption_count > 0, "Expected preemptions under tiny pool memory pressure!"

    # 2. No deadlock: queues are empty
    assert len(scheduler.running) == 0
    assert len(scheduler.waiting) == 0

    # 3. No leaked blocks: all 40 blocks returned
    assert len(allocator.free_block_ids) == total_blocks
    assert len(allocator.ref_counts) == 0

    # 4. Every sequence finished with the correct token count
    for seq in sequences:
        assert seq.status == SeqStatus.FINISHED
        # Total generated tokens across all prefill + decode iterations equals max_new_tokens
        # Note: if preempted, generated tokens so far were absorbed into prompt_token_ids,
        # and re-prefill generated the rest. Total tokens added = initial_prompt + max_new_tokens.
        initial_prompt_len = 6
        expected_total_len = initial_prompt_len + target_token_counts[seq.seq_id]
        total_tokens = len(seq.prompt_token_ids) + len(seq.generated_ids)
        assert total_tokens == expected_total_len, (
            f"Seq {seq.seq_id} ended with {total_tokens} tokens, expected {expected_total_len}!"
        )


def test_single_sequence_exceeds_pool_fails():
    """
    Verifies that if a single sequence can never fit in the pool,
    it fails with a clear MemoryError instead of looping forever.
    """
    total_blocks = 2
    block_size = 4
    allocator = BlockAllocator(num_blocks=total_blocks)
    scheduler = Scheduler(
        block_allocator=allocator,
        block_size=block_size,
        watermark_blocks=1,
    )

    # Prompt requires 3 blocks (10 tokens / 4 = 3 blocks) > 2 total blocks!
    scheduler.add_request("too_big", max_new_tokens=5, prompt_token_ids=[1] * 10)

    with pytest.raises(MemoryError, match="exceeding total pool capacity"):
        scheduler.step()

    # Verify no leaked blocks
    assert len(allocator.free_block_ids) == total_blocks


def test_prefix_cache_shares_blocks_and_recomputes_a_prompt_token():
    allocator = BlockAllocator(num_blocks=8, enable_prefix_cache=True)
    scheduler = Scheduler(
        block_allocator=allocator,
        block_size=2,
        watermark_blocks=1,
        debug_invariants=True,
    )
    fake_engine = FakeModelEngine()
    prompt_ids = [10, 11, 12, 13, 14, 15]

    first = scheduler.add_request("first", max_new_tokens=3, prompt_token_ids=prompt_ids)
    first_batch = scheduler.step()
    scheduler.post_step(first_batch, fake_engine.step(first_batch))
    first_blocks = list(first.block_table.physical_block_ids)

    second = scheduler.add_request("second", max_new_tokens=3, prompt_token_ids=prompt_ids)
    second_batch = scheduler.step()
    assert second_batch.batch_type == BatchType.PREFILL
    assert second.num_cached_tokens == 4
    assert second.block_table.physical_block_ids[:2] == first_blocks[:2]
    assert second.block_table.physical_block_ids[2] != first_blocks[2]
    scheduler.post_step(second_batch, fake_engine.step(second_batch))
    assert allocator.ref_counts[first_blocks[0]] == 2
    assert allocator.ref_counts[first_blocks[1]] == 2
    scheduler._check_invariants()


def test_prefix_cache_hit_leaves_exactly_one_prompt_token_to_recompute():
    allocator = BlockAllocator(num_blocks=4, enable_prefix_cache=True)
    scheduler = Scheduler(
        block_allocator=allocator,
        block_size=2,
        watermark_blocks=0,
        prefix_mode="full",
    )
    fake_engine = FakeModelEngine()

    seed = scheduler.add_request("seed", max_new_tokens=1, prompt_token_ids=[1, 2])
    seed_batch = scheduler.step()
    scheduler.post_step(seed_batch, fake_engine.step(seed_batch))
    assert seed.status == SeqStatus.FINISHED

    target = scheduler.add_request("target", max_new_tokens=1, prompt_token_ids=[1, 2, 3])
    target_batch = scheduler.step()
    assert target_batch.batch_type == BatchType.PREFILL
    assert target.num_cached_tokens == 2
    assert target.block_table.num_tokens - target.num_cached_tokens == 1
    scheduler.post_step(target_batch, fake_engine.step(target_batch))

    assert allocator.ref_counts == {}
    assert allocator.num_available() == allocator.num_blocks


def test_finished_prompt_stays_cached_and_is_reused():
    allocator = BlockAllocator(num_blocks=6, enable_prefix_cache=True)
    scheduler = Scheduler(block_allocator=allocator, block_size=2, watermark_blocks=1)
    fake_engine = FakeModelEngine()
    prompt_ids = [20, 21, 22, 23, 24, 25]

    first = scheduler.add_request("first", max_new_tokens=1, prompt_token_ids=prompt_ids)
    first_batch = scheduler.step()
    first_blocks = list(first.block_table.physical_block_ids)
    scheduler.post_step(first_batch, fake_engine.step(first_batch))
    assert first.status == SeqStatus.FINISHED
    assert allocator.ref_counts == {}
    assert len(allocator.cached_free) == 3

    second = scheduler.add_request("second", max_new_tokens=1, prompt_token_ids=prompt_ids)
    second_batch = scheduler.step()
    assert second.num_cached_tokens == 4
    assert second.block_table.physical_block_ids[:2] == first_blocks[:2]
    scheduler.post_step(second_batch, fake_engine.step(second_batch))

    assert allocator.ref_counts == {}
    assert allocator.num_available() == allocator.num_blocks
    assert len(allocator.free_block_ids) + len(allocator.cached_free) == allocator.num_blocks


def test_tiny_pool_admission_evicts_cached_prefix_blocks():
    allocator = BlockAllocator(num_blocks=2, enable_prefix_cache=True)
    scheduler = Scheduler(block_allocator=allocator, block_size=2, watermark_blocks=0)
    fake_engine = FakeModelEngine()

    cached_keys = []
    for token in (30, 40):
        seq = scheduler.add_request(
            f"prompt-{token}",
            max_new_tokens=1,
            prompt_token_ids=[token, token + 1],
        )
        batch = scheduler.step()
        scheduler.post_step(batch, fake_engine.step(batch))
        cached_keys.append((None, (token, token + 1)))
        assert cached_keys[-1] in allocator.cache

    newest = scheduler.add_request("newest", max_new_tokens=1, prompt_token_ids=[50, 51])
    batch = scheduler.step()
    assert batch.batch_type == BatchType.PREFILL
    assert cached_keys[0] not in allocator.cache
    assert cached_keys[1] in allocator.cache
    scheduler.post_step(batch, fake_engine.step(batch))
    assert newest.status == SeqStatus.FINISHED
    assert allocator.ref_counts == {}
    assert allocator.num_available() == allocator.num_blocks


def test_preempted_sequence_reuses_its_cached_prefix_on_readmission():
    allocator = BlockAllocator(num_blocks=3, enable_prefix_cache=True)
    scheduler = Scheduler(
        block_allocator=allocator,
        block_size=2,
        watermark_blocks=0,
        max_batch_size=2,
        debug_invariants=True,
    )
    fake_engine = FakeModelEngine()

    victim = scheduler.add_request("victim", max_new_tokens=3, prompt_token_ids=[1, 2, 3])
    other = scheduler.add_request("other", max_new_tokens=3, prompt_token_ids=[8, 9])

    victim_prefill = scheduler.step()
    scheduler.post_step(victim_prefill, fake_engine.step(victim_prefill))
    victim_prefix_ids = list(victim.block_table.physical_block_ids)
    other_prefill = scheduler.step()
    scheduler.post_step(other_prefill, fake_engine.step(other_prefill))

    # The other sequence needs a new block first and preempts the victim.
    preempting_batch = scheduler.step()
    assert victim in scheduler.waiting
    assert victim.block_table is None
    assert preempting_batch.sequences == [other]
    scheduler.post_step(preempting_batch, fake_engine.step(preempting_batch))

    # The victim's cached hit is not counted as capacity for its remaining block;
    # after the other sequence finishes, it can be admitted and reuse that block.
    other_decode = scheduler.step()
    assert other_decode.batch_type == BatchType.DECODE
    scheduler.post_step(other_decode, fake_engine.step(other_decode))
    readmission = scheduler.step()
    assert readmission.batch_type == BatchType.PREFILL
    assert readmission.sequences == [victim]
    assert victim.num_cached_tokens == 2
    assert victim.block_table.physical_block_ids[: victim.num_cached_tokens // 2] == (
        victim_prefix_ids[: victim.num_cached_tokens // 2]
    )
    scheduler.post_step(readmission, fake_engine.step(readmission))

    while scheduler.has_unfinished_requests():
        batch = scheduler.step()
        if batch is None:
            continue
        scheduler.post_step(batch, fake_engine.step(batch))

    assert scheduler.preemption_count > 0
    assert allocator.ref_counts == {}
    assert allocator.num_available() == allocator.num_blocks
    assert len(allocator.free_block_ids) + len(allocator.cached_free) == allocator.num_blocks
