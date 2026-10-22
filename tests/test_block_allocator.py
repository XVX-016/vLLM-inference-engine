"""
Unit tests for the Phase 2 BlockAllocator free-list logic. These only
exercise the allocator itself (pure Python, no torch/GPU needed) so they
run anywhere, including CI without a GPU runner.

    pytest tests/test_block_allocator.py -v
"""
import random
from collections import OrderedDict

import pytest

from engine.memory.block_allocator import BlockAllocator, BlockTable


def test_allocate_and_free_basic():
    alloc = BlockAllocator(num_blocks=4)
    assert alloc.can_allocate(4)
    assert not alloc.can_allocate(5)

    b0 = alloc.allocate()
    b1 = alloc.allocate()
    assert b0 != b1
    assert len(alloc.free_block_ids) == 2

    alloc.free(b0)
    assert len(alloc.free_block_ids) == 3
    assert alloc.can_allocate(3)


def test_allocate_raises_when_exhausted():
    alloc = BlockAllocator(num_blocks=1)
    alloc.allocate()
    with pytest.raises(MemoryError):
        alloc.allocate()


def test_no_double_free_frees_block_early():
    """
    A block with ref_count 2 (forked/shared) must not return to the free
    list until BOTH holders have freed it — this is the invariant prefix
    caching (Phase 6) depends on.
    """
    alloc = BlockAllocator(num_blocks=2)
    b0 = alloc.allocate()
    alloc.fork(b0)  # ref_count now 2
    assert alloc.ref_counts[b0] == 2

    alloc.free(b0)  # ref_count -> 1, should NOT be returned to free list yet
    assert b0 not in alloc.free_block_ids

    alloc.free(b0)  # ref_count -> 0, now it should be free
    assert b0 in alloc.free_block_ids


def test_zero_fragmentation_alloc_free_cycle():
    """
    Repeated alloc/free cycles of varying sizes should never leak blocks —
    at the end, every block must be back in the free list. This directly
    tests the "fixes 60-80% VRAM waste from fragmentation" claim.
    """
    num_blocks = 16
    alloc = BlockAllocator(num_blocks=num_blocks)

    for _ in range(50):
        held = [alloc.allocate() for _ in range(min(5, len(alloc.free_block_ids)))]
        for b in held:
            alloc.free(b)

    assert len(alloc.free_block_ids) == num_blocks
    assert alloc.ref_counts == {}


def test_block_table_tracks_blocks_needed():
    table = BlockTable(seq_id=0, block_size=16)
    assert table.num_blocks_needed_for(1) == 1
    assert table.num_blocks_needed_for(16) == 1
    assert table.num_blocks_needed_for(17) == 2
    assert table.num_blocks_needed_for(32) == 2


def test_prefix_cache_evicts_in_lru_order():
    alloc = BlockAllocator(num_blocks=2, enable_prefix_cache=True)
    first, second = alloc.allocate(), alloc.allocate()
    first_key = (None, (1, 2))
    second_key = (None, (3, 4))
    alloc.register(first, first_key)
    alloc.register(second, second_key)
    alloc.release(first)
    alloc.release(second)
    alloc.acquire(first)
    alloc.release(first)

    alloc.allocate()
    assert second not in alloc.block_key
    assert first in alloc.block_key
    alloc.check_invariants()


def test_referenced_block_is_not_evicted():
    alloc = BlockAllocator(num_blocks=2, enable_prefix_cache=True)
    cached, referenced = alloc.allocate(), alloc.allocate()
    key = (None, (1, 2))
    alloc.register(cached, key)
    alloc.register(referenced, (None, (3, 4)))
    alloc.release(cached)

    assert alloc.allocate() == cached
    assert alloc.ref_counts[referenced] == 1
    assert alloc.block_key[referenced] == (None, (3, 4))
    alloc.check_invariants()


def test_lookup_prefix_stops_at_first_miss():
    alloc = BlockAllocator(num_blocks=3, enable_prefix_cache=True)
    first, second, unrelated = [alloc.allocate() for _ in range(3)]
    first_key = (None, (1, 2))
    second_key = (first_key, (3, 4))
    alloc.register(first, first_key)
    alloc.register(second, second_key)
    alloc.register(unrelated, ((None, (8, 9)), (10, 11)))

    before = alloc.ref_counts.copy()
    assert alloc.lookup_prefix([1, 2, 3, 4, 5, 6]) == [first, second]
    assert alloc.ref_counts == before
    alloc.check_invariants()


def test_reacquiring_cached_free_block_prevents_eviction():
    alloc = BlockAllocator(num_blocks=2, enable_prefix_cache=True)
    first, second = alloc.allocate(), alloc.allocate()
    key = (None, (1, 2))
    alloc.register(first, key)
    alloc.release(first)
    alloc.acquire(first)

    assert first not in alloc.cached_free
    assert alloc.ref_counts[first] == 1
    with pytest.raises(MemoryError):
        alloc.allocate()
    assert alloc.cache[key] == first
    alloc.check_invariants()


@pytest.mark.parametrize("seed", [7, 19, 2025, 65537])
def test_prefix_cache_randomized_state_machine(seed):
    rng = random.Random(seed)
    num_blocks = 8
    alloc = BlockAllocator(num_blocks=num_blocks, enable_prefix_cache=True)
    plain_free = list(range(num_blocks))
    ref_counts = {}
    cached_free = OrderedDict()
    cache = {}
    block_key = {}
    next_token = 1

    def model_lookup(tokens):
        result = []
        offset = 0
        parent = None
        while offset < len(tokens):
            matches = [
                (key, block_id)
                for key, block_id in cache.items()
                if key[0] == parent
                and tuple(tokens[offset : offset + len(key[1])]) == key[1]
            ]
            if not matches:
                break
            key, block_id = max(matches, key=lambda item: len(item[0][1]))
            result.append(block_id)
            offset += len(key[1])
            parent = key
        return result

    for _ in range(4000):
        available = bool(plain_free or cached_free)
        referenced = list(ref_counts)
        cached = list(cached_free)
        unkeyed_referenced = [block_id for block_id in referenced if block_id not in block_key]
        choices = ["lookup"]
        if available:
            choices.append("allocate")
        if referenced:
            choices.extend(("acquire", "release"))
        if cached or referenced:
            choices.append("acquire")
        if unkeyed_referenced:
            choices.append("register")
        operation = rng.choice(choices)

        if operation == "allocate":
            if plain_free:
                expected = plain_free.pop()
            else:
                expected, _ = cached_free.popitem(last=False)
                del cache[block_key.pop(expected)]
            assert alloc.allocate() == expected
            ref_counts[expected] = 1
        elif operation == "register":
            block_id = rng.choice(unkeyed_referenced)
            parent = None
            if cache and rng.random() < 0.5:
                parent = rng.choice(list(cache))
            key = (parent, (next_token, rng.randrange(1000)))
            next_token += 1
            alloc.register(block_id, key)
            cache[key] = block_id
            block_key[block_id] = key
        elif operation == "acquire":
            block_id = rng.choice(referenced + cached)
            if block_id in cached_free:
                del cached_free[block_id]
                ref_counts[block_id] = 1
            else:
                ref_counts[block_id] += 1
            alloc.acquire(block_id)
        elif operation == "release":
            block_id = rng.choice(referenced)
            ref_counts[block_id] -= 1
            if ref_counts[block_id] == 0:
                del ref_counts[block_id]
                if block_id in block_key:
                    cached_free[block_id] = None
                else:
                    plain_free.append(block_id)
            alloc.release(block_id)
        else:
            tokens = [rng.randrange(1000) for _ in range(rng.randrange(1, 10))]
            before_refs = alloc.ref_counts.copy()
            assert alloc.lookup_prefix(tokens) == model_lookup(tokens)
            assert alloc.ref_counts == before_refs

        alloc.check_invariants()
        assert alloc.ref_counts == ref_counts
        assert alloc.free_block_ids == plain_free
        assert list(alloc.cached_free) == list(cached_free)
        assert alloc.cache == cache
        assert alloc.block_key == block_key
        assert alloc.num_available() == len(plain_free) + len(cached_free)
