from types import SimpleNamespace
import logging

import torch

from engine.kernels.paged_attention_integration import KernelPagedManualDecode
from engine.memory.block_allocator import BlockTable


class FakePrefillModel:
    def __init__(self):
        self.config = SimpleNamespace(
            num_hidden_layers=1,
            num_key_value_heads=1,
            head_dim=2,
        )
        self.calls = []

    def __call__(
        self,
        input_ids,
        use_cache,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        logits_to_keep=0,
    ):
        self.calls.append(
            {
                "input_ids": input_ids.clone(),
                "attention_mask": attention_mask,
                "position_ids": position_ids,
                "past_key_values": past_key_values,
                "logits_to_keep": logits_to_keep,
                "inference_mode": torch.is_inference_mode_enabled(),
            }
        )
        prompt_len = input_ids.shape[1]
        positions = (
            position_ids[0]
            if position_ids is not None
            else torch.arange(prompt_len, device=input_ids.device)
        )
        key = torch.stack((positions.float(), positions.float() + 1), dim=-1)
        key = key.reshape(1, 1, prompt_len, 2)
        value = key + 100
        logits = torch.zeros((1, prompt_len, 5), dtype=torch.float32)
        logits[:, :, 3] = 1
        if logits_to_keep:
            logits = logits[:, -logits_to_keep:, :]

        if past_key_values is not None:
            layer = past_key_values.layers[0]
            layer.keys = torch.cat((layer.keys, key), dim=2)
            layer.values = torch.cat((layer.values, value), dim=2)
            return SimpleNamespace(logits=logits, past_key_values=past_key_values)
        return SimpleNamespace(logits=logits, past_key_values=((key, value),))


def test_prefill_keeps_hit_blocks_and_copies_only_owned_blocks(caplog):
    caplog.set_level(logging.INFO, logger="engine.kernels.paged_attention_integration")
    block_size = 2
    runner = KernelPagedManualDecode(
        model=FakePrefillModel(),
        tokenizer=None,
        block_size=block_size,
        num_blocks=2,
        device="cpu",
        dtype=torch.float32,
        debug_prefix_cache=True,
    )
    table = BlockTable(seq_id=0, block_size=block_size, physical_block_ids=[0, 1], num_tokens=4)

    expected_keys = torch.tensor([[[0, 1]], [[1, 2]]], dtype=torch.float32)
    expected_values = expected_keys + 100
    runner.pool.k_cache[0][0] = expected_keys
    runner.pool.v_cache[0][0] = expected_values
    seq = SimpleNamespace(
        prompt_token_ids=[10, 11, 12, 13],
        block_table=table,
        num_cached_tokens=block_size,
        last_logits=None,
    )

    assert runner.prefill(seq) == 3

    assert torch.equal(runner.pool.k_cache[0][0], expected_keys)
    assert torch.equal(runner.pool.v_cache[0][0], expected_values)
    assert torch.equal(runner.pool.k_cache[0][1], torch.tensor([[[2, 3]], [[3, 4]]]))
    assert torch.equal(runner.pool.v_cache[0][1], torch.tensor([[[102, 103]], [[103, 104]]]))
    assert "layer=0 max_abs_difference=0" in caplog.text


def test_full_prefill_uses_cached_prefix_and_only_forwards_suffix():
    model = FakePrefillModel()
    runner = KernelPagedManualDecode(
        model=model,
        tokenizer=None,
        block_size=2,
        num_blocks=2,
        device="cpu",
        dtype=torch.float32,
        prefix_mode="full",
    )
    table = BlockTable(seq_id=0, block_size=2, physical_block_ids=[0, 1], num_tokens=4)
    runner.pool.k_cache[0][0] = torch.tensor([[[0, 1]], [[1, 2]]], dtype=torch.float32)
    runner.pool.v_cache[0][0] = torch.tensor([[[100, 101]], [[101, 102]]], dtype=torch.float32)
    seq = SimpleNamespace(
        prompt_token_ids=[10, 11, 12, 13],
        block_table=table,
        num_cached_tokens=2,
        last_logits=None,
    )

    assert runner.prefill(seq) == 3

    call = model.calls[0]
    assert torch.equal(call["input_ids"], torch.tensor([[12, 13]]))
    assert torch.equal(call["position_ids"], torch.tensor([[2, 3]]))
    assert torch.equal(call["attention_mask"], torch.ones((1, 4), dtype=torch.long))
    assert call["past_key_values"].layers[0].keys.shape == (1, 1, 4, 2)
    assert call["logits_to_keep"] == 1
    assert call["inference_mode"]
    assert torch.equal(runner.pool.k_cache[0][1], torch.tensor([[[2, 3]], [[3, 4]]]))
    assert torch.equal(runner.pool.v_cache[0][1], torch.tensor([[[102, 103]], [[103, 104]]]))
    assert seq.last_logits.shape == (5,)


def test_full_prefill_with_zero_hit_uses_normal_full_prompt_path():
    model = FakePrefillModel()
    runner = KernelPagedManualDecode(
        model=model,
        tokenizer=None,
        block_size=2,
        num_blocks=2,
        device="cpu",
        dtype=torch.float32,
        prefix_mode="full",
    )
    table = BlockTable(seq_id=0, block_size=2, physical_block_ids=[0, 1], num_tokens=3)
    seq = SimpleNamespace(
        prompt_token_ids=[10, 11, 12],
        block_table=table,
        num_cached_tokens=0,
        last_logits=None,
    )

    runner.prefill(seq)
    assert torch.equal(model.calls[0]["input_ids"], torch.tensor([[10, 11, 12]]))
    assert model.calls[0]["past_key_values"] is None
    assert model.calls[0]["inference_mode"]


def test_full_prefill_handles_cached_prefix_and_single_remaining_token():
    model = FakePrefillModel()
    runner = KernelPagedManualDecode(
        model=model,
        tokenizer=None,
        block_size=2,
        num_blocks=2,
        device="cpu",
        dtype=torch.float32,
        prefix_mode="full",
    )
    table = BlockTable(seq_id=0, block_size=2, physical_block_ids=[0, 1], num_tokens=3)
    runner.pool.k_cache[0][0] = torch.tensor([[[0, 1]], [[1, 2]]], dtype=torch.float32)
    runner.pool.v_cache[0][0] = torch.tensor([[[100, 101]], [[101, 102]]], dtype=torch.float32)
    seq = SimpleNamespace(
        prompt_token_ids=[10, 11, 12],
        block_table=table,
        num_cached_tokens=2,
        last_logits=None,
    )

    runner.prefill(seq)
    call = model.calls[0]
    assert torch.equal(call["input_ids"], torch.tensor([[12]]))
    assert torch.equal(call["position_ids"], torch.tensor([[2]]))
    assert torch.equal(call["attention_mask"], torch.ones((1, 3), dtype=torch.long))
    assert call["past_key_values"].get_seq_length() == 3
    assert call["inference_mode"]
