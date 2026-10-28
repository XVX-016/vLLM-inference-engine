"""
Precise measurement with CUDA events and logged GPU clocks of:
(a) raw HF prefill (489 tokens)
(b) engine prefill with 'off' (full prompt copy)
(c) engine prefill with 'full' (0 cached vs 481 cached)
Runs 30 repetitions each after warmup.
"""
import time
import subprocess
import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from engine.memory.block_allocator import PhysicalBlockPool, BlockAllocator, BlockTable
from engine.kernels.paged_attention_integration import KernelPagedManualDecode
from engine.scheduler.continuous_batching import Sequence

def get_gpu_clocks():
    try:
        r = subprocess.run(
            ["nvidia-smi", "--query-gpu=clocks.current.graphics,clocks.current.memory,temperature.gpu,power.draw",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, check=True
        )
        return r.stdout.strip()
    except Exception as e:
        return str(e)

device = "cuda"
MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
tok = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float16).to(device).eval()

sys_text = "The transformer architecture relies on multi-head attention to capture relationships. " * 40
suffix_text = "Explain paged attention in detail."
full_prompt = f"{sys_text}\n\n{suffix_text}"
ids = tok.encode(full_prompt, return_tensors="pt").to(device)
prompt_len = ids.shape[1]
print(f"Prompt length: {prompt_len} tokens. Initial GPU clocks: {get_gpu_clocks()}")

N_REPS = 30

# ── (a) Raw HF prefill ─────────────────────────────────────────────────────────
for _ in range(5):
    with torch.no_grad():
        _ = model(input_ids=ids, use_cache=True)
torch.cuda.synchronize()

start_event = torch.cuda.Event(enable_timing=True)
end_event = torch.cuda.Event(enable_timing=True)

times_hf = []
for _ in range(N_REPS):
    torch.cuda.synchronize()
    start_event.record()
    with torch.no_grad():
        _ = model(input_ids=ids, use_cache=True)
    end_event.record()
    torch.cuda.synchronize()
    times_hf.append(start_event.elapsed_time(end_event))

print(f"(a) Raw HF prefill (489 tokens): mean = {np.mean(times_hf):.2f} ms, p50 = {np.median(times_hf):.2f} ms")

# ── (b) Engine prefill with 'off' ──────────────────────────────────────────────
runner_off = KernelPagedManualDecode(model=model, tokenizer=tok, num_blocks=512, block_size=16, prefix_mode="off", device=device)
allocator = BlockAllocator(num_blocks=512)

times_engine_off = []
times_copy_only = []
for _ in range(N_REPS):
    bt = BlockTable(seq_id=1, block_size=16)
    num_blocks = (prompt_len + 15) // 16
    for _ in range(num_blocks):
        bt.physical_block_ids.append(allocator.allocate())
    bt.num_tokens = prompt_len
    seq = Sequence(seq_id=1, prompt=full_prompt, max_new_tokens=16, prompt_token_ids=ids[0].tolist(), block_table=bt)

    torch.cuda.synchronize()
    start_event.record()
    with torch.no_grad():
        runner_off.prefill(seq)
    end_event.record()
    torch.cuda.synchronize()
    times_engine_off.append(start_event.elapsed_time(end_event))

    for b in bt.physical_block_ids:
        allocator.free(b)

print(f"(b) Engine prefill 'off' (HF + layer-by-layer copy): mean = {np.mean(times_engine_off):.2f} ms, p50 = {np.median(times_engine_off):.2f} ms")
copy_overhead = np.mean(times_engine_off) - np.mean(times_hf)
print(f"    Copy overhead into block pool: {copy_overhead:.2f} ms")

# ── (c) Engine prefill 'full' with 0 vs 480 cached tokens ──────────────────────
runner_full = KernelPagedManualDecode(model=model, tokenizer=tok, num_blocks=512, block_size=16, prefix_mode="full", device=device)
# Populate cache for 480 tokens (30 blocks)
sys_ids = tok.encode(sys_text)
cached_blocks = (len(sys_ids) // 16)
cached_tokens = cached_blocks * 16

# Prime the pool with the prefix
prime_bt = BlockTable(seq_id=99, block_size=16)
for _ in range(cached_blocks):
    prime_bt.physical_block_ids.append(allocator.allocate())
prime_bt.num_tokens = cached_tokens
prime_seq = Sequence(seq_id=99, prompt=sys_text, max_new_tokens=1, prompt_token_ids=sys_ids[:cached_tokens], block_table=prime_bt)
with torch.no_grad():
    runner_full.prefill(prime_seq)

times_engine_full_cached = []
for _ in range(N_REPS):
    bt = BlockTable(seq_id=2, block_size=16)
    bt.physical_block_ids.extend(prime_seq.block_table.physical_block_ids) # shared
    uncached_blocks = num_blocks - cached_blocks
    for _ in range(uncached_blocks):
        bt.physical_block_ids.append(allocator.allocate())
    bt.num_tokens = prompt_len
    seq = Sequence(seq_id=2, prompt=full_prompt, max_new_tokens=16, prompt_token_ids=ids[0].tolist(), block_table=bt, num_cached_tokens=cached_tokens)

    torch.cuda.synchronize()
    start_event.record()
    with torch.no_grad():
        runner_full.prefill(seq)
    end_event.record()
    torch.cuda.synchronize()
    times_engine_full_cached.append(start_event.elapsed_time(end_event))

    for b in bt.physical_block_ids[cached_blocks:]:
        allocator.free(b)

print(f"(c) Engine prefill 'full' with {cached_tokens} cached tokens: mean = {np.mean(times_engine_full_cached):.2f} ms, p50 = {np.median(times_engine_full_cached):.2f} ms")
print(f"Ending GPU clocks: {get_gpu_clocks()}")
