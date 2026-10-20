import time, torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
from transformers.cache_utils import StaticCache

from engine.serving.naive_server import NaiveHFModelRunner, PendingNaiveRequest
from engine.kernels.paged_attention_integration import KernelPagedManualDecode
from engine.memory.block_allocator import BlockAllocator
from engine.scheduler.continuous_batching import Scheduler, SequenceGroup

model_name = 'Qwen/Qwen2.5-0.5B-Instruct'
cfg = AutoConfig.from_pretrained(model_name)
tok = AutoTokenizer.from_pretrained(model_name)
tok.pad_token = tok.eos_token

def get_gpu_clocks():
    import subprocess
    res = subprocess.run(['nvidia-smi', '--query-gpu=clocks.current.graphics,temperature.gpu', '--format=csv,noheader,nounits'], capture_output=True, text=True)
    return res.stdout.strip()

print('=== BENCHMARK DECODE STEPS (10 warmup, 30 timed) ===')
print(f'Initial GPU: {get_gpu_clocks()}')

batch_sizes = [1, 3, 4, 8]
results = {'plain': {}, 'naive': {}, 'paged': {}}

# 1. Plain HF decode loop
print('\n--- 1. Plain HF Decode Loop ---')
model_plain = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.float16, device_map='cuda')
for bs in batch_sizes:
    cache = StaticCache(config=cfg, max_batch_size=bs, max_cache_len=192, device='cuda', dtype=torch.float16)
    inputs = tok(['Hello world'] * bs, return_tensors='pt', padding=True).to('cuda')
    cache.reset()
    out = model_plain(input_ids=inputs.input_ids, attention_mask=inputs.attention_mask, past_key_values=cache, use_cache=True)
    next_tok = torch.argmax(out.logits[:, -1, :], dim=-1, keepdim=True)
    cur_mask = torch.cat([inputs.attention_mask, torch.ones((bs, 1), dtype=inputs.attention_mask.dtype, device='cuda')], dim=-1)
    
    # Warmup 10 steps
    with torch.inference_mode():
        for _ in range(10):
            dec = model_plain(input_ids=next_tok, attention_mask=cur_mask, past_key_values=cache, use_cache=True)
            next_tok = torch.argmax(dec.logits[:, -1, :], dim=-1, keepdim=True)
            cur_mask = torch.cat([cur_mask, torch.ones((bs, 1), dtype=cur_mask.dtype, device='cuda')], dim=-1)
        
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(30):
            dec = model_plain(input_ids=next_tok, attention_mask=cur_mask, past_key_values=cache, use_cache=True)
            next_tok = torch.argmax(dec.logits[:, -1, :], dim=-1, keepdim=True)
            cur_mask = torch.cat([cur_mask, torch.ones((bs, 1), dtype=cur_mask.dtype, device='cuda')], dim=-1)
        torch.cuda.synchronize()
        dt_ms = (time.perf_counter() - t0) / 30.0 * 1000.0
    results['plain'][bs] = dt_ms
    print(f'  Plain HF BS={bs}: {dt_ms:.2f} ms/step  [GPU: {get_gpu_clocks()}]')

del model_plain
torch.cuda.empty_cache()

# 2. Naive Runner
print('\n--- 2. Naive Runner ---')
for bs in batch_sizes:
    runner_n = NaiveHFModelRunner(model_name=model_name, max_slots=bs, max_seq_len=192, device='cuda', dtype=torch.float16)
    prompts = ['Hello world ' + str(i) for i in range(bs)]
    batch = [PendingNaiveRequest(f'req-{i}', prompts[i], [1, 2], 45, True, None) for i in range(bs)]
    # Run execute_batch: 1 prefill + 44 decodes (14 warmup + 30 timed)
    runner_n.execute_batch(batch, lambda r, tok: None)
    times = runner_n.decode_times
    # last 30 steps
    timed = times[-30:] if len(times) >= 30 else times
    dt_ms = (sum(timed) / len(timed)) * 1000.0
    results['naive'][bs] = dt_ms
    print(f'  Naive Runner BS={bs}: {dt_ms:.2f} ms/step  [GPU: {get_gpu_clocks()}]')
    del runner_n
    torch.cuda.empty_cache()

# 3. Paged Runner
print('\n--- 3. Paged Runner ---')
block_size = 16
total_blocks = 128
hf_model_p = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.float16, device_map='cuda')
runner_p = KernelPagedManualDecode(
    model=hf_model_p,
    tokenizer=tok,
    block_size=block_size,
    num_blocks=total_blocks,
    device='cuda',
    dtype=torch.float16,
)

for bs in batch_sizes:
    allocator = BlockAllocator(num_blocks=total_blocks)
    scheduler = Scheduler(block_allocator=allocator, block_size=block_size, watermark_blocks=1)
    
    # Add bs requests
    for i in range(bs):
        scheduler.add_request(f'paged-req-{i}', prompt_tokens=[10, 20, 30], max_tokens=60)
    
    # Prefill step
    batch_p = scheduler.schedule()
    runner_p.step(batch_p)
    
    # Warmup 10 decode steps
    with torch.inference_mode():
        for _ in range(10):
            batch_d = scheduler.schedule()
            runner_p.step(batch_d)
        
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(30):
            batch_d = scheduler.schedule()
            runner_p.step(batch_d)
        torch.cuda.synchronize()
        dt_ms = (time.perf_counter() - t0) / 30.0 * 1000.0
    results['paged'][bs] = dt_ms
    print(f'  Paged Runner BS={bs}: {dt_ms:.2f} ms/step  [GPU: {get_gpu_clocks()}]')

del hf_model_p
del runner_p
torch.cuda.empty_cache()

print('\n=== COMPARISON SUMMARY ===')
print(f'{"BS":<4} | {"Plain HF (ms)":<14} | {"Naive (ms)":<12} | {"Paged (ms)":<12} | {"Naive/Plain":<12} | {"Paged/Plain":<12}')
print('-' * 76)
for bs in batch_sizes:
    pl = results['plain'][bs]
    nv = results['naive'][bs]
    pg = results['paged'][bs]
    print(f'{bs:<4} | {pl:<14.2f} | {nv:<12.2f} | {pg:<12.2f} | {nv/pl:<12.2f} | {pg/pl:<12.2f}')
