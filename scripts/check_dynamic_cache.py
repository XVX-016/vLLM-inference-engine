"""
Inspect the DynamicCache structure from this transformers version to confirm
what API _copy_prefill_kv should use.
"""
import sys, torch
sys.path.insert(0, r'c:\Computing\vLLM-inference-engine')
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained(
    'Qwen/Qwen2.5-0.5B-Instruct', dtype=torch.float16, device_map='cuda'
)
tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen2.5-0.5B-Instruct')
enc = tokenizer('The capital of France is', return_tensors='pt').to('cuda')

with torch.inference_mode():
    out = model(input_ids=enc.input_ids, use_cache=True)
pkv = out.past_key_values

print('pkv type:', type(pkv).__name__)
print('has layers:', hasattr(pkv, 'layers'))

if hasattr(pkv, 'layers'):
    layer0 = pkv.layers[0]
    print('layer0 type:', type(layer0).__name__)
    layer_attrs = [x for x in dir(layer0) if not x.startswith('_')]
    print('layer0 non-private attrs:', layer_attrs)
    for attr in layer_attrs:
        val = getattr(layer0, attr)
        if isinstance(val, torch.Tensor):
            print(f'  {attr}: Tensor shape={val.shape}, dtype={val.dtype}')

# Also try iterating
print()
print('Iterating pkv...')
try:
    for i, item in enumerate(pkv):
        if i >= 1:
            break
        print(f'  iter item {i}: type={type(item).__name__}')
        if isinstance(item, (list, tuple)):
            for j, sub in enumerate(item):
                print(f'    [{j}]: type={type(sub).__name__}, shape={getattr(sub, "shape", None)}')
except Exception as e:
    print(f'  iteration failed: {e}')
