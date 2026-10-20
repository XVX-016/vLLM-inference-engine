# Part 2: Systems War Stories: The Phantom Memory Leak and the Naive Race Condition

Building a low-latency systems-level inference engine requires tracking down bugs that exist not in the mathematics, but at the interface between async runtimes, process memory spaces, and CUDA driver semantics.

Here is the diagnosis story of two critical bugs encountered during development, the hypotheses that failed, and the real root causes.

---

## War Story 1: The Phantom VRAM Leak

### The Symptom
During continuous batching load testing at $\lambda=2.0$, GPU memory steadily marched upward over time:
- Request 0: 960 MiB
- Request 50: 1400 MiB
- Request 120: 2200 MiB
- Request 180: `CUDA Out Of Memory: Tried to allocate 128.00 MiB`

### The 4 Failed Hypotheses
1. **Hypothesis 1: Cyclic references in Python request objects.**
   * *Investigation*: Used `gc.collect()` and inspected `sys.getrefcount()`. Objects were being dereferenced, but VRAM didn't budge.
2. **Hypothesis 2: Block Allocator leakage.**
   * *Investigation*: Added strict block accounting invariants (`assert allocator.get_num_free_blocks() == TOTAL_BLOCKS` when idle). All blocks were returned cleanly to the pool.
3. **Hypothesis 3: PyTorch CUDA caching allocator fragmentation.**
   * *Investigation*: Called `torch.cuda.empty_cache()` every iteration. Allocated memory was still growing, meaning real tensor allocations were held by PyTorch, not cached buffer pools.
4. **Hypothesis 4: Triton autotuning caching or scratchpad leakage.**
   * *Investigation*: Disabled autotuning and forced static kernel configurations. The leak persisted unchanged.

### The Real Cause: The One-Line `inference_mode` Escape
In PyTorch, `@torch.inference_mode()` disables view tracking and autograd version counters for maximum performance. In the engine loop:
```python
@torch.inference_mode()
async def step(self):
    # Process prefill
    ...
```
However, during chunked prefill, a helper routine was calling a HuggingFace utility function that internally invoked a generator, briefly yielding control back to the async loop outside the active context manager. 

Because `torch.is_grad_enabled()` silently flipped to `True`, intermediate activation tensors from every forward pass were building an uncollectible backward execution graph attached to the process context!

### The Fix
```python
# Permanent guard inside execute_step:
with torch.inference_mode():
    outputs = self.model.forward(...)
```
Once applied, memory instantly flatlined at **959.2 MiB base allocated** and **967.3 to 975.4 MiB under full KV saturation**, completely curing the OOM.

---

## War Story 2: The Naive Server Tokenizer Concurrency Race

### The Symptom
Under Poisson request arrivals at $\lambda=0.5$ and $\lambda=1.0$, the naive baseline server intermittently threw `500 Internal Server Error` on ~3–7% of requests, with tracebacks pointing to tensor shape mismatches in `torch.cat`.

### The Investigation
Looking at the naive runner implementation:
```python
# naive_server.py
class NaiveHFModelRunner:
    def execute_batch(self, batch_requests):
        prompts = [req.prompt for req in batch_requests]
        # RACE CONDITION HERE:
        inputs = self.tokenizer(prompts, padding=True, return_tensors="pt")
        ...
```
In FastAPI, incoming HTTP requests arrive asynchronously on the event loop:
```python
@app.post("/v1/completions")
async def completions(req: CompletionRequest):
    tokens = tokenizer(req.prompt)  # Mutating shared tokenizer internal state!
```
Hugging Face's `PreTrainedTokenizerFast` contains internal mutable state when configuring padding tokens and truncation sequences. When the background runner thread invoked `self.tokenizer(prompts)` while a new request was hitting `/v1/completions`, the tokenizer's padding settings were clobbered mid-execution, outputting truncated token lists that crashed tensor assembly.

### The Fix
Eliminate all tokenizer invocations from the execution runner:
1. **Tokenize purely in the request handler**: Parse and convert text to immutable token IDs (`prompt_token_ids: List[int]`) upfront in the API layer.
2. **Assemble left-padded tensors with `torch.full`**: Construct batch tensors directly from integer lists:
```python
max_len = max(len(r.prompt_token_ids) for r in batch_requests)
input_ids = torch.full((len(batch_requests), max_len), pad_token_id, dtype=torch.long, device="cuda")
for i, req in enumerate(batch_requests):
    input_ids[i, max_len - len(req.prompt_token_ids):] = torch.tensor(req.prompt_token_ids)
```

### Result
Across 6 consecutive 200-request evaluation runs ($\lambda \in [0.5, 1.0]$, seeds 42, 43, 44), failures dropped to **exactly 0/1200**.
