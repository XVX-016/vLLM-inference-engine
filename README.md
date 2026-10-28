# Paged-KV Inference Engine

A lightweight, from-scratch LLM serving engine built from first principles to demonstrate **paged KV-cache memory allocation** and **iteration-level continuous batching** (mini-vLLM).

> **Core Result**: Under an equal KV memory budget of 768 tokens (9.0 MiB), paged continuous batching sustains **1.2 to 1.5× the throughput** of static batching at saturation. An ablation capping paged batch size to 3 sequences matches static batching, isolating the throughput gain to memory efficiency and concurrency.

---

## Architecture Overview

```mermaid
graph TD
    Client["Client / Load Generator"] -->|HTTP /v1/completions| FastAPIServer["FastAPI Server (api_server.py)"]
    FastAPIServer -->|Async Queue| AsyncEngine["AsyncEngine (async_engine.py)"]
    
    subgraph Core Engine Execution
        AsyncEngine --> Scheduler["Continuous Batching Scheduler"]
        Scheduler --> BlockAllocator["Block Allocator (block_allocator.py)"]
        BlockAllocator --> PhysicalPool["Physical KV Cache Pool (blocks x block_size)"]
        
        Scheduler --> PagedRunner["Model Runner (paged_runner.py)"]
        PagedRunner --> PagedAttn["Paged Attention Integration"]
        PagedAttn --> GPU["NVIDIA GPU (Custom Triton Kernel / Flash-Paged)"]
    end
```

### Components
1. **Physical Block Pool (`engine/memory/block_pool.py`)**: Pre-allocated contiguous GPU tensor representing fixed-size KV blocks (e.g. 16 tokens/block).
2. **Block Allocator (`engine/memory/block_allocator.py`)**: Manages free/used block indices, dynamic allocation per sequence, and copy-on-write mapping.
3. **Continuous Batching Scheduler (`engine/scheduler/continuous_batching.py`)**: Iteration-level scheduling managing request states (`WAITING`, `RUNNING`, `SWAPPED`). Interleaves chunked prefill and single-token decode steps without head-of-line blocking.
4. **Paged Model Runner (`engine/serving/paged_runner.py`)**: Executes forward passes with Qwen2.5-0.5B-Instruct using gathered block tables.
5. **FastAPI Serving Layer (`engine/serving/api_server.py`)**: OpenAI-compatible `/v1/completions` endpoint with streaming SSE, disconnect reclamation, and prometheus-style `/metrics`.

---

## Evaluation & Empirical Results

All evaluations were conducted under strict invariant controls:
- **Equal KV Memory Budget**: Exactly 9.0 MiB of KV cache.
  - **Naive Static Server**: 3 sequence slots × 256 Maximum Sequence Length (MSL) = **768 tokens**.
  - **Paged Continuous Server**: 48 physical blocks × 16 tokens/block = **768 tokens**.
- **Model**: `Qwen/Qwen2.5-0.5B-Instruct` (FP16 weights: 959.2 MiB in-process memory).
- **Environment**: NVIDIA GeForce RTX 3050 Laptop GPU (4 GB VRAM), AC Power connected, Windows Best Performance scheme.
- **Sysmem Policy**: NVIDIA Control Panel 3D Settings set to *"CUDA - Sysmem Fallback Policy: Prefer No Sysmem Fallback"*.

### Throughput & Goodput vs. Offered Load
Below saturation ($\lambda \le 2.0$), both systems are equal: throughput matches offered load, and P99 Time-To-First-Token (TTFT) remains under 0.45 s.

At saturation ($\lambda \in [3.0, 4.0]$):
- **Naive Static Batching** plateaus at **43.4 to 48.5 tok/s** due to head-of-line blocking and internal slot fragmentation.
- **Paged Ablation (`max_batch_size=3`)** plateaus at **40.0 to 43.4 tok/s**, matching the static server and proving that concurrency limits explain the ceiling.
- **Paged Continuous (Unbounded)** breaks through to **56.4 to 65.6 tok/s** (**1.2 to 1.5× higher throughput** across seeds).

| Arrival Rate $\lambda$ | Seed | Offered Load | Naive Static | Paged Ablation (BS=3) | Paged Unbounded | Speedup Range |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **0.5** | 42–44 | ~10–11 tok/s | ~10.7 tok/s | — | ~10.7 tok/s | 1.0× (unsat.) |
| **1.0** | 42–44 | ~19–23 tok/s | ~21.4 tok/s | — | ~21.4 tok/s | 1.0× (unsat.) |
| **1.5** | 42 | 33.54 tok/s | 33.04 tok/s | 33.22 tok/s | 33.23 tok/s | 1.0× |
| **1.5** | 43 | 28.77 tok/s | 28.43 tok/s | 28.57 tok/s | 28.55 tok/s | 1.0× |
| **2.0** | 42 | 44.72 tok/s | 43.44 tok/s | 43.94 tok/s | 44.15 tok/s | 1.0× |
| **2.0** | 43 | 38.36 tok/s | 37.88 tok/s | 37.95 tok/s | 37.90 tok/s | 1.0× |
| **3.0** | 42 | 67.08 tok/s | 48.55 tok/s | 41.80 tok/s | **65.62 tok/s** | **1.35× (seed 42)** |
| **3.0** | 43 | 57.54 tok/s | 47.91 tok/s | 39.97 tok/s | **56.43 tok/s** | **1.18× (seed 43)** |
| **4.0** | 42 | 89.44 tok/s | 44.46 tok/s | 43.38 tok/s | **64.78 tok/s** | **1.46× (seed 42)** |
| **4.0** | 43 | 76.72 tok/s | 43.40 tok/s | 42.00 tok/s | **59.22 tok/s** | **1.36× (seed 43)** |

### Latency (TTFT P99)
- Below saturation ($\lambda \le 2.0$), both naive and paged servers achieve P99 TTFT between **0.21 s and 0.45 s**.
- At $\lambda=4.0$, Naive P99 TTFT spikes to **2.05 – 2.30 s**, while Paged Unbounded sustains **0.65 – 0.68 s** (a **3 to 3.5× latency improvement**).

### Decode Step-Time Benchmark (Interleaved ABAB Blocks after 30s Warmup)

| Batch Size | Plain HF Median | Paged Median | Speedup (Paged / Plain HF) |
| :---: | :---: | :---: | :---: |
| **BS = 1** | 42.93 ms | 40.83 ms | **1.05×** |
| **BS = 3** | 43.75 ms | 46.99 ms | **0.93×** |
| **BS = 4** | 43.94 ms | 48.91 ms | **0.90×** |
| **BS = 8** | 45.35 ms | 60.78 ms | **0.75×** |

> **Analysis**: Single-step decode time is roughly flat (~45 ms) for Plain HF across batch sizes 1 to 8, which explains why aggregate throughput scales linearly with concurrency. Paged attention is slightly slower at higher batch sizes (0.75× at BS=8) due to Python scheduling overhead and block-table metadata dispatch. Eliminating this step-time gap via CUDA Graphs or C++ bindings is the primary architectural headroom.

---

## Prefix Caching (Radix / Chain-Keyed Block Sharing)

Phase 6 implements prefix caching directly in the memory allocator and continuous scheduler, distinguishing between **memory sharing** (enabling higher concurrent sequence density within the fixed KV budget) and **compute skipping** (bypassing prefill forward passes for cached prefix tokens).

### Design & Mechanics
1. **Immutable Full-Block Cache**: Only complete blocks of size 16 are cached. Blocks are keyed by `(parent_block_key, tuple(token_ids_in_block))`, using the exact token ID tuple to eliminate hash collision risks.
2. **Cached-Free LRU Pool**: When a sequence finishes or releases a cached block, its reference count decrements to 0. Rather than returning immediately to the unkeyed free pool, it moves to an `OrderedDict`-backed LRU `cached_free` pool.
3. **Lazy Eviction on Demand**: New allocations take unkeyed free blocks first. If plain free blocks are exhausted, the oldest block in `cached_free` is evicted (unregistered from the cache) and reused. `num_available()` accurately counts both free and cached-free blocks.
4. **Last Prompt Token Recompute Invariant**: If a request matches prefix blocks up to prompt length $L$, the maximum usable cached tokens are capped at $\min(h, L - 1)$ rounded down to a block boundary. Recomputing at least the final prompt token ensures fresh, non-divergent next-token logits.
5. **Decode Write Isolation**: An explicit invariant assertion ensures no decode step ever writes into a block with `ref_count > 1`. Shared prefix blocks are strictly read-only.

### Invariants & Testing
Prefix caching invariants are enforced continuously and verified via property tests:
- **Conservation of Blocks**: $\text{Referenced Blocks} + \text{Plain Free} + \text{Cached Free} == \text{Total Pool Size}$ at every scheduling step and memory operation.
- **Reference Count Parity**: For every block in any active sequence's block table, its reference count exactly equals the number of sequences sharing that physical block.
- **Randomized State Machine**: Tested against a 4,000-operation randomized transition model across multiple seeds (`test_prefix_cache_randomized_state_machine`), exercising random allocate, register, acquire, release, lookup, and preemption cycles.
- **Token Output Parity**: Greedy completions with `prefix_mode="full"` and `prefix_mode="share"` match `prefix_mode="off"` bit-identically except for occasional floating-point near-ties where top-1/top-2 logit margins are under $0.05$.

### Empirical Results & Invariant Analysis (Phase 6 Matrix)

![Prefix Caching 4-Panel Results](benchmarks/results/prefix_cache/prefix_cache_4panel.png)

#### 1. Core Throughput & Dose-Response Scaling (768-Token Budget)
Across all runs, **throughput gains strictly scale with prefix sharing density**, exhibiting a clear monotonic dose-response relationship:

| Profile ($S=64$, Budget = 768T) | Offered Load $\lambda$ | `off` Throughput | `share` Throughput | `full` Throughput | Measured Hit Rate | Throughput Speedup |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **P = 1** (High sharing) | $\lambda=3.0$ | 38.9 tok/s (s42) / 39.1 (s43) | 65.6 tok/s (s42) / 56.5 (s43) | 65.6 tok/s (s42) / 56.5 (s43) | **49.8%** | **1.45–1.69×** |
| | $\lambda=4.0$ | 38.8 tok/s (s42) / 41.7 (s43) | 68.5 tok/s (s42) / 68.7 (s43) | 65.8 tok/s (s42) / 67.9 (s43) | **49.9%** | **1.63–1.77×** |
| **P = 4** (Moderate sharing) | $\lambda=3.0$ | 36.5 tok/s (s42) / 40.0 (s43) | 55.5 tok/s (s42) / 54.1 (s43) | 55.2 tok/s (s42) / 54.6 (s43) | **48.3%** | **1.35–1.52×** |
| | $\lambda=4.0$ | 36.7 tok/s (s42) / 36.1 (s43) | 56.6 tok/s (s42) / 54.4 (s43) | 55.0 tok/s (s42) / 57.8 (s43) | **48.4%** | **1.51–1.60×** |
| **P = 16** (Eviction stress test) | $\lambda=3.0$ | 38.9 tok/s (s42) / 39.0 (s43) | 41.6 tok/s (s42) / 42.2 (s43) | 41.1 tok/s (s42) / 42.0 (s43) | **4.0%** | **1.06–1.08×** |
| | $\lambda=4.0$ | 38.7 tok/s (s42) / 41.1 (s43) | 42.6 tok/s (s42) / 42.3 (s43) | 41.3 tok/s (s42) / 42.8 (s43) | **4.1%** | **1.03–1.10×** |
| **Control** (Zero-sharing, $P=200$) | $\lambda=3.0$ | 26.7 tok/s | 28.4 tok/s | 26.0 tok/s | **0.0%** | **1.00× (Within noise)** |

- **Hit-Rate Disambiguation**: For P=4, the measured hit rate is **48.3%** out of a theoretical shareable ceiling of **50.6%** ($48\text{ shared tokens} / 95\text{ avg tokens}$), achieving **95.4% cache efficiency**. For P=16 under the 768-token budget, the measured hit rate collapses to **4.0%** (**7.9% efficiency**) because 16 active prefixes require $16 \times 64 = 1,024$ tokens, exceeding the 768-token physical capacity and causing continuous LRU eviction thrashing.
- **Zero-Sharing Control Overhead**: In the $P=200$ control (where unique prefix headers guarantee a clean 0.0% hit rate by construction), throughputs for `off`, `share`, and `full` are indistinguishable within 2-seed noise (26.7, 28.4, 26.0 tok/s), confirming zero runtime penalty when prefixes cannot be shared.

#### 2. Fair Caching Scaling: P=16 at Higher Budgets
When the pool budget is expanded to comfortably hold all 16 distinct system prompts, cache thrashing vanishes and throughput scales proportionally:

| KV Budget | Mode | Throughput (tok/s) | Hit Rate | Evictions | Preemptions | TTFT P50 (s) | Speedup vs `off` |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **768 Tokens** (Eviction Stress) | `off` | 38.9 tok/s | 0.0% | 0 | 23 | 24.04 s | 1.00× |
| | `share` | 41.6 tok/s | 4.0% | 1,528 | 26 | 21.58 s | 1.07× |
| | `full` | 41.1 tok/s | 3.7% | 1,545 | 24 | 21.37 s | 1.06× |
| **2048 Tokens** | `off` | 45.0 tok/s | 0.0% | 0 | 8 | 7.61 s | 1.00× |
| | `share` | 54.1 tok/s | 25.1% | 509 | 11 | 4.27 s | **1.20×** |
| | `full` | 52.9 tok/s | 22.9% | 523 | 15 | 5.08 s | **1.18×** |
| **4096 Tokens** (Full Residency) | `off` | 49.5 tok/s | 0.0% | 0 | 10 | 4.75 s | 1.00× |
| | `share` | 67.1 tok/s | 49.7% | 190 | 0 | 0.57 s | **1.36×** |
| | `full` | 66.3 tok/s | 49.8% | 186 | 0 | 0.46 s | **1.34×** |

#### 3. Anatomy of the TTFT Disparity: Memory Effect vs Compute Skipping
Under long system prefixes ($P=1, S=512$, budget = 4096 tokens, $\lambda \in \{0.5, 1.0\}$), naive execution reported massive TTFT gaps between `off` and `share`/`full`:
- At $\lambda=0.5$: `off` TTFT P50 = **2.36–7.34 s** (up to 8 preemptions), while `share`/`full` TTFT P50 = **0.16–0.19 s** (0 preemptions).
- At $\lambda=1.0$: `off` TTFT P50 = **69.98–83.53 s** (up to 13 preemptions), while `share`/`full` TTFT P50 = **0.16–0.17 s** (0 preemptions).

**The Root Cause is Memory Sharing, Not Compute FLOPs**:
Because each 512-token prefix occupies 32 blocks, unshared execution (`off`) quickly exhausts the 4096-token pool (which holds only 8 concurrent full-length requests), forcing severe queueing delays and repeated preemption cycles. Prefix sharing (`share`) amortizes the 32 blocks across all requests, reducing active memory consumption by over $85\%$, completely eliminating preemptions, and maintaining near-instantaneous queue clearance.

**Prefill Compute-Skip Profiling**:
In-process GPU microbenchmarking isolating raw forward math (with an isolated 481-token prefix and an 8-token suffix) demonstrates:
- **Full prefill (489 tokens)**: $53.84\text{ ms}$ (launch-bound on RTX Laptop GPU)
- **Suffix prefill (8 tokens, 481 cached)**: $46.31\text{ ms}$
- **True raw compute saving**: **$7.53\text{ ms}$ ($1.16\times$ speedup)**.

Because single-step forward passes at this parameter scale are launch-bound (~45–50 ms per forward step), raw prefill compute skipping saves only tens of milliseconds. The multi-second TTFT reductions observed in live serving are entirely queueing delay mitigations afforded by memory deduplication.

#### 4. Hardware Telemetry & Thermal Clocks
Logged GPU graphics clocks across all matrix runs:
- `off` runs: Mean $1,038.9\text{ MHz}$ (median $795\text{ MHz}$, range $210–2,565\text{ MHz}$, $n=1,679$)
- `share` runs: Mean $1,099.0\text{ MHz}$ (median $810\text{ MHz}$, range $210–2,565\text{ MHz}$, $n=1,373$)
- `full` runs: Mean $937.0\text{ MHz}$ (median $600\text{ MHz}$, range $210–2,565\text{ MHz}$, $n=1,395$)

Latin-square mode interleaving within each $(profile, \lambda, seed)$ cell successfully averaged thermal drift across modes. Minor throttling into lower P-states occurred during long saturation runs without creating systematic bias.

---

## Known Limitations

1. **Model Scope**: Validated on single-GPU `Qwen/Qwen2.5-0.5B-Instruct` (FP16).
2. **Sampling**: Supports greedy decoding (`temperature=0.0`); top-p / top-k sampling kernels are omitted.
3. **Preemption Strategy**: Memory exhaustion triggers recompute preemption (aborting tail tokens and re-running prefill upon block availability) rather than host-memory swapping.
4. **Hardware**: Single laptop GPU setup (RTX 3050 Laptop GPU, 4 GB VRAM); multi-GPU tensor parallelism is not implemented.
5. **Trace Workload**: Synthetic Poisson arrival traces generated from random prompts and target lengths.

---

## The Debugging Story

Building this engine exposed several subtle systems bugs:
1. **The Phantom Memory Leak**: Under continuous batching, memory steadily expanded across requests until hitting OOM. Four hypotheses failed: tensor reference cycles, block table leaks, allocator double-frees, and Triton cache fragmentation. The actual root cause was a one-line PyTorch behavior: executing inside `@torch.inference_mode()` on the request runner loop prevented autograd tracking, but sub-calls invoked during prefill reset the inference context. Wrapping all engine step routines in explicit inference mode permanently flattened memory to 959–975 MiB.
2. **The Naive Tokenizer Concurrency Race**: Under Poisson load, the naive server experienced intermittent `500 Internal Server Errors`. The runner thread was calling mutating tokenizer methods (`tokenizer(prompts, ...)` and `set_truncation_and_padding()`) concurrently with the FastAPI request handler. The fix refactored `NaiveHFModelRunner` to take pre-tokenized IDs from the async handler and build left-padded tensors purely with `torch.full`, driving server failure rates to exactly zero.
3. **Preemption Variance Across Seeds**: At $\lambda=4.0$, seed 42 experienced 37 preemptions while seed 43 experienced 0. Deeper trace analysis revealed that seed 42 contained a 142-token burst prompt with a cluster of 16 arrivals within a 2.0-second window, exhausting the 768-token budget temporarily and forcing preemption, whereas seed 43's arrivals were more evenly spaced.

---

## Setup & Reproduction

### Prerequisites
- Python 3.10 or 3.11
- NVIDIA GPU with CUDA 12.x (or CPU for unit tests)

```bash
git clone https://github.com/your-username/vLLM-inference-engine.git
cd vLLM-inference-engine
python -m venv .venv
source .venv/bin/activate  # Or .venv\Scripts\activate on Windows
pip install -r requirements.txt
```

### Reproduce Evaluation Figures
To reproduce the consolidated evaluation figure from the validated experimental runs:
```bash
python benchmarks/reproduce_figures.py
# Generated plot saved to: results/serving_evaluation_consolidated.png
```

### Run CI Unit Tests (CPU Runnable)
```bash
python -m pytest tests/test_block_allocator.py tests/test_scheduler_continuous_batching.py tests/test_paged_attention_reference.py tests/test_paged_attention_prefix_cache.py -v
```
