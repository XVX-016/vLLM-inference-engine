# Paged-KV Inference Engine

A from-scratch, mini-vLLM-style LLM serving engine, built in phases:

| Phase | Component | Status |
|-------|-----------|--------|
| 0 | Naive baseline engine (contiguous KV-cache) + instrumentation | scaffolded |
| 1 | Manual autoregressive decode loop (own KV-cache tensors) | scaffolded |
| 2 | Block-based paged memory allocator | stubbed |
| 3 | Triton paged-attention kernel | stubbed |
| 4 | Continuous batching scheduler | stubbed |
| 5 | FastAPI serving layer + benchmark harness | stubbed |
| 6 | Stretch: prefix caching / speculative decoding / KV quantization | not started |

## Why this exists

Standard LLM serving pre-allocates KV-cache for the worst-case sequence length,
wasting 60-80% of VRAM. This project rebuilds vLLM's core ideas — paged
KV-cache allocation and iteration-level continuous batching — from first
principles, in stages that are each independently testable and benchmarkable.

## Repo layout

```
engine/
  baseline/       Phase 0-1: naive engine, manual decode loop, instrumentation
  memory/         Phase 2: BlockAllocator, BlockTable, physical block pool
  kernels/        Phase 3: Triton paged-attention kernel + correctness tests
  scheduler/       Phase 4: continuous batching scheduler (waiting/running/swapped)
  serving/        Phase 5: FastAPI server, OpenAI-compatible endpoint
benchmarks/       Load-test harness, throughput/latency/VRAM plots
tests/            Unit + numerical-correctness tests for every phase
scripts/          Convenience run scripts
configs/          Model/engine config files (YAML)
```

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Requires a CUDA-capable GPU for anything past Phase 1 (Triton kernels won't
run meaningfully on CPU or Apple Silicon). Phase 0/1 can be smoke-tested on
CPU with a tiny model, but real VRAM numbers need a real GPU (a T4/A10 on
Colab/Lambda/RunPod is enough to start).

## Running each phase

```bash
# Phase 0: naive baseline, logs VRAM + tok/s to benchmarks/results/
python -m benchmarks.bench_baseline --model Qwen/Qwen2.5-0.5B --num-seqs 1 4 16

# Phase 1: manual decode loop sanity check (compares logits to HF generate())
python -m tests.test_manual_decode

# Phase 2 (once implemented): allocator unit tests
pytest tests/test_block_allocator.py -v

# Phase 3 (once implemented): kernel correctness vs naive PyTorch reference
pytest tests/test_paged_attention_kernel.py -v
```

## Development order

Build bottom-up, one verifiable layer at a time. Do not start Phase 2 until
Phase 0/1 produce believable numbers; do not start Phase 4 until Phase 3's
kernel passes numerical correctness tests against a naive reference. Every
phase should leave you with something that runs end-to-end, even if slow.
