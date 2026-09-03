"""
Phase 0 benchmark: sweep concurrency levels against the naive static-cache
engine and record VRAM + throughput. Run this FIRST, before touching the
paged allocator, so you have real control-group numbers.

Usage:
    python -m benchmarks.bench_baseline --model Qwen/Qwen2.5-0.5B --num-seqs 1 4 16
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from engine.baseline.naive_baseline import NaiveStaticCacheEngine, GenerationRequest

SAMPLE_PROMPTS = [
    "Explain the difference between TCP and UDP in two sentences.",
    "Write a haiku about distributed systems.",
    "What causes VRAM fragmentation in LLM serving?",
    "Summarize the plot of a heist movie in one paragraph.",
    "List three benefits of paged memory management.",
    "Describe continuous batching in one sentence.",
    "What is the capital of France?",
    "Give a one-line definition of a GPU kernel.",
]


def build_requests(n: int, max_new_tokens: int) -> list[GenerationRequest]:
    return [
        GenerationRequest(prompt=SAMPLE_PROMPTS[i % len(SAMPLE_PROMPTS)], max_new_tokens=max_new_tokens)
        for i in range(n)
    ]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-0.5B")
    parser.add_argument("--num-seqs", type=int, nargs="+", default=[1, 4, 16])
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--max-seq-len", type=int, default=2048)
    args = parser.parse_args()

    engine = NaiveStaticCacheEngine(args.model, max_seq_len=args.max_seq_len)

    all_summaries = []
    for n in args.num_seqs:
        requests = build_requests(n, args.max_new_tokens)
        run_name = f"baseline_naive_n{n}"
        print(f"\n=== Running {run_name} ===")
        logger = engine.run_batch(requests, run_name=run_name)
        summary = logger.summary()
        print(json.dumps(summary, indent=2))
        all_summaries.append(summary)

    out_path = Path("benchmarks/results/baseline_sweep_summary.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(all_summaries, indent=2))
    print(f"\nSaved sweep summary to {out_path}")


if __name__ == "__main__":
    main()
