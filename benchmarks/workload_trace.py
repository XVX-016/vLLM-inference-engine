"""
Workload trace generator for comparing serving engines.

Generates Poisson request arrival traces with lognormal prompt and output lengths,
saved to JSON so both Paged/Continuous and Naive/Static engines replay the exact same workload.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import List, Optional

import numpy as np


# Rich bank of diverse English sentences for synthetic prompt creation
VOCAB_SENTENCES = [
    "The transformer architecture relies on multi-head attention to capture relationships across sequence positions.",
    "In deep learning systems, key-value caching prevents redundant matrix multiplications across generation steps.",
    "Paged memory management divides physical memory into fixed-size chunks to eliminate external fragmentation.",
    "Continuous batching operates at the iteration level, scheduling tokens immediately rather than waiting for batches to finish.",
    "Prefill compute is compute-bound with high arithmetic intensity, while autoregressive decode is memory-bandwidth bound.",
    "Operating systems use page tables and translation lookaside buffers to map virtual addresses to physical pages.",
    "Large language model inference serving requires careful orchestration of GPU high-bandwidth memory and compute cores.",
    "When multiple clients stream tokens concurrently, early preemption and re-scheduling guarantee system responsiveness.",
    "Distributed tensor parallelism partitions weight matrices across multiple GPUs using collective communications.",
    "FlashAttention restructures the attention computation into tiled SRAM blocks to avoid slow High Bandwidth Memory accesses.",
    "KV-cache paging allows zero-waste copy-on-write sharing for parallel sampling and common system prompts.",
    "Autoregressive generation generates tokens one by one by computing the argmax or sampling from output logits.",
]


@dataclass
class TraceRequest:
    request_id: int
    arrival_time_s: float
    prompt: str
    prompt_len: int
    output_len: int
    ignore_eos: bool = True


def generate_synthetic_prompt(target_token_count: int, seed: int) -> str:
    """Generates synthetic prompt text of approximately target_token_count words/tokens."""
    rng = np.random.default_rng(seed)
    words: List[str] = []
    
    while len(words) < target_token_count:
        idx = rng.integers(0, len(VOCAB_SENTENCES))
        sent_words = VOCAB_SENTENCES[idx].split()
        words.extend(sent_words)

    trimmed = words[:target_token_count]
    return " ".join(trimmed)


def generate_workload_trace(
    num_requests: int = 100,
    arrival_rate: float = 10.0,  # lambda requests / second (Poisson process)
    prompt_mu: float = 3.5,      # lognormal mean parameter for prompt length
    prompt_sigma: float = 0.5,   # lognormal sigma parameter for prompt length
    output_mu: float = 3.0,      # lognormal mean parameter for output length
    output_sigma: float = 0.4,   # lognormal sigma parameter for output length
    min_prompt_len: int = 4,
    max_prompt_len: int = 512,
    min_output_len: int = 4,
    max_output_len: int = 256,
    seed: int = 42,
    ignore_eos: bool = True,
) -> List[TraceRequest]:
    """
    Generates a reproducible workload trace using a Poisson arrival process
    and lognormal distributions for prompt and completion lengths.
    """
    rng = np.random.default_rng(seed)

    # 1. Inter-arrival times ~ Exponential(lambda)
    inter_arrivals = rng.exponential(scale=1.0 / arrival_rate, size=num_requests)
    arrival_times = np.cumsum(inter_arrivals)
    # Start first request at t=0
    arrival_times = arrival_times - arrival_times[0]

    # 2. Prompt lengths ~ Lognormal(mu, sigma)
    raw_prompt_lens = rng.lognormal(mean=prompt_mu, sigma=prompt_sigma, size=num_requests)
    prompt_lens = np.clip(np.round(raw_prompt_lens).astype(int), min_prompt_len, max_prompt_len)

    # 3. Output lengths ~ Lognormal(mu, sigma)
    raw_output_lens = rng.lognormal(mean=output_mu, sigma=output_sigma, size=num_requests)
    output_lens = np.clip(np.round(raw_output_lens).astype(int), min_output_len, max_output_len)

    requests: List[TraceRequest] = []
    for i in range(num_requests):
        p_len = int(prompt_lens[i])
        o_len = int(output_lens[i])
        prompt_text = generate_synthetic_prompt(p_len, seed=seed + i)
        requests.append(
            TraceRequest(
                request_id=i,
                arrival_time_s=float(round(arrival_times[i], 4)),
                prompt=prompt_text,
                prompt_len=p_len,
                output_len=o_len,
                ignore_eos=ignore_eos,
            )
        )

    return requests


def save_trace_to_json(trace: List[TraceRequest], filepath: Path | str) -> None:
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = [asdict(r) for r in trace]
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def load_trace_from_json(filepath: Path | str) -> List[TraceRequest]:
    path = Path(filepath)
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return [TraceRequest(**item) for item in data]


def main():
    parser = argparse.ArgumentParser(description="Generate benchmark workload trace")
    parser.add_argument("--num-requests", type=int, default=100)
    parser.add_argument("--arrival-rate", type=float, default=8.0, help="Poisson lambda (req/sec)")
    parser.add_argument("--prompt-mu", type=float, default=3.5)
    parser.add_argument("--prompt-sigma", type=float, default=0.5)
    parser.add_argument("--output-mu", type=float, default=3.0)
    parser.add_argument("--output-sigma", type=float, default=0.4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str, default="benchmarks/workload_trace.json")
    args = parser.parse_args()

    trace = generate_workload_trace(
        num_requests=args.num_requests,
        arrival_rate=args.arrival_rate,
        prompt_mu=args.prompt_mu,
        prompt_sigma=args.prompt_sigma,
        output_mu=args.output_mu,
        output_sigma=args.output_sigma,
        seed=args.seed,
    )
    save_trace_to_json(trace, args.output)
    print(f"Generated {len(trace)} requests trace and saved to {args.output}")


if __name__ == "__main__":
    main()
