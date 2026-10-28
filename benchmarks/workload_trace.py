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
from typing import Any, List, Optional

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


@dataclass
class SharedPrefixTrace:
    requests: List[TraceRequest]
    metadata: dict[str, Any]


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


def generate_shared_prefix_trace(
    tokenizer: Any,
    num_requests: int = 200,
    arrival_rate: float = 1.0,
    prefix_count: int = 4,
    system_prompt_tokens: int = 64,
    seed: int = 42,
    block_size: int = 16,
    ignore_eos: bool = True,
    suffix_mu: float = 3.5,
    suffix_sigma: float = 0.5,
    min_suffix_tokens: int = 4,
    max_suffix_tokens: int = 512,
    output_mu: float = 3.0,
    output_sigma: float = 0.4,
    min_output_tokens: int = 4,
    max_output_tokens: int = 256,
) -> SharedPrefixTrace:
    """Generate tokenizer-measured requests sharing one of a pool of system prompts."""
    if num_requests < 1 or prefix_count < 1 or system_prompt_tokens < 1:
        raise ValueError("num_requests, prefix_count, and system_prompt_tokens must be positive.")
    if arrival_rate <= 0 or block_size < 1:
        raise ValueError("arrival_rate and block_size must be positive.")

    rng = np.random.default_rng(seed)
    arrivals = np.cumsum(rng.exponential(scale=1.0 / arrival_rate, size=num_requests))
    arrivals -= arrivals[0]
    suffix_lengths = np.clip(
        np.round(rng.lognormal(suffix_mu, suffix_sigma, num_requests)).astype(int),
        min_suffix_tokens,
        max_suffix_tokens,
    )
    output_lengths = np.clip(
        np.round(rng.lognormal(output_mu, output_sigma, num_requests)).astype(int),
        min_output_tokens,
        max_output_tokens,
    )

    def encode(text: str) -> list[int]:
        return list(tokenizer.encode(text, add_special_tokens=True))

    system_prompts: list[str] = []
    for prefix_idx in range(prefix_count):
        # When generating multiple distinct prefixes, prepend a unique tag so
        # no two system prompts share their initial block(s).
        tag = f"Prefix {prefix_idx:04d}: " if prefix_count > 1 else ""
        candidate = tag + generate_synthetic_prompt(
            target_token_count=max(system_prompt_tokens * 3, 8),
            seed=seed * 1009 + prefix_idx,
        )
        candidate_ids = encode(candidate)
        if len(candidate_ids) < system_prompt_tokens:
            raise ValueError("Unable to generate enough system-prompt tokens.")
        system_prompt = tokenizer.decode(
            candidate_ids[:system_prompt_tokens],
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )
        system_ids = encode(system_prompt)
        if len(system_ids) != system_prompt_tokens:
            raise ValueError(
                "Tokenizer does not round-trip the requested system prompt length: "
                f"wanted {system_prompt_tokens}, got {len(system_ids)}."
            )
        system_prompts.append(system_prompt)

    requests: list[TraceRequest] = []
    prompt_token_ids: dict[int, list[int]] = {}
    prefix_assignments: dict[int, int] = {}
    for req_id in range(num_requests):
        prefix_idx = req_id % prefix_count
        suffix_text = generate_synthetic_prompt(
            target_token_count=int(suffix_lengths[req_id]),
            seed=seed * 65537 + req_id,
        )
        prompt_text = f"{system_prompts[prefix_idx]}\n\n{suffix_text}"
        token_ids = encode(prompt_text)
        share_prefix_ids = encode(f"{system_prompts[prefix_idx]}\n\n")
        if token_ids[: len(share_prefix_ids)] != share_prefix_ids:
            raise ValueError(
                "Prompt tokenization changed the system-prefix token sequence at the "
                f"boundary for request {req_id}; use a tokenizer-stable separator."
            )
        prompt_token_ids[req_id] = token_ids
        prefix_assignments[req_id] = prefix_idx
        requests.append(
            TraceRequest(
                request_id=req_id,
                arrival_time_s=float(round(arrivals[req_id], 6)),
                prompt=prompt_text,
                prompt_len=len(token_ids),
                output_len=int(output_lengths[req_id]),
                ignore_eos=ignore_eos,
            )
        )

    shareable_tokens = 0
    for prefix_idx in range(prefix_count):
        group_ids = [
            prompt_token_ids[req_id]
            for req_id, assigned_prefix in prefix_assignments.items()
            if assigned_prefix == prefix_idx
        ]
        if len(group_ids) < 2:
            continue
        common_tokens = min(len(ids) for ids in group_ids)
        for position in range(common_tokens):
            token = group_ids[0][position]
            if any(ids[position] != token for ids in group_ids[1:]):
                common_tokens = position
                break
        # The first request in the prefix group populates the cache (cold miss);
        # only subsequent requests (len(group_ids) - 1) can hit the cached prefix blocks.
        for ids in group_ids[1:]:
            shareable_tokens += min(
                (common_tokens // block_size) * block_size,
                max(0, ((len(ids) - 1) // block_size) * block_size),
            )

    total_prompt_tokens = sum(request.prompt_len for request in requests)
    metadata = {
        "seed": seed,
        "arrival_rate_rps": arrival_rate,
        "num_requests": num_requests,
        "prefix_count": prefix_count,
        "system_prompt_tokens": system_prompt_tokens,
        "block_size": block_size,
        "suffix_length_distribution": {
            "name": "lognormal",
            "mu": suffix_mu,
            "sigma": suffix_sigma,
            "min": min_suffix_tokens,
            "max": max_suffix_tokens,
        },
        "output_length_distribution": {
            "name": "lognormal",
            "mu": output_mu,
            "sigma": output_sigma,
            "min": min_output_tokens,
            "max": max_output_tokens,
        },
        "system_prompts": system_prompts,
        "theoretical_shareable_prompt_tokens": shareable_tokens,
        "total_prompt_tokens": total_prompt_tokens,
        "theoretical_shareable_fraction": (
            shareable_tokens / total_prompt_tokens if total_prompt_tokens else 0.0
        ),
    }
    return SharedPrefixTrace(requests=requests, metadata=metadata)


def save_shared_prefix_trace(trace: SharedPrefixTrace, filepath: Path | str) -> None:
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "metadata": trace.metadata,
        "requests": [asdict(request) for request in trace.requests],
    }
    with path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2)


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
    parser.add_argument("--prefix-count", type=int, default=0, help="Generate a shared-prefix trace with this pool size")
    parser.add_argument("--system-prompt-tokens", type=int, default=64)
    parser.add_argument("--model-name", type=str, default="Qwen/Qwen2.5-0.5B-Instruct")
    args = parser.parse_args()

    if args.prefix_count:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(args.model_name)
        shared_trace = generate_shared_prefix_trace(
            tokenizer=tokenizer,
            num_requests=args.num_requests,
            arrival_rate=args.arrival_rate,
            prefix_count=args.prefix_count,
            system_prompt_tokens=args.system_prompt_tokens,
            seed=args.seed,
        )
        save_shared_prefix_trace(shared_trace, args.output)
        print(
            f"Generated {len(shared_trace.requests)} shared-prefix requests "
            f"(P={args.prefix_count}, S={args.system_prompt_tokens}, "
            f"theoretical shareable fraction="
            f"{shared_trace.metadata['theoretical_shareable_fraction']:.4f}) "
            f"and saved to {args.output}"
        )
    else:
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
