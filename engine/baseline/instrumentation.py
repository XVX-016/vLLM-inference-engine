"""
Lightweight instrumentation for VRAM usage and decode throughput.

Used across every phase so results stay comparable: Phase 0's naive engine
and Phase 4's continuous-batching engine should be measured with the exact
same StepTimer/MemoryTracker so the final benchmark numbers are apples-to-apples.
"""
from __future__ import annotations

import time
import json
import dataclasses
from pathlib import Path
from typing import Optional

import torch


@dataclasses.dataclass
class StepRecord:
    step: int
    num_active_seqs: int
    tokens_generated_this_step: int
    step_time_s: float
    vram_allocated_bytes: int
    vram_reserved_bytes: int


@dataclasses.dataclass
class IterationRecord:
    iteration: int
    iteration_type: str  # "prefill" or "decode"
    running_count: int
    waiting_count: int
    free_blocks: int
    preemptions: int
    step_time_s: float
    tokens_generated: int = 0
    vram_allocated_bytes: int = 0
    vram_reserved_bytes: int = 0


@dataclasses.dataclass
class RequestRecord:
    seq_id: int
    prompt_len: int
    arrival_time_s: float
    time_to_first_token_s: Optional[float] = None
    finish_time_s: Optional[float] = None
    total_tokens_generated: int = 0

    @property
    def latency_s(self) -> Optional[float]:
        if self.finish_time_s is not None:
            return self.finish_time_s - self.arrival_time_s
        return None


class MemoryTracker:
    """Thin wrapper around torch.cuda memory stats. No-ops gracefully on CPU."""

    def __init__(self, device: Optional[torch.device] = None):
        self.device = device or (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )
        self.has_cuda = self.device.type == "cuda"
        if self.has_cuda:
            torch.cuda.reset_peak_memory_stats(self.device)

    def snapshot(self) -> tuple[int, int]:
        if not self.has_cuda:
            return 0, 0
        return (
            torch.cuda.memory_allocated(self.device),
            torch.cuda.memory_reserved(self.device),
        )

    def peak(self) -> tuple[int, int]:
        if not self.has_cuda:
            return 0, 0
        return (
            torch.cuda.max_memory_allocated(self.device),
            torch.cuda.max_memory_reserved(self.device),
        )


class RunLogger:
    """Collects StepRecords and IterationRecords for a single run and writes them to JSON."""

    def __init__(self, run_name: str, out_dir: str = "benchmarks/results"):
        self.run_name = run_name
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.records: list[StepRecord] = []
        self.iterations: list[IterationRecord] = []
        self.requests: dict[int, RequestRecord] = {}
        self._t_start = time.perf_counter()

    def log_step(
        self,
        step: int,
        num_active_seqs: int,
        tokens_generated_this_step: int,
        step_time_s: float,
        mem: MemoryTracker,
    ) -> None:
        allocated, reserved = mem.snapshot()
        self.records.append(
            StepRecord(
                step=step,
                num_active_seqs=num_active_seqs,
                tokens_generated_this_step=tokens_generated_this_step,
                step_time_s=step_time_s,
                vram_allocated_bytes=allocated,
                vram_reserved_bytes=reserved,
            )
        )

    def log_iteration(
        self,
        iteration: int,
        iteration_type: str,
        running_count: int,
        waiting_count: int,
        free_blocks: int,
        preemptions: int,
        step_time_s: float,
        tokens_generated: int = 0,
        mem: Optional[MemoryTracker] = None,
    ) -> None:
        """Logs per-iteration metrics for continuous batching."""
        allocated, reserved = (0, 0) if mem is None else mem.snapshot()
        self.iterations.append(
            IterationRecord(
                iteration=iteration,
                iteration_type=iteration_type,
                running_count=running_count,
                waiting_count=waiting_count,
                free_blocks=free_blocks,
                preemptions=preemptions,
                step_time_s=step_time_s,
                tokens_generated=tokens_generated,
                vram_allocated_bytes=allocated,
                vram_reserved_bytes=reserved,
            )
        )

    def record_request_arrival(self, seq_id: int, prompt_len: int, arrival_time_s: Optional[float] = None) -> None:
        """Records arrival of a request."""
        arr_time = arrival_time_s if arrival_time_s is not None else (time.perf_counter() - self._t_start)
        self.requests[seq_id] = RequestRecord(
            seq_id=seq_id,
            prompt_len=prompt_len,
            arrival_time_s=arr_time,
        )

    def record_request_first_token(self, seq_id: int, first_token_time_s: Optional[float] = None) -> None:
        """Records time-to-first-token for a request."""
        curr_time = first_token_time_s if first_token_time_s is not None else (time.perf_counter() - self._t_start)
        req = self.requests.get(seq_id)
        if req is not None and req.time_to_first_token_s is None:
            req.time_to_first_token_s = curr_time - req.arrival_time_s

    def record_request_finish(self, seq_id: int, total_tokens: int, finish_time_s: Optional[float] = None) -> None:
        """Records request finish time."""
        curr_time = finish_time_s if finish_time_s is not None else (time.perf_counter() - self._t_start)
        req = self.requests.get(seq_id)
        if req is not None:
            req.finish_time_s = curr_time
            req.total_tokens_generated = total_tokens

    def summary(self) -> dict:
        total_tokens = sum(r.tokens_generated_this_step for r in self.records)
        if not total_tokens and self.iterations:
            total_tokens = sum(it.tokens_generated for it in self.iterations)

        total_time = sum(r.step_time_s for r in self.records)
        if not total_time and self.iterations:
            total_time = sum(it.step_time_s for it in self.iterations)

        peak_alloc = max((r.vram_allocated_bytes for r in self.records), default=0)
        peak_reserved = max((r.vram_reserved_bytes for r in self.records), default=0)
        if self.iterations:
            peak_alloc = max(peak_alloc, max((it.vram_allocated_bytes for it in self.iterations), default=0))
            peak_reserved = max(peak_reserved, max((it.vram_reserved_bytes for it in self.iterations), default=0))

        # Per-request metrics summary
        req_latencies = [r.latency_s for r in self.requests.values() if r.latency_s is not None]
        req_ttfts = [r.time_to_first_token_s for r in self.requests.values() if r.time_to_first_token_s is not None]

        return {
            "run_name": self.run_name,
            "total_tokens": total_tokens,
            "total_time_s": total_time,
            "tokens_per_sec": total_tokens / total_time if total_time > 0 else 0.0,
            "peak_vram_allocated_bytes": peak_alloc,
            "peak_vram_reserved_bytes": peak_reserved,
            "peak_vram_allocated_mb": peak_alloc / (1024 ** 2),
            "peak_vram_reserved_mb": peak_reserved / (1024 ** 2),
            "wall_clock_s": time.perf_counter() - self._t_start,
            "total_requests": len(self.requests),
            "avg_ttft_s": sum(req_ttfts) / len(req_ttfts) if req_ttfts else 0.0,
            "avg_latency_s": sum(req_latencies) / len(req_latencies) if req_latencies else 0.0,
        }

    def save(self) -> Path:
        out_path = self.out_dir / f"{self.run_name}.json"
        payload = {
            "summary": self.summary(),
            "steps": [dataclasses.asdict(r) for r in self.records],
            "iterations": [dataclasses.asdict(it) for it in self.iterations],
            "requests": {
                seq_id: dataclasses.asdict(req) for seq_id, req in self.requests.items()
            },
        }
        out_path.write_text(json.dumps(payload, indent=2))
        return out_path
