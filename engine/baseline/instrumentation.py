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
    """Collects StepRecords for a single run and writes them to JSON for plotting."""

    def __init__(self, run_name: str, out_dir: str = "benchmarks/results"):
        self.run_name = run_name
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.records: list[StepRecord] = []
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

    def summary(self) -> dict:
        total_tokens = sum(r.tokens_generated_this_step for r in self.records)
        total_time = sum(r.step_time_s for r in self.records)
        peak_alloc = max((r.vram_allocated_bytes for r in self.records), default=0)
        peak_reserved = max((r.vram_reserved_bytes for r in self.records), default=0)
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
        }

    def save(self) -> Path:
        out_path = self.out_dir / f"{self.run_name}.json"
        payload = {
            "summary": self.summary(),
            "steps": [dataclasses.asdict(r) for r in self.records],
        }
        out_path.write_text(json.dumps(payload, indent=2))
        return out_path
