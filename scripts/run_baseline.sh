#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

MODEL="${1:-Qwen/Qwen2.5-0.5B}"

echo "Running Phase 0 baseline sweep with model: $MODEL"
python -m benchmarks.bench_baseline --model "$MODEL" --num-seqs 1 4 16 --max-new-tokens 64
