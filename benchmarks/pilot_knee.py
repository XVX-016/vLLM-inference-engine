"""
Pilot runner to locate each system's TTFT knee.
Requirement 3:
- One run per server at lambda in {0.25, 0.5, 1.0, 2.0, 4.0}
- 60 requests each
- Report TTFT knee, TTFT P99, E2E P99, token_tps, failures
"""
import asyncio
import json
import os
import sys
import time
from pathlib import Path
from benchmarks.run_benchmarks import (
    PORT_NAIVE,
    PORT_PAGED,
    RESULTS_DIR,
    start_naive,
    start_paged,
    stop_server,
    gen_trace,
    replay,
    compute_summary,
    choose_budget,
)

PILOT_LAMBDAS = [0.25, 0.5, 1.0, 2.0, 4.0]
PILOT_REQS = 60
PILOT_SEED = 42

async def run_pilot():
    print("=" * 70, flush=True)
    print("Starting Pilot Knee Search", flush=True)
    print(f"Lambdas: {PILOT_LAMBDAS}, Requests per point: {PILOT_REQS}, Seed: {PILOT_SEED}", flush=True)
    print("=" * 70, flush=True)

    # Use fixed budget of 7.0 MiB and MSL=192 so Naive gets exactly 3 slots
    budget_mib = 7.0
    msl = 192
    clamped_slots = 3
    print(f"KV Budget: {budget_mib} MiB, Max Seq Len: {msl}, Naive Slots: {clamped_slots}", flush=True)

    results = {"naive": {}, "paged": {}}

    for lam in PILOT_LAMBDAS:
        trace = gen_trace(PILOT_REQS, lam=lam, seed=PILOT_SEED)
        print(f"\n--- Pilot Point: lambda = {lam} req/s ---", flush=True)

        # 1. Run Naive Server
        print(f"  Starting Naive Server on :{PORT_NAIVE}...", flush=True)
        proc_n = start_naive(budget_mib, msl, PORT_NAIVE)
        try:
            # Short warmup (e.g. 5s)
            res_n, samp_n, dur_n = await replay(
                f"http://127.0.0.1:{PORT_NAIVE}", trace, warmup_s=5.0
            )
            summary_n = compute_summary(res_n, samp_n, dur_n, lam, budget_mib=budget_mib)
            results["naive"][str(lam)] = summary_n
            print(
                f"  [Naive] lam={lam}: completed={summary_n['completed']}/{summary_n['total_eval']} "
                f"failed={summary_n['failed_eval']} "
                f"tps={summary_n['token_tps']:.1f} "
                f"TTFT_P99={summary_n.get('ttft_p99')}s "
                f"E2E_P99={summary_n.get('e2e_p99')}s",
                flush=True
            )
        finally:
            stop_server(proc_n)

        # 2. Run Paged Server
        print(f"  Starting Paged Server on :{PORT_PAGED}...", flush=True)
        proc_p = start_paged(budget_mib, PORT_PAGED)
        try:
            res_p, samp_p, dur_p = await replay(
                f"http://127.0.0.1:{PORT_PAGED}", trace, warmup_s=5.0
            )
            summary_p = compute_summary(res_p, samp_p, dur_p, lam, budget_mib=budget_mib)
            results["paged"][str(lam)] = summary_p
            print(
                f"  [Paged] lam={lam}: completed={summary_p['completed']}/{summary_p['total_eval']} "
                f"failed={summary_p['failed_eval']} "
                f"tps={summary_p['token_tps']:.1f} "
                f"TTFT_P99={summary_p.get('ttft_p99')}s "
                f"E2E_P99={summary_p.get('e2e_p99')}s "
                f"[blocked={summary_p.get('paged_ever_blocked')} preempted={summary_p.get('paged_ever_preempted')}]",
                flush=True
            )
        finally:
            stop_server(proc_p)

    out_path = RESULTS_DIR / "pilot_knee_results.json"
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nPilot results saved to {out_path}", flush=True)

    # Summary Table
    print("\n" + "=" * 75, flush=True)
    print("PILOT KNEE SUMMARY TABLE", flush=True)
    print(f"{'Lambda':<8} | {'Naive TTFT P99':<16} | {'Naive Fail':<10} | {'Paged TTFT P99':<16} | {'Paged Fail':<10}", flush=True)
    print("-" * 75, flush=True)
    for lam in PILOT_LAMBDAS:
        n_res = results["naive"].get(str(lam), {})
        p_res = results["paged"].get(str(lam), {})
        n_ttft = f"{n_res.get('ttft_p99', 0.0):.3f}s" if n_res.get('ttft_p99') is not None else "N/A"
        p_ttft = f"{p_res.get('ttft_p99', 0.0):.3f}s" if p_res.get('ttft_p99') is not None else "N/A"
        n_fail = n_res.get("failed_eval", 0)
        p_fail = p_res.get("failed_eval", 0)
        print(f"{lam:<8} | {n_ttft:<16} | {n_fail:<10} | {p_ttft:<16} | {p_fail:<10}", flush=True)
    print("=" * 75, flush=True)

if __name__ == "__main__":
    asyncio.run(run_pilot())
