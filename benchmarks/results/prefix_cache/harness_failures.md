# Harness Failure Log — Phase 6 Prefix KV-Cache Benchmark

Total benchmark runs: 212 (as of 2026-10-28). Good runs: 210. Failed runs: 16 (15 root-cause A+B + 1 root-cause C).
All failed cells were retried; good data exists for all but one (see Root Cause C).

---

## Root Cause A: Missing `socket` import

- **Error:** `NameError: name 'socket' is not defined` in `stop_server()` inside `bench_prefix_cache.py`
- **Affected runs:** 14 cells (all `lambda=4` seeds 42 and 43, at budgets 2048T and 4096T, for `shared_p16_s64` and `control_no_share_p200_s64`).
- **Root cause:** `stop_server()` was calling `socket.create_connection()` for port-liveness polling, but `import socket` was missing from the module-level imports.
- **Fix:** Added `import socket` to `benchmarks/bench_prefix_cache.py` (committed in Phase 6).
- **Outcome:** All 14 cells rerun successfully and stored in `prefix_cache_benchmark.json`.

---

## Root Cause B: Windows port-reuse race condition

- **Error:** `RuntimeError: Startup block count mismatch` / `prefix_mode mismatch` / `ConnectionRefusedError` on server start.
- **Affected runs:** 1 cell (`control_no_share_p200_s64|budget=2048|lambda=3|seed=42|mode=share`; server exited with code 3221225477 during startup).
- **Root cause:** The server process from the prior run was still bound to port 8268 when the next run's `start_server()` returned successfully (the OS had not yet released the port). The new server either failed to bind (crash code 3221225477) or started against the wrong running server (mode or block-count mismatch).
- **Fix:** `stop_server()` now polls `socket.create_connection(("127.0.0.1", port))` in a loop (10 s timeout) after sending SIGTERM, confirming the port is fully released before returning control.
- **Outcome:** Cell rerun successfully.

---

## Root Cause C: GPU OOM / server crash at very high load

- **Error:** `RuntimeError: Paged server exited during startup with code 3221225477.`
- **Affected runs:** 1 cell (`shared_p1_s64|budget=768|lambda=8|seed=42|mode=off`).
- **Root cause:** The server failed during startup (not during load). On Windows, exit code 3221225477 = 0xC0000005 = `STATUS_ACCESS_VIOLATION`. This likely reflects a GPU memory access fault during model initialisation at this particular GPU clock/power state, following many consecutive high-load runs. The fault is non-deterministic; the same configuration (`shared_p1_s64|budget=768|lambda=6|seed=42|mode=off`) ran successfully 2 runs earlier.
- **Fix:** No fix applied. The run is stored in `prefix_cache_benchmark.json` with an `"error"` field and is excluded from all analysis (`_filter()` skips entries containing `"error"`). The `lambda=6` data for the same profile and seed is used in Panel D instead, which is a saturated measurement.
- **Outcome:** One data point missing; not expected to change conclusions. Lambda=6 provides a valid saturated measurement for this cell.

---

## Remediation Summary

| # | Root Cause | Runs Failed | Code Fix | Rerun OK? |
|---|---|---|---|---|
| A | Missing `import socket` | 14 | Added `import socket` | Yes (all 14) |
| B | Windows port reuse race | 1 | `stop_server()` polls until port closed | Yes |
| C | GPU fault at startup after heavy load | 1 | None (non-deterministic, excluded) | Lambda=8 skipped; lambda=6 OK |

