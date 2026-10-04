# Expert streaming A/B benchmark

A = `STRATA_PREFILL_STREAM_AHEAD=0`; B = `1`. Positive gain means B is faster.
Times are batched prefill only. Warmup and residual-check runs are excluded.
Each sample starts a fresh engine; this is not a persistent-server benchmark.

| Workload | Tokens | Requested chunk | Memory / KV / cache / context | Status | Pairs | A ms | B ms | A tok/s | B tok/s | Gain |
|---|---:|---|---|---|---:|---:|---:|---:|---:|---:|
| ramp | 1041 | 1024 | config / config / config / 2048 | ok | 1 | 2303.4 | 2277.2 | 451.9 | 457.1 | +1.15% |
| ramp | 1041 | 256 | config / config / config / 2048 | ok | 1 | 5559.8 | 5485.1 | 187.2 | 189.8 | +1.36% |
| ramp | 529 | 1024 | config / config / config / 2048 | ok | 1 | 1632.6 | 1608.4 | 324.0 | 328.9 | +1.50% |
| ramp | 529 | 256 | config / config / config / 2048 | ok | 1 | 3130.1 | 2987.1 | 169.0 | 177.1 | +4.79% |

JSON/CSV include run ranges, p95 and paired bootstrap intervals (at least three pairs).
Intervals from small samples are exploratory, not proof of a hardware-wide speedup.
Compare A/B within a row. Different chunk sizes, prompts and capacities are different workloads.

Skipped combinations: 0. See summary.json for reasons.
