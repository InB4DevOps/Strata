# Expert streaming A/B benchmark

A = `STRATA_PREFILL_STREAM_AHEAD=0`; B = `1`. Positive gain means B is faster.
Times are batched prefill only. Warmup and residual-check runs are excluded.
Each sample starts a fresh engine; this is not a persistent-server benchmark.

| Workload | Tokens | Requested chunk | Memory / KV / cache / context | Status | Pairs | A ms | B ms | A tok/s | B tok/s | Gain |
|---|---:|---|---|---|---:|---:|---:|---:|---:|---:|
| ramp | 4096 | 256 | config / config / config / 4160 | ok | 3 | 20843.5 | 20272.9 | 196.5 | 202.0 | +2.81% |
| ramp | 529 | 256 | config / config / config / 4160 | ok | 3 | 2845.3 | 2792.9 | 185.9 | 189.4 | +1.88% |
| ramp | 2048 | 768 | config / config / config / 4160 | ok | 3 | 5118.9 | 5013.6 | 400.1 | 408.5 | +2.10% |
| ramp | 2048 | 512 | config / config / config / 4160 | ok | 3 | 6071.9 | 5947.4 | 337.3 | 344.4 | +2.09% |
| ramp | 1024 | 128 | config / config / config / 4160 | ok | 3 | 7715.6 | 7571.5 | 132.7 | 135.2 | +1.90% |
| ramp | 2048 | 128 | config / config / config / 4160 | ok | 3 | 16529.7 | 16155.6 | 123.9 | 126.8 | +2.32% |
| ramp | 1024 | 512 | config / config / config / 4160 | ok | 3 | 3007.5 | 2936.4 | 340.5 | 348.7 | +2.42% |
| ramp | 256 | 128 | config / config / config / 4160 | ok | 3 | 1701.5 | 1674.9 | 150.5 | 152.8 | +1.59% |
| ramp | 529 | 512 | config / config / config / 4160 | ok | 3 | 2091.5 | 2030.6 | 252.9 | 260.5 | +3.00% |
| ramp | 529 | 768 | config / config / config / 4160 | ok | 3 | 1570.1 | 1543.6 | 336.9 | 342.7 | +1.72% |
| ramp | 256 | 512 | config / config / config / 4160 | ok | 3 | 1183.7 | 1169.4 | 216.3 | 218.9 | +1.22% |
| ramp | 256 | 768 | config / config / config / 4160 | ok | 3 | 1185.9 | 1167.5 | 215.9 | 219.3 | +1.58% |
| ramp | 256 | 1024 | config / config / config / 4160 | ok | 3 | 1189.2 | 1171.8 | 215.3 | 218.5 | +1.48% |
| ramp | 4096 | 1024 | config / config / config / 4160 | ok | 3 | 6518.4 | 6523.4 | 628.4 | 627.9 | -0.08% |
| ramp | 4096 | 512 | config / config / config / 4160 | ok | 3 | 12391.8 | 12091.5 | 330.5 | 338.8 | +2.48% |
| ramp | 1024 | 256 | config / config / config / 4160 | ok | 3 | 4773.6 | 4701.5 | 214.5 | 217.8 | +1.53% |
| ramp | 529 | 1024 | config / config / config / 4160 | ok | 3 | 1572.8 | 1536.5 | 336.3 | 344.3 | +2.36% |
| ramp | 2048 | 1024 | config / config / config / 4160 | ok | 3 | 3270.4 | 3266.3 | 626.2 | 627.0 | +0.13% |
| ramp | 4096 | 768 | config / config / config / 4160 | ok | 3 | 10273.6 | 10065.7 | 398.7 | 406.9 | +2.07% |
| ramp | 529 | 128 | config / config / config / 4160 | ok | 3 | 4099.8 | 4009.3 | 129.0 | 131.9 | +2.26% |
| ramp | 1024 | 1024 | config / config / config / 4160 | ok | 3 | 1688.7 | 1699.0 | 606.4 | 602.7 | -0.61% |
| ramp | 2048 | 256 | config / config / config / 4160 | ok | 3 | 10032.6 | 9869.6 | 204.1 | 207.5 | +1.65% |
| ramp | 1024 | 768 | config / config / config / 4160 | ok | 3 | 3090.4 | 3027.0 | 331.3 | 338.3 | +2.09% |
| ramp | 256 | 256 | config / config / config / 4160 | ok | 3 | 1181.4 | 1163.6 | 216.7 | 220.0 | +1.53% |
| ramp | 4096 | 128 | config / config / config / 4160 | ok | 3 | 34641.9 | 34116.3 | 118.2 | 120.1 | +1.54% |

JSON/CSV include run ranges, p95 and paired bootstrap intervals (at least three pairs).
Intervals from small samples are exploratory, not proof of a hardware-wide speedup.
Compare A/B within a row. Different chunk sizes, prompts and capacities are different workloads.

Skipped combinations: 0. See summary.json for reasons.
