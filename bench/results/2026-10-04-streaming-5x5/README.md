# Expert-streaming A/B: five prompt lengths by five chunk sizes

**Historical base:** this matrix was measured on `4fa31ee` plus the local streaming/profiling
changes, before moving the change onto upstream `main` at `6f32ec0`. It is not a measurement
of the updated base. The original manifest, binary hash and raw results are retained;
the [post-update smoke checks](../2026-10-04-streaming-main-smoke/README.md) are separate.

Completed on 2026-10-04: **25 cases passed, none failed or skipped**. There are 200
successful engine runs: 50 discarded warmup runs and 150 measured runs (three A/B
pairs per cell). Every pair passed the output-token, GDN-state-hash, actual chunk-count,
GPU and expert-placement checks. Residual dumps were disabled for this timing matrix.

## Setup

- NVIDIA GeForce RTX 3060, 12,288 MiB, driver 580.178.04; Intel Core i7-12700KF; Linux.
- Coder IQ1_M native pack, INT8 KV, fixed 4,160-token context capacity across all cases,
  `--kv-resident 0`, automatic expert cache, pinned resident expert arena.
- Synthetic ramp: token IDs `100 + i % 200`. Each input includes one additional final
  prompt token handled by decoding. One output token, greedy, `--pcie-frac 0`.
- A: `STRATA_PREFILL_STREAM_AHEAD=0`. B: `1`. Both use the same rebuilt binary.
- Seeded case shuffle, alternating A/B order, one warmup pair and three measured pairs
  per cell. Each run starts a fresh engine; primary times exclude model loading.
- Profiling disabled. No other GPU workload was running at the start; engine processes
  ran sequentially, and the GPU was idle again after completion.
- Engine SHA-256: `ed25fdcb42418bb0321fcf1c54c2f56c1db3beedc9684d7015180d194a62e326`.
  Revision and local modifications, hardware, model asset metadata and full configuration
  are recorded in [manifest.json](manifest.json).

## Measured throughput gain

Each cell is `100 * (median old prefill ms / median new prefill ms - 1)`.
Positive means the new schedule is faster. Columns are **requested** chunk sizes;
a prompt shorter than its requested chunk uses its actual shorter length.

| Fresh prefill tokens | Chunk 128 | Chunk 256 | Chunk 512 | Chunk 768 | Chunk 1024 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 256 | +1.59% | +1.53% | +1.22% | +1.58% | +1.48% |
| 529 | +2.26% | +1.88% | +3.00% | +1.72% | +2.36% |
| 1024 | +1.90% | +1.53% | +2.42% | +2.09% | -0.61% |
| 2048 | +2.32% | +1.65% | +2.09% | +2.10% | +0.13% |
| 4096 | +1.54% | +2.81% | +2.48% | +2.07% | -0.08% |

The 22 routed-only combinations all had positive median gains, ranging from **1.22% to
3.00%** on this workload/machine. The three pure 1,024-token-chunk controls showed
**-0.61%, +0.13%, and -0.08%**; they do not demonstrate an improvement for the full-stream
path. The 256- and 529-token prompts still use routed-only processing in the 1,024 column.

Three pairs per cell give a limited estimate of variability, not a universal GPU claim.
The JSON/CSV reports include ranges, paired gains and exploratory paired bootstrap
intervals. This is a synthetic throughput test, not a natural-prompt or answer-quality test.

The gains do not imply that smaller chunks are faster overall. For example, the new
schedule processes the 4,096-token prompt at **120.1 tok/s with chunk 128**, **202.0 tok/s
with chunk 256**, and **627.9 tok/s with chunk 1024**. Chunk choice changes total work and
transfer traffic much more than this scheduling optimization does.

## Detailed results and reproduction

- [summary.md](summary.md): old/new milliseconds and tokens/sec for every cell.
- [summary.csv](summary.csv), [summary.json](summary.json): metrics, spread and pair statistics.
- `cases/`: all 200 raw engine logs and parsed run records, plus checked per-case reports.
- `prompts/`: the exact five input token files, named by SHA-256.

Command executed from the repository root:

```sh
bash bench-expert-streaming-suite.sh \
  --lengths 256,529,1024,2048,4096 \
  --chunks 128,256,512,768,1024 \
  --rounds 3 --warmup-pairs 1 \
  --output bench/results/2026-10-04-streaming-5x5
```

Use a new output directory to repeat the experiment. `--resume` with this exact
configuration verifies and reuses completed records instead of repeating the measurement.
