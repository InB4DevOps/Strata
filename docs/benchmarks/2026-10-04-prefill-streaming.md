# Routed expert streaming: RTX 3060

These measurements precede the integration onto upstream `6f32ec0`. The original
[25-case evidence packet](../../bench/results/2026-10-04-streaming-5x5/README.md) names its
base and binary hash. [Post-update smoke checks](../../bench/results/2026-10-04-streaming-main-smoke/README.md)
validate the integration separately; the historical speedups are not new-base measurements.

Linux, NVIDIA GeForce RTX 3060 12 GB, Intel Core i7-12700KF, 62 GiB system RAM, CUDA build for sm_86,
Coder IQ1_M native pack (`Qwen3.8-Flash-Next-GSQ-RCO-IQ1_M`), 8-bit KV,
`--expert-cache auto`, shipped Coder expert profile, pinned resident expert arena,
`--spec 4`, `--max-new 1`, `--pcie-frac 0`. Synthetic token IDs
`100 + i % 200`, with one additional final prompt token processed by decoding.
These are scheduling benchmarks, not natural-prompt throughput or quality measurements.

## Change

The routed-only path now:

1. Reads back the router before running the shared expert, then runs the shared expert
   while the host groups routed rows and queues expert uploads. Previously the routing
   synchronization also waited for the shared expert to finish.
2. Keeps up to eight **streamed experts** ahead rather than eight positions in the mixed
   resident/streamed order. Resident experts no longer consume lookahead capacity.

The ring remains eight slots. No additional GPU buffer is allocated. Expert ordering,
MMQ grouping and arithmetic are unchanged. Ring slots are only reissued after the
compute stream has recorded their release events. The full-stream path used for chunks
of at least 1,024 tokens keeps its existing schedule.

`STRATA_PREFILL_STREAM_AHEAD=0` selects the old schedule for A/B measurements.
The default is enabled. Both changes apply to the routed-only path, including small
final chunks. The same CUDA API spellings map through Strata's HIP compatibility layer;
these measurements and checks were on NVIDIA only.

## Profiling-disabled measurements

Two pairs per workload, in the order old/new, new/old. Each process loads the same model.
The times below are the engine's batched-prefill time, excluding model loading and cache
refill. GDN state hashes are collected after that timing boundary. PLE startup and other
run-to-run variation remain in the measurement.

| Fresh prompt tokens | Chunk | Old runs (ms) | New runs (ms) | Median time reduction |
| ---: | ---: | --- | --- | ---: |
| 1,024 | 256 | 4716.7, 4715.1 | 4603.2, 4622.6 | 2.18% |
| 2,048 | 512 | 5997.3, 6018.4 | 5895.6, 5925.9 | 1.62% |

Generated output tokens and all reported GDN state hashes matched for each workload.
The measured improvement is modest; it does not remove the underlying PCIe bandwidth cost.

## Additional correctness checks

- **Unpinned source and ring wraparound:** `--mmap-experts`, `STRATA_STAGER_RING=2`,
  529 fresh tokens in chunks of 256, 256 and 17. Old/new generated tokens, GDN hashes,
  and SHA-256 of the sampled FP32 residual dump matched.
- **Normal auto path:** 1,024 fresh tokens with `--prefill auto` used one full-stream
  chunk. Old/new outputs, GDN hashes and residual dumps matched. A single pair measured
  1661.5 / 1669.0 ms; this change does not target that path.
- **Transition between paths:** 1,041 fresh tokens, chunk size 1,024, exercised a
  full-stream chunk followed by a 17-token routed-only tail. Outputs, GDN hashes and
  sampled residual bytes matched.
- **FP16 fallback:** `STRATA_PREFILL_MMQ=0`, 65 fresh tokens. Outputs, GDN hashes and
  sampled residual bytes matched. The unpinned two-slot stager was also checked separately
  with a 65-token MMQ prompt.

The residual dump samples every 64th position after all layers; it is not a full tensor
comparison. It adds synchronization and is used for correctness checks, not speed claims.

Grouped gathers with the original ring, a larger ring, and direct-to-MMQ staging were
also tried. They showed no reliable improvement on this setup and were discarded.

## Reproduce

For sweeps over prompt lengths, chunk sizes, workloads, KV/context capacities and expert
residency, see the [resumable benchmark suite](../PREFILL_BENCHMARK_SUITE.md).

For the local Coder configuration and `build/strata`, run from the repository root:

```sh
bash bench-expert-streaming.sh
```

This runs three alternating A/B pairs with profiling disabled and prints median prefill
times, tokens/sec, run ranges and the measured gain (negative for a regression).
Use `bash bench-expert-streaming.sh --tokens 2048 --chunk 512` for the second workload.

Use a configuration pointing to the same model and a freshly built engine:

```sh
python3 tools/test_prefill_streaming.py --config strata-coder-iq1_m.json \
  --exe build/strata --chunk 256 --tokens 1024 --rounds 2 --benchmark
python3 tools/test_prefill_streaming.py --config strata-coder-iq1_m.json \
  --exe build/strata --chunk 512 --tokens 2048 --rounds 2 --benchmark
python3 tools/test_prefill_streaming.py --config strata-coder-iq1_m.json \
  --exe build/strata --chunk 256 --tokens 529 --rounds 1 --mmap-experts --stager-ring 2
python3 tools/test_prefill_streaming.py --config strata-coder-iq1_m.json \
  --exe build/strata --chunk auto --tokens 1024 --rounds 1
```

The tool writes logs and `results.json` to a new temporary directory, printed at startup.
It also accepts `--tokens-file` to check real pretokenized prompts. Save that report with
the engine revision, model identity, build settings and any environment overrides when
comparing other hardware or workloads.
