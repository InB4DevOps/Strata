# Streaming overlap: smoke checks after updating main

The change was reapplied to upstream `main` at
`6f32ec070f23ced9f50e704d854d775da52591ab`. Integration kept upstream's asynchronous
layer-split hand-off lifecycle, temporary helper-stage ownership and non-P2P return-copy
path. This machine has one GPU; those multi-GPU paths were not hardware-tested.

CUDA engine SHA-256:
`e4669099caf58fcbb0b79b9e607ab2d19387b7171c023fb58489474ac7e8ba6e`.

RTX 3060 12 GB, Intel i7-12700KF, Linux, Coder IQ1_M, INT8 KV, fixed 2,048-token
capacity, resident arena, automatic expert cache. Synthetic ramp IDs 100–299. Each
prompt contains one extra final input token, handled by decoding.

**Four cases, 16 engine runs, all passed:**

- Fresh prefill lengths 529 and 1,041; chunk sizes 256 and 1,024.
- One separate residual-check A/B pair and one timing A/B pair for each case.
- Matching generated token, GDN hashes, GPU identity, actual chunk count, expert residency
  and streaming counts in every pair.
- Matching SHA-256 of sampled FP32 residual bytes in every correctness pair.
- The 1,041/1,024 case exercises a full-stream chunk followed by a 17-token routed-only tail.

One timing pair per case is a smoke check, not a replacement for the historical 25-case
benchmark. [summary.md](summary.md), [summary.csv](summary.csv), [summary.json](summary.json)
and the raw logs record these runs separately. The `.residuals` files remain local;
their hashes are recorded in the check-run JSON files.

Additional checks on this base:

- `python3 tools/test_prefill_profile.py`: 5 tests passed.
- `python3 tools/test_bench_prefill_suite.py`: 13 tests passed.
- CUDA build (`sm_86`) completed.
- `ctest --test-dir build -R '^prefill_profile_test$' --output-on-failure`: passed;
  verifies 12,000 intervals across event reuse and repeated folds.

Command:

```sh
bash bench-expert-streaming-suite.sh \
  --lengths 529,1041 --chunks 256,1024 \
  --rounds 1 --warmup-pairs 0 --check-residuals \
  --output bench/results/2026-10-04-streaming-main-smoke
```
