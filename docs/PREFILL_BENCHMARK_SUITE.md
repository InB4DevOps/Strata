# Expert streaming benchmark suite

`tools/bench_prefill_suite.py` compares the old and new expert-streaming schedules over
a matrix of prompt lengths, chunk sizes, workloads, memory settings, KV formats, expert
cache sizes and context capacities. It needs Python 3 and the model/GPU used by your
normal Strata configuration; no Python packages are required.

- **A:** `STRATA_PREFILL_STREAM_AHEAD=0` (old scheduling).
- **B:** `STRATA_PREFILL_STREAM_AHEAD=1` (new scheduling).

The same engine binary runs both arms, one process at a time. See the
[implementation measurements](benchmarks/2026-10-04-prefill-streaming.md) for what the
change does. It primarily targets routed-only chunks, normally below 1,024 tokens,
including small final chunks. Large chunks are useful controls in the matrix.

## Start here

From the repository root, inspect the default workload without running the GPU:

```sh
bash bench-expert-streaming-suite.sh --dry-run
```

The shell wrapper selects your local `strata-coder-iq1_m.json` configuration and the
rebuilt `build/strata` binary. Extra arguments override these defaults. The existing
`bench-expert-streaming.sh` remains the simpler, single-workload A/B script.

For an initial smoke run:

```sh
bash bench-expert-streaming-suite.sh \
  --preset smoke --rounds 1 --warmup-pairs 0 \
  --output bench/results/streaming-smoke
```

For the standard matrix with three measured pairs and one discarded warmup pair per case:

```sh
bash bench-expert-streaming-suite.sh \
  --preset standard --output bench/results/streaming-standard
```

For a different model, or on Windows, invoke Python directly:

```sh
python tools/bench_prefill_suite.py \
  --config strata-MODEL.json --exe build/strata \
  --preset smoke --rounds 1 --warmup-pairs 0
```

Use `python3` where `python` is not installed, and your `.exe` path on Windows. Run with
the GPU otherwise idle. The suite does not start an HTTP server or modify your model config.

## Presets and cost

The planner prints the exact number of cases and engine invocations before execution.
`--dry-run` creates no output files and performs no GPU work. `--list-cases` prints every
case rather than the first twelve. Each engine invocation reloads the model, so large
matrices can take considerably longer than their summed prefill times.

| Preset | Fresh prefill lengths | Requested chunks | Default workloads | Cases | Invocations with defaults |
| --- | --- | --- | --- | ---: | ---: |
| `smoke` | 65, 1041 | 256, 1024, auto | ramp | 6 | 48 |
| `standard` | 65, 256, 529, 1023, 1024, 1041, 2048, 4096, 8192 | 128, 256, 512, 768, 1024, 2048, auto | ramp | 63 | 504 |
| `full` | 1, 17, 65, 127, 128, 129, 255, 256, 257, 511, 512, 513, 767, 768, 769, 1023, 1024, 1025, 1041, 2048, 4096, 8192, 16384, 32768 | 64, 128, 256, 512, 768, 1024, 2048, 4096, 8192, auto | ramp, random, repeat | 720 | 5760 |

These counts use one memory profile, KV mode, cache setting and context capacity, with
no extra residual check. Adding another dimension multiplies the work. The formula is:

```text
engine invocations = cases × 2 × (measured pairs + warmup pairs + residual-check pairs)
```

There is one residual-check pair per case when `--check-residuals` is enabled. Skipped
combinations, such as a prompt that cannot fit the selected context, do not run.

The boundary lengths exercise exact chunks, one-token tails, short tails, and transitions
between large-chunk streaming and routed-only streaming. Requested chunk sizes larger
than the prompt are intentionally retained: their allocation/selection behavior is part
of the experiment. The report also records the **actual chunk count** emitted by the engine.

## Build your own matrix

Each comma-separated list is a dimension of a Cartesian product. Explicit lists replace
the corresponding preset defaults.

### Sweep length and chunk size

```sh
bash bench-expert-streaming-suite.sh \
  --lengths 65,256,512,1023,1024,1025,1041,2048,4096 \
  --chunks 128,256,512,768,1024,2048,auto \
  --rounds 5 --warmup-pairs 1 \
  --output bench/results/streaming-lengths
```

### Compare expert memory paths

```sh
bash bench-expert-streaming-suite.sh \
  --lengths 65,529,1041 --chunks 256,1024 \
  --memory config,arena,mmap,mmap-ring2 \
  --check-residuals --rounds 3 \
  --output bench/results/streaming-memory
```

- `config` preserves the configuration's expert-memory options.
- `arena` removes mmap/resident-limited/shared-arena overrides, selecting the engine's
  default resident arena. RAM availability still determines how much can be pinned.
- `mmap` requests `--mmap-experts`, removing conflicting resident/shared-arena options.
- `mmap-ring2` also sets `STRATA_STAGER_RING=2` to stress host-buffer reuse.

The engine's streamed, direct-DMA and resident expert counts are saved for every run.
These counters, rather than the profile's name, describe the actual transfer path.

### Compare KV, context capacity and expert residency

```sh
bash bench-expert-streaming-suite.sh \
  --lengths 512,1041,4096 --chunks 256,1024,auto \
  --kv-modes int8,fp16 --contexts 8192,32768 \
  --expert-caches auto,1024 \
  --output bench/results/streaming-residency
```

`--expert-caches` values are **slot counts**, not MiB. `config` keeps the original setting.
KV choices are `config,int8,fp16,q4_0,k8v4`; unsupported model/backend combinations are
recorded as failed cases rather than silently substituted.

By default, `--contexts auto` sets **one fixed capacity for the entire suite**:
`max(2048, largest requested prefill length + 64)`. It does not resize capacity for each
prompt. This avoids accidentally giving short prompts more expert residency than long
prompts in the same sweep. Explicit capacities are a separate matrix dimension.

KV streaming is disabled by default with `--kv-resident 0`; set `--kv-resident N` to
exercise it. A combination with insufficient context for the prompt and output is listed
as skipped, with a reason in `summary.json`.

## Prompt workloads

Synthetic workloads are deterministic and are not natural-language or quality benchmarks:

- `ramp`: cycle through the selected token-ID range, matching the original A/B script.
- `random`: seeded uniform draws from that range, useful for different routing patterns.
- `repeat`: one repeated token, a low-diversity workload.

The default range is 100–299. Use `--token-min` and `--token-max` to change it, choosing
IDs valid for your model. `--seed` controls synthetic generation and case ordering. Shorter
prompts are prefixes of longer ones in the same workload.

```sh
bash bench-expert-streaming-suite.sh \
  --lengths 256,1024,4096 --chunks 256,512,1024,auto \
  --workloads ramp,random,repeat --seed 1729 \
  --output bench/results/streaming-patterns
```

For real workloads, provide comma- or whitespace-separated token IDs from your model's
tokenizer. Multiple files produce independent workloads:

```sh
bash bench-expert-streaming-suite.sh \
  --tokens-file coding.tokens --tokens-file conversation.tokens \
  --lengths 512,2048,8192 --chunks 256,512,1024,auto \
  --output bench/results/streaming-real-prompts
```

With token files, synthetic workloads default to none. Add `--workloads ramp,random` if
you want both. Files are used as prefixes and are **never repeated to fill a longer case**;
too-short combinations are skipped. The suite does not download a tokenizer or model.

Lengths mean **fresh tokens processed by batched prefill**. Each input contains one more
token because the engine processes the final input token in its decoding path. A length
of 1,024 therefore needs at least 1,025 IDs. Output generation is fixed at one token.

## Measurement and correctness

For each case:

1. Run the requested discarded warmup pairs.
2. Optionally run a separate residual-check pair.
3. Run measured pairs, alternating A/B order. The initial order varies across cases.

Cases are shuffled with a fixed seed by default; `--ordered` keeps matrix order. The suite
runs processes sequentially, so A and B never compete for the GPU.

Every sample starts a fresh engine. Warmups can warm OS/file caches, but **do not create a
warm persistent GPU session**. They are not a controlled cold-cache experiment either:
the suite does not flush OS caches. The reported primary metric is the engine's batched
prefill time, excluding model loading, subsequent expert-cache refill and the final input
token's decoding work. Process wall time, the broader engine prefill time and engine TTFT
are saved separately where available.

The suite fixes one output token, disables sampling and PCIe-expert decode offload, and
owns the prompt/chunk/context flags. It removes inherited profiling/debug timing and
residual-dump variables that would contaminate the measurement. Other Strata overrides
are recorded; the memory and KV settings above are applied consistently to both arms.

All runs collect the engine's GDN state hashes after its batched timing boundary. Each
pair must match output tokens, GDN hashes, actual chunk count, GPU identity, cache/borrowed
slot counts, and streamed/DMA/resident expert counts. A mismatch invalidates the case's
performance summary. This catches correctness differences and placement drift that would
otherwise confound an A/B comparison.

`--check-residuals` adds a pair that compares SHA-256 hashes of the engine's sampled FP32
residual dumps. Those synchronized runs are **excluded** from performance statistics.
The dump samples every 64th input position; it is not a complete tensor comparison.
Correctness is compared within a case/pair, not across different chunk sizes or KV formats,
whose arithmetic may legitimately differ.

## Read the output

Each case prints a result such as:

```text
A 4715.9 ms -> B 4612.9 ms; 217.1 -> 222.0 tok/s; gain +2.23%
```

The figures here illustrate the format using the earlier 1,024-token, 256-chunk RTX 3060
measurement; your matrix's results are computed when you run it.

The results directory contains:

```text
manifest.json                 # config, binary/runner hashes, hardware, environment, assets, revision
summary.md                    # readable table for all cases, including failures/pending cases
summary.csv                   # spreadsheet-friendly metrics
summary.json                  # structured metrics and skipped combinations
prompts/<sha256>.tokens       # exact inputs, including the final input token
cases/<case-id>/
  warmup-0-A.log / .json       # raw engine output and parsed record
  check-0-A.log / .json        # optional separate correctness runs
  check-0-A.residuals          # optional sampled residual bytes
  measure-0-A.log / .json      # analogous B records and additional pairs
  case.json                   # checked measured pairs and aggregate statistics
```

The main gain is `100 × (median_A_ms / median_B_ms − 1)`, a throughput increase.
Time saved is `100 × (1 − median_B_ms / median_A_ms)`. They are different percentages.
Positive means improvement; negative means regression.

JSON/CSV also contain min/max, interpolated p95, actual chunk counts, and the median of
the **paired** throughput gains. With at least three pairs, a seeded 2,000-resample
bootstrap gives an exploratory 95% interval for that paired median. A few samples do not
establish a precise p95 or a hardware-wide speedup; inspect the run spread and repeat.

Compare A/B **within each row**. The suite does not average unlike prompt lengths, KV
formats or cache placements into a single advertised gain. Model loading is excluded
from the speedup calculation even though it contributes to how long the suite takes.

## Failure handling and resume

Engine exits, missing/incomplete timing records, incorrect token counts, timeouts and
correctness/placement mismatches are recorded as failures. Failed cases have no reported
speedup. Other cases continue by default; `--fail-fast` stops after the first failure.
`--timeout` is a per-process limit in seconds, default 600.

Reports are updated after every case and on Ctrl+C. To resume, repeat the same command
with the same output directory and add `--resume`:

```sh
bash bench-expert-streaming-suite.sh \
  --preset standard --output bench/results/streaming-standard --resume
```

Successful pairs are reused. Both arms of a half-finished pair are rerun together rather
than comparing an old measurement with a much later new one. Failed runs are retried.
Completed warmup pairs are also reused; resuming does not promise the same OS-cache or
thermal state as the original session.

Resume rejects a different binary, runner, configuration, matrix, statistical settings,
recorded hardware or relevant environment. Saved prompt/log/residual hashes are checked.
Model asset sizes and modification times are recorded for explicit model paths and files
under their directories; huge model weights are not read merely to hash them. This is
not a cryptographic identity check of all implicitly discovered model shards. Keep the
model and machine settings fixed throughout an experiment.

Use a new output directory to compare a different build or matrix. Existing output
directories are not overwritten by a fresh run.

## Tests

```sh
python3 tools/test_bench_prefill_suite.py
```

These CPU-only tests cover matrix generation and boundaries, real-prompt truncation,
command overrides, result parsing, statistics, warmup/check exclusion, subprocess failure
and timeout handling, report generation, resume and artifact corruption. A fake engine
allows the runner itself to be tested without a GPU or model download.
