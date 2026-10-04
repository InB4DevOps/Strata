# Profiling prompt processing

The batched prompt path can report where its time goes, by chunk, layer and operation.
This measures **prefill**, before output-token decoding. It requires a rebuilt engine with these changes.

## Capture and rank a profile

Set `STRATA_PREFILL_PROFILE` to a JSONL output file before starting the engine or Python server.
The variable is inherited by the engine. For example, from the Strata directory on Linux:

```sh
STRATA_PREFILL_PROFILE=/tmp/prefill.jsonl python serve/server.py --engine strata --config strata-MODEL.json
```

Use your own configuration (with `exe` pointing to the rebuilt `build/strata`) and send a request with a fresh prompt. An already-running
server must be restarted to receive the environment variable. On PowerShell:

```powershell
$env:STRATA_PREFILL_PROFILE = "$PWD\prefill.jsonl"
# Start your usual Strata launcher here.
```

Then, using Python 3:

```sh
python tools/prefill_profile.py /tmp/prefill.jsonl
python tools/prefill_profile.py /tmp/prefill.jsonl --group layer --top 30
python tools/prefill_profile.py /tmp/prefill.jsonl --group chunk
python tools/prefill_profile.py /tmp/prefill.jsonl --group implementation
python tools/prefill_profile.py /tmp/candidate.jsonl --baseline /tmp/baseline.jsonl
python tools/prefill_profile.py /tmp/prefill.jsonl --json
```

Use `python3` if your system has no `python` command. No Python packages are required.

The file is appended to. Use a new path for each experiment; one file per engine process.
Concurrent layer-split stages within one process can share a file. Output contains timings and
dimensions, not prompt text. `STRATA_PREFILL_TIMING=1` still enables just the stderr summary;
the JSONL option enables that summary too. Unset both variables for normal throughput measurements.

## What the numbers mean

- **GPU rows** are intervals between events on a compute stream. They include queued kernels,
  stream waits, and gaps while the host is preparing or launching work. They are not individual
  kernel execution times. Percentages use that stream's measured total.
- **Host rows** use a monotonic CPU clock. `routing download+sync` includes waiting for earlier
  GPU work; it is not a measure of the cost of the routing algorithm. `expert grouping+upload enqueue`
  measures CPU grouping and enqueue work after that wait. Host measurements overlap GPU intervals.
- **Chunk callbacks** include draft-KV preparation and progress/checkpoint callbacks where present.
  The GPU `chunk finish+callbacks` interval also includes the gap between chunks. Earlier layer-split
  stages' hand-off/callback costs appear in this GPU interval; the host callback rows cover the final stage.
- **Wait copy** is the exposed delay on the compute stream, including waiting for the host issuer.
  It is not total DMA duration. `primary_expert_copy_bytes` counts enqueued expert transfers to the
  primary device's staging ring, including re-copies across chunks. It excludes peer transfers,
  cache refill, PLE and KV uploads. Do not divide those bytes by exposed wait time to infer bandwidth.
- **Dequant/weight gather** is weight preparation: dequantization for the FP16 path or quantized
  weight gathering/conversion for MMQ. `gemm down` also includes activation preparation where
  it occurs before the down product. Fused expert processing has its own label, rather than being
  attributed entirely to gate/up GEMM.
- **Implementation** identifies MMQ versus fused versus dequantize/FP16 expert paths, expert
  GGUF type IDs, batched versus per-token PLE, and tensor prompt attention versus its decode-kernel
  fallback. `default` means the phase is identified but its internal kernel dispatch is not recorded.
- **Layer -1** means chunk-level work. `chunk` is the absolute starting token position, not an ordinal.
  `tokens` is the actual size of that chunk, including a short final chunk.
- **Intervals** counts timed spans, not kernels, experts, or tokens. A phase may contain several
  kernels. `max_ms` is the longest single span, not the largest sum for a layer or chunk.
- **Wall time** covers one `Prefill::run_impl` stage invocation, including profiling overhead. It excludes model loading,
  initialization, subsequent expert-cache refill and the final prompt token handled by decoding.
  Layer-split stages overlap: the sum of their wall times is not request latency. A stage can
  finish before a downstream stage fails, so stage completion alone does not establish request success.

No new synchronization is introduced by the collector. It collects at existing synchronization
points and queries completed events during long runs. Each timer reuses at most 8,192 events;
if the GPU cannot keep up with that bound, or a timing API fails, it reports an incomplete profile
rather than silently misattributing time. Collection/output can still add host work and launch gaps.
Compare enabled versus disabled runs to measure the effect on your workload.

## Find an optimization candidate

Start with phase totals, then inspect the expensive phases by layer, chunk and implementation.

| Large measured cost | Code to inspect and next measurement |
| --- | --- |
| `wait copy` | Expert streaming/issuer in `src/prefill/prefill.cpp`; compare chunk size, ring size and expert residency. Inspect transfer/compute overlap in a GPU timeline. |
| `dequant/weight gather`, expert GEMMs | Dispatch in `prefill.cpp`, then `moe_mmq.cu`, `moe_fused*.cu`, or `gemm.cu`; compare the selected quantization formats and routed batch sizes. |
| `gdn`, `gdn out proj`, `qsa proj`, `shared expert` | Projection calls in `prefill.cpp` and native GEMM/dequantization in `gemm.cu`. |
| `gdn recurrence`, `gdn conv+gates` | `src/prefill/kernels.cu`; inspect the selected kernel with a GPU profiler. |
| `qsa indexer`, `qsa select`, `qsa attn` | The corresponding QSA kernels in `src/kernels/`; compare early and late chunks. |
| `ple wait+upload` | PLE gather/read-ahead in `prefill.cpp` and the reader in `src/ngram/`; compare first and later chunks. |
| CPU expert grouping | Grouping in `prefill.cpp`; distinguish CPU work from the preceding GPU wait. |

Use Nsight Systems/Compute on NVIDIA or the ROCm profiling tools on AMD for the next, kernel-level
investigation. This profiler is intended to select what to investigate, not predict a speedup.

The routed-only streaming path overlaps shared-expert computation with grouping/uploads and
counts only actual transfers against its eight-slot lookahead. `STRATA_PREFILL_STREAM_AHEAD=0`
restores the previous schedule for comparison. See the [RTX 3060 measurements and correctness
checks](benchmarks/2026-10-04-prefill-streaming.md) and `tools/test_prefill_streaming.py`.
For an extensive, resumable A/B matrix across lengths, chunks and memory settings, use
the [prefill benchmark suite](PREFILL_BENCHMARK_SUITE.md).

For comparisons, keep the model, prompt tokens, chunk size, KV settings, expert residency, device,
build flags and environment fixed except for the change being tested. Save the command/config,
revision, model identity and engine log beside the profile. The tool checks GPU, stage range,
token positions/counts, chunk shapes and run count before comparing; it cannot verify prompt contents
or model identity. Compare fresh processing rather than a prompt-cache hit. Separate cold starts
from warm runs, repeat measurements, and confirm any improvement with profiling disabled.

## File format and checks

Schema 1 has `run`, `configuration`, `gpu`, `host`, `counters`, and `end` records, joined by a run ID.
GPU rows may be partial aggregates emitted at several collection points; sum matching keys.
Records from separate stages may interleave. A successful run ends with `complete: true` and
`valid: true`. Cancelled/failed runs have `complete: false`; a killed process may leave no end record.
The analysis tool excludes all of these incomplete/invalid runs and reports how many it excluded.

```sh
python tools/test_prefill_profile.py
# With STRATA_BUILD_TESTS=ON in the CUDA or HIP build:
cmake --build build --target prefill_profile_test
ctest --test-dir build -R '^prefill_profile_test$' --output-on-failure
```

The runtime test exercises 12,000 intervals across event reuse and repeated partial folds. It needs
a GPU but no model. The Python tests cover aggregation, overlapping streams, incomplete runs,
malformed records and workload comparison.
