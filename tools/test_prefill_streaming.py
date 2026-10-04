#!/usr/bin/env python3
"""Model/GPU integration check and A/B benchmark for routed expert streaming.

Uses a normal Strata engine config. Default: compare sampled residual bytes, GDN
state hashes and output tokens. --benchmark disables residual dumps for timing.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import statistics
import subprocess
import tempfile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--exe", type=Path, help="override the config's engine binary")
    parser.add_argument("--chunk", default="256", help="engine --prefill setting, including auto")
    parser.add_argument("--tokens", type=int, default=529, help="synthetic prefill tokens, plus one final prompt token")
    parser.add_argument("--tokens-file", type=Path, help="use a real pretokenized prompt instead")
    parser.add_argument("--rounds", type=int, default=2, help="alternating A/B pairs")
    parser.add_argument("--benchmark", action="store_true", help="disable residual dumps for throughput measurements")
    parser.add_argument("--mmap-experts", action="store_true", help="exercise the unpinned expert path")
    parser.add_argument("--stager-ring", type=int, help="host staging ring depth; 2 stresses buffer reuse")
    parser.add_argument("--timeout", type=int, default=180, help="seconds per engine process")
    args = parser.parse_args()
    if args.rounds < 1 or args.tokens < 1 or args.timeout < 1:
        parser.error("rounds, tokens and timeout must be positive")
    config = json.loads(args.config.read_text(encoding="utf-8"))
    cwd = Path(config.get("cwd", ".")).resolve()
    exe = args.exe.resolve() if args.exe else Path(config["exe"])
    command = [str(exe), *config["args"]]
    if "--serve" in command:
        parser.error("use a generation config without --serve")
    if args.tokens_file:
        tokens = [int(t) for t in re.split(r"[\s,]+", args.tokens_file.read_text().strip()) if t]
        if len(tokens) < 2:
            parser.error("the prompt needs at least two tokens")
    else:
        tokens = [100 + i % 200 for i in range(args.tokens + 1)]
    output = Path(tempfile.mkdtemp(prefix="strata-prefill-streaming-"))
    token_file = output / "prompt.tokens"
    token_file.write_text(",".join(map(str, tokens)), encoding="utf-8")
    command += ["--prefill", args.chunk, "--max-context", str(max(2048, len(tokens) - 1 + 64)),
                "--kv-resident", "0", "--pcie-frac", "0", "--max-new", "1", "--tokens-file", str(token_file)]
    if args.mmap_experts:
        command += ["--mmap-experts"]
    print(f"Logs/results: {output}", flush=True)
    print(f"A = old streaming; B = new streaming. {args.rounds} alternating A/B pairs.\n"
          f"Prompt: {len(tokens) - 1} prefill tokens; chunk setting: {args.chunk}.\n"
          "Reported times exclude model loading. Each run starts a fresh engine.", flush=True)
    results = []
    for pair in range(args.rounds):
        for enabled in ([0, 1] if pair % 2 == 0 else [1, 0]):
            label = "B (new)" if enabled else "A (old)"
            print(f"\nPair {pair + 1}/{args.rounds}: running {label}...", flush=True)
            stem = output / f"pair-{pair}-ahead-{enabled}"
            env = os.environ.copy()
            for key in ("STRATA_PREFILL_PROFILE", "STRATA_PREFILL_TIMING", "STRATA_PREFILL_DUMP_R"):
                env.pop(key, None)
            env["STRATA_PREFILL_STREAM_AHEAD"] = str(enabled)
            env["STRATA_STATE_HASH_GDN"] = "1"
            if not args.benchmark:
                env["STRATA_PREFILL_DUMP_R"] = str(stem) + ".residuals"
            if args.stager_ring is not None:
                env["STRATA_STAGER_RING"] = str(args.stager_ring)
            with open(str(stem) + ".log", "w", encoding="utf-8") as log:
                subprocess.run(command, cwd=cwd, env=env, stdout=log, stderr=subprocess.STDOUT,
                               check=True, timeout=args.timeout)
            text = Path(str(stem) + ".log").read_text(encoding="utf-8", errors="replace")
            timing = re.search(r"prefill (\d+) tokens in (\d+) chunks, ([\d.]+) ms", text)
            state = re.search(r"GDN_HASH ([^\n]+)", text)
            generated = re.search(r"output  : ([^\n]+)", text)
            if not all((timing, state, generated)):
                raise RuntimeError(f"No complete generation result in {stem}.log")
            record = {"enabled": enabled, "pair": pair, "tokens": int(timing[1]),
                      "chunks": int(timing[2]), "ms": float(timing[3]),
                      "gdn_hashes": state[1].strip(), "output": generated[1].strip()}
            if not args.benchmark:
                residuals = Path(str(stem) + ".residuals").read_bytes()
                if not residuals:
                    raise RuntimeError("empty residual dump")
                record["residual_sha256"] = hashlib.sha256(residuals).hexdigest()
            results.append(record)
            print(f"  {record['ms']:.1f} ms, {1000 * record['tokens'] / record['ms']:.1f} tokens/sec, "
                  f"{record['chunks']} chunks; output token(s): {record['output']}", flush=True)
    for field in ("tokens", "chunks", "gdn_hashes", "output") + (() if args.benchmark else ("residual_sha256",)):
        if len({r[field] for r in results}) != 1:
            raise RuntimeError(f"A/B mismatch: {field}; inspect {output}")
    old = statistics.median(r["ms"] for r in results if not r["enabled"])
    new = statistics.median(r["ms"] for r in results if r["enabled"])
    report = {"command": command, "config": str(args.config.resolve()), "benchmark": args.benchmark,
              "results": results, "legacy_median_ms": old, "ahead_median_ms": new,
              "time_reduction_percent": 100 * (1 - new / old),
              "throughput_gain_percent": 100 * (old / new - 1)}
    (output / "results.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    count = results[0]["tokens"]
    print("\n=== Expert streaming A/B results ===")
    print(f"{'Version':<16} {'Median prefill':>16} {'Tokens/sec':>12} {'Run range (ms)':>22}")
    for enabled, label, median in ((0, "A: old", old), (1, "B: new", new)):
        times = [r["ms"] for r in results if r["enabled"] == enabled]
        print(f"{label:<16} {median:13.1f} ms {1000 * count / median:12.1f} "
              f"{min(times):9.1f} - {max(times):9.1f}")
    print(f"\nThroughput gain: {report['throughput_gain_percent']:+.2f}%")
    print(f"Prefill time saved: {old - new:+.1f} ms ({report['time_reduction_percent']:+.2f}%)")
    print("Positive = improvement; negative = regression. Compare the run ranges for variability.")
    print("PASS: output/state checks match.")
    print(f"Full results: {output / 'results.json'}")
    if not args.benchmark:
        print("Residual dumps synchronize each chunk; use --benchmark for performance comparisons.")


if __name__ == "__main__":
    main()
