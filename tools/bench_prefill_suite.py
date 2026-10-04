#!/usr/bin/env python3
"""Resumable, sequential A/B benchmark matrix for Strata expert streaming (Python 3, no packages)."""
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import platform
import random
import re
import statistics
import subprocess
import sys
import time

VERSION = 1
ROOT = Path(__file__).resolve().parents[1]
PRESETS = {
    "smoke": ([65, 1041], ["256", "1024", "auto"]),
    "standard": ([65, 256, 529, 1023, 1024, 1041, 2048, 4096, 8192],
                 ["128", "256", "512", "768", "1024", "2048", "auto"]),
    "full": ([1, 17, 65, 127, 128, 129, 255, 256, 257, 511, 512, 513,
              767, 768, 769, 1023, 1024, 1025, 1041, 2048, 4096, 8192, 16384, 32768],
             ["64", "128", "256", "512", "768", "1024", "2048", "4096", "8192", "auto"]),
}
# The suite owns the prompt, timing boundaries and deterministic generation settings.
MANAGED_VALUES = {"--tokens", "--tokens-file", "--prefill", "--max-context", "--kv-resident",
                  "--max-new", "--pcie-frac", "--dump-logits", "--logits-stride", "--prefill-until",
                  "--seed", "--temperature", "--top-k", "--top-p"}
MEMORY_FLAGS = {"--mmap-experts", "--resident-experts", "--resident-cpu-experts"}
MEMORY_VALUES = {"--shared-expert-arena", "--resident-budget-gib"}
DISABLED_ENV = {"STRATA_PREFILL_PROFILE", "STRATA_PREFILL_TIMING", "STRATA_PREFILL_DUMP_R",
                "STRATA_VERIFY_PROFILE", "STRATA_DRAFT_TIMING", "STRATA_DBG_NAN", "STRATA_TRACE",
                "STRATA_QSA_DUMP", "STRATA_STATE_HASH"}


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def file_hash(path):
    h = hashlib.sha256()
    with open(path, "rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def save_json(path, value):
    """Only a fully written record is eligible for resume."""
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temp.replace(path)


def items(text):
    return list(dict.fromkeys(part.strip() for part in text.split(",") if part.strip()))


def positive_list(text):
    values = [int(v) for v in items(text)]
    if not values or any(v < 1 for v in values):
        raise ValueError("expected a comma-separated list of positive integers")
    return values


def strip_args(arguments, values, flags=()):
    out = []
    i = 0
    while i < len(arguments):
        arg = arguments[i]
        if arg in values:
            if i + 1 == len(arguments) or arguments[i + 1].startswith("--"):
                raise ValueError(f"missing value after {arg} in config")
            i += 2
        elif arg in flags:
            i += 1
        else:
            out.append(arg)
            i += 1
    return out


def build_plan(args):
    default_lengths, default_chunks = PRESETS[args.preset]
    lengths = positive_list(args.lengths) if args.lengths else default_lengths
    chunks = items(args.chunks) if args.chunks else default_chunks
    if not chunks or any(c != "auto" and (not c.isdecimal() or int(c) < 1) for c in chunks):
        raise ValueError("chunks must be positive integers or auto")
    memories, kvs, caches = items(args.memory), items(args.kv_modes), items(args.expert_caches)
    if not memories or not set(memories) <= {"config", "arena", "mmap", "mmap-ring2"}:
        raise ValueError("memory profiles: config,arena,mmap,mmap-ring2")
    if not kvs or not set(kvs) <= {"config", "int8", "fp16", "q4_0", "k8v4"}:
        raise ValueError("KV modes: config,int8,fp16,q4_0,k8v4")
    if not caches or any(c not in ("config", "auto") and not c.isdecimal() for c in caches):
        raise ValueError("expert caches must be config,auto or nonnegative slot counts")
    context_options = items(args.contexts)
    if not context_options or any(c != "auto" and (not c.isdecimal() or int(c) < 1) for c in context_options):
        raise ValueError("contexts must be positive capacities or auto")
    # A single automatic capacity across the matrix avoids silently changing residency with prompt length.
    contexts = list(dict.fromkeys(max(2048, max(lengths) + 64) if c == "auto" else int(c)
                                  for c in context_options))
    workloads = items(args.workloads) if args.workloads is not None else (
        [] if args.tokens_file else ["ramp", "random", "repeat"] if args.preset == "full" else ["ramp"])
    if not set(workloads) <= {"ramp", "random", "repeat"}:
        raise ValueError("synthetic workloads: ramp,random,repeat")
    sources = []
    for kind in workloads:
        rng = random.Random(args.seed)
        n = max(lengths) + 1
        width = args.token_max - args.token_min + 1
        tokens = ([args.token_min + i % width for i in range(n)] if kind == "ramp" else
                  [rng.randint(args.token_min, args.token_max) for _ in range(n)] if kind == "random" else
                  [args.token_min] * n)
        sources.append({"name": kind, "tokens": tokens, "source": "synthetic"})
    for path in args.tokens_file:
        raw = path.read_text(encoding="utf-8").strip()
        tokens = [int(t) for t in re.split(r"[\s,]+", raw) if t]
        if len(tokens) < 2 or any(t < 0 for t in tokens):
            raise ValueError(f"{path}: need at least two nonnegative token IDs")
        sources.append({"name": path.name, "tokens": tokens, "source": str(path.resolve()),
                        "file_sha256": file_hash(path)})
    if not sources:
        raise ValueError("choose a synthetic workload or provide --tokens-file")
    cases, skipped, prompts = [], [], {}
    for source, length, chunk, memory, kv, cache, context in itertools.product(
            sources, lengths, chunks, memories, kvs, caches, contexts):
        case = {"workload": source["name"], "source": source["source"], "tokens": length,
                "chunk": chunk, "memory": memory, "kv": kv, "expert_cache": cache, "context": context}
        if len(source["tokens"]) < length + 1 or context < length + 2:
            skipped.append(dict(case, reason="prompt file too short" if len(source["tokens"]) < length + 1
                                else "context too small for prompt and output"))
            continue
        payload = ",".join(map(str, source["tokens"][:length + 1])) + "\n"
        sha = hashlib.sha256(payload.encode()).hexdigest()
        prompts[sha] = payload
        case["prompt_sha256"] = sha
        case["id"] = digest(case)[:16]
        cases.append(case)
    cases = list({case["id"]: case for case in cases}.values())
    if not args.ordered:
        random.Random(args.seed).shuffle(cases)
    return cases, skipped, prompts


def make_command(config_args, exe, case, prompt, kv_resident):
    values = set(MANAGED_VALUES)
    flags = set()
    if case["memory"] != "config":
        values |= MEMORY_VALUES
        flags |= MEMORY_FLAGS
    if case["kv"] != "config":
        values.add("--kv")
    if case["expert_cache"] != "config":
        values.add("--expert-cache")
    command = [str(exe), *strip_args(config_args, values, flags),
               "--tokens-file", str(prompt), "--prefill", case["chunk"], "--max-context", str(case["context"]),
               "--kv-resident", str(kv_resident), "--max-new", "1", "--pcie-frac", "0"]
    if case["memory"].startswith("mmap"):
        command.append("--mmap-experts")
    if case["kv"] != "config":
        command += ["--kv", case["kv"]]
    if case["expert_cache"] != "config":
        command += ["--expert-cache", case["expert_cache"]]
    return command


def parse_log(text, expected_tokens):
    matches = re.findall(r"strata generate: prefill (\d+) tokens in (\d+) chunks, ([\d.eE+-]+) ms.*?"
                         r"experts streamed (\d+) \((\d+) by DMA, host ([\d.eE+-]+) ms\), "
                         r"resident (\d+); PLE ([\d.eE+-]+) ms", text)
    states = re.findall(r"strata prefill: GDN_HASH ([^\n]+)", text)
    outputs = re.findall(r"^output[ \t]*:[ \t]*(\d+(?:[ \t]+\d+)*)[ \t]*$", text, re.MULTILINE)
    if len(matches) != 1 or len(states) != 1 or len(outputs) != 1:
        raise ValueError("expected one completed prefill, GDN hash and output record; inspect engine log")
    n, chunks, ms, streamed, dma, host_ms, resident, ple_ms = matches[0]
    if int(n) != expected_tokens or int(chunks) < 1 or not math.isfinite(float(ms)) or float(ms) <= 0:
        raise ValueError("incorrect token count or invalid prefill duration/chunk count")
    slots = re.findall(r"expert cache (\d+) slots", text)
    borrowed = re.findall(r"prompt path borrows (\d+) cache slots", text)
    result = {"tokens": int(n), "chunks": int(chunks), "ms": float(ms),
              "streamed": int(streamed), "dma": int(dma), "resident": int(resident),
              "host_staging_ms": float(host_ms), "ple_ms": float(ple_ms),
              "cache_slots": int(slots[-1]) if slots else None,
              "borrowed_slots": int(borrowed[-1]) if borrowed else None,
              "gpu": re.findall(r"strata generate: GPU \d+: ([^\n]+)", text),
              "gdn_hashes": states[0].strip(), "output": outputs[0].strip()}
    final = re.search(r"^prefill\s+\d+ tokens in ([\d.]+) ms.*?time to first token ([\d.]+) ms", text, re.MULTILINE)
    if final:
        result.update(engine_prefill_ms=float(final[1]), engine_ttft_ms=float(final[2]))
    if any(not math.isfinite(result[k]) or result[k] < 0 for k in ("host_staging_ms", "ple_ms")):
        raise ValueError("invalid host timing")
    return result


def validate_pair(a, b):
    for field in ("tokens", "chunks", "gpu", "gdn_hashes", "output", "residual_sha256",
                  "streamed", "dma", "resident", "cache_slots", "borrowed_slots"):
        if a.get(field) != b.get(field):
            raise ValueError(f"A/B mismatch in {field}: pair excluded (placement drift also invalidates comparisons)")


def percentile(values, fraction):
    ordered = sorted(values)
    pos = (len(ordered) - 1) * fraction
    lo, hi = math.floor(pos), math.ceil(pos)
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (pos - lo)


def summarize(pairs, seed):
    old = [p["A"]["ms"] for p in pairs]
    new = [p["B"]["ms"] for p in pairs]
    gains = [100 * (a / b - 1) for a, b in zip(old, new)]
    a, b = statistics.median(old), statistics.median(new)
    tokens = pairs[0]["A"]["tokens"]
    result = {"pairs": len(pairs), "old_ms": a, "new_ms": b, "old_tps": 1000 * tokens / a,
              "new_tps": 1000 * tokens / b, "throughput_gain_pct": 100 * (a / b - 1),
              "time_saved_pct": 100 * (1 - b / a), "paired_gain_median_pct": statistics.median(gains),
              "old_min_ms": min(old), "old_max_ms": max(old), "new_min_ms": min(new), "new_max_ms": max(new),
              "old_p95_ms": percentile(old, .95), "new_p95_ms": percentile(new, .95),
              "actual_chunks": sorted({p["A"]["chunks"] for p in pairs}),
              "paired_gain_ci95_low": None, "paired_gain_ci95_high": None}
    if len(gains) >= 3:
        rng = random.Random(seed)
        boot = [statistics.median(rng.choices(gains, k=len(gains))) for _ in range(2000)]
        result["paired_gain_ci95_low"] = percentile(boot, .025)
        result["paired_gain_ci95_high"] = percentile(boot, .975)
    return result


def probe(command, cwd=None):
    try:
        result = subprocess.run(command, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                text=True, timeout=10, check=False)
        return result.stdout.strip() if result.returncode == 0 else None
    except (OSError, subprocess.TimeoutExpired):
        return None


def asset_metadata(arguments, cwd):
    """Stat model assets without reading huge weights into the OS cache just to hash them."""
    files = {}
    for i, arg in enumerate(arguments[:-1]):
        if arg not in ("--pack", "--native", "--ple-gguf", "--mtp", "--expert-profile", "--embd-gguf",
                       "--native-dense-gguf", "--native-head-gguf"):
            continue
        path = (cwd / arguments[i + 1]).resolve()
        paths = [path, *sorted(path.rglob("*"))] if path.is_dir() else [path]
        for item in paths:
            if item.is_file():
                stat = item.stat()
                files[str(item)] = {"bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns}
    return files


def environment(config):
    env = os.environ.copy()
    for key in DISABLED_ENV:
        env.pop(key, None)
    env["STRATA_STATE_HASH_GDN"] = "1"  # outside the engine's batched-prefill timing boundary
    dirs = [str(p) for p in config.get("lib_dirs", [])]
    if dirs:
        variable = "PATH" if os.name == "nt" else "LD_LIBRARY_PATH"
        env[variable] = os.pathsep.join(dirs + ([env[variable]] if env.get(variable) else []))
    return env


def relevant_env(env):
    return {k: v for k, v in sorted(env.items()) if k.startswith(("STRATA_", "GGML_")) or
            k in ("CUDA_VISIBLE_DEVICES", "CUDA_MODULE_LOADING", "HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES",
                  "OMP_NUM_THREADS", "MKL_NUM_THREADS", "LD_LIBRARY_PATH")}


def run_engine(command, cwd, env, stem, expected_tokens, timeout, residuals, resume):
    record_path = stem.with_suffix(".json")
    log_path = stem.with_suffix(".log")
    if resume and record_path.exists():
        record = json.loads(record_path.read_text(encoding="utf-8"))
        if (record.get("status") == "ok" and log_path.exists() and record.get("command") == command
                and record.get("stream_ahead") == env["STRATA_PREFILL_STREAM_AHEAD"]
                and bool(record.get("residual_sha256")) == residuals):
            if record.get("log_sha256") != file_hash(log_path):
                raise ValueError(f"saved log changed: {log_path}")
            if residuals and (not stem.with_suffix(".residuals").exists() or
                              record["residual_sha256"] != file_hash(stem.with_suffix(".residuals"))):
                raise ValueError(f"saved residual dump changed: {stem}")
            return record
    run_env = dict(env)
    residual_path = stem.with_suffix(".residuals")
    if residuals:
        run_env["STRATA_PREFILL_DUMP_R"] = str(residual_path)
    start = time.monotonic()
    record = {"command": command, "stream_ahead": run_env["STRATA_PREFILL_STREAM_AHEAD"],
              "started_utc": datetime.now(timezone.utc).isoformat(),
              "environment": relevant_env(run_env), "log": str(log_path), "status": "failed"}
    try:
        with log_path.open("w", encoding="utf-8") as log:
            process = subprocess.run(command, cwd=cwd, env=run_env, stdout=log, stderr=subprocess.STDOUT,
                                     timeout=timeout, check=False)
        record["returncode"] = process.returncode
        if process.returncode:
            raise ValueError(f"engine exit {process.returncode}")
        record.update(parse_log(log_path.read_text(encoding="utf-8", errors="replace"), expected_tokens))
        if residuals:
            if not residual_path.exists() or not residual_path.stat().st_size:
                raise ValueError("missing/empty residual dump")
            record["residual_sha256"] = file_hash(residual_path)
        record["status"] = "ok"
    except (OSError, ValueError, subprocess.TimeoutExpired) as exc:
        record["error"] = str(exc)
    except KeyboardInterrupt:
        record["error"] = "interrupted"
        raise
    finally:
        record["process_wall_ms"] = 1000 * (time.monotonic() - start)
        if log_path.exists():
            record["log_sha256"] = file_hash(log_path)
        save_json(record_path, record)
    return record


def run_case(case, index, args, config_args, exe, cwd, env, output):
    directory = output / "cases" / case["id"]
    directory.mkdir(parents=True, exist_ok=True)
    command = make_command(config_args, exe, case, output / "prompts" / (case["prompt_sha256"] + ".tokens"),
                           args.kv_resident)
    case_env = dict(env)
    if case["memory"] == "mmap-ring2":
        case_env["STRATA_STAGER_RING"] = "2"
    result = dict(case=case, status="failed", pairs=[])
    try:
        phases = [("warmup", args.warmup_pairs), ("check", int(args.check_residuals)), ("measure", args.rounds)]
        for phase, count in phases:
            for pair in range(count):
                records = {}
                # Re-run both arms of a half-finished pair so newly measured A/B runs remain adjacent.
                reusable = args.resume
                for arm in ("A", "B"):
                    path = directory / f"{phase}-{pair}-{arm}.json"
                    if (not path.exists() or not path.with_suffix(".log").exists() or
                            json.loads(path.read_text()).get("status") != "ok"):
                        reusable = False
                order = ("A", "B") if (index + pair + args.seed) % 2 == 0 else ("B", "A")
                for arm in order:
                    print(f"  {phase} {pair + 1}/{count} {arm}", end=" ", flush=True)
                    case_env["STRATA_PREFILL_STREAM_AHEAD"] = "0" if arm == "A" else "1"
                    record = run_engine(command, cwd, case_env, directory / f"{phase}-{pair}-{arm}",
                                        case["tokens"], args.timeout, phase == "check", reusable)
                    if record["status"] != "ok":
                        raise ValueError(f"{phase} {pair} {arm}: {record.get('error', 'invalid run')}; {record['log']}")
                    print(f"{record['ms']:.1f} ms / {record['chunks']} chunks", flush=True)
                    records[arm] = record
                validate_pair(records["A"], records["B"])
                if phase == "measure":
                    result["pairs"].append(records)
        result["summary"] = summarize(result["pairs"], args.seed)
        result["status"] = "ok"
    except (OSError, ValueError) as exc:
        result["error"] = str(exc)
    save_json(directory / "case.json", result)
    return result


def write_reports(output, cases, results, skipped):
    by_id = {r["case"]["id"]: r for r in results}
    rows = []
    for case in cases:
        result = by_id.get(case["id"], {})
        rows.append(dict(case, status=result.get("status", "pending"), error=result.get("error", ""),
                         **result.get("summary", {})))
    save_json(output / "summary.json", {"schema": VERSION, "cases": rows, "skipped": skipped})
    columns = list(dict.fromkeys(k for row in rows for k in row))
    with (output / "summary.csv").open("w", newline="", encoding="utf-8") as dest:
        writer = csv.DictWriter(dest, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    lines = ["# Expert streaming A/B benchmark", "",
             "A = `STRATA_PREFILL_STREAM_AHEAD=0`; B = `1`. Positive gain means B is faster.",
             "Times are batched prefill only. Warmup and residual-check runs are excluded.",
             "Each sample starts a fresh engine; this is not a persistent-server benchmark.", "",
             "| Workload | Tokens | Requested chunk | Memory / KV / cache / context | Status | Pairs | A ms | B ms | A tok/s | B tok/s | Gain |",
             "|---|---:|---|---|---|---:|---:|---:|---:|---:|---:|"]
    for row in rows:
        name = row["workload"].replace("|", "\\|")
        settings = f"{row['memory']} / {row['kv']} / {row['expert_cache']} / {row['context']}"
        prefix = f"| {name} | {row['tokens']} | {row['chunk']} | {settings} | {row['status']} |"
        if row["status"] == "ok":
            lines.append(prefix + f" {row['pairs']} | {row['old_ms']:.1f} | {row['new_ms']:.1f} | "
                         f"{row['old_tps']:.1f} | {row['new_tps']:.1f} | {row['throughput_gain_pct']:+.2f}% |")
        else:
            lines.append(prefix + " — | — | — | — | — | — |")
    lines += ["", "JSON/CSV include run ranges, p95 and paired bootstrap intervals (at least three pairs).",
              "Intervals from small samples are exploratory, not proof of a hardware-wide speedup.",
              "Compare A/B within a row. Different chunk sizes, prompts and capacities are different workloads.",
              f"\nSkipped combinations: {len(skipped)}. See summary.json for reasons."]
    for row in rows:
        if row["error"]:
            lines.append(f"\n- Failed {row['id']}: {row['error']}")
    (output / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--exe", type=Path)
    p.add_argument("--preset", choices=PRESETS, default="standard")
    p.add_argument("--lengths", help="comma-separated freshly prefilled token counts")
    p.add_argument("--chunks", help="comma-separated chunk sizes and/or auto")
    p.add_argument("--workloads", help="ramp,random,repeat; defaults to ramp (full: all three; files: none)")
    p.add_argument("--tokens-file", type=Path, action="append", default=[], help="real token corpus; repeatable")
    p.add_argument("--token-min", type=int, default=100)
    p.add_argument("--token-max", type=int, default=299)
    p.add_argument("--memory", default="config", help="config,arena,mmap,mmap-ring2")
    p.add_argument("--kv-modes", default="config", help="config,int8,fp16,q4_0,k8v4")
    p.add_argument("--expert-caches", default="config", help="config,auto or slot counts")
    p.add_argument("--contexts", default="auto", help="capacities; auto fixes one capacity for the entire matrix")
    p.add_argument("--kv-resident", type=int, default=0)
    p.add_argument("--rounds", type=int, default=3, help="measured A/B pairs per case")
    p.add_argument("--warmup-pairs", type=int, default=1, help="discarded A/B pairs per case; warms OS caches")
    p.add_argument("--check-residuals", action="store_true", help="extra correctness pair, excluded from timings")
    p.add_argument("--seed", type=int, default=1729)
    p.add_argument("--ordered", action="store_true", help="disable seeded case shuffling")
    p.add_argument("--timeout", type=float, default=600, help="seconds per engine invocation")
    p.add_argument("--output", type=Path, help="new results directory; required with --resume")
    p.add_argument("--resume", action="store_true", help="reuse successful runs with matching manifest")
    p.add_argument("--dry-run", action="store_true", help="show matrix and process count without GPU work or files")
    p.add_argument("--list-cases", action="store_true", help="print the entire planned matrix")
    p.add_argument("--fail-fast", action="store_true", help="stop after the first failed case")
    return p


def main(argv=None):
    p = parser()
    args = p.parse_args(argv)
    if (args.rounds < 1 or args.warmup_pairs < 0 or args.timeout <= 0 or not math.isfinite(args.timeout)
            or args.token_min < 0 or args.token_max < args.token_min or args.kv_resident < 0):
        p.error("invalid rounds, warmup count, timeout, token range or KV residency")
    if args.resume and not args.output:
        p.error("--resume requires --output")
    try:
        config = json.loads(args.config.read_text(encoding="utf-8"))
        config_args = config["args"]
        if not isinstance(config_args, list) or not all(isinstance(a, str) for a in config_args):
            raise ValueError("config args must be a list of strings")
        if "--serve" in config_args or "--layer-split" in config_args:
            raise ValueError("this suite uses standalone generation; --serve/--layer-split configs are not supported")
        cwd = Path(config.get("cwd", ROOT)).resolve()
        exe = args.exe.resolve() if args.exe else (cwd / config["exe"]).resolve()
        cases, skipped, prompts = build_plan(args)
        if not cases:
            raise ValueError("no runnable cases; check lengths, context capacities and token files")
        for case in cases:
            make_command(config_args, exe, case, Path("prompt.tokens"), args.kv_resident)
    except (OSError, KeyError, TypeError, ValueError) as exc:
        p.error(str(exc))
    invocations = len(cases) * 2 * (args.rounds + args.warmup_pairs + int(args.check_residuals))
    print(f"{len(cases)} cases; {len(skipped)} skipped combinations; {invocations} engine invocations.")
    print(f"Per case: {args.rounds} measured pairs, {args.warmup_pairs} discarded warmup pairs, "
          f"{int(args.check_residuals)} residual-check pairs. A=old; B=new.")
    for case in cases if args.list_cases else cases[:12]:
        print(f"  {case['id']} {case['workload']} tokens={case['tokens']} chunk={case['chunk']} "
              f"memory={case['memory']} kv={case['kv']} cache={case['expert_cache']} context={case['context']}")
    if not args.list_cases and len(cases) > 12:
        print(f"  ... {len(cases) - 12} more; --list-cases shows all combinations.")
    if args.dry_run:
        return 0
    if not exe.is_file() or not cwd.is_dir():
        p.error(f"engine or working directory missing: {exe}, {cwd}")
    env = environment(config)
    hardware = {"platform": platform.platform(), "cpu": platform.processor(), "logical_cpus": os.cpu_count(),
                "nvidia": probe(["nvidia-smi", "--query-gpu=name,uuid,driver_version,pci.bus_id,memory.total",
                                 "--format=csv,noheader"]),
                "amd": probe(["rocm-smi", "--showproductname", "--showdriverversion", "--json"])}
    if Path("/proc/cpuinfo").exists():
        cpu = re.search(r"^model name\s*:\s*(.+)$", Path("/proc/cpuinfo").read_text(), re.MULTILINE)
        if cpu:
            hardware["cpu"] = cpu[1]
    if hasattr(os, "sysconf"):
        hardware["ram_bytes"] = os.sysconf("SC_PHYS_PAGES") * os.sysconf("SC_PAGE_SIZE")
    spec = {"schema": VERSION, "config": config, "config_path": str(args.config.resolve()),
            "exe": str(exe), "exe_sha256": file_hash(exe), "runner_sha256": file_hash(Path(__file__)),
            "cwd": str(cwd), "cases": cases,
            "rounds": args.rounds, "warmup_pairs": args.warmup_pairs, "check_residuals": args.check_residuals,
            "seed": args.seed, "kv_resident": args.kv_resident, "timeout": args.timeout,
            "environment": relevant_env(env), "hardware": hardware, "model_assets": asset_metadata(config_args, cwd)}
    fingerprint = digest(spec)
    output = (args.output or ROOT / "bench" / "results" /
              ("prefill-suite-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ"))).resolve()
    manifest_path = output / "manifest.json"
    if args.resume:
        if not manifest_path.exists() or json.loads(manifest_path.read_text())["fingerprint"] != fingerprint:
            p.error("resume manifest differs or is missing: use the same binary, hardware, config, matrix and settings")
    else:
        if output.exists():
            p.error("output directory already exists; choose a new path or use --resume")
        output.mkdir(parents=True)
        save_json(manifest_path, dict(spec, fingerprint=fingerprint, python=sys.version,
                                     revision=probe(["git", "rev-parse", "HEAD"], ROOT),
                                     git_status=probe(["git", "status", "--short"], ROOT),
                                     created_utc=datetime.now(timezone.utc).isoformat()))
    prompt_dir = output / "prompts"
    prompt_dir.mkdir(exist_ok=True)
    for sha, text in prompts.items():
        path = prompt_dir / (sha + ".tokens")
        if path.exists() and file_hash(path) != sha:
            p.error(f"saved prompt changed: {path}")
        if not path.exists():
            path.write_text(text, encoding="utf-8", newline="")
    results = []
    print(f"Results: {output}", flush=True)
    interrupted = False
    try:
        for index, case in enumerate(cases):
            print(f"\n[{index + 1}/{len(cases)}] {case['workload']}: {case['tokens']} tokens, chunk {case['chunk']}, "
                  f"{case['memory']}, KV {case['kv']}, cache {case['expert_cache']}, context {case['context']}", flush=True)
            result = run_case(case, index, args, config_args, exe, cwd, env, output)
            results.append(result)
            if result["status"] == "ok":
                s = result["summary"]
                print(f"  A {s['old_ms']:.1f} ms -> B {s['new_ms']:.1f} ms; "
                      f"{s['old_tps']:.1f} -> {s['new_tps']:.1f} tok/s; gain {s['throughput_gain_pct']:+.2f}%", flush=True)
            else:
                print(f"  FAILED: {result['error']}", flush=True)
            write_reports(output, cases, results, skipped)
            if result["status"] != "ok" and args.fail_fast:
                break
    except KeyboardInterrupt:
        interrupted = True
        print("\nInterrupted. Completed run records are saved; rerun with --resume.")
    finally:
        write_reports(output, cases, results, skipped)
    passed = sum(r["status"] == "ok" for r in results)
    print(f"\n{passed} passed, {len(results) - passed} failed, {len(cases) - len(results)} pending. "
          f"Reports: {output / 'summary.md'} (also .csv and .json)")
    return 130 if interrupted else 0 if passed == len(cases) else 1


if __name__ == "__main__":
    raise SystemExit(main())
