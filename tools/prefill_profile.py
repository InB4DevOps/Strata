#!/usr/bin/env python3
"""Rank STRATA_PREFILL_PROFILE JSONL stream intervals; no third-party packages."""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import sys


def load(path):
    runs = {}
    with open(path, encoding="utf-8") as source:
        for line_number, line in enumerate(source, 1):
            try:
                row = json.loads(line)
                if row.get("schema") != 1:
                    raise ValueError("unsupported schema")
                run = runs.setdefault(row["run"], {"records": []})
                kind = row["type"]
                if kind == "run":
                    if "metadata" in run:
                        raise ValueError("duplicate run header")
                    run["metadata"] = row
                elif kind == "end":
                    if "end" in run:
                        raise ValueError("duplicate run end")
                    run["end"] = row
                else:
                    run["records"].append(row)
            except (ValueError, KeyError, TypeError, AttributeError) as exc:
                raise ValueError(f"{path}:{line_number}: {exc}") from exc
    accepted = [r for r in runs.values() if "metadata" in r and
                r.get("end", {}).get("complete") is True and r["end"].get("valid") is True]
    skipped = len(runs) - len(accepted)
    if not accepted:
        raise ValueError(f"{path}: no complete, valid prefill runs ({skipped} excluded)")
    return accepted, skipped


def signature(runs):
    """Compare workload shapes, not run IDs or selected implementation (which may change)."""
    result = Counter()
    for run in runs:
        m = run["metadata"]
        chunks = sorted({(r["chunk"], r["tokens"]) for r in run["records"] if r["type"] == "gpu"})
        result[(m["gpu"], m["device"], m["position"], m["tokens"], m["layer_begin"],
                m["layer_end"], tuple(chunks))] += 1
    return result


def summarize(runs, group="phase"):
    totals = defaultdict(lambda: {"ms": 0.0, "intervals": 0, "max_ms": 0.0})
    host = Counter()
    counters = Counter()
    streams = Counter()
    for run in runs:
        device = run["metadata"]["device"]
        for row in run["records"]:
            if row["type"] == "host":
                host[row["phase"]] += row["ms"]
            elif row["type"] == "counters":
                counters.update({k: v for k, v in row.items() if k not in ("schema", "run", "type")})
            elif row["type"] == "gpu":
                stream = f"device {row.get('device', device)}/{row['stream']}"
                label = row["phase"]
                if group == "layer":
                    label = f"layer {row['layer']} / {label}"
                elif group == "chunk":
                    label = f"position {row['chunk']} ({row['tokens']} tokens) / {label}"
                elif group == "implementation":
                    label = f"{row['implementation']} [{row['gu_type']}/{row['down_type']}] / {label}"
                t = totals[(stream, label)]
                t["ms"] += row["ms"]
                t["intervals"] += row["intervals"]
                t["max_ms"] = max(t["max_ms"], row["max_ms"])
                streams[stream] += row["ms"]
    rows = []
    for (stream, label), t in totals.items():
        rows.append(dict(stream=stream, label=label, **t,
                         mean_ms=t["ms"] / t["intervals"] if t["intervals"] else 0,
                         percent=100 * t["ms"] / streams[stream] if streams[stream] else 0))
    return {"runs": len(runs), "stage_wall_ms": sum(r["end"]["wall_ms"] for r in runs),
            "rows": sorted(rows, key=lambda r: -r["ms"]), "host_ms": dict(host),
            "counters": dict(counters)}


def compare(report, baseline):
    previous = {(r["stream"], r["label"]): r for r in baseline["rows"]}
    current = {(r["stream"], r["label"]): r for r in report["rows"]}
    rows = []
    for key in previous.keys() | current.keys():
        old, new = previous.get(key, {}).get("ms", 0), current.get(key, {}).get("ms", 0)
        rows.append({"stream": key[0], "label": key[1], "baseline_ms": old, "ms": new,
                     "delta_ms": new - old, "change_percent": 100 * (new / old - 1) if old else None})
    return sorted(rows, key=lambda r: -abs(r["delta_ms"]))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("profile", type=Path)
    parser.add_argument("--group", choices=("phase", "layer", "chunk", "implementation"), default="phase")
    parser.add_argument("--baseline", type=Path)
    parser.add_argument("--top", type=int, default=20)
    parser.add_argument("--json", action="store_true", help="emit the complete summary as JSON")
    args = parser.parse_args(argv)
    if args.top < 1:
        parser.error("--top must be positive")
    try:
        runs, skipped = load(args.profile)
        report = summarize(runs, args.group)
        report["excluded_runs"] = skipped
        if args.baseline:
            baseline, baseline_skipped = load(args.baseline)
            if signature(runs) != signature(baseline):
                raise ValueError("baseline workload differs: match GPU, stage range, token positions/counts, chunks and run count")
            report["comparison"] = compare(report, summarize(baseline, args.group))
            report["baseline_excluded_runs"] = baseline_skipped
    except (OSError, ValueError, KeyError, TypeError) as exc:
        parser.error(str(exc))
    if args.json:
        print(json.dumps(report, indent=2))
        return
    print(f"{report['runs']} completed runs; {skipped} incomplete/invalid runs excluded")
    print(f"Sum of stage wall times: {report['stage_wall_ms']:.3f} ms (overlapping stages are not request latency)")
    print("GPU stream intervals include waits and host launch gaps. Percentages are per stream.")
    print(f"{'total ms':>12} {'stream %':>9} {'intervals':>10} {'max ms':>10}  phase")
    for row in report["rows"][:args.top]:
        print(f"{row['ms']:12.3f} {row['percent']:9.2f} {row['intervals']:10d} {row['max_ms']:10.3f}  "
              f"{row['stream']}: {row['label']}")
    print("\nHost measurements (may overlap GPU work and each other):")
    for label, ms in sorted(report["host_ms"].items(), key=lambda item: -item[1]):
        print(f"{ms:12.3f} ms  {label}")
    if report["counters"]:
        print("\nCounters: " + ", ".join(f"{k}={v}" for k, v in report["counters"].items()))
    if "comparison" in report:
        print("\nCandidate minus baseline (negative is less time):")
        for r in report["comparison"][:args.top]:
            change = f"{r['change_percent']:+.1f}%" if r["change_percent"] is not None else "new"
            print(f"{r['delta_ms']:+12.3f} ms {change:>9}  {r['stream']}: {r['label']}")


if __name__ == "__main__":
    main()
