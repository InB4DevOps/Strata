#!/usr/bin/env bash
# Full matrix runner. Use --dry-run to inspect the workload before running it.
set -euo pipefail
ROOT="$(dirname -- "$(realpath -- "${BASH_SOURCE[0]}")")"
exec python3 "$ROOT/tools/bench_prefill_suite.py" \
    --config "$ROOT/strata-coder-iq1_m.json" --exe "$ROOT/build/strata" "$@"
