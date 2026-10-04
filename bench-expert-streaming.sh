#!/usr/bin/env bash
# A/B the rebuilt engine with the local Coder configuration. Extra arguments override defaults.
set -euo pipefail
ROOT="$(dirname -- "$(realpath -- "${BASH_SOURCE[0]}")")"
exec python3 "$ROOT/tools/test_prefill_streaming.py" \
    --config "$ROOT/strata-coder-iq1_m.json" \
    --exe "$ROOT/build/strata" \
    --chunk 256 --tokens 1024 --rounds 3 --benchmark "$@"
