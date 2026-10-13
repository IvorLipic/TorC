#!/usr/bin/env bash
# Builds TorC at the pinned ref, runs the trace programs and writes site/data/*.json.
#   tools/trace/run.sh           -> writes site/data/autograd_trace.json
#   tools/trace/run.sh --check   -> regenerates into a temp file and fails if site/data differs (CI drift check, numeric tolerance)
# Environment: CXX (default: clang++-18, otherwise c++; needs clang>=18 or GCC>=14), TORC_SRC=<dir> to use a local TorC checkout.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
BUILD="$ROOT/build/trace"
CXX="${CXX:-$(command -v clang++-18 || command -v c++)}"
ARGS=(-S "$ROOT/tools/trace" -B "$BUILD" -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_COMPILER="$CXX")
[ -n "${TORC_SRC:-}" ] && ARGS+=("-DFETCHCONTENT_SOURCE_DIR_TORC=$TORC_SRC")
cmake "${ARGS[@]}" >/dev/null
cmake --build "$BUILD" --target autograd_trace -j"$(nproc)" >/dev/null
OUT="$ROOT/site/data/autograd_trace.json"
if [ "${1:-}" = "--check" ]; then
  TMP="$(mktemp)"; "$BUILD/autograd_trace" > "$TMP"
  node "$ROOT/tools/trace/compare.mjs" "$OUT" "$TMP" || { echo "site/data/autograd_trace.json is stale; run tools/trace/run.sh and commit"; exit 1; }
else
  mkdir -p "$(dirname "$OUT")"; "$BUILD/autograd_trace" > "$OUT"; echo "wrote ${OUT#$ROOT/}"
fi
