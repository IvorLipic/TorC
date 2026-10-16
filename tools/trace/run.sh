#!/usr/bin/env bash
# Builds TorC at the pinned ref, runs every trace program and writes site/data/*.json.
#   tools/trace/run.sh           -> (re)writes all JSON files listed in TRACES below
#   tools/trace/run.sh --check   -> regenerates into a temp dir and fails if any site/data/*.json differs from a fresh run
#                                   (numeric tolerance, see compare.mjs), is missing, or has no generator (CI drift check)
# Environment: CXX (default: clang++-18, otherwise c++; needs clang>=18 or GCC>=14), TORC_SRC=<dir> to use a local TorC checkout.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
BUILD="$ROOT/build/trace"
DATA="$ROOT/site/data"
CXX="${CXX:-$(command -v clang++-18 || command -v c++)}"

# CMake target -> JSON file in site/data. Add a line here (and to TRACE_TARGETS in CMakeLists.txt) for a new trace program.
TRACES=(
  "autograd_trace:autograd_trace.json"
  "trace_why:why-gradients.json"
  "trace_optim:optim.json"
  "trace_linear:linear.json"
  "trace_tensors:tensors.json"
)

ARGS=(-S "$ROOT/tools/trace" -B "$BUILD" -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_COMPILER="$CXX")
[ -n "${TORC_SRC:-}" ] && ARGS+=("-DFETCHCONTENT_SOURCE_DIR_TORC=$TORC_SRC")
cmake "${ARGS[@]}" >/dev/null
TARGETS=(); for t in "${TRACES[@]}"; do TARGETS+=("${t%%:*}"); done
cmake --build "$BUILD" --target "${TARGETS[@]}" -j"$(nproc)" >/dev/null

run_all() {  # $1 = output directory
  mkdir -p "$1"
  for t in "${TRACES[@]}"; do
    "$BUILD/${t%%:*}" > "$1/${t##*:}" || { echo "trace program ${t%%:*} failed (a built-in self-check or exception); see its stderr above" >&2; exit 1; }
  done
}

if [ "${1:-}" = "--check" ]; then
  TMP="$(mktemp -d)"; trap 'rm -rf "$TMP"' EXIT
  run_all "$TMP"
  bad=0
  for t in "${TRACES[@]}"; do
    f="${t##*:}"
    if [ ! -f "$DATA/$f" ]; then echo "site/data/$f is missing; run tools/trace/run.sh and commit"; bad=1; continue; fi
    node "$ROOT/tools/trace/compare.mjs" "$DATA/$f" "$TMP/$f" || { echo "site/data/$f is stale; run tools/trace/run.sh and commit"; bad=1; }
  done
  for f in "$DATA"/*.json; do
    [ -e "$f" ] || continue
    b="$(basename "$f")"; known=0
    for t in "${TRACES[@]}"; do [ "${t##*:}" = "$b" ] && known=1; done
    [ "$known" = 1 ] || { echo "site/data/$b has no generator in tools/trace/run.sh (site/data is generated-only)"; bad=1; }
  done
  [ "$bad" = 0 ] && echo "all ${#TRACES[@]} trace files match a fresh run" || exit 1
else
  mkdir -p "$DATA"
  run_all "$DATA"
  for t in "${TRACES[@]}"; do echo "wrote site/data/${t##*:}"; done
fi
