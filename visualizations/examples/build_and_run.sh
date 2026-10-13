#!/usr/bin/env bash
# Builds the torc library (needs clang >= 18 or GCC >= 14: tensor.hpp uses C++23 deducing-this),
# compiles autograd_trace.cpp and writes visualizations/generated/autograd_trace.json.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
CXX="${CXX:-clang++-18}"
BUILD="$ROOT/build-viz"
cmake -S "$ROOT" -B "$BUILD" -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_COMPILER="$CXX" >/dev/null
cmake --build "$BUILD" --target torc -j"$(nproc)" >/dev/null
"$CXX" -std=c++2b -O1 -march=native -I"$ROOT/include" "$ROOT/visualizations/examples/autograd_trace.cpp" "$BUILD/libtorc.a" -o "$BUILD/autograd_trace"
mkdir -p "$ROOT/visualizations/generated"
"$BUILD/autograd_trace" > "$ROOT/visualizations/generated/autograd_trace.json"
echo "wrote visualizations/generated/autograd_trace.json"
