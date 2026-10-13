#!/usr/bin/env bash
# Copies the real-library trace into the deployable static site and runs the site tests.
# Run after examples/build_and_run.sh. (The site is plain HTML/JS; there is no bundling step.)
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
mkdir -p "$ROOT/site/data"
cp "$ROOT/visualizations/generated/autograd_trace.json" "$ROOT/site/data/"
node --test "$ROOT/site/test/engine.test.mjs"
echo "site/ updated and tests passed"
