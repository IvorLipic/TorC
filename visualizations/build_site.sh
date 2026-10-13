#!/usr/bin/env bash
# Copies generated data and rendered videos into site/ (the deployable static directory).
# Run after examples/build_and_run.sh and the manim render (-qm).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
mkdir -p "$ROOT/site/data" "$ROOT/site/media"
cp "$ROOT/visualizations/generated/autograd_trace.json" "$ROOT/site/data/"
for s in ReuseScene ChainScene MlpScene; do
  cp "$ROOT/visualizations/media/videos/torc_scenes/720p30/$s.mp4" "$ROOT/site/media/"
  ffmpeg -v error -y -i "$ROOT/visualizations/media/videos/torc_scenes/720p30/$s.mp4" -c copy -movflags +faststart "$ROOT/site/media/$s.mp4"
  ffmpeg -v error -y -i "$ROOT/site/media/$s.mp4" -c:v libvpx-vp9 -crf 34 -b:v 0 -an "$ROOT/site/media/$s.webm"   # fallback for browsers without H.264
done
echo "site/ updated"
