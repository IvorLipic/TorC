#!/usr/bin/env bash
# Extracts the visible prose of the site and runs the ASD-STE100 linter (.claude/skills/asd-ste100, vendored) on it.
# Hard findings (semicolons, long sentences, phrasal verbs, nominalizations, marketing adjectives, synonym rotation) fail;
# passive voice and compound tenses are advisory. See docs/authoring.md "Writing rules".
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
node "$ROOT/tools/prose/extract.mjs"
python3 -I "$ROOT/.claude/skills/asd-ste100/scripts/ste-lint.py" "$ROOT"/build/prose/*.md
