#!/usr/bin/env bash
# Extracts the visible prose of the site (pages, course texts, home page, and the UI strings of the widgets and lesson-boot.js,
# brief D15) and runs the ASD-STE100 linter (.claude/skills/asd-ste100, vendored, run unchanged) on it.
# Hard findings (semicolons, long sentences, phrasal verbs, nominalizations, marketing adjectives, synonym rotation) fail;
# passive voice and compound tenses are advisory. Then the site rules that the STE linter does not know (banned teacher-voice
# tics, provenance words, dashes and colons in UI strings) run as site/test/prose.test.mjs. See docs/authoring.md "Writing rules".
# Set PROSE_REPORT=1 to print the banned-tic count per lesson.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
node "$ROOT/tools/prose/extract.mjs"
python3 -I "$ROOT/.claude/skills/asd-ste100/scripts/ste-lint.py" "$ROOT"/build/prose/*.md
node --test "$ROOT/site/test/prose.test.mjs" > "$ROOT/build/prose/site-rules.log" 2>&1 || { cat "$ROOT/build/prose/site-rules.log"; exit 1; }
echo "site rules (banned tics, label colons, UI strings): ok"
