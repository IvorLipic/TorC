# Vendored copy

Source: https://github.com/danyuchn/asd-ste100-skill at commit 3944283 (MIT License, see LICENSE; copyright Dustin Yuchen Teng).
Copied unchanged. Update by copying a newer commit over these files and changing the commit above.

How TorC Learn uses it (see CLAUDE.md and docs/authoring.md "Writing rules"):
- Every sentence a visitor sees is written in the skill's **STE-flavored** mode, plus the site's own rules.
- `tools/prose/check.sh` extracts the visible prose of every page to `build/prose/*.md` and runs `scripts/ste-lint.py` on it.
  CI runs it in `.github/workflows/verify.yml`. Hard findings fail the run; passive voice and compound tenses are advisory.
