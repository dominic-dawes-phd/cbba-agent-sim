#!/usr/bin/env bash
# Runs the test suite and the linters, the same checks as before every commit (about 3 minutes).
# Usage, from the repo root:  bash scripts/check.sh
# Needs the venv with the dev extras: pip install -e .[dev,analysis,solver] (README, "Setup").
set -euo pipefail
cd "$(dirname "$0")/.."

if [ -x .venv/Scripts/python ]; then
  bin=.venv/Scripts      # Windows
else
  bin=.venv/bin          # Linux, macOS
fi

echo "== tests"
"$bin/python" -m pytest -q
echo "== lint"
"$bin/ruff" check .
# Format-check the Python packages only: on "." ruff would also reformat code blocks in docs/*.md.
"$bin/ruff" format --check allocation experiments llm sim tests
echo "all checks passed"
