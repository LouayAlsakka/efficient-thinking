#!/usr/bin/env bash
# ET-IV checker CI — definition of done for machine-queue item 1: all three checkers green.
# Usage: bash checkers/run_tests.sh   (run from repo root)
set -euo pipefail
cd "$(dirname "$0")/.."
V=./.venv/bin/python
echo "=== ET-IV checker test suite ==="
$V checkers/tests/test_meter_rhyme.py
$V checkers/tests/test_lyric_fit.py
$V checkers/tests/test_counterpoint.py
echo "=== ALL CHECKERS GREEN ==="
