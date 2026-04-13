#!/usr/bin/env bash
# Run all test suites: pytest for tests/ + standalone scripts in optimisers/.
# Usage: bash repo/tests/run_all_tests.sh
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PYTHON="${REPO_ROOT}/../.conda/bin/python"
export PYTHONPATH="${REPO_ROOT}:${REPO_ROOT}/parameters:${REPO_ROOT}/analysis"

echo "========================================"
echo "Running pytest tests/"
echo "========================================"
"$PYTHON" -m pytest "${REPO_ROOT}/tests/" -v --tb=short
echo

passed=0
failed=0
failed_files=()

run_script() {
    local f="$1"
    local name
    name="$(basename "$f")"
    echo "========================================"
    echo "Running: $name"
    echo "========================================"
    if "$PYTHON" "$f"; then
        passed=$((passed + 1))
    else
        failed=$((failed + 1))
        failed_files+=("$name")
    fi
    echo
}

# Existing optimizer functional tests
for f in "${REPO_ROOT}"/optimisers/test_*.py; do
    [ -f "$f" ] && run_script "$f"
done

# Existing parity checks
for f in "${REPO_ROOT}"/optimisers/compare_*.py; do
    [ -f "$f" ] && run_script "$f"
done

echo "========================================"
echo "SCRIPTS: ${passed} passed, ${failed} failed"
if [ "$failed" -gt 0 ]; then
    echo "Failed: ${failed_files[*]}"
fi
echo "========================================"
[ "$failed" -eq 0 ]
