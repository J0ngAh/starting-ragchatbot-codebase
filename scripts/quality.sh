#!/bin/bash
# Run all code quality checks (format + lint + test)

set -e

cd "$(dirname "$0")/.."

echo "=== Formatting Code ==="
./scripts/format.sh

echo ""
echo "=== Running Linter ==="
./scripts/lint.sh

echo ""
echo "=== Running Tests ==="
uv run pytest

echo ""
echo "All quality checks passed!"
