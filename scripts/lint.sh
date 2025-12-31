#!/bin/bash
# Run linting checks on Python code

set -e

cd "$(dirname "$0")/.."

echo "Running ruff..."
uv run ruff check backend/

echo "Checking black formatting..."
uv run black --check backend/

echo "Checking isort..."
uv run isort --check-only backend/

echo "All checks passed!"
