#!/bin/bash
# Format Python code with black and isort

set -e

cd "$(dirname "$0")/.."

echo "Running isort..."
uv run isort backend/

echo "Running black..."
uv run black backend/

echo "Formatting complete!"
