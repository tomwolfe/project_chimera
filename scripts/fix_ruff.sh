#!/bin/bash
# This script is intended to be run once to fix existing Ruff formatting issues.
# It should not be part of the regular CI or pre-commit hooks.

echo "Running Ruff formatter to fix existing issues..."
ruff format .

if [ $? -eq 0 ]; then
    echo "Ruff formatting fixed successfully."
else
    echo "Error: Ruff formatter encountered issues."
    exit 1
fi
