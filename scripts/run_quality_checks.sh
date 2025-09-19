#!/bin/bash

echo "Running Ruff (Linter Check)..."
ruff check . --output-format=github --exit-non-zero-on-fix
if [ $? -ne 0 ]; then
  echo "Ruff linting failed."
  exit 1
fi

echo "Running Ruff (Formatter Validation)..."
ruff format --check --diff --exit-non-zero-on-fix
if [ $? -ne 0 ]; then
  echo "Ruff formatting failed."
  exit 1
fi

echo "Running Bandit Security Scan..."
bandit -r . --config pyproject.toml --severity-level HIGH --confidence-level HIGH --exit-on-error
if [ $? -ne 0 ]; then
  echo "Bandit security scan failed."
  exit 1
fi

echo "Checking dependencies for vulnerabilities with Safety..."
# Allow Safety to fail without failing the build immediately, report is still generated
safety check --full-report || true

echo "All quality and security checks completed."
