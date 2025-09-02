# Project Chimera Context

## Overview

Project Chimera is an AI-powered system designed for [briefly describe the project's main purpose]. This document provides essential context for understanding the project's structure, development practices, and analysis frameworks.

## Project Structure

- `src/`: Contains the core application logic, including LLM integration, persona management, and self-improvement mechanisms.
  - `ai_core/`: Core AI functionalities.
  - `personas/`: Definitions for different AI personas.
  - `utils/`: Utility functions and helpers.
  - `llm_provider.py`: Interface for interacting with Large Language Models.
  - `database_operations.py`: Handles data persistence.
  - `code_validator.py`: Tools for code analysis and validation.
  - `metrics_collector.py`: Collects performance and quality metrics.
  - `prompt_engineering.py`: Utilities for prompt construction and management.
- `tests/`: Unit and integration tests for various components.
- `ai_core/`: Contains the core self-improvement loop and related logic.
- `config/`: Configuration files.
- `docs/`: Documentation files, including this context document.

## Development Environment & Tooling

- **Python Version:** 3.11
- **Dependency Management:** `requirements.txt` (production), `requirements-dev.txt` (development - assumed, not provided)
- **Code Formatting & Linting:** Ruff (`pyproject.toml` configuration)
- **Security Analysis:** Bandit (`pyproject.toml` configuration)
- **Testing Framework:** Pytest
- **Pre-commit Hooks:** Configured in `.pre-commit-config.yaml` for automated checks before commits.

## Development Workflow

1. **Code Changes:** Implement features or fixes.
2. **Pre-commit Hooks:** Run `pre-commit install` and ensure all hooks pass on commit.
3. **Local Testing:** Execute relevant unit and integration tests using `pytest`.
4. **Code Quality Checks:** Run `ruff check .` and `ruff format --check .` locally.
5. **Security Scan:** Run `bandit -r . -ll -c pyproject.toml` locally.
6. **Commit:** Commit changes with a descriptive message.
7. **CI Pipeline:** Push to trigger the CI pipeline, which includes automated checks for formatting, linting, security, and testing.

## Key Metrics & Goals

- **Code Quality:** Maintain high standards through consistent linting and formatting (Ruff).
- **Security:** Proactively identify and mitigate security vulnerabilities (Bandit).
- **Robustness:** Ensure system stability through comprehensive testing and error handling.
- **Efficiency:** Optimize token usage and processing time.
- **Maintainability:** Improve code readability, modularity, and testability.

## Analysis Framework

This document serves as the foundation for the self-improvement analysis. Future analyses should build upon this context, referencing specific components and metrics to provide targeted recommendations.