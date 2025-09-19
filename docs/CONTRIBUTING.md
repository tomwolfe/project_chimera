# Contributing to Project Chimera

We welcome contributions to Project Chimera! Please follow these guidelines to ensure a smooth and consistent development process.

## Project Overview

Project Chimera is an innovative AI system that leverages a Socratic self-debate methodology to solve complex problems, generate code, and continuously improve its own capabilities. It features a dynamic multi-persona architecture, context-aware reasoning, and a Streamlit-based interactive interface.

## Code Standards

*   Follow [PEP 8 guidelines](https://peps.python.org/pep-0008/) for Python code.
*   Use clear, descriptive, and consistent variable and function names.
*   Add comprehensive docstrings to all public modules, classes, and functions, explaining their purpose, arguments, and return values.
*   Ensure inline comments are used for complex logic or non-obvious implementations.

## Branching Strategy

*   **`main` branch:** Represents the latest stable release. All new features and bug fixes are merged into `main` after review.
*   **`develop` branch:** Used for ongoing development. Feature branches are typically based off `develop` and merged back into it.
*   **Feature branches:** Create a new branch for each feature or bug fix (e.g., `feature/add-new-reasoning-module`, `bugfix/fix-token-counting`).
*   Aim to keep branches focused on a single task.

## Pull Request Process

1.  **Fork the repository** and clone your fork.
2.  **Create a new branch** from `main` or `develop` (depending on the scope of your change).
3.  **Make your changes**, ensuring they adhere to the [Code Standards](#code-standards).
4.  **Run tests** (`pytest`) and ensure existing tests pass. Add new unit and/or integration tests for your changes, aiming for high code coverage.
5.  **Install and run pre-commit hooks** (`pre-commit install && pre-commit run --all-files`) to check for linting, formatting, and basic security issues before committing.
6.  **Commit your changes** with clear and concise commit messages (e.g., `feat: Add new feature`, `fix: Resolve bug in X`).
7.  **Push your branch** to your fork.
8.  **Submit a Pull Request** to the `main` branch of the original repository.
9.  **Provide a clear description** of your changes, their rationale, and expected impact. Reference any related issues.
10. **Be prepared to address feedback** from reviewers.

## Project Structure

*   `app.py`: Main Streamlit web application UI.
*   `core.py`: The central Socratic Debate orchestration engine.
*   `src/`: Main application source code.
    *   `config/`: Configuration-related modules.
    *   `context/`: Codebase scanning and context relevance analysis.
    *   `database/`: Database operations.
    *   `exceptions.py`: Custom exception classes.
    *   `llm_provider.py`: Interface for Google Gemini API.
    *   `llm_tokenizers/`: Tokenizer implementations.
    *   `models.py`: Pydantic data models for structured data.
    *   `persona/`: Persona definitions, routing logic.
    *   `persona_manager.py`: Manages persona configurations and performance.
    *   `resilience/`: Circuit breaker and rate limiter implementations.
    *   `self_improvement/`: Modules for metrics collection, content validation.
    *   `token_tracker.py`: Tracks token usage and costs.
    *   `utils/`: General utility functions (parsers, validators, loggers, UI helpers, etc.).
*   `docs/`: Project documentation.
*   `prompts/`: Jinja2 templates for dynamic persona prompts.
*   `schemas/`: JSON schemas for validating LLM outputs.
*   `scripts/`: Utility scripts.
*   `tests/`: Unit and integration tests.

## Reporting Issues

If you encounter bugs, have suggestions for new features, or notice areas for improvement, please open an issue on the project's GitHub repository. Provide as much detail as possible, including steps to reproduce bugs, expected behavior, and relevant environment information.
