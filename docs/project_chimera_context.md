# Project Chimera Codebase Context

This document provides a high-level overview of the Project Chimera codebase structure, key components, and dependencies. It serves as a reference for the AI's self-improvement analysis, ensuring that code modifications and suggestions are grounded in the actual project architecture.

## Project Purpose
Project Chimera is an advanced reasoning engine designed for complex problem-solving and code generation through Socratic self-debate methodology. It aims to be a self-optimizing AI development framework.

## Directory Structure

- **`/` (Project Root):**
  - `app.py`: Main Streamlit web application.
  - `core.py`: Core Socratic debate engine logic.
  - `personas.yaml`: Centralized definitions and configurations for all AI personas and their sets.
  - `pyproject.toml`: Project configuration for tools like Ruff, Bandit, and pytest.
  - `requirements.txt`: Development Python dependencies.
  - `requirements-prod.txt`: Production Python dependencies.
  - `Dockerfile`: Docker container definition for deployment.
  - `.dockerignore`: Specifies files/directories to ignore when building Docker images.
  - `.gitignore`: Specifies intentionally untracked files to ignore.
  - `.pre-commit-config.yaml`: Pre-commit hook configurations.
  - `README.md`: Project overview and setup instructions.
  - `LICENSE`: Project's license file.

- **`.github/`:** GitHub Actions workflows.
  - `workflows/`: Contains CI/CD and self-analysis workflow definitions.
    - `analysis.yml`: Workflow for triggering self-improvement analysis.
    - `ci.yml`: Continuous Integration pipeline.

- **`config/`:** Application settings, model registry, and persistence logic.
  - `config.yaml`: Main application settings.
  - `user_overrides.yaml`: User-defined overrides for personas and frameworks.
  - `model_registry.py`: Defines LLM model specifications.
  - `persistence.py`: Handles loading/saving of configurations.
  - `settings.py`: Pydantic-based application settings.

- **`docs/`:** Project documentation and guidelines.
  - `persona_alignment_guidelines.md`: Guidelines for persona output.
  - `project_chimera_context.md`: Contextual information for AI self-analysis (this file).
  - `prompt_optimizations.md`: Strategies for prompt optimization.
  - `system_improvement_strategy.md`: Overall system improvement strategy.
  - `token_optimization_strategy.md`: Detailed token optimization strategy.

- **`prompts/`:** Jinja2 templates for dynamic persona prompts.

- **`schemas/`:** JSON schemas for validating LLM outputs.
  - `analysis_schema.json`: Schema for self-improvement analysis output.

- **`scripts/`:** Utility scripts.
  - `fix_ruff.sh`: Script to auto-fix Ruff issues.
  - `run_analysis.py`: Script to trigger self-improvement analysis.

- **`src/`:** Main application source code.
  - `__init__.py`: Python package initializer.
  - `config/`: Configuration-related modules.
  - `constants.py`: Global constants.
  - `context/`: Codebase scanning and context relevance analysis.
    - `context_analyzer.py`: Core logic for context analysis and codebase scanning.
  - `database/`: Database operations (e.g., SQLite for user data).
    - `db_operations.py`: Database interaction logic.
  - `exceptions.py`: Custom exception classes.
  - `llm_provider.py`: Interface for Google Gemini API.
  - `llm_tokenizers/`: Tokenizer implementations.
    - `base.py`: Base tokenizer interface.
    - `gemini_tokenizer.py`: Gemini-specific tokenizer.
  - `models.py`: Pydantic data models for structured data.
  - `persona/`: Persona definitions, routing logic.
    - `routing.py`: Persona routing logic.
  - `persona_manager.py`: Manages persona configurations and performance.
  - `resilience/`: Circuit breaker and rate limiter implementations.
    - `circuit_breaker.py`: Circuit breaker implementation.
  - `self_improvement/`: Modules for metrics collection, content validation.
    - `content_validator.py`: Validates content alignment.
    - `metrics_collector.py`: Collects self-improvement metrics.
  - `services.py`: General service functions.
  - `token_tracker.py`: Tracks token usage and costs.
  - `utils/`: General utility functions.
    - `api_key_validator.py`: API key validation.
    - `code_utils.py`: Code manipulation and analysis utilities.
    - `code_validator.py`: Code validation (Ruff, Bandit, AST).
    - `command_executor.py`: Safe command execution.
    - `domain_recommender.py`: Recommends domains from keywords.
    - `error_handler.py`: Standardized error handling.
    - `file_io/`: File I/O utilities.
    - `json_utils.py`: JSON utility functions.
    - `output_formatter.py`: Formats analysis results.
    - `output_parser.py`: Parses and validates LLM outputs.
    - `path_utils.py`: Path manipulation and validation.
    - `prompt_analyzer.py`: Analyzes prompt complexity.
    - `prompt_engineering.py`: Prompt formatting utilities.
    - `prompt_optimizer.py`: Optimizes prompts.
    - `report_generator.py`: Generates Markdown reports.
    - `session/`: Streamlit session management.
    - `validation/`: Validation utilities.

- **`tests/`:** Unit and integration tests.
  - `integration/`: Integration tests.
  - `quality/`: Code quality tests.
  - `unit/`: Unit tests.

## Key Dependencies and Tools

*   **Python 3.11+**
*   **Google Generative AI API**
*   **Streamlit**: For the interactive UI.
*   **Ruff**: Linting and formatting.
*   **Bandit**: Security scanning.
*   **pytest**: Testing framework.
*   **Sentence-Transformers**: For semantic search in context analysis.
*   **Pydantic**: For data validation.
*   **GitHub Actions**: CI/CD.
*   **Docker**: For containerization.

## Self-Analysis Focus Areas

When performing self-analysis, the AI should prioritize:
*   **Reasoning Quality:** Clarity, consistency, and logical soundness of the debate process.
*   **Robustness:** Error handling, resilience, and stability of core components.
*   **Efficiency:** Token usage, processing speed, and resource management.
*   **Maintainability:** Code structure, readability, documentation, and testability.
*   **Security:** Vulnerability identification and mitigation.
*   **CI/CD & Deployment:** Automation, reliability, and operational posture.