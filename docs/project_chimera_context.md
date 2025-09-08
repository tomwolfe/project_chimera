# Project Chimera Codebase Context

This document provides a high-level overview of the Project Chimera codebase structure, key components, and dependencies. It serves as a reference for the AI's self-improvement analysis, ensuring that code modifications and suggestions are grounded in the actual project architecture.

## Project Purpose
Project Chimera is an advanced reasoning engine designed for complex problem-solving and code generation through Socratic self-debate methodology. It aims to be a self-optimizing AI development framework.

## Key Directories and Files

*   **`/` (Project Root):**
    *   `app.py`: Main Streamlit web application.
    *   `core.py`: Core Socratic debate engine logic.
    *   `personas.yaml`: Centralized definitions and configurations for all AI personas and their sets.
    *   `pyproject.toml`: Project configuration for tools like Ruff, Bandit, and pytest.
    *   `requirements.txt`: Development Python dependencies.
    *   `requirements-prod.txt`: Production Python dependencies.
    *   `Dockerfile`: Docker container definition for deployment.
    *   `.github/workflows/ci.yml`: GitHub Actions CI/CD pipeline.
    *   `.pre-commit-config.yaml`: Pre-commit hook configurations.
    *   `README.md`: Project overview and setup instructions.
*   **`src/`:** Main application source code.
    *   `src/config/`: Configuration settings, model registry, and persistence logic.
    *   `src/context/`: Modules for codebase context analysis and relevance.
    *   `src/llm_provider.py`: Interface for interacting with Large Language Models (currently Gemini).
    *   `src/models.py`: Pydantic models for structured data validation.
    *   `src/persona/`: Persona routing and management logic.
    *   `src/resilience/`: Circuit breaker and rate limiter implementations.
    *   `src/self_improvement/`: Modules specific to the self-improvement process (metrics, content validation, prompt optimization).
    *   `src/tokenizers/`: Token counting implementations.
    *   `src/utils/`: General utility functions (command execution, error handling, output parsing, UI helpers, path utilities).
*   **`tests/`:** Unit and integration tests.
    *   `tests/integration/`: Integration tests.
    *   `tests/unit/`: Unit tests.
*   **`docs/`:** Project documentation.
    *   `docs/persona_alignment_guidelines.md`: Guidelines for persona output.
    *   `docs/prompt_optimizations.md`: Strategies for prompt optimization.
    *   `docs/system_improvement_strategy.md`: Overall system improvement strategy.
    *   `docs/token_optimization_strategy.md`: Detailed token optimization strategy.

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