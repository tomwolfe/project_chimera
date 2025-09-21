# File: project_chimera_context.md
# Project Chimera Codebase Context

This document outlines the necessary structure and files required for effective self-improvement analysis.

## Required Files for Analysis
- `core.py`: Main application logic
- `app.py`: Streamlit web application entry point
- `src/llm_provider.py`: LLM interaction and API handling
- `src/services.py`: Core service implementations
- `src/self_improvement/critique_engine.py`: Critique and analysis logic
- `src/self_improvement/improvement_applicator.py`: Application of improvements
- `src/self_improvement/metrics_collector.py`: Metrics collection for self-improvement
- `src/config/settings.py`: Application settings
- `src/constants.py`: Global constants
- `docs/project_chimera_context.md`: This documentation file itself
- `schemas/analysis_schema.json`: Schema for self-improvement output
- `personas.yaml`: Persona definitions
- `pyproject.toml`: Project configuration
- `.github/workflows/analysis.yml`: Self-analysis workflow
- `.github/workflows/ci.yml`: CI workflow
- `.ruff.toml`: Ruff linter/formatter configuration
- `.pre-commit-config.yaml`: Pre-commit hooks configuration
- `requirements.txt`: Development Python dependencies
- `requirements-prod.txt`: Production Python dependencies
- `Dockerfile`: Docker container definition
- All other files in `src/` and `tests/` directories

## How to Provide Context
When running self-improvement analysis, ensure these files are included in the context provided to the analysis system. The context should include:
- File paths
- File contents
- Directory structure
- Any relevant configuration files

This will enable the system to perform a thorough analysis and identify meaningful improvements.