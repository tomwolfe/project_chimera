# Project Chimera Context Documentation

## Project Purpose
Project Chimera is an advanced reasoning engine designed for complex problem-solving and code generation through Socratic self-debate methodology. This innovative system critically analyzes its own codebase to identify and implement improvements, creating a self-optimizing AI development framework.

## Architecture Overview
Project Chimera follows a multi-layered architecture:
- **Core Engine**: Handles the Socratic debate process and manages persona interactions
- **LLM Interface**: Provides abstraction for different LLM providers (currently Gemini)
- **Self-Improvement Framework**: Analyzes code quality, security, and performance
- **Validation Tools**: Integrates Ruff, Bandit, and pytest for code quality checks

## Key Components
- **`src/socratic_debate.py`**: Core implementation of the Socratic debate framework
- **`src/llm_provider.py`**: Interface for interacting with Large Language Models
- **`src/personas/`**: Directory containing persona definitions and configurations
- **`src/utils/`**: Utility functions for token counting, logging, and prompt engineering
- **`tests/`**: Test suite for core functionality
- **`.github/workflows/ci.yml`**: CI pipeline configuration
- **`pyproject.toml`**: Project configuration for Ruff, Bandit, and other tools

## Dependencies
- Python 3.11+
- Google Generative AI API
- Streamlit (for UI)
- Ruff (linting and formatting)
- Bandit (security scanning)
- pytest (testing framework)

## Setup Instructions
1. Clone the repository: `git clone https://github.com/tomwolfe/project_chimera.git`
2. Install dependencies: `pip install -r requirements.txt`
3. Set up environment variables: `export GEMINI_API_KEY=your_api_key`
4. Run the application: `streamlit run main.py`

## Analysis Context
When performing self-analysis, focus on:
- Reasoning quality of the debate process
- Robustness of error handling
- Efficiency of token usage
- Maintainability of code structure
- Security of API key management