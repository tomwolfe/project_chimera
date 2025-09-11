# 🧬 Project Chimera

[![CI Status](https://github.com/tomwolfe/project_chimera/actions/workflows/ci.yml/badge.svg)](https://github.com/tomwolfe/project_chimera/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.11+](https://img.shields.io/badge/Python-3.11%2B-blue.svg)](https://www.python.org/downloads/)
[![Ruff](https://img.shields.io/badge/linter/formatter-Ruff-blue.svg)](https://docs.astral.sh/ruff/)
[![Bandit](https://img.shields.io/badge/security-Bandit-yellow.svg)](https://bandit.readthedocs.io/en/latest/)
[![Test Coverage](https://img.shields.io/badge/Coverage-80%25%2B-brightgreen.svg)](https://github.com/tomwolfe/project_chimera/blob/main/pyproject.toml#L48)
[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://project-chimera-406972693661.us-central1.run.app/)
[![Live Demo](https://img.shields.io/badge/Live%20Demo-Google%20Cloud%20Run-blueviolet.svg)](https://project-chimera-406972693661.us-central1.run.app/)

**Project Chimera** is your AI-powered partner for continuous code evolution. It intelligently analyzes, diagnoses, and refactors your Python codebase using static analysis (Bandit, Ruff) and advanced LLM reasoning. Chimera automates code maintenance, generates actionable improvement plans, and maintains a transparent audit trail, embodying a philosophy of perpetual self-reflection and improvement.

In the fast-evolving landscape of software development, maintaining high code quality, security, and efficiency is paramount. Project Chimera addresses this challenge by allowing your codebase to intelligently evolve alongside your development efforts, turning maintenance into an automated, intelligent, and philosophically grounded workflow.

---

## 🚀 Features

-   🔍 **Self-Scanning**: Recursively analyzes `.py` files in the codebase for security and style violations.
-   🛡️ **Static Analysis**: Integrates **Bandit** (security-focused) and **Ruff** (blazing-fast linter/formatter).
-   🤖 **AI-Powered Fixes**: Uses LLMs to generate structured, line-specific improvement plans with rationale and impact summaries.
-   📜 **Audit Trail**: Automatically logs every scan and fix attempt with timestamps, file changes, and success/failure status.
-   🔄 **Safe Auto-Apply**: Preview fixes before applying — or run in `--dry-run` mode for zero-risk experimentation.
-   🌐 **Streamlit UI**: Beautiful, interactive dashboard for scanning, reviewing issues, generating fixes, and viewing logs.
-   ⚙️ **CI/CD Ready**: Preconfigured GitHub Actions workflow for automated scanning on every `push` or `pull_request`.
-   🧪 **Test Coverage**: Includes unit and integration tests to ensure reliability as the system evolves.

---

## 🛠️ Installation & Setup

### Prerequisites

-   Python 3.9 or higher (Python 3.11+ recommended)
-   `pip` (or `poetry` for dependency management)
-   Access to a **Google Gemini-compatible LLM API** (e.g., Google AI Studio).

### Quick Start

1.  **Clone the repository**

    ```bash
    git clone https://github.com/tomwolfe/project_chimera.git
    cd project_chimera
    ```

2.  **Install dependencies**

    With `pip`:

    ```bash
    pip install -r requirements.txt
    ```

    With `poetry` (if `pyproject.toml` is configured for it):

    ```bash
    poetry install
    ```

3.  **Configure environment (optional but recommended)**

    Create a `.env` file in the project root and add your Gemini API key:

    ```env
    GEMINI_API_KEY=your-gemini-api-key-here
    # Optional: customize model or endpoint
    # GEMINI_BASE_URL=https://api.gemini.com/v1
    # MODEL_NAME=gemini-2.5-pro
    ```
    *Note: The application prioritizes API keys from environment variables or Streamlit secrets for security.*

4.  **Launch the app**

    ```bash
    streamlit run app.py
    ```

    ➡️ Open your browser to `http://localhost:8501`

---

## 🧪 Usage

### 💻 Web Interface (Recommended)

Use the Streamlit dashboard to:

-   ✅ Initiate code scans
-   📊 View categorized issues (Security / Style)
-   🧠 Generate AI improvement plans
-   👀 Preview proposed file edits
-   🛠️ Apply fixes with one click
-   📖 Browse audit logs with filters and search

### 🖥️ Command Line Interface

Scan codebase and output results:

```bash
python chimera/scanner.py
```

Generate fixes without applying them:

```bash
python chimera/fixer.py --dry-run
```

Apply generated fixes to disk:

```bash
python chimera/fixer.py --apply
```

---

## 📁 Project Structure

```
project_chimera/
├── app.py                  # Main Streamlit application
├── core.py                 # Core Socratic debate engine logic
├── personas.yaml           # Centralized definitions and configurations for AI personas
├── pyproject.toml          # Project configuration (Ruff, Bandit, Pytest, etc.)
├── requirements.txt        # Development dependencies
├── requirements-prod.txt   # Production dependencies
├── Dockerfile              # Docker container definition for deployment
├── .github/workflows/
│   └── ci.yml              # GitHub Actions CI/CD pipeline
├── .pre-commit-config.yaml # Pre-commit hook configurations
├── README.md               # Project overview and setup instructions
├── .env.example            # Template for environment variables
├── LICENSE                 # Project license file
├── src/                    # Main application source code
│   ├── config/             # Configuration settings, model registry, persistence
│   ├── context/            # Modules for codebase context analysis
│   ├── llm_provider.py     # Interface for LLM interactions
│   ├── models.py           # Pydantic models for structured data
│   ├── persona/            # Persona routing and management logic
│   ├── resilience/         # Circuit breaker and rate limiter implementations
│   ├── self_improvement/   # Modules specific to self-improvement analysis
│   ├── tokenizers/         # Token counting implementations
│   └── utils/              # General utility functions
├── tests/                  # Unit and integration tests
│   ├── integration/        # Integration tests
│   └── unit/               # Unit tests
└── docs/                   # Project documentation
    ├── persona_alignment_guidelines.md
    ├── prompt_optimizations.md
    ├── system_improvement_strategy.md
    └── token_optimization_strategy.md
```

---

## 🧪 Testing & Quality

Run the full test suite:

```bash
pytest tests/
```

Generate coverage report:

```bash
pytest tests/ --cov=src --cov-report=html
```

> 💡 Tests include mocking LLM calls and verifying file patch logic — safe to run even without an API key.

---

## 🤖 AI Integration

Chimera uses LLMs to interpret static analysis output and generate context-aware, minimal-impact fixes. It:

-   Structures output using Pydantic for reliability
-   Targets specific line numbers for surgical edits
-   Explains *why* each change is recommended
-   Avoids hallucinated or unsafe modifications
-   Logs prompts and responses for full traceability

*Note: You must provide your own LLM API key. Local LLMs (via Ollama, LM Studio, etc.) are supported by setting `GEMINI_BASE_URL` if your LLM provider supports an OpenAI-compatible API endpoint.*

---

## 🔄 CI/CD Automation

The included GitHub Actions workflow:

1.  Installs Python dependencies
2.  Runs Bandit and Ruff for security and style checks
3.  ❗ Fails the build if critical issues are detected
4.  (Future) Auto-generates fixes in PR comments

Enable by pushing to any branch — no setup required.

---

## 📜 License

**MIT License** — Use freely in personal or commercial projects. Fork it, modify it, break it, fix it.
See `LICENSE` file for full terms.

---

## 🤝 Contributing

We welcome contributions to Project Chimera! Whether it's improving existing features, adding new ones, fixing bugs, or enhancing documentation, your input is valuable.

1.  **Fork the repository** on GitHub.
2.  **Clone your fork** locally: `git clone https://github.com/your-username/project_chimera.git`
3.  **Create a new branch** for your feature or fix: `git checkout -b feat/your-feature-branch`
4.  **Make your changes**, ensuring they adhere to the project's coding standards (linting and formatting enforced by Ruff).
5.  **Add tests** for any new functionality or bug fixes.
6.  **Commit your changes** with clear and descriptive messages.
7.  **Push your branch** to your fork: `git push origin feat/your-feature-branch`
8.  **Open a Pull Request** against the `main` branch of the original repository.

Please also review our [Code of Conduct](https://github.com/tomwolfe/project_chimera/blob/main/CODE_OF_CONDUCT.md) to foster a positive and inclusive community.

---

## 🌐 Connect With Us

*   **GitHub**: [tomwolfe/project_chimera](https://github.com/tomwolfe/project_chimera)
*   **Twitter**: [@Proj_Chimera](https://x.com/Proj_Chimera)
*   **Live Demo**: [Google Cloud Run](https://project-chimera-406972693661.us-central1.run.app/)

## 🙏 Acknowledgments

Project Chimera builds upon numerous open-source projects and frameworks. Special thanks to the Python, Streamlit, Bandit, Ruff, and GitHub Actions communities for their excellent tools and libraries.

---

> "The only true wisdom is in knowing you know nothing." - Socrates
> Project Chimera embodies this philosophy through continuous self-reflection and improvement.
