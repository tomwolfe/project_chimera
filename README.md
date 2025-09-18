# Project Chimera: The Self-Improving AI Reasoning Engine

Project Chimera is an innovative AI system that leverages a Socratic self-debate methodology to solve complex problems, generate code, and continuously improve its own capabilities. It features a dynamic multi-persona architecture, context-aware reasoning, and a Streamlit-based interactive interface.

## ✨ Key Features

*   **🚀 Socratic Debate Core:** Orchestrates a debate among specialized AI personas to explore problems from multiple angles, fostering robust and nuanced solutions.
*   **🧠 Self-Improvement Loop:** An experimental, automated feature allowing the AI to analyze its own codebase and debate process, identify areas for improvement (reasoning quality, robustness, efficiency, maintainability), and suggest actionable code changes.
*   **🌐 Context-Aware Reasoning:** Utilizes semantic search and codebase scanning to provide relevant contextual information to AI personas, significantly enhancing performance for code-related tasks.
*   **🛠️ Dynamic Persona Management:** Configurable AI personas with distinct roles, system prompts, temperatures, and token limits, supporting domain-specific reasoning frameworks (e.g., Software Engineering, Business, Science, Creative, Self-Improvement).
*   **💻 Code Generation & Validation:** Generates code snippets and modifications, with integrated validation for syntax, style (Ruff), and security (Bandit, AST checks).
*   **🛡️ Robust LLM Interaction:** Implements circuit breakers, rate limiting, and retry mechanisms for resilient and cost-effective interaction with Google Gemini models.
*   **💡 Token Optimization:** Dynamically adjusts prompts and summarizes content using a Hugging Face summarization pipeline to manage token consumption and optimize API costs.
*   **🤝 Conflict Resolution:** Automatically detects and attempts to resolve disagreements or malformed outputs between debating personas, ensuring a coherent final answer.
*   **📊 Comprehensive CI/CD:** GitHub Actions workflows for linting (Ruff), formatting (Ruff), security scanning (Bandit, Safety), and unit/integration testing with coverage checks.
*   **✨ Interactive Streamlit UI:** A user-friendly web interface for defining prompts, configuring settings, uploading codebase context, and visualizing the debate process and results.

## 🚀 Getting Started

These instructions will guide you through setting up Project Chimera on your local machine.

### Prerequisites

*   **Python:** Version 3.11+
*   **Google Gemini API Key:** Obtain one from [Google AI Studio](https://aistudio.google.com/apikey).

### Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/tomwolfe/project_chimera.git
    cd project_chimera
    ```
2.  **Create and activate a virtual environment:**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```
3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
    *(Note: `requirements-prod.txt` is used for production deployments and includes fewer development-specific tools.)*

### Configuration

1.  **Set your Google Gemini API Key:**
    Provide your Gemini API Key in one of the following ways (prioritized in this order):
    *   **Environment Variable (Recommended):**
        ```bash
        export GEMINI_API_KEY="YOUR_GEMINI_API_KEY"
        ```
    *   **Streamlit Sidebar:** Enter it directly into the "Enter your Gemini API Key" text input in the application's sidebar.
    *   **Streamlit Secrets (for deployment):** If deploying to Streamlit Cloud, use `st.secrets["GEMINI_API_KEY"]`.

2.  **Review `config.yaml`:**
    This file contains global application settings such as `max_tokens_limit` (total token budget for a debate) and `context_token_budget_ratio`. Adjust these as needed.

### Running the Application

Once installed and configured, launch the Streamlit application:

```bash
streamlit run app.py
```

The application will typically open in your default web browser at `http://localhost:8080`.

## 📂 Project Structure

```
.
├── .github/                       # GitHub Actions workflows (CI/CD, self-analysis)
├── config/                        # Application settings, model registry, persona persistence
│   ├── custom_frameworks/         # User-saved custom persona frameworks
│   ├── model_registry.py          # LLM model specifications
│   ├── persistence.py             # Configuration loading/saving logic
│   └── settings.py                # Pydantic-based application settings
├── docs/                          # Project documentation and guidelines
│   ├── persona_alignment_guidelines.md
│   ├── project_chimera_context.md # Contextual information for AI self-analysis
│   ├── prompt_optimizations.md
│   ├── system_improvement_strategy.md
│   └── token_optimization_strategy.md
├── prompts/                       # Jinja2 templates for dynamic persona prompts
├── schemas/                       # JSON schemas for validating LLM outputs
│   └── analysis_schema.json       # Schema for self-improvement analysis
├── scripts/                       # Utility scripts
│   ├── fix_ruff.sh                # Script to auto-fix Ruff issues
│   └── run_analysis.py            # Script to trigger self-improvement analysis
├── src/                           # Main application source code
│   ├── config/                    # Configuration-related modules
│   ├── context/                   # Codebase scanning and context relevance analysis
│   ├── database/                  # Database operations (e.g., SQLite for user data)
│   ├── exceptions.py              # Custom exception classes
│   ├── llm_provider.py            # Interface for Google Gemini API
│   ├── llm_tokenizers/            # Tokenizer implementations
│   ├── models.py                  # Pydantic data models for structured data
│   ├── persona/                   # Persona definitions, routing logic
│   ├── persona_manager.py         # Manages persona configurations and performance
│   ├── resilience/                # Circuit breaker and rate limiter implementations
│   ├── self_improvement/          # Modules for metrics collection, content validation
│   ├── token_tracker.py           # Tracks token usage and costs
│   └── utils/                     # General utilities (parsers, validators, loggers, UI helpers, etc.)
├── app.py                         # The main Streamlit web application UI
├── core.py                        # The central Socratic Debate orchestration engine
├── Dockerfile                     # Docker container definition for deployment
├── LICENSE                        # MIT License
├── personas.yaml                  # Default configurations for all AI personas and frameworks
├── pyproject.toml                 # Project configuration for development tools (Ruff, Bandit, Pytest)
├── README.md                      # Project overview (this file)
├── requirements.txt               # Python dependencies for development
└── requirements-prod.txt          # Python dependencies for production deployments
```

## 💡 Usage

1.  **Enter API Key:** Provide your Google Gemini API Key in the sidebar. The system will validate its format and functionality.
2.  **Select Prompt:** Choose from example prompts categorized by domain (e.g., "Coding & Implementation", "Analysis & Problem Solving") or enter your own custom prompt.
3.  **Select Reasoning Framework:** Choose a domain-specific framework (e.g., "Software Engineering", "Self-Improvement") that best suits your task. You can also manage and save custom frameworks.
4.  **Provide Codebase Context (Optional):** For code-related tasks, you can upload relevant files. For "Self-Improvement" prompts, the system will automatically load its own codebase for analysis.
5.  **Run Socratic Debate:** Click the "🚀 Run Socratic Debate" button. The UI will display real-time progress and token usage.
6.  **Review Results:** After the debate, the "Results" section displays the final synthesized answer, proposed code changes (if applicable), and detailed intermediate reasoning steps. You can also download a comprehensive Markdown report.

## 🤖 Self-Improvement & CI/CD

Project Chimera is designed for continuous self-improvement and maintains high code quality through robust CI/CD pipelines.

*   **Automated Self-Analysis:** The `.github/workflows/analysis.yml` workflow can be manually triggered from GitHub or run on push/pull request. It executes `scripts/run_analysis.py` to perform a self-analysis of the Project Chimera codebase using the `Self_Improvement_Analyst` persona. The output is validated against `schemas/analysis_schema.json` and uploaded as an artifact.
*   **Continuous Integration:** The `.github/workflows/ci.yml` pipeline ensures code quality and security for every push and pull request by running:
    *   **Ruff:** Linting and formatting checks to enforce Python style guidelines.
    *   **Bandit:** Security vulnerability scanning to identify common security issues.
    *   **Safety:** Dependency vulnerability checks to ensure installed packages are secure.
    *   **Pytest:** Unit and integration tests with an 80% code coverage enforcement.
*   **Pre-commit Hooks:** The `.pre-commit-config.yaml` enforces code quality standards locally before commits, running tools like `ruff` and `bandit` to catch issues early.

## 🤝 Contributing

We welcome contributions to Project Chimera! To contribute:

1.  Fork the repository.
2.  Create a new branch for your feature or bug fix (`git checkout -b feature/your-feature-name`).
3.  Make your changes, ensuring they adhere to the project's coding standards.
4.  Run tests (`pytest`) and ensure code coverage remains above 80%.
5.  Install and run pre-commit hooks (`pre-commit install && pre-commit run --all-files`) to check for linting, formatting, and basic security issues.
6.  Commit your changes (`git commit -m 'feat: Add new feature'`).
7.  Push to your branch (`git push origin feature/your-feature-name`).
8.  Open a Pull Request, describing your changes and their impact.

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🌐 Connect With Us

*   **GitHub**: [tomwolfe/project_chimera](https://github.com/tomwolfe/project_chimera)
*   **Twitter**: [@Proj_Chimera](https://x.com/Proj_Chimera)
*   **Live Demo**: [Google Cloud Run](https://project-chimera-406972693661.us-central1.run.app/)

## 🙏 Acknowledgments

Project Chimera builds upon numerous open-source projects and frameworks. Special thanks to the Python, Streamlit, Bandit, Ruff, and GitHub Actions communities for their excellent tools and libraries.

---

> "The only true wisdom is in knowing you know nothing." - Socrates
> Project Chimera embodies this philosophy through continuous self-reflection and improvement.