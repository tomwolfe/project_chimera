# Project Chimera

Project Chimera is an innovative AI system that leverages a Socratic self-debate methodology to solve complex problems, generate code, and continuously improve its own capabilities. It features a dynamic multi-persona architecture, context-aware reasoning, and a Streamlit-based interactive interface.

## ✨ Features

*   **Socratic Debate Engine**: A central orchestrator (`core.py`) that manages a multi-turn debate among specialized AI personas to arrive at robust solutions.
*   **Dynamic Multi-Persona Architecture**: Utilizes a `PersonaManager` to dynamically select and configure AI personas based on the problem domain, prompt complexity, and historical performance.
*   **Context-Aware Reasoning (RAG)**: Integrates a Retrieval-Augmented Generation (RAG) system with a `CodebaseScanner` and `ContextRelevanceAnalyzer` to provide relevant code and documentation context to personas, enhancing their problem-solving capabilities.
*   **Self-Improvement Capabilities**: Features a dedicated `Self_Improvement_Analyst` persona and a suite of tools (`FocusedMetricsCollector`, `CritiqueEngine`, `ImprovementApplicator`) that allow the AI to analyze its own codebase, debate process, and performance metrics to suggest and apply improvements.
*   **Robust LLM Interaction**: Employs `GeminiProvider` with built-in retry mechanisms (`tenacity`), rate limiting, and a circuit breaker pattern to ensure resilient and fault-tolerant communication with the Google Gemini API.
*   **Structured Output & Validation**: Leverages Pydantic models (`src/models.py`) and JSON schemas (`schemas/`) to enforce strict output formats for LLM responses, ensuring consistency and reliability.
*   **Interactive Streamlit UI**: Provides a user-friendly web interface (`app.py`) for configuring debate parameters, inputting prompts, viewing real-time progress, and reviewing structured results.
*   **Prompt Optimization**: Includes a `PromptOptimizer` to dynamically adjust and truncate prompts based on token limits, persona performance, and overall token budget, improving efficiency and reducing costs.
*   **Conflict Resolution**: A `ConflictResolutionManager` mediates disagreements or malformed outputs from personas, attempting automated self-correction or synthesizing coherent responses.
*   **Code Quality & Security Checks**: Integrates `Ruff` (linting, formatting), `Bandit` (security scanning), and AST-based checks (`src/utils/validation/code_validator.py`) to validate generated code and analyze the project's own codebase.
*   **Containerization**: A `Dockerfile` is provided for easy deployment and consistent environments.
*   **Continuous Integration**: GitHub Actions workflows (`.github/workflows/ci.yml`, `analysis.yml`) automate testing, code quality checks, and trigger self-improvement analysis.

## 🚀 Getting Started

Follow these steps to get Project Chimera up and running on your local machine.

### Prerequisites

*   Python 3.11+
*   Git
*   A Google Gemini API Key (obtainable from [Google AI Studio](https://aistudio.google.com/apikey))

### Installation

1.  **Clone the repository**:
    ```bash
    git clone https://github.com/tomwolfe/project_chimera.git
    cd project_chimera
    ```

2.  **Create and activate a virtual environment**:
    ```bash
    python -m venv venv
    source venv/bin/activate # On Windows: .\venv\Scripts\activate
    ```

3.  **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

4.  **Install pre-commit hooks**:
    ```bash
    pre-commit install
    ```
    This will ensure code quality and formatting checks run automatically before each commit.

### Configuration

1.  **Set your Gemini API Key**:
    Create a `.env` file in the root of the project and add your API key:
    ```
    GEMINI_API_KEY="YOUR_GEMINI_API_KEY"
    ```
    Alternatively, you can enter your API key directly in the Streamlit sidebar when the application is running.

2.  **Adjust advanced settings (Optional)**:
    Modify `config.yaml` for advanced configurations such as token budgets, persona-specific settings, or domain keywords.

### Running the Application

To start the interactive Streamlit application:

```bash
streamlit run app.py
```

Open your web browser and navigate to `http://localhost:8080` (or the address displayed in your terminal).

### Running Self-Analysis (CLI)

You can manually trigger a self-analysis of the Project Chimera codebase via a script:

```bash
python scripts/run_analysis.py > analysis_output.json
```

This script will output a structured JSON analysis to `analysis_output.json`. This process is also automated via GitHub Actions.

## 📂 Project Structure

The project is organized into the following main directories:

```
.
├── .github/                      # GitHub Actions workflows (CI, self-analysis)
├── config/                       # Application settings, model registry, persistence
├── docs/                         # Project documentation and guidelines
├── prompts/                      # Jinja2 templates for dynamic persona prompts
├── schemas/                      # JSON schemas for validating LLM outputs
├── scripts/                      # Utility scripts (e.g., run_analysis.py, run_quality_checks.sh)
├── src/                          # Main application source code
│   ├── config/                   # Configuration-related modules
│   ├── constants.py              # Global constants
│   ├── context/                  # Codebase scanning, context relevance analysis (RAG)
│   ├── database/                 # Database operations
│   ├── exceptions.py             # Custom exception classes
│   ├── llm_provider.py           # Interface for Google Gemini API
│   ├── llm_tokenizers/           # Tokenizer implementations
│   ├── models.py                 # Pydantic data models for structured data
│   ├── persona/                  # Persona definitions, routing logic
│   ├── persona_manager.py        # Manages persona configurations and performance
│   ├── resilience/               # Circuit breaker and rate limiter implementations
│   ├── self_improvement/         # Modules for metrics collection, critique, improvement application
│   ├── services/                 # General service functions
│   ├── token_tracker.py          # Tracks token usage and costs
│   └── utils/                    # General utility functions (parsers, validators, loggers, UI helpers, etc.)
├── tests/                        # Unit and integration tests
├── app.py                        # Main Streamlit web application UI
├── core.py                       # The central Socratic Debate orchestration engine
├── personas.yaml                 # Centralized definitions and configurations for all AI personas
├── pyproject.toml                # Project configuration for tools like Ruff, Bandit, and pytest
├── requirements.txt              # Development Python dependencies
├── requirements-prod.txt         # Production Python dependencies
├── Dockerfile                    # Docker container definition for deployment
├── README.md                     # Project overview (this file)
└── LICENSE                       # Project's license file
```

For a more detailed overview of the codebase, refer to `docs/project_chimera_context.md`.

## 🤝 Contributing

Contributions are welcome! Please see `docs/CONTRIBUTING.md` for guidelines on how to contribute, including code standards, branching strategy, and pull request process.

## 📄 License

This project is licensed under the MIT License - see the `LICENSE` file for details.

## 🌐 Connect With Us

*   **GitHub**: [tomwolfe/project_chimera](https://github.com/tomwolfe/project_chimera)
*   **Twitter**: [@Proj_Chimera](https://x.com/Proj_Chimera)
*   **Live Demo**: [Google Cloud Run](https://project-chimera-406972693661.us-central1.run.app/)

## 🙏 Acknowledgments

Project Chimera builds upon numerous open-source projects and frameworks. Special thanks to the Python, Streamlit, Bandit, Ruff, and GitHub Actions communities for their excellent tools and libraries.

---

> "The only true wisdom is in knowing you know nothing." - Socrates
> Project Chimera embodies this philosophy through continuous self-reflection and improvement.
