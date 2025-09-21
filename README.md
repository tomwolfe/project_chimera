# Project Chimera

An AI System for Socratic Self-Debate and Continuous Improvement

## ✨ Overview

Project Chimera is an advanced AI system designed to tackle complex problems through a unique Socratic self-debate methodology. It orchestrates a dynamic team of specialized AI personas, each contributing a distinct perspective to analyze, critique, and refine solutions. With a focus on continuous self-improvement, Chimera can analyze its own codebase, identify areas for enhancement in reasoning, robustness, efficiency, and maintainability, and even propose actionable code modifications. The system features an interactive Streamlit interface, robust integration with the Google Gemini API, and intelligent context-awareness through codebase scanning and semantic search.

## 🚀 Key Features

*   **Socratic Debate Engine**: Orchestrates multi-turn debates between specialized AI personas to explore problems from diverse angles, fostering comprehensive analysis and solution refinement.
*   **Dynamic Multi-Persona Architecture**: Leverages a configurable set of AI personas (e.g., Code Architect, Security Auditor, Devil's Advocate) whose sequence is dynamically routed based on prompt analysis, domain, and intermediate debate results.
*   **Context-Aware Reasoning (RAG)**: Integrates Retrieval-Augmented Generation (RAG) by scanning provided codebases, computing semantic embeddings, and retrieving the most relevant files and snippets to inform the debate, especially for code-related tasks.
*   **Self-Improvement Capabilities**: Enables the AI to critically analyze its own codebase, identify impactful improvements (following the 80/20 principle), and suggest concrete code changes for enhanced reasoning quality, robustness, efficiency, and maintainability.
*   **Interactive Streamlit UI**: Provides an intuitive web interface for users to input prompts, configure debate parameters, manage personas, and visualize the debate process and final results.
*   **Robust LLM Interaction**: Features resilient integration with the Google Gemini API, incorporating retry mechanisms, rate limiting, and a circuit breaker pattern for reliable communication.
*   **Structured Output & Validation**: Ensures AI outputs adhere to predefined Pydantic schemas, with advanced parsing and repair heuristics to handle malformed responses and ensure data integrity.
*   **Code Quality & Security Checks**: Integrates static analysis tools (Ruff, Bandit, AST-based checks) to validate generated code and identify potential issues, ensuring high-quality outputs.
*   **Token Optimization & Cost Tracking**: Intelligently manages token usage across the debate, optimizes prompts to stay within budget, and tracks estimated API costs.
*   **Customizable Reasoning Frameworks**: Allows users to save, load, import, and export custom persona configurations and debate frameworks, enabling tailored AI behaviors.

## 🛠️ Technologies Used

*   **Python**: Core programming language.
*   **Streamlit**: For the interactive web application.
*   **Google Gemini API**: Large Language Model provider.
*   **Pydantic**: For data validation and settings management.
*   **Sentence Transformers**: For semantic search and context embedding.
*   **Hugging Face Transformers**: For summarization capabilities.
*   **Ruff**: High-performance Python linter and formatter.
*   **Bandit**: Security linter for Python code.
*   **Jinja2**: For dynamic prompt templating.
*   **GitHub Actions**: For CI/CD, self-analysis, and quality checks.
*   **Docker**: For containerization and deployment.

## ⚙️ Installation

To set up Project Chimera locally, follow these steps:

1.  **Clone the repository**:
    ```bash
    git clone https://github.com/tomwolfe/project_chimera.git
    cd project_chimera
    ```

2.  **Create and activate a virtual environment**:
    ```bash
    python -m venv venv
    # On Windows:
    # .\venv\Scripts\activate
    # On macOS/Linux:
    source venv/bin/activate
    ```

3.  **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

4.  **Set up your Google Gemini API Key**:
    Project Chimera requires a Google Gemini API key. You can obtain one from [Google AI Studio](https://aistudio.google.com/apikey).
    Store your API key in one of the following ways (prioritized in order):
    *   **Environment Variable**: Create a `.env` file in the project root and add:
        ```dotenv
        GEMINI_API_KEY="YOUR_GEMINI_API_KEY"
        ```
    *   **Streamlit Secrets**: If deploying to Streamlit Cloud, use Streamlit's secrets management.
    *   **Direct Input**: You can also enter the key directly into the sidebar of the Streamlit application (not recommended for production).

5.  **Install pre-commit hooks (recommended)**:
    To ensure code quality and consistency, install the pre-commit hooks:
    ```bash
    pre-commit install
    ```
    This will automatically run linters, formatters, and security checks before each commit.

## 💡 Usage

1.  **Run the Streamlit application**:
    ```bash
    streamlit run app.py
    ```
    This will open the application in your web browser, typically at `http://localhost:8080`.

2.  **Interact with the UI**:
    *   **Sidebar Configuration**: Enter your Gemini API Key, select the LLM model (`gemini-2.5-flash-lite` is recommended for most tasks), and adjust token budgets.
    *   **Prompt Input**: Choose from various example prompts categorized by domain (e.g., "Coding & Implementation", "Analysis & Problem Solving") or enter your own custom prompt.
    *   **Reasoning Framework**: Select a domain-specific reasoning framework (e.g., "Software Engineering", "Self-Improvement") to guide the AI personas.
    *   **Codebase Context (Optional)**: For code-related tasks or self-analysis, upload relevant files or let the system scan its own codebase.
    *   **Run Socratic Debate**: Click the "🚀 Run Socratic Debate" button to initiate the AI's reasoning process.
    *   **Review Results**: The main area will display the AI's final synthesized answer, along with intermediate reasoning steps and a detailed process log. You can also download a comprehensive Markdown report.

3.  **Self-Improvement Analysis**:
    To have Project Chimera analyze its own codebase, select the "Self-Improvement" domain and choose the example prompt: "Critically analyze the entire Project Chimera codebase...". The system will automatically scan its internal files and provide actionable suggestions for improvement.

## 📂 Project Structure

*   `app.py`: The main Streamlit web application interface.
*   `core.py`: The central Socratic Debate orchestration engine.
*   `src/`: Contains the core application logic.
    *   `config/`: Application settings, model registry, and configuration persistence.
    *   `context/`: Codebase scanning and context relevance analysis.
    *   `database/`: Database operations (e.g., for user data, session management).
    *   `exceptions.py`: Custom exception classes.
    *   `llm_provider.py`: Interface for Google Gemini API interactions.
    *   `llm_tokenizers/`: Tokenizer implementations for different LLMs.
    *   `models.py`: Pydantic data models for structured data validation.
    *   `persona/`: Persona definitions and dynamic routing logic.
    *   `persona_manager.py`: Manages persona configurations, sets, and performance metrics.
    *   `rag_system.py`: Retrieval-Augmented Generation components.
    *   `resilience/`: Circuit breaker and rate limiter implementations.
    *   `self_improvement/`: Modules for metrics collection, content validation, and improvement application.
    *   `token_tracker.py`: Tracks token usage and costs.
    *   `utils/`: General utility functions (error handling, JSON utilities, path utilities, UI helpers, etc.).
*   `docs/`: Project documentation, including architecture, contributing guidelines, and improvement strategies.
*   `prompts/`: Jinja2 templates for dynamic persona prompts.
*   `schemas/`: JSON schemas for validating LLM outputs.
*   `scripts/`: Utility scripts for quality checks, token reports, etc.
*   `tests/`: Unit and integration tests.
*   `.github/workflows/`: GitHub Actions CI/CD and self-analysis workflows.
*   `.pre-commit-config.yaml`: Configuration for pre-commit hooks.
*   `.ruff.toml`: Ruff linter and formatter configuration.
*   `pyproject.toml`: Project configuration, including pytest and tool settings.
*   `requirements.txt`: Development Python dependencies.
*   `requirements-prod.txt`: Production Python dependencies.
*   `Dockerfile`: Docker container definition for deployment.

## 🤝 Contributing

We welcome contributions to Project Chimera! Please refer to our [CONTRIBUTING.md](docs/CONTRIBUTING.md) for detailed guidelines on how to get started, code standards, branching strategy, and pull request process.

## 📄 License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## 🏛️ Architecture

For a detailed overview of the system's design and component interactions, please refer to the [ARCHITECTURE.md](docs/ARCHITECTURE.md) document.

## 🧠 Self-Improvement Strategy

Our approach to continuous self-improvement is outlined in [system_improvement_strategy.md](docs/system_improvement_strategy.md).

## 🌐 Connect With Us

*   **GitHub**: [tomwolfe/project_chimera](https://github.com/tomwolfe/project_chimera)
*   **Twitter**: [@Proj_Chimera](https://x.com/Proj_Chimera)
*   **Live Demo**: [Google Cloud Run](https://project-chimera-406972693661.us-central1.run.app/)

## 🙏 Acknowledgments

Project Chimera builds upon numerous open-source projects and frameworks. Special thanks to the Python, Streamlit, Bandit, Ruff, and GitHub Actions communities for their excellent tools and libraries.

---

> "The only true wisdom is in knowing you know nothing." - Socrates
> Project Chimera embodies this philosophy through continuous self-reflection and improvement.