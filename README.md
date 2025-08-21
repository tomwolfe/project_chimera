# Project Chimera: Socratic Self-Debate Engine

Project Chimera is an advanced, open-source reasoning engine designed for complex problem-solving, code generation, and self-improvement through a multi-agent Socratic debate framework. It leverages large language models (LLMs) to simulate a collaborative and critical thinking process, leading to more robust and well-reasoned solutions.

-   **Project Repository**: [https://github.com/tomwolfe/project_chimera](https://github.com/tomwolfe/project_chimera)
-   **Twitter**: [@Proj_Chimera](https://x.com/Proj_Chimera)

## Table of Contents

-   [Features](#features)
-   [Getting Started](#getting-started)
    -   [Prerequisites](#prerequisites)
    -   [Installation](#installation)
    -   [Configuration](#configuration)
    -   [Running the Application](#running-the-application)
    -   [Running with Docker](#running-with-docker)
-   [Usage](#usage)
    -   [Core LLM Settings](#core-llm-settings)
    -   [Project Setup & Input](#project-setup--input)
    -   [Reasoning Frameworks](#reasoning-frameworks)
    -   [Codebase Context](#codebase-context)
    -   [Persona Management](#persona-management)
    -   [Results & Analysis](#results--analysis)
-   [Project Structure](#project-structure)
-   [Contributing](#contributing)
-   [License](#license)
-   [Contact](#contact)

## Features

Project Chimera is built with a focus on advanced reasoning, robustness, and developer experience:

*   **Socratic Self-Debate Core**: Orchestrates a multi-turn debate among specialized AI personas to explore problems from various angles, identify flaws, and synthesize comprehensive solutions.
*   **Dynamic Persona Sequencing**: Intelligently selects and orders AI personas based on the user's prompt, chosen domain, and real-time context analysis, ensuring the most relevant experts contribute to the debate.
*   **Context-Aware Reasoning**: Utilizes semantic search and intelligent summarization to identify and incorporate relevant codebase files, providing the LLMs with precise, token-efficient context.
*   **Structured Output & Validation**: Employs Pydantic models to enforce strict JSON schema adherence for LLM outputs, ensuring predictable and machine-readable results.
*   **Code Generation & Validation**: When generating code, it performs automated checks for:
    *   **PEP8 Style Compliance**: Ensures code adheres to Python's style guide.
    *   **Static Security Analysis (Bandit)**: Identifies common security vulnerabilities.
    *   **Abstract Syntax Tree (AST) Security Checks**: Detects dangerous patterns like `eval()`, `exec()`, `os.system()`, unsafe `subprocess` calls, and insecure deserialization (e.g., `pickle.load`, `yaml.load` without safe loaders).
    *   **Unit Test Presence**: Suggests missing unit tests for new or modified Python files.
*   **Adaptive LLM Parameter Adjustment**: Dynamically fine-tunes persona-specific LLM parameters (e.g., temperature, max tokens) based on real-time performance metrics like schema adherence and truncation rates.
*   **Conflict Resolution Sub-Debate**: Introduces a "Devils_Advocate" persona to identify and report conflicts, triggering a focused sub-debate among involved personas to resolve disagreements.
*   **Self-Improvement Analysis**: Features a dedicated "Self_Improvement_Analyst" persona that collects objective metrics (code quality, security, debate efficiency, robustness) from the codebase and debate history to propose actionable improvements to Project Chimera itself.
*   **Robustness & Resilience**:
    *   **Rate Limiting**: Prevents abuse and manages API usage.
    *   **Circuit Breaker**: Protects against cascading failures by temporarily halting calls to consistently failing LLM services, with adaptive thresholds.
    *   **Structured Error Handling**: Catches and logs exceptions with detailed context, providing user-friendly, action-oriented error messages.
    *   **Prompt Sanitization**: Mitigates prompt injection and XSS risks in user inputs.
    *   **Secure Path Handling**: Validates and sanitizes file paths to prevent directory traversal and other file system vulnerabilities.
*   **Flexible Persona Management**: Allows users to view, edit, and reset individual persona parameters, and save/load custom reasoning frameworks for different problem domains.
*   **Structured Logging**: Implements JSON-formatted logging with request IDs for easier debugging, monitoring, and analysis in production environments.
*   **Streamlit Web UI**: Provides an intuitive and interactive web interface for configuring debates, providing context, and visualizing results.

## Getting Started

Follow these steps to get Project Chimera up and running on your local machine.

### Prerequisites

*   Python 3.10+
*   `pip` (Python package installer)
*   A Google Gemini API Key (obtainable from [Google AI Studio](https://aistudio.google.com/apikey))
*   (Optional) Docker for containerized deployment

### Installation

1.  **Clone the repository**:
    ```bash
    git clone https://github.com/tomwolfe/project_chimera.git
    cd project_chimera
    ```

2.  **Create and activate a virtual environment**:
    ```bash
    python -m venv venv
    # On Windows:
    .\venv\Scripts\activate
    # On macOS/Linux:
    source venv/bin/activate
    ```

3.  **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```
    For production environments, use `requirements-prod.txt`:
    ```bash
    pip install -r requirements-prod.txt
    ```

4.  **Install pre-commit hooks (recommended for development)**:
    ```bash
    pre-commit install
    ```

### Configuration

The main configuration for Project Chimera is handled via `config.yaml` and the Streamlit UI.

1.  **`config.yaml`**: This file defines global settings like domain keywords for prompt classification.
    ```yaml
    # config.yaml
    domain_keywords:
      Science: ["scientific", "research", ...]
      Business: ["business", "market", ...]
      Creative: ["creative", "art", ...]
      Software Engineering: ["code", "implement", ...]
    context_token_budget_ratio: 0.25
    ```
    You can adjust `context_token_budget_ratio` to control how much of the total token budget is allocated to context analysis.

2.  **Gemini API Key**: You will need to enter your Gemini API Key directly into the Streamlit sidebar when running the application. It is not stored persistently by the application.

### Running the Application

Once installed, you can run the Streamlit web application:

```bash
streamlit run app.py
```

This will open the application in your default web browser, usually at `http://localhost:8501`.

### Running with Docker

For a containerized deployment, you can use the provided `Dockerfile`:

1.  **Build the Docker image**:
    ```bash
    docker build -t project-chimera .
    ```
    This process includes a multi-stage build to pre-download the SentenceTransformer model, making the final image smaller and faster to deploy.

2.  **Run the Docker container**:
    ```bash
    docker run -p 8080:8080 -e GEMINI_API_KEY="YOUR_GEMINI_API_KEY" project-chimera
    ```
    Replace `"YOUR_GEMINI_API_KEY"` with your actual Gemini API key.
    The application will be accessible at `http://localhost:8080`.

## Usage

The Project Chimera web application provides a user-friendly interface to interact with the Socratic debate engine.

### Core LLM Settings (Sidebar)

*   **Gemini API Key**: Enter your API key here. The "Run" button will be disabled until a valid key is provided.
*   **Select LLM Model**: Choose between `gemini-2.5-flash-lite` (default, cost-effective), `gemini-2.5-flash`, or `gemini-2.5-pro` (more capable, potentially higher cost).
*   **Max Total Tokens Budget**: Set the maximum number of tokens the entire debate process can consume. Higher budgets allow for more extensive debates but increase cost.
*   **Show Intermediate Reasoning Steps**: Toggle to display detailed outputs from each persona's turn in the results section.
*   **Context Token Budget Ratio**: Adjust the percentage of the total token budget allocated specifically for context analysis (e.g., codebase summarization). The app provides smart defaults based on your prompt.

### Project Setup & Input

*   **What would you like to do?**:
    *   **Example Prompts**: Explore pre-defined prompts categorized by domain (e.g., "Coding & Implementation", "Analysis & Problem Solving"). Selecting an example will pre-fill the prompt and suggest a relevant reasoning framework.
    *   **Custom Prompt**: Enter your own detailed prompt. The system will attempt to recommend a suitable reasoning framework based on keywords in your custom prompt.

### Reasoning Frameworks

*   **Select Framework**: Choose a pre-defined persona set (e.g., "Software Engineering", "Science", "Business", "Creative", "Self-Improvement") or a custom framework you've saved. Each framework activates a specific set of AI personas tailored to that domain.
*   **Custom Framework Management**:
    *   **Save Current Framework**: Save the currently selected framework, including any persona edits you've made, as a new custom framework. This allows you to persist your customized persona configurations.
    *   **Load/Manage Frameworks**: Load previously saved custom frameworks.

### Codebase Context (Optional)

*   **Upload relevant files**: For "Software Engineering" tasks, you can upload code files (e.g., `.py`, `.js`, `.ts`, `.json`, `.yaml`) to provide context to the AI. The system will analyze these files to generate more accurate and consistent code changes.
*   **Demo Context**: For example prompts in the "Software Engineering" domain, a small demo codebase context is automatically loaded.

### Persona Management

*   **View and Edit Personas**: Expand this section to inspect and modify the `system_prompt`, `temperature`, and `max_tokens` for each persona within the currently selected framework.
    *   Changes are temporary unless saved as a custom framework.
    *   A "Reset All Personas for Current Framework" button allows you to revert all changes to the default configurations.
    *   Individual persona reset buttons are also available.

### Results & Analysis

After running the debate, the "Results" section will appear:

*   **Structured Summary**: Provides a concise commit message suggestion, a rationale for the solution, and details on token usage and estimated cost.
*   **Validation & Quality Report**: For "Software Engineering" tasks, this section provides a detailed report on the proposed code changes, including:
    *   **Malformed Output Detection**: Flags any parts of the LLM's output that did not conform to the expected JSON schema.
    *   **Code Quality Issues**: Reports PEP8 violations, Bandit security findings, and AST-based security vulnerabilities.
    *   **Missing Unit Tests**: Highlights if corresponding unit tests are not proposed for new or modified Python files.
*   **Proposed Code Changes**: Displays the suggested code modifications with a diff view for `MODIFY` actions and full content for `ADD` actions. You can download individual files.
*   **Final Synthesized Answer**: For non-coding domains, this section presents the final output from the debate.
*   **Download Analysis**: Download a complete Markdown report of the entire debate process, including configuration, process logs, intermediate steps, and the final answer.
*   **Show Intermediate Steps & Process Log**: Expand to view the detailed output from each persona's turn and the raw process log, which includes internal system messages and LLM interactions.

## Project Structure

```
.
├── .dockerignore             # Files to ignore when building Docker image
├── .gitignore                # Files to ignore in Git
├── .pre-commit-config.yaml   # Configuration for pre-commit hooks
├── app.py                    # Streamlit web application UI
├── core.py                   # Core Socratic Debate orchestration logic
├── data/
│   └── demo_codebase_context.json # Sample codebase for demo purposes
├── Dockerfile                # Docker build instructions
├── LICENSE                   # Project license (MIT)
├── llm_provider.py           # Interface for LLM API (GeminiProvider)
├── personas.yaml             # Default persona definitions and persona sets/frameworks
├── requirements-prod.txt     # Production dependencies
├── requirements.txt          # Development dependencies (includes linters, formatters)
└── src/
    ├── config/
    │   ├── persistence.py    # Handles saving/loading custom frameworks
    │   └── settings.py       # Centralized application settings (Pydantic model)
    ├── constants.py          # Global constants, self-analysis keywords, negation patterns
    ├── context/
    │   └── context_analyzer.py # Semantic context analysis, file relevance, summarization
    ├── exceptions.py         # Custom exception hierarchy for structured error handling
    ├── logging_config.py     # Structured JSON logging configuration
    ├── middleware/
    │   └── rate_limiter.py   # In-memory rate limiting for API calls
    ├── models.py             # Pydantic data models for LLM inputs/outputs, personas, etc.
    ├── persona/
    │   └── routing.py        # Dynamic persona selection and sequencing logic
    ├── persona_manager.py    # Manages persona configurations, sets, and adaptive adjustments
    ├── resilience/
    │   └── circuit_breaker.py # Circuit breaker implementation for LLM calls
    ├── self_improvement/
    │   └── metrics_collector.py # Collects metrics for self-improvement analysis
    ├── tokenizers/
    │   ├── __init__.py
    │   ├── base.py           # Abstract base class for tokenizers
    │   └── gemini_tokenizer.py # Gemini-specific tokenizer implementation
    └── utils/
        ├── __init__.py
        ├── code_validator.py # Code quality (PEP8), security (Bandit, AST), and test validation
        ├── domain_recommender.py # Recommends domain based on prompt keywords
        ├── error_handler.py  # Decorator for standardized error logging and wrapping
        └── output_parser.py  # Robust JSON extraction, repair, and Pydantic validation
```

## Contributing

We welcome contributions to Project Chimera! If you're interested in improving the debate engine, adding new personas, enhancing validation, or contributing to the UI, please feel free to:

1.  Fork the repository.
2.  Create a new branch (`git checkout -b feature/your-feature-name`).
3.  Make your changes.
4.  Ensure your code adheres to the project's style guidelines (run `pre-commit run --all-files`).
5.  Write clear, concise commit messages.
6.  Open a Pull Request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contact

For questions, feedback, or collaborations, you can reach out via:

*   **GitHub Issues**: [https://github.com/tomwolfe/project_chimera/issues](https://github.com/tomwolfe/project_chimera/issues)
*   **Twitter**: [@Proj_Chimera](https://x.com/Proj_Chimera)
