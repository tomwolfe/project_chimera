# Project Chimera 🐉: Socratic Self-Debate Engine

> **"An advanced reasoning engine for complex problem-solving and code generation. This project's core software is open-source and available on [GitHub](https://github.com/tomwolfe/project_chimera)."**

Project Chimera is an innovative AI system that uses Socratic self-debate methodology to critically analyze and improve its own codebase. Unlike traditional software, Chimera embodies continuous self-reflection and improvement, creating a truly adaptive reasoning engine that evolves through internal dialogue between specialized AI personas.

## 🌟 Key Features

*   **Socratic Self-Debate Framework**: Multiple AI personas engage in structured debate to analyze and improve the system.
*   **Self-Improvement Capability**: The system identifies and implements its own enhancements following the 80/20 Pareto principle.
*   **Multi-Persona Architecture**: Specialized AI roles including:
    *   **Self_Improvement_Analyst**: Prioritizes reasoning quality, robustness, efficiency.
    *   **Code_Architect**: Focuses on structural improvements and maintainability.
    *   **Security_Auditor**: Identifies and mitigates vulnerabilities.
    *   **DevOps_Engineer**: Optimizes deployment and operational efficiency.
    *   **Test_Engineer**: Enhances testing strategies and coverage.
    *   **Constructive_Critic**: Provides balanced critique on logic and best practices.
    *   **Devils_Advocate**: Challenges assumptions and identifies risks.
    *   **Impartial_Arbitrator**: Mediates debates and synthesizes consensus.
    *   **Context_Aware_Assistant**: Provides codebase context analysis.
    *   **General_Synthesizer**: Integrates diverse perspectives into a final output.
    *   **Visionary_Generator**: Proposes innovative solutions.
    *   **Skeptical_Generator**: Identifies flaws and limitations.
*   **Schema-Validated Outputs**: Strict JSON schema adherence ensures reliable reasoning quality and data integrity.
*   **Token Optimization**: Advanced tracking and minimization of token usage for cost-efficiency.
*   **Comprehensive Validation**: Integrated Ruff (linting/formatting), Bandit (security scanning), AST checks, and pytest for code quality and security.
*   **Robust Error Handling & Resilience**: Incorporates circuit breakers, rate limiting, and comprehensive error handling.
*   **Streamlit UI**: Provides an interactive web interface for easy interaction, configuration, and visualization of results.
*   **Configuration Management**: Flexible configuration via `config.yaml`, custom persona sets, and dynamic framework loading/saving.
*   **CI/CD & Development Workflow**: Includes GitHub Actions for automated testing, validation, and deployment, along with pre-commit hooks.

## 🏗️ Architecture Overview

Project Chimera follows a multi-layered architecture designed for self-reflection and improvement:

```
├── app.py                   # Main Streamlit application file
├── core.py                  # Core Socratic debate engine logic
├── src/                     # Main application logic
│   ├── config/              # Configuration settings and persistence
│   ├── context/             # Codebase context analysis modules
│   ├── llm_provider.py      # LLM interface (Gemini)
│   ├── models.py            # Pydantic models for structured data
│   ├── persona/             # Persona routing and management
│   ├── resilience/          # Circuit breaker, rate limiter
│   ├── self_improvement/    # Self-improvement specific modules
│   ├── tokenizers/          # Token counting implementations
│   └── utils/               # Various utility functions (prompt engineering, validation, etc.)
├── tests/                   # Unit and integration tests
├── docs/                    # Project documentation
├── .github/workflows/       # CI/CD pipelines
├── pyproject.toml           # Project configuration (Ruff, Bandit, Pytest)
├── requirements.txt         # Python dependencies
└── ...                      # Other configuration files (e.g., personas.yaml)
```

## ⚙️ Setup Instructions

### Prerequisites

*   Python 3.11+
*   Google Generative AI API Key

### Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/tomwolfe/project_chimera.git
    cd project_chimera
    ```

2.  **Create and activate a virtual environment:**
    ```bash
    python -m venv venv
    source venv/bin/activate  # Linux/MacOS
    # OR
    venv\Scripts\activate    # Windows
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Set up environment variables:**
    ```bash
    export GEMINI_API_KEY='YOUR_API_KEY'
    ```
    (Alternatively, use a `.env` file and `python-dotenv`).

### Running the Application

Launch the Streamlit application:
```bash
streamlit run app.py
```

## 💡 Self-Improvement Methodology

Project Chimera employs a structured approach to self-improvement, prioritizing high-impact changes based on the 80/20 Pareto principle:

### 1. **Enhanced Observability & Instrumentation**
   - Collects comprehensive system metrics: reasoning quality, robustness, efficiency (token usage), maintainability, security, and test coverage.
   - Tracks token usage for cost optimization.
   - Assesses persona performance and reasoning quality through detailed metrics.

### 2. **Multi-Persona Analysis Framework**
   - Specialized AI personas analyze different aspects of the system:
     - **Self_Improvement_Analyst**: Identifies top 1-3 high-impact improvements.
     - **Code_Architect**: Evaluates structure, scalability, and maintainability.
     - **Security_Auditor**: Focuses on vulnerabilities and threat models.
     - **DevOps_Engineer**: Optimizes deployment, CI/CD, and operations.
     - **Test_Engineer**: Enhances testing strategies and coverage.
     - **Constructive_Critic**: Provides balanced critique on logic and best practices.
     - **Devils_Advocate**: Challenges assumptions and identifies risks.
     - **Impartial_Arbitrator**: Mediates debates and synthesizes consensus.
     - **Context_Aware_Assistant**: Analyzes codebase context relevant to the prompt.
     - **General_Synthesizer**: Integrates diverse perspectives into a final output.
     - **Visionary_Generator**: Proposes innovative solutions.
     - **Skeptical_Generator**: Identifies flaws and limitations.
   - Utilizes a Socratic debate process with conflict resolution mechanisms.

### 3. **Schema-Validated Output Process**
   - Strict adherence to JSON schemas for reliable outputs.
   - Malformed block tracking for continuous improvement of LLM outputs.
   - Consolidation of code changes affecting the same file.

### 4. **Implementation & Validation**
   - Proposed code changes follow a defined JSON schema (`FILE_PATH`, `ACTION`, `FULL_CONTENT`/`DIFF_CONTENT`/`LINES`).
   - Integrates Ruff (linting/formatting), Bandit (security scanning), AST checks, and pytest for code quality and security validation.
   - Includes automated testing via `pytest`.
   - Tracks improvement history in `data/improvement_history.jsonl`.

## 🧪 Validation Tools

Project Chimera integrates industry-standard tools for code quality and security:

*   **Ruff**: Linting and formatting (replaces Flake8, Black).
*   **Bandit**: Security vulnerability scanning.
*   **pytest**: Testing framework.
*   **GitHub Actions**: CI/CD pipeline for automated testing, linting, security scans, and deployment.

## 🤝 Contributing

We welcome contributions! Please follow these steps:

1.  Fork the repository.
2.  Create a new feature branch (`git checkout -b feature/YourFeature`).
3.  Commit your changes (`git commit -m 'feat: Add AmazingFeature'`).
4.  Push to the branch (`git push origin feature/YourFeature`).
5.  Open a Pull Request.

When contributing, focus on high-impact changes, provide clear rationale, include specific code modifications, and adhere to the project's schemas and validation standards.

## 📄 License

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