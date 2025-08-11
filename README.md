# Project Chimera: Socratic Self-Debate & Code Intelligence

Project Chimera is an advanced AI reasoning engine that simulates a Socratic self-debate process to tackle complex problems. It leverages a diverse set of AI personas, dynamic routing, and sophisticated codebase context analysis to generate comprehensive, well-reasoned, and actionable outputs. It particularly excels in software engineering tasks, offering robust code validation and self-analysis capabilities.

---

## 🚀 Project Status & Connect

[![Python Version](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?style=flat&logo=pre-commit)](https://pre-commit.com/)
[![Docker](https://img.shields.io/badge/docker-enabled-0db7ed?style=flat&logo=docker)](https://www.docker.com/)
[![GitHub](https://img.shields.io/badge/GitHub-Project_Chimera-000?style=flat-square&logo=github)](https://github.com/tomwolfe/project_chimera)
[![X (Twitter)](https://img.shields.io/badge/X-Proj_Chimera-1DA1F2?style=flat-square&logo=x)](https://x.com/Proj_Chimera)
[![Live Demo](https://img.shields.io/badge/Live%20Demo-Project%20Chimera-green?style=flat-square&logo=streamlit)](https://project-chimera-406972693661.us-central1.run.app)

---

## ✨ Key Features

*   **Socratic Debate Engine:** Simulates a structured debate among AI personas to explore multifaceted aspects of a problem, identify potential issues, and synthesize optimal solutions.
*   **Persona Management:** Utilizes a diverse set of AI personas (e.g., Visionary, Skeptic, Architect, Security Auditor, Test Engineer, DevOps Engineer) with configurable system prompts, descriptions, temperatures, and token limits.
*   **Dynamic Persona Routing:** Intelligently selects and sequences personas based on prompt analysis, domain keywords, and intermediate reasoning results to tailor the debate flow.
*   **Self-Analysis Prompt Detection:** Automatically identifies prompts requesting analysis or improvement of codebases (including Project Chimera itself) and activates specialized persona sequences.
*   **Codebase Context Analysis:** For software engineering prompts, it analyzes uploaded code files to understand project structure, identify key modules, and provide relevant code snippets for context-aware generation and analysis.
*   **Robust Code Validation Suite:** Integrates static analysis tools (`pycodestyle`, `bandit`) and AST-based checks to validate generated code for PEP8 compliance, security vulnerabilities, and common anti-patterns.
*   **Customizable Reasoning Frameworks:** Allows users to define, save, and load custom reasoning frameworks, grouping specific personas and configurations for tailored problem-solving workflows.
*   **Token Budget Management:** Dynamically allocates token budgets across different phases of the reasoning process (context analysis, debate, synthesis) based on prompt complexity and available resources, ensuring efficient operation.
*   **Interactive Streamlit UI:** Provides a user-friendly web interface for inputting prompts, managing configurations, running debates, and viewing results.
*   **Robust Error Handling:** Implements comprehensive error handling and retry mechanisms for LLM interactions and internal processes.

---

## 🚀 Getting Started

### 1. Installation

1.  **Clone the Repository:**
    ```bash
    git clone https://github.com/tomwolfe/project_chimera.git
    cd project_chimera
    ```

2.  **Set up a Python Environment:**
    It's recommended to use a virtual environment:
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Obtain a Gemini API Key:**
    You'll need an API key from Google AI Studio: [https://aistudio.google.com/app/apikey](https://aistudio.google.com/app/apikey)

5.  **Configure API Key:**
    *   **Recommended (Secure & Convenient):** Set the `GEMINI_API_KEY` environment variable before running the app:
        ```bash
        export GEMINI_API_KEY='YOUR_API_KEY'
        ```
    *   **Alternatively:** Enter your API key directly into the "API Key" field in the Streamlit sidebar when running the application.

### 2. Run the Application

```bash
streamlit run app.py
```

### 3. Usage via Web UI

*   **API Key:** Enter your Gemini API key (if not set as an environment variable).
*   **Prompt:** Select an example prompt or enter your own.
*   **Framework:** Choose a reasoning framework (e.g., "General", "Science", "Business", "Creative", "Software Engineering").
*   **Codebase Context (for Software Engineering):** Upload relevant code files to provide context for AI analysis and generation.
*   **Persona Configuration:** Expand "View and Edit Personas" to review and modify persona settings for the selected framework.
*   **Run Debate:** Click "🚀 Run Socratic Debate".
*   **View Results:** Monitor progress, intermediate steps, logs, and the final synthesized answer.
*   **Download Reports:** Save full reports or final answer JSON using provided buttons.

---

## 🔧 Configuration Files

*   **`config.yaml`**: Configures domain keywords for prompt analysis and default token budget ratios.
*   **`personas.yaml`**: Defines core AI personas, their system prompts, parameters, and organizes them into domain-specific persona sets (e.g., `Software Engineering`, `Science`).
*   **Custom Frameworks**: User-defined frameworks are saved as JSON files in the `custom_frameworks/` directory, allowing for personalized reasoning workflows.

---

## 📦 Development

*   **Pre-commit Hooks:** Ensures code quality and formatting. Install with `pip install pre-commit` and run `pre-commit install`.
*   **Docker Support:** Includes a `Dockerfile` for easy containerization and deployment.

---

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.