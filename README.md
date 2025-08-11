# Project Chimera

[![GitHub](https://img.shields.io/badge/GitHub-Project_Chimera-000?style=flat-square&logo=github)](https://github.com/tomwolfe/project_chimera)
[![X (Twitter)](https://img.shields.io/badge/X-Proj_Chimera-1DA1F2?style=flat-square&logo=x)](https://x.com/Proj_Chimera)
[![Live Demo](https://img.shields.io/badge/Live%20Demo-Project%20Chimera-green?style=flat-square&logo=streamlit)](https://project-chimera-406972693661.us-central1.run.app)

Project Chimera is an advanced AI reasoning engine designed for complex problem-solving, code generation, and critical self-analysis. It employs a sophisticated **Socratic debate methodology**, where multiple specialized AI personas collaborate to refine solutions, identify potential flaws, and generate high-quality, robust code. Each persona contributes a unique viewpoint to iteratively improve a solution or piece of code, facilitating comprehensive analysis, risk assessment, and optimization.

Powered by Google's Gemini API, Chimera offers a unique approach to AI-assisted development and critical thinking.

## 🚀 Key Features

*   **Socratic Debate Engine:**
    *   Simulates a collaborative discussion among diverse AI personas (e.g., Visionary, Skeptic, Architect, Security Auditor, Test Engineer).
    *   Each persona contributes a unique viewpoint to iteratively refine solutions, identify potential flaws, and generate high-quality, robust code.
    *   Facilitates comprehensive analysis, risk assessment, and optimization.

*   **Domain-Specific Reasoning Frameworks:**
    *   Supports specialized reasoning tailored for **Science**, **Business**, **Creative** endeavors, and **Software Engineering**.
    *   Users can select a framework to guide the AI's persona selection and approach.

*   **Codebase Context Analysis:**
    *   For Software Engineering tasks, users can upload code files to provide rich context.
    *   Enables the AI to understand project structure, dependencies, and coding standards, leading to more relevant and consistent code generation and analysis.

*   **Self-Analysis & Improvement:**
    *   Chimera can critically analyze its own codebase.
    *   Identifies areas for enhancement in reasoning quality, robustness, efficiency, and developer maintainability, providing actionable code modification suggestions.

*   **Persona & Framework Management:**
    *   Load default personas and frameworks or create and save custom ones directly from the UI.
    *   Allows for highly tailored AI workflows by defining custom persona sets and reasoning strategies.

*   **Robustness & Quality Assurance:**
    *   Features dynamic token budget management and adaptive phase allocation.
    *   Includes built-in retry mechanisms for LLM calls and integrated code quality/security checks (PEP8, Bandit, AST-based security patterns).

*   **Actionable & Validated Output:**
    *   Generates structured outputs including commit messages, rationales, and specific code changes.
    *   Outputs are validated against predefined schemas (like `LLMOutput`) to ensure correctness and usability.

## 💡 Technology Stack

*   **AI Model:** Google Gemini API
*   **Web Framework:** Streamlit
*   **Core Libraries:** `google-genai`, `pydantic`, `sentence-transformers`, `rich`, `pyyaml`
*   **Code Analysis Tools:** `pycodestyle`, `bandit`, `ast` (for static analysis)

## 🌐 Live Demo

Experience Project Chimera in action:
[**Project Chimera Live Demo**](https://project-chimera-406972693661.us-central1.run.app)

## 🛠️ Getting Started

### Prerequisites

*   **Python:** Version 3.9 or higher.
*   **Gemini API Key:** Obtain an API key from [Google AI Studio](https://aistudio.google.com/apikey).

### Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/tomwolfe/project_chimera.git
    cd project_chimera
    ```

2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    # On Windows: venv\Scripts\activate
    # On macOS/Linux: source venv/bin/activate
    source venv/bin/activate
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Configure API Key:**
    *   You can enter your Gemini API key directly in the Streamlit app's sidebar.
    *   Alternatively, set it as an environment variable: `export GEMINI_API_KEY='YOUR_API_KEY'` (or `set GEMINI_API_KEY=YOUR_API_KEY` on Windows).

### Running the Application

Start the Streamlit application:
```bash
streamlit run app.py
```

## 📝 Usage

1.  **Enter API Key:** Provide your Gemini API key in the sidebar.
2.  **Select Prompt:** Choose an example prompt or enter your own custom prompt in the main area.
3.  **Select Framework:** Choose a reasoning framework (e.g., "General", "Software Engineering") from the dropdown to guide the AI's persona selection.
4.  **Provide Codebase Context (Optional):** If using the "Software Engineering" framework, upload relevant code files. This context helps the AI understand your project for more accurate analysis and code generation.
5.  **Edit Personas (Optional):** Access the "⚙️ View and Edit Personas" expander to customize persona behavior by adjusting their system prompts and parameters.
6.  **Run Debate:** Click the "🚀 Run Socratic Debate" button to start the AI's reasoning process.
7.  **View Results:** Examine the final synthesized answer, intermediate reasoning steps, process logs, and download comprehensive reports.

## 🗄️ Configuration

*   **`config.yaml`**: Defines domain keywords used for suggesting appropriate reasoning frameworks based on your prompt.
*   **`personas.yaml`**: Contains the default personas, their configurations, and predefined persona sets for different domains.
*   **`custom_frameworks/`**: This directory is automatically created to store any custom frameworks you save directly from the application.

## 🧠 Self-Analysis Feature

Project Chimera can analyze its own codebase to identify areas for improvement. To trigger this:

1.  Select the **"Software Engineering"** framework.
2.  Use a prompt that explicitly requests self-analysis, such as:
    *"Critically analyze the entire Project Chimera codebase. Identify the most impactful code changes for self-improvement, focusing on the 80/20 Pareto principle. Prioritize enhancements to reasoning quality, robustness, efficiency, and developer maintainability. For each suggestion, provide a clear rationale and a specific, actionable code modification."*

## 🐳 Docker

The project includes a `Dockerfile` for containerized deployment.

1.  **Build the Docker image:**
    ```bash
    docker build -t project-chimera .
    ```
2.  **Run the Docker container:**
    ```bash
    docker run -p 8080:8080 -e GEMINI_API_KEY=$GEMINI_API_KEY project-chimera
    ```
    Access the application at `http://localhost:8080`.

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🤝 Contributing

Contributions are welcome! Please refer to the [CONTRIBUTING.md](CONTRIBUTING.md) file (if available) for details on how to contribute, report issues, or suggest features.