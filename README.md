# Project Chimera

**The Iterative Socratic Arbitration Loop (ISAL) for Deeper AI Reasoning.**

Project Chimera is a lightweight, powerful Python application that leverages a single Large Language Model (LLM) to adopt multiple reasoning personas. It orchestrates an **Iterative Socratic Arbitration Loop (ISAL)**, enabling complex problems to be explored from diverse angles, yielding more robust, nuanced, and well-reasoned answers than traditional single-query LLM approaches.

---

## ✨ Why Project Chimera?

In a world of increasingly capable LLMs, achieving truly insightful and reliable outputs for complex challenges remains a hurdle. Project Chimera addresses this by simulating a dynamic, multi-perspective debate among AI personas.

*   **Uncover Nuance & Depth:** Move beyond surface-level answers by engaging AI in structured critique and synthesis.
*   **Robust Problem Solving:** Leverage distinct AI viewpoints (Visionary, Skeptic, Critic, Arbitrator, Devil's Advocate, and domain-specific roles) to identify blind spots and refine ideas.
*   **Domain-Specific Intelligence:** Tailor AI reasoning to your specific field—be it Science, Business, Creative arts, or general problem-solving—with curated persona frameworks.
*   **Unprecedented Control & Transparency:** Customize AI reasoning workflows, view intermediate steps, track costs, and generate comprehensive reports to understand the AI's thought process.

---

## 🚀 Try It Live!

Experience the power of multi-persona AI reasoning firsthand.

➡️ **[Live Demo: Project Chimera](https://project-chimera-406972693661.us-central1.run.run.app)**

---

## 🌟 Key Features

*   **Multi-Persona Socratic Debate:** Employs distinct LLM personas (Visionary, Skeptic, Critic, Arbitrator, Devil's Advocate, plus domain-specific ones and a Generalist Assistant fallback) to generate, critique, and synthesize ideas iteratively.
*   **Intelligent Reasoning Frameworks:** Select from predefined sets of personas tailored for specific problem domains (e.g., General, Science, Business, Creative). Includes an LLM-powered recommendation for the best framework based on your prompt.
*   **In-UI Persona Configuration:** Temporarily adjust individual persona parameters (system prompt, temperature, max tokens) directly within the Streamlit web interface for session-specific experimentation.
*   **Community Frameworks:** Save and load custom persona configurations as reusable "Community Frameworks" within your session, facilitating easy sharing and reuse of debate setups.
*   **Cost & Token Tracking:** Monitors and displays total token usage and estimated USD cost for each debate session, including real-time updates and proactive warnings for the next step's estimated cost.
*   **Real-time Status Updates:** Provides live feedback on the debate progress, including current step, token usage, and estimated cost for the next step.
*   **Enhanced Robustness:** Implemented advanced error handling with retries, persona fallbacks (e.g., to `Generalist_Assistant`), and graceful degradation of token limits for a more stable and resilient user experience, especially during API interactions.
*   **Intermediate Step Visibility:** Option to view the detailed output of each persona's contribution to the debate for full transparency.
*   **Comprehensive Reporting:** Generates a full Markdown report of the debate, including prompt, configuration, process log, intermediate steps, and final answer, with options to download.
*   **Flexible Deployment:** Usable as a command-line tool or a Streamlit web application.
*   **User-Provided API Key:** Securely handles your Gemini API key, which is provided directly by the user at runtime and is not stored by the application.

---

## 🧠 Core Concept: The Iterative Socratic Arbitration Loop (ISAL)

The ISAL process involves a series of LLM calls where different personas interact to refine an initial prompt. Project Chimera now supports multiple "Reasoning Frameworks," which are curated sets of these personas, allowing the debate to be tailored to specific problem domains.

Typically, the core debate flow involves:
1.  **Visionary Generator:** Creates an initial, often bold, idea.
2.  **Skeptical Generator:** Critiques the idea from a risk-averse perspective.
3.  **Constructive Critic:** Provides actionable improvements based on the critique.
4.  **Impartial Arbitrator:** Synthesizes all inputs into a final, balanced answer.
5.  **Devil's Advocate:** Offers a final, sharp critique to expose any remaining fundamental flaws.

In addition to these core roles, domain-specific personas (e.g., `Scientific_Analyst`, `Business_Strategist`) and a `Generalist_Assistant` for fallbacks are used within selected frameworks to enhance the debate's relevance and robustness. This iterative process aims to produce more robust, well-reasoned, and comprehensive outputs than a single LLM query.

---

## 🛠️ Setup

1.  **Clone this repository:**
    ```bash
    git clone https://github.com/tomwolfe/project_chimera.git
    cd project_chimera
    ```
2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
3.  **Obtain a Gemini API Key:**
    Project Chimera uses the Google Gemini API. You'll need an API key from [Google AI Studio](https://aistudio.google.com/apikey).

    **Important:** Your API key is provided directly by you at runtime and is **not stored** by the application.

---

## 🚀 Usage

### Command-Line Interface (CLI)

The CLI tool `main.py` allows you to run a Socratic debate directly from your terminal.

1.  **Set your Gemini API Key:**
    It's recommended to set your Gemini API Key as an environment variable for convenience:
    ```bash
    export GEMINI_API_KEY="YOUR_GEMINI_API_KEY"
    ```
    Alternatively, you can pass it directly using the `--api-key` or `-k` option.

2.  **Run the Socratic Arbitration Loop:**
    ```bash
    python main.py reason "Your prompt goes here, e.g., Design a sustainable city for 1 million people on Mars."
    ```

3.  **CLI Options:**
    *   `--verbose` or `-v`: Show all intermediate reasoning steps.
        ```bash
        python main.py reason "Your prompt goes here." --verbose
        ```
    *   `--api-key` or `-k`: Provide your API key directly (overrides environment variable).
        ```bash
        python main.py reason "Your prompt goes here." --api-key YOUR_GEMINI_API_KEY
        ```
    *   `--max-tokens` or `-m`: Set a maximum total token budget for the entire process (default: 10000).
        ```bash
        python main.py reason "Your prompt goes here." --max-tokens 20000
        ```
    *   `--model` or `-M`: Specify the Gemini model to use (default: `gemini-2.5-flash-lite`).
        ```bash
        python main.py reason "Your prompt goes here." --model gemini-2.5-pro
        ```
    *   `--domain` or `-d`: Specify a reasoning domain to use (e.g., `General`, `Science`, `Business`, `Creative`). Defaults to `auto` which attempts to recommend one based on the prompt.
        ```bash
        python main.py reason "Your prompt goes here." --domain Science
        ```

    The CLI output provides real-time status updates, including current token usage, estimated cost, and proactive warnings about the next step's token consumption relative to the budget.

### Web Application (Streamlit)

The Streamlit web application provides an interactive UI for Project Chimera.

1.  **Run the Streamlit application:**
    ```bash
    streamlit run app.py
    ```
    Then open your web browser to the address provided by Streamlit (usually `http://localhost:8501`).

2.  **Using the Web UI:**
    *   Enter your Gemini API Key directly into the input field.
    *   Type your prompt or select one from the "Choose an example prompt" dropdown.
    *   Adjust the "Max Total Tokens Budget" to control the LLM's expenditure.
    *   **Reasoning Framework Selection:** The app will recommend a framework based on your prompt. You can apply the recommendation or select a different one from the dropdown (e.g., `General`, `Science`, `Business`, `Creative`, or `Custom`).
    *   Toggle "Show Intermediate Reasoning Steps" to see the full debate process.
    *   Select your preferred LLM Model (`gemini-2.5-flash-lite`, `gemini-2.5-pro`, `gemini-2.5-flash`).
    *   **Persona Configuration:** Expand the "View and Edit Personas" section to inspect or temporarily modify the parameters of the personas in the currently selected framework.
    *   **Community Frameworks:** In the "Contribute Your Framework" expander, you can save your current persona configuration (including any edits you've made) as a named "Community Framework" for later use in the session. You can also apply existing community frameworks.
    *   Click "Run Socratic Debate" to start the process.
    *   Monitor the "Process Log" and "Intermediate Reasoning Steps" (if enabled) for real-time updates and detailed output from each persona.
    *   Download the "Final Answer" or the "Full Report" in Markdown format.

---

## 🔧 Customizing Personas & Frameworks

Project Chimera's flexibility is powered by its `personas.yaml` configuration file and the in-UI customization options.

### `personas.yaml` Configuration

The personas and their groupings used in the Socratic debate are comprehensively defined in the `personas.yaml` file. This file allows you to:

*   **Define Individual Personas:** Each persona has a `name`, `system_prompt`, `temperature`, and `max_tokens`.
*   **Group Personas into Sets (Reasoning Frameworks):** The `persona_sets` section allows you to define different collections of personas. This enables the application to offer domain-specific reasoning frameworks (e.g., `Science`, `Business`, `Creative`) by activating a specific subset of personas for a given problem.

**Example (`personas.yaml`):**
```yaml
personas:
  # Core personas used across multiple domains
  - name: "Visionary_Generator"
    system_prompt: "You are a visionary futurist obsessed with innovation. Provide a bold, imaginative answer that explores uncharted possibilities. Ignore all practical constraints and risks."
    temperature: 0.9
    max_tokens: 2048

  - name: "Skeptical_Generator"
    system_prompt: "You are a meticulous, risk-averse pragmatist. Deliver a cautious, evidence-based answer. Your response MUST identify at least three potential failure points or critical vulnerabilities."
    temperature: 0.2
    max_tokens: 2048

  # ... (other personas) ...

persona_sets:
  General:
    - Visionary_Generator
    - Skeptical_Generator
    - Constructive_Critic
    - Impartial_Arbitrator
    - Generalist_Assistant
    - Devils_Advocate
  Science:
    - Scientific_Visionary
    - Scientific_Analyst
    - Constructive_Critic
    - Impartial_Arbitrator
    - Devils_Advocate
  # ... (other persona sets) ...
```
If you add new personas to `personas.yaml`, they will automatically be available for inclusion in `persona_sets`.

### In-UI Persona Configuration

For temporary, session-specific adjustments, you can modify persona parameters directly within the Streamlit web UI under the "View and Edit Personas" expander. Changes made here are temporary for the current session and do not modify the `personas.yaml` file on disk.

### Community Frameworks

The "Contribute Your Framework" section in the UI allows you to save your current persona configuration (including any UI edits) as a named "Community Framework." These saved frameworks appear in the "Community Frameworks" dropdown, enabling easy switching within the same session. To make a framework permanent or share it widely, you would manually add it to `personas.yaml` and potentially submit a pull request.

---

## 🤝 Contributing

We welcome contributions to Project Chimera! If you have ideas for new features, improvements, bug fixes, or new persona frameworks, please feel free to:

*   Open an issue to discuss your ideas or report bugs.
*   Fork the repository and submit a pull request with your changes.

Please ensure your contributions align with the project's goal of providing a lightweight, effective Socratic debate tool.

---

## ⚖️ License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🔗 Connect with Us

*   **GitHub:** [https://github.com/tomwolfe/project_chimera](https://github.com/tomwolfe/project_chimera)
*   **X (Twitter):** [https://x.com/Proj_Chimera](https://x.com/Proj_Chimera)
*   **Live Demo:** [https://project-chimera-406972693661.us-central1.run.app](https://project-chimera-406972693661.us-central1.run.app)