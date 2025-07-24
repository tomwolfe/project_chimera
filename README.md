# Project Chimera

A lightweight Python application that runs an **Iterative Socratic Arbitration Loop (ISAL)** using a single LLM adopting different reasoning personas. It's designed to explore complex problems from multiple angles, providing a more robust and nuanced answer than a single LLM query.

## Features

-   **Multi-Persona Socratic Debate:** Employs distinct LLM personas (Visionary, Skeptic, Critic, Arbitrator, Devil's Advocate) to generate, critique, and synthesize ideas.
-   **Configurable Debate Flow:** Personas and their roles are defined in a `personas.yaml` file, allowing for easy customization of the debate participants.
-   **Cost & Token Tracking:** Monitors and displays total token usage and estimated USD cost for each debate session, helping users manage their LLM budget.
-   **Real-time Status Updates:** Provides live feedback on the debate progress, including current step, token usage, and estimated cost for the next step.
-   **Enhanced Robustness:** Improved error handling and graceful recovery for a more stable user experience, especially during API interactions.
-   **Intermediate Step Visibility:** Option to view the detailed output of each persona's contribution to the debate.
-   **Comprehensive Reporting:** Generates a full Markdown report of the debate, including prompt, configuration, process log, intermediate steps, and final answer.
-   **Flexible Deployment:** Usable as a command-line tool or a Streamlit web application.
-   **User-Provided API Key:** Securely handles your Gemini API key, which is provided directly by the user and not stored.

## Core Concept: The Iterative Socratic Arbitration Loop (ISAL)

## Setup

1.  Clone this repository:
    ```bash
    git clone https://github.com/your-username/project-chimera.git
    cd project_chimera
    ```
2.  Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```
3.  **Obtain a Gemini API Key:**
    Project Chimera uses the Google Gemini API. You'll need an API key from [Google AI Studio](https://aistudio.google.com/apikey).

    **Important:** Your API key is provided directly by you at runtime and is **not stored** by the application.

## Usage (CLI)

The CLI tool `main.py` allows you to run a Socratic debate directly from your terminal.

1.  **Set your Gemini API Key:**
    It's recommended to set your Gemini API Key as an environment variable for convenience:
    ```bash
    export GEMINI_API_KEY="YOUR_GEMINI_API_KEY"
    ```
    Alternatively, you can pass it directly using the `--api-key` or `-k` option.

2.  **Run the Socratic Arbitration Loop:**
    ```bash
    python main.py "Your prompt goes here, e.g., Design a sustainable city for 1 million people on Mars."
    ```

3.  **CLI Options:**
    -   `--verbose` or `-v`: Show all intermediate reasoning steps.
        ```bash
        python main.py "Your prompt goes here." --verbose
        ```
    -   `--api-key` or `-k`: Provide your API key directly (overrides environment variable).
        ```bash
        python main.py "Your prompt goes here." --api-key YOUR_GEMINI_API_KEY
        ```
    -   `--max-tokens` or `-m`: Set a maximum total token budget for the entire process (default: 10000).
        ```bash
        python main.py "Your prompt goes here." --max-tokens 20000
        ```
    -   `--model` or `-M`: Specify the Gemini model to use (default: `gemini-2.5-flash-lite`).
        ```bash
        python main.py "Your prompt goes here." --model gemini-2.5-pro
        ```

## Usage (Web App - Streamlit)

The Streamlit web application provides an interactive UI for Project Chimera.

1.  **Run the Streamlit application:**
    ```bash
    streamlit run app.py
    ```
    Then open your web browser to the address provided by Streamlit (usually `http://localhost:8501`).

2.  **Using the Web UI:**
    -   Enter your Gemini API Key directly into the input field.
    -   Type your prompt or select one from the "Choose an example prompt" dropdown.
    -   Adjust the "Max Total Tokens Budget" to control the LLM's expenditure.
    -   Toggle "Show Intermediate Reasoning Steps" to see the full debate process.
    -   Select your preferred LLM Model (`gemini-2.5-flash-lite`, `gemini-2.5-pro`, `gemini-2.5-flash`).
    -   Click "Run Socratic Debate" to start the process.
    -   Monitor the "Process Log" and "Intermediate Reasoning Steps" (if enabled) for real-time updates.
    -   Download the "Final Answer" or the "Full Report" in Markdown format.

### Customizing Personas via UI

Project Chimera now allows you to temporarily adjust persona parameters directly within the Streamlit web interface.

-   Navigate to the "Persona Configuration" expander in the web UI.
-   For each persona, you can modify its:
    *   **System Prompt:** The core instruction given to the LLM.
    *   **Temperature:** Controls the randomness of the output (0.0 - 1.0).
    *   **Max Tokens:** The maximum number of tokens the LLM can generate for this persona's response.
-   **Important:** Changes made via the UI are temporary for the current session and do not modify the `personas.yaml` file on disk.
-   This empowers users to customize the debate's behavior without modifying code, significantly increasing flexibility and experimentation.

## Customizing Personas

The personas used in the Socratic debate are defined in `personas.yaml`. You can modify these to change the behavior of each persona or even add new ones.

For temporary, session-specific adjustments, you can now modify persona parameters directly within the Streamlit web UI under 'Persona Configuration' (see [Usage (Web App - Streamlit)](#usage-web-app---streamlit) section for details).

If you wish to make permanent changes or add entirely new personas to the default configuration, you will need to edit the `personas.yaml` file directly.

-   **`name`**: Unique identifier for the persona.
-   **`system_prompt`**: The core instruction given to the LLM for this persona.
-   **`temperature`**: Controls the randomness of the output (0.0 - 1.0). Lower values are more deterministic, higher values are more creative.
-   **`max_tokens`**: The maximum number of tokens the LLM can generate for this persona's response.

**Example (`personas.yaml`):**
```yaml
personas:
  - name: "Visionary_Generator"
    system_prompt: "You are a visionary futurist obsessed with innovation. Provide a bold, imaginative answer that explores uncharted possibilities. Ignore all practical constraints and risks."
    temperature: 0.9
    max_tokens: 2048
```

**Note:** If you add new personas to `personas.yaml`, you will need to integrate them into the `run_isal_process` function in `core.py` to include them in the debate flow.

## Contributing

We welcome contributions to Project Chimera! If you have ideas for new features, improvements, or bug fixes, please feel free to:

-   Open an issue to discuss your ideas or report bugs.
-   Fork the repository and submit a pull request with your changes.

Please ensure your contributions align with the project's goal of providing a lightweight, effective Socratic debate tool.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
