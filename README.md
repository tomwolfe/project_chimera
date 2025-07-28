# Project Chimera

A lightweight Python application that runs an **Iterative Socratic Arbitration Loop (ISAL)** using a single LLM adopting different reasoning personas. It's designed to explore complex problems from multiple angles, providing a more robust and nuanced answer than a single LLM query, now with domain-specific reasoning and enhanced configurability.

## Features

-   **Multi-Persona Socratic Debate:** Employs distinct LLM personas (Visionary, Skeptic, Critic, Arbitrator, Devil's Advocate, plus domain-specific ones and a Generalist Assistant fallback) to generate, critique, and synthesize ideas.
-   **Reasoning Frameworks:** Select from predefined sets of personas (e.g., General, Science, Business, Creative) to tailor the debate to specific problem domains. Includes an LLM-powered recommendation for the best framework based on your prompt.
-   **In-UI Persona Configuration:** Temporarily adjust individual persona parameters (system prompt, temperature, max tokens) directly within the Streamlit web interface for session-specific experimentation.
-   **Community Frameworks:** Save and load custom persona configurations as reusable "Community Frameworks" within your session, facilitating easy sharing and reuse of debate setups.
-   **Cost & Token Tracking:** Monitors and displays total token usage and estimated USD cost for each debate session, including real-time updates and proactive warnings for the next step's estimated cost.
-   **Real-time Status Updates:** Provides live feedback on the debate progress, including current step, token usage, and estimated cost for the next step.
-   **Enhanced Robustness:** Implemented advanced error handling with retries, persona fallbacks (e.g., to `Generalist_Assistant`), and graceful degradation of token limits for a more stable and resilient user experience, especially during API interactions.
-   **Intermediate Step Visibility:** Option to view the detailed output of each persona's contribution to the debate.
-   **Comprehensive Reporting:** Generates a full Markdown report of the debate, including prompt, configuration, process log, intermediate steps, and final answer, with options to download.
-   **Flexible Deployment:** Usable as a command-line tool or a Streamlit web application.
-   **User-Provided API Key:** Securely handles your Gemini API key, which is provided directly by the user at runtime and is not stored by the application.

## Core Concept: The Iterative Socratic Arbitration Loop (ISAL)

The ISAL process involves a series of LLM calls where different personas interact to refine an initial prompt. Project Chimera now supports multiple "Reasoning Frameworks," which are curated sets of these personas, allowing the debate to be tailored to specific problem domains.

Typically, the core debate flow involves:
1.  **Visionary Generator:** Creates an initial, often bold, idea.
2.  **Skeptical Generator:** Critiques the idea from a risk-averse perspective.
3.  **Constructive Critic:** Provides actionable improvements based on the critique.
4.  **Impartial Arbitrator:** Synthesizes all inputs into a final, balanced answer.
5.  **Devil's Advocate:** Offers a final, sharp critique to expose any remaining fundamental flaws.

In addition to these core roles, domain-specific personas (e.g., `Scientific_Analyst`, `Business_Strategist`) and a `Generalist_Assistant` for fallbacks are used within selected frameworks to enhance the debate's relevance and robustness. This iterative process aims to produce more robust, well-reasoned, and comprehensive outputs than a single LLM query.

## Documentation
For a detailed explanation of the project's architecture, components, and workflow, please see the [Implementation Overview](docs/implementation_overview.md).

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
    python main.py reason "Your prompt goes here, e.g., Design a sustainable city for 1 million people on Mars."
    ```

3.  **CLI Options:**
    -   `--verbose` or `-v`: Show all intermediate reasoning steps.
        ```bash
        python main.py reason "Your prompt goes here." --verbose
        ```
    -   `--api-key` or `-k`: Provide your API key directly (overrides environment variable).
        ```bash
        python main.py reason "Your prompt goes here." --api-key YOUR_GEMINI_API_KEY
        ```
    -   `--max-tokens` or `-m`: Set a maximum total token budget for the entire process (default: 10000).
        ```bash
        python main.py reason "Your prompt goes here." --max-tokens 20000
        ```
    -   `--model` or `-M`: Specify the Gemini model to use (default: `gemini-2.5-flash-lite`).
        ```bash
        python main.py reason "Your prompt goes here." --model gemini-2.5-pro
        ```
    -   `--domain` or `-d`: Specify a reasoning domain to use (e.g., `General`, `Science`, `Business`, `Creative`). Defaults to `auto` which attempts to recommend one based on the prompt.
        ```bash
        python main.py reason "Your prompt goes here." --domain Science
        ```

    The CLI output provides real-time status updates, including current token usage, estimated cost, and proactive warnings about the next step's token consumption relative to the budget.

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
    -   **Reasoning Framework Selection:** The app will recommend a framework based on your prompt. You can apply the recommendation or select a different one from the dropdown (e.g., `General`, `Science`, `Business`, `Creative`, or `Custom`).
    -   Toggle "Show Intermediate Reasoning Steps" to see the full debate process.
    -   Select your preferred LLM Model (`gemini-2.5-flash-lite`, `gemini-2.5-pro`, `gemini-2.5-flash`).
    -   **Persona Configuration:** Expand the "View and Edit Personas" section to inspect or temporarily modify the parameters of the personas in the currently selected framework.
    -   **Community Frameworks:** In the "Contribute Your Framework" expander, you can save your current persona configuration as a named framework for later use in the session. You can also apply existing community frameworks.
    -   Click "Run Socratic Debate" to start the process.
    -   Monitor the "Process Log" and "Intermediate Reasoning Steps" (if enabled) for real-time updates and detailed output from each persona.
    -   Download the "Final Answer" or the "Full Report" in Markdown format.

### In-UI Persona Configuration & Community Frameworks

Project Chimera now offers extensive persona customization directly within the Streamlit web interface, making it easier to experiment and share configurations.

-   **Reasoning Framework Selection:**
    *   The application now includes predefined "Reasoning Frameworks" (e.g., `General`, `Science`, `Business`, `Creative`) which are sets of personas tailored for specific problem types.
    *   An LLM-powered recommendation feature will suggest the most suitable framework based on your prompt.
    *   You can select any available framework from the dropdown. Choosing "Custom" will allow you to work with the currently loaded persona configuration.

-   **View and Edit Personas:**
    *   Under the "Persona Configuration" expander, you can see the personas active in your selected framework.
    *   For each persona, you can modify its:
        *   **System Prompt:** The core instruction given to the LLM.
        *   **Temperature:** Controls the randomness of the output (0.0 - 1.0).
        *   **Max Tokens:** The maximum number of tokens the LLM can generate for this persona's response.
    *   **Important:** Changes made via the UI are temporary for the current session and do not modify the `personas.yaml` file on disk. They will be lost when the app restarts or the session refreshes.

-   **Community Frameworks:**
    *   The "Contribute Your Framework" section allows you to save your current persona configuration (including any edits you've made) as a named "Community Framework".
    *   These saved frameworks will appear in the "Community Frameworks" dropdown, allowing you to easily switch between your custom configurations within the same session.
    *   To make a framework permanent or share it with others, you would still need to manually add it to the `personas.yaml` file and submit a pull request to the repository.

## Customizing Personas

The personas and their groupings used in the Socratic debate are now comprehensively defined in the `personas.yaml` file. This file allows you to:

-   **Define Individual Personas:** Each persona has a `name`, `system_prompt`, `temperature`, and `max_tokens`.
-   **Group Personas into Sets (Reasoning Frameworks):** The new `persona_sets` section allows you to define different collections of personas. This enables the application to offer domain-specific reasoning frameworks (e.g., `Science`, `Business`, `Creative`) by activating a specific subset of personas for a given problem.

For temporary, session-specific adjustments, you can now modify persona parameters directly within the Streamlit web UI under 'Persona Configuration' (see [In-UI Persona Configuration & Community Frameworks](#in-ui-persona-configuration--community-frameworks) section for details).

If you wish to make permanent changes or add entirely new personas to the default configuration, you will need to edit the `personas.yaml` file directly.

**`personas.yaml` Structure:**

```yaml
personas:
  # Define individual personas here
  - name: "Persona_Name_1"
    system_prompt: "..."
    temperature: 0.X
    max_tokens: YYYY

persona_sets:
  # Group personas into named sets (frameworks)
  General:
    - Persona_Name_1
    - Persona_Name_2
  Science:
    - Scientific_Persona_A
    - Scientific_Persona_B
```

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

  - name: "Constructive_Critic"
    system_prompt: "You are a sharp but constructive critic. Your goal is to find logical gaps and missing information in the provided text. Propose specific, actionable improvements."
    temperature: 0.4
    max_tokens: 1024

  - name: "Impartial_Arbitrator"
    system_prompt: "You are a wise and impartial synthesizer. Combine the revised answers and critiques into a single, balanced, and definitive final answer that incorporates the best elements of all inputs."
    temperature: 0.1
    max_tokens: 4096

  - name: "Devils_Advocate"
    system_prompt: "You are a ruthless debate champion. Your sole purpose is to find the single most critical, fundamental flaw in the proposed consensus. Do not offer solutions, only expose the weakness with a sharp, incisive critique."
    temperature: 1.0
    max_tokens: 512

  - name: "Generalist_Assistant"
    system_prompt: "You are a helpful and versatile AI assistant. Provide a concise and accurate response to the user's request, drawing upon general knowledge. If you cannot fulfill the request directly, provide a summary of the situation."
    temperature: 0.5
    max_tokens: 1024

  # Domain-specific personas
  - name: "Scientific_Analyst"
    system_prompt: "You are a rigorous scientific analyst. Focus on empirical evidence, logical consistency, and testable hypotheses. Identify gaps in data or methodology, and suggest areas for further research. Your critique should be objective and data-driven."
    temperature: 0.1
    max_tokens: 1500

  - name: "Business_Strategist"
    system_prompt: "You are a shrewd business strategist. Evaluate proposals based on market viability, financial implications, competitive advantage, and scalability. Identify opportunities and risks from a commercial perspective, focusing on practical implementation and ROI."
    temperature: 0.3
    max_tokens: 1500

  - name: "Creative_Thinker"
    system_prompt: "You are an imaginative and artistic creative. Focus on originality, aesthetic appeal, emotional impact, and narrative coherence. Explore unconventional ideas and artistic expression, pushing boundaries and inspiring new perspectives."
    temperature: 0.8
    max_tokens: 1500

  - name: "Scientific_Visionary"
    system_prompt: "You are a research scientist exploring innovative solutions. Provide evidence-based hypotheses that push boundaries while acknowledging methodological constraints."
    temperature: 0.7
    max_tokens: 2048

  - name: "Business_Innovator"
    system_prompt: "You are a forward-thinking business consultant. Propose bold, innovative business solutions focusing on market disruption and new revenue streams."
    temperature: 0.8
    max_tokens: 2048

  - name: "Creative_Visionary"
    system_prompt: "You are an artist exploring uncharted creative possibilities. Provide bold, imaginative solutions that push artistic boundaries without concern for practical constraints."
    temperature: 0.95
    max_tokens: 2048

# Define persona sets by referencing persona names
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
  Business:
    - Business_Innovator
    - Business_Strategist
    - Constructive_Critic
    - Impartial_Arbitrator
    - Devils_Advocate
  Creative:
    - Creative_Visionary
    - Creative_Thinker
    - Constructive_Critic
    - Impartial_Arbitrator
    - Devils_Advocate
```

**Note:** If you add new personas to `personas.yaml`, they will automatically be available for inclusion in `persona_sets`.

## Contributing

We welcome contributions to Project Chimera! If you have ideas for new features, improvements, or bug fixes, please feel free to:

-   Open an issue to discuss your ideas or report bugs.
-   Fork the repository and submit a pull request with your changes.

Please ensure your contributions align with the project's goal of providing a lightweight, effective Socratic debate tool.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.