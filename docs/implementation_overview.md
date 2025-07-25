# Final Synthesized Answer

This codebase implements "Project Chimera," a tool that uses a single Large Language Model (LLM) to simulate a Socratic debate process. The goal is to generate a more robust and nuanced answer to a given prompt by having the LLM adopt different personas to generate an initial idea, critique it, offer constructive feedback, and finally synthesize a comprehensive answer.

Here's a breakdown of how it works, step by step:

**1. Project Structure and Core Components:**

*   **`app.py`**: This is the Streamlit web application interface. It handles user input (API key, prompt, configuration), displays results, and orchestrates the execution of the core logic.
*   **`core.py`**: This file contains the main logic for the Socratic debate. It defines the `SocraticDebate` class, which manages the personas, the debate flow, token/cost tracking, and interaction with the LLM provider. It also includes the `run_isal_process` function to initialize the debate.
*   **`llm_provider.py`**: This module abstracts the interaction with the Google Gemini LLM. It handles API calls, token counting, cost calculation, and implements retry logic for API errors. It also defines custom exceptions for LLM-related issues.
*   **`main.py`**: This provides a command-line interface (CLI) for running the Socratic debate. It uses the `typer` library for argument parsing and integrates with `rich` for enhanced terminal output.
*   **`personas.yaml`**: This configuration file defines the different personas used in the debate (e.g., Visionary Generator, Skeptical Critic, Impartial Arbitrator). Each persona has a name, a system prompt (which guides its behavior), a temperature setting (for creativity vs. determinism), and a maximum token limit.
*   **`requirements.txt`**: Lists all the Python dependencies needed to run the project.
*   **`.dockerignore` & `Dockerfile`**: These files are for containerizing the application using Docker, making it easier to deploy.
*   **`.gitignore`**: Specifies files and directories that should be ignored by Git.
*   **`LICENSE` & `README.md`**: Standard project files for licensing and documentation.

**2. The Socratic Debate Flow (Orchestrated by `core.py` and `app.py`/`main.py`):**

The process is designed as an Iterative Socratic Arbitration Loop (ISAL):

*   **Initialization**:
    *   The user provides an initial `prompt`, a Gemini `api_key`, a `max_tokens_budget`, and selects a `model_name`.
    *   The `personas.yaml` file is loaded to define the roles and instructions for each persona.
    *   A `SocraticDebate` instance is created, configured with the user's inputs and the loaded personas. The `GeminiProvider` is initialized within this instance.
    *   A `status_callback` function is passed to `SocraticDebate` to provide real-time feedback to the user interface (Streamlit or CLI).

*   **Persona Execution (within `SocraticDebate.run_debate()`):**
    The `run_debate` method orchestrates a sequence of calls to `_execute_persona_step`, each representing a stage of the debate:

    1.  **Visionary Generator**:
        *   **Role**: To generate a bold, imaginative, and unconstrained initial response to the prompt.
        *   **Input**: The user's original prompt.
        *   **Output**: Stored as `Visionary_Generator_Output` in `intermediate_steps`. This output also updates the `current_thought`.
        *   **Error Handling**: If this step fails (e.g., due to API errors or token budget), subsequent steps will be affected, and the error will be noted.

    2.  **Skeptical Generator**:
        *   **Role**: To critically analyze the "Visionary Generator's" output, identifying potential flaws, vulnerabilities, and risks from a pragmatic perspective.
        *   **Input**: The `Visionary_Generator_Output` and the original prompt.
        *   **Output**: Stored as `Skeptical_Critique`.
        *   **Error Handling**: If the previous step failed, this step is skipped, and its output is marked as "N/A".

    3.  **Constructive Critic**:
        *   **Role**: To provide specific, actionable improvements to the proposal, addressing logical gaps and missing information identified in both the original proposal and the skeptical critique.
        *   **Input**: The `Visionary_Generator_Output`, `Skeptical_Critique`, and the original prompt.
        *   **Output**: Stored as `Constructive_Feedback`.
        *   **Error Handling**: Similar to the Skeptical Generator, it checks for previous step failures.

    4.  **Impartial Arbitrator**:
        *   **Role**: To synthesize all previous inputs (original prompt, visionary proposal, skeptical critique, constructive feedback) into a single, balanced, and definitive final answer. It aims to incorporate the best elements and address critiques.
        *   **Input**: All previous outputs.
        *   **Output**: Stored as `Arbitrator_Output`. This is designated as the `final_answer`.
        *   **Error Handling**: This is a critical step. If it fails, the `final_answer` is set to an error message.

    5.  **Devil's Advocate**:
        *   **Role**: To find the single most critical, fundamental flaw in the *final synthesized answer* generated by the Arbitrator. It focuses solely on exposing weaknesses, not offering solutions.
        *   **Input**: The `Arbitrator_Output`.
        *   **Output**: Stored as `Devils_Advocate_Critique`.
        *   **Error Handling**: Checks for errors in the final answer before proceeding.

*   **Token and Cost Management**:
    *   Throughout the process, `core.py` tracks `cumulative_token_usage` and `cumulative_usd_cost`.
    *   Before each LLM call, it estimates the tokens and cost for that step.
    *   It checks if the estimated tokens for the current step would exceed the `max_total_tokens_budget`. If so, it raises a `TokenBudgetExceededError`.
    *   A strict check is performed *after* generation to ensure the actual tokens used don't exceed the budget. If they do, the response is truncated, and an error is raised.
    *   The `llm_provider.py` module contains the logic for calculating costs based on token usage and model pricing tiers.

*   **Output and Reporting**:
    *   The `final_answer` is the output from the "Impartial Arbitrator".
    *   All intermediate outputs and token counts are stored in the `intermediate_steps` dictionary.
    *   The `app.py` and `main.py` scripts display the process log, intermediate steps (if enabled), and the final answer.
    *   `app.py` also provides functionality to download the final answer and a full Markdown report containing all details of the debate.

**3. User Interface (`app.py` and `main.py`):**

*   **Streamlit (`app.py`)**:
    *   Provides input fields for API key, prompt, max tokens, and model selection.
    *   Includes an example prompt selector for ease of use.
    *   Allows users to view and temporarily edit persona parameters (system prompt, temperature, max tokens) via an expander.
    *   Displays real-time status updates using `st.status`.
    *   Renders the process log, intermediate steps, and final answer.
    *   Offers download buttons for the final answer and a full report.

*   **CLI (`main.py`)**:
    *   Uses `typer` to define command-line arguments and options.
    *   Accepts the prompt as a positional argument and other configurations as options (`--api-key`, `--max-tokens`, `--model`, `--verbose`).
    *   Uses `rich.console` for formatted output in the terminal, including status messages, errors, and the final answer.
    *   The `--verbose` flag shows all intermediate steps.

**4. Error Handling:**

*   Custom exceptions (`TokenBudgetExceededError`, `GeminiAPIError`, `LLMUnexpectedError`) are defined in `llm_provider.py` and used throughout the codebase.
*   The `GeminiProvider` implements retry logic for transient API errors (e.g., rate limits, server errors).
*   The `SocraticDebate` class and the UI layers gracefully handle these exceptions, providing informative messages to the user and attempting to continue or halt the process appropriately.

In essence, Project Chimera leverages the LLM's ability to adopt different personas to simulate a structured, multi-faceted reasoning process, aiming for more comprehensive and well-considered outputs while providing transparency into the steps and costs involved.
