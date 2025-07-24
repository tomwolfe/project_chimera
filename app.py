# app.py
import streamlit as st
import os
from core import run_isal_process
from core import TokenBudgetExceededError # Import new exception
import sys
from rich.text import Text
from rich.syntax import Syntax
import io
import contextlib
from llm_provider import GeminiAPIError, LLMUnexpectedError # Import specific exceptions
from llm_provider import GeminiProvider # Import GeminiProvider to access cost calculation
import re # Added for stripping ANSI codes
import datetime # Added for timestamp in report
from typing import Dict, Any # Add this line

# Redirect rich console output to a string buffer for Streamlit display
@contextlib.contextmanager
def capture_rich_output():
    buffer = io.StringIO()
    
    # Temporarily replace sys.stdout to capture print statements
    old_stdout = sys.stdout
    sys.stdout = buffer 
    
    # Patch the global rich Console instance used by main.py/core.py
    # This is a workaround to capture output from imported modules that use rich.
    # A more robust solution for larger projects would be to pass the Console object explicitly.
    try:
        from main import console as typer_console_instance
        old_typer_console_file = typer_console_instance.__dict__['_file']
        typer_console_instance.__dict__['_file'] = buffer # Redirect rich console output
        typer_console_instance.force_terminal = True # Ensure colors/formatting are written to buffer
    except ImportError:
        # main.py might not be imported if app.py is run directly without main.py being in sys.modules
        # In this case, rich console output from core.py won't be captured this way.
        # For simplicity of MVP, we assume main.py's console is the one to patch.
        pass
    
    yield buffer
    
    # Restore original stdout and rich console file
    sys.stdout = old_stdout
    try:
        from main import console as typer_console_instance
        typer_console_instance.__dict__['_file'] = old_typer_console_file
        typer_console_instance.force_terminal = False # Restore default
    except ImportError:
        pass

# Function to strip ANSI escape codes from text
ansi_escape_re = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')
def strip_ansi_codes(text):
    return ansi_escape_re.sub('', text)

# --- Helper function for Markdown Report Generation ---
def generate_markdown_report(user_prompt: str, final_answer: str, intermediate_steps: Dict[str, Any], process_log_output: str, config_params: Dict[str, Any]) -> str:
    """
    Generates a comprehensive Markdown report of the Socratic Debate process.
    """
    report_date = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    md_content = f"# Project Chimera Socratic Debate Report\n\n"
    md_content += f"**Date:** {report_date}\n"
    md_content += f"**Original Prompt:** {user_prompt}\n\n"

    md_content += "---\n\n"
    md_content += "## Configuration\n\n"
    md_content += f"*   **Model:** {config_params.get('model_name', 'N/A')}\n"
    md_content += f"*   **Max Total Tokens Budget:** {config_params.get('max_tokens_budget', 'N/A')}\n"
    md_content += f"*   **Intermediate Steps Shown in UI:** {'Yes' if config_params.get('show_intermediate_steps', False) else 'No'}\n\n"

    md_content += "---\n\n"
    md_content += "## Process Log\n\n"
    md_content += "```text\n"
    md_content += strip_ansi_codes(process_log_output) # Strip ANSI codes for cleaner text file
    md_content += "\n```\n\n"

    if config_params.get('show_intermediate_steps', True): # Always include in full report if available
        md_content += "---\n\n"
        md_content += "## Intermediate Reasoning Steps\n\n"
        
        # Create a list of step names to process, excluding token counts and the total
        step_keys_to_process = [k for k in intermediate_steps.keys() 
                                if not k.endswith("_Tokens_Used") and k != "Total_Tokens_Used" and k != "Total_Estimated_Cost_USD"]
        
        for step_key in step_keys_to_process: # step_key here is the content key
            # Determine the display name for the section
            display_name = step_key.replace('_Output', '').replace('_Critique', '').replace('_Feedback', '').replace('_', ' ').title()
            
            content = intermediate_steps.get(step_key, "N/A")
            
            # Find the corresponding token count key based on core.py's naming
            token_base_name = step_key
            if step_key.endswith("_Output"):
                token_base_name = step_key.replace("_Output", "")
            # For Skeptical_Critique, Constructive_Feedback, Devils_Advocate_Critique,
            # the step_key itself is the base for the token key.
            
            token_count_key = f"{token_base_name}_Tokens_Used"
            tokens_used = intermediate_steps.get(token_count_key, "N/A")

            md_content += f"### {display_name}\n\n"
            md_content += f"```markdown\n{content}\n```\n"
            md_content += f"**Tokens Used for this step:** {tokens_used}\n\n"

    md_content += "---\n\n"
    md_content += "## Final Synthesized Answer\n\n"
    md_content += f"{final_answer}\n\n" # Final answer is already markdown

    md_content += "---\n\n"
    md_content += "## Summary\n\n"
    md_content += f"**Total Tokens Consumed:** {intermediate_steps.get('Total_Tokens_Used', 'N/A')}\n"
    md_content += f"**Total Estimated Cost:** {intermediate_steps.get('Total_Estimated_Cost_USD', 'N/A')}\n" # Add cost to report

    return md_content


st.set_page_config(layout="wide", page_title="Project Chimera Web App")

st.title("Project Chimera: Socratic Self-Debate")
st.markdown("Run an Iterative Socratic Arbitration Loop (ISAL) using a single LLM with multiple personas.")
st.markdown("This project's core software is open-source and available on [GitHub](https://github.com/tomwolfe/project_chimera).")

# Pre-defined Prompt Templates/Examples
EXAMPLE_PROMPTS = {
    "Design a Mars City": "Design a sustainable city for 1 million people on Mars, considering resource scarcity and human psychology.",
    "Ethical AI Framework": "Develop an ethical framework for an AI system designed to assist in judicial sentencing, addressing bias, transparency, and accountability.",
    "Future of Education": "Describe the future of education in 2050, incorporating AI, virtual reality, and personalized learning paths.",
    "Climate Change Solution": "Propose an innovative, scalable solution to mitigate the effects of climate change, focusing on a specific sector (e.g., energy, agriculture, transportation).",
    "Space Tourism Business Plan": "Outline a business plan for a luxury space tourism company, detailing target audience, unique selling propositions, and operational challenges.",
    # --- New Example Prompts ---
    "Quantum Computing Impact": "Analyze the potential societal and economic impacts of widespread quantum computing by 2040, including both opportunities and risks.",
    "Personalized Medicine Ethics": "Discuss the ethical implications of highly personalized medicine, considering data privacy, equitable access, and genetic manipulation.",
    "Sustainable Urban Farming": "Detail a comprehensive plan for implementing sustainable urban farming on a large scale in a major metropolitan area, addressing infrastructure, community engagement, and economic viability.",
    "AI in Creative Arts": "Explore how AI will transform creative industries (e.g., music, visual art, writing) over the next decade, focusing on collaboration between human and AI artists.",
    "Global Water Crisis Solution": "Propose a multi-faceted, international strategy to address the global water crisis, integrating technological, policy, and behavioral changes.",
    "Future of Work": "Describe the future of work in a world increasingly dominated by automation and AI, considering new job roles, economic models, and the role of human creativity.",
    "Interstellar Travel Challenges": "Identify and propose solutions for the primary scientific and engineering challenges of achieving interstellar travel within the next 200 years.",
}

# --- Reset Function ---
def reset_app_state():
    """Resets all input fields and clears outputs."""
    st.session_state.api_key_input = ""
    st.session_state.user_prompt_input = EXAMPLE_PROMPTS["Design a Mars City"] # Reset to default example
    st.session_state.max_tokens_budget_input = 10000
    st.session_state.show_intermediate_steps_checkbox = True
    st.session_state.selected_model_selectbox = "gemini-2.5-flash-lite"
    # When resetting, set example_selector to the key of the default prompt
    st.session_state.example_selector = "Design a Mars City" 
    
    # Clear all output-related session states
    st.session_state.debate_ran = False
    st.session_state.final_answer_output = ""
    st.session_state.intermediate_steps_output = {}
    st.session_state.process_log_output_text = ""
    st.session_state.last_config_params = {}
    # No st.rerun() needed here. Streamlit will rerun automatically after the callback.

# Initialize all session state variables at the top
if "api_key_input" not in st.session_state:
    st.session_state.api_key_input = os.getenv("GEMINI_API_KEY", "")
if "user_prompt_input" not in st.session_state:
    st.session_state.user_prompt_input = EXAMPLE_PROMPTS["Design a Mars City"]
if "max_tokens_budget_input" not in st.session_state:
    st.session_state.max_tokens_budget_input = 10000
if "show_intermediate_steps_checkbox" not in st.session_state:
    st.session_state.show_intermediate_steps_checkbox = True
if "selected_model_selectbox" not in st.session_state:
    st.session_state.selected_model_selectbox = "gemini-2.5-flash-lite"
# Initialize example_selector to the key of the default prompt
if "example_selector" not in st.session_state:
    st.session_state.example_selector = "Design a Mars City"

# New session state variables for outputs and control
if "debate_ran" not in st.session_state:
    st.session_state.debate_ran = False
if "final_answer_output" not in st.session_state:
    st.session_state.final_answer_output = ""
if "intermediate_steps_output" not in st.session_state:
    st.session_state.intermediate_steps_output = {}
if "process_log_output_text" not in st.session_state:
    st.session_state.process_log_output_text = ""
if "last_config_params" not in st.session_state: # Store config for report generation
    st.session_state.last_config_params = {}


# API Key Input
api_key = st.text_input(
    "Enter your Gemini API Key",
    type="password",
    help="Your API key will not be stored. For deployed apps, consider using Streamlit Secrets (`st.secrets`).",
    key="api_key_input" # Value is implicitly taken from st.session_state.api_key_input
)
st.markdown("Need a Gemini API key? Get one from [Google AI Studio](https://aistudio.google.com/apikey).")


# Callback function to update the prompt text area when an example is selected
def update_prompt_from_example():
    selected_example_name = st.session_state.example_selector
    if selected_example_name:
        st.session_state.user_prompt_input = EXAMPLE_PROMPTS[selected_example_name]

# Add Example Prompt Selector
st.selectbox(
    "Choose an example prompt:",
    options=[""] + list(EXAMPLE_PROMPTS.keys()), # Add an empty option for "no selection"
    # Removed 'index' parameter. Streamlit will automatically select based on st.session_state.example_selector
    help="Select a pre-defined prompt to quickly get started or see examples.",
    key="example_selector", # Key for the selectbox
    on_change=update_prompt_from_example # Callback to run when selection changes
)

# Prompt Input
user_prompt = st.text_area(
    "Enter your prompt here:",
    height=150,
    key="user_prompt_input" # Value is implicitly taken from st.session_state.user_prompt_input
)

# Configuration Options
col1, col2 = st.columns(2)
with col1:
    max_tokens_budget = st.number_input(
        "Max Total Tokens Budget:",
        min_value=1000,
        max_value=50000, # Set a reasonable upper limit
        step=1000,
        help="Controls the maximum number of tokens used across all LLM calls to manage cost.",
        key="max_tokens_budget_input" # Value is implicitly taken from st.session_state.max_tokens_budget_input
    )
with col2:
    show_intermediate_steps = st.checkbox(
        "Show Intermediate Reasoning Steps",
        key="show_intermediate_steps_checkbox" # Value is implicitly taken from st.session_state.show_intermediate_steps_checkbox
    )

# Model Selection
selected_model = st.selectbox(
    "Select LLM Model",
    ["gemini-2.5-flash-lite", "gemini-2.5-pro", "gemini-2.5-flash"],
    index=["gemini-2.5-flash-lite", "gemini-2.5-pro", "gemini-2.5-flash"].index(st.session_state.selected_model_selectbox), # Set index based on current model
    help="Choose the Gemini model based on your needs for speed, cost, or capability.",
    key="selected_model_selectbox" # Value is implicitly taken from st.session_state.selected_model_selectbox
)

# Run and Reset Buttons
run_col, reset_col = st.columns([0.7, 0.3]) # Adjust column width for buttons
with run_col:
    run_button_clicked = st.button("Run Socratic Debate", type="primary")
with reset_col:
    st.button("Reset All Inputs & Outputs", on_click=reset_app_state)


if run_button_clicked:
    if not api_key:
        st.error("Please enter your Gemini API Key to proceed.")
    elif not user_prompt.strip():
        st.error("Please enter a prompt.")
    else:
        # Reset output states at the start of a new run
        st.session_state.debate_ran = False # Set to False initially, then True on success/error
        st.session_state.final_answer_output = ""
        st.session_state.intermediate_steps_output = {}
        st.session_state.process_log_output_text = ""
        st.session_state.last_config_params = {}

        # Use st.status for real-time feedback during the process
        with st.status("Initializing Socratic Debate...", expanded=True) as status:
            # Placeholders for real-time metrics
            st.markdown("---")
            metric_col1, metric_col2, metric_col3 = st.columns(3)
            total_tokens_placeholder = metric_col1.empty()
            total_cost_placeholder = metric_col2.empty()
            next_step_warning_placeholder = metric_col3.empty()
            st.markdown("---")

            # Define the callback function to update Streamlit UI elements
            def streamlit_status_callback(message: str, state: str = "running", expanded: bool = True,
                                          current_total_tokens: int = 0, current_total_cost: float = 0.0,
                                          estimated_next_step_tokens: int = 0, estimated_next_step_cost: float = 0.0):
                status.update(label=message, state=state, expanded=expanded)
                total_tokens_placeholder.metric("Total Tokens Used", f"{current_total_tokens}")
                total_cost_placeholder.metric("Estimated Cost (USD)", f"${current_total_cost:.4f}")

                # Proactive warning for next step
                if estimated_next_step_tokens > 0:
                    budget_remaining = max_tokens_budget - current_total_tokens
                    if estimated_next_step_tokens > budget_remaining:
                        next_step_warning_placeholder.warning(
                            f"⚠️ Next step ({estimated_next_step_tokens} tokens) "
                            f"will exceed budget ({budget_remaining} remaining). "
                            f"Estimated cost: ${estimated_next_step_cost:.4f}"
                        )
                    else:
                        next_step_warning_placeholder.info(
                            f"Next step estimated: {estimated_next_step_tokens} tokens "
                            f"(${(estimated_next_step_cost):.4f}). "
                            f"Budget remaining: {budget_remaining} tokens."
                        )
                else:
                    next_step_warning_placeholder.empty() # Clear warning if no next step estimate

            # Use the context manager to capture rich output for the log display
            with capture_rich_output() as rich_output_buffer:
                try:
                    final_answer, intermediate_steps = run_isal_process(
                        user_prompt, api_key, max_total_tokens_budget=max_tokens_budget,
                        streamlit_status_callback=streamlit_status_callback, # Pass the callback
                        model_name=selected_model # Pass the selected model name
                    )
                    st.session_state.process_log_output_text = rich_output_buffer.getvalue()
                    st.session_state.final_answer_output = final_answer
                    st.session_state.intermediate_steps_output = intermediate_steps
                    st.session_state.last_config_params = {
                        "max_tokens_budget": max_tokens_budget,
                        "model_name": selected_model,
                        "show_intermediate_steps": show_intermediate_steps
                    }
                    st.session_state.debate_ran = True # Set to True only on successful completion
                    
                    # Update status to complete after successful execution
                    status.update(label="Socratic Debate Complete!", state="complete", expanded=False)
                    # Ensure final metrics are displayed
                    final_total_tokens = intermediate_steps.get('Total_Tokens_Used', 0)
                    final_total_cost_str = intermediate_steps.get('Total_Estimated_Cost_USD', "$0.0000")
                    # Remove '$' for float conversion if needed, but for display, keep as string
                    final_total_cost = float(final_total_cost_str.replace('$', '')) if isinstance(final_total_cost_str, str) else 0.0

                    total_tokens_placeholder.metric("Total Tokens Used", f"{final_total_tokens}")
                    total_cost_placeholder.metric("Estimated Cost (USD)", f"{final_total_cost_str}")
                    next_step_warning_placeholder.empty() # Clear any pending warnings

                except TokenBudgetExceededError as e: # Catch the new specific error
                    st.session_state.process_log_output_text = rich_output_buffer.getvalue()
                    st.session_state.intermediate_steps_output = intermediate_steps # Capture partial steps
                    st.session_state.last_config_params = { # Capture config even on error
                        "max_tokens_budget": max_tokens_budget,
                        "model_name": selected_model,
                        "show_intermediate_steps": show_intermediate_steps
                    }
                    st.session_state.debate_ran = True # Still show error output
                    error_message = str(e)
                    user_advice = "The process was stopped because it would exceed the maximum token budget."
                    status.update(label=f"Socratic Debate Failed: {user_advice}", state="error", expanded=True)
                    st.error(f"**Error:** {user_advice}\n\n**Details:** {error_message}")
                    # Ensure final metrics are displayed even on error
                    total_tokens_placeholder.metric("Total Tokens Used", f"{intermediate_steps.get('Total_Tokens_Used', 0)}")
                    total_cost_placeholder.metric("Estimated Cost (USD)", f"{intermediate_steps.get('Total_Estimated_Cost_USD', '$0.0000')}")

                except GeminiAPIError as e:
                    st.session_state.process_log_output_text = rich_output_buffer.getvalue()
                    st.session_state.intermediate_steps_output = intermediate_steps
                    st.session_state.last_config_params = {
                        "max_tokens_budget": max_tokens_budget,
                        "model_name": selected_model,
                        "show_intermediate_steps": show_intermediate_steps
                    }
                    st.session_state.debate_ran = True
                    error_message = str(e) # Get the message from the exception object
                    user_advice = "An issue occurred with the Gemini API."
                    if "invalid API key" in error_message.lower() or (hasattr(e, 'code') and e.code == 401):
                        user_advice = "Invalid API Key. Please check your Gemini API key and ensure it is correct."
                    elif "rate limit" in error_message.lower() or (hasattr(e, 'code') and e.code == 429):
                        user_advice = "Rate Limit Exceeded. You've made too many requests. Please wait a moment and try again."
                    elif "quota" in error_message.lower() or (hasattr(e, 'code') and e.code == 403):
                        user_advice = "Quota Exceeded or Access Denied. Please check your Gemini API quota or permissions."
                    elif "model" in error_message.lower() and "not found" in error_message.lower():
                        user_advice = f"Selected model '{selected_model}' not found or not available. Please choose a different model."
                    
                    status.update(label=f"Socratic Debate Failed: {user_advice}", state="error", expanded=True)
                    st.error(f"**Error:** {user_advice}\n\n**Details:** {error_message}")
                    # Ensure final metrics are displayed even on error
                    total_tokens_placeholder.metric("Total Tokens Used", f"{intermediate_steps.get('Total_Tokens_Used', 0)}")
                    total_cost_placeholder.metric("Estimated Cost (USD)", f"{intermediate_steps.get('Total_Estimated_Cost_USD', '$0.0000')}")

                except LLMUnexpectedError as e:
                    st.session_state.process_log_output_text = rich_output_buffer.getvalue()
                    st.session_state.intermediate_steps_output = intermediate_steps
                    st.session_state.last_config_params = {
                        "max_tokens_budget": max_tokens_budget,
                        "model_name": selected_model,
                        "show_intermediate_steps": show_intermediate_steps
                    }
                    st.session_state.debate_ran = True
                    error_message = str(e)
                    user_advice = "An unexpected issue occurred with the LLM provider (e.g., network problem, malformed response). Please try again later."
                    
                    status.update(label=f"Socratic Debate Failed: {user_advice}", state="error", expanded=True)
                    st.error(f"**Error:** {user_advice}\n\n**Details:** {error_message}")
                    # Ensure final metrics are displayed even on error
                    total_tokens_placeholder.metric("Total Tokens Used", f"{intermediate_steps.get('Total_Tokens_Used', 0)}")
                    total_cost_placeholder.metric("Estimated Cost (USD)", f"{intermediate_steps.get('Total_Estimated_Cost_USD', '$0.0000')}")

                except Exception as e:
                    st.session_state.process_log_output_text = rich_output_buffer.getvalue()
                    st.session_state.intermediate_steps_output = intermediate_steps
                    st.session_state.last_config_params = {
                        "max_tokens_budget": max_tokens_budget,
                        "model_name": selected_model,
                        "show_intermediate_steps": show_intermediate_steps
                    }
                    st.session_state.debate_ran = True
                    # Update status to error if an exception occurs
                    status.update(label=f"Socratic Debate Failed: {e}", state="error", expanded=True)
                    st.error(f"An unexpected error occurred during the process: {e}")
                    # Ensure final metrics are displayed even on error
                    total_tokens_placeholder.metric("Total Tokens Used", f"{intermediate_steps.get('Total_Tokens_Used', 0)}")
                    total_cost_placeholder.metric("Estimated Cost (USD)", f"{intermediate_steps.get('Total_Estimated_Cost_USD', '$0.0000')}")

# This block will only render if a debate has been run (successfully or with error)
if st.session_state.debate_ran:
    st.subheader("Process Log")
    st.code(strip_ansi_codes(st.session_state.process_log_output_text), language="text")

    if st.session_state.show_intermediate_steps_checkbox: # Use session state for checkbox value
        st.subheader("Intermediate Reasoning Steps")
        # Filter out Total_Tokens_Used and Total_Estimated_Cost_USD for this display, it's shown separately
        display_steps = {k: v for k, v in st.session_state.intermediate_steps_output.items() 
                         if k not in ["Total_Tokens_Used", "Total_Estimated_Cost_USD"]}
        
        # Group step outputs with their token counts for display
        processed_keys = set()
        for content_key, content in display_steps.items(): # content_key is like 'Visionary_Generator_Output', 'Skeptical_Critique'
            if content_key.endswith("_Tokens_Used") or content_key in processed_keys:
                continue

            # Determine the display name for the expander
            display_name = content_key.replace('_Output', '').replace('_Critique', '').replace('_Feedback', '').replace('_', ' ').title()
            
            # Determine the actual token count key from core.py's naming convention
            # The token key is always the content_key with '_Tokens_Used' appended,
            # UNLESS the content_key itself is a "base" name like 'Visionary_Generator'
            # For consistency, core.py uses the step_name_prefix for the token key.
            # We need to map the content_key back to that step_name_prefix.
            
            token_base_name = content_key
            if content_key.endswith("_Output"): # e.g., Visionary_Generator_Output, Arbitrator_Output
                # The token key is based on the part before '_Output'
                token_base_name = content_key.replace("_Output", "")
            # For Skeptical_Critique, Constructive_Feedback, Devils_Advocate_Critique,
            # the content_key itself is the base for the token key.
            
            token_count_key = f"{token_base_name}_Tokens_Used"
            
            # Get the content and token count
            step_content = content
            step_tokens = display_steps.get(token_count_key, "N/A (not recorded)")

            with st.expander(f"### {display_name}"):
                st.code(step_content, language="markdown")
                st.write(f"**Tokens used for this step:** {step_tokens}")
            
            processed_keys.add(content_key)
            processed_keys.add(token_count_key)

    st.subheader("Final Synthesized Answer")
    st.markdown(st.session_state.final_answer_output) # Use session state for final answer

    # --- Export Functionality ---
    st.markdown("---")
    st.subheader("Export Results")

    # Use the prompt from the text area, as it's the current state of the prompt input
    final_answer_md = f"# Final Synthesized Answer\n\n{st.session_state.final_answer_output}"
    st.download_button(
        label="Download Final Answer (Markdown)",
        data=final_answer_md,
        file_name="final_answer.md",
        mime="text/markdown"
    )

    full_report_md = generate_markdown_report(
        user_prompt=user_prompt, # This is the current user_prompt from the text_area
        final_answer=st.session_state.final_answer_output,
        intermediate_steps=st.session_state.intermediate_steps_output,
        process_log_output=st.session_state.process_log_output_text,
        config_params=st.session_state.last_config_params # Use stored config
    )
    st.download_button(
        label="Download Full Report (Markdown)",
        data=full_report_md,
        file_name="socratic_debate_report.md",
        mime="text/markdown"
    )
    st.info("To generate a PDF, download the Markdown report and use your browser's 'Print to PDF' option (usually accessible via Ctrl+P or Cmd+P).")