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
@contextlib.contextmanager # Corrected: lowercase 'm' in contextmanager
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
# Moved this function definition here, before it's called in the main Streamlit logic.
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
        
        for step_key in step_keys_to_process:
            display_name = step_key.replace('_', ' ').title()
            content = intermediate_steps.get(step_key, "N/A")
            
            # Find the corresponding token count
            token_count_key = f"{step_key.replace('_Output', '').replace('_Critique', '').replace('_Feedback', '')}_Tokens_Used"
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

# API Key Input
api_key = st.text_input(
    "Enter your Gemini API Key",
    type="password",
    help="Your API key will not be stored. For deployed apps, consider using Streamlit Secrets (`st.secrets`).",
    value=os.getenv("GEMINI_API_KEY") # Pre-fill if env var is set
)
st.markdown("Need a Gemini API key? Get one from [Google AI Studio](https://aistudio.google.com/apikey).")

# Pre-defined Prompt Templates/Examples
EXAMPLE_PROMPTS = {
    "Design a Mars City": "Design a sustainable city for 1 million people on Mars, considering resource scarcity and human psychology.",
    "Ethical AI Framework": "Develop an ethical framework for an AI system designed to assist in judicial sentencing, addressing bias, transparency, and accountability.",
    "Future of Education": "Describe the future of education in 2050, incorporating AI, virtual reality, and personalized learning paths.",
    "Climate Change Solution": "Propose an innovative, scalable solution to mitigate the effects of climate change, focusing on a specific sector (e.g., energy, agriculture, transportation).",
    "Space Tourism Business Plan": "Outline a business plan for a luxury space tourism company, detailing target audience, unique selling propositions, and operational challenges.",
}

# Initialize session state for the prompt input if not already present
if "user_prompt_input" not in st.session_state:
    st.session_state.user_prompt_input = EXAMPLE_PROMPTS["Design a Mars City"] # Default initial prompt

# Callback function to update the prompt text area when an example is selected
def update_prompt_from_example():
    selected_example_name = st.session_state.example_selector
    if selected_example_name:
        st.session_state.user_prompt_input = EXAMPLE_PROMPTS[selected_example_name]

# Add Example Prompt Selector
st.selectbox(
    "Choose an example prompt:",
    options=[""] + list(EXAMPLE_PROMPTS.keys()), # Add an empty option for "no selection"
    index=0, # Default to the empty option
    help="Select a pre-defined prompt to quickly get started or see examples.",
    key="example_selector", # Key for the selectbox
    on_change=update_prompt_from_example # Callback to run when selection changes
)

# Prompt Input
user_prompt = st.text_area(
    "Enter your prompt here:",
    height=150,
    key="user_prompt_input" # This key links the widget to the session state variable
)

# Configuration Options
col1, col2 = st.columns(2)
with col1:
    max_tokens_budget = st.number_input(
        "Max Total Tokens Budget:",
        min_value=1000,
        max_value=50000, # Set a reasonable upper limit
        value=10000,
        step=1000,
        help="Controls the maximum number of tokens used across all LLM calls to manage cost."
    )
with col2:
    show_intermediate_steps = st.checkbox("Show Intermediate Reasoning Steps", value=True)

# Model Selection
selected_model = st.selectbox(
    "Select LLM Model",
    ["gemini-2.5-flash-lite", "gemini-2.5-pro", "gemini-2.5-flash"],
    index=0, # Default to gemini-2.5-flash-lite
    help="Choose the Gemini model based on your needs for speed, cost, or capability."
)

if st.button("Run Socratic Debate", type="primary"):
    if not api_key:
        st.error("Please enter your Gemini API Key to proceed.")
    elif not user_prompt.strip():
        st.error("Please enter a prompt.")
    else:
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
                final_answer = "" # Initialize to empty string
                intermediate_steps = {} # Initialize to empty dict
                process_log_output = "" # Initialize process log

                try:
                    final_answer, intermediate_steps = run_isal_process(
                        user_prompt, api_key, max_total_tokens_budget=max_tokens_budget,
                        streamlit_status_callback=streamlit_status_callback, # Pass the callback
                        model_name=selected_model # Pass the selected model name
                    )
                    process_log_output = rich_output_buffer.getvalue()
                    
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

                    # Display captured console output (e.g., "Running persona: ...")
                    st.subheader("Process Log")
                    st.code(strip_ansi_codes(process_log_output), language="text")

                    if show_intermediate_steps:
                        st.subheader("Intermediate Reasoning Steps")
                        # Filter out Total_Tokens_Used and Total_Estimated_Cost_USD for this display, it's shown separately
                        display_steps = {k: v for k, v in intermediate_steps.items() 
                                         if k not in ["Total_Tokens_Used", "Total_Estimated_Cost_USD"]}
                        
                        # Group step outputs with their token counts for display
                        processed_keys = set()
                        for step_name, content in display_steps.items():
                            if step_name.endswith("_Tokens_Used") or step_name in processed_keys:
                                continue

                            # Determine the base name and corresponding token key
                            base_name = step_name.replace('_Output', '').replace('_Critique', '').replace('_Feedback', '')
                            token_count_key = f"{base_name}_Tokens_Used"
                            
                            # Get the content and token count
                            step_content = content
                            step_tokens = display_steps.get(token_count_key, "N/A (not recorded)")

                            with st.expander(f"### {base_name.replace('_', ' ').title()}"):
                                st.code(step_content, language="markdown")
                                st.write(f"**Tokens used for this step:** {step_tokens}")
                            
                            processed_keys.add(step_name)
                            processed_keys.add(token_count_key)

                    st.subheader("Final Synthesized Answer")
                    st.markdown(final_answer) # Use markdown for final answer

                    # --- Export Functionality ---
                    st.markdown("---")
                    st.subheader("Export Results")

                    # Generate Markdown for Final Answer
                    final_answer_md = f"# Final Synthesized Answer\n\n{final_answer}"
                    st.download_button(
                        label="Download Final Answer (Markdown)",
                        data=final_answer_md,
                        file_name="final_answer.md",
                        mime="text/markdown"
                    )

                    # Generate Full Report Markdown
                    full_report_md = generate_markdown_report(
                        user_prompt=user_prompt,
                        final_answer=final_answer,
                        intermediate_steps=intermediate_steps,
                        process_log_output=process_log_output,
                        config_params={
                            "max_tokens_budget": max_tokens_budget,
                            "model_name": selected_model,
                            "show_intermediate_steps": show_intermediate_steps
                        }
                    )
                    st.download_button(
                        label="Download Full Report (Markdown)",
                        data=full_report_md,
                        file_name="socratic_debate_report.md",
                        mime="text/markdown"
                    )
                    st.info("To generate a PDF, download the Markdown report and use your browser's 'Print to PDF' option (usually accessible via Ctrl+P or Cmd+P).")

                except TokenBudgetExceededError as e: # Catch the new specific error
                    error_message = str(e)
                    user_advice = "The process was stopped because it would exceed the maximum token budget."
                    status.update(label=f"Socratic Debate Failed: {user_advice}", state="error", expanded=True)
                    st.error(f"**Error:** {user_advice}\n\n**Details:** {error_message}")
                    # Ensure final metrics are displayed even on error
                    total_tokens_placeholder.metric("Total Tokens Used", f"{intermediate_steps.get('Total_Tokens_Used', 0)}")
                    total_cost_placeholder.metric("Estimated Cost (USD)", f"{intermediate_steps.get('Total_Estimated_Cost_USD', '$0.0000')}")
                    st.code(strip_ansi_codes(rich_output_buffer.getvalue()), language="text")

                except GeminiAPIError as e:
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
                    st.code(strip_ansi_codes(rich_output_buffer.getvalue()), language="text")

                except LLMUnexpectedError as e:
                    error_message = str(e)
                    user_advice = "An unexpected issue occurred with the LLM provider (e.g., network problem, malformed response). Please try again later."
                    
                    status.update(label=f"Socratic Debate Failed: {user_advice}", state="error", expanded=True)
                    st.error(f"**Error:** {user_advice}\n\n**Details:** {error_message}")
                    # Ensure final metrics are displayed even on error
                    total_tokens_placeholder.metric("Total Tokens Used", f"{intermediate_steps.get('Total_Tokens_Used', 0)}")
                    total_cost_placeholder.metric("Estimated Cost (USD)", f"{intermediate_steps.get('Total_Estimated_Cost_USD', '$0.0000')}")
                    st.code(strip_ansi_codes(rich_output_buffer.getvalue()), language="text")

                except Exception as e:
                    # Update status to error if an exception occurs
                    status.update(label=f"Socratic Debate Failed: {e}", state="error", expanded=True)
                    st.error(f"An unexpected error occurred during the process: {e}")
                    # Ensure final metrics are displayed even on error
                    total_tokens_placeholder.metric("Total Tokens Used", f"{intermediate_steps.get('Total_Tokens_Used', 0)}")
                    total_cost_placeholder.metric("Estimated Cost (USD)", f"{intermediate_steps.get('Total_Estimated_Cost_USD', '$0.0000')}")
                    st.code(strip_ansi_codes(rich_output_buffer.getvalue()), language="text") # Show logs even on error