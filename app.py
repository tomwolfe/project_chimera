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

# Prompt Input
user_prompt = st.text_area(
    "Enter your prompt here:",
    "Design a sustainable city for 1 million people on Mars, considering resource scarcity and human psychology.",
    height=150
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
            # Use the context manager to capture rich output for the log display
            with capture_rich_output() as rich_output_buffer:
                try:
                    final_answer, intermediate_steps = run_isal_process(
                        user_prompt, api_key, max_total_tokens_budget=max_tokens_budget,
                        streamlit_status=status, # Pass the status object for granular updates
                        model_name=selected_model # Pass the selected model name
                    )
                    
                    # Update status to complete after successful execution
                    status.update(label="Socratic Debate Complete!", state="complete", expanded=False)

                    # Display captured console output (e.g., "Running persona: ...")
                    st.subheader("Process Log")
                    st.code(rich_output_buffer.getvalue(), language="text")

                    if show_intermediate_steps:
                        st.subheader("Intermediate Reasoning Steps")
                        for step_name, content in intermediate_steps.items():
                            with st.expander(f"### {step_name.replace('_', ' ').title()}"):
                                # Use Streamlit's markdown or code block for display
                                # Check for the special token count step
                                if step_name.endswith("_Tokens_Used"): # New: Per-step token display
                                    original_step_name = step_name.replace("_Tokens_Used", "")
                                    st.write(f"Tokens used for '{original_step_name.replace('_', ' ').title()}': {content}")
                                elif step_name == "Total_Tokens_Used":
                                    st.write(f"Total tokens consumed: {content}")
                                elif "Output" in step_name or "Critique" in step_name or "Feedback" in step_name or "[ERROR]" in content:
                                    st.code(content, language="markdown")
                                else:
                                    st.write(content)

                    st.subheader("Final Synthesized Answer")
                    st.markdown(final_answer) # Use markdown for final answer
                    
                except TokenBudgetExceededError as e: # Catch the new specific error
                    error_message = str(e)
                    user_advice = "The process was stopped because it would exceed the maximum token budget."
                    status.update(label=f"Socratic Debate Failed: {user_advice}", state="error", expanded=True)
                    st.error(f"**Error:** {user_advice}\n\n**Details:** {error_message}")
                    st.code(rich_output_buffer.getvalue(), language="text")

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
                    st.code(rich_output_buffer.getvalue(), language="text")

                except LLMUnexpectedError as e:
                    error_message = str(e)
                    user_advice = "An unexpected issue occurred with the LLM provider (e.g., network problem, malformed response). Please try again later."
                    
                    status.update(label=f"Socratic Debate Failed: {user_advice}", state="error", expanded=True)
                    st.error(f"**Error:** {user_advice}\n\n**Details:** {error_message}")
                    st.code(rich_output_buffer.getvalue(), language="text")

                except Exception as e:
                    # Update status to error if an exception occurs
                    status.update(label=f"Socratic Debate Failed: {e}", state="error", expanded=True)
                    st.error(f"An unexpected error occurred during the process: {e}")
                    st.code(rich_output_buffer.getvalue(), language="text") # Show logs even on error