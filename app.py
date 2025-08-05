# app.py
import streamlit as st
import json
import os
import io
import contextlib
import re
import datetime
from typing import Dict, Any, Optional, List
import yaml
import logging # <-- ADDED: This import was suggested by the LLM for the LLMOutputParser
from rich.console import Console
import core # Moved import to top for standard practice
from utils import parse_llm_code_output, validate_code_output, format_git_diff # Import from new utils.py

# --- Configuration Loading ---
@st.cache_resource
def load_config(file_path: str = "config.yaml") -> Dict[str, Any]:
    """Loads configuration from a YAML file with enhanced error handling."""
    try:
        with open(file_path, 'r') as f:
            config = yaml.safe_load(f)
        st.success(f"Configuration loaded successfully from {file_path}.") # User feedback
        return config
    except FileNotFoundError:
        st.error(f"Configuration file not found at '{file_path}'. Please ensure it exists.")
        # Removed: return {} # Return empty dict on error - this would hide the error message
    except yaml.YAMLError as e:
        st.error(f"Error parsing configuration file '{file_path}'. Please check its format: {e}")
        # Removed: return {}
    except IOError as e:
        st.error(f"IO error reading configuration file '{file_path}'. Check permissions: {e}")
        # Removed: return {}
    return {} # Explicitly return empty dict if any error occurs

app_config = load_config()
DOMAIN_KEYWORDS = app_config.get("domain_keywords", {})
CONTEXT_TOKEN_BUGET_RATIO = app_config.get("context_token_budget_ratio", 0.25) # Default fallback

# --- Demo Codebase Context Loading ---
@st.cache_data
def load_demo_codebase_context(file_path: str = "data/demo_codebase_context.json") -> Dict[str, str]:
    """Loads demo codebase context from a JSON file with enhanced error handling."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        st.error(f"Demo context file not found at '{file_path}'.")
        return {}
    except json.JSONDecodeError as e:
        st.error(f"Error decoding JSON from '{file_path}'. Please check its format: {e}")
        return {}
    except IOError as e:
        st.error(f"IO error reading demo context file '{file_path}'. Check permissions: {e}")
        return {}

# Redirect rich console output to a string buffer for Streamlit display
# Configure console to capture output, force terminal for ANSI codes, and use soft wrapping
@contextlib.contextmanager
def capture_rich_output_and_get_console():
    """Captures rich output (like Streamlit elements) and returns the captured content."""
    buffer = io.StringIO()
    # Use a specific console instance for capturing, ensuring it's configured correctly
    # force_terminal=True helps ensure ANSI codes are generated for rich formatting
    # soft_wrap=True prevents lines from being cut off prematurely
    console_instance = Console(file=buffer, force_terminal=True, soft_wrap=True)
    yield buffer, console_instance # Provide the buffer and console instance to the 'with' block
    # Cleanup is implicitly handled by the context manager exiting

# Helper function to strip ANSI escape codes from text
ansi_escape_re = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')
def strip_ansi_codes(text):
    return ansi_escape_re.sub('', text)

# --- LLM Output Parsing Class ---
class LLMOutputParser:
    """Handles parsing and validation of LLM-generated structured output."""
    def __init__(self, llm_provider: core.GeminiProvider): # Use core.GeminiProvider for type hinting
        self.llm_provider = llm_provider
        self.logger = logging.getLogger(self.__class__.__name__) # This line now works because 'logging' is imported

    def parse_and_validate(self, raw_output: str) -> Dict[str, Any]:
        """Parses raw LLM output, validates JSON structure, and handles escaping."""
        try:
            # Attempt to parse JSON
            parsed_json = json.loads(raw_output)
            self.logger.info("Successfully parsed LLM output as JSON.")

            # Basic validation of expected top-level keys
            if not all(k in parsed_json for k in ["COMMIT_MESSAGE", "RATIONALE", "CODE_CHANGES"]):
                self.logger.warning("Parsed JSON missing expected top-level keys.")
                # Decide how to handle: raise error, return partial, etc.
                # For now, we'll proceed but log a warning.

            # Validate CODE_CHANGES structure if present
            if "CODE_CHANGES" in parsed_json:
                if not isinstance(parsed_json["CODE_CHANGES"], list):
                    self.logger.error("CODE_CHANGES is not a list.")
                    raise ValueError("Invalid CODE_CHANGES format: must be a list.")
                
                for change in parsed_json["CODE_CHANGES"]:
                    if not isinstance(change, dict) or not all(k in change for k in ["file_path", "action"]):
                        self.logger.error(f"Invalid change item format: {change}")
                        raise ValueError("Invalid item in CODE_CHANGES: missing required keys.")
                    if change["action"] in ["ADD", "MODIFY"] and "full_content" not in change:
                        self.logger.error(f"Missing 'full_content' for ADD/MODIFY action: {change}")
                        raise ValueError(f"Missing 'full_content' for ADD/MODIFY action in change: {change}")
                    if change["action"] == "REMOVE" and "lines" not in change:
                        self.logger.error(f"Missing 'lines' for REMOVE action: {change}")
                        raise ValueError("Missing 'lines' for REMOVE action.")

            return parsed_json

        except json.JSONDecodeError as e:
            self.logger.error(f"Failed to decode JSON: {e}. Raw output:\n{raw_output[:500]}...")
            raise ValueError(f"LLM output is not valid JSON. Error: {e}") from e
        except ValueError as e:
            self.logger.error(f"JSON structure validation failed: {e}")
            raise e
        except Exception as e:
            self.logger.error(f"An unexpected error occurred during LLM output parsing: {e}")
            raise RuntimeError(f"Failed to process LLM output: {e}") from e

# --- Helper function for Markdown Report Generation ---
def generate_markdown_report(user_prompt: str, final_answer: str, intermediate_steps: Dict[str, Any], process_log_output: str, config_params: Dict[str, Any]) -> str:
    report_date = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    md_content = f"# Project Chimera Socratic Debate Report\n\n" # Added newline for spacing
    md_content += f"**Date:** {report_date}\n"
    md_content += f"**Original Prompt:** {user_prompt}\n\n"

    md_content += "---\n\n"
    md_content += "## Configuration\n\n"
    md_content += f"*   **Model:** {config_params.get('model_name', 'N/A')}\n"
    md_content += f"*   **Max Total Tokens Budget:** {config_params.get('max_tokens_budget', 'N/A')}\n"
    md_content += f"*   **Intermediate Steps Shown in UI:** {'Yes' if config_params.get('show_intermediate_steps', False) else 'No'}\n"
    md_content += f"*   **Reasoning Framework:** {config_params.get('domain', 'N/A')}\n"

    md_content += "---\n\n"
    md_content += "## Process Log\n\n"
    md_content += "```text\n"
    md_content += strip_ansi_codes(process_log_output)
    md_content += "\n```\n\n"
    
    if config_params.get('show_intermediate_steps', True):
        md_content += "---\n\n"
        md_content += "## Intermediate Reasoning Steps\n\n"
        # Filter keys to only include actual persona outputs and their token counts
        step_keys_to_process = sorted([k for k in intermediate_steps.keys()
                                       if not k.endswith("_Tokens_Used") and k != "Total_Tokens_Used" and k != "Total_Estimated_Cost_USD" and k != "debate_history"],
                                      key=lambda x: (x.split('_')[0], x)) # Sort by persona name, then type
        
        for step_key in step_keys_to_process:
            display_name = step_key.replace('_Output', '').replace('_Critique', '').replace('_Feedback', '').replace('_', ' ').title()
            content = intermediate_steps.get(step_key, "N/A")
            # Find the corresponding token count key
            token_base_name = step_key.replace("_Output", "").replace("_Critique", "").replace("_Feedback", "")
            token_count_key = f"{token_base_name}_Tokens_Used"
            tokens_used = intermediate_steps.get(token_count_key, "N/A")
            
            md_content += f"### {display_name}\n\n"
            md_content += f"```markdown\n{content}\n```\n"
            md_content += f"**Tokens Used for this step:** {tokens_used}\n\n"

    md_content += "---\n\n"
    md_content += "## Final Synthesized Answer\n\n"
    md_content += f"{final_answer}\n\n"

    md_content += "---\n\n"
    md_content += "## Summary\n\n"
    md_content += f"**Total Tokens Consumed:** {intermediate_steps.get('Total_Tokens_Used', 0):,}\n"
    md_content += f"**Total Estimated Cost:** ${intermediate_steps.get('Total_Estimated_Cost_USD', 0.0):.4f}\n"

    return md_content # Return the generated markdown content

st.set_page_config(layout="wide", page_title="Project Chimera Web App")

st.title("Project Chimera: Socratic Self-Debate")
st.markdown("An advanced reasoning engine for complex problem-solving and code generation. This project's core software is open-source and available on [GitHub](https://github.com/tomwolfe/project_chimera).")

EXAMPLE_PROMPTS = {
    "Design a Mars City": "Design a sustainable city for 1 million people on Mars, considering resource scarcity and human psychology.",
    "Ethical AI Framework": "Develop an ethical framework for an AI system designed to assist in judicial sentencing, addressing bias, transparency, and accountability.",
    "Implement Python API Endpoint": "Implement a new FastAPI endpoint `/items/{item_id}` that retrieves an item from a dictionary. Include basic error handling for non-existent items and add a corresponding unit test.",
    "Refactor a Python Function": "Refactor the given Python function to improve its readability and performance. It currently uses a nested loop; see if you can optimize it.",
    "Fix a Bug in a Script": "The provided Python script is supposed to calculate the average of a list of numbers but fails with a `TypeError` if the list contains non-numeric strings. Fix the bug by safely ignoring non-numeric values.",
    "Climate Change Solution": "Propose an innovative, scalable solution to mitigate the effects of climate change, focusing on a specific sector (e.g., energy, agriculture, transportation).",
}

def reset_app_state():
    """Resets all session state variables to their default values."""
    st.session_state.api_key_input = os.getenv("GEMINI_API_KEY", "")
    # Default to the first example prompt
    st.session_state.user_prompt_input = EXAMPLE_PROMPTS[list(EXAMPLE_PROMPTS.keys())[0]]
    st.session_state.max_tokens_budget_input = 1000000
    st.session_state.show_intermediate_steps_checkbox = True
    st.session_state.selected_model_selectbox = "gemini-2.5-flash-lite"
    # Reset to the first example key as the default selected example
    st.session_state.selected_example_name = list(EXAMPLE_PROMPTS.keys())[0]

    # Reset persona set to default based on loaded personas
    if "persona_sets" in st.session_state and "General" in st.session_state.persona_sets:
        st.session_state.selected_persona_set = "General"
    elif "available_domains" in st.session_state and st.session_state.available_domains:
        st.session_state.selected_persona_set = st.session_state.available_domains[0]
    else:
        st.session_state.available_domains = ["General"]
        st.session_state.selected_persona_set = "General"

    st.session_state.debate_ran = False
    st.session_state.final_answer_output = ""
    st.session_state.intermediate_steps_output = {}
    st.session_state.process_log_output_text = ""
    st.session_state.last_config_params = {}
    st.session_state.codebase_context = {}
    st.session_state.uploaded_files = [] # Clear uploaded files
    st.session_state.context_token_budget_ratio = CONTEXT_TOKEN_BUGET_RATIO # Reset ratio
    st.session_state.example_selector_widget = st.session_state.selected_example_name # Reset widget state
    st.session_state.selected_persona_set_widget = st.session_state.selected_persona_set # Reset widget state

    st.rerun() # Rerun to apply changes

# --- Session State Initialization ---
# Ensure all persona-related session state is initialized first and robustly
if "all_personas" not in st.session_state:
    # Load personas, persona sets, persona sequence, and the default persona set name
    try:
        all_personas, persona_sets, persona_sequence, default_persona_set_name = core.load_personas()
        st.session_state.all_personas = all_personas
        st.session_state.persona_sets = persona_sets
        st.session_state.persona_sequence = persona_sequence # Store the loaded persona sequence
        st.session_state.available_domains = list(persona_sets.keys())
        st.session_state.selected_persona_set = default_persona_set_name # Use the actual default persona set name
    except Exception as e:
        st.error(f"Failed to load personas from personas.yaml: {e}")
        st.session_state.all_personas = {} # Fallback to empty if load fails
        st.session_state.persona_sets = {}
        st.session_state.persona_sequence = [] # Fallback for sequence
        st.session_state.available_domains = ["General"]
        st.session_state.selected_persona_set = "General" # Fallback if loading fails

if "api_key_input" not in st.session_state:
    st.session_state.api_key_input = os.getenv("GEMINI_API_KEY", "")
if "user_prompt_input" not in st.session_state:
    st.session_state.user_prompt_input = EXAMPLE_PROMPTS[list(EXAMPLE_PROMPTS.keys())[0]]
if "max_tokens_budget_input" not in st.session_state:
    st.session_state.max_tokens_budget_input = 1000000
if "show_intermediate_steps_checkbox" not in st.session_state:
    st.session_state.show_intermediate_steps_checkbox = True
if "selected_model_selectbox" not in st.session_state:
    st.session_state.selected_model_selectbox = "gemini-2.5-flash-lite"
if "selected_example_name" not in st.session_state:
    st.session_state.selected_example_name = list(EXAMPLE_PROMPTS.keys())[0]
if "debate_ran" not in st.session_state:
    st.session_state.debate_ran = False
if "final_answer_output" not in st.session_state:
    st.session_state.final_answer_output = ""
if "intermediate_steps_output" not in st.session_state:
    st.session_state.intermediate_steps_output = {}
if "process_log_output_text" not in st.session_state:
    st.session_state.process_log_output_text = ""
if "last_config_params" not in st.session_state:
    st.session_state.last_config_params = {}
if "codebase_context" not in st.session_state:
    st.session_state.codebase_context = {}
if "uploaded_files" not in st.session_state: # Keep track of uploaded files
    st.session_state.uploaded_files = []
if "context_token_budget_ratio" not in st.session_state:
    st.session_state.context_token_budget_ratio = CONTEXT_TOKEN_BUGET_RATIO
# Initialize widget state keys if they don't exist
if "example_selector_widget" not in st.session_state:
    st.session_state.example_selector_widget = st.session_state.selected_example_name
if "selected_persona_set_widget" not in st.session_state:
    st.session_state.selected_persona_set_widget = st.session_state.selected_persona_set


# --- Sidebar for Configuration ---
with st.sidebar:
    st.header("Configuration")
    st.text_input("Enter your Gemini API Key", type="password", key="api_key_input", help="Your API key will not be stored.")
    st.markdown("Need a Gemini API key? Get one from [Google AI Studio](https://aistudio.google.com/apikey).")
    st.markdown("**Security Note:** Input sanitization is applied to mitigate prompt injection risks, but it is not foolproof against highly sophisticated adversarial attacks.")
    st.markdown("---")
    st.selectbox("Select LLM Model", ["gemini-2.5-flash-lite", "gemini-2.5-pro", "gemini-2.5-flash"], key="selected_model_selectbox")
    st.markdown("---")
    st.number_input("Max Total Tokens Budget:", min_value=1000, max_value=1000000, step=1000, key="max_tokens_budget_input")
    st.checkbox("Show Intermediate Reasoning Steps", key="show_intermediate_steps_checkbox")

# --- Main Content Area ---
st.header("Project Setup & Input")

api_key_feedback_placeholder = st.empty()
if not st.session_state.api_key_input.strip():
    api_key_feedback_placeholder.warning("Please enter your Gemini API Key in the sidebar to enable the 'Run' button.")

# --- Prompt Input ---
CUSTOM_PROMPT_KEY = "Custom Prompt"
SELECTBOX_PROMPT_OPTIONS = [CUSTOM_PROMPT_KEY] + list(EXAMPLE_PROMPTS.keys())

# Determine the initial index for the selectbox based on the current session state
current_example_index = 0
if st.session_state.selected_example_name == CUSTOM_PROMPT_KEY:
    current_example_index = SELECTBOX_PROMPT_OPTIONS.index(CUSTOM_PROMPT_KEY)
elif st.session_state.selected_example_name in EXAMPLE_PROMPTS:
    current_example_index = SELECTBOX_PROMPT_OPTIONS.index(st.session_state.selected_example_name)
else: # Fallback if session state somehow got corrupted
    st.session_state.selected_example_name = list(EXAMPLE_PROMPTS.keys())[0]
    current_example_index = SELECTBOX_PROMPT_OPTIONS.index(st.session_state.selected_example_name)

# Selectbox for example prompts
selected_option_from_widget = st.selectbox(
    "Choose an example prompt:",
    options=SELECTBOX_PROMPT_OPTIONS,
    index=current_example_index,
    key="example_selector_widget", # Static key for the widget
    help="Select a pre-defined prompt or choose 'Custom Prompt' to enter your own."
)

# --- Logic to handle changes in prompt selection or user input ---

# 1. If the selectbox value changed (user selected a different option)
if selected_option_from_widget != st.session_state.selected_example_name:
    st.session_state.selected_example_name = selected_option_from_widget # Update our internal state

    if st.session_state.selected_example_name == CUSTOM_PROMPT_KEY:
        st.session_state.user_prompt_input = "" # Clear prompt for custom input
        st.session_state.codebase_context = {} # Clear codebase context
        st.session_state.uploaded_files = [] # Clear uploaded files
        # Reset persona set if it was Software Engineering and we're moving to custom
        if st.session_state.selected_persona_set == "Software Engineering":
            st.session_state.selected_persona_set = "General"
    else:
        # Populate prompt from the selected example
        st.session_state.user_prompt_input = EXAMPLE_PROMPTS[st.session_state.selected_example_name]
        
        # Check if the selected example is a Software Engineering one and update context accordingly
        if st.session_state.selected_example_name in ["Implement Python API Endpoint", "Refactor a Python Function", "Fix a Bug in a Script"]:
            st.session_state.codebase_context = load_demo_codebase_context()
            # Simulate uploaded_files for UI consistency in the file uploader
            st.session_state.uploaded_files = [
                type('obj', (object,), {'name': k, 'size': len(v.encode('utf-8')), 'getvalue': lambda val=v: val.encode('utf-8')})()
                for k, v in st.session_state.codebase_context.items()
            ]
            st.session_state.selected_persona_set = "Software Engineering"
        else:
            # Clear context if not a SE prompt
            st.session_state.codebase_context = {}
            st.session_state.uploaded_files = []
            # Reset persona set if it was Software Engineering and we're moving to a non-SE example
            if st.session_state.selected_persona_set == "Software Engineering":
                st.session_state.selected_persona_set = "General"
                
    st.rerun() # Rerun to apply the changes immediately to the text area and context

# Text area for the user's prompt. It's bound to st.session_state.user_prompt_input.
user_prompt = st.text_area("Enter your prompt here:", height=150, key="user_prompt_input")


# --- Reasoning Framework & Context Input ---
col1, col2 = st.columns(2)

with col1:
    st.subheader("Reasoning Framework")
    def recommend_domain_from_keywords(prompt: str) -> Optional[str]:
        prompt_lower = prompt.lower()
        scores = {domain: sum(1 for keyword in keywords if keyword in prompt_lower) for domain, keywords in DOMAIN_KEYWORDS.items()} # Use DOMAIN_KEYWORDS from config
        if not any(scores.values()): return None
        return max(scores, key=scores.get)

    if user_prompt.strip():
        suggested_domain = recommend_domain_from_keywords(user_prompt)
        # Only show recommendation if it's different from current selection and valid
        if suggested_domain and suggested_domain != st.session_state.selected_persona_set and suggested_domain in st.session_state.available_domains:
            st.info(f"💡 Recommendation: Use the **'{suggested_domain}'** framework for this prompt.")
            if st.button(f"Apply '{suggested_domain}' Framework", type="primary", use_container_width=True):
                st.session_state.selected_persona_set = suggested_domain
                st.session_state.selected_persona_set_widget = suggested_domain # Update widget state too
                st.rerun() # Rerun to update the selectbox immediately

    st.selectbox(
        "Select Framework",
        options=st.session_state.available_domains,
        index=st.session_state.available_domains.index(st.session_state.selected_persona_set) if st.session_state.selected_persona_set in st.session_state.available_domains else 0,
        key="selected_persona_set_widget", # Use the widget key here
        help="Choose a domain-specific reasoning framework."
    )

    st.subheader("Context Budget")
    st.slider(
        "Context Token Budget Ratio", min_value=0.05, max_value=0.5, value=st.session_state.context_token_budget_ratio, # Access from session state
        step=0.05, key="context_token_budget_ratio", help="Percentage of total token budget allocated to context analysis."
    )

with col2:
    st.subheader("Codebase Context (Optional)")
    if st.session_state.selected_persona_set == "Software Engineering":
        uploaded_files = st.file_uploader(
            "Upload up to 25 relevant files",
            accept_multiple_files=True,
            type=['py', 'js', 'ts', 'html', 'css', 'json', 'yaml', 'md', 'txt', 'java', 'go', 'rb', 'php'],
            help="Provide files for context. The AI will analyze them to generate consistent code.",
            key="code_context_uploader" # Unique key for the uploader
        )
        
        # Logic to handle file uploads and update session state
        if uploaded_files: # User has actively uploaded files
            # Check if the uploaded files have changed since last run
            current_uploaded_file_info = [(f.name, f.size) for f in uploaded_files]
            previous_uploaded_file_info = [(f.name, f.size) for f in st.session_state.uploaded_files]

            if current_uploaded_file_info != previous_uploaded_file_info:
                if len(uploaded_files) > 25:
                    st.warning("Please upload a maximum of 25 files. Truncating to the first 25.")
                    uploaded_files = uploaded_files[:25] # Truncate if too many

                temp_context = {}
                for file in uploaded_files:
                    try:
                        content = file.getvalue().decode("utf-8")
                        temp_context[file.name] = content
                    except Exception as e:
                        st.error(f"Error reading {file.name}: {e}")
                st.session_state.codebase_context = temp_context
                st.session_state.uploaded_files = uploaded_files # Store the actual uploaded file objects for comparison
                st.toast(f"{len(st.session_state.codebase_context)} file(s) loaded for context.")
        elif st.session_state.codebase_context and not uploaded_files: # Only show if context exists and no new files were uploaded
            # No new files uploaded, but context exists (e.g., from demo load or previous upload)
            st.success(f"{len(st.session_state.codebase_context)} file(s) already loaded for context.")
        else:
            # Uploader is empty and no context exists in session state
            st.session_state.codebase_context = {}
            st.session_state.uploaded_files = [] # Ensure this is empty
    else:
        st.info("Select the 'Software Engineering' framework to provide codebase context.")
        st.session_state.codebase_context = {}
        st.session_state.uploaded_files = [] # Clear uploaded files if domain changes


# --- Run and Reset Buttons ---
st.markdown("---")
run_col, reset_col = st.columns([0.8, 0.2])
with run_col:
    run_button_clicked = st.button("🚀 Run Socratic Debate", type="primary", use_container_width=True)
with reset_col:
    st.button("🔄 Reset All", on_click=reset_app_state, use_container_width=True)

# --- Execution Logic ---
if run_button_clicked:
    api_key_feedback_placeholder.empty()
    if not st.session_state.api_key_input.strip():
        st.error("Please enter your Gemini API Key in the sidebar to proceed.")
    elif not user_prompt.strip():
        st.error("Please enter a prompt.")
    else:
        st.session_state.debate_ran = False
        with st.status("Initializing Socratic Debate...", expanded=True) as status:
            # Placeholders for real-time metrics
            st.markdown("---")
            metric_col1, metric_col2, metric_col3 = st.columns(3)
            total_tokens_placeholder = metric_col1.empty()
            total_cost_placeholder = metric_col2.empty()
            next_step_warning_placeholder = metric_col3.empty()
            st.markdown("---")

            def streamlit_status_callback(message: str, state: str = "running", expanded: bool = True,
                                          current_total_tokens: int = 0, current_total_cost: float = 0.0,
                                          estimated_next_step_tokens: int = 0, estimated_next_step_cost: float = 0.0):
                status.update(label=message, state=state, expanded=expanded)
                total_tokens_placeholder.metric("Total Tokens Used", f"{current_total_tokens:,}")
                total_cost_placeholder.metric("Estimated Cost (USD)", f"${current_total_cost:.4f}")

                if estimated_next_step_tokens > 0:
                    budget_remaining = st.session_state.max_tokens_budget_input - current_total_tokens
                    if estimated_next_step_tokens > budget_remaining:
                        next_step_warning_placeholder.warning(
                            f"⚠️ Next step ({estimated_next_step_tokens:,} tokens) "
                            f"will exceed budget ({budget_remaining:,} remaining). "
                            f"Estimated cost: ${estimated_next_step_cost:.4f}"
                        )
                    else:
                        next_step_warning_placeholder.info(
                            f"Next step estimated: {estimated_next_step_tokens:,} tokens "
                            f"(${(estimated_next_step_cost):.4f}). "
                            f"Budget remaining: {budget_remaining:,} tokens."
                        )
                else:
                    next_step_warning_placeholder.empty()

            debate_instance = None
            try:
                all_personas = st.session_state.all_personas
                persona_sets = st.session_state.persona_sets
                domain_for_run = st.session_state.selected_persona_set
                personas_for_run = {name: all_personas[name] for name in persona_sets.get(domain_for_run, [])} # Use .get for safety

                # Instantiate GeminiProvider with the correct model name from session state
                gemini_provider_instance = core.GeminiProvider( # Use core.GeminiProvider
                    api_key=st.session_state.api_key_input,
                    model_name=st.session_state.selected_model_selectbox,
                    _status_callback=streamlit_status_callback # Corrected parameter name
                )

                # Capture rich console output for the log display
                with capture_rich_output_and_get_console() as (rich_output_buffer, rich_console_instance):
                    # Instantiate SocraticDebate
                    debate_instance = core.SocraticDebate( # Use core.SocraticDebate
                        initial_prompt=user_prompt,
                        api_key=st.session_state.api_key_input,
                        max_total_tokens_budget=st.session_state.max_tokens_budget_input,
                        model_name=st.session_state.selected_model_selectbox,
                        personas=personas_for_run,
                        all_personas=all_personas,
                        persona_sequence=st.session_state.persona_sequence, # Pass the loaded persona sequence
                        persona_sets=persona_sets,
                        domain=domain_for_run,
                        gemini_provider=gemini_provider_instance, # Pass the instantiated provider
                        status_callback=streamlit_status_callback,
                        rich_console=rich_console_instance,
                        codebase_context=st.session_state.get('codebase_context', {}),
                        context_token_budget_ratio=st.session_state.context_token_budget_ratio # Pass the configurable ratio
                    )
                    
                    final_answer, intermediate_steps = debate_instance.run_debate()

                    st.session_state.process_log_output_text = rich_output_buffer.getvalue()
                    st.session_state.final_answer_output = final_answer
                    st.session_state.intermediate_steps_output = intermediate_steps
                    st.session_state.last_config_params = {
                        "max_tokens_budget": st.session_state.max_tokens_budget_input,
                        "model_name": st.session_state.selected_model_selectbox, # Model name is correctly captured
                        "show_intermediate_steps": st.session_state.show_intermediate_steps_checkbox,
                        "domain": domain_for_run
                    }
                    st.session_state.debate_ran = True
                    status.update(label="Socratic Debate Complete!", state="complete", expanded=False)

                    final_total_tokens = intermediate_steps.get('Total_Tokens_Used', 0)
                    final_total_cost = intermediate_steps.get('Total_Estimated_Cost_USD', 0.0)
                    total_tokens_placeholder.metric("Total Tokens Used", f"{final_total_tokens:,}")
                    total_cost_placeholder.metric("Estimated Cost (USD)", f"${final_total_cost:.4f}")
                    next_step_warning_placeholder.empty()

            except (core.TokenBudgetExceededError, Exception) as e: # Use core.TokenBudgetExceededError
                # Ensure process log is captured even on error
                st.session_state.process_log_output_text = rich_output_buffer.getvalue() if 'rich_output_buffer' in locals() else ""
                status.update(label=f"Socratic Debate Failed: {e}", state="error", expanded=True)
                st.error(f"**Error:** {e}")
                st.session_state.debate_ran = True
                if debate_instance:
                    st.session_state.intermediate_steps_output = debate_instance.intermediate_steps
                    # --- FIX FOR "Malformed Output Detected" ---
                    # Ensure final_answer is populated with an error message if the debate failed
                    if not st.session_state.final_answer_output or "Process did not complete" in st.session_state.final_answer_output:
                        st.session_state.final_answer_output = f"Error during debate: {e}"
                    # --- END FIX ---
                # Update metrics even on error if debate_instance exists
                total_tokens_placeholder.metric("Total Tokens Used", f"{debate_instance.intermediate_steps.get('Total_Tokens_Used', 0):,}" if debate_instance else "N/A")
                total_cost_placeholder.metric("Estimated Cost (USD)", f"${debate_instance.intermediate_steps.get('Total_Estimated_Cost_USD', 0.0):.4f}" if debate_instance else "N/A")
                next_step_warning_placeholder.empty()

# --- Results Display ---
if st.session_state.debate_ran:
    st.markdown("---")
    st.header("Results")

    # Handle Software Engineering output specifically
    if st.session_state.last_config_params.get("domain") == "Software Engineering":
        raw_output = st.session_state.final_answer_output
        
        # Use the new LLMOutputParser for parsing and validation
        # Instantiate parser with the provider used for the debate
        # Ensure gemini_provider_instance is accessible here, or re-instantiate if necessary
        # For simplicity, we assume it's available in the scope. If not, it needs to be passed or recreated.
        # A safer approach might be to pass it from the 'try' block or ensure it's stored in session state.
        # For now, assuming gemini_provider_instance is still valid or accessible.
        # If gemini_provider_instance is not defined here, it means the 'try' block failed before its creation.
        # In that case, we should handle it.
        
        parser = None
        if 'gemini_provider_instance' in locals() and gemini_provider_instance:
            parser = LLMOutputParser(gemini_provider_instance)
        else:
            # Fallback if gemini_provider_instance wasn't created (e.g., due to an earlier error)
            # This might require a dummy provider or re-instantiation with API key from session state
            try:
                fallback_provider = core.GeminiProvider( # Use core.GeminiProvider
                    api_key=st.session_state.api_key_input,
                    model_name=st.session_state.selected_model_selectbox,
                    _status_callback=None # No status callback for parser instantiation if not running debate
                )
                parser = LLMOutputParser(fallback_provider)
                st.warning("Re-instantiated GeminiProvider for parsing as the original instance was not available.")
            except Exception as e:
                st.error(f"Could not instantiate LLMOutputParser due to missing provider: {e}")
                # Handle this case gracefully, perhaps by skipping the parsing section
                parser = None # Ensure parser is None if instantiation fails

        validation_results = {'issues': [], 'malformed_blocks': []}
        parsed_data = {'commit_message': 'Not generated.', 'rationale': 'Not generated.', 'code_changes': []} # Default to empty structure

        if parser:
            try:
                parsed_data = parser.parse_and_validate(raw_output)
                # Use the existing validate_code_output from utils.py for detailed validation
                # Ensure codebase_context is correctly retrieved from session state
                validation_results = validate_code_output_batch(parsed_data, st.session_state.get('codebase_context', {})) 
            except (ValueError, RuntimeError) as e:
                # If parsing fails, capture the error and mark blocks as malformed
                # The error message from parse_and_validate is already descriptive
                validation_results['malformed_blocks'].append(f"Error parsing LLM output: {e}\nRaw Output:\n{raw_output}")
                # parsed_data remains default empty structure, ensuring "Not generated" is shown

        # --- Structured Summary ---
        st.subheader("Structured Summary")
        summary_col1, summary_col2 = st.columns(2)
        with summary_col1:
            st.markdown("**Commit Message Suggestion**")
            # --- FIX: Access top-level key directly ---
            st.code(parsed_data.get('commit_message', 'Not generated.'), language='text')
            # --- END FIX ---
        with summary_col2:
            st.markdown("**Token Usage**")
            total_tokens = st.session_state.intermediate_steps_output.get('Total_Tokens_Used', 0)
            total_cost = st.session_state.intermediate_steps_output.get('Total_Estimated_Cost_USD', 0.0)
            st.metric("Total Tokens Consumed", f"{total_tokens:,}")
            st.metric("Total Estimated Cost (USD)", f"${total_cost:.4f}")
        
        st.markdown("**Rationale**")
        # --- FIX: Access top-level key directly ---
        st.markdown(parsed_data.get('rationale', 'Not generated.'))
        # --- END FIX ---

        if parsed_data.get('conflict_resolution'):
            st.markdown("**Conflict Resolution**")
            st.info(parsed_data['conflict_resolution'])
        if parsed_data.get('unresolved_conflict'):
            st.markdown("**Unresolved Conflict**")
            st.warning(parsed_data['unresolved_conflict'])

        # --- Validation Report ---
        with st.expander("🔍 Validation & Quality Report", expanded=True):
            if not validation_results['issues'] and not validation_results['malformed_blocks']:
                st.success("✅ No syntax, style, or formatting issues detected.")
            else:
                for issue in validation_results['issues']:
                    st.warning(f"**{issue['type']} in `{issue['file']}`:** {issue['message']} (Line: {issue.get('line', 'N/A')})")
                if validation_results['malformed_blocks']:
                     st.error(f"**Malformed Output Detected:** The LLM produced {len(validation_results['malformed_blocks'])} block(s) that could not be parsed. The raw output is provided as a fallback.")

        # --- Proposed Code Changes ---
        st.subheader("Proposed Code Changes")
        if not parsed_data.get('code_changes') and not validation_results['malformed_blocks']:
            st.info("No code changes were proposed.")
        
        # Iterate over the list of changes
        for change in parsed_data.get('code_changes', []):
            with st.expander(f"📄 **{change.get('file_path', 'N/A')}** (`{change.get('action', 'N/A')}`)", expanded=False): # Changed expanded to False by default
                st.write(f"**Action:** {change.get('action')}")
                st.write(f"**File Path:** {change.get('file_path')}")
                
                if change.get('action') in ['ADD', 'MODIFY']:
                    st.write("**Content:**")
                    # Use 'python' as default language, but could be dynamic based on file_path (e.g., based on extension)
                    # Truncate for display if content is very long
                    display_content = change.get('full_content', '')
                    st.code(display_content[:1000] + ('...' if len(display_content) > 1000 else ''), language='python') 
                elif change.get('action') == 'REMOVE':
                    st.write("**Lines to Remove:**")
                    st.write(change.get('lines', []))
        
        # Display malformed blocks as fallbacks if any were detected
        for block in validation_results['malformed_blocks']:
            with st.expander(f"📄 **Unknown File (Malformed Block)**", expanded=True):
                st.error("This block was malformed and could not be parsed correctly. Raw output is shown below.")
                st.code(block, language='text')

    # Handle all other frameworks
    else:
        st.subheader("Final Synthesized Answer")
        st.markdown(st.session_state.final_answer_output)

    # --- Intermediate Steps & Log ---
    with st.expander("Show Intermediate Steps & Process Log"):
        if st.session_state.show_intermediate_steps_checkbox:
            st.subheader("Intermediate Reasoning Steps")
            # Filter and sort intermediate steps for display
            display_steps = {k: v for k, v in st.session_state.intermediate_steps_output.items() 
                             if not k.endswith("_Tokens_Used") and k != "Total_Tokens_Used" and k != "Total_Estimated_Cost_USD" and k != "debate_history"}
            
            # Sort keys for consistent display order
            sorted_step_keys = sorted(display_steps.keys(), key=lambda x: (x.split('_')[0], x)) # Sort by persona name, then type

            for step_key in sorted_step_keys:
                persona_name = step_key.split('_')[0] # Extract persona name
                display_name = step_key.replace('_Output', '').replace('_Critique', '').replace('_Feedback', '').replace('_', ' ').title()
                content = display_steps.get(step_key, "N/A")
                
                # Find the corresponding token count key
                cleaned_step_key = step_key.replace("_Output", "").replace("_Critique", "").replace("_Feedback", "")
                token_count_key = f"{cleaned_step_key}_Tokens_Used"
                tokens_used = st.session_state.intermediate_steps_output.get(token_count_key, "N/A")
                
                with st.expander(f"**{display_name}** (Tokens: {tokens_used})"):
                    st.markdown(f"```markdown\n{content}\n```")
        
        st.subheader("Process Log")
        st.code(strip_ansi_codes(st.session_state.process_log_output_text), language="text")

    # --- Export Functionality ---
    st.markdown("---")
    st.subheader("Export Results")

    final_answer_md = f"# Final Synthesized Answer\n\n{st.session_state.final_answer_output}"
    st.download_button(
        label="Download Final Answer (Markdown)",
        data=final_answer_md,
        file_name="final_answer.md",
        mime="text/markdown"
    )

    full_report_md = generate_markdown_report(
        user_prompt=user_prompt,
        final_answer=st.session_state.final_answer_output,
        intermediate_steps=st.session_state.intermediate_steps_output, # Pass the intermediate steps
        process_log_output=st.session_state.process_log_output_text,
        config_params=st.session_state.last_config_params
    )
    st.download_button(
        label="Download Full Report (Markdown)",
        data=full_report_md,
        file_name="socratic_debate_report.md",
        mime="text/markdown"
    )
    st.info("To generate a PDF, download the Markdown report and use your browser's 'Print to PDF' option (usually accessible via Ctrl+P or Cmd+P).")