# -*- coding: utf-8 -*-
# app.py
import streamlit as st
import json
import os
import io
import contextlib
import re
import datetime
from typing import Dict, Any, List, Optional
import yaml
import logging
from rich.console import Console
import core
# Import models from src.models for consistency
from src.models import PersonaConfig, ReasoningFrameworkConfig, LLMOutput, ContextAnalysisOutput
from src.utils import LLMOutputParser, validate_code_output_batch # Import LLMOutputParser
# Import PersonaManager
from src.persona_manager import PersonaManager
# Import ConfigPersistence for unified configuration management
from src.config.persistence import ConfigPersistence
# Import custom exceptions from core.py (or a dedicated exceptions module if refactored)
from src.exceptions import ChimeraError, LLMResponseValidationError, TokenBudgetExceededError, SchemaValidationError # Added SchemaValidationError import
# --- ADDED IMPORT FOR SUGGESTION 1 ---
from src.constants import SELF_ANALYSIS_KEYWORDS
# --- END ADDED IMPORT ---

import traceback # Needed for error handling in app.py

# Configure logging for app.py
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Configuration Loading ---
@st.cache_resource
def load_config(file_path: str = "config.yaml") -> Dict[str, Any]:
    """Loads configuration from a YAML file with enhanced error handling."""
    try:
        with open(file_path, 'r') as f:
            config = yaml.safe_load(f)
        st.success(f"Configuration loaded successfully from {file_path}.")
        return config
    except FileNotFoundError:
        st.error(f"Configuration file not found at '{file_path}'. Please ensure it exists.")
    except yaml.YAMLError as e:
        st.error(f"Error parsing configuration file '{file_path}'. Please check its format: {e}")
    except IOError as e:
        st.error(f"IO error reading configuration file '{file_path}'. Check permissions: {e}")
    return {}
app_config = load_config()
DOMAIN_KEYWORDS = app_config.get("domain_keywords", {})
CONTEXT_TOKEN_BUDGET_RATIO = app_config.get("context_token_budget_ratio", 0.25)

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
@contextlib.contextmanager
def capture_rich_output_and_get_console():
    """Captures rich output (like Streamlit elements) and returns the captured content."""
    buffer = io.StringIO()
    console_instance = Console(file=buffer, force_terminal=True, soft_wrap=True)
    yield buffer, console_instance

ansi_escape_re = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')
def strip_ansi_codes(text):
    return ansi_escape_re.sub('', text)

# --- Helper function for Markdown Report Generation ---
def generate_markdown_report(user_prompt: str, final_answer: Any, intermediate_steps: Dict[str, Any], process_log_output: str, config_params: Dict[str, Any], persona_audit_log: List[Dict[str, Any]]) -> str:
    report_date = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    md_content = f"# Project Chimera Socratic Debate Report\n\n"
    md_content += f"**Date:** {report_date}\n"
    md_content += f"**Original Prompt:** {user_prompt}\n\n"
    md_content += "---\n\n"
    md_content += "## Configuration\n\n"
    md_content += f"*   **Model:** {config_params.get('model_name', 'N/A')}\n"
    md_content += f"*   **Max Total Tokens Budget:** {config_params.get('max_tokens_budget', 'N/A')}\n"
    md_content += f"*   **Intermediate Steps Shown in UI:** {'Yes' if config_params.get('show_intermediate_steps', False) else 'No'}\n"
    md_content += f"*   **Reasoning Framework:** {config_params.get('domain', 'N/A')}\n"
    md_content += "---\n\n"

    if persona_audit_log:
        md_content += "## Persona Configuration Audit Trail (Current Session)\n\n"
        md_content += "| Timestamp | Persona | Parameter | Old Value | New Value |\n"
        md_content += "|---|---|---|---|---|\n"
        for entry in persona_audit_log:
            old_val = str(entry.get('old_value')).replace('\n', '\\n')[:50] + '...' if len(str(entry.get('old_value'))) > 50 else str(entry.get('old_value')).replace('\n', '\\n')
            new_val = str(entry.get('new_value')).replace('\n', '\\n')[:50] + '...' if len(str(entry.get('new_value'))) > 50 else str(entry.get('new_value')).replace('\n', '\\n')
            md_content += f"| {entry.get('timestamp')} | {entry.get('persona')} | {entry.get('parameter')} | `{old_val}` | `{new_val}` |\n"
        md_content += "\n---\n\n"

    md_content += "## Process Log\n\n"
    md_content += "```text\n"
    md_content += strip_ansi_codes(process_log_output)
    md_content += "\n```\n\n"
    
    if config_params.get('show_intermediate_steps', True):
        md_content += "---\n\n"
        md_content += "## Intermediate Reasoning Steps\n\n"
        step_keys_to_process = sorted([k for k in intermediate_steps.keys()
                                       if not k.endswith("_Tokens_Used") and k != "Total_Tokens_Used" and k != "Total_Estimated_Cost_USD" and k != "debate_history" and not k.startswith("Final_Answer_") and not k.startswith("Context_Analysis") and not k.startswith("Persona_Sequence") and not k.startswith("Partial_Result_Warning")],
                                      key=lambda x: (x.split('_')[0], x))
        
        for step_key in step_keys_to_process:
            display_name = step_key.replace('_Output', '').replace('_Critique', '').replace('_Feedback', '').replace('_', ' ').title()
            content = intermediate_steps.get(step_key, "N/A")
            token_base_name = step_key.replace("_Output", "").replace("_Critique", "").replace("_Feedback", "")
            token_count_key = f"{token_base_name}_Tokens_Used"
            tokens_used = intermediate_steps.get(token_count_key, "N/A")
            
            md_content += f"### {display_name}\n\n"
            # MODIFIED: Handle final_answer which might be a dict or string
            if isinstance(content, dict):
                md_content += "```json\n"
                md_content += json.dumps(content, indent=2)
                md_content += "\n```\n"
            else:
                md_content += f"```markdown\n{content}\n```\n"
            md_content += f"**Tokens Used for this step:** {tokens_used}\n\n"
    md_content += "---\n\n"
    md_content += "## Final Synthesized Answer\n\n"
    # MODIFIED: Handle final_answer which might be a dict or string
    if isinstance(final_answer, dict):
        md_content += "```json\n"
        md_content += json.dumps(final_answer, indent=2)
        md_content += "\n```\n\n"
    else:
        md_content += f"{final_answer}\n\n"
    md_content += "---\n\n"
    md_content += "## Summary\n\n"
    md_content += f"**Total Tokens Consumed:** {intermediate_steps.get('Total_Tokens_Used', 0):,}\n"
    md_content += f"**Total Estimated Cost:** ${intermediate_steps.get('Total_Estimated_Cost_USD', 0.0):.4f}\n"
    
    # Add phase breakdown to report
    if "Tokens_Used_Per_Phase" in intermediate_steps:
        md_content += "\n### Token Usage Breakdown by Phase:\n"
        for phase, tokens in intermediate_steps["Tokens_Used_Per_Phase"].items():
            budget = intermediate_steps.get("phase_budgets", {}).get(phase, "N/A")
            md_content += f"*   **{phase.capitalize()}**: {tokens:,} tokens (Budget: {budget:,})\n"
            
    return md_content

st.set_page_config(layout="wide", page_title="Project Chimera Web App")
st.title("Project Chimera: Socratic Self-Debate")
st.markdown("An advanced reasoning engine for complex problem-solving and code generation. This project's core software is open-source and available on [GitHub](https://github.com/tomwolfe/project_chimera).")

EXAMPLE_PROMPTS = {
    "Design a Mars City": "Design a sustainable city for 1 million people on Mars, considering resource scarcity and human psychology.",
    "Ethical AI Framework": "Develop an ethical framework for an AI system designed to assist in judicial sentencing, addressing bias, transparency, and accountability.",
    "Implement Python API Endpoint": "Implement a new FastAPI endpoint `/items/{item_id}` that retrieves an item from a dictionary. Include basic error handling for non-existent items and add a corresponding unit test.",
    "Refactor a Python Function": "Refactor the given Python function to improve its readability and performance. It currently uses a nested loop; see if you can optimize it.",
    "Fix a Bug in a Script": "The provided Python script is supposed to calculate the average of a list of numbers but fails with a `TypeError` if the list contains non-numeric strings. Fix the bug by safely ignoring non-numeric values.",
    "Critically analyze the entire Project Chimera codebase. Identify the most impactful code changes for self-improvement, focusing on the 80/20 Pareto principle. Prioritize enhancements to reasoning quality, robustness, efficiency, and developer maintainability. For each suggestion, provide a clear rationale and a specific, actionable code modification.": "Critically analyze the entire Project Chimera codebase. Identify the most impactful code changes for self-improvement, focusing on the 80/20 Pareto principle. Prioritize enhancements to reasoning quality, robustness, efficiency, and developer maintainability. For each suggestion, provide a clear rationale and a specific, actionable code modification.",
    "Climate Change Solution": "Propose an innovative, scalable solution to mitigate the effects of climate change, focusing on a specific sector (e.g., energy, agriculture, transportation).",
}

# Initialize PersonaManager once (it's cached by st.cache_resource)
persona_manager_instance = PersonaManager()

# --- Session State Initialization ---
# Initialize context_token_budget_ratio unconditionally at the top level
st.session_state.context_token_budget_ratio = CONTEXT_TOKEN_BUDGET_RATIO

def _initialize_session_state(pm: PersonaManager):
    """Initializes or resets all session state variables to their default values."""
    st.session_state.api_key_input = os.getenv("GEMINI_API_KEY", "")
    st.session_state.user_prompt_input = EXAMPLE_PROMPTS[list(EXAMPLE_PROMPTS.keys())[0]]
    st.session_state.max_tokens_budget_input = 1000000
    st.session_state.show_intermediate_steps_checkbox = True
    st.session_state.selected_model_selectbox = "gemini-2.5-flash-lite"
    st.session_state.selected_example_name = list(EXAMPLE_PROMPTS.keys())[0]
    
    # Initialize persona-related states from PersonaManager
    st.session_state.persona_manager = pm # Store the cached instance
    # Load initial personas and sets from PersonaManager
    st.session_state.all_personas = pm.all_personas
    st.session_state.persona_sets = pm.persona_sets
    st.session_state.persona_sequence = pm.persona_sequence
    st.session_state.available_domains = pm.available_domains # This should include custom frameworks loaded by PersonaManager
    st.session_state.selected_persona_set = pm.default_persona_set_name

    st.session_state.debate_ran = False
    st.session_state.final_answer_output = ""
    st.session_state.intermediate_steps_output = {}
    st.session_state.process_log_output_text = ""
    st.session_state.last_config_params = {}
    st.session_state.codebase_context = {}
    st.session_state.uploaded_files = []
    # st.session_state.context_token_budget_ratio is already initialized above
    st.session_state.example_selector_widget = st.session_state.selected_example_name
    st.session_state.selected_persona_set_widget = st.session_state.selected_persona_set
    st.session_state.persona_audit_log = [] # Reset audit log
    st.session_state.persona_edit_mode = False # Reset edit mode
    
    # Framework saving/loading state
    st.session_state.save_framework_input = ""
    st.session_state.framework_description = "" # For saving new frameworks
    st.session_state.load_framework_select = "" # For loading frameworks

# Initialize session state if not already done
if "api_key_input" not in st.session_state: # Check for any key to determine if state is initialized
    _initialize_session_state(persona_manager_instance)

def reset_app_state():
    """Resets all session state variables to their default values."""
    _initialize_session_state(st.session_state.persona_manager)
    st.rerun()

# --- Persona Change Logging ---
def _log_persona_change(persona_name: str, parameter: str, old_value: Any, new_value: Any):
    """Logs a change to a persona parameter in the session audit log."""
    st.session_state.persona_audit_log.append({
        "timestamp": datetime.datetime.now().isoformat(),
        "persona": persona_name,
        "parameter": parameter,
        "old_value": old_value,
        "new_value": new_value
    })

with st.sidebar:
    st.header("Configuration")
    st.text_input("Enter your Gemini API Key", type="password", key="api_key_input", help="Your API key will not be stored.")
    st.markdown("Need a Gemini API key? Get one from [Google AI Studio](https://aistudio.google.com/apikey).")
    st.markdown("---")
    st.markdown("Security Note: Input sanitization is applied to mitigate prompt injection risks, but it is not foolproof against highly sophisticated adversarial attacks.")
    st.markdown("---")
    st.selectbox("Select LLM Model", ["gemini-2.5-flash-lite", "gemini-2.5-pro", "gemini-2.5-flash"], key="selected_model_selectbox")
    st.markdown("---")
    st.number_input("Max Total Tokens Budget:", min_value=1000, max_value=1000000, step=1000, key="max_tokens_budget_input")
    st.checkbox("Show Intermediate Reasoning Steps", key="show_intermediate_steps_checkbox")

st.header("Project Setup & Input")
api_key_feedback_placeholder = st.empty()
if not st.session_state.api_key_input.strip():
    api_key_feedback_placeholder.warning("Please enter your Gemini API Key in the sidebar to enable the 'Run' button.")

CUSTOM_PROMPT_KEY = "Custom Prompt"
SELECTBOX_PROMPT_OPTIONS = [CUSTOM_PROMPT_KEY] + list(EXAMPLE_PROMPTS.keys())
current_example_index = 0
if st.session_state.selected_example_name == CUSTOM_PROMPT_KEY:
    current_example_index = SELECTBOX_PROMPT_OPTIONS.index(CUSTOM_PROMPT_KEY)
elif st.session_state.selected_example_name in EXAMPLE_PROMPTS:
    current_example_index = SELECTBOX_PROMPT_OPTIONS.index(st.session_state.selected_example_name)
else:
    st.session_state.selected_example_name = list(EXAMPLE_PROMPTS.keys())[0]
    current_example_index = SELECTBOX_PROMPT_OPTIONS.index(st.session_state.selected_example_name)

selected_option_from_widget = st.selectbox(
    "Choose an example prompt:",
    options=SELECTBOX_PROMPT_OPTIONS,
    index=current_example_index,
    key="example_selector_widget",
    help="Select a pre-defined prompt or choose 'Custom Prompt' to enter your own."
)

if selected_option_from_widget != st.session_state.selected_example_name:
    st.session_state.selected_example_name = selected_option_from_widget
    if st.session_state.selected_example_name == CUSTOM_PROMPT_KEY:
        st.session_state.user_prompt_input = ""
        st.session_state.codebase_context = {}
        st.session_state.uploaded_files = []
        if st.session_state.selected_persona_set == "Software Engineering":
            st.session_state.selected_persona_set = "General"
    else:
        st.session_state.user_prompt_input = EXAMPLE_PROMPTS[st.session_state.selected_example_name]
        if st.session_state.selected_example_name in ["Implement Python API Endpoint", "Refactor a Python Function", "Fix a Bug in a Script", "Critically analyze the entire Project Chimera codebase. Identify the most impactful code changes for self-improvement, focusing on the 80/20 Pareto principle. Prioritize enhancements to reasoning quality, robustness, efficiency, and developer maintainability. For each suggestion, provide a clear rationale and a specific, actionable code modification."]: # Added the new prompt to trigger SE framework
            st.session_state.codebase_context = load_demo_codebase_context()
            st.session_state.uploaded_files = [
                type('obj', (object,), {'name': k, 'size': len(v.encode('utf-8')), 'getvalue': lambda val=v: val.encode('utf-8')})()
                for k, v in st.session_state.codebase_context.items()
            ]
            st.session_state.selected_persona_set = "Software Engineering"
        else:
            st.session_state.codebase_context = {}
            st.session_state.uploaded_files = []
            if st.session_state.selected_persona_set == "Software Engineering":
                st.session_state.selected_persona_set = "General"
    # ADDED: Force a rerun to update the UI, especially the framework selectbox
    st.rerun()

user_prompt = st.text_area("Enter your prompt here:", height=150, key="user_prompt_input")

col1, col2 = st.columns(2)
with col1:
    st.subheader("Reasoning Framework")
    def recommend_domain_from_keywords(prompt: str) -> Optional[str]:
        prompt_lower = prompt.lower()
        scores = {domain: sum(1 for keyword in keywords if keyword in prompt_lower) for domain, keywords in DOMAIN_KEYWORDS.items()}
        if not any(scores.values()): return None
        return max(scores, key=scores.get)

    if user_prompt.strip():
        suggested_domain = recommend_domain_from_keywords(user_prompt)
        if suggested_domain and suggested_domain != st.session_state.selected_persona_set and suggested_domain in st.session_state.available_domains:
            st.info(f"💡 Recommendation: Use the **'{suggested_domain}'** framework for this prompt.")
            if st.button(f"Apply '{suggested_domain}' Framework", type="primary", use_container_width=True):
                st.session_state.selected_persona_set_widget = suggested_domain
                st.rerun()

    # Use available_domains from PersonaManager (which now includes custom frameworks)
    available_framework_options = st.session_state.available_domains
    unique_framework_options = sorted(list(set(available_framework_options)))
    
    current_framework_selection = st.session_state.selected_persona_set
    # Ensure the current selection is valid, otherwise default to the first available
    if current_framework_selection not in unique_framework_options:
        current_framework_selection = unique_framework_options[0] if unique_framework_options else "General"
        st.session_state.selected_persona_set = current_framework_selection
        
    selected_framework_for_widget = st.selectbox(
        "Select Framework",
        options=unique_framework_options,
        index=unique_framework_options.index(current_framework_selection) if current_framework_selection in unique_framework_options else 0,
        key="selected_persona_set_widget",
        help="Choose a domain-specific reasoning framework or a custom saved framework."
    )
    # Update session state if the selection changes
    if selected_framework_for_widget != st.session_state.selected_persona_set:
        st.session_state.selected_persona_set = selected_framework_for_widget
        # Rerun to load personas for the new framework
        st.rerun()

    # Load personas for the selected framework
    if st.session_state.selected_persona_set:
        # Use PersonaManager to get the correct personas for the selected framework
        current_domain_personas = st.session_state.persona_manager.all_personas.get(st.session_state.selected_persona_set, {})
        if not current_domain_personas: # Fallback if framework not found directly
            current_domain_persona_names = st.session_state.persona_sets.get(st.session_state.selected_persona_set, [])
            current_domain_personas = {name: st.session_state.all_personas[name] for name in current_domain_persona_names if name in st.session_state.all_personas}
        
        st.session_state.personas = current_domain_personas # Store active personas for the current framework

    st.subheader("Save Current Framework")
    new_framework_name_input = st.text_input("Enter a name for your framework:", key='save_framework_input')
    # Add a text area for framework description
    framework_description_input = st.text_area("Framework Description (Optional):", key='framework_description', height=50)

    if st.button("Save Current Framework") and new_framework_name_input:
        # Get current active personas for the selected framework
        current_framework_name = st.session_state.selected_persona_set
        current_domain_persona_names = st.session_state.persona_sets.get(current_framework_name, [])
        current_active_personas_data = {
            p_name: st.session_state.all_personas[p_name] # Pass PersonaConfig objects directly
            for p_name in current_domain_persona_names
            if p_name in st.session_state.all_personas
        }

        # Use PersonaManager's save_framework method
        if st.session_state.persona_manager.save_framework(new_framework_name_input, current_framework_name, current_active_personas_data):
            st.rerun() # Rerun to update UI elements like the selectbox

    st.subheader("Load Framework")
    # Get all available frameworks (default + custom) from PersonaManager
    all_available_frameworks_for_load = [""] + st.session_state.persona_manager.available_domains
    unique_framework_options_for_load = sorted(list(set(all_available_frameworks_for_load)))
    
    current_selection_for_load = ""
    if st.session_state.selected_persona_set in unique_framework_options_for_load:
        current_selection_for_load = st.session_state.selected_persona_set
    # Check if the current selection is a custom framework name not directly in persona_sets
    elif st.session_state.selected_persona_set in st.session_state.persona_manager.all_custom_frameworks_data:
        current_selection_for_load = st.session_state.selected_persona_set
    
    selected_framework_to_load = st.selectbox(
        "Select a framework to load:",
        options=unique_framework_options_for_load,
        index=unique_framework_options_for_load.index(current_selection_for_load) if current_selection_for_load in unique_framework_options_for_load else 0,
        key='load_framework_select'
    )
    if st.button("Load Selected Framework") and selected_framework_to_load:
        # PersonaManager handles loading both default and custom frameworks
        loaded_personas_dict, loaded_persona_sets_dict, new_selected_framework_name = st.session_state.persona_manager.load_framework_into_session(selected_framework_to_load)
        
        if loaded_personas_dict:
            # Update session state with loaded data
            st.session_state.all_personas.update(loaded_personas_dict)
            st.session_state.persona_sets.update(loaded_persona_sets_dict)
            st.session_state.selected_persona_set = new_selected_framework_name
            st.rerun()
    
    st.subheader("Context Budget")
    st.slider(
        "Context Token Budget Ratio", min_value=0.05, max_value=0.5, value=st.session_state.context_token_budget_ratio,
        step=0.05, key="context_token_budget_ratio", help="Percentage of total token budget allocated to context analysis."
    )

with col2:
    st.subheader("Codebase Context (Optional)")
    if st.session_state.selected_persona_set == "Software Engineering":
        uploaded_files = st.file_uploader(
            "Upload up to 100 relevant files", # MODIFIED: Changed limit from 25 to 100
            accept_multiple_files=True,
            type=['py', 'js', 'ts', 'html', 'css', 'json', 'yaml', 'md', 'txt', 'java', 'go', 'rb', 'php'],
            help="Provide files for context. The AI will analyze them to generate consistent code.",
            key="code_context_uploader"
        )
        if uploaded_files:
            current_uploaded_file_info = [(f.name, f.size) for f in uploaded_files]
            previous_uploaded_file_info = [(f.name, f.size) for f in st.session_state.uploaded_files]
            if current_uploaded_file_info != previous_uploaded_file_info:
                if len(uploaded_files) > 100: # MODIFIED: Changed limit from 25 to 100
                    st.warning("Please upload a maximum of 100 files. Truncating to the first 100.") # MODIFIED: Changed message
                    uploaded_files = uploaded_files[:100]
                temp_context = {}
                for file in uploaded_files:
                    try:
                        content = file.getvalue().decode("utf-8")
                        temp_context[file.name] = content
                    except Exception as e:
                        st.error(f"Error reading {file.name}: {e}")
                st.session_state.codebase_context = temp_context
                st.session_state.uploaded_files = uploaded_files
                st.toast(f"{len(st.session_state.codebase_context)} file(s) loaded for context.")
        elif st.session_state.codebase_context and not uploaded_files:
            st.success(f"{len(st.session_state.codebase_context)} file(s) already loaded for context.")
        else:
            st.session_state.codebase_context = {}
            st.session_state.uploaded_files = []
    else:
        st.info("Select the 'Software Engineering' framework to provide codebase context.")
        st.session_state.codebase_context = {}
        st.session_state.uploaded_files = []

# --- NEW: Persona Editing UI ---
st.markdown("---")
with st.expander("⚙️ View and Edit Personas", expanded=st.session_state.persona_edit_mode):
    st.session_state.persona_edit_mode = True # Keep expander open if user interacts
    st.info("Edit persona parameters for the **currently selected framework**. Changes are temporary unless saved as a custom framework.")
    
    # Sort personas for consistent display
    sorted_persona_names = sorted(st.session_state.personas.keys()) # Assuming st.session_state.personas is populated correctly

    for p_name in sorted_persona_names:
        # Access PersonaConfig object via PersonaManager
        persona: PersonaConfig = st.session_state.persona_manager.all_personas.get(p_name)
        if not persona:
            st.warning(f"Persona '{p_name}' not found in manager. Skipping.")
            continue

        with st.expander(f"**{persona.name.replace('_', ' ')}**", expanded=False):
            st.markdown(f"**Description:** {persona.description}")
            
            # Track changes for visual feedback
            changes_made = False

            # System Prompt
            new_system_prompt = st.text_area(
                "System Prompt",
                value=persona.system_prompt,
                height=200,
                key=f"system_prompt_{p_name}",
                help="The core instructions for this persona."
            )
            if new_system_prompt != persona.system_prompt:
                changes_made = True
                # Use PersonaManager to update
                if st.session_state.persona_manager.update_persona_config(p_name, "system_prompt", new_system_prompt):
                    # No immediate rerun needed here, changes are managed by the manager
                    pass # The manager updates its internal state
                else:
                    st.error(f"Failed to update system prompt for {p_name}.")

            # Temperature
            new_temperature = st.slider(
                "Temperature",
                min_value=0.0,
                max_value=1.0,
                value=persona.temperature,
                step=0.05,
                key=f"temperature_{p_name}",
                help="Controls the randomness of the output. Lower values mean less random."
            )
            if new_temperature != persona.temperature:
                changes_made = True
                if st.session_state.persona_manager.update_persona_config(p_name, "temperature", new_temperature):
                    pass
                else:
                    st.error(f"Failed to update temperature for {p_name}.")
            
            # Max Tokens
            new_max_tokens = st.number_input(
                "Max Output Tokens",
                min_value=1,
                max_value=8192, # Max tokens for Gemini models can be higher
                value=persona.max_tokens,
                step=128,
                key=f"max_tokens_{p_name}",
                help="Maximum number of tokens the LLM can generate in response."
            )
            if new_max_tokens != persona.max_tokens:
                changes_made = True
                if st.session_state.persona_manager.update_persona_config(p_name, "max_tokens", new_max_tokens):
                    pass
                else:
                    st.error(f"Failed to update max tokens for {p_name}.")
            
            # Display status and reset button
            if changes_made:
                st.info("Changes detected. Click 'Reset' to revert or save as a custom framework.")
            
            # Reset button for individual persona
            if st.button(f"Reset {p_name.replace('_', ' ')} to Default", key=f"reset_persona_{p_name}"):
                if st.session_state.persona_manager.reset_persona_to_default(p_name):
                    st.toast(f"Persona '{p_name.replace('_', ' ')}' reset to default.")
                    st.rerun() # Rerun to update UI widgets with reset values
                else:
                    st.error(f"Could not reset persona '{p_name}'.")

# --- END NEW: Persona Editing UI ---

st.markdown("---")
run_col, reset_col = st.columns([0.8, 0.2])
with run_col:
    run_button_clicked = st.button("🚀 Run Socratic Debate", type="primary", use_container_width=True)
with reset_col:
    st.button("🔄 Reset All", on_click=reset_app_state, use_container_width=True)

if run_button_clicked:
    api_key_feedback_placeholder.empty()
    if not st.session_state.api_key_input.strip():
        st.error("Please enter your Gemini API Key in the sidebar to proceed.")
    elif not user_prompt.strip():
        st.error("Please enter a prompt.")
    else:
        st.session_state.debate_ran = False
        with st.status("Initializing Socratic Debate...", expanded=True) as status:
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
            final_total_tokens = 0
            final_total_cost = 0.0
            try:
                # Ensure personas_for_run uses the potentially modified PersonaConfig objects from all_personas
                domain_for_run = st.session_state.selected_persona_set
                current_domain_persona_names = st.session_state.persona_sets.get(domain_for_run, [])
                # Use the potentially modified personas from session_state.all_personas
                personas_for_run = {name: st.session_state.all_personas[name] for name in current_domain_persona_names if name in st.session_state.all_personas}

                if not personas_for_run:
                    raise ValueError(f"No personas found for the selected framework '{domain_for_run}'. Please check your configuration.")
                
                with capture_rich_output_and_get_console() as (rich_output_buffer, rich_console_instance):
                    debate_instance = core.SocraticDebate(
                        initial_prompt=user_prompt,
                        api_key=st.session_state.api_key_input,
                        max_total_tokens_budget=st.session_state.max_tokens_budget_input,
                        model_name=st.session_state.selected_model_selectbox,
                        all_personas=st.session_state.all_personas, # Pass the full set of personas
                        persona_sets=st.session_state.persona_sets,
                        persona_sequence=st.session_state.persona_sequence,
                        domain=domain_for_run,
                        status_callback=streamlit_status_callback,
                        rich_console=rich_console_instance,
                        codebase_context=st.session_state.get('codebase_context', {}),
                        context_token_budget_ratio=st.session_state.context_token_budget_ratio
                    )
                    # Pass context_analysis_results to run_debate if available
                    context_analysis_results = debate_instance.intermediate_steps.get("Context_Analysis") # This will be populated inside run_debate
                    
                    final_answer, intermediate_steps = debate_instance.run_debate()
                    
                    st.session_state.process_log_output_text = rich_output_buffer.getvalue()
                    st.session_state.final_answer_output = final_answer
                    st.session_state.intermediate_steps_output = intermediate_steps
                    st.session_state.last_config_params = {
                        "max_tokens_budget": st.session_state.max_tokens_budget_input,
                        "model_name": st.session_state.selected_model_selectbox,
                        "show_intermediate_steps": st.session_state.show_intermediate_steps_checkbox,
                        "domain": domain_for_run
                    }
                    st.session_state.debate_ran = True
                    status.update(label="Socratic Debate Complete!", state="complete", expanded=False)
                    final_total_tokens = intermediate_steps.get('Total_Tokens_Used', 0)
                    final_total_cost = intermediate_steps.get('Total_Estimated_Cost_USD', 0.0)
            
            # --- UPDATED ERROR HANDLING ---
            except (TokenBudgetExceededError, SchemaValidationError, ChimeraError, GeminiAPIError, LLMProviderError) as e:
                # Capture any output buffer if it exists
                if 'rich_output_buffer' in locals():
                    st.session_state.process_log_output_text = rich_output_buffer.getvalue()
                else:
                    st.session_state.process_log_output_text = "" # Ensure it's empty if buffer wasn't created

                status.update(label=f"Socratic Debate Failed: {e}", state="error", expanded=True)
                st.error(f"**Error:** {e}")
                st.session_state.debate_ran = True
                if debate_instance:
                    st.session_state.intermediate_steps_output = debate_instance.intermediate_steps
                
                # Populate final_answer_output with error details
                st.session_state.final_answer_output = {
                    "COMMIT_MESSAGE": "Debate Failed",
                    "RATIONALE": f"An error occurred during the Socratic debate: {str(e)}",
                    "CODE_CHANGES": [],
                    "CONFLICT_RESOLUTION": None,
                    "UNRESOLVED_CONFLICT": None,
                    "malformed_blocks": [{"type": "DEBATE_ERROR", "message": str(e), "details": getattr(e, 'details', {})}] # Add details from exception
                }
                final_total_tokens = st.session_state.intermediate_steps_output.get('Total_Tokens_Used', 0)
                final_total_cost = st.session_state.intermediate_steps_output.get('Total_Estimated_Cost_USD', 0.0)
            except Exception as e: # Catch any other unexpected errors not covered by specific exceptions
                if 'rich_output_buffer' in locals():
                    st.session_state.process_log_output_text = rich_output_buffer.getvalue()
                else:
                    st.session_state.process_log_output_text = ""

                status.update(label=f"Socratic Debate Failed: An unexpected error occurred: {e}", state="error", expanded=True)
                st.error(f"**Unexpected Error:** {e}")
                st.session_state.debate_ran = True
                if debate_instance:
                    st.session_state.intermediate_steps_output = debate_instance.intermediate_steps
                
                # Populate final_answer_output with error details
                st.session_state.final_answer_output = {
                    "COMMIT_MESSAGE": "Debate Failed (Unexpected Error)",
                    "RATIONALE": f"An unexpected error occurred during the Socratic debate: {str(e)}",
                    "CODE_CHANGES": [],
                    "CONFLICT_RESOLUTION": None,
                    "UNRESOLVED_CONFLICT": None,
                    "malformed_blocks": [{"type": "UNEXPECTED_ERROR", "message": str(e), "details": {"traceback": traceback.format_exc()}}] # Include traceback for unexpected errors
                }
                final_total_tokens = st.session_state.intermediate_steps_output.get('Total_Tokens_Used', 0)
                final_total_cost = st.session_state.intermediate_steps_output.get('Total_Estimated_Cost_USD', 0.0)
            # --- END UPDATED ERROR HANDLING ---

            total_tokens_placeholder.metric("Total Tokens Used", f"{final_total_tokens:,}")
            total_cost_placeholder.metric("Estimated Cost (USD)", f"${final_total_cost:.4f}")
            next_step_warning_placeholder.empty()

if st.session_state.debate_ran:
    st.markdown("---")
    st.header("Results")

    # --- ADDED: Download Buttons ---
    download_cols = st.columns(2)
    with download_cols[0]:
        # Generate the full markdown report content
        full_report_content = generate_markdown_report(
            user_prompt=user_prompt,
            final_answer=st.session_state.final_answer_output,
            intermediate_steps=st.session_state.intermediate_steps_output,
            process_log_output=st.session_state.process_log_output_text,
            config_params=st.session_state.last_config_params,
            persona_audit_log=st.session_state.persona_audit_log # Pass audit log
        )
        st.download_button(
            label="Download Full Report (Markdown)",
            data=full_report_content,
            file_name=f"project_chimera_report_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
            mime="text/markdown",
            use_container_width=True
        )
    with download_cols[1]:
        # Generate the final answer JSON content if applicable
        if isinstance(st.session_state.final_answer_output, dict):
            final_answer_json = json.dumps(st.session_state.final_answer_output, indent=2)
            st.download_button(
                label="Download Final Answer (JSON)",
                data=final_answer_json,
                file_name=f"project_chimera_final_answer_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json",
                use_container_width=True
            )
        else:
            st.info("Final answer is not in JSON format for download.")
    # --- END ADDED: Download Buttons ---

    if st.session_state.last_config_params.get("domain") == "Software Engineering":
        # --- REVISED: LLMOutput Handling and Validation Display ---
        parsed_llm_output: LLMOutput
        malformed_blocks_from_parser = []
        
        # Get the raw output from session state
        raw_output_data = st.session_state.final_answer_output

        # --- APPLY FIX HERE ---
        # Check if the raw output is a list, which is unexpected for direct LLMOutput instantiation
        if isinstance(raw_output_data, list):
            if not raw_output_data:
                logger.error("Parser returned an empty list, no valid output found.")
                # Create an error structure that app.py can handle
                st.session_state.final_answer_output = {
                    "COMMIT_MESSAGE": "Debate Failed - No Output",
                    "RATIONALE": "The LLM parser returned an empty list, indicating no valid output was found.",
                    "CODE_CHANGES": [],
                    "malformed_blocks": [{"type": "EMPTY_LIST_OUTPUT", "message": "Parser returned an empty list."}]
                }
                # Re-run to show the error message
                st.rerun() 
            else:
                # Take the first item if it's a list of potential outputs
                logger.debug(f"Parser returned {len(raw_output_data)} output blocks, using the first one for LLMOutput instantiation.")
                # Update raw_output_data to be the first item
                raw_output_data = raw_output_data[0]
        # --- END FIX ---

        # Now, proceed with the assumption that raw_output_data is either a dict or was handled if it was a list
        if isinstance(raw_output_data, dict):
            try:
                # Attempt to parse the final_answer_output into an LLMOutput model
                parsed_llm_output = LLMOutput(**raw_output_data)
                # Capture malformed blocks reported by the parser during the arbitrator's step
                malformed_blocks_from_parser = raw_output_data.get('malformed_blocks', [])
            except ValidationError as e:
                st.error(f"Failed to parse final LLM output into LLMOutput model: {e}")
                # Create a fallback LLMOutput instance with error details
                parsed_llm_output = LLMOutput(
                    COMMIT_MESSAGE="Parsing Error",
                    RATIONALE=f"Failed to parse final LLM output into expected structure. Error: {e}",
                    CODE_CHANGES=[],
                    malformed_blocks=[{"type": "UI_PARSING_ERROR", "message": str(e), "raw_string_snippet": str(raw_output_data)[:500]}]
                )
                malformed_blocks_from_parser.extend(parsed_llm_output.malformed_blocks) # Add to the list
        else:
            # This case handles when final_answer_output is not a dictionary (e.g., a simple string error message)
            st.error(f"Final answer is not a structured dictionary or a list of dictionaries. Raw output type: {type(raw_output_data).__name__}")
            parsed_llm_output = LLMOutput(
                COMMIT_MESSAGE="Error: Output not structured.",
                RATIONALE=f"Error: Output not structured. Raw output type: {type(raw_output_data).__name__}",
                CODE_CHANGES=[],
                malformed_blocks=[{"type": "UI_PARSING_ERROR", "message": f"Final answer was not a dictionary or list. Type: {type(raw_output_data).__name__}", "raw_string_snippet": str(raw_output_data)[:500]}]
            )
            malformed_blocks_from_parser.extend(parsed_llm_output.malformed_blocks) # Add to the list

        # The validate_code_output_batch function expects a dictionary, so pass the model_dump
        # Ensure that the input to validate_code_output_batch is a dictionary, even if it's an error dict.
        validation_results_by_file = validate_code_output_batch(
            parsed_llm_output.model_dump(by_alias=True) if isinstance(parsed_llm_output, LLMOutput) else raw_output_data, # Pass the dictionary representation
            st.session_state.get('codebase_context', {})
        )

        # Aggregate all issues for display
        all_issues = []
        for file_issues_list in validation_results_by_file.values():
            if isinstance(file_issues_list, list): # Ensure it's a list of issues
                all_issues.extend(file_issues_list)

        # Combine malformed blocks from parser and any from validation
        all_malformed_blocks = malformed_blocks_from_parser
        # Check if validate_code_output_batch itself reported malformed blocks (e.g., if its input was bad)
        if 'malformed_blocks' in validation_results_by_file:
            all_malformed_blocks.extend(validation_results_by_file['malformed_blocks'])

        st.subheader("Structured Summary")
        summary_col1, summary_col2 = st.columns(2)
        with summary_col1:
            st.markdown("**Commit Message Suggestion**")
            st.code(parsed_llm_output.commit_message, language='text')
        with summary_col2:
            st.markdown("**Token Usage**")
            total_tokens = st.session_state.intermediate_steps_output.get('Total_Tokens_Used', 0)
            total_cost = st.session_state.intermediate_steps_output.get('Total_Estimated_Cost_USD', 0.0)
            st.metric("Total Tokens Consumed", f"{total_tokens:,}")
            st.metric("Total Estimated Cost (USD)", f"${total_cost:.4f}")
        st.markdown("**Rationale**")
        st.markdown(parsed_llm_output.rationale)
        if parsed_llm_output.conflict_resolution:
            st.markdown("**Conflict Resolution**")
            st.info(parsed_llm_output.conflict_resolution)
        if parsed_llm_output.unresolved_conflict:
            st.markdown("**Unresolved Conflict**")
            st.warning(parsed_llm_output.unresolved_conflict)

        with st.expander("✅ Validation & Quality Report", expanded=True):
            if not all_issues and not all_malformed_blocks:
                st.success("✅ No syntax, style, or formatting issues detected.")
            else:
                if all_malformed_blocks:
                     st.error(f"**Malformed Output Detected:** The LLM produced {len(all_malformed_blocks)} block(s) that could not be parsed or validated correctly. Raw output snippets are provided below.")
                for issue in all_issues:
                    # --- FIX APPLIED HERE ---
                    file_info = issue.get('file', 'N/A')
                    st.warning(f"**{issue['type']} in `{file_info}`:** {issue['message']} (Line: {issue.get('line', 'N/A')})")
        
        st.subheader("Proposed Code Changes")
        if not parsed_llm_output.code_changes and not all_malformed_blocks:
            st.info("No code changes were proposed.")
        for change in parsed_llm_output.code_changes:
            with st.expander(f"📝 **{change.file_path}** (`{change.action}`)", expanded=False):
                st.write(f"**Action:** {change.action}")
                st.write(f"**File Path:** {change.file_path}")
                if change.action in ['ADD', 'MODIFY']:
                    st.write("**Content:**")
                    display_content = change.full_content or ''
                    # Limit display content to prevent overwhelming the UI
                    st.code(display_content[:2000] + ('...' if len(display_content) > 2000 else ''), language='python')
                elif change.action == 'REMOVE':
                    st.write("**Lines to Remove:**")
                    st.write(change.lines)
        # Display malformed blocks separately if they exist
        for block_info in all_malformed_blocks:
            with st.expander(f"⚠️ **Malformed Block ({block_info.get('type', 'Unknown')})**", expanded=True):
                st.error(f"This block was malformed and could not be parsed correctly. Message: {block_info.get('message', 'N/A')}")
                # Display a snippet of the raw output that caused the issue
                raw_snippet = block_info.get('raw_string_snippet', block_info.get('raw_item', 'N/A'))
                st.code(raw_snippet[:1000] + ('...' if len(raw_snippet) > 1000 else ''), language='text')
    else:
        st.subheader("Final Synthesized Answer")
        # Display the final answer, handling both dict and string cases
        if isinstance(st.session_state.final_answer_output, dict):
            st.json(st.session_state.final_answer_output)
        else:
            st.markdown(st.session_state.final_answer_output)

    with st.expander("Show Intermediate Steps & Process Log"):
        if st.session_state.show_intermediate_steps_checkbox:
            st.subheader("Intermediate Reasoning Steps")
            # Filter out non-step related items for display
            display_steps = {k: v for k, v in st.session_state.intermediate_steps_output.items()
                             if not k.endswith("_Tokens_Used") and k != "Total_Tokens_Used" and k != "Total_Estimated_Cost_USD" and k != "debate_history" and not k.startswith("Final_Answer_") and not k.startswith("Context_Analysis") and not k.startswith("Persona_Sequence") and not k.startswith("Partial_Result_Warning") and not k.startswith("Tokens_Used_Per_Phase") and not k.startswith("Tokens_Used_Per_Step")}
            sorted_step_keys = sorted(display_steps.keys(), key=lambda x: (x.split('_')[0], x))
            for step_key in sorted_step_keys:
                persona_name = step_key.split('_')[0]
                display_name = step_key.replace('_Output', '').replace('_Critique', '').replace('_Feedback', '').replace('_', ' ').title()
                content = display_steps.get(step_key, "N/A")
                cleaned_step_key = step_key.replace("_Output", "").replace("_Critique", "").replace("_Feedback", "")
                token_count_key = f"{cleaned_step_key}_Tokens_Used"
                tokens_used = st.session_state.intermediate_steps_output.get(token_count_key, "N/A")
                with st.expander(f"**{display_name}** (Tokens: {tokens_used})"):
                    # If content is a dict (e.g., final answer, context analysis), display as JSON
                    if isinstance(content, dict):
                        st.json(content)
                    else:
                        st.markdown(f"```markdown\n{content}\n```")
        
        st.subheader("Process Log")
        st.code(strip_ansi_codes(st.session_state.process_log_output_text), language='text')
        
        # Display token usage breakdown if available
        if "Tokens_Used_Per_Phase" in st.session_state.intermediate_steps_output:
            st.subheader("Token Usage Breakdown")
            token_usage_data = st.session_state.intermediate_steps_output
            
            col_phase_1, col_phase_2, col_phase_3 = st.columns(3)
            with col_phase_1:
                st.metric("Context Phase", f"{token_usage_data['Tokens_Used_Per_Phase'].get('context', 0):,} / {token_usage_data['phase_budgets'].get('context', 0):,}")
            with col_phase_2:
                st.metric("Debate Phase", f"{token_usage_data['Tokens_Used_Per_Phase'].get('debate', 0):,} / {token_usage_data['phase_budgets'].get('debate', 0):,}")
            with col_phase_3:
                st.metric("Synthesis Phase", f"{token_usage_data['Tokens_Used_Per_Phase'].get('synthesis', 0):,} / {token_usage_data['phase_budgets'].get('synthesis', 0):,}")
            
            st.markdown("**Step-by-Step Token Usage:**")
            for step_name, tokens in token_usage_data.get("Tokens_Used_Per_Step", {}).items():
                st.write(f"- {step_name}: {tokens:,} tokens")