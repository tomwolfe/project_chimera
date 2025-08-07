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
                                       if not k.endswith("_Tokens_Used") and k != "Total_Tokens_Used" and k != "Total_Estimated_Cost_USD" and k != "debate_history"],
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

def reset_app_state():
    """Resets all session state variables to their default values."""
    st.session_state.api_key_input = os.getenv("GEMINI_API_KEY", "")
    st.session_state.user_prompt_input = EXAMPLE_PROMPTS[list(EXAMPLE_PROMPTS.keys())[0]]
    st.session_state.max_tokens_budget_input = 1000000
    st.session_state.show_intermediate_steps_checkbox = True
    st.session_state.selected_model_selectbox = "gemini-2.5-flash-lite"
    st.session_state.selected_example_name = list(EXAMPLE_PROMPTS.keys())[0]
    if "available_domains" in st.session_state and st.session_state.available_domains:
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
    st.session_state.uploaded_files = []
    st.session_state.context_token_budget_ratio = CONTEXT_TOKEN_BUDGET_RATIO
    st.session_state.example_selector_widget = st.session_state.selected_example_name
    st.session_state.selected_persona_set_widget = st.session_state.selected_persona_set
    st.session_state.persona_audit_log = [] # Reset audit log
    st.session_state.persona_edit_mode = False # Reset edit mode

# --- Custom Framework File Management ---
CUSTOM_FRAMEWORKS_DIR = "custom_frameworks"
def ensure_custom_frameworks_dir():
    """Ensures the directory for custom frameworks exists."""
    if not os.path.exists(CUSTOM_FRAMEWORKS_DIR):
        try:
            os.makedirs(CUSTOM_FRAMEWORKS_DIR)
            st.toast(f"Created directory for custom frameworks: '{CUSTOM_FRAMEWORKS_DIR}'")
        except OSError as e:
            st.error(f"Error creating custom frameworks directory: {e}")

def sanitize_framework_filename(name: str) -> str:
    """Sanitizes a framework name to be used as a valid filename."""
    sanitized = re.sub(r'[<>:"/\\|?*\s]+', '_', name)
    sanitized = re.sub(r'^[^a-zA-Z0-9_]+|[^a-zA-Z0-9_]+$', '', sanitized)
    if not sanitized:
        sanitized = "unnamed_framework"
    return sanitized

def get_saved_custom_framework_names() -> List[str]:
    """Loads and returns the names of all custom frameworks saved as JSON files."""
    ensure_custom_frameworks_dir()
    framework_names = []
    try:
        for filename in os.listdir(CUSTOM_FRAMEWORKS_DIR):
            if filename.endswith(".json"):
                framework_names.append(os.path.splitext(filename)[0])
        return sorted(framework_names)
    except OSError as e:
        st.error(f"Error listing custom frameworks: {e}")
        return []

def load_custom_framework_config(name: str) -> Dict[str, Any]:
    """Loads a specific custom framework configuration from a JSON file."""
    filename = f"{sanitize_framework_filename(name)}.json"
    filepath = os.path.join(CUSTOM_FRAMEWORKS_DIR, filename)
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            config_data = json.load(f)
        return config_data
    except (FileNotFoundError, json.JSONDecodeError, OSError) as e:
        if isinstance(e, FileNotFoundError):
            st.error(f"Custom framework '{name}' not found at '{filepath}'.")
        elif isinstance(e, json.JSONDecodeError):
            st.error(f"Error decoding JSON from '{filepath}'. Please check its format: {e}")
        st.error(f"Error reading custom framework file '{filepath}': {e}")
        return {}

def save_current_framework_to_file(name: str):
    """Saves the current framework configuration (personas, persona_sets) to a JSON file."""
    if not name:
        st.warning("Please enter a name for the framework before saving.")
        return
    framework_name_sanitized = sanitize_framework_filename(name)
    if not framework_name_sanitized:
        st.error("Invalid framework name provided after sanitization.")
        return
    
    # Get the personas currently active in the selected framework, including any UI edits
    current_persona_names_in_set = st.session_state.persona_sets.get(st.session_state.selected_persona_set, [])
    current_personas_dict = {
        p_name: st.session_state.all_personas[p_name].model_dump()
        for p_name in current_persona_names_in_set if p_name in st.session_state.all_personas
    }

    # If this is an existing custom framework, increment its version
    version = 1
    if framework_name_sanitized in st.session_state.all_custom_frameworks_data:
        version = st.session_state.all_custom_frameworks_data[framework_name_sanitized].get('version', 0) + 1

    try:
        temp_config_validation = ReasoningFrameworkConfig(
            framework_name=name,
            personas={p_name: PersonaConfig(**p_data) for p_name, p_data in current_personas_dict.items()},
            persona_sets={st.session_state.selected_persona_set: current_persona_names_in_set}, # Only save the current set
            version=version
        )
        config_to_save = temp_config_validation.model_dump()
    except Exception as e:
        st.error(f"Cannot save framework: Invalid data structure. {e}")
        return
    filename = f"{framework_name_sanitized}.json"
    filepath = os.path.join(CUSTOM_FRAMEWORKS_DIR, filename)
    try:
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(config_to_save, f, indent=2)
        st.toast(f"Framework '{name}' saved successfully to '{filepath}'!")
        
        # Update session state with the new/updated custom framework
        st.session_state.all_custom_frameworks_data[framework_name_sanitized] = config_to_save
        if framework_name_sanitized not in st.session_state.available_domains:
             st.session_state.available_domains.append(framework_name_sanitized)
        
        st.session_state.selected_persona_set = framework_name_sanitized # Select the newly saved framework
        st.rerun() # Rerun to update UI with new framework selected
    except OSError as e:
        st.error(f"Error saving framework '{name}' to '{filepath}': {e}")
    except Exception as e:
        st.error(f"An unexpected error occurred while saving framework '{name}': {e}")

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

# --- Initialize Session State ---
if 'all_custom_frameworks_data' not in st.session_state:
    ensure_custom_frameworks_dir()
    st.session_state.all_custom_frameworks_data = {}
    saved_names = get_saved_custom_framework_names()
    for name in saved_names:
        try:
            config = load_custom_framework_config(name)
        except Exception as e:
            st.error(f"Failed to load custom framework '{name}': {e}")
        if config:
            st.session_state.all_custom_frameworks_data[name] = config

if "all_personas" not in st.session_state:
    try:
        all_personas_from_yaml, persona_sets_from_yaml, persona_sequence_from_yaml, default_persona_set_name = core.load_personas()
        st.session_state.all_personas = all_personas_from_yaml
        st.session_state.persona_sets = persona_sets_from_yaml
        st.session_state.persona_sequence = persona_sequence_from_yaml
        st.session_state.available_domains = list(persona_sets_from_yaml.keys())
        st.session_state.selected_persona_set = default_persona_set_name
    except Exception as e:
        st.error(f"Failed to load personas from personas.yaml: {e}")
        st.session_state.all_personas = {}
        st.session_state.persona_sets = {}
        st.session_state.persona_sequence = []
        st.session_state.available_domains = ["General"]
        st.session_state.selected_persona_set = "General"

# Initialize other session state variables
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
if "uploaded_files" not in st.session_state:
    st.session_state.uploaded_files = []
if "context_token_budget_ratio" not in st.session_state:
    st.session_state.context_token_budget_ratio = CONTEXT_TOKEN_BUDGET_RATIO
if "example_selector_widget" not in st.session_state:
    st.session_state.example_selector_widget = st.session_state.selected_example_name
if "selected_persona_set_widget" not in st.session_state:
    st.session_state.selected_persona_set_widget = st.session_state.selected_persona_set
if "persona_audit_log" not in st.session_state: # NEW: Audit log
    st.session_state.persona_audit_log = []
if "persona_edit_mode" not in st.session_state: # NEW: Edit mode toggle
    st.session_state.persona_edit_mode = False

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

    available_framework_options = st.session_state.available_domains + list(st.session_state.all_custom_frameworks_data.keys())
    unique_framework_options = sorted(list(set(available_framework_options)))
    current_framework_selection = st.session_state.selected_persona_set
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
    st.session_state.selected_persona_set = selected_framework_for_widget

    # Load personas for the selected framework
    if st.session_state.selected_persona_set:
        if st.session_state.selected_persona_set in st.session_state.all_custom_frameworks_data:
            custom_data = st.session_state.all_custom_frameworks_data[st.session_state.selected_persona_set]
            # Update all_personas with custom framework's personas
            for name, data in custom_data.get('personas', {}).items():
                st.session_state.all_personas[name] = PersonaConfig(**data)
            st.session_state.persona_sets.update(custom_data.get('persona_sets', {}))
            try:
                current_domain_persona_names = st.session_state.persona_sets.get(st.session_state.selected_persona_set, [])
                st.session_state.personas = {name: st.session_state.all_personas[name] for name in current_domain_persona_names if name in st.session_state.all_personas}
            except Exception as e:
                st.error(f"Error loading custom framework '{st.session_state.selected_persona_set}': Invalid persona data. {e}")
                st.session_state.personas = {}
        elif st.session_state.selected_persona_set in st.session_state.persona_sets:
            current_domain_persona_names = st.session_state.persona_sets.get(st.session_state.selected_persona_set, [])
            st.session_state.personas = {name: st.session_state.all_personas[name] for name in current_domain_persona_names if name in st.session_state.all_personas}
        else:
            st.warning(f"Selected framework '{st.session_state.selected_persona_set}' has no associated personas defined.")
            st.session_state.personas = {}

    st.subheader("Save Current Framework")
    new_framework_name_input = st.text_input("Enter a name for your current framework:", key='save_framework_input')
    if st.button("Save Current Framework") and new_framework_name_input:
        save_current_framework_to_file(new_framework_name_input)
        # Rerun is handled inside save_current_framework_to_file

    st.subheader("Load Framework")
    custom_framework_names = list(st.session_state.all_custom_frameworks_data.keys())
    all_available_frameworks_for_load = [""] + st.session_state.available_domains + custom_framework_names
    unique_framework_options_for_load = sorted(list(set(all_available_frameworks_for_load)))
    current_selection_for_load = ""
    if st.session_state.selected_persona_set in unique_framework_options_for_load:
        current_selection_for_load = st.session_state.selected_persona_set
    elif st.session_state.selected_persona_set in st.session_state.all_custom_frameworks_data:
        current_selection_for_load = st.session_state.selected_persona_set
    selected_framework_to_load = st.selectbox(
        "Select a framework to load:",
        options=unique_framework_options_for_load,
        index=unique_framework_options_for_load.index(current_selection_for_load) if current_selection_for_load in unique_framework_options_for_load else 0,
        key='load_framework_select'
    )
    if st.button("Load Selected Framework") and selected_framework_to_load:
        if selected_framework_to_load in st.session_state.all_custom_frameworks_data:
            loaded_config_data = st.session_state.all_custom_frameworks_data[selected_framework_to_load]
            # Update all_personas with loaded custom framework's personas
            for name, data in loaded_config_data.get('personas', {}).items():
                st.session_state.all_personas[name] = PersonaConfig(**data)
            st.session_state.persona_sets.update(loaded_config_data.get('persona_sets', {}))
            try:
                current_domain_persona_names = st.session_state.persona_sets.get(selected_framework_to_load, [])
                st.session_state.personas = {name: st.session_state.all_personas[name] for name in current_domain_persona_names if name in st.session_state.all_personas}
                st.session_state.current_framework_name = selected_framework_to_load
                st.session_state.selected_persona_set = selected_framework_to_load
                st.success(f"Loaded custom framework: '{selected_framework_to_load}'")
                st.rerun()
            except Exception as e:
                st.error(f"Error loading custom framework '{selected_framework_to_load}': Invalid persona data. {e}")
                st.session_state.personas = {}
        elif selected_framework_to_load in st.session_state.available_domains:
            st.session_state.selected_persona_set = selected_framework_to_load
            st.success(f"Loaded default framework: '{selected_framework_to_load}'")
            st.rerun()
        else:
            st.error(f"Framework '{selected_framework_to_load}' not found.")
    st.subheader("Context Budget")
    st.slider(
        "Context Token Budget Ratio", min_value=0.05, max_value=0.5, value=st.session_state.context_token_budget_ratio,
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
            key="code_context_uploader"
        )
        if uploaded_files:
            current_uploaded_file_info = [(f.name, f.size) for f in uploaded_files]
            previous_uploaded_file_info = [(f.name, f.size) for f in st.session_state.uploaded_files]
            if current_uploaded_file_info != previous_uploaded_file_info:
                if len(uploaded_files) > 25:
                    st.warning("Please upload a maximum of 25 files. Truncating to the first 25.")
                    uploaded_files = uploaded_files[:25]
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
    sorted_persona_names = sorted(st.session_state.personas.keys())

    for p_name in sorted_persona_names:
        persona: PersonaConfig = st.session_state.all_personas[p_name] # Get the PersonaConfig object

        with st.expander(f"**{persona.name.replace('_', ' ')}**", expanded=False):
            st.markdown(f"**Description:** {persona.description}")
            
            # System Prompt
            new_system_prompt = st.text_area(
                "System Prompt",
                value=persona.system_prompt,
                height=200,
                key=f"system_prompt_{p_name}",
                help="The core instructions for this persona."
            )
            if new_system_prompt != persona.system_prompt:
                _log_persona_change(p_name, "system_prompt", persona.system_prompt, new_system_prompt)
                persona.system_prompt = new_system_prompt

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
                _log_persona_change(p_name, "temperature", persona.temperature, new_temperature)
                persona.temperature = new_temperature

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
                _log_persona_change(p_name, "max_tokens", persona.max_tokens, new_max_tokens)
                persona.max_tokens = new_max_tokens
            
            # Reset button for individual persona
            if st.button(f"Reset {p_name.replace('_', ' ')} to Default", key=f"reset_persona_{p_name}"):
                # Reload original persona config from the initial load
                original_personas, _, _, _ = core.load_personas() # Reload from personas.yaml
                if p_name in original_personas:
                    original_persona_config = original_personas[p_name]
                    if persona.system_prompt != original_persona_config.system_prompt:
                        _log_persona_change(p_name, "system_prompt", persona.system_prompt, original_persona_config.system_prompt)
                        persona.system_prompt = original_persona_config.system_prompt
                    if persona.temperature != original_persona_config.temperature:
                        _log_persona_change(p_name, "temperature", persona.temperature, original_persona_config.temperature)
                        persona.temperature = original_persona_config.temperature
                    if persona.max_tokens != original_persona_config.max_tokens:
                        _log_persona_change(p_name, "max_tokens", persona.max_tokens, original_persona_config.max_tokens)
                        persona.max_tokens = original_persona_config.max_tokens
                    st.toast(f"Persona '{p_name.replace('_', ' ')}' reset to default.")
                    st.rerun() # Rerun to update UI widgets

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
            gemini_provider_instance = None
            final_total_tokens = 0
            final_total_cost = 0.0
            try:
                # Ensure personas_for_run uses the potentially modified PersonaConfig objects from all_personas
                domain_for_run = st.session_state.selected_persona_set
                current_domain_persona_names = st.session_state.persona_sets.get(domain_for_run, [])
                personas_for_run = {name: st.session_state.all_personas[name] for name in current_domain_persona_names if name in st.session_state.all_personas}

                if not personas_for_run:
                    raise ValueError(f"No personas found for the selected framework '{domain_for_run}'. Please check your configuration.")
                
                gemini_provider_instance = core.GeminiProvider(
                    api_key=st.session_state.api_key_input,
                    model_name=st.session_state.selected_model_selectbox
                )
                with capture_rich_output_and_get_console() as (rich_output_buffer, rich_console_instance):
                    debate_instance = core.SocraticDebate(
                        initial_prompt=user_prompt,
                        api_key=st.session_state.api_key_input,
                        max_total_tokens_budget=st.session_state.max_tokens_budget_input,
                        model_name=st.session_state.selected_model_selectbox,
                        personas=personas_for_run, # Pass the potentially modified personas
                        all_personas=st.session_state.all_personas, # Pass all personas for fallback
                        persona_sets=st.session_state.persona_sets,
                        persona_sequence=st.session_state.persona_sequence,
                        domain=domain_for_run,
                        gemini_provider=gemini_provider_instance,
                        status_callback=streamlit_status_callback,
                        rich_console=rich_console_instance,
                        codebase_context=st.session_state.get('codebase_context', {}),
                        context_token_budget_ratio=st.session_state.context_token_budget_ratio
                    )
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
            except (core.TokenBudgetExceededError, Exception) as e:
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
                # final_answer_output is now a dict, so populate it with error info
                st.session_state.final_answer_output = {
                    "COMMIT_MESSAGE": "Debate Failed",
                    "RATIONALE": f"Error during debate: {e}",
                    "CODE_CHANGES": [],
                    "CONFLICT_RESOLUTION": None,
                    "UNRESOLVED_CONFLICT": None,
                    "malformed_blocks": [f"Error during debate: {e}"]
                }
                final_total_tokens = st.session_state.intermediate_steps_output.get('Total_Tokens_Used', 0)
                final_total_cost = st.session_state.intermediate_steps_output.get('Total_Estimated_Cost_USD', 0.0)
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
        # final_llm_response_data is the output from LLMOutputParser.parse_and_validate
        final_llm_response_data = st.session_state.final_answer_output 
        
        validation_results = {'issues': [], 'malformed_blocks': []}

        if isinstance(final_llm_response_data, dict):
            # Extract malformed blocks reported by the parser during the arbitrator's step.
            # These are critical for informing the user about parsing failures.
            validation_results['malformed_blocks'].extend(final_llm_response_data.get('malformed_blocks', []))

            # Check if the primary structure (COMMIT_MESSAGE, RATIONALE, CODE_CHANGES) is present
            # This implies the LLMOutputParser successfully created at least a partial LLMOutput dict
            # and that the 'malformed_blocks' list is empty (meaning no critical parsing/schema errors).
            if final_llm_response_data.get('COMMIT_MESSAGE') is not None and \
               final_llm_response_data.get('RATIONALE') is not None and \
               isinstance(final_llm_response_data.get('CODE_CHANGES'), list):
                
                # If there are no malformed blocks from the initial parsing, proceed to content validation.
                if not validation_results['malformed_blocks']:
                    # --- Perform Domain-Specific Validation (e.g., Code Syntax) ---
                    # This is where validate_code_output_batch or similar logic should reside.
                    # It validates the *content* of the code changes.
                    
                    # Pass the already parsed and structured data to validate_code_output_batch.
                    # validate_code_output_batch expects a dictionary with 'CODE_CHANGES'.
                    try:
                        code_validation_issues_by_file = validate_code_output_batch(final_llm_response_data, st.session_state.get('codebase_context', {}))
                        for file_issues_list in code_validation_issues_by_file.values():
                            validation_results['issues'].extend(file_issues_list)
                    except Exception as e:
                        st.error(f"Error during batch code content validation: {e}")
                        validation_results['issues'].append({'type': 'Internal Error', 'file': 'N/A', 'message': f'Batch code content validation failed: {e}'})
                else:
                    # If there were malformed blocks from the arbitrator's output,
                    # consider the overall output malformed and don't proceed with content validation.
                    # The malformed_blocks will be displayed separately.
                    validation_results['issues'].append({
                        'type': 'Malformed Output',
                        'file': 'N/A',
                        'message': 'LLM output structure was malformed and could not be fully parsed by the arbitrator. See "Malformed LLM Blocks" below.'
                    })
            else:
                # This case means final_llm_response_data is a dictionary, but it's missing critical fields
                # required by LLMOutput (e.g., COMMIT_MESSAGE, RATIONALE, CODE_CHANGES).
                # The malformed_blocks should already contain details about why parsing failed.
                validation_results['issues'].append({
                    'type': 'Structural Error',
                    'file': 'N/A',
                    'message': 'LLM output did not conform to the expected primary structure (COMMIT_MESSAGE, RATIONALE, CODE_CHANGES). See "Malformed LLM Blocks" for details.'
                })
        else:
            # This case means final_llm_response_data is not a dictionary at all (e.g., plain string error from LLM).
            st.error(f"Cannot display structured output: Final answer is not a valid structured dictionary. Raw output: {final_llm_response_data}")
            # Re-assign to a structured error dict for display consistency.
            final_llm_response_data = { 
                "COMMIT_MESSAGE": "Error: Output not structured.",
                "RATIONALE": f"Error: Output not structured. Raw output: {final_llm_response_data}",
                "CODE_CHANGES": [],
                "CONFLICT_RESOLUTION": None,
                "UNRESOLVED_CONFLICT": None,
                "malformed_blocks": [f"Final answer was not a dictionary. Raw: {final_llm_response_data}"]
            }
            validation_results['malformed_blocks'].append(f"Final answer was not a dictionary. Raw: {final_llm_response_data}")

        # --- Consolidate and Display Results ---
        st.subheader("Structured Summary")
        summary_col1, summary_col2 = st.columns(2)
        with summary_col1:
            st.markdown("**Commit Message Suggestion**")
            st.code(final_llm_response_data.get('COMMIT_MESSAGE', 'Not generated.'), language='text')
        with summary_col2:
            st.markdown("**Token Usage**")
            total_tokens = st.session_state.intermediate_steps_output.get('Total_Tokens_Used', 0)
            total_cost = st.session_state.intermediate_steps_output.get('Total_Estimated_Cost_USD', 0.0)
            st.metric("Total Tokens Consumed", f"{total_tokens:,}")
            st.metric("Total Estimated Cost (USD)", f"${total_cost:.4f}")
        st.markdown("**Rationale**")
        st.markdown(final_llm_response_data.get('RATIONALE', 'Not generated.'))
        if final_llm_response_data.get('CONFLICT_RESOLUTION'):
            st.markdown("**Conflict Resolution**")
            st.info(final_llm_response_data['CONFLICT_RESOLUTION'])
        if final_llm_response_data.get('UNRESOLVED_CONFLICT'):
            st.markdown("**Unresolved Conflict**")
            st.warning(final_llm_response_data['UNRESOLVED_CONFLICT'])

        with st.expander("✅ Validation & Quality Report", expanded=True):
            if not validation_results['issues'] and not validation_results['malformed_blocks']:
                st.success("✅ No syntax, style, or formatting issues detected.")
            else:
                if validation_results['malformed_blocks']:
                     st.error(f"**Malformed Output Detected:** The LLM produced {len(validation_results['malformed_blocks'])} block(s) that could not be parsed. The raw output is provided as a fallback.")
                for issue in validation_results['issues']:
                    st.warning(f"**{issue['type']} in `{issue['file']}`:** {issue['message']} (Line: {issue.get('line', 'N/A')})")

        st.subheader("Proposed Code Changes")
        if not final_llm_response_data.get('CODE_CHANGES') and not validation_results['malformed_blocks']: 
            st.info("No code changes were proposed.")
        for change in final_llm_response_data.get('CODE_CHANGES', []): 
            with st.expander(f"📝 **{change.get('FILE_PATH', 'N/A')}** (`{change.get('ACTION', 'N/A')}`)", expanded=False):
                st.write(f"**Action:** {change.get('ACTION')}")
                st.write(f"**File Path:** {change.get('FILE_PATH')}")
                if change.get('ACTION') in ['ADD', 'MODIFY']:
                    st.write("**Content:**")
                    display_content = change.get('FULL_CONTENT', '')
                    st.code(display_content[:1000] + ('...' if len(display_content) > 1000 else ''), language='python')
                elif change.get('ACTION') == 'REMOVE':
                    st.write("**Lines to Remove:**")
                    st.write(change.get('LINES', []))
        # Display malformed blocks separately if they exist
        for block in validation_results['malformed_blocks']:
            with st.expander(f"⚠️ **Unknown File (Malformed Block)**", expanded=True):
                st.error("This block was malformed and could not be parsed correctly. Raw output is shown below.")
                st.code(block, language='text')
    else:
        st.subheader("Final Synthesized Answer")
        st.markdown(st.session_state.final_answer_output)

    with st.expander("Show Intermediate Steps & Process Log"):
        if st.session_state.show_intermediate_steps_checkbox:
            st.subheader("Intermediate Reasoning Steps")
            display_steps = {k: v for k, v in st.session_state.intermediate_steps_output.items()
                             if not k.endswith("_Tokens_Used") and k != "Total_Tokens_Used" and k != "Total_Estimated_Cost_USD" and k != "debate_history"}
            sorted_step_keys = sorted(display_steps.keys(), key=lambda x: (x.split('_')[0], x))
            for step_key in sorted_step_keys:
                persona_name = step_key.split('_')[0]
                display_name = step_key.replace('_Output', '').replace('_Critique', '').replace('_Feedback', '').replace('_', ' ').title()
                content = display_steps.get(step_key, "N/A")
                cleaned_step_key = step_key.replace("_Output", "").replace("_Critique", "").replace("_Feedback", "")
                token_count_key = f"{cleaned_step_key}_Tokens_Used"
                tokens_used = st.session_state.intermediate_steps_output.get(token_count_key, "N/A")
                with st.expander(f"**{display_name}** (Tokens: {tokens_used})"):
                    # If content is a dict (e.g., final answer), display as JSON
                    if isinstance(content, dict):
                        st.json(content)
                    else:
                        st.markdown(f"```markdown\n{content}\n```")
        st.subheader("Process Log")
        st.code(strip_ansi_codes(st.session_state.process_log_output_text), language='text')