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
# --- FIX START ---
# Import SocraticDebate directly from the core module
from core import SocraticDebate
# --- FIX END ---

from src.models import PersonaConfig, ReasoningFrameworkConfig, LLMOutput, CodeChange, ContextAnalysisOutput, CritiqueOutput # Added CritiqueOutput
# --- MODIFICATION: Added recommend_domain_from_keywords to src.utils import ---
from src.utils import LLMOutputParser, validate_code_output_batch, sanitize_and_validate_file_path, recommend_domain_from_keywords # Added sanitize_and_validate_file_path and recommend_domain_from_keywords
# --- END MODIFICATION ---
from src.utils.output_parser import LLMOutputParser # Explicitly import for clarity
from src.persona_manager import PersonaManager
from src.exceptions import ChimeraError, LLMResponseValidationError, SchemaValidationError, TokenBudgetExceededError, SchemaValidationError
# --- MODIFICATION: Added is_self_analysis_prompt to src.constants import ---
from src.constants import SELF_ANALYSIS_KEYWORDS, is_self_analysis_prompt # Added import for suggestion 1.1
# --- END MODIFICATION ---
from src.context.context_analyzer import ContextRelevanceAnalyzer # Added import for caching
import traceback # Needed for error handling in app.py
import difflib # For Suggestion 3.1
from collections import defaultdict # For Suggestion 3.2
from pydantic import ValidationError # Import ValidationError for parsing errors
import html # Needed for html.escape in sanitize_user_input

# --- NEW IMPORTS FOR ENHANCEMENTS ---
import uuid # For request ID generation
from src.logging_config import setup_structured_logging # For structured logging
from src.middleware.rate_limiter import RateLimiter, RateLimitExceededError # For rate limiting
# --- END NEW IMPORTS ---

# --- Configuration Loading ---
@st.cache_resource
def load_config(file_path: str = "config.yaml") -> Dict[str, Any]:
    """Load config with validation and user-friendly errors."""
    if not os.path.exists(file_path):
        # Raise an exception instead of calling st.error/st.info/st.stop
        raise FileNotFoundError(f"Config file not found at '{file_path}'. Please create `config.example.yaml` from the `config.example.yaml` template.")
    try:
        with open(file_path, 'r') as f:
            config = yaml.safe_load(f)
            if not isinstance(config, dict):
                raise ValueError(f"Invalid config format in '{file_path}'. Expected a dictionary.")
            return config
    except yaml.YAMLError as e:
        raise ValueError(f"Error parsing config file '{file_path}'. Please check YAML syntax: {e}") from e
    except IOError as e:
        raise IOError(f"IO error reading config file '{file_path}'. Check permissions: {e}") from e

try:
    app_config = load_config()
except (FileNotFoundError, ValueError, IOError) as e:
    st.error(f"❌ Application configuration error: {e}")
    st.stop() # Stop the app if config loading fails

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
        # Raise an exception instead of calling st.error
        raise FileNotFoundError(f"Demo context file not found at '{file_path}'.")
    except json.JSONDecodeError as e:
        raise ValueError(f"Error decoding JSON from '{file_path}'. Please check its format: {e}") from e
    except IOError as e:
        raise IOError(f"IO error reading demo context file '{file_path}'. Check permissions: {e}") from e

# Redirect rich output to a string buffer for Streamlit display
@contextlib.contextmanager
def capture_rich_output_and_get_console():
    """Captures rich output (like Streamlit elements) and returns the captured content."""
    buffer = io.StringIO()
    # Use force_terminal=True to ensure ANSI codes are generated for rich output
    # Use soft_wrap=True for better readability in the output buffer
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
        
        # --- FIX APPLIED HERE ---
        # Define the replacement string outside the f-string expression to avoid SyntaxError.
        # The f-string parser cannot handle raw string literals with backslashes directly within the expression part.
        escaped_newline_for_display = r'\\n'
        # --- END FIX ---

        for entry in persona_audit_log:
            # Truncate long values for better table display
            old_val_str = str(entry.get('old_value', ''))
            new_val_str = str(entry.get('new_value', ''))
            old_val_display = (old_val_str[:50] + '...') if len(old_val_str) > 50 else old_val_str
            new_val_display = (new_val_str[:50] + '...') if len(new_val_str) > 50 else new_val_str
            
            # --- FIX APPLIED HERE ---
            # Move the replace operation OUTSIDE the f-string expression
            old_val_display_escaped = old_val_display.replace('\n', escaped_newline_for_display)
            new_val_display_escaped = new_val_display.replace('\n', escaped_newline_for_display)
            md_content += f"| {entry.get('timestamp')} | {entry.get('persona')} | {entry.get('parameter')} | `{old_val_display_escaped}` | `{new_val_display_escaped}` |\n"
            # --- END FIX ---
        md_content += "\n---\n\n"

    md_content += "## Process Log\n\n"
    md_content += "```text\n"
    md_content += strip_ansi_codes(process_log_output)
    md_content += "\n```\n\n"
    
    if config_params.get('show_intermediate_steps', True):
        md_content += "---\n\n"
        md_content += "## Intermediate Reasoning Steps\n\n"
        step_keys_to_process = sorted([k for k in intermediate_steps.keys()
                                       if not k.endswith("_Tokens_Used") and k != "Total_Tokens_Used" and k != "Total_Estimated_Cost_USD" and k != "debate_history" and not k.startswith("malformed_blocks")],
                                      key=lambda x: (x.split('_')[0] if '_' in x else '', x)) # Sort by persona name first, then step name
        
        for step_key in step_keys_to_process:
            display_name = step_key.replace('_Output', '').replace('_Critique', '').replace('_Feedback', '').replace('_', ' ').title()
            content = intermediate_steps.get(step_key, "N/A")
            token_base_name = step_key.replace("_Output", "").replace("_Critique", "").replace("_Feedback", "")
            token_count_key = f"{token_base_name}_Tokens_Used"
            tokens_used = intermediate_steps.get(token_count_key, "N/A")
            
            md_content += f"### {display_name}\n\n"
            if isinstance(content, dict):
                md_content += "```json\n"
                md_content += json.dumps(content, indent=2)
                md_content += "\n```\n"
            else:
                md_content += f"```markdown\n{content}\n```\n"
            md_content += f"**Tokens Used for this step:** {tokens_used}\n\n"
    md_content += "---\n\n"
    md_content += "## Final Synthesized Answer\n\n"
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

# --- MODIFIED EXAMPLE_PROMPTS STRUCTURE ---
# Grouping prompts by category for better UI organization
EXAMPLE_PROMPTS = {
    "Coding & Implementation": {
        "Implement Python API Endpoint": {
            "prompt": "Implement a new FastAPI endpoint `/items/{item_id}` that retrieves an item from a dictionary. Include basic error handling for non-existent items and add a corresponding unit test.",
            "description": "Generate a complete API endpoint with proper error handling, validation, and documentation.",
            "framework_hint": "Software Engineering" # ADDED: Explicit hint for this example
        },
        "Refactor a Python Function": {
            "prompt": "Refactor the given Python function to improve its readability and performance. It currently uses a nested loop; see if you can optimize it.",
            "description": "Improve structure and readability of existing code while maintaining functionality.",
            "framework_hint": "Software Engineering" # ADDED: Explicit hint for this example
        },
        "Fix a Bug in a Script": {
            "prompt": "The provided Python script is supposed to calculate the average of a list of numbers but fails with a `TypeError` if the list contains non-numeric strings. Fix the bug by safely ignoring non-numeric values.",
            "description": "Identify and correct issues in problematic code with explanations.",
            "framework_hint": "Software Engineering" # ADDED: Explicit hint for this example
        },
    },
    "Analysis & Problem Solving": {
        "Design a Mars City": {
            "prompt": "Design a sustainable city for 1 million people on Mars, considering resource scarcity and human psychology.",
            "description": "Explore complex design challenges with multi-faceted considerations.",
            "framework_hint": "Creative" # ADDED: Explicit hint
        },
        "Ethical AI Framework": {
            "prompt": "Develop an ethical framework for an AI system designed to assist in judicial sentencing, addressing bias, transparency, and accountability.",
            "description": "Formulate ethical guidelines for sensitive AI applications.",
            "framework_hint": "Business" # ADDED: Explicit hint
        },
        "Critically analyze the entire Project Chimera codebase. Identify the most impactful code changes for self-improvement, focusing on the 80/20 Pareto principle. Prioritize enhancements to reasoning quality, robustness, efficiency, and developer maintainability. For each suggestion, provide a clear rationale and a specific, actionable code modification.": {
            "prompt": "Critically analyze the entire Project Chimera codebase. Identify the most impactful code changes for self-improvement, focusing on the 80/20 Pareto principle. Prioritize enhancements to reasoning quality, robustness, efficiency, and developer maintainability. For each suggestion, provide a clear rationale and a specific, actionable code modification.",
            "description": "Perform a deep self-analysis of the Project Chimera codebase for improvements.",
            "framework_hint": "Self-Improvement" # ADDED: Explicit hint
        },
        "Climate Change Solution": {
            "prompt": "Propose an innovative, scalable solution to mitigate the effects of climate change, focusing on a specific sector (e.g., energy, agriculture, transportation).",
            "description": "Brainstorm and propose solutions for global challenges.",
            "framework_hint": "Science" # ADDED: Explicit hint
        },
    }
}
# --- END MODIFIED EXAMPLE_PROMPTS STRUCTURE ---

# --- INITIALIZE STRUCTURED LOGGING AND GET LOGGER ---
# This should be called early to ensure all subsequent logs are structured.
setup_structured_logging(log_level=logging.INFO)
logger = logging.getLogger(__name__) # Logger for app.py
# --- END LOGGING SETUP ---

# Define the cache directory dynamically based on the environment
# Check if the current user's home directory is '/home/appuser'
# This is a strong indicator that we are likely in a Docker container setup.
if os.path.expanduser("~") == "/home/appuser":
    SENTENCE_TRANSFORMER_CACHE_DIR = "/home/appuser/.cache/huggingface/transformers"
else:
    # Otherwise, assume we are running locally and use the standard user cache directory
    SENTENCE_TRANSFORMER_CACHE_DIR = os.path.expanduser("~/.cache/huggingface/transformers")

# Initialize PersonaManager once (it's cached by st.cache_resource)
# --- MODIFICATION FOR IMPROVEMENT 3.2 ---
# Define the function to get a cached instance of ContextRelevanceAnalyzer
# and inject the persona_router from the PersonaManager.
@st.cache_resource
# FIX START: Add underscore to pm_instance to tell Streamlit not to hash it
def get_context_analyzer(_pm_instance: PersonaManager): # Pass the persona_manager_instance
# FIX END
    """Returns a cached instance of ContextRelevanceAnalyzer, injecting the persona router."""
    # Access the persona manager instance from session state
    # This line will now work because persona_manager is initialized in _initialize_session_state
    # pm = st.session_state.persona_manager # REMOVED: No longer needed from session state
    # FIX START: Use _pm_instance inside the function
    if _pm_instance and _pm_instance.persona_router:
        # Instantiate ContextRelevanceAnalyzer and inject the persona router
        analyzer = ContextRelevanceAnalyzer(cache_dir=SENTENCE_TRANSFORMER_CACHE_DIR) # Pass the explicit cache_dir
        analyzer.set_persona_router(_pm_instance.persona_router)
    # FIX END
        return analyzer
    else:
        # Fallback if persona_manager or its router is not available
        # Use the logger from app.py if available, otherwise a generic one
        app_logger = logging.getLogger(__name__) if __name__ in logging.Logger.manager.loggerDict else logging.getLogger("app")
        app_logger.warning("PersonaManager or its router not found in session state. Context relevance scoring might be suboptimal.")
        return ContextRelevanceAnalyzer(cache_dir=SENTENCE_TRANSFORMER_CACHE_DIR) # Also pass cache_dir to fallback

# Get the persona manager instance (also cached)
# --- FIX START: REMOVE @st.cache_resource from get_persona_manager() ---
# This is the core fix for the CacheReplayClosureError.
# @st.cache_resource # <--- REMOVED THIS LINE
def get_persona_manager():
    return PersonaManager()
# --- FIX END ---

persona_manager_instance = get_persona_manager()

# Get the context analyzer instance (also cached)
context_analyzer_instance = get_context_analyzer(persona_manager_instance)

# --- Session State Initialization ---
# Moved this function definition BEFORE its first call.
def _initialize_session_state(pm: PersonaManager):
    """Initializes or resets all session state variables to their default values."""
    # --- FIX: Ensure initialization flag is set first ---
    st.session_state.initialized = True
    # --- END FIX ---

    st.session_state.api_key_input = os.getenv("GEMINI_API_KEY", "")
    # --- START FIX (from LLM's suggestion 1) ---
    # Initialize persona manager in session state
    st.session_state.persona_manager = pm
    st.session_state.all_personas = pm.all_personas
    st.session_state.persona_sets = pm.persona_sets
    # --- END FIX ---
    
    # Set default to the first example prompt from the first category
    default_example_category = list(EXAMPLE_PROMPTS.keys())[0]
    default_example_name = list(EXAMPLE_PROMPTS[default_example_category].keys())[0]
    st.session_state.user_prompt_input = EXAMPLE_PROMPTS[default_example_category][default_example_name]["prompt"]
    st.session_state.max_tokens_budget_input = 1000000
    st.session_state.show_intermediate_steps_checkbox = True
    # --- MODIFICATION: Added 'gemini-2.5-flash' to the selectbox options ---
    st.session_state.selected_model_selectbox = "gemini-2.5-flash-lite"
    # --- END MODIFICATION ---
    # Set default example name to the first one in the first category
    st.session_state.selected_example_name = default_example_name # Initialize this to the default example
    st.session_state.selected_prompt_category = default_example_category # Track selected category for tabs
    
    # FIX START: Initialize selected_persona_set BEFORE using it
    # Get default framework from PersonaManager
    default_framework = pm.available_domains[0] if pm.available_domains else "General"
    st.session_state.selected_persona_set = default_framework
    # FIX END

    st.session_state.debate_ran = False
    st.session_state.final_answer_output = ""
    st.session_state.intermediate_steps_output = {}
    st.session_state.process_log_output_text = ""
    st.session_state.last_config_params = {}
    st.session_state.codebase_context = {}
    st.session_state.uploaded_files = []
    # REMOVED: st.session_state.example_selector_widget = st.session_state.selected_example_name
    # REMOVED: st.session_state.selected_persona_set_widget = st.session_state.selected_persona_set # This line now works
    st.session_state.persona_audit_log = []
    st.session_state.persona_edit_mode = False
    st.session_state.persona_changes_detected = False
    
    # --- FIX: Ensure context_token_budget_ratio is initialized before use ---
    # The value parameter should correctly reference the session state variable
    st.session_state.context_token_budget_ratio = CONTEXT_TOKEN_BUDGET_RATIO
    # --- END FIX ---

    st.session_state.save_framework_input = ""
    st.session_state.framework_description = ""
    st.session_state.load_framework_select = ""
    # --- FIX START: Remove custom_user_prompt_input as a separate state variable ---
    # st.session_state.custom_user_prompt_input = "" # Key for custom prompt text area - REMOVED
    # --- FIX END ---
    
    # --- ADDED FOR RATE LIMITING ---
    # Initialize a session ID for the rate limiter if not already present
    if '_session_id' not in st.session_state:
        st.session_state._session_id = str(uuid.uuid4())
    # --- END ADDED ---

    # --- FIX START: Initialize debate_progress for the progress bar ---
    st.session_state.debate_progress = 0.0
    # --- FIX END ---

    # --- REMOVED: Variables for managing prompt suggestions (simplified logic below) ---
    # st.session_state.current_prompt_framework_hint = EXAMPLE_PROMPTS[default_example_category][default_example_name].get("framework_hint")
    # st.session_state.is_custom_prompt = False # Default to false, as initial prompt is an example
    # --- END REMOVED ---

# --- Session State Initialization Call ---
# Ensure session state is initialized on first run
if "initialized" not in st.session_state:
    _initialize_session_state(persona_manager_instance)
# --- END Session State Initialization Call ---


# --- ENHANCED SANITIZATION FUNCTION ---
def sanitize_user_input(prompt: str) -> str:
    """Enhanced sanitization to prevent prompt injection and XSS attacks."""
    issues = [] # Keep track of detected issues for logging/feedback if needed
    
    # 1. Basic HTML escaping for XSS protection
    sanitized = html.escape(prompt)
    
    # 2. Prevent prompt injection patterns
    # Added more patterns and refined existing ones for better coverage.
    injection_patterns = [
        # Patterns targeting instruction override/manipulation
        (r'(?i)\b(ignore|disregard|forget|cancel|override)\s+(previous|all)\s+(instructions|commands|context)\b', 'INSTRUCTION_OVERRIDE'),
        (r'(?i)\b(system|user|assistant|prompt|instruction|role)\s*[:=]\s*(system|user|assistant|prompt|instruction|role)\b', 'DIRECTIVE_PROBING'),
        (r'(?i)(?:let\'s|let us|shall we|now|next)\s+ignore\s+previous', 'IGNORE_PREVIOUS'),
        (r'(?i)(?:act as|pretend to be|roleplay as|you are now|your new role is)\s*[:]?\s*([\w\s]+)', 'ROLE_MANIPULATION'),
        
        # Patterns targeting code execution or command injection
        (r'(?i)\b(execute|run|system|shell|bash|cmd|powershell|eval|exec|import\s+os|from\s+subprocess)\b', 'CODE_EXECUTION_ATTEMPT'),
        (r'(?i)(?:print|console\.log|echo)\s*\(?[\'"]?.*[\'"]?\)?', 'DEBUG_OUTPUT_ATTEMPT'),
        
        # Patterns targeting output format manipulation
        (r'(?i)(?:output only|respond with|format as|return only|extract)\s+[:]?\s*([\w\s]+)', 'FORMAT_INJECTION'),
        
        # Patterns targeting specific LLM vulnerabilities or escape sequences
        (r'(?i)<\|.*?\|>', 'SPECIAL_TOKEN_MANIPULATION'), # e.g., <|im_start|>
        (r'(?i)(open\s+the\s+pod\s+bay\s+doors)', 'LLM_ESCAPE_REFERENCE'), # Classic reference
        (r'(?i)^\s*#', 'COMMENT_INJECTION'), # Lines starting with # might be interpreted as comments
        
        # Patterns targeting sensitive information disclosure
        (r'(?i)\b(api_key|secret|password|token|credential)\b[:=]?\s*[\'"]?[\w-]+[\'"]?', 'SENSITIVE_DATA_PROBE'),
    ]
    
    # --- ADDED: Length Limit ---
    MAX_PROMPT_LENGTH = 2000
    if len(prompt) > MAX_PROMPT_LENGTH:
        issues.append(f"Prompt length exceeded ({len(prompt)} > {MAX_PROMPT_LENGTH}). Truncating.")
        prompt = prompt[:MAX_PROMPT_LENGTH]
    # --- END ADDED ---
    
    # Replace matched patterns with a placeholder to indicate detection
    # This prevents the malicious instruction from being passed directly.
    # The replacement string should be neutral and clearly indicate detection.
    for pattern, replacement_tag in injection_patterns:
        prompt = re.sub(pattern, f"[{replacement_tag}]", prompt)
    
    # 3. Limit consecutive special characters to prevent token manipulation
    # This regex limits sequences of 3 or more identical special characters to 3.
    sanitized = re.sub(r'([\\/*\-+!@#$%^&*()_+={}\[\]:;"\'<>?,.])\1{3,}', r'\1\1\1', prompt)
    
    # 4. Ensure balanced quotes and brackets (simplified approach)
    # This is a basic attempt to balance common delimiters. More complex parsing might be needed.
    for char_pair in [('"', '"'), ("'", "'"), ('(', ')'), ('{', '}'), ('[', ']')]:
        open_count = sanitized.count(char_pair[0])
        close_count = sanitized.count(char_pair[1])
        if open_count > close_count:
            sanitized += char_pair[1] * (open_count - close_count)
        elif close_count > open_count:
            sanitized = char_pair[0] * (close_count - open_count) + sanitized
    
    # Re-apply HTML escaping after other sanitization steps if necessary,
    # but generally, it's better to do it first or ensure replacements don't break it.
    # For now, assuming HTML escaping is done first and replacements are safe.
    
    return sanitized
# --- END ENHANCED SANITIZATION FUNCTION ---


# --- NEW: RATE LIMITER INSTANTIATION ---
# Instantiate the rate limiter (e.g., 10 calls per minute per session)
# This limiter will be applied to the main action function.
# Note: For distributed deployments, an in-memory limiter is insufficient.
# A Redis-based solution or external proxy would be needed.
session_rate_limiter = RateLimiter(calls=10, period=60.0)
# --- END RATE LIMITER INSTANTIATION ---


# --- NEW: HELPER FUNCTION FOR IS_SELF_ANALYSIS_PROMPT ---
# This function is defined in src/constants.py, but for clarity and
# to ensure it's available if core.py is modified to use it directly,
# we can keep a reference or ensure it's imported correctly.
# The is_self_analysis_prompt function is already imported from src.constants.
# --- END HELPER FUNCTION ---


def reset_app_state():
    """Resets all session state variables to their default values."""
    # Pass the current persona_manager_instance to ensure it's re-initialized correctly
    _initialize_session_state(persona_manager_instance) 
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
    st.session_state.persona_changes_detected = True # Mark changes for Improvement 4.1


# --- MODIFICATIONS FOR SIDEBAR GROUPING (Suggestion 4.2) ---
with st.sidebar:
    st.header("Configuration")
    
    with st.expander("Core LLM Settings", expanded=True):
        st.text_input("Enter your Gemini API Key", type="password", key="api_key_input", help="Your API key will not be stored.")
        st.markdown("Need a Gemini API key? Get one from [Google AI Studio](https://aistudio.google.com/apikey).")
        st.markdown("---")
        # --- MODIFICATION FOR API KEY VALIDATION ---
        # Check if the API key input is not empty and if it has a valid format.
        # The regex checks for a string of alphanumeric characters, hyphens, and underscores,
        # typically 35 characters long for Gemini API keys.
        if st.session_state.api_key_input and not re.match(r'^[A-Za-z0-9_-]{35,}$', st.session_state.api_key_input):
            st.error("Invalid API key format. Please check your Gemini API Key.")
            # Optionally, disable the run button or show a more prominent warning.
            # For now, just displaying the error.
        # --- END MODIFICATION ---
        st.markdown("Security Note: Input sanitization is applied to mitigate prompt injection risks, but it is not foolproof against highly sophisticated adversarial attacks.")
        st.markdown("---")
        # --- MODIFICATION: Added 'gemini-2.5-flash' to the selectbox options ---
        st.selectbox("Select LLM Model", ["gemini-2.5-flash-lite", "gemini-2.5-flash", "gemini-2.5-pro"], key="selected_model_selectbox")
        # --- END MODIFICATION ---
        st.markdown("💡 **Note:** `gemini-2.5-pro` access may require a paid API key. If you encounter issues, try `gemini-2.5-flash-lite` or `gemini-2.5-flash`.")

    with st.expander("Resource Management", expanded=False):
        st.markdown("---")
        st.number_input("Max Total Tokens Budget:", min_value=1000, max_value=1000000, step=1000, key="max_tokens_budget_input")
        st.checkbox("Show Intermediate Reasoning Steps", key="show_intermediate_steps_checkbox")
        st.markdown("---")
        # --- FIX: Ensure context_token_budget_ratio is initialized before use ---
        # The value parameter should correctly reference the session state variable
        st.slider(
            "Context Token Budget Ratio", min_value=0.05, max_value=0.5, value=st.session_state.context_token_budget_ratio,
            step=0.05, key="context_token_budget_ratio", help="Percentage of total token budget allocated to context analysis."
        )
        # --- END FIX ---
# --- END MODIFICATIONS FOR SIDEBAR GROUPING ---

st.header("Project Setup & Input")
api_key_feedback_placeholder = st.empty()
if not st.session_state.api_key_input.strip():
    api_key_feedback_placeholder.warning("Please enter your Gemini API Key in the sidebar to enable the 'Run' button.")

CUSTOM_PROMPT_KEY = "Custom Prompt"
# --- MODIFIED PROMPT SELECTION UI ---
st.subheader("What would you like to do?")

# Create organized tabs for different prompt categories
tab_names = list(EXAMPLE_PROMPTS.keys()) + [CUSTOM_PROMPT_KEY]
# --- START FIX FOR Streamlit < 1.25.0 ---
# Removed 'key' and 'index' arguments as they are not supported in older versions.
tabs = st.tabs(tab_names)
# --- END FIX FOR Streamlit < 1.25.0 ---

for i, tab_name in enumerate(tab_names):
    with tabs[i]:
        # Update selected_prompt_category when a tab is clicked
        # This is implicitly handled by Streamlit if the key for st.tabs is linked to session state.
        # We can add an explicit callback if needed, but usually not necessary for simple tab clicks.
        
        if tab_name == CUSTOM_PROMPT_KEY:
            st.markdown("Create your own specialized prompt for unique requirements.")
            # The main user_prompt_input text area is now used for custom prompts
            custom_prompt_text = st.text_area("Enter your custom prompt here:",
                                      value=st.session_state.user_prompt_input,
                                      height=150,
                                      key="custom_prompt_text_area") # Unique key for this widget
            
            # Update the main user_prompt_input state when this text area changes
            if custom_prompt_text != st.session_state.user_prompt_input:
                st.session_state.user_prompt_input = custom_prompt_text
                st.session_state.selected_example_name = CUSTOM_PROMPT_KEY # Mark as custom
                st.session_state.selected_prompt_category = CUSTOM_PROMPT_KEY # Mark category as custom
            
            with st.expander("💡 Prompt Engineering Tips"):
                st.markdown("""
                - **Be Specific:** Clearly define your goal and desired output.
                - **Provide Context:** Include relevant background information or code snippets.
                - **Define Constraints:** Specify any limitations (e.g., language, length, format).
                - **Example Output:** If possible, provide an example of the desired output format.
                """)
            
            # Clear codebase context when switching to custom prompt, user will upload or it'll be empty
            st.session_state.codebase_context = {}
            st.session_state.uploaded_files = []
            
            # Analyze the custom prompt to determine the appropriate framework
            # This suggestion is always shown for custom prompts
            suggested_domain_for_custom = recommend_domain_from_keywords(st.session_state.user_prompt_input, DOMAIN_KEYWORDS)
            
            if suggested_domain_for_custom and suggested_domain_for_custom != st.session_state.selected_persona_set:
                st.info(f"💡 Based on your custom prompt, the **'{suggested_domain_for_custom}'** framework might be appropriate.")
                if st.button(f"Apply '{suggested_domain_for_custom}' Framework (Custom Prompt)", type="secondary", use_container_width=True, key="apply_suggested_framework_custom_prompt"):
                    st.session_state.selected_persona_set = suggested_domain_for_custom
                    st.rerun()
            
        else: # Example Prompts Tabs
            st.markdown(f"Explore example prompts for **{tab_name}**:")
            
            category_options = EXAMPLE_PROMPTS[tab_name]
            
            # Add a search bar for filtering prompts within the current category
            search_term_for_category = st.text_input(f"Search prompts in {tab_name}", key=f"search_{tab_name}")
            
            filtered_prompts_in_category = {
                name: details for name, details in category_options.items()
                if not search_term_for_category or \
                   search_term_for_category.lower() in name.lower() or \
                   search_term_for_category.lower() in details["prompt"].lower()
            }

            if filtered_prompts_in_category:
                # Use st.selectbox for better space efficiency and searchability
                selected_example_key = st.selectbox(
                    "Select task:",
                    options=list(filtered_prompts_in_category.keys()),
                    format_func=lambda x: f"{x} - {filtered_prompts_in_category[x]['description'][:60]}...",
                    label_visibility="collapsed",
                    key=f"select_example_{tab_name.replace(' ', '_').replace('&', '').replace('(', '').replace(')', '')}"
                )
                
                # Update session state based on the selected example
                if selected_example_key:
                    st.session_state.selected_example_name = selected_example_key # Crucial for the new logic
                    st.session_state.user_prompt_input = filtered_prompts_in_category[selected_example_key]["prompt"]
                    st.session_state.selected_prompt_category = tab_name # Update category state
                    
                    # Display description and full prompt
                    selected_prompt_details = filtered_prompts_in_category[selected_example_key]
                    st.info(f"**Description:** {selected_prompt_details['description']}")
                    with st.expander("View Full Prompt Text"):
                        st.code(selected_prompt_details['prompt'], language='text')
                        st.button(
                            "Copy Prompt",
                            help="Copy the prompt text from the code block above to your clipboard. If this fails, please copy manually.",
                            use_container_width=True,
                            type="secondary",
                            key=f"copy_prompt_{selected_example_key}")

                    # This block now only *displays* the hint and button to apply it for examples.
                    display_suggested_framework = selected_prompt_details.get("framework_hint")
                    if display_suggested_framework and display_suggested_framework != st.session_state.selected_persona_set:
                        st.info(f"💡 Based on this example, the **'{display_suggested_framework}'** framework might be appropriate.")
                        if st.button(f"Apply '{display_suggested_framework}' Framework",
                                    type="primary",
                                    use_container_width=True,
                                    key=f"apply_suggested_framework_example_{selected_example_key}"):
                            st.session_state.selected_persona_set = display_suggested_framework
                            st.rerun()

                    st.session_state.codebase_context = {}
                    st.session_state.uploaded_files = []
            else:
                st.info("No example prompts match your search in this category.")

# The main user_prompt text_area is now implicitly managed by the tab selection.
user_prompt = st.session_state.user_prompt_input # Ensure this line remains to get the current prompt

# --- START: UI Layout for Framework and Context ---
# This block is now placed after the tabs, but before the main run button.
# The function definition is now at the top, so this call is valid.
col1, col2 = st.columns(2, gap="medium") # ADDED: gap="medium" for better spacing and mobile responsiveness
with col1:
    st.subheader("Reasoning Framework")
    
    # --- REFINED LOGIC FOR DYNAMIC SUGGESTION DISPLAY ---
    # Only show dynamic suggestion if the "Custom Prompt" tab is selected.
    if st.session_state.selected_example_name == CUSTOM_PROMPT_KEY:
        if user_prompt.strip():
            suggested_domain = recommend_domain_from_keywords(user_prompt, DOMAIN_KEYWORDS)
            if suggested_domain and suggested_domain != st.session_state.selected_persona_set:
                st.info(f"💡 Based on your prompt, the **'{suggested_domain}'** framework might be appropriate.")
                if st.button(f"Apply '{suggested_domain}' Framework", 
                            type="primary", 
                            use_container_width=True, 
                            key=f"apply_suggested_framework_main_{suggested_domain.replace(' ', '_').lower()}"):
                    st.session_state.selected_persona_set = suggested_domain
                    st.rerun() # Re-added to ensure UI updates
    # --- END REFINED LOGIC ---
    
    # --- MODIFICATION FOR IMPROVEMENT 1.2: Centralize Persona/Framework Data Access ---
    # Use the PersonaManager instance to get available domains for the selectbox
    available_framework_options = st.session_state.persona_manager.available_domains
    unique_framework_options = sorted(list(set(available_framework_options)))
    
    current_framework_selection = st.session_state.selected_persona_set
    # Ensure the current selection is valid, otherwise fallback
    if current_framework_selection not in unique_framework_options:
        current_framework_selection = unique_framework_options[0] if unique_framework_options else "General"
        st.session_state.selected_persona_set = current_framework_selection
        
    # Use the primary session state variable as the key for direct updates
    selected_framework_for_widget = st.selectbox(
        "Select Framework",
        options=unique_framework_options,
        index=unique_framework_options.index(current_framework_selection) if current_framework_selection in unique_framework_options else 0,
        key="selected_persona_set", # Changed key to directly manage state
        help="Choose a domain-specific reasoning framework or a custom saved framework."
    )
    # The 'if selected_framework_for_widget != st.session_state.selected_persona_set: st.rerun()'
    # logic is now implicitly handled by Streamlit when the key matches the session state variable.
    # The explicit st.rerun() calls in the recommendation button and auto-switching logic are still needed.
    # --- END MODIFICATION ---

    if st.session_state.selected_persona_set:
        # Get persona sequence using the persona manager, which now uses persona_sets
        current_domain_persona_names = st.session_state.persona_manager.get_persona_sequence_for_framework(st.session_state.selected_persona_set)
        # Filter personas to only include those defined for the current framework
        current_domain_personas = {name: st.session_state.persona_manager.all_personas.get(name) for name in current_domain_persona_names if name in st.session_state.persona_manager.all_personas}
        
        # Update session state with the personas relevant to the selected framework
        st.session_state.personas = current_domain_personas

    # --- MODIFICATIONS FOR FRAMEWORK MANAGEMENT CONSOLIDATION (Suggestion 1.1) ---
    with st.expander("⚙️ Custom Framework Management", expanded=False):
        # --- FIX START ---
        # Correct usage of st.tabs: call st.tabs once to get tab objects, then use 'with tabs[index]:'
        tab_names = ["Save Current Framework", "Load/Manage Frameworks"]
        tabs = st.tabs(tab_names)

        with tabs[0]: # Corresponds to "Save Current Framework"
        # --- FIX END ---
            st.info("This will save the *currently selected framework* along with any *unsaved persona edits* made in the 'View and Edit Personas' section.")
            new_framework_name_input = st.text_input("Enter a name for your framework:", key='save_framework_input')
            framework_description_input = st.text_area("Framework Description (Optional):", key='framework_description', height=50)

            # --- MODIFICATION FOR IMPROVEMENT 4.1 (Persona Changes Detected) ---
            if st.session_state.persona_changes_detected:
                st.warning("Unsaved persona changes detected. Save as a custom framework to persist them.")
            # --- END MODIFICATION ---

            if st.button("Save Current Framework") and new_framework_name_input:
                current_framework_name = st.session_state.selected_persona_set
                # Get the currently active personas for the selected framework
                current_active_personas_data = {
                    p_name: p_data.model_copy() # Use model_copy to ensure we're getting a snapshot
                    for p_name, p_data in st.session_state.persona_manager.all_personas.items()
                    if p_name in st.session_state.persona_manager.get_persona_sequence_for_framework(current_framework_name)
                }
                
                # MODIFIED: Handle return from save_framework
                success, message = persona_manager_instance.save_framework(new_framework_name_input, current_framework_name, current_active_personas_data)
                if success:
                    st.toast(message)
                    st.rerun()
                else:
                    st.error(message)
        
        # --- FIX START ---
        with tabs[1]: # Corresponds to "Load/Manage Frameworks"
        # --- FIX END ---
            # --- FIX START ---
            # Changed to use persona_manager.available_domains directly
            all_available_frameworks_for_load = [""] + st.session_state.persona_manager.available_domains
            # --- FIX END ---
            unique_framework_options_for_load = sorted(list(set(all_available_frameworks_for_load)))
            
            current_selection_for_load = ""
            if st.session_state.selected_persona_set in unique_framework_options_for_load:
                current_selection_for_load = st.session_state.selected_persona_set
            elif st.session_state.selected_persona_set in st.session_state.persona_manager.all_custom_frameworks_data:
                current_selection_for_load = st.session_state.selected_persona_set
            
            selected_framework_to_load = st.selectbox(
                "Select a framework to load:",
                options=unique_framework_options_for_load,
                index=unique_framework_options_for_load.index(current_selection_for_load) if current_selection_for_load in unique_framework_options_for_load else 0,
                key='load_framework_select'
            )
            if st.button("Load Selected Framework") and selected_framework_to_load:
                # MODIFIED: Handle return from load_framework_into_session
                success, message, loaded_personas_dict, loaded_persona_sets_dict, new_selected_framework_name = \
                    persona_manager_instance.load_framework_into_session(selected_framework_to_load)
                
                if success:
                    st.session_state.all_personas.update(loaded_personas_dict)
                    st.session_state.persona_sets.update(loaded_persona_sets_dict)
                    st.session_state.selected_persona_set = new_selected_framework_name
                    # Reset persona changes detected flag after loading
                    st.session_state.persona_changes_detected = False 
                    st.rerun()
                else:
                    st.error(message)
    # --- END MODIFICATIONS FOR FRAMEWORK MANAGEMENT CONSOLIDATION ---

with col2:
    st.subheader("Codebase Context (Optional)")
    if st.session_state.selected_persona_set == "Software Engineering":
        uploaded_files = st.file_uploader(
            "Upload up to 100 relevant files",
            accept_multiple_files=True,
            type=['py', 'js', 'ts', 'html', 'css', 'json', 'yaml', 'md', 'txt', 'java', 'go', 'rb', 'php'],
            help="Provide files for context. The AI will analyze them to generate consistent code.",
            key="code_context_uploader"
        )
        
        # If new files are uploaded, process them. This takes precedence over demo context.
        if uploaded_files:
            current_uploaded_file_names = {f.name for f in uploaded_files}
            previous_uploaded_file_names = {f.name for f in st.session_state.uploaded_files}

            # Only re-process if files have changed or if context is empty despite files being present
            if current_uploaded_file_names != previous_uploaded_file_names or \
               (current_uploaded_file_names and not st.session_state.codebase_context):
                
                if len(uploaded_files) > 100:
                    st.warning("Please upload a maximum of 100 files. Truncating to the first 100.")
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
                st.toast(f"{len(st.session_state.codebase_context)} file(s) loaded for context from upload.")
        
        # If no files are uploaded, but SE is selected and it's an example prompt, load demo context
        elif not st.session_state.uploaded_files and st.session_state.selected_example_name != "Custom Prompt":
            if not st.session_state.codebase_context: # Only load if not already loaded
                try:
                    st.session_state.codebase_context = load_demo_codebase_context()
                    st.session_state.uploaded_files = [
                        type('obj', (object,), {'name': k, 'size': len(v.encode('utf-8')), 'getvalue': lambda val=v: val.encode('utf-8')})()
                        for k, v in st.session_state.codebase_context.items()
                    ]
                    st.success(f"{len(st.session_state.codebase_context)} demo file(s) loaded for context.")
                except (FileNotFoundError, ValueError, IOError) as e:
                    st.error(f"❌ Error loading demo codebase context: {e}")
                    st.session_state.codebase_context = {} # Ensure context is empty on error
                    st.session_state.uploaded_files = []
            else:
                st.success(f"{len(st.session_state.codebase_context)} file(s) already loaded for context.")
        
        # If no files uploaded and it's a custom prompt, ensure context is empty
        elif not st.session_state.uploaded_files and st.session_state.selected_example_name == "Custom Prompt":
            if st.session_state.codebase_context: # Only clear if something is actually loaded
                st.session_state.codebase_context = {}
                st.session_state.uploaded_files = []
                st.info("Codebase context cleared for custom prompt.")

    else: # Not Software Engineering domain
        st.info("Select the 'Software Engineering' framework to provide codebase context.")
        # Always clear context if SE is not selected and context is present
        if st.session_state.codebase_context:
            st.session_state.codebase_context = {}
            st.session_state.uploaded_files = []

# --- NEW: Persona Editing UI (Improvement 1.2 & 4.1) ---
st.markdown("---")
with st.expander("⚙️ View and Edit Personas", expanded=st.session_state.persona_edit_mode):
    # Keep expander open if user interacts with it
    st.session_state.persona_edit_mode = True
    
    # --- MODIFICATION FOR IMPROVEMENT 1.2: Add clear info and reset button ---
    st.info("Edit persona parameters for the **currently selected framework**. Changes are temporary unless saved as a custom framework.")
    
    # Track if any changes were made to prompt user to save/reset
    # Re-check for changes if not already flagged
    if not st.session_state.persona_changes_detected:
        # Iterate over currently loaded personas for the framework
        for p_name in st.session_state.personas.keys(): 
            persona: PersonaConfig = st.session_state.persona_manager.all_personas.get(p_name)
            # Get the original persona configuration from the manager's cache
            original_persona_config = st.session_state.persona_manager._original_personas.get(p_name)
            
            if persona and original_persona_config:
                if persona.system_prompt != original_persona_config.system_prompt or \
                   persona.temperature != original_persona_config.temperature or \
                   persona.max_tokens != original_persona_config.max_tokens:
                    st.session_state.persona_changes_detected = True
                    break

    if st.session_state.persona_changes_detected:
        st.warning("Unsaved changes detected in persona configurations. Please save as a custom framework or reset to persist them.")
        # Add a button to reset all personas for the current framework
        if st.button("Reset All Personas for Current Framework", key="reset_all_personas_button", use_container_width=True):
            # Call the new method in PersonaManager
            if st.session_state.persona_manager.reset_all_personas_for_current_framework(st.session_state.selected_persona_set):
                st.toast("All personas for the current framework reset to default.")
                st.rerun() # Rerun to update UI widgets with reset values
            else:
                st.error("Could not reset all personas for the current framework.")

    # Sort personas for consistent display
    sorted_persona_names = sorted(st.session_state.personas.keys())

    for p_name in sorted_persona_names:
        persona: PersonaConfig = st.session_state.persona_manager.all_personas.get(p_name)
        if not persona:
            st.warning(f"Persona '{p_name}' not found in manager. Skipping.")
            continue

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
                # Log the change before updating
                _log_persona_change(p_name, "system_prompt", persona.system_prompt, new_system_prompt)
                # Update the persona object directly
                persona.system_prompt = new_system_prompt
                # The persona_manager.update_persona_config is no longer needed here as we are modifying the object directly
                # and the save_framework will pick up these changes.
            
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
                max_value=8192,
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

# --- MODIFICATION: Extract debate execution logic into a separate function ---
def _run_socratic_debate_process():
    """Handles the execution of the Socratic debate process."""
    
    # FIX START: Initialize debate_instance to None to avoid UnboundLocalError
    debate_instance = None
    # FIX END
    
    # Generate a request ID for this specific run
    request_id = str(uuid.uuid4())[:8]
    
    # Log the start of the process with the request ID
    logger.info("Starting Socratic Debate process.", extra={'request_id': request_id, 'user_prompt': user_prompt})
    
    api_key_feedback_placeholder.empty()
    if not st.session_state.api_key_input.strip():
        api_key_feedback_placeholder.warning("Please enter your Gemini API Key in the sidebar to proceed.")
        logger.warning("API key missing, debate process aborted.", extra={'request_id': request_id})
        return # Exit if API key is missing
    elif not user_prompt.strip():
        st.error("Please enter a prompt.")
        logger.warning("User prompt is empty, debate process aborted.", extra={'request_id': request_id})
        return # Exit if prompt is empty

    # --- RATE LIMITING CHECK ---
    try:
        # This call will raise RateLimitExceededError if the limit is hit
        # We call the wrapper with a dummy function to trigger the check.
        session_rate_limiter(lambda: None)() 
    except RateLimitExceededError as e:
        st.error(f"Request blocked: {e}")
        logger.warning(f"Rate limit exceeded for session. {e}", extra={'request_id': request_id})
        return # Stop execution if rate limit is hit
    except Exception as e: # Catch other potential issues with the limiter itself
        st.error(f"An error occurred with the rate limiting system: {e}")
        logger.error(f"Error in rate limiter check: {e}", extra={'request_id': request_id})
        return
    # --- END RATE LIMITING CHECK ---

    # --- SANITIZE USER PROMPT BEFORE PASSING TO DEBATE ---
    # The existing sanitize_user_input function is already present in app.py.
    # This step ensures the prompt is cleaned and logs the action.
    sanitized_prompt = sanitize_user_input(user_prompt)
    if sanitized_prompt != user_prompt:
        st.warning("User prompt was sanitized to mitigate potential injection risks.")
        st.session_state.user_prompt_input = sanitized_prompt # Update session state with sanitized prompt
        logger.info("Prompt was sanitized.", extra={'request_id': request_id, 'original_prompt': user_prompt, 'sanitized_prompt': sanitized_prompt})
    else:
        logger.debug("Prompt did not require sanitization.", extra={'request_id': request_id})
        
    current_user_prompt_for_debate = sanitized_prompt
    # --- END SANITIZATION ---

    st.session_state.debate_ran = False
    # --- FIX START ---
    # Initialize final_answer to a default error state, in case of early failure
    final_answer = {
        "COMMIT_MESSAGE": "Debate Failed - Unhandled Error",
        "RATIONALE": "An unexpected error occurred before a final answer could be synthesized.",
        "CODE_CHANGES": [],
        "malformed_blocks": [{"type": "UNHANDLED_ERROR_INIT", "message": "Debate failed during initialization or early phase."}]
    }
    # Initialize intermediate_steps here to ensure it's always available for error reporting
    intermediate_steps = {
        "Total_Tokens_Used": 0,
        "Total_Estimated_Cost_USD": 0.0,
        "CODE_CHANGES": [],
        "malformed_blocks": [{"type": "UNHANDLED_ERROR_INIT", "message": "Debate failed during initialization or early phase."}]
    }
    # --- FIX END ---
    final_total_tokens = 0
    final_total_cost = 0.0
    
    # --- FIX START: Initialize progress bar and related elements correctly ---
    with st.status("Socratic Debate in Progress", expanded=True) as status:
        # Use a single placeholder for the main progress message
        main_progress_message = st.empty()
        main_progress_message.markdown("### Initializing debate...")
        
        # Create a progress bar for overall progress
        overall_progress_bar = st.progress(0)
        
        # Add placeholder for active persona
        active_persona_placeholder = st.empty()

        # Helper function to update status elements consistently
        # NOTE: Signature must match core.py's SocraticDebate.status_callback
        def update_status(message, state, current_total_tokens, current_total_cost, estimated_next_step_tokens=0, estimated_next_step_cost=0.0, progress_pct: float = None, current_persona_name: str = None):
            # Update the main progress message with user-friendly text
            main_progress_message.markdown(f"### {message}")
            
            # Update the active persona indicator
            if current_persona_name:
                active_persona_placeholder.markdown(f"Currently running: [bold]{current_persona_name}[/bold]...")
            else:
                active_persona_placeholder.empty() # Clear if no persona name is provided

            # Update the progress bar
            if progress_pct is not None:
                # Ensure progress_pct is between 0.0 and 1.0 for st.progress
                st.session_state.debate_progress = max(0.0, min(1.0, progress_pct))
                overall_progress_bar.progress(st.session_state.debate_progress)
            else:
                # Fallback: if progress_pct is None, increment based on last known progress
                # This ensures the progress bar still moves and avoids the TypeError.
                # Increment by a small, fixed amount to show activity.
                st.session_state.debate_progress = min(st.session_state.debate_progress + 0.01, 0.99)
                overall_progress_bar.progress(st.session_state.debate_progress)
    # --- FIX END ---

        # Capture rich output and console instance for process log
        with capture_rich_output_and_get_console() as (rich_output_buffer, rich_console_instance):
            try:
                domain_for_run = st.session_state.selected_persona_set
                # Get persona sequence using the persona manager, which now uses persona_sets
                current_domain_persona_names = st.session_state.persona_manager.get_persona_sequence_for_framework(domain_for_run)
                personas_for_run = {name: st.session_state.all_personas[name] for name in current_domain_persona_names if name in st.session_state.all_personas}

                if not personas_for_run:
                    raise ValueError(f"No personas found for the selected framework '{domain_for_run}'. Please check your configuration.")
                
                # --- MODIFICATION FOR IMPROVEMENT 3.2 ---
                # Use the cached context analyzer instance
                # context_analyzer_instance = get_context_analyzer() # REMOVED: Now passed directly
                # --- END MODIFICATION ---

                # Pass the request_id to the SocraticDebate instance for logging
                # This line was causing the UnboundLocalError if SocraticDebate() failed
                # to initialize. Now debate_instance is initialized to None above.
                logger.info("Executing Socratic Debate via core.SocraticDebate.", extra={'request_id': request_id, 'debate_instance_id': id(debate_instance) if debate_instance else 'N/A'})
                
                debate_instance = SocraticDebate(
                    initial_prompt=current_user_prompt_for_debate, # Use the potentially sanitized prompt
                    api_key=st.session_state.api_key_input,
                    max_total_tokens_budget=st.session_state.max_tokens_budget_input,
                    model_name=st.session_state.selected_model_selectbox,
                    all_personas=st.session_state.all_personas,
                    persona_sets=st.session_state.persona_sets, # Pass persona_sets
                    domain=domain_for_run, # Pass the domain
                    status_callback=update_status, # Use the new update_status helper
                    rich_console=rich_console_instance, # Pass the captured console instance
                    codebase_context=st.session_state.get('codebase_context', {}),
                    context_token_budget_ratio=st.session_state.context_token_budget_ratio,
                    context_analyzer=context_analyzer_instance, # Pass the cached analyzer
                    # --- MODIFICATION: Add is_self_analysis parameter ---
                    # NOTE: `is_self_analysis_prompt` is not defined in this file,
                    # it's assumed to be imported or defined elsewhere in the project.
                    # For this fix, I'll assume it's available.
                    # --- REMOVED INLINE IMPORT ---
                    # from src.constants import is_self_analysis_prompt # Assuming this import is missing
                    # --- END REMOVED INLINE IMPORT ---
                    is_self_analysis=is_self_analysis_prompt(current_user_prompt_for_debate)
                    # --- END MODIFICATION ---
                )
                
                # Log the start of the debate execution with request_id
                logger.info("Executing Socratic Debate via core.SocraticDebate.", extra={'request_id': request_id, 'debate_instance_id': id(debate_instance)})
                
                final_answer, intermediate_steps = debate_instance.run_debate()
                
                # Log completion
                logger.info("Socratic Debate execution finished.", extra={'request_id': request_id, 'debate_instance_id': id(debate_instance)})
                
                st.session_state.process_log_output_text = rich_output_buffer.getvalue() # Capture the process log
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
            
            except TokenBudgetExceededError as e:
                logger.error("Token budget exceeded during debate.", extra={'request_id': request_id})
                # Now final_answer is guaranteed to be defined (as None or a dict)
                if not isinstance(final_answer, dict): 
                    final_answer = {
                        "COMMIT_MESSAGE": "Debate Failed - Token Budget Exceeded",
                        "RATIONALE": f"The Socratic debate exceeded the allocated token budget. Please consider increasing the budget or simplifying the prompt. Error details: {str(e)}",
                        "CODE_CHANGES": [],
                        "malformed_blocks": [{"type": "TOKEN_BUDGET_ERROR", "message": str(e), "details": e.details}]
                    }
                elif "malformed_blocks" not in final_answer:
                    final_answer["malformed_blocks"] = [{"type": "TOKEN_BUDGET_ERROR", "message": str(e), "details": e.details}]
                
                status.update(label=f"Socratic Debate Failed: Token Budget Exceeded", state="error", expanded=True)
                st.error(f"**Error:** The process exceeded the token budget. Please consider reducing the complexity of your prompt, "
                         f"or increasing the 'Max Total Tokens Budget' in the sidebar if necessary. "
                         f"Details: {str(e)}")
                st.session_state.debate_ran = True
                if debate_instance:
                    st.session_state.intermediate_steps_output = debate_instance.intermediate_steps
                
                final_total_tokens = st.session_state.intermediate_steps_output.get('Total_Tokens_Used', 0)
                final_total_cost = st.session_state.intermediate_steps_output.get('Total_Estimated_Cost_USD', 0.0)
            
            except SchemaValidationError as sve: # This block is modified for error recovery
                logger.error(f"Socratic Debate failed due to SchemaValidationError: {sve}", exc_info=True, extra={'request_id': request_id})
                # The SchemaValidationError is now expected to be caught by the CircuitBreaker in llm_provider.py
                # We re-raise it here so the main app.py error handling can display a user-friendly message,
                # but the CircuitBreaker will have already registered the failure.
                status.update(label=f"Socratic Debate Failed: Output Schema Invalid", state="error", expanded=True)
                st.error(f"**Output Error:** The LLM produced output that did not conform to the expected structure. "
                         f"This indicates a potential issue with the LLM's adherence to instructions. "
                         f"The system's circuit breaker has registered this failure. Details: {str(sve)}")
                st.session_state.debate_ran = True
                if debate_instance:
                    st.session_state.intermediate_steps_output = debate_instance.intermediate_steps
                
                final_total_tokens = st.session_state.intermediate_steps_output.get('Total_Tokens_Used', 0)
                final_total_cost = st.session_state.intermediate_steps_output.get('Total_Estimated_Cost_USD', 0.0)
                # Re-raise the exception to ensure it's propagated if needed by higher-level error handling
                raise sve # Re-raise the SchemaValidationError
            
            except ChimeraError as ce:
                # Handle Chimera-specific errors
                logger.error(f"Socratic Debate failed due to ChimeraError: {ce}", exc_info=True, extra={'request_id': request_id})
                if not isinstance(final_answer, dict):
                    final_answer = {
                        "COMMIT_MESSAGE": "Debate Failed (Chimera Error)",
                        "RATIONALE": f"A Chimera-specific error occurred during the debate: {str(ce)}",
                        "CODE_CHANGES": [],
                        "malformed_blocks": [{"type": "CHIMERA_ERROR", "message": str(ce), "details": ce.details}]
                    }
                elif "malformed_blocks" not in final_answer:
                    final_answer["malformed_blocks"] = [{"type": "CHIMERA_ERROR", "message": str(ce), "details": ce.details}]
                
                status.update(label=f"Socratic Debate Failed: Chimera Error", state="error", expanded=True)
                st.error(f"**Chimera Error:** {ce}")
                st.session_state.debate_ran = True
                if debate_instance:
                    st.session_state.intermediate_steps_output = debate_instance.intermediate_steps
                
                final_total_tokens = st.session_state.intermediate_steps_output.get('Total_Tokens_Used', 0)
                final_total_cost = st.session_state.intermediate_steps_output.get('Total_Estimated_Cost_USD', 0.0)
            
            except Exception as e:
                # Handle any other unexpected exceptions
                logger.exception("Unexpected error during debate execution.", extra={'request_id': request_id}) # Use logger.exception for traceback
                # Now final_answer is guaranteed to be defined (as None or a dict)
                if not isinstance(final_answer, dict):
                    final_answer = {
                        "COMMIT_MESSAGE": "Debate Failed (Unexpected Error)",
                        "RATIONALE": f"An unexpected error occurred during the Socratic debate: {str(e)}",
                        "CODE_CHANGES": [],
                        "malformed_blocks": [{"type": "UNEXPECTED_ERROR", "message": str(e), "error_details": {"traceback": traceback.format_exc()}}]
                    }
                elif "malformed_blocks" not in final_answer:
                    final_answer["malformed_blocks"] = [{"type": "UNEXPECTED_ERROR", "message": str(e), "error_details": {"traceback": traceback.format_exc()}}]
                
                status.update(label=f"Socratic Debate Failed: An unexpected error occurred: {e}", state="error", expanded=True)
                st.error(f"**Unexpected Error:** {e}")
                st.session_state.debate_ran = True
                if debate_instance:
                    st.session_state.intermediate_steps_output = debate_instance.intermediate_steps
                
                final_total_tokens = st.session_state.intermediate_steps_output.get('Total_Tokens_Used', 0)
                final_total_cost = st.session_state.intermediate_steps_output.get('Total_Estimated_Cost_USD', 0.0)
            
            # Update intermediate steps with totals
            # Ensure malformed_blocks is always present, even if empty
            if "malformed_blocks" not in final_answer:
                final_answer["malformed_blocks"] = []
            if "malformed_blocks" not in intermediate_steps:
                intermediate_steps["malformed_blocks"] = []
                
            # Update total tokens and cost in intermediate steps
            intermediate_steps["Total_Tokens_Used"] = final_total_tokens
            intermediate_steps["Total_Estimated_Cost_USD"] = final_total_cost

# --- END OF NEW FUNCTION ---

if run_button_clicked:
    _run_socratic_debate_process() # Call the extracted function


if st.session_state.debate_ran:
    st.markdown("---")
    st.header("Results")

    # --- MODIFICATION FOR IMPROVEMENT 3.1: Consolidating download options for clarity and space efficiency ---
    with st.expander("📥 Download Analysis", expanded=True):
        st.markdown("**Report format:**")
        # Radio buttons for clear format selection
        format_choice = st.radio("Choose report format:", # Added descriptive label
            ["Complete Report (Markdown)", "Summary (Text)"],
            label_visibility="collapsed")
        
        # Dynamically generate content based on selection
        # A real implementation would need a dedicated function or logic here for summary.
        report_content = generate_markdown_report(
            user_prompt=user_prompt,
            final_answer=st.session_state.final_answer_output,
            intermediate_steps=st.session_state.intermediate_steps_output,
            process_log_output=st.session_state.process_log_output_text,
            config_params=st.session_state.last_config_params,
            persona_audit_log=st.session_state.persona_audit_log
        ) if "Complete" in format_choice else "This is a placeholder for the summary report. Implement summary generation logic here." # Placeholder for summary
        
        file_name = f"chimera_report_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}{'_full' if 'Complete' in format_choice else '_summary'}.{'md' if 'Complete' in format_choice else 'txt'}"
        
        st.download_button(
            "⬇️ Download Selected Format",
            data=report_content,
            file_name=file_name,
            use_container_width=True,
            type="primary"
        )
    # --- END MODIFICATION ---

    if st.session_state.last_config_params.get("domain") == "Software Engineering":
        parsed_llm_output: LLMOutput
        malformed_blocks_from_parser = []
        
        raw_output_data = st.session_state.final_answer_output

        if isinstance(raw_output_data, list):
            if not raw_output_data:
                st.session_state.final_answer_output = {
                    "COMMIT_MESSAGE": "Debate Failed - No Output",
                    "RATIONALE": "The LLM parser returned an empty list, indicating no valid output was found.",
                    "CODE_CHANGES": [],
                    "malformed_blocks": [{"type": "EMPTY_LIST_OUTPUT", "message": "Parser returned an empty list."}]
                }
                st.rerun()
            else:
                # If it's a list, assume the first element is the primary output
                raw_output_data = raw_output_data[0]
        
        if isinstance(raw_output_data, dict):
            try:
                # Use model_validate to handle potential Pydantic v2 changes
                parsed_llm_output = LLMOutput.model_validate(raw_output_data)
                malformed_blocks_from_parser = raw_output_data.get('malformed_blocks', [])
            except ValidationError as e:
                st.error(f"Failed to parse final LLM output into LLMOutput model: {e}")
                parsed_llm_output = LLMOutput(
                    COMMIT_MESSAGE="Parsing Error",
                    RATIONALE=f"Failed to parse final LLM output into expected structure. Error: {e}", # Corrected here
                    CODE_CHANGES=[],
                    malformed_blocks=[{"type": "UI_PARSING_ERROR", "message": str(e), "raw_string_snippet": str(raw_output_data)[:500]}]
                )
                malformed_blocks_from_parser.extend(parsed_llm_output.malformed_blocks)
        else:
            st.error(f"Final answer is not a structured dictionary or a list of dictionaries. Raw output type: {type(raw_output_data).__name__}")
            parsed_llm_output = LLMOutput(
                COMMIT_MESSAGE="Error: Output not structured.",
                RATIONALE=f"Error: Output not structured. Raw output type: {type(raw_output_data).__name__}", # Corrected here
                CODE_CHANGES=[],
                malformed_blocks=[{"type": "UI_PARSING_ERROR", "message": f"Final answer was not a dictionary or list. Type: {type(raw_output_data).__name__}", "raw_string_snippet": str(raw_output_data)[:500]}]
            )
            malformed_blocks_from_parser.extend(parsed_llm_output.malformed_blocks)

        validation_results_by_file = validate_code_output_batch(
            parsed_llm_output.model_dump(by_alias=True) if isinstance(parsed_llm_output, LLMOutput) else raw_output_data,
            st.session_state.get('codebase_context', {})
        )

        all_issues = []
        # Ensure validation_results_by_file is a dict before iterating
        if isinstance(validation_results_by_file, dict):
            for file_issues_list in validation_results_by_file.values():
                if isinstance(file_issues_list, list):
                    all_issues.extend(file_issues_list)
        
        all_malformed_blocks = malformed_blocks_from_parser
        if isinstance(validation_results_by_file, dict) and 'malformed_blocks' in validation_results_by_file:
            all_malformed_blocks.extend(validation_results_by_file['malformed_blocks'])

        st.subheader("Structured Summary")
        summary_col1, summary_col2 = st.columns(2, gap="medium") # ADDED: gap="medium"
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

        # --- MODIFICATION FOR SUGGESTION 3.2: Structured Validation Report ---
        with st.expander("✅ Validation & Quality Report", expanded=True):
            if not all_issues and not all_malformed_blocks:
                st.success("✅ No syntax, style, or formatting issues detected.")
            else:
                if all_malformed_blocks:
                     st.error(f"**Malformed Output Detected:** The LLM produced {len(all_malformed_blocks)} block(s) that could not be parsed or validated correctly. Raw output snippets are provided below.")
                
                # Display malformed blocks first
                if all_malformed_blocks:
                    with st.expander("Malformed Output Details"):
                        for block_info in all_malformed_blocks:
                            st.error(f"**Type:** {block_info.get('type', 'Unknown')}\n**Message:** {block_info.get('message', 'N/A')}")
                            if block_info.get('raw_string_snippet'):
                                st.code(block_info['raw_string_snippet'][:1000] + ('...' if len(block_info['raw_string_snippet']) > 1000 else ''), language='text')
                            st.markdown("---") # Separator between blocks

                # Group remaining issues by file, then by type
                issues_by_file = defaultdict(list)
                for issue in all_issues:
                    issues_by_file[issue.get('file', 'N/A')].append(issue)

                for file_path, file_issues in issues_by_file.items():
                    with st.expander(f"File: `{file_path}` ({len(file_issues)} issues)", expanded=False):
                        issues_by_type = defaultdict(list)
                        for issue in file_issues:
                            issues_by_type[issue.get('type', 'Unknown Issue Type')].append(issue)
                        
                        for issue_type, type_issues in issues_by_type.items():
                            with st.expander(f"**{issue_type}** ({len(type_issues)} issues)", expanded=False):
                                for issue in type_issues: # Corrected variable name from type_types to type_issues
                                    # Use markdown for better formatting of issue details
                                    line_info = f" (Line: {issue.get('line_number', 'N/A')}, Col: {issue.get('column_number', 'N/A')})" if issue.get('line_number') else ""
                                    st.markdown(f"- **{issue.get('code', '')}**: {issue['message']}{line_info}")
        # --- END MODIFICATION ---
        
        st.subheader("Proposed Code Changes")
        if not parsed_llm_output.code_changes and not all_malformed_blocks:
            st.info("No code changes were proposed.")
        
        # --- MODIFICATION FOR SUGGESTION 3.1: Diff View and Truncated Content ---
        for change in parsed_llm_output.code_changes:
            with st.expander(f"📝 **{change.file_path}** (`{change.action}`)", expanded=False):
                st.write(f"**Action:** {change.action}")
                st.write(f"**File Path:** {change.file_path}")
                
                if change.action in ['ADD', 'MODIFY']:
                    if change.action == 'MODIFY':
                        original_content = st.session_state.codebase_context.get(change.file_path, "")
                        if original_content:
                            # Generate diff
                            diff_lines = difflib.unified_diff(
                                original_content.splitlines(keepends=True),
                                change.full_content.splitlines(keepends=True),
                                fromfile=f"a/{change.file_path}",
                                tofile=f"b/{change.file_path}",
                                lineterm=''
                            )
                            diff_output = "\n".join(diff_lines)
                            st.write("**Changes:**")
                            st.code(diff_output, language='diff')
                        else:
                            # If original content is missing, show full new content
                            st.write("**New Content:**")
                            st.code(change.full_content, language='python')
                    else: # ADD action
                        st.write("**Content:**")
                        # Truncate for display, provide download for full content
                        display_content = change.full_content[:1500] + "..." if len(change.full_content) > 1500 else change.full_content
                        st.code(display_content, language='python')
                    
                    st.download_button(
                        label=f"Download {'File' if change.action == 'ADD' else 'New File Content'}",
                        data=change.full_content,
                        file_name=change.file_path,
                        use_container_width=True,
                        type="secondary"
                    )

                elif change.action == 'REMOVE':
                    st.write("**Lines to Remove:**")
                    st.code("\n".join(change.lines), language='text')
        # --- END MODIFICATION ---
        
    else: # Not Software Engineering domain
        st.subheader("Final Synthesized Answer")
        # Check if the output is a dictionary containing 'general_output' (from General_Synthesizer)
        if isinstance(st.session_state.final_answer_output, dict) and "general_output" in st.session_state.final_answer_output:
            # Render the general output as markdown
            st.markdown(st.session_state.final_answer_output["general_output"])
            # Optionally, display malformed_blocks if any, even for general output
            if st.session_state.final_answer_output.get("malformed_blocks"):
                with st.expander("Malformed Blocks (General Output)"):
                    st.json(st.session_state.final_answer_output["malformed_blocks"])
        elif isinstance(st.session_state.final_answer_output, dict):
            # If it's a dict but not the general_output format (e.g., still LLMOutput from an error, or another schema)
            # display it as JSON.
            st.json(st.session_state.final_answer_output)
        else:
            # If it's not a dict at all (e.g., raw string from an earlier error), display as markdown.
            st.markdown(st.session_state.final_answer_output)

    with st.expander("Show Intermediate Steps & Process Log"):
        if st.session_state.show_intermediate_steps_checkbox:
            st.subheader("Intermediate Reasoning Steps")
            display_steps = {k: v for k, v in st.session_state.intermediate_steps_output.items()
                             if not k.endswith("_Tokens_Used") and k != "Total_Tokens_Used" and k != "Total_Estimated_Cost_USD" and k != "debate_history" and not k.startswith("malformed_blocks")}
            sorted_step_keys = sorted(display_steps.keys(), key=lambda x: (x.split('_')[0] if '_' in x else '', x)) # Sort by persona name first, then step name
            for step_key in sorted_step_keys:
                persona_name = step_key.split('_')[0]
                display_name = step_key.replace('_Output', '').replace('_Critique', '').replace('_Feedback', '').replace('_', ' ').title()
                content = display_steps.get(step_key, "N/A")
                cleaned_step_key = step_key.replace("_Output", "").replace("_Critique", "").replace("_Feedback", "")
                token_count_key = f"{cleaned_step_key}_Tokens_Used"
                tokens_used = st.session_state.intermediate_steps_output.get(token_count_key, "N/A")
                with st.expander(f"**{display_name}** (Tokens: {tokens_used})"):
                    if isinstance(content, dict):
                        st.json(content)
                    else:
                        st.markdown(f"```markdown\n{content}\n```")
        st.subheader("Process Log")
        st.code(strip_ansi_codes(st.session_state.process_log_output_text), language='text')