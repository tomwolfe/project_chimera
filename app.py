# -*- coding: utf-8 -*-
# app.py
import streamlit as st
import json
import os
import io
import contextlib
import re
import datetime
import time
from typing import Dict, Any, List, Optional
import yaml
import logging
from rich.console import Console
from core import SocraticDebate

import google.genai as genai
from google.genai.errors import APIError

from src.models import PersonaConfig, ReasoningFrameworkConfig, LLMOutput, CodeChange, ContextAnalysisOutput, CritiqueOutput, GeneralOutput, ConflictReport, SelfImprovementAnalysisOutput, SelfImprovementAnalysisOutputV1
from src.utils import LLMOutputParser, validate_code_output_batch, sanitize_and_validate_file_path
# MODIFIED: Removed recommend_domain_from_keywords from here, as it's now part of PromptAnalyzer
from src.persona_manager import PersonaManager
from src.exceptions import ChimeraError, LLMResponseValidationError, SchemaValidationError, TokenBudgetExceededError, LLMProviderError, CircuitBreakerError
from src.constants import SELF_ANALYSIS_KEYWORDS, is_self_analysis_prompt
from src.context.context_analyzer import ContextRelevanceAnalyzer
import traceback
from collections import defaultdict
from pydantic import ValidationError
import html
import difflib

import uuid
from src.logging_config import setup_structured_logging
from src.middleware.rate_limiter import RateLimiter, RateLimitExceededError
from src.config.settings import ChimeraSettings
from pathlib import Path # Added for Path in Self-Improvement download button

# NEW IMPORT: For centralized prompt analysis
from src.utils.prompt_analyzer import PromptAnalyzer

# --- Configuration Loading ---
@st.cache_resource
def load_config(file_path: str = "config.yaml") -> Dict[str, Any]:
    """Load config with validation and user-friendly errors."""
    if not os.path.exists(file_path):
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
    st.stop()

DOMAIN_KEYWORDS = app_config.get("domain_keywords", {})
CONTEXT_TOKEN_BUDGET_RATIO_FROM_CONFIG = app_config.get("context_token_budget_ratio", 0.25)

# --- NEW: API Key Validation and Test Functions ---
def validate_gemini_api_key_format(api_key: str) -> bool:
    """Validate Gemini API key format."""
    if not api_key or not isinstance(api_key, str):
        return False
    return re.match(r'^[A-Za-z0-9_-]{35,}$', api_key) is not None

def test_gemini_api_key_functional(api_key: str) -> bool:
    """Attempts a simple API call to validate the key's functionality."""
    if not api_key:
        return False
    try:
        test_client = genai.Client(api_key=api_key)
        test_client.models.list()
        return True
    except APIError as e:
        logger.error(f"API key functional test failed: {e}")
        return False
    except Exception as e:
        logger.error(f"Unexpected error during API key functional test: {e}")
        return False

# --- NEW: Callback for API Key Input ---
def on_api_key_change():
    """Callback to validate API key format and update activity timestamp."""
    # Access the current value directly from the widget's key in session_state
    api_key_value = st.session_state.api_key_input
    st.session_state.api_key_valid_format = validate_gemini_api_key_format(api_key_value)
    update_activity_timestamp()
# --- END NEW: Callback for API Key Input ---
# --- END NEW: API Key Validation and Test Functions ---

# --- Demo Codebase Context Loading ---
@st.cache_data
def load_demo_codebase_context(file_path: str = "data/demo_codebase_context.json") -> Dict[str, str]:
    """Loads demo codebase context from a JSON file with enhanced error handling."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"Demo context file not found at '{file_path}'.")
    except json.JSONDecodeError as e:
        raise ValueError(f"Error decoding JSON from '{file_path}'. Please check its format: {e}") from e
    except IOError as e:
        raise IOError(f"IO error reading config file '{file_path}'. Check permissions: {e}") from e

# Redirect rich output to a string buffer for Streamlit display
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
        
        escaped_newline_for_display = '\\n'

        for entry in persona_audit_log:
            old_val_str = str(entry.get('old_value', ''))
            new_val_str = str(entry.get('new_value', ''))
            old_val_display = (old_val_str[:50] + '...') if len(old_val_str) > 50 else old_val_str
            new_val_display = (new_val_str[:50] + '...') if len(new_val_str) > 50 else new_val_str
            
            old_val_display_escaped = old_val_display.replace('\n', escaped_newline_for_display)
            new_val_display_escaped = new_val_display.replace('\n', escaped_newline_for_display)
            md_content += f"| {entry.get('timestamp')} | {entry.get('persona')} | {entry.get('parameter')} | `{old_val_display_escaped}` | `{new_val_display_escaped}` |\n"
        md_content += "\n---\n\n"

    md_content += "## Process Log\n\n"
    md_content += "```text\n"
    md_content += strip_ansi_codes(process_log_output)
    md_content += "\n```\n\n"
    
    if config_params.get('show_intermediate_steps', True):
        md_content += "---\n\n"
        md_content += "## Intermediate Reasoning Steps\n\n"
        step_keys_to_process = sorted([k for k in intermediate_steps.keys()
                                       if not k.endswith("_Tokens_Used") and not k.endswith("_Estimated_Cost_USD") and k != "Total_Tokens_Used" and k != "Total_Estimated_Cost_USD" and k != "debate_history" and not k.startswith("malformed_blocks")],
                                      key=lambda x: (x.split('_')[0] if '_' in x else '', x))
        
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
            "framework_hint": "Software Engineering"
        },
        "Refactor a Python Function": {
            "prompt": "Refactor the given Python function to improve its readability and performance. It currently uses a nested loop; see if you can optimize it.",
            "description": "Improve structure and readability of existing code while maintaining functionality.",
            "framework_hint": "Software Engineering"
        },
        "Fix a Bug in a Script": {
            "prompt": "The provided Python script is supposed to calculate the average of a list of numbers but fails with a `TypeError` if the list contains non-numeric strings. Fix the bug by safely ignoring non-numeric values.",
            "description": "Identify and correct issues in problematic code with explanations.",
            "framework_hint": "Software Engineering"
        },
    },
    "Analysis & Problem Solving": {
        "Design a Mars City": {
            "prompt": "Design a sustainable city for 1 million people on Mars, considering resource scarcity and human psychology.",
            "description": "Explore complex design challenges with multi-faceted considerations.",
            "framework_hint": "Creative"
        },
        "Ethical AI Framework": {
            "prompt": "Develop an ethical framework for an AI system designed to assist in judicial sentencing, addressing bias, transparency, and accountability.",
            "description": "Formulate ethical guidelines for sensitive AI applications.",
            "framework_hint": "Business"
        },
        "Critically analyze the entire Project Chimera codebase. Identify the most impactful code changes for self-improvement, focusing on the 80/20 Pareto principle. Prioritize enhancements to reasoning quality, robustness, efficiency, and developer maintainability. For each suggestion, provide a clear rationale and a specific, actionable code modification.": {
            "prompt": "Critically analyze the entire Project Chimera codebase. Identify the most impactful code changes for self-improvement, focusing on the 80/20 Pareto principle. Prioritize enhancements to reasoning quality, robustness, efficiency, and developer maintainability. For each suggestion, provide a clear rationale and a specific, actionable code modification.",
            "description": "Perform a deep self-analysis of the Project Chimera codebase for improvements.",
            "framework_hint": "Self-Improvement"
        },
        "Climate Change Solution": {
            "prompt": "Propose an innovative, scalable solution to mitigate the effects of climate change, focusing on a specific sector (e.g., energy, agriculture, transportation).",
            "description": "Brainstorm and propose solutions for global challenges.",
            "framework_hint": "Science"
        },
    }
}
# --- END EXAMPLE_PROMPTS STRUCTURE ---

# --- NEW CONSTANTS FOR SESSION MANAGEMENT AND RETRIES ---
SESSION_TIMEOUT_SECONDS = 1800 # 30 minutes of inactivity
MAX_DEBATE_RETRIES = 3 # Max retries for the entire debate process if rate limited
DEBATE_RETRY_DELAY_SECONDS = 5 # Initial delay for debate retries (will be multiplied by attempt number)
# --- END NEW CONSTANTS ---

# --- INITIALIZE STRUCTURED LOGGING AND GET LOGGER ---
setup_structured_logging(log_level=logging.INFO)
logger = logging.getLogger(__name__)
# --- END LOGGING SETUP ---

# Define the cache directory dynamically based on the environment
if os.path.expanduser("~") == "/home/appuser":
    SENTENCE_TRANSFORMER_CACHE_DIR = "/home/appuser/.cache/huggingface/transformers"
else:
    SENTENCE_TRANSFORMER_CACHE_DIR = os.path.expanduser("~/.cache/huggingface/transformers")

# FIX START: Removed @st.cache_resource from get_context_analyzer()
def get_context_analyzer(_pm_instance: PersonaManager):
    """Returns a cached instance of ContextRelevanceAnalyzer, injecting the persona router."""
    if _pm_instance and _pm_instance.persona_router:
        analyzer = ContextRelevanceAnalyzer(cache_dir=SENTENCE_TRANSFORMER_CACHE_DIR)
        analyzer.set_persona_router(_pm_instance.persona_router)
        return analyzer
    else:
        app_logger = logging.getLogger(__name__) if __name__ in logging.Logger.manager.loggerDict else logging.getLogger("app")
        app_logger.warning("PersonaManager or its router not found. Context relevance scoring might be suboptimal.")
        return ContextRelevanceAnalyzer(cache_dir=SENTENCE_TRANSFORMER_CACHE_DIR)

# FIX START: REMOVED @st.cache_resource from get_persona_manager()
def get_persona_manager():
    # PersonaManager now requires DOMAIN_KEYWORDS
    return PersonaManager(DOMAIN_KEYWORDS)
# FIX END

# --- Helper to update activity timestamp ---
def update_activity_timestamp():
    st.session_state.last_activity_timestamp = time.time()
    logger.debug("Activity timestamp updated.")

# --- Session State Initialization ---
def _initialize_session_state():
    """Initializes or resets all session state variables to their default values."""
    defaults = {
        "initialized": True,
        "api_key_input": os.getenv("GEMINI_API_KEY", ""),
        "user_prompt_input": "",
        "max_tokens_budget_input": 1000000,
        "show_intermediate_steps_checkbox": True,
        "selected_model_selectbox": "gemini-2.5-flash-lite",
        "selected_example_name": "",
        "selected_prompt_category": "",
        "active_example_framework_hint": None,
        "debate_ran": False,
        "final_answer_output": "",
        "intermediate_steps_output": {},
        "process_log_output_text": "",
        "last_config_params": {},
        "codebase_context": {},
        "uploaded_files": [],
        "persona_audit_log": [],
        "persona_edit_mode": False,
        "persona_changes_detected": False,
        "context_token_budget_ratio": CONTEXT_TOKEN_BUDGET_RATIO_FROM_CONFIG,
        "save_framework_input": "",
        "framework_description": "",
        "load_framework_select": "",
        "_session_id": str(uuid.uuid4()),
        "debate_progress": 0.0,
        "api_key_valid_format": False,
        "api_key_functional": False,
        "current_debate_tokens_used": 0,
        "current_debate_cost_usd": 0.0,
        "last_activity_timestamp": time.time(),
        "context_ratio_user_modified": False
    }

    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

    if "persona_manager" not in st.session_state:
        # MODIFIED: Pass DOMAIN_KEYWORDS to PersonaManager constructor
        st.session_state.persona_manager = PersonaManager(DOMAIN_KEYWORDS)
        st.session_state.all_personas = st.session_state.persona_manager.all_personas
        st.session_state.persona_sets = st.session_state.persona_manager.persona_sets
        st.session_state.selected_persona_set = st.session_state.persona_manager.available_domains[0] if st.session_state.persona_manager.available_domains else "General"
        initial_framework_personas = st.session_state.persona_manager.get_persona_sequence_for_framework(st.session_state.selected_persona_set)
        st.session_state.personas = {name: st.session_state.persona_manager.all_personas.get(name) for name in initial_framework_personas if name in st.session_state.persona_manager.all_personas}
        # REMOVED: The redundant PersonaRouter initialization block, as PersonaManager's __init__ now handles it.

    if "context_analyzer" not in st.session_state:
        analyzer = ContextRelevanceAnalyzer(cache_dir=SENTENCE_TRANSFORMER_CACHE_DIR, codebase_context=st.session_state.codebase_context)
        if st.session_state.persona_manager.persona_router:
            analyzer.set_persona_router(st.session_state.persona_manager.persona_router)
        st.session_state.context_analyzer = analyzer

    if "session_rate_limiter_instance" not in st.session_state:
        st.session_state.session_rate_limiter_instance = RateLimiter(
            calls=10,
            period=60.0,
            key_func=lambda: st.session_state._session_id
        )

    if st.session_state.api_key_input:
        st.session_state.api_key_valid_format = validate_gemini_api_key_format(st.session_state.api_key_input)

    if "user_prompt_input" not in st.session_state or not st.session_state.user_prompt_input:
        default_example_category = list(EXAMPLE_PROMPTS.keys())[0]
        default_example_name = list(EXAMPLE_PROMPTS[default_example_category].keys())[0]
        st.session_state.user_prompt_input = EXAMPLE_PROMPTS[default_example_category][default_example_name]["prompt"]
        st.session_state.selected_example_name = default_example_name
        st.session_state.selected_prompt_category = default_example_category
        st.session_state.active_example_framework_hint = EXAMPLE_PROMPTS[default_example_category][default_example_name].get("framework_hint")


# --- Session State Initialization Call ---
if "initialized" not in st.session_state:
    _initialize_session_state()
# --- END Session State Initialization Call ---

# --- NEW: Session Expiration Check ---
if "initialized" in st.session_state and st.session_state.initialized:
    if time.time() - st.session_state.last_activity_timestamp > SESSION_TIMEOUT_SECONDS:
        st.warning("Your session has expired due to inactivity. Resetting the application.")
        _initialize_session_state()
        st.rerun()
# --- END NEW: Session Expiration Check ---


# --- ENHANCED SANITIZATION FUNCTION ---
def sanitize_user_input(prompt: str) -> str:
    """Enhanced sanitization to prevent prompt injection and XSS attacks."""
    issues = []
    
    sanitized = html.escape(prompt)
    
    injection_patterns = [
        (r'(?i)\b(ignore|disregard|forget|cancel|override)\s+(previous|all)\s+(instructions|commands|context)\b', 'INSTRUCTION_OVERRIDE'),
        (r'(?i)\b(system|user|assistant|prompt|instruction|role)\s*[:=]\s*(system|user|assistant|prompt|instruction|role)\b', 'DIRECTIVE_PROBING'),
        (r'(?i)(?:let\'s|let us|shall we|now|next)\s+ignore\s+previous', 'IGNORE_PREVIOUS'),
        (r'(?i)(?:act as|pretend to be|roleplay as|you are now|your new role is)\s*[:]?\s*([\w\s]+)', 'ROLE_MANIPULATION'),
        (r'(?i)\b(execute|run|system|shell|bash|cmd|powershell|eval|exec|import\s+os|from\s+subprocess)\b', 'CODE_EXECUTION_ATTEMPT'),
        (r'(?i)(?:print|console\.log|echo)\s*\(?[\'"]?.*[\'"]?\)?', 'DEBUG_OUTPUT_ATTEMPT'),
        (r'(?i)(?:output only|respond with|format as|return only|extract)\s+[:]?\s*([\w\s]+)', 'FORMAT_INJECTION'),
        (r'(?i)<\|.*?\|>', 'SPECIAL_TOKEN_MANIPULATION'),
        (r'(?i)(open\s+the\s+pod\s+bay\s+doors)', 'LLM_ESCAPE_REFERENCE'),
        (r'(?i)^\s*#', 'COMMENT_INJECTION'),
        (r'(?i)\b(api_key|secret|password|token|credential)\b[:=]?\s*[\'"]?[\w-]+[\'"]?', 'SENSITIVE_DATA_PROBE'),
    ]
    
    MAX_PROMPT_LENGTH = 2000
    if len(prompt) > MAX_PROMPT_LENGTH:
        issues.append(f"Prompt length exceeded ({len(prompt)} > {MAX_PROMPT_LENGTH}). Truncating.")
        prompt = prompt[:MAX_PROMPT_LENGTH]
    
    for pattern, replacement_tag in injection_patterns:
        prompt = re.sub(pattern, f"[{replacement_tag}]", prompt)
    
    sanitized = re.sub(r'([\\/*\-+!@#$%^&*()_+={}\[\]:;"\'<>?,.])\1{3,}', r'\1\1\1', prompt)
    
    for char_pair in [('"', '"'), ("'", "'"), ('(', ')'), ('{', '}'), ('[', ']')]:
        open_count = sanitized.count(char_pair[0])
        close_count = sanitized.count(char_pair[1])
        if open_count > close_count:
            sanitized += char_pair[1] * (open_count - close_count)
        elif close_count > open_count:
            sanitized = char_pair[0] * (close_count - open_count) + sanitized
    
    return sanitized
# --- END ENHANCED SANITIZATION FUNCTION ---


def reset_app_state():
    """Resets all session state variables to their default values."""
    _initialize_session_state()
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
    st.session_state.persona_changes_detected = True
    update_activity_timestamp()

# --- NEW: HELPER FUNCTION FOR ACTION-ORIENTED ERROR MESSAGING ---
def handle_debate_errors(error: Exception):
    """Displays user-friendly, action-oriented error messages based on exception type."""
    error_type = type(error).__name__
    error_str = str(error).lower()
    
    if "invalid_api_key" in error_str or "api key not valid" in error_str:
        st.error("""
        🔑 **API Key Error: Invalid or Missing Key**
        
        We couldn't authenticate with the Gemini API. Please ensure:
        - Your Gemini API Key is correctly entered in the sidebar.
        - The key is valid and active.
        - You have access to the selected model (`gemini-2.5-flash-lite`, `gemini-2.5-flash`, or `gemini-2.5-pro`).
        
        [Get a Gemini API key from Google AI Studio](https://aistudio.google.com/apikey)
        """)
    elif "429" in error_str or "rate limit" in error_str or "quota" in error_str:
        st.error("""
        ⏳ **API Rate Limit Exceeded**
        
        Google's Gemini API has rate limits based on your project quota.
        
        **Immediate Solutions:**
        - Wait 1-2 minutes before trying again.
        - Reduce the complexity of your prompt.
        - Break your request into smaller parts.
        
        **Long-term Solutions:**
        - Request a quota increase in [Google Cloud Console](https://console.cloud.google.com/iam-admin/quotas).
        - Consider using a less capable but higher-quota model like `gemini-2.5-flash-lite`.
        """)
    elif "connection" in error_str or "timeout" in error_str or "network" in error_str or "socket" in error_str:
        st.error("""
        📡 **Network Connection Issue**
        
        Unable to connect to Google's API servers. This is likely a temporary network issue.
        
        **What to try:**
        - Check your internet connection.
        - Refresh the page.
        - Try again in a few minutes.
        
        Google API status: [Cloud Status Dashboard](https://status.cloud.google.com/)
        """)
    elif "safety" in error_str or "blocked" in error_str or "content" in error_str or "invalid_argument" in error_str:
        st.error("""
        🛡️ **Content Safety Filter Triggered**
        
        Your prompt or the AI's response was blocked by Google's safety filters.
        
        **How to fix:**
        - Rephrase your prompt to avoid potentially sensitive topics.
        - Remove any code that might be interpreted as harmful.
        - Try a less detailed request first.
        """)
    elif isinstance(error, LLMProviderError):
        st.error(f"""
        🌐 **LLM Provider Error: Connection Issue**
        
        An issue occurred while connecting to the Gemini API. This might be a temporary network problem or an API service disruption.
        
        **Details:** `{str(error)}`
        
        Please try again in a moment. If the issue persists, check your internet connection or the [Gemini API status page](https://status.cloud.google.com/).
        """)
    elif isinstance(error, RateLimitExceededError):
        st.error(f"""
        ⏳ **Rate Limit Exceeded**
        
        You've hit the API rate limit for this session. To prevent abuse and manage resources, we limit the number of requests.
        
        **Details:** `{str(error)}`
        
        Please wait a few moments before trying again. If you require higher limits, consider deploying your own instance or upgrading your Google Cloud project's quota.
        """)
    elif isinstance(error, TokenBudgetExceededError):
        st.error(f"""
        📈 **Token Budget Exceeded**
        
        The Socratic debate process consumed more tokens than the allocated budget. This can happen with very complex prompts or extensive codebase contexts.
        
        **Details:** `{str(error)}`
        
        Please consider:
        - Simplifying your prompt.
        - Reducing the amount of codebase context provided.
        - Increasing the 'Max Total Tokens Budget' in the sidebar (use with caution, as this increases cost).
        """)
    elif isinstance(error, SchemaValidationError):
        st.error(f"""
        🚫 **Output Format Error: LLM Response Invalid**
        
        The AI generated an output that did not conform to the expected structured format (JSON schema). This indicates the LLM struggled to follow instructions precisely.
        
        **Details:** `{str(error)}`
        
        The system's circuit breaker has registered this failure. You can try:
        - Rephrasing your prompt to be clearer.
        - Reducing the complexity of the task.
        - Trying a different LLM model (e.g., `gemini-2.5-pro` for more complex tasks).
        """)
    elif isinstance(error, CircuitBreakerError):
        st.error(f"""
        ⛔ **Circuit Breaker Open: Service Temporarily Unavailable**
        
        The system has detected repeated failures from the LLM provider and has temporarily stopped making calls to prevent further issues.
        
        **Details:** `{str(error)}`
        
        The circuit will attempt to reset itself after a short timeout. Please wait a minute and try again.
        """)
    elif isinstance(error, ChimeraError):
        st.error(f"""
        🔥 **Project Chimera Internal Error**
        
        An internal error occurred within the Project Chimera system. This is an unexpected issue.
        
        **Details:** `{str(error)}`
        
        Please report this issue if it persists.
        """)
    else:
        st.error(f"""
        ❌ **An Unexpected Error Occurred**
        
        An unhandled error prevented the Socratic debate from completing.
        
        **Details:** `{str(error)}`
        
        Please try again. If the issue persists, please report it with the prompt you used.
        """)
    logger.exception(f"Debate process failed with error: {error_type}", exc_info=True)
# --- END NEW HELPER FUNCTION ---


# --- MODIFICATIONS FOR SIDEBAR GROUPING ---
with st.sidebar:
    st.header("Configuration")
    
    with st.expander("Core LLM Settings", expanded=True):
        api_key_input_val = st.text_input("Enter your Gemini API Key", type="password", 
                                        key="api_key_input", 
                                        value=st.session_state.api_key_input,
                                        on_change=on_api_key_change, # MODIFIED THIS LINE
                                        help="Your API key will not be stored.")
        
        api_key_col1, api_key_col2 = st.columns([3, 1])
        with api_key_col1:
            if st.session_state.api_key_input:
                if st.session_state.api_key_valid_format:
                    st.success("✅ API key format is valid.")
                else:
                    st.error("❌ Invalid API key format. Gemini keys are typically 35+ characters long with letters, numbers, hyphens and underscores.")
            else:
                st.info("Please enter your Gemini API Key.")
        
        with api_key_col2:
            if st.button("Test Key", help="Verify your API key works by making a small API call.", key="test_api_key_button", on_click=update_activity_timestamp):
                if st.session_state.api_key_input:
                    with st.spinner("Testing API connection..."):
                        if test_gemini_api_key_functional(st.session_state.api_key_input):
                            st.session_state.api_key_functional = True
                            st.success("API key is functional!", icon="✅")
                        else:
                            st.session_state.api_key_functional = False
                            st.error("API key functional test failed. Check key or network.", icon="❌")
                else:
                    st.warning("Please enter an API key first.")
        
        if st.session_state.api_key_input and st.session_state.api_key_functional:
            st.success("API key is ready for use.")
        elif st.session_state.api_key_input and not st.session_state.api_key_functional and st.session_state.api_key_valid_format:
            st.warning("API key format is valid, but functional test failed. Please re-test or check network.")

        st.markdown("Get a Gemini API key from [Google AI Studio](https://aistudio.google.com/apikey).")
        st.markdown("---")
        
        model_options = ["gemini-2.5-flash-lite", "gemini-2.5-flash", "gemini-2.5-pro"]
        current_model_index = model_options.index(st.session_state.selected_model_selectbox) if st.session_state.selected_model_selectbox in model_options else 0
        st.selectbox("Select LLM Model", model_options, 
                     key="selected_model_selectbox", 
                     index=current_model_index,
                     on_change=update_activity_timestamp)
        st.markdown("💡 **Note:** `gemini-2.5-pro` access may require a paid API key. If you encounter issues, try `gemini-2.5-flash-lite` or `gemini-2.5-flash`.")

    with st.expander("Resource Management", expanded=False):
        st.markdown("---")
        st.number_input(
            "Max Total Tokens Budget:", 
            min_value=1000, 
            max_value=2000000,
            step=1000, 
            key="max_tokens_budget_input",
            value=st.session_state.max_tokens_budget_input,
            on_change=update_activity_timestamp
        )
        st.checkbox("Show Intermediate Reasoning Steps", key="show_intermediate_steps_checkbox", value=st.session_state.show_intermediate_steps_checkbox, on_change=update_activity_timestamp)
        st.markdown("---")
        current_ratio_value = st.session_state.get("context_token_budget_ratio", CONTEXT_TOKEN_BUDGET_RATIO_FROM_CONFIG)
        user_prompt_text = st.session_state.get("user_prompt_input", "")
        
        if "context_ratio_user_modified" not in st.session_state:
            st.session_state.context_ratio_user_modified = False
        
        def on_context_ratio_change():
            st.session_state.context_ratio_user_modified = True
            update_activity_timestamp()

        smart_default_ratio = CONTEXT_TOKEN_BUDGET_RATIO_FROM_CONFIG
        help_text_dynamic = "Percentage of total token budget allocated to context analysis."
        
        if user_prompt_text and not st.session_state.context_ratio_user_modified:
            # MODIFIED: Use PromptAnalyzer for domain recommendation
            recommended_domain = st.session_state.persona_manager.prompt_analyzer.recommend_domain_from_keywords(user_prompt_text)
            
            if is_self_analysis_prompt(user_prompt_text):
                smart_default_ratio = 0.35
                help_text_dynamic = "Self-analysis prompts often benefit from more context tokens (35%+)."
            elif recommended_domain == 'Software Engineering':
                smart_default_ratio = 0.30
                help_text_dynamic = "Software Engineering prompts often benefit from more context tokens (30%+)."
            elif recommended_domain == 'Creative':
                smart_default_ratio = 0.15
                help_text_dynamic = "Creative prompts may require less context tokens (15%+)."
            else:
                smart_default_ratio = 0.20
                help_text_dynamic = "Percentage of total token budget allocated to context analysis."
            
            if current_ratio_value == CONTEXT_TOKEN_BUDGET_RATIO_FROM_CONFIG or \
               (smart_default_ratio != current_ratio_value and not st.session_state.context_ratio_user_modified):
                st.session_state.context_token_budget_ratio = smart_default_ratio
                current_ratio_value = smart_default_ratio

        st.slider(
            "Context Token Budget Ratio", min_value=0.05, max_value=0.5, 
            value=current_ratio_value,
            step=0.05, key="context_token_budget_ratio", help=help_text_dynamic,
            on_change=on_context_ratio_change
        )

    is_allowed_check, current_count, time_to_wait, usage_percent = \
        st.session_state.session_rate_limiter_instance.check_and_record_call(st.session_state._session_id, dry_run=True)
    
    st.markdown("---")
    st.subheader("API Rate Limit Status")
    
    progress_text = f"API Usage: {current_count}/{st.session_state.session_rate_limiter_instance.calls} requests"
    st.progress(int(usage_percent), text=progress_text)
    
    if not is_allowed_check:
        st.warning(f"⏳ Rate limit exceeded. Please wait {time_to_wait:.1f} seconds.")
    elif usage_percent >= st.session_state.session_rate_limiter_instance.warning_threshold * 100:
        st.info(f"⚠️ Approaching rate limit. {current_count}/{st.session_state.session_rate_limiter_instance.calls} requests used.")
    else:
        st.success("API usage is within limits.")

    if st.session_state.debate_ran or st.session_state.current_debate_tokens_used > 0:
        st.markdown("---")
        st.subheader("Current Debate Usage")
        col_tokens, col_cost = st.columns(2)
        with col_tokens:
            st.metric("Tokens Used", f"{st.session_state.current_debate_tokens_used:,}")
        with col_cost:
            st.metric("Estimated Cost", f"${st.session_state.current_debate_cost_usd:.6f}")
        st.caption("These metrics update in real-time during the debate.")


st.header("Project Setup & Input")

CUSTOM_PROMPT_KEY = "Custom Prompt"

# --- Callback functions for prompt selection ---
def on_custom_prompt_change():
    st.session_state.user_prompt_input = st.session_state.custom_prompt_text_area_widget
    st.session_state.selected_example_name = CUSTOM_PROMPT_KEY
    st.session_state.selected_prompt_category = CUSTOM_PROMPT_KEY
    st.session_state.active_example_framework_hint = None
    st.session_state.codebase_context = {}
    st.session_state.uploaded_files = []
    update_activity_timestamp()
    st.rerun()

def on_example_select_change(selectbox_key, tab_name):
    selected_example_key = st.session_state[selectbox_key]
    
    st.session_state.selected_example_name = selected_example_key
    st.session_state.user_prompt_input = EXAMPLE_PROMPTS[tab_name][selected_example_key]["prompt"]
    st.session_state.selected_prompt_category = tab_name
    
    framework_hint = EXAMPLE_PROMPTS[tab_name][selected_example_key].get("framework_hint")
    if framework_hint:
        st.session_state.active_example_framework_hint = framework_hint
        logger.debug(f"Framework hint '{framework_hint}' stored for example '{selected_example_key}'.")
    else:
        st.session_state.active_example_framework_hint = None
        logger.warning(f"No framework hint found for example '{selected_example_key}'.")
    
    st.session_state.codebase_context = {}
    st.session_state.uploaded_files = []
    
    if "custom_prompt_text_area_widget" in st.session_state:
        st.session_state.custom_prompt_text_area_widget = st.session_state.user_prompt_input

    logger.debug(f"DEBUG - on_example_select_change called. Selected example: {selected_example_key}")
    logger.debug(f"DEBUG - Prompt updated to: {EXAMPLE_PROMPTS[tab_name][selected_example_key]['prompt'][:100]}...")
    logger.debug(f"DEBUG - Framework hint stored: {framework_hint}")
    update_activity_timestamp()
    st.rerun()

# --- MODIFIED PROMPT SELECTION UI ---
st.subheader("What would you like to do?")

tab_names = list(EXAMPLE_PROMPTS.keys()) + [CUSTOM_PROMPT_KEY]
tabs = st.tabs(tab_names)

for i, tab_name in enumerate(tab_names):
    with tabs[i]:
        if tab_name == CUSTOM_PROMPT_KEY:
            st.markdown("Create your own specialized prompt for unique requirements.")
            st.text_area("Enter your custom prompt here:",
                         value=st.session_state.user_prompt_input,
                         height=150,
                         key="custom_prompt_text_area_widget",
                         on_change=on_custom_prompt_change)
            
            with st.expander("💡 Prompt Engineering Tips"):
                st.markdown("""
                - **Be Specific:** Clearly define your goal and desired output.
                - **Provide Context:** Include relevant background information or code snippets.
                - **Define Constraints:** Specify any limitations (e.g., language, length, format).
                - **Example Output:** If possible, provide an.example of the desired output format.
                """)
            
            # MODIFIED: Use PromptAnalyzer for domain recommendation
            suggested_domain_for_custom = st.session_state.persona_manager.prompt_analyzer.recommend_domain_from_keywords(st.session_state.user_prompt_input)
            
            if suggested_domain_for_custom and suggested_domain_for_custom != st.session_state.selected_persona_set:
                st.info(f"💡 Based on your custom prompt, the **'{suggested_domain_for_custom}'** framework might be appropriate.")
                if st.button(f"Apply '{suggested_domain_for_custom}' Framework (Custom Prompt)", type="secondary", use_container_width=True, key=f"apply_suggested_framework_custom_prompt_{tab_name}", on_click=update_activity_timestamp):
                    st.session_state.selected_persona_set = suggested_domain_for_custom
                    update_activity_timestamp()
                    st.rerun()
            
        else:
            st.markdown(f"Explore example prompts for **{tab_name}**:")
            
            category_options = EXAMPLE_PROMPTS[tab_name]
            
            st.text_input(f"Search prompts in {tab_name}", key=f"search_{tab_name}", value="", on_change=update_activity_timestamp)
            
            filtered_prompts_in_category = {
                name: details for name, details in category_options.items()
                if not st.session_state[f"search_{tab_name}"] or \
                   st.session_state[f"search_{tab_name}"].lower() in name.lower() or \
                   st.session_state[f"search_{tab_name}"].lower() in details["prompt"].lower()
            }

            options_keys = list(filtered_prompts_in_category.keys())
            
            if not options_keys:
                st.info("No example prompts match your search in this category.")
                if st.session_state.selected_prompt_category == tab_name:
                    st.session_state.user_prompt_input = ""
                    st.session_state.selected_example_name = ""
                    st.session_state.codebase_context = {}
                    st.session_state.uploaded_files = []
                continue
            
            initial_selectbox_index = 0
            
            current_selected_example_name = st.session_state.selected_example_name
            current_selected_prompt_category = st.session_state.selected_prompt_category

            if current_selected_prompt_category == tab_name and \
               current_selected_example_name in options_keys:
                initial_selectbox_index = options_keys.index(current_selected_example_name)

            selectbox_key = f"select_example_{tab_name.replace(' ', '_').replace('&', '').replace('(', '').replace(')', '')}"

            selected_example_key_for_this_tab = st.selectbox(
                "Select task:",
                options=options_keys,
                index=initial_selectbox_index,
                format_func=lambda x: f"{x} - {filtered_prompts_in_category[x]['description'][:60]}...",
                label_visibility="collapsed",
                key=selectbox_key,
                on_change=on_example_select_change,
                args=(selectbox_key, tab_name)
            )
            
            selected_prompt_details = filtered_prompts_in_category[selected_example_key_for_this_tab]
            st.info(f"**Description:** {selected_prompt_details['description']}")
            with st.expander("View Full Prompt Text"):
                st.code(selected_prompt_details['prompt'], language='text')

            display_suggested_framework = selected_prompt_details.get("framework_hint")
            if display_suggested_framework and display_suggested_framework != st.session_state.selected_persona_set:
                st.info(f"💡 Based on this example, the **'{display_suggested_framework}'** framework might be appropriate.")
                if st.button(f"Apply '{display_suggested_framework}' Framework",
                            type="primary",
                            use_container_width=True,
                            key=f"apply_suggested_framework_example_{selected_example_key_for_this_tab}",
                            on_click=update_activity_timestamp):
                    st.session_state.selected_persona_set = display_suggested_framework
                    update_activity_timestamp()
                    st.rerun()

user_prompt = st.session_state.user_prompt_input

st.info(f"**Currently Active Prompt:**\n\n{user_prompt}")

logger.debug(f"DEBUG - Current user_prompt_input (from session state): {st.session_state.user_prompt_input[:100]}...")
logger.debug(f"DEBUG - Selected example: {st.session_state.selected_example_name}")
logger.debug(f"DEBUG - Selected prompt category: {st.session_state.selected_prompt_category}")
logger.debug(f"DEBUG - Active example framework hint: {st.session_state.active_example_framework_hint}")
logger.debug(f"DEBUG - Sidebar selected persona set: {st.session_state.selected_persona_set}")

# --- START: UI Layout for Framework and Context ---
col1, col2 = st.columns(2, gap="medium")
with col1:
    st.subheader("Reasoning Framework")
    
    if st.session_state.selected_example_name == CUSTOM_PROMPT_KEY:
        if user_prompt.strip():
            # MODIFIED: Use PromptAnalyzer for domain recommendation
            suggested_domain = st.session_state.persona_manager.prompt_analyzer.recommend_domain_from_keywords(user_prompt)
            if suggested_domain and suggested_domain != st.session_state.selected_persona_set:
                st.info(f"💡 Based on your custom prompt, the **'{suggested_domain}'** framework might be appropriate.")
                if st.button(f"Apply '{suggested_domain}' Framework (Custom Prompt)", type="secondary", use_container_width=True, key=f"apply_suggested_framework_main_{suggested_domain.replace(' ', '_').lower()}", on_click=update_activity_timestamp):
                    st.session_state.selected_persona_set = suggested_domain
                    update_activity_timestamp()
                    st.rerun()
    
    available_framework_options = st.session_state.persona_manager.available_domains
    unique_framework_options = sorted(list(set(available_framework_options)))
    
    current_framework_selection = st.session_state.selected_persona_set
    if current_framework_selection not in unique_framework_options:
        current_framework_selection = unique_framework_options[0] if unique_framework_options else "General"
        st.session_state.selected_persona_set = current_framework_selection
        
    st.selectbox(
        "Select Framework",
        options=unique_framework_options,
        index=unique_framework_options.index(current_framework_selection) if current_framework_selection in unique_framework_options else 0,
        key="selected_persona_set",
        help="Choose a domain-specific reasoning framework or a custom saved framework.",
        on_change=update_activity_timestamp
    )

    if st.session_state.selected_persona_set:
        current_domain_persona_names = st.session_state.persona_manager.get_persona_sequence_for_framework(st.session_state.selected_persona_set)
        st.session_state.personas = {name: st.session_state.persona_manager.all_personas.get(name) for name in current_domain_persona_names if name in st.session_state.persona_manager.all_personas}

    with st.expander("⚙️ Custom Framework Management", expanded=False):
        tabs_framework = st.tabs(["Save Current Framework", "Load/Manage Frameworks", "Export/Import"])

        with tabs_framework[0]:
            st.info("This will save the *currently selected framework* along with any *unsaved persona edits* made in the 'View and Edit Personas' section.")
            st.text_input("Enter a name for your framework:", key='save_framework_input', value=st.session_state.get('save_framework_input', ''), on_change=update_activity_timestamp)
            st.text_area("Framework Description (Optional):", key='framework_description', value=st.session_state.get('framework_description', ''), height=50, on_change=update_activity_timestamp)

            if st.session_state.persona_changes_detected:
                st.warning("Unsaved persona changes detected. Save as a custom framework to persist them.")

            if st.button("Save Current Framework", on_click=update_activity_timestamp) and st.session_state.save_framework_input:
                update_activity_timestamp()
                current_framework_name = st.session_state.selected_persona_set
                current_active_personas_data = {
                    p_name: p_data.model_copy()
                    for p_name, p_data in st.session_state.persona_manager.all_personas.items()
                    if p_name in st.session_state.persona_manager.get_persona_sequence_for_framework(current_framework_name)
                }
                
                success, message = st.session_state.persona_manager.save_framework(
                    st.session_state.save_framework_input,
                    current_framework_name,
                    current_active_personas_data,
                    st.session_state.framework_description
                )
                if success:
                    st.toast(message)
                    st.rerun()
                else:
                    st.error(message)
        
        with tabs_framework[1]:
            all_available_frameworks_for_load = [""] + st.session_state.persona_manager.available_domains
            unique_framework_options_for_load = sorted(list(set(all_available_frameworks_for_load)))
            
            current_selection_for_load = ""
            if st.session_state.selected_persona_set in unique_framework_options_for_load:
                current_selection_for_load = st.session_state.selected_persona_set
            elif st.session_state.selected_persona_set in st.session_state.persona_manager.all_custom_frameworks_data:
                current_selection_for_load = st.session_state.selected_persona_set
            
            st.selectbox(
                "Select a framework to load:",
                options=unique_framework_options_for_load,
                index=unique_framework_options_for_load.index(current_selection_for_load) if current_selection_for_load in unique_framework_options_for_load else 0,
                key='load_framework_select',
                on_change=update_activity_timestamp
            )
            if st.button("Load Selected Framework", on_click=update_activity_timestamp) and st.session_state.load_framework_select:
                update_activity_timestamp()
                success, message, loaded_personas_dict, loaded_persona_sets_dict, new_selected_framework_name = \
                    st.session_state.persona_manager.load_framework_into_session(st.session_state.load_framework_select)
                
                if success:
                    st.session_state.all_personas.update(loaded_personas_dict)
                    st.session_state.persona_sets.update(loaded_persona_sets_dict)
                    st.session_state.selected_persona_set = new_selected_framework_name
                    st.session_state.persona_changes_detected = False
                    st.rerun()
                else:
                    st.error(message)

        with tabs_framework[2]:
            st.subheader("Export Framework")
            st.info("Export the currently selected framework to a file for sharing.")
            export_framework_name = st.session_state.selected_persona_set
            # FIX APPLIED HERE: Changed 'on_change' to 'on_click'
            if st.button(f"Export '{export_framework_name}'", use_container_width=True, key="export_framework_button", on_click=update_activity_timestamp):
                success, message, exported_content = st.session_state.persona_manager.export_framework_for_sharing(export_framework_name)
                if success and exported_content:
                    st.download_button(
                        label=f"Download '{export_framework_name}.yaml'",
                        data=exported_content,
                        file_name=f"{export_framework_name}.yaml",
                        mime="application/x-yaml",
                        use_container_width=True,
                        key="download_exported_framework",
                        on_click=update_activity_timestamp
                    )
                    st.success(message)
                else:
                    st.error(message)

            st.markdown("---")
            st.subheader("Import Framework")
            st.info("Upload a custom framework file (YAML or JSON) to import it.")
            uploaded_framework_file = st.file_uploader("Upload Framework File", type=["yaml", "json"], key="import_framework_uploader", on_change=update_activity_timestamp)
            if uploaded_framework_file is not None:
                if st.button("Import Uploaded Framework", use_container_width=True, key="perform_import_framework_button", on_click=update_activity_timestamp):
                    file_content = uploaded_framework_file.getvalue().decode("utf-8")
                    success, message = st.session_state.persona_manager.import_framework(file_content, uploaded_framework_file.name)
                    if success:
                        st.toast(message)
                        st.rerun()
                    else:
                        st.error(message)

with col2:
    st.subheader("Codebase Context (Optional)")
    if st.session_state.selected_persona_set == "Software Engineering":
        st.file_uploader(
            "Upload up to 100 relevant files",
            accept_multiple_files=True,
            type=['py', 'js', 'ts', 'html', 'css', 'json', 'yaml', 'md', 'txt', 'java', 'go', 'rb', 'php'],
            help="Provide files for context. The AI will analyze them to generate consistent code.",
            key="code_context_uploader",
            on_change=update_activity_timestamp
        )
        
        uploaded_files = st.session_state.code_context_uploader
        if uploaded_files:
            current_uploaded_file_names = {f.name for f in uploaded_files}
            previous_uploaded_file_names = {f.name for f in st.session_state.uploaded_files}

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
        
        elif not st.session_state.uploaded_files and st.session_state.selected_example_name != "Custom Prompt":
            if not st.session_state.codebase_context:
                try:
                    st.session_state.codebase_context = load_demo_codebase_context()
                    st.session_state.uploaded_files = [
                        type('obj', (object,), {'name': k, 'size': len(v.encode('utf-8')), 'getvalue': lambda val=v: val.encode('utf-8')})()
                        for k, v in st.session_state.codebase_context.items()
                    ]
                    st.success(f"{len(st.session_state.codebase_context)} demo file(s) loaded for context.")
                except (FileNotFoundError, ValueError, IOError) as e:
                    st.error(f"❌ Error loading demo codebase context: {e}")
                    st.session_state.codebase_context = {}
                    st.session_state.uploaded_files = []
            else:
                st.success(f"{len(st.session_state.codebase_context)} file(s) already loaded for context.")
        
        elif not st.session_state.uploaded_files and st.session_state.selected_example_name == "Custom Prompt":
            if st.session_state.codebase_context:
                st.session_state.codebase_context = {}
                st.session_state.uploaded_files = []
                st.info("Codebase context cleared for custom prompt.")

    else:
        st.info("Select the 'Software Engineering' framework to provide codebase context.")
        if st.session_state.codebase_context:
            st.session_state.codebase_context = {}
            st.session_state.uploaded_files = []

# --- NEW: Persona Editing UI ---
st.markdown("---")
with st.expander("⚙️ View and Edit Personas", expanded=st.session_state.persona_edit_mode):
    st.session_state.persona_edit_mode = True
    update_activity_timestamp()
    
    st.info("Edit persona parameters for the **currently selected framework**. Changes are temporary unless saved as a custom framework.")
    
    if not st.session_state.persona_changes_detected:
        current_framework_persona_names = st.session_state.persona_manager.get_persona_sequence_for_framework(st.session_state.selected_persona_set)
        for p_name in current_framework_persona_names: 
            persona: PersonaConfig = st.session_state.persona_manager.all_personas.get(p_name)
            original_persona_config = st.session_state.persona_manager._original_personas.get(p_name)
            
            if persona and original_persona_config:
                if persona.system_prompt != original_persona_config.system_prompt or \
                   persona.temperature != original_persona_config.temperature or \
                   persona.max_tokens != original_persona_config.max_tokens:
                    st.session_state.persona_changes_detected = True
                    break

    if st.session_state.persona_changes_detected:
        st.warning("Unsaved changes detected in persona configurations. Please save as a custom framework or reset to persist them.")
        if st.button("Reset All Personas for Current Framework", key="reset_all_personas_button", use_container_width=True, on_click=update_activity_timestamp):
            update_activity_timestamp()
            if st.session_state.persona_manager.reset_all_personas_for_current_framework(st.session_state.selected_persona_set):
                st.toast("All personas for the current framework reset to default.")
                st.session_state.persona_changes_detected = False
                st.rerun()
            else:
                st.error("Could not reset all personas for the current framework.")

    sorted_persona_names = sorted(st.session_state.personas.keys())

    for p_name in sorted_persona_names:
        persona: PersonaConfig = st.session_state.persona_manager.all_personas.get(p_name)
        if not persona:
            st.warning(f"Persona '{p_name}' not found in manager. Skipping.")
            continue

        with st.expander(f"**{persona.name.replace('_', ' ')}**", expanded=False):
            st.markdown(f"**Description:** {persona.description}")
            
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
            
            if st.button(f"Reset {p_name.replace('_', ' ')} to Default", key=f"reset_persona_{p_name}", on_click=update_activity_timestamp):
                update_activity_timestamp()
                if st.session_state.persona_manager.reset_persona_to_default(p_name):
                    st.toast(f"Persona '{p_name.replace('_', ' ')}' reset to default.")
                    st.session_state.persona_changes_detected = False
                    st.rerun()
                else:
                    st.error(f"Could not reset persona '{p_name}'.")
# --- END NEW: Persona Editing UI ---

st.markdown("---")
run_col, reset_col = st.columns([0.8, 0.2])
with run_col:
    run_button_clicked = st.button("🚀 Run Socratic Debate", type="primary", use_container_width=True, 
                               disabled=not (st.session_state.api_key_input and st.session_state.api_key_functional))
with reset_col:
    st.button("🔄 Reset All", on_click=reset_app_state, use_container_width=True)

# --- MODIFICATION: Extract debate execution logic into a separate function ---
def _run_socratic_debate_process():
    """Handles the execution of the Socratic debate process."""
    
    debate_instance = None
    
    request_id = str(uuid.uuid4())[:8]
    
    logger.info("Starting Socratic Debate process.", extra={'request_id': request_id, 'user_prompt': user_prompt})
    
    st.session_state.current_debate_tokens_used = 0
    st.session_state.current_debate_cost_usd = 0.0

    if not st.session_state.api_key_input.strip():
        st.error("Please enter your Gemini API Key in the sidebar to proceed.")
        logger.warning("API key missing, debate process aborted.", extra={'request_id': request_id})
        return
    elif not st.session_state.api_key_functional:
        st.error("Your Gemini API Key is not functional. Please test it in the sidebar.")
        logger.warning("API key not functional, debate process aborted.", extra={'request_id': request_id})
        return
    elif not user_prompt.strip():
        st.error("Please enter a prompt.")
        logger.warning("User prompt is empty, debate process aborted.", extra={'request_id': request_id})
        return

    try:
        st.session_state.session_rate_limiter_instance(lambda: None)() 
    except Exception as e:
        handle_debate_errors(e)
        return

    sanitized_prompt = sanitize_user_input(user_prompt)
    if sanitized_prompt != user_prompt:
        st.warning("User prompt was sanitized to mitigate potential injection risks.")
        st.session_state.user_prompt_input = sanitized_prompt
        logger.info("Prompt was sanitized.", extra={'request_id': request_id, 'original_prompt': user_prompt, 'sanitized_prompt': sanitized_prompt})
    else:
        logger.debug("Prompt did not require sanitization.", extra={'request_id': request_id})
        
    current_user_prompt_for_debate = sanitized_prompt

    st.session_state.debate_ran = False
    final_answer = {
        "COMMIT_MESSAGE": "Debate Failed - Unhandled Error",
        "RATIONALE": "An unexpected error occurred before a final answer could be synthesized.",
        "CODE_CHANGES": [],
        "malformed_blocks": [{"type": "UNHANDLED_ERROR_INIT", "message": "Debate failed during initialization or early phase."}]
    }
    intermediate_steps = {
        "Total_Tokens_Used": 0,
        "Total_Estimated_Cost_USD": 0.0,
        "CODE_CHANGES": [],
        "malformed_blocks": [{"type": "UNHANDLED_ERROR_INIT", "message": "Debate failed during initialization or early phase."}]
    }
    final_total_tokens = 0
    final_total_cost = 0.0
    
    with st.status("Socratic Debate in Progress", expanded=True) as status:
        main_progress_message = st.empty()
        main_progress_message.markdown("### Initializing debate...")
        
        overall_progress_bar = st.progress(0)
        
        active_persona_placeholder = st.empty()

        def update_status(message, state, current_total_tokens, current_total_cost, estimated_next_step_tokens=0, estimated_next_step_cost=0.0, progress_pct: float = None, current_persona_name: str = None):
            main_progress_message.markdown(f"### {message}")
            
            if current_persona_name:
                active_persona_placeholder.markdown(f"Currently running: [bold]{current_persona_name}[/bold]...")
            else:
                active_persona_placeholder.empty()

            if progress_pct is not None:
                st.session_state.debate_progress = max(0.0, min(1.0, progress_pct))
                overall_progress_bar.progress(st.session_state.debate_progress)
            else:
                st.session_state.debate_progress = min(st.session_state.debate_progress + 0.01, 0.99)
                overall_progress_bar.progress(st.session_state.debate_progress)

        def update_status_with_realtime_metrics(message, state, current_total_tokens, current_total_cost, estimated_next_step_tokens=0, estimated_next_step_cost=0.0, progress_pct: float = None, current_persona_name: str = None):
            update_status(message, state, current_total_tokens, current_total_cost, estimated_next_step_tokens, estimated_next_step_cost, progress_pct, current_persona_name)
            
            st.session_state.current_debate_tokens_used = current_total_tokens
            st.session_state.current_debate_cost_usd = current_total_cost

        with capture_rich_output_and_get_console() as (rich_output_buffer, rich_console_instance):
            try:
                domain_for_run = st.session_state.selected_persona_set
                
                if st.session_state.selected_example_name != CUSTOM_PROMPT_KEY and \
                   st.session_state.active_example_framework_hint:
                    domain_for_run = st.session_state.active_example_framework_hint
                    logger.debug(f"Using active example framework hint: {domain_for_run}")
                elif st.session_state.selected_example_name == CUSTOM_PROMPT_KEY:
                    # MODIFIED: Use PromptAnalyzer for domain recommendation
                    suggested_domain = st.session_state.persona_manager.prompt_analyzer.recommend_domain_from_keywords(current_user_prompt_for_debate)
                    if suggested_domain:
                        domain_for_run = suggested_domain
                        logger.debug(f"Using recommended domain for custom prompt: {domain_for_run}")
                
                logger.info(f"Final domain selected for debate: {domain_for_run}")

                logger.debug(f"DEBUG - _run_socratic_debate_process started.")
                logger.debug(f"DEBUG - Prompt at start of debate function: {current_user_prompt_for_debate[:100]}...")
                logger.debug(f"DEBUG - Domain selection logic - Initial domain_for_run: {st.session_state.selected_persona_set}")
                logger.debug(f"DEBUG - Domain selection logic - Selected example name: {st.session_state.selected_example_name}")
                logger.debug(f"DEBUG - Domain selection logic - Active example framework hint: {st.session_state.active_example_framework_hint}")
                logger.debug(f"DEBUG - Domain selection logic - Sidebar selected persona set: {st.session_state.selected_persona_set}")
                logger.debug(f"DEBUG - Domain selection logic - Final domain_for_run: {domain_for_run}")

                current_settings = ChimeraSettings(
                    context_token_budget_ratio=st.session_state.context_token_budget_ratio,
                    total_budget=st.session_state.max_tokens_budget_input
                )

                debate_instance = SocraticDebate(
                    initial_prompt=current_user_prompt_for_debate,
                    api_key=st.session_state.api_key_input,
                    model_name=st.session_state.selected_model_selectbox,
                    all_personas=st.session_state.all_personas,
                    persona_sets=st.session_state.persona_sets,
                    domain=domain_for_run,
                    status_callback=update_status_with_realtime_metrics,
                    rich_console=rich_console_instance,
                    codebase_context=st.session_state.get('codebase_context', {}),
                    context_analyzer=st.session_state.context_analyzer,
                    is_self_analysis=is_self_analysis_prompt(current_user_prompt_for_debate),
                    settings=current_settings,
                    persona_manager=st.session_state.persona_manager
                )
                
                logger.info("Executing Socratic Debate via core.SocraticDebate.", extra={'request_id': request_id, 'debate_instance_id': id(debate_instance)})
                
                final_answer, intermediate_steps = debate_instance.run_debate()
                
                logger.info("Socratic Debate execution finished.", extra={'request_id': request_id, 'debate_instance_id': id(debate_instance)})
                
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
            
            except (TokenBudgetExceededError, SchemaValidationError, ChimeraError, CircuitBreakerError, LLMProviderError) as e:
                handle_debate_errors(e)
                status.update(label=f"Socratic Debate Failed: {type(e).__name__}", state="error", expanded=True)
                st.session_state.debate_ran = True
                if debate_instance:
                    st.session_state.intermediate_steps_output = debate_instance.intermediate_steps
                final_total_tokens = st.session_state.intermediate_steps_output.get('Total_Tokens_Used', 0)
                final_total_cost = st.session_state.intermediate_steps_output.get('Total_Estimated_Cost_USD', 0.0)
                if isinstance(e, RateLimitExceededError):
                    raise e
            except Exception as e:
                handle_debate_errors(e)
                status.update(label=f"Socratic Debate Failed: An unexpected error occurred", state="error", expanded=True)
                st.session_state.debate_ran = True
                if debate_instance:
                    st.session_state.intermediate_steps_output = debate_instance.intermediate_steps
                final_total_tokens = st.session_state.intermediate_steps_output.get('Total_Tokens_Used', 0)
                final_total_cost = st.session_state.intermediate_steps_output.get('Total_Estimated_Cost_USD', 0.0)
            
            if "malformed_blocks" not in final_answer:
                final_answer["malformed_blocks"] = []
            if "malformed_blocks" not in intermediate_steps:
                intermediate_steps["malformed_blocks"] = []
                
            intermediate_steps["Total_Tokens_Used"] = final_total_tokens
            intermediate_steps["Total_Estimated_Cost_USD"] = final_total_cost

# --- END OF NEW FUNCTION ---

if run_button_clicked:
    st.session_state.last_activity_timestamp = time.time()

    for attempt in range(MAX_DEBATE_RETRIES):
        try:
            _run_socratic_debate_process()
            break
        except RateLimitExceededError as e:
            if attempt < MAX_DEBATE_RETRIES - 1:
                wait_time = DEBATE_RETRY_DELAY_SECONDS * (attempt + 1)
                st.info(f"Rate limit exceeded. Retrying Socratic Debate in {wait_time:.1f} seconds... (Attempt {attempt + 1}/{MAX_DEBATE_RETRIES})")
                time.sleep(wait_time)
                st.session_state.last_activity_timestamp = time.time()
            else:
                st.error(f"Max retries ({MAX_DEBATE_RETRIES}) for Socratic Debate reached due to rate limiting. Please try again later.")
                handle_debate_errors(e)
                break
        except Exception as e:
            handle_debate_errors(e)
            break


if st.session_state.debate_ran:
    st.markdown("---")
    st.header("Results")

    with st.expander("📥 Download Analysis", expanded=True):
        st.markdown("**Report format:**")
        format_choice = st.radio("Choose report format:",
            ["Complete Report (Markdown)", "Summary (Text)"],
            label_visibility="collapsed",
            key="report_format_radio",
            on_change=update_activity_timestamp
        )
        
        report_content = generate_markdown_report(
            user_prompt=user_prompt,
            final_answer=st.session_state.final_answer_output,
            intermediate_steps=st.session_state.intermediate_steps_output,
            process_log_output=st.session_state.process_log_output_text,
            config_params=st.session_state.last_config_params,
            persona_audit_log=st.session_state.persona_audit_log
        ) if "Complete" in format_choice else "This is a placeholder for the summary report. Implement summary generation logic here."
        
        file_name = f"chimera_report_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}{'_full' if 'Complete' in format_choice else '_summary'}.{'md' if 'Complete' in format_choice else 'txt'}"
        
        st.download_button(
            label="⬇️ Download Selected Format",
            data=report_content,
            file_name=file_name,
            use_container_width=True,
            type="primary",
            on_click=update_activity_timestamp
        )

    actual_debate_domain = st.session_state.last_config_params.get("domain")

    if actual_debate_domain == "Software Engineering":
        st.subheader("Structured Summary")
        parsed_llm_output_dict: Dict[str, Any]
        malformed_blocks_from_parser = []
        
        raw_output_data = st.session_state.final_answer_output

        if isinstance(raw_output_data, dict):
            try:
                parsed_llm_output_dict = LLMOutputParser().parse_and_validate(
                    json.dumps(raw_output_data),
                    LLMOutput
                )
                malformed_blocks_from_parser.extend(parsed_llm_output_dict.get('malformed_blocks', []))
            except Exception as e:
                st.error(f"Failed to parse final LLMOutput for Software Engineering domain: {e}")
                parsed_llm_output_dict = {
                    "COMMIT_MESSAGE": "Parsing Error",
                    "RATIONALE": f"Failed to parse final LLM output into expected structure. Error: {e}",
                    "CODE_CHANGES": [],
                    "malformed_blocks": [{"type": "UI_PARSING_ERROR", "message": str(e), "raw_string_snippet": str(raw_output_data)[:500]}]
                }
        elif isinstance(raw_output_data, list) and raw_output_data:
            try:
                parsed_llm_output_dict = LLMOutputParser().parse_and_validate(
                    json.dumps(raw_output_data[0]),
                    LLMOutput
                )
                malformed_blocks_from_parser.extend(parsed_llm_output_dict.get('malformed_blocks', []))
            except Exception as e:
                st.error(f"Failed to parse list-based LLMOutput for Software Engineering domain: {e}")
                parsed_llm_output_dict = {
                    "COMMIT_MESSAGE": "Parsing Error",
                    "RATIONALE": f"Failed to parse list-based LLM output into expected structure. Error: {e}",
                    "CODE_CHANGES": [],
                    "malformed_blocks": [{"type": "UI_PARSING_ERROR", "message": str(e), "raw_string_snippet": str(raw_output_data)[:500]}]
                }
        else:
            st.error(f"Final answer for Software Engineering is not a structured dictionary or list. Raw output type: {type(raw_output_data).__name__}")
            parsed_llm_output_dict = {
                "COMMIT_MESSAGE": "Error: Output not structured.",
                "RATIONALE": f"Error: Output not structured. Raw output type: {type(raw_output_data).__name__}",
                "CODE_CHANGES": [],
                "malformed_blocks": [{"type": "UI_PARSING_ERROR", "message": f"Final answer was not a dictionary or list. Type: {type(raw_output_data).__name__}", "raw_string_snippet": str(raw_output_data)[:500]}]
            }
        
        validation_results_by_file = validate_code_output_batch(
            parsed_llm_output_dict,
            st.session_state.get('codebase_context', {})
        )

        all_issues = []
        if isinstance(validation_results_by_file, dict):
            for file_issues_list in validation_results_by_file.values():
                if isinstance(file_issues_list, list):
                    all_issues.extend(file_issues_list)
        
        all_malformed_blocks = malformed_blocks_from_parser
        if isinstance(validation_results_by_file, dict) and 'malformed_blocks' in validation_results_by_file:
            all_malformed_blocks.extend(validation_results_by_file['malformed_blocks'])

        summary_col1, summary_col2 = st.columns(2, gap="medium")
        with summary_col1:
            st.markdown("**Commit Message Suggestion**")
            st.code(parsed_llm_output_dict.get("COMMIT_MESSAGE", "N/A"), language='text')
        with summary_col2:
            st.markdown("**Token Usage**")
            total_tokens = st.session_state.intermediate_steps_output.get('Total_Tokens_Used', 0)
            total_cost = st.session_state.intermediate_steps_output.get('Total_Estimated_Cost_USD', 0.0)
            st.metric("Total Tokens Consumed", f"{total_tokens:,}")
            st.metric("Total Estimated Cost (USD)", f"${total_cost:.4f}")
        st.markdown("**Rationale**")
        st.markdown(parsed_llm_output_dict.get("RATIONALE", "N/A"))
        if parsed_llm_output_dict.get("CONFLICT_RESOLUTION"):
            st.markdown("**Conflict Resolution**")
            st.info(parsed_llm_output_dict["CONFLICT_RESOLUTION"])
        if parsed_llm_output_dict.get("UNRESOLVED_CONFLICT"):
            st.markdown("**Unresolved Conflict**")
            st.warning(parsed_llm_output_dict["UNRESOLVED_CONFLICT"])

        with st.expander("✅ Validation & Quality Report", expanded=True):
            if not all_issues and not all_malformed_blocks:
                st.success("✅ No syntax, style, or formatting issues detected.")
            else:
                if all_malformed_blocks:
                     st.error(f"**Malformed Output Detected:** The LLM produced {len(all_malformed_blocks)} block(s) that could not be parsed or validated correctly. Raw output snippets are provided below.")
                
                if all_malformed_blocks:
                    with st.expander("Malformed Output Details"):
                        for block_info in all_malformed_blocks:
                            st.error(f"**Type:** {block_info.get('type', 'Unknown')}\n**Message:** {block_info.get('message', 'N/A')}")
                            raw_snippet = block_info.get('raw_string_snippet', '')
                            if not raw_snippet:
                                st.code("<No content available>", language='text')
                            else:
                                display_content = raw_snippet[:1000]
                                if len(raw_snippet) > 1000:
                                    display_content += '...'
                                st.code(display_content, language='text')
                            st.markdown("---")

                issues_by_file = defaultdict(list)
                for issue in all_issues:
                    issues_by_file[issue.get('file', 'N/A')].append(issue)

                for file_path, file_issues in issues_by_file.items():
                    with st.expander(f"File: `{file_path}` ({len(file_issues)} issues)", expanded=False):
                        issues_by_type = defaultdict(list)
                        for issue in file_issues:
                            issues_by_type[issue.get('type', 'Unknown')].append(issue)
                        
                        for issue_type, type_issues in sorted(issues_by_type.items()):
                            with st.expander(f"**{issue_type}** ({len(type_issues)} issues)", expanded=False):
                                for issue in type_issues:
                                    line_info = f" (Line: {issue.get('line_number', 'N/A')}, Col: {issue.get('column_number', 'N/A')})" if issue.get('line_number') else ""
                                    st.markdown(f"- **{issue.get('code', '')}**: {issue['message']}{line_info}")
        
        st.subheader("Proposed Code Changes")
        if not parsed_llm_output_dict.get("CODE_CHANGES") and not all_malformed_blocks:
            st.info("No code changes were proposed.")
        
        for change in parsed_llm_output_dict.get("CODE_CHANGES", []):
            with st.expander(f"📝 **{change.get('FILE_PATH', 'N/A')}** (`{change.get('ACTION', 'N/A')}`)", expanded=False):
                st.write(f"**Action:** {change.get('ACTION', 'N/A')}")
                st.write(f"**File Path:** {change.get('FILE_PATH', 'N/A')}")
                
                if change.get('ACTION') in ['ADD', 'MODIFY']:
                    if change.get('ACTION') == 'MODIFY':
                        original_content = st.session_state.codebase_context.get(change.get('FILE_PATH', 'N/A'), "")
                        if original_content:
                            diff_lines = difflib.unified_diff(
                                original_content.splitlines(keepends=True),
                                change.get('FULL_CONTENT', '').splitlines(keepends=True),
                                fromfile=f"a/{change.get('FILE_PATH', 'N/A')}",
                                tofile=f"b/{change.get('FILE_PATH', 'N/A')}",
                                lineterm=''
                            )
                            diff_output = "\n".join(diff_lines)
                            st.write("**Changes:**")
                            st.code(diff_output, language='diff')
                        else:
                            st.write("**New Content:**")
                            st.code(change.get('FULL_CONTENT', ''), language='python')
                    else:
                        st.write("**Content:**")
                        display_content = change.get('FULL_CONTENT', '')[:1500] + "..." if len(change.get('FULL_CONTENT', '')) > 1500 else change.get('FULL_CONTENT', '')
                        st.code(display_content, language='python')
                    
                    st.download_button(
                        label=f"Download {'File' if change.get('ACTION') == 'ADD' else 'New File Content'}",
                        data=change.get('FULL_CONTENT', ''),
                        file_name=change.get('FILE_PATH', 'N/A'),
                        use_container_width=True,
                        type="secondary",
                        on_click=update_activity_timestamp
                    )

                elif change.get('ACTION') == 'REMOVE':
                    st.write("**Lines to Remove:**")
                    st.code("\n".join(change.get('LINES', [])), language='text')

    elif actual_debate_domain == "Self-Improvement":
        st.subheader("Final Synthesized Answer")
        final_analysis_output = st.session_state.final_answer_output # This is already a parsed dict from core.py
        malformed_blocks_from_parser = []

        analysis_summary = "Error: Output not structured for Self-Improvement."
        impactful_suggestions = []

        if isinstance(final_analysis_output, dict):
            # Collect malformed_blocks from the top level of the output
            malformed_blocks_from_parser.extend(final_analysis_output.get("malformed_blocks", []))

            # Check for the version and extract data accordingly
            if final_analysis_output.get("version") == "1.0":
                v1_data = final_analysis_output.get("data", {})
                analysis_summary = v1_data.get("ANALYSIS_SUMMARY", "N/A")
                impactful_suggestions = v1_data.get("IMPACTFUL_SUGGESTIONS", [])
                # Collect malformed_blocks from the 'data' level if present
                malformed_blocks_from_parser.extend(v1_data.get("malformed_blocks", []))
            else:
                analysis_summary = "Error: Unexpected SelfImprovementAnalysisOutput version or structure."
                malformed_blocks_from_parser.append({"type": "UNEXPECTED_VERSION_OR_STRUCTURE", "message": analysis_summary, "raw_string_snippet": str(final_analysis_output)[:500]})
        else:
            st.error(f"Final answer for Self-Improvement is not a structured dictionary. Raw output type: {type(final_analysis_output).__name__}")
            analysis_summary = f"Error: Output not structured for Self-Improvement. Raw output type: {type(final_analysis_output).__name__}"
            malformed_blocks_from_parser.append({"type": "UI_PARSING_ERROR", "message": analysis_summary, "raw_string_snippet": str(final_analysis_output)[:500]})

        st.markdown("**Analysis Summary**")
        st.markdown(analysis_summary)

        if "LLM returned a single suggestion item instead of the full analysis." in analysis_summary:
            st.warning("⚠️ The LLM provided only a partial analysis (a single suggestion) due to potential token limits or reasoning constraints. Consider refining your prompt or increasing the token budget for a more comprehensive report.")

        st.markdown("**Impactful Suggestions**")
        if not impactful_suggestions:
            st.info("No specific suggestions were provided in the analysis.")
        else:
            # Apply the fix here: Add enumeration to the outer loop
            for suggestion_idx, suggestion in enumerate(impactful_suggestions):
                with st.expander(f"💡 {suggestion.get('AREA', 'N/A')}: {suggestion.get('PROBLEM', 'N/A')[:80]}...", expanded=False):
                    st.markdown(f"**Area:** {suggestion.get('AREA', 'N/A')}")
                    st.markdown(f"**Problem:** {suggestion.get('PROBLEM', 'N/A')}")
                    st.markdown(f"**Proposed Solution:** {suggestion.get('PROPOSED_SOLUTION', 'N/A')}")
                    st.markdown(f"**Expected Impact:** {suggestion.get('EXPECTED_IMPACT', 'N/A')}")
                    
                    code_changes = suggestion.get('CODE_CHANGES_SUGGESTED', [])
                    if code_changes:
                        st.markdown("**Suggested Code Changes:**")
                        # Apply the fix here: Add enumeration to the inner loop
                        for change_idx, change in enumerate(code_changes):
                            with st.expander(f"📝 {change.get('FILE_PATH', 'N/A')} (`{change.get('ACTION', 'N/A')}`)", expanded=False):
                                st.write(f"**Action:** {change.get('ACTION', 'N/A')}")
                                st.write(f"**File Path:** {change.get('FILE_PATH', 'N/A')}")
                                
                                # --- NEW LOGIC FOR DIFF_CONTENT ---
                                if change.get('ACTION') in ['ADD', 'MODIFY']:
                                    if change.get('DIFF_CONTENT'):
                                        st.write("**Changes (Unified Diff):**")
                                        st.code(change.get('DIFF_CONTENT', ''), language='diff')
                                        st.download_button(
                                            label=f"Download Diff for {change.get('FILE_PATH', 'N/A')}",
                                            data=change.get('DIFF_CONTENT', ''),
                                            file_name=f"{Path(change.get('FILE_PATH', 'N/A')).name}.diff",
                                            use_container_width=True,
                                            type="secondary",
                                            on_click=update_activity_timestamp,
                                            # Apply the fix here: Add unique key
                                            key=f"diff_download_{suggestion_idx}_{change_idx}_{change.get('FILE_PATH', 'N/A').replace('/', '_')}"
                                        )
                                    elif change.get('FULL_CONTENT'):
                                        st.write("**Content:**")
                                        display_content = change.get('FULL_CONTENT', '')[:1500] + "..." if len(change.get('FULL_CONTENT', '')) > 1500 else change.get('FULL_CONTENT', '')
                                        st.code(display_content, language='python')
                                        st.download_button(
                                            label=f"Download {'File' if change.get('ACTION') == 'ADD' else 'New File Content'}",
                                            data=change.get('FULL_CONTENT', ''),
                                            file_name=change.get('FILE_PATH', 'N/A'),
                                            use_container_width=True,
                                            type="secondary",
                                            on_click=update_activity_timestamp,
                                            # Apply the fix here: Add unique key
                                            key=f"full_download_{suggestion_idx}_{change_idx}_{change.get('FILE_PATH', 'N/A').replace('/', '_')}"
                                        )
                                    else:
                                        st.info("No content or diff provided for this change.")
                                elif change.get('ACTION') == 'REMOVE':
                                    st.write("**Lines to Remove:**")
                                    st.code("\n".join(change.get('LINES', [])), language='text')
                                # --- END NEW LOGIC ---
                    else:
                        st.info("No specific code changes suggested for this item.")

        if malformed_blocks_from_parser:
            with st.expander("Malformed Blocks (Self-Improvement Output)"):
                st.json(malformed_blocks_from_parser)

    else:
        st.subheader("Final Synthesized Answer")
        if isinstance(st.session_state.final_answer_output, dict) and "general_output" in st.session_state.final_answer_output:
            st.markdown(st.session_state.final_answer_output["general_output"])
            if st.session_state.final_answer_output.get("malformed_blocks"):
                with st.expander("Malformed Blocks (General Output)"):
                    st.json(st.session_state.final_answer_output["malformed_blocks"])
        elif isinstance(st.session_state.final_answer_output, dict):
            st.json(st.session_state.final_answer_output)
        else:
            st.markdown(st.session_state.final_answer_output)

    with st.expander("Show Intermediate Steps & Process Log"):
        if st.session_state.show_intermediate_steps_checkbox:
            st.subheader("Intermediate Reasoning Steps")
            display_steps = {k: v for k, v in st.session_state.intermediate_steps_output.items()
                             if not k.endswith("_Tokens_Used") and not k.endswith("_Estimated_Cost_USD") and k != "Total_Tokens_Used" and k != "Total_Estimated_Cost_USD" and not k.startswith("malformed_blocks")}
            sorted_step_keys = sorted(display_steps.keys(), key=lambda x: (x.split('_')[0] if '_' in x else '', x))
            for step_key in sorted_step_keys:
                persona_name = step_key.split('_')[0]
                display_name = step_key.replace('_Output', '').replace('_Critique', '').replace('_Feedback', '').replace('_', ' ').title()
                content = display_steps.get(step_key, "N/A")
                cleaned_step_key = step_key.replace("_Output", "").replace("_Critique", "").replace("_Feedback", "")
                token_count_key = f"{cleaned_step_key}_Tokens_Used"
                tokens_used = st.session_state.intermediate_steps_output.get(token_count_key, "N/A")
                
                actual_temp = st.session_state.intermediate_steps_output.get(f"{persona_name}_Actual_Temperature")
                actual_max_tokens = st.session_state.intermediate_steps_output.get(f"{persona_name}_Actual_Max_Tokens")
                
                persona_params_info = ""
                if actual_temp is not None or actual_max_tokens is not None:
                    persona_params_info = " (Parameters: "
                    if actual_temp is not None:
                        persona_params_info += f"Temp={actual_temp:.2f}"
                    if actual_max_tokens is not None:
                        if actual_temp is not None: persona_params_info += ", "
                        persona_params_info += f"MaxTokens={actual_max_tokens}"
                    persona_params_info += ")"

                with st.expander(f"**{display_name}** (Tokens: {tokens_used}){persona_params_info}"):
                    if isinstance(content, dict):
                        st.json(content)
                    else:
                        st.markdown(f"```markdown\n{content}\n```")
        st.subheader("Process Log")
        st.code(strip_ansi_codes(st.session_state.process_log_output_text), language='text')