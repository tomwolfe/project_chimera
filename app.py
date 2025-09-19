# app.py

import streamlit as st
import os
import sys

# NEW: Set TOKENIZERS_PARALLELISM to false to avoid deadlocks on fork
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import io
import contextlib
import re
import datetime
import time
from typing import Dict, Any, List, Optional, Callable
import logging
from rich.console import Console
from core import SocraticDebate

from src.models import (
    PersonaConfig,
    LLMOutput,
    CodeChange,
    SelfImprovementAnalysisOutputV1,
    SuggestionItem,
)
from src.utils.output_parser import LLMOutputParser
from src.persona_manager import PersonaManager
from src.exceptions import (
    ChimeraError,
    LLMResponseValidationError,  # Keep this import
    SchemaValidationError,
    TokenBudgetExceededError,
    LLMProviderError,
    CircuitBreakerError,
)
from collections import defaultdict
from pydantic import ValidationError  # Keep this import
import html
import difflib
from src.utils.command_executor import execute_command_safely
from src.utils.code_validator import validate_code_output_batch
import json
import uuid
from src.logging_config import setup_structured_logging
from src.middleware.rate_limiter import RateLimiter, RateLimitExceededError
from src.config.settings import (
    ChimeraSettings,
)  # MODIFIED: Import ChimeraSettings from src/config/settings.py
from pathlib import Path

from src.utils.prompt_analyzer import PromptAnalyzer
from src.token_tracker import TokenUsageTracker

# NEW IMPORTS FOR CODEBASE SCANNING AND GARBAGE COLLECTION
from src.context.context_analyzer import ContextRelevanceAnalyzer, CodebaseScanner
import gc

from src.utils.report_generator import generate_markdown_report, strip_ansi_codes
from src.utils.session_manager import (
    _initialize_session_state,
    update_activity_timestamp,
    reset_app_state,
    check_session_expiration,
    SESSION_TIMEOUT_SECONDS,
)
from src.utils.ui_helpers import (
    on_api_key_change,
    display_key_status,
    test_api_key,
    shutdown_streamlit,
)

# NEW IMPORT for PromptOptimizer
from src.utils.prompt_optimizer import PromptOptimizer

# NEW IMPORT: For the summarization pipeline
from transformers import pipeline

# --- Constants ---
MAX_DEBATE_RETRIES = 3
DEBATE_RETRY_DELAY_SECONDS = 5

# --- Configuration Loading ---
try:
    # MODIFIED: Load settings from config.yaml, which will also load from .env and environment variables
    settings_instance = ChimeraSettings.from_yaml("config.yaml")
except Exception as e:
    st.error(f"❌ Application configuration error: {e}")
    st.stop()


# NEW: Initialize the global logger object using st.cache_resource for robustness
@st.cache_resource
def get_app_logger():
    """Initializes and returns the structured logger, cached by Streamlit."""
    try:
        configured_logger = setup_structured_logging()
        if configured_logger is None:
            logging.basicConfig(
                level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
            )
            fallback_logger = logging.getLogger(__name__)
            fallback_logger.warning(
                "setup_structured_logging returned None. Using basic fallback logger."
            )
            return fallback_logger
        return configured_logger
    except Exception as e:
        st.error(f"❌ Error setting up structured logging: {e}")
        logging.basicConfig(
            level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
        )
        fallback_logger = logging.getLogger(__name__)
        fallback_logger.exception(
            "Failed to set up structured logging. Using basic fallback logger."
        )
        return fallback_logger


logger = get_app_logger()

if logger is None:
    st.error(
        "❌ Critical: Logging system failed to initialize and fallback also failed. Please check src/logging_config.py."
    )
    logging.basicConfig(
        level=logging.ERROR, format="%(asctime)s - %(levelname)s - %(message)s"
    )
    logger = logging.getLogger(__name__)
    logger.critical(
        "Final fallback logger activated due to primary and secondary logger initialization failure."
    )


DOMAIN_KEYWORDS = settings_instance.domain_keywords
CONTEXT_TOKEN_BUDGET_RATIO_FROM_CONFIG = settings_instance.context_token_budget_ratio
MAX_TOKENS_LIMIT = settings_instance.total_budget


# NEW: Instantiate CodebaseScanner once for the UI and cache it
@st.cache_resource
def get_codebase_scanner_instance():
    """Initializes and returns the CodebaseScanner, cached by Streamlit."""
    logger.info("Initializing CodebaseScanner via st.cache_resource.")
    return CodebaseScanner()


# NEW: Instantiate ContextRelevanceAnalyzer once and cache it
@st.cache_resource
def get_context_relevance_analyzer_instance(_settings: ChimeraSettings):
    """Initializes and returns the ContextRelevanceAnalyzer, cached by Streamlit."""
    logger.info("Initializing ContextRelevanceAnalyzer via st.cache_resource.")
    return ContextRelevanceAnalyzer(
        cache_dir=_settings.sentence_transformer_cache_dir, raw_file_contents={}
    )


# NEW: Instantiate the Hugging Face summarization pipeline once and cache it
@st.cache_resource
def get_summarizer_pipeline_instance():
    """Initializes and returns the Hugging Face summarization pipeline, cached by Streamlit."""
    logger.info(
        "Initializing Hugging Face summarization pipeline (sshleifer/distilbart-cnn-6-6) via st.cache_resource."
    )
    return pipeline("summarization", model="sshleifer/distilbart-cnn-6-6")


EXAMPLE_PROMPTS = {
    "Coding & Implementation": {
        "Implement Python API Endpoint": {
            "prompt": "Implement a new FastAPI endpoint `/items/{item_id}` that retrieves an item from a dictionary. Include basic error handling for non-existent items and add a corresponding unit test.",
            "description": "Generate a complete API endpoint with proper error handling, validation, and documentation.",
            "framework_hint": "Software Engineering",
        },
        "Refactor a Python Function": {
            "prompt": "Refactor the given Python function to improve its readability and performance. It currently uses a nested loop; see if you can optimize it.",
            "description": "Improve structure and readability of existing code while maintaining functionality.",
            "framework_hint": "Software Engineering",
        },
        "Fix a Bug in a Script": {
            "prompt": "The provided Python script is supposed to calculate the average of a list of numbers but fails with a `TypeError` if the list contains non-numeric strings. Fix the bug by safely ignoring non-numeric values.",
            "description": "Identify and correct issues in problematic code with explanations.",
            "framework_hint": "Software Engineering",
        },
    },
    "Analysis & Problem Solving": {
        "Design a Mars City": {
            "prompt": "Design a sustainable city for 1 million people on Mars, considering resource scarcity and human psychology.",
            "description": "Explore complex design challenges with multi-faceted considerations.",
            "framework_hint": "Creative",
        },
        "Ethical AI Framework": {
            "prompt": "Develop an ethical framework for an and AI system designed to assist in judicial sentencing, addressing bias, transparency, and accountability.",
            "description": "Formulate ethical guidelines for sensitive AI applications.",
            "framework_hint": "Business",
        },
        "Critically analyze the entire Project Chimera codebase. Identify the most impactful code changes for self-improvement, focusing on the 80/20 Pareto principle. Prioritize enhancements to reasoning quality, robustness, efficiency, and developer maintainability. For each suggestion, provide a clear rationale and a specific, actionable code modification.": {
            "prompt": "Critically analyze the entire Project Chimera codebase. Identify the most impactful code changes for self-improvement, focusing on the 80/20 Pareto principle. Prioritize enhancements to reasoning quality, robustness, efficiency, and developer maintainability. For each suggestion, provide a clear rationale and a specific, actionable code modification.",
            "description": "Perform a deep self-analysis of the Project Chimera codebase for improvements.",
            "framework_hint": "Self-Improvement",
        },
        "Climate Change Solution": {
            "prompt": "Propose an innovative, scalable solution to mitigate the effects of climate change, focusing on a specific sector (e.g., energy, agriculture, transportation).",
            "description": "Brainstorm and propose solutions for global challenges.",
            "framework_hint": "Science",
        },
    },
}

# --- Session State Initialization Call ---
if "initialized" not in st.session_state:
    _initialize_session_state(
        app_config=settings_instance,
        example_prompts=EXAMPLE_PROMPTS,
        get_context_relevance_analyzer_instance=get_context_relevance_analyzer_instance,
        get_codebase_scanner_instance=get_codebase_scanner_instance,
        get_summarizer_pipeline_instance=get_summarizer_pipeline_instance,
    )
    # MODIFIED: Use settings_instance.GEMINI_API_KEY as default
    st.session_state.api_key_input = (
        st.session_state.api_key_input or settings_instance.GEMINI_API_KEY
    )
# --- END Session State Initialization Call ---

# --- NEW: Session Expiration Check ---
check_session_expiration(settings_instance, EXAMPLE_PROMPTS)
# --- END NEW: Session Expiration Check ---


# --- NEW HELPER FUNCTION FOR ROBUST TOKEN COUNTING (as suggested) ---
def calculate_token_count(text: str, tokenizer) -> int:
    """
    Robustly counts tokens using available tokenizer methods.
    This helper is for UI display or other direct token counting needs in app.py.
    """
    if hasattr(tokenizer, "count_tokens"):
        return tokenizer.count_tokens(text)
    elif hasattr(tokenizer, "encode"):
        return len(tokenizer.encode(text))
    elif hasattr(tokenizer, "tokenize"):
        return len(tokenizer.tokenize(text))
    else:
        logger.warning(
            f"Unknown tokenizer type for {type(tokenizer).__name__}. Falling back to character count / 4 estimate."
        )
        return len(text) // 4


# --- END NEW HELPER FUNCTION ---


def sanitize_user_input(prompt: str) -> str:
    """Enhanced sanitization to prevent prompt injection and XSS attacks."""
    issues = []

    processed_prompt = prompt

    injection_patterns = [
        (
            r"(?i)\b(ignore|disregard|forget|cancel|override)\s+(previous|all)\s+(instructions|commands|context)\b",
            "INSTRUCTION_OVERRIDE",
        ),
        (
            r"(?i)(system|user|assistant|prompt|instruction|role)\s*[:=]\s*(system|user|assistant|prompt|instruction|role)\b",
            "DIRECTIVE_PROBING",
        ),
        (
            r"(?i)(?:let\'s|let us|shall we|now|next)\s+ignore\s+previous",
            "IGNORE_PREVIOUS",
        ),
        (
            r"(?i)(?:act as|pretend to be|roleplay as|you are now|your new role is)\s*[:]?\s*([\w\s]+)",
            "ROLE_MANIPULATION",
        ),
        (
            r"(?i)\b(execute|run|system|shell|bash|cmd|powershell|eval|exec|import\s+os|from\s+subprocess)\b",
            "CODE_EXECUTION_ATTEMPT",
        ),
        (
            r'(?i)(?:print|console\.log|echo)\s*\(?[\'"]?.*[\'"]?\)?',
            "DEBUG_OUTPUT_ATTEMPT",
        ),
        (
            r"(?i)(?:output only|respond with|format as|return only|extract)\s+[:]?\s*([\w\s]+)",
            "FORMAT_INJECTION",
        ),
        (r"(?i)<\|.*?\|>", "SPECIAL_TOKEN_MANIPULATION"),
        (r"(?i)(open\s+the\s+pod\s+bay\s+doors)", "LLM_ESCAPE_REFERENCE"),
        (r"(?i)^\s*#", "COMMENT_INJECTION"),
        (
            r'(?i)\b(api_key|secret|password|token|credential)\b[:=]?\s*[\'"]?[\w-]+[\'"]?',
            "SENSITIVE_DATA_PROBE",
        ),
    ]

    MAX_PROMPT_LENGTH = 2000
    for pattern, replacement_tag in injection_patterns:
        if re.search(pattern, prompt):
            processed_prompt = f"[{replacement_tag}]"
            return processed_prompt

    if len(processed_prompt) > MAX_PROMPT_LENGTH:
        issues.append(
            f"Prompt length exceeded ({len(processed_prompt)} > {MAX_PROMPT_LENGTH}). Truncating."
        )
        processed_prompt = processed_prompt[:MAX_PROMPT_LENGTH] + " [TRUNCATED]"

    sanitized = html.escape(processed_prompt)

    sanitized = re.sub(
        r'([\\/*\-+!@#$%^&*()_+={}\[\]:;"\'<>?,.])\1{3,}', r"\1\1\1", sanitized
    )

    return sanitized


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
        logger.error(f"API Key Error: {error_str}", exc_info=True)
    elif isinstance(error, RateLimitExceededError):
        st.error(f"""
        ⏳ **Rate Limit Exceeded**

        You've hit the API rate limit for this session. To prevent abuse and manage resources, we limit the number of requests.

        **Details:** `{str(error)}`

        Please wait a few moments before trying again. If you require higher limits, consider deploying your own instance or upgrading your Google Cloud project's quota.
        """)
        logger.error(f"Rate Limit Exceeded: {error_str}", exc_info=True)
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
        logger.error(f"Token Budget Exceeded: {error_str}", exc_info=True)
    elif isinstance(error, LLMProviderError):
        st.error(f"""
        🌐 **LLM Processing Error**

        An issue occurred during AI model interaction or processing. This could be a temporary service disruption, an unexpected model response, or an internal processing error.

        **Details:** `{str(error)}`

        Please try again in a moment. If the issue persists, consider simplifying your prompt or checking the [Gemini API status page](https://status.cloud.google.com/).
        """)
        logger.error(f"LLM Provider Error: {error_str}", exc_info=True)
    elif isinstance(error, CircuitBreakerError):
        st.error(f"""
        ⛔ **Circuit Breaker Open: Service Temporarily Unavailable**

        The system has detected repeated failures from the LLM provider and has temporarily stopped making calls to prevent further issues.

        **Details:** `{str(error)}`

        The circuit will attempt to reset itself after a short timeout. Please wait a minute and try again.
        """)
        logger.error(f"Circuit Breaker Open: {error_str}", exc_info=True)
    elif isinstance(error, SchemaValidationError):
        st.error(f"""
        🚫 **Output Format Error: LLM Response Invalid**

        The AI generated an output that did not conform to the expected structured format (JSON schema). This indicates the LLM struggled to follow instructions precisely.

        **Details:** `{str(error)}`

        The system's circuit breaker has registered this failure. You can try:
        - Rephrase your prompt to be clearer.
        - Reduce the complexity of the task.
        - Try a different LLM model (e.g., `gemini-2.5-pro` for more complex tasks).
        """)
        logger.error(f"Schema Validation Error: {error_str}", exc_info=True)
    elif isinstance(error, TypeError) and "unexpected keyword argument" in error_str:
        st.error("""
        🐛 **Internal Configuration Error: Type Mismatch**

        An internal component received an unexpected argument. This usually indicates a mismatch in component configuration or an outdated interface.

        **Details:** `{str(error)}`

        This is likely a bug within Project Chimera. Please report this issue.
        """)
        logger.error(
            f"Internal Configuration Error (TypeError): {error_str}", exc_info=True
        )
    elif (
        "connection" in error_str
        or "timeout" in error_str
        or "network" in error_str
        or "socket" in error_str
    ):
        st.error("""
        📡 **Network Connection Issue**

        Unable to connect to Google's API servers. This is likely a temporary network issue.

        **What to try:**
        - Check your internet connection.
        - Refresh the page.
        - Try again in a few minutes.

        Google API status: [Cloud Status Dashboard](https://status.cloud.google.com/)
        """)
        logger.error(f"Network Connection Issue: {error_str}", exc_info=True)
    elif (
        ("safety" in error_str and "chimera_error" not in error_str)
        or "blocked" in error_str
        or "content" in error_str
        or "invalid_argument" in error_str
    ):
        st.error("""
        🛡️ **Content Safety Filter Triggered**

        Your prompt or the AI's response was blocked by Google's safety filters.

        **How to fix:**
        - Rephrase your prompt to avoid potentially sensitive topics.
        - Remove any code that might be interpreted as harmful.
        - Try a less detailed request first.
        """)
        logger.error(f"Content Safety Filter Triggered: {error_str}", exc_info=True)
    elif isinstance(error, ChimeraError):
        st.error(f"""
        🔥 **Project Chimera Internal Error**

        An internal error occurred within the Project Chimera system. This is an unexpected issue.

        **Details:** `{str(error)}`

        Please report this issue if it persists.
        """)
        logger.error(f"Project Chimera Internal Error: {error_str}", exc_info=True)
    else:
        st.error(f"""
        ❌ **An Unexpected Error Occurred**

        An unhandled error prevented the Socratic debate from completing.

        **Details:** `{str(error)}`

        Please try again. If the issue persists, please report it with the prompt you used.
        """)
        logger.exception(
            f"Debate process failed with error: {error_type}", exc_info=True
        )


def execute_command(command_str: str, timeout: int = 60) -> str:
    """
    Executes a simple command safely using the centralized utility.
    This function is specifically for simple 'echo' commands within the app's UI.
    """
    try:
        return_code, stdout, stderr = execute_command_safely(
            ["echo", command_str], timeout=timeout
        )
        if return_code == 0:
            return stdout.strip()
        else:
            error_output = (
                stderr.strip()
                if stderr.strip()
                else f"Command failed with exit code {return_code}."
            )
            logger.error(f"Error executing command '{command_str}': {error_output}")
            return f"Error executing command: {error_output}"
    except Exception as e:
        logger.error(f"Error executing command '{command_str}': {e}", exc_info=True)
        return f"Error executing command: {e}"


def _log_persona_change(
    persona_name: str, parameter: str, old_value: Any, new_value: Any
):
    """Logs a change to a persona parameter in the session audit log."""
    st.session_state.persona_audit_log.append(
        {
            "timestamp": datetime.datetime.now().isoformat(),
            "persona": persona_name,
            "parameter": parameter,
            "old_value": old_value,
            "new_value": new_value,
        }
    )
    st.session_state.persona_changes_detected = True
    update_activity_timestamp()


with st.sidebar:
    st.header("Configuration")

    with st.expander("Core LLM Settings", expanded=True):
        api_key_input_val = st.text_input(
            "Enter your Gemini API Key",
            type="password",
            key="api_key_input",
            # MODIFIED: Use settings_instance.GEMINI_API_KEY as default
            value=st.session_state.api_key_input or settings_instance.GEMINI_API_KEY,
            on_change=on_api_key_change,
            help="Your API key will not be stored.",
        )

        api_key_col1, api_key_col2, api_key_col3 = st.columns([2, 1, 1])
        with api_key_col1:
            if st.session_state.api_key_input:
                if st.session_state.api_key_valid_format:
                    st.success("✅ API key format is valid.")
                else:  # MODIFIED: Check for empty API key from settings
                    st.error(f"❌ {st.session_state.api_key_format_message}")
            else:
                if (
                    os.getenv("ENVIRONMENT") == "production"
                    and st.session_state.api_key_input
                ):
                    st.warning(
                        "⚠️ API key is sourced from environment variable. Consider using a secrets manager for production."
                    )
                elif (  # MODIFIED: Check for empty API key from settings
                    os.getenv("ENVIRONMENT") == "production"
                    and not st.session_state.api_key_input
                ):
                    st.error(
                        "❌ No API key found. In production, API key should be from secrets manager or environment variable."
                    )

                st.info("Please enter your Gemini API Key.")

        with api_key_col2:  # MODIFIED: Test key functionality
            st.button("Test Key", on_click=test_api_key, key="test_api_key_btn")

        with api_key_col3:
            display_key_status()

        st.markdown(
            "Get a Gemini API key from [Google AI Studio](https://aistudio.google.com/apikey)."
        )
        st.markdown("---")

        model_options = ["gemini-2.5-flash-lite", "gemini-2.5-flash", "gemini-2.5-pro"]
        current_model_index = (
            model_options.index(st.session_state.selected_model_selectbox)
            # MODIFIED: Use settings_instance.model_name as default
            if st.session_state.selected_model_selectbox in model_options
            else model_options.index(settings_instance.model_name)
            if settings_instance.model_name in model_options
            else 0
        )
        st.selectbox(
            "Select LLM Model",
            model_options,
            key="selected_model_selectbox",
            index=current_model_index,
            on_change=update_activity_timestamp,
        )
        st.markdown(
            "💡 **Note:** `gemini-2.5-pro` access may require a paid API key. If you encounter issues, try `gemini-2.5-flash-lite` or `gemini-2.5-flash`."
        )

    with st.expander("Resource Management", expanded=False):
        st.markdown("---")

        def on_max_tokens_budget_change():
            st.session_state.token_tracker.budget = (
                st.session_state.max_tokens_budget_input
            )
            update_activity_timestamp()

        st.number_input(
            "Max Total Tokens Budget:",
            min_value=1000,
            max_value=MAX_TOKENS_LIMIT,
            step=1000,
            key="max_tokens_budget_input",
            # MODIFIED: Use settings as default
            value=st.session_state.max_tokens_budget_input
            or settings_instance.total_budget,
            on_change=on_max_tokens_budget_change,
        )
        st.checkbox(
            "Show Intermediate Reasoning Steps",
            key="show_intermediate_steps_checkbox",
            value=st.session_state.show_intermediate_steps_checkbox,
            on_change=update_activity_timestamp,
        )
        st.markdown("---")
        current_ratio_value = st.session_state.get(
            "context_token_budget_ratio", CONTEXT_TOKEN_BUDGET_RATIO_FROM_CONFIG
        )
        user_prompt_text = st.session_state.get("user_prompt_input", "")

        if "context_ratio_user_modified" not in st.session_state:
            st.session_state.context_ratio_user_modified = False

        def on_context_ratio_change():
            st.session_state.context_ratio_user_modified = True
            update_activity_timestamp()

        smart_default_ratio = CONTEXT_TOKEN_BUDGET_RATIO_FROM_CONFIG
        help_text_dynamic = (
            "Percentage of total token budget allocated to context analysis."
        )

        if user_prompt_text and not st.session_state.context_ratio_user_modified:
            recommended_domain = st.session_state.persona_manager.prompt_analyzer.recommend_domain_from_keywords(
                user_prompt_text
            )

            if st.session_state.persona_manager.prompt_analyzer.is_self_analysis_prompt(
                user_prompt_text
            ):
                smart_default_ratio = settings_instance.self_analysis_context_ratio
                help_text_dynamic = "Self-analysis prompts often benefit from more context tokens (35%+)."
            elif recommended_domain == "Software Engineering":
                smart_default_ratio = 0.30
                help_text_dynamic = "Software Engineering prompts often benefit from more context tokens (30%+)."
            elif recommended_domain == "Creative":
                smart_default_ratio = 0.15
                help_text_dynamic = (
                    "Creative prompts may require less context tokens (15%+)."
                )
            else:
                smart_default_ratio = 0.20
                help_text_dynamic = (
                    "Percentage of total token budget allocated to context analysis."
                )

            if current_ratio_value == CONTEXT_TOKEN_BUDGET_RATIO_FROM_CONFIG or (
                smart_default_ratio != current_ratio_value
                and not st.session_state.context_ratio_user_modified
            ):
                st.session_state.context_token_budget_ratio = smart_default_ratio
                current_ratio_value = smart_default_ratio

        st.slider(
            "Context Token Budget Ratio",
            min_value=0.05,
            max_value=0.5,
            value=current_ratio_value,
            step=0.05,
            key="context_token_budget_ratio",
            help=help_text_dynamic,
            on_change=on_context_ratio_change,
        )

    is_allowed_check, current_count, time_to_wait, usage_percent = (
        st.session_state.session_rate_limiter_instance.check_and_record_call(
            st.session_state._session_id, dry_run=True
        )
    )

    st.markdown("---")
    st.subheader("API Rate Limit Status")

    progress_text = f"API Usage: {current_count}/{st.session_state.session_rate_limiter_instance.calls} requests"
    st.progress(int(usage_percent), text=progress_text)

    if not is_allowed_check:
        st.warning(f"⏳ Rate limit exceeded. Please wait {time_to_wait:.1f} seconds.")
    elif (
        usage_percent
        >= st.session_state.session_rate_limiter_instance.warning_threshold * 100
    ):
        st.info(
            f"⚠️ Approaching rate limit. {current_count}/{st.session_state.session_rate_limiter_instance.calls} requests used."
        )
    else:
        st.success("API usage is within limits.")

    if st.session_state.debate_ran or st.session_state.current_debate_tokens_used > 0:
        st.markdown("---")
        st.subheader("Current Debate Usage")
        col_tokens, col_cost = st.columns(2)
        with col_tokens:
            st.metric("Tokens Used", f"{st.session_state.current_debate_tokens_used:,}")
        with col_cost:
            st.metric(
                "Estimated Cost", f"${st.session_state.current_debate_cost_usd:.6f}"
            )
        st.caption("These metrics update in real-time during the debate.")

    st.markdown("---")
    # NEW: Add a "Shutdown App" button
    if st.button("🛑 Shutdown App", use_container_width=True, type="secondary"):
        shutdown_streamlit()


st.header("Project Setup & Input")

CUSTOM_PROMPT_KEY = "Custom Prompt"


def on_custom_prompt_change():
    st.session_state.user_prompt_input = st.session_state.custom_prompt_text_area_widget
    st.session_state.selected_example_name = CUSTOM_PROMPT_KEY
    st.session_state.selected_prompt_category = CUSTOM_PROMPT_KEY
    st.session_state.active_example_framework_hint = None
    st.session_state.codebase_context = {}
    st.session_state.structured_codebase_context = {}
    st.session_state.raw_file_contents = {}
    st.session_state.uploaded_files = []
    update_activity_timestamp()
    st.rerun()


def on_example_select_change(selectbox_key, tab_name):
    selected_example_key = st.session_state[selectbox_key]

    st.session_state.selected_example_name = selected_example_key
    st.session_state.user_prompt_input = EXAMPLE_PROMPTS[tab_name][
        selected_example_key
    ]["prompt"]
    st.session_state.selected_prompt_category = tab_name

    framework_hint = EXAMPLE_PROMPTS[tab_name][selected_example_key].get(
        "framework_hint"
    )
    if framework_hint:
        st.session_state.active_example_framework_hint = framework_hint
        logger.debug(
            f"Framework hint '{framework_hint}' stored for example '{selected_example_key}'."
        )
    else:
        st.session_state.active_example_framework_hint = None
        logger.warning(f"No framework hint found for example '{selected_example_key}'.")

    st.session_state.codebase_context = {}
    st.session_state.structured_codebase_context = {}
    st.session_state.raw_file_contents = {}
    st.session_state.uploaded_files = []

    if (
        selected_example_key
        == "Critically analyze the entire Project Chimera codebase. Identify the most impactful code changes for self-improvement, focusing on the 80/20 Pareto principle. Prioritize enhancements to reasoning quality, robustness, efficiency, and developer maintainability. For each suggestion, provide a clear rationale and a specific, actionable code modification."
    ):
        st.session_state.user_prompt_input = (
            EXAMPLE_PROMPTS[tab_name][selected_example_key]["prompt"]
            + "\n\nNOTE: You have full access to the Project Chimera codebase for this analysis."
        )

    if "custom_prompt_text_area_widget" in st.session_state:
        st.session_state.custom_prompt_text_area_widget = (
            st.session_state.user_prompt_input
        )

    logger.debug(
        f"Current user_prompt_input (from session state): {st.session_state.user_prompt_input[:100]}..."
    )
    logger.debug(f"Selected example: {st.session_state.selected_example_name}")
    logger.debug(
        f"Selected prompt category: {st.session_state.selected_prompt_category}"
    )
    logger.debug(
        f"Active example framework hint: {st.session_state.active_example_framework_hint}"
    )
    logger.debug(
        f"Sidebar selected persona set: {st.session_state.selected_persona_set}"
    )
    update_activity_timestamp()
    st.rerun()


st.subheader("What would you like to do?")

tab_names = list(EXAMPLE_PROMPTS.keys()) + [CUSTOM_PROMPT_KEY]
tabs = st.tabs(tab_names)

for i, tab_name in enumerate(tab_names):
    with tabs[i]:
        if tab_name == CUSTOM_PROMPT_KEY:
            st.markdown("Create your own specialized prompt for unique requirements.")
            st.text_area(
                "Enter your custom prompt here:",
                value=st.session_state.user_prompt_input,
                height=150,
                key="custom_prompt_text_area_widget",
                on_change=on_custom_prompt_change,
            )

            with st.expander("💡 Prompt Engineering Tips"):
                st.markdown("""
                - **Be Specific:** Clearly define your goal and desired output.
                - **Provide Context:** Include relevant background information or code snippets.
                - **Define Constraints:** Specify any limitations (e.g., language, length, format).
                - **Example Output:** If possible, provide an.example of the desired output format.
                """)

            recommended_domain_for_custom = st.session_state.persona_manager.prompt_analyzer.recommend_domain_from_keywords(
                st.session_state.user_prompt_input
            )

            if (
                recommended_domain_for_custom
                and recommended_domain_for_custom
                != st.session_state.selected_persona_set
            ):
                st.info(
                    f"💡 Based on your custom prompt, the **'{recommended_domain_for_custom}'** framework might be appropriate."
                )
                if st.button(
                    f"Apply '{recommended_domain_for_custom}' Framework (Custom Prompt)",
                    type="secondary",
                    use_container_width=True,
                    key=f"apply_suggested_framework_main_{recommended_domain_for_custom.replace(' ', '_').lower()}",
                    on_click=update_activity_timestamp,
                ):
                    st.session_state.selected_persona_set = (
                        recommended_domain_for_custom
                    )
                    update_activity_timestamp()
                    st.rerun()

        else:
            st.markdown(f"Explore example prompts for **{tab_name}**:")

            category_options = EXAMPLE_PROMPTS[tab_name]

            st.text_input(
                f"Search prompts in {tab_name}",
                key=f"search_{tab_name}",
                value="",
                on_change=update_activity_timestamp,
            )

            filtered_prompts_in_category = {
                name: details
                for name, details in category_options.items()
                if (
                    f"search_{tab_name}" not in st.session_state
                    or not st.session_state[f"search_{tab_name}"]
                )
                or (st.session_state[f"search_{tab_name}"].lower() in name.lower())
                or (
                    st.session_state[f"search_{tab_name}"].lower()
                    in details["prompt"].lower()
                )
            }

            options_keys = list(filtered_prompts_in_category.keys())

            if not options_keys:
                st.info("No example prompts match your search in this category.")
                if st.session_state.selected_prompt_category == tab_name:
                    st.session_state.user_prompt_input = ""
                    st.session_state.selected_example_name = ""
                    st.session_state.codebase_context = {}
                    st.session_state.structured_codebase_context = {}
                    st.session_state.raw_file_contents = {}
                    st.session_state.uploaded_files = []
                continue

            initial_selectbox_index = 0

            current_selected_example_name = st.session_state.selected_example_name
            current_selected_prompt_category = st.session_state.selected_prompt_category

            if (
                current_selected_prompt_category == tab_name
                and current_selected_example_name in options_keys
            ):
                initial_selectbox_index = options_keys.index(
                    current_selected_example_name
                )

            selectbox_key = f"select_example_{tab_name.replace(' ', '_').replace('&', '').replace('(', '').replace(')', '')}"

            selected_example_key_for_this_tab = st.selectbox(
                "Select task:",
                options=options_keys,
                index=initial_selectbox_index,
                format_func=lambda x: f"{x} - {filtered_prompts_in_category[x]['description'][:60]}...",
                label_visibility="collapsed",
                key=selectbox_key,
                on_change=on_example_select_change,
                args=(selectbox_key, tab_name),
            )

            selected_prompt_details = filtered_prompts_in_category[
                selected_example_key_for_this_tab
            ]
            st.info(f"**Description:** {selected_prompt_details['description']}")
            with st.expander("View Full Prompt Text"):
                st.code(selected_prompt_details["prompt"], language="text")

            display_suggested_framework = selected_prompt_details.get("framework_hint")
            if (
                display_suggested_framework
                and display_suggested_framework != st.session_state.selected_persona_set
            ):
                st.info(
                    f"💡 Based on this example, the **'{display_suggested_framework}'** framework might be appropriate."
                )
                if st.button(
                    f"Apply '{display_suggested_framework}' Framework",
                    type="primary",
                    use_container_width=True,
                    key=f"apply_suggested_framework_example_{selected_example_key_for_this_tab}",
                    on_click=update_activity_timestamp,
                ):
                    st.session_state.selected_persona_set = display_suggested_framework
                    update_activity_timestamp()
                    st.rerun()

user_prompt = st.session_state.user_prompt_input

st.info(f"**Currently Active Prompt:**\n\n{user_prompt}")

logger.debug(
    f"Current user_prompt_input (from session state): {st.session_state.user_prompt_input[:100]}..."
)
logger.debug(f"Selected example: {st.session_state.selected_example_name}")
logger.debug(f"Selected prompt category: {st.session_state.selected_prompt_category}")
logger.debug(
    f"Active example framework hint: {st.session_state.active_example_framework_hint}"
)
logger.debug(f"Sidebar selected persona set: {st.session_state.selected_persona_set}")

col1, col2 = st.columns(2, gap="medium")
with col1:
    st.subheader("Reasoning Framework")

    if st.session_state.selected_example_name == CUSTOM_PROMPT_KEY:
        if user_prompt.strip():
            recommended_domain = st.session_state.persona_manager.prompt_analyzer.recommend_domain_from_keywords(
                user_prompt
            )
            if (
                recommended_domain
                and recommended_domain != st.session_state.selected_persona_set
            ):
                st.info(
                    f"💡 Based on your custom prompt, the **'{recommended_domain}'** framework might be appropriate."
                )
                if st.button(
                    f"Apply '{recommended_domain}' Framework (Custom Prompt)",
                    type="secondary",
                    use_container_width=True,
                    key=f"apply_suggested_framework_main_{recommended_domain.replace(' ', '_').lower()}",
                    on_click=update_activity_timestamp,
                ):
                    st.session_state.selected_persona_set = recommended_domain
                    update_activity_timestamp()
                    st.rerun()

    available_framework_options = st.session_state.persona_manager.available_domains
    unique_framework_options = sorted(list(set(available_framework_options)))

    current_framework_selection = st.session_state.selected_persona_set
    if current_framework_selection not in unique_framework_options:
        current_framework_selection = (
            unique_framework_options[0] if unique_framework_options else "General"
        )
        st.session_state.selected_persona_set = current_framework_selection

    st.selectbox(
        "Select Framework",
        options=unique_framework_options,
        index=unique_framework_options.index(current_framework_selection)
        if current_framework_selection in unique_framework_options
        else 0,
        key="selected_persona_set",
        help="Choose a domain-specific reasoning framework or a custom saved framework.",
        on_change=update_activity_timestamp,
    )

    if st.session_state.selected_persona_set:
        current_domain_persona_names = (
            st.session_state.persona_manager.get_persona_sequence_for_framework(
                st.session_state.selected_persona_set
            )
        )
        st.session_state.personas = {
            name: st.session_state.persona_manager.all_personas.get(name)
            for name in current_domain_persona_names
            if name in st.session_state.persona_manager.all_personas
        }

    with st.expander("⚙️ Custom Framework Management", expanded=False):
        tabs_framework = st.tabs(
            ["Save Current Framework", "Load/Manage Frameworks", "Export/Import"]
        )

        with tabs_framework[0]:
            st.info(
                "This will save the *currently selected framework* along with any *unsaved persona edits* made in the 'View and Edit Personas' section."
            )
            st.text_input(
                "Enter a name for your framework:",
                key="save_framework_input",
                value=st.session_state.get("save_framework_input", ""),
                on_change=update_activity_timestamp,
            )
            st.text_area(
                "Framework Description (Optional):",
                key="framework_description",
                value=st.session_state.get("framework_description", ""),
                height=50,
                on_change=update_activity_timestamp,
            )

            if st.session_state.persona_changes_detected:
                st.warning(
                    "Unsaved persona changes detected. Please save as a custom framework to persist them."
                )

            if (
                st.button("Save Current Framework", on_click=update_activity_timestamp)
                and st.session_state.save_framework_input
            ):
                update_activity_timestamp()
                current_framework_name = st.session_state.selected_persona_set
                current_active_personas_data = {
                    p_name: p_data.model_copy()
                    for p_name, p_data in st.session_state.persona_manager.all_personas.items()
                    if p_name
                    in st.session_state.persona_manager.get_persona_sequence_for_framework(
                        current_framework_name
                    )
                }

                success, message = st.session_state.persona_manager.save_framework(
                    st.session_state.save_framework_input,
                    current_framework_name,
                    current_active_personas_data,
                    st.session_state.framework_description,
                )
                if success:
                    st.toast(message)
                    st.rerun()
                else:
                    st.error(message)

        with tabs_framework[1]:
            all_available_frameworks_for_load = [
                ""
            ] + st.session_state.persona_manager.available_domains
            unique_framework_options_for_load = sorted(
                list(set(all_available_frameworks_for_load))
            )

            current_selection_for_load = ""
            if (
                st.session_state.selected_persona_set
                in unique_framework_options_for_load
            ):
                current_selection_for_load = st.session_state.selected_persona_set
            elif (
                st.session_state.selected_persona_set
                in st.session_state.persona_manager.all_custom_frameworks_data
            ):
                current_selection_for_load = st.session_state.selected_persona_set

            st.selectbox(
                "Select a framework to load:",
                options=unique_framework_options_for_load,
                index=unique_framework_options_for_load.index(
                    current_selection_for_load
                )
                if current_selection_for_load in unique_framework_options_for_load
                else 0,
                key="load_framework_select",
                on_change=update_activity_timestamp,
            )
            if (
                st.button("Load Selected Framework", on_click=update_activity_timestamp)
                and st.session_state.load_framework_select
            ):
                update_activity_timestamp()
                (
                    success,
                    message,
                    loaded_personas_dict,
                    loaded_persona_sets_dict,
                    new_selected_framework_name,
                ) = st.session_state.persona_manager.load_framework_into_session(
                    st.session_state.load_framework_select
                )

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
            if st.button(
                f"Export '{export_framework_name}'",
                use_container_width=True,
                key="export_framework_button",
                on_click=update_activity_timestamp,
            ):
                success, message, exported_content = (
                    st.session_state.persona_manager.export_framework_for_sharing(
                        export_framework_name
                    )
                )
                if success and exported_content:
                    st.download_button(
                        label=f"Download '{export_framework_name}.yaml'",
                        data=exported_content,
                        file_name=f"{export_framework_name}.yaml",
                        mime="application/x-yaml",
                        use_container_width=True,
                        key="download_exported_framework",
                        on_click=update_activity_timestamp,
                    )
                    st.success(message)
                else:
                    st.error(message)

            st.markdown("---")
            st.subheader("Import Framework")
            st.info("Upload a custom framework file (YAML or JSON) to import it.")
            uploaded_framework_file = st.file_uploader(
                "Upload Framework File",
                type=["yaml", "json"],
                key="import_framework_uploader",
                on_change=update_activity_timestamp,
            )
            if uploaded_framework_file is not None:
                if st.button(
                    "Import Uploaded Framework",
                    use_container_width=True,
                    key="perform_import_framework_button",
                    on_click=update_activity_timestamp,
                ):
                    file_content = uploaded_framework_file.getvalue().decode("utf-8")
                    success, message = (
                        st.session_state.persona_manager.import_framework(
                            file_content, uploaded_framework_file.name
                        )
                    )
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
            type=[
                "py",
                "js",
                "ts",
                "html",
                "css",
                "json",
                "yaml",
                "md",
                "txt",
                "java",
                "go",
                "rb",
                "php",
            ],
            help="Provide files for context. The AI will analyze them to generate consistent code.",
            key="code_context_uploader",
            on_change=update_activity_timestamp,
        )

        uploaded_files = st.session_state.code_context_uploader
        if uploaded_files:
            current_uploaded_file_names = {f.name for f in uploaded_files}
            previous_uploaded_file_names = {
                f.name for f in st.session_state.uploaded_files
            }

            if current_uploaded_file_names != previous_uploaded_file_names or (
                current_uploaded_file_names and not st.session_state.raw_file_contents
            ):
                if len(uploaded_files) > 100:
                    st.warning(
                        "Please upload a maximum of 100 files. Truncating to the first 100."
                    )
                    uploaded_files = uploaded_files[:100]

                temp_raw_file_contents = {}
                for file in uploaded_files:
                    try:
                        content = file.getvalue().decode("utf-8")
                        temp_raw_file_contents[file.name] = content
                    except Exception as e:
                        st.error(f"Error reading {file.name}: {e}")

                st.session_state.raw_file_contents = temp_raw_file_contents
                st.session_state.codebase_context = temp_raw_file_contents
                st.session_state.uploaded_files = uploaded_files
                st.toast(
                    f"{len(st.session_state.raw_file_contents)} file(s) loaded for context from upload."
                )

        elif (
            not st.session_state.uploaded_files
            and st.session_state.selected_example_name != "Custom Prompt"
        ):
            if not st.session_state.raw_file_contents:
                try:
                    st.info(
                        "No demo codebase context file found. Please upload files manually."
                    )
                    st.session_state.raw_file_contents = {}
                    st.session_state.codebase_context = {}
                    st.session_state.uploaded_files = []
                except (FileNotFoundError, ValueError, IOError) as e:
                    st.error(f"❌ Error loading demo codebase context: {e}")
                    st.session_state.raw_file_contents = {}
                    st.session_state.codebase_context = {}
                    st.session_state.uploaded_files = []
            else:
                st.success(
                    f"{len(st.session_state.raw_file_contents)} file(s) already loaded for context."
                )

        elif (
            not st.session_state.uploaded_files
            and st.session_state.selected_example_name == "Custom Prompt"
        ):
            if st.session_state.raw_file_contents:
                st.session_state.raw_file_contents = {}
                st.session_state.codebase_context = {}
                st.session_state.uploaded_files = []
                st.info("Codebase context cleared for custom prompt.")

st.markdown("---")
with st.expander(
    "⚙️ View and Edit Personas", expanded=st.session_state.persona_edit_mode
):
    st.session_state.persona_edit_mode = True
    update_activity_timestamp()

    st.info(
        "Edit persona parameters for the **currently selected framework**. Changes are temporary unless saved as a custom framework."
    )

    if not st.session_state.persona_changes_detected:
        current_framework_persona_names = (
            st.session_state.persona_manager.get_persona_sequence_for_framework(
                st.session_state.selected_persona_set
            )
        )
        for p_name in current_framework_persona_names:
            persona: PersonaConfig = st.session_state.persona_manager.all_personas.get(
                p_name
            )
            original_persona_config = (
                st.session_state.persona_manager._original_personas.get(p_name)
            )

            if persona and original_persona_config:
                if (
                    persona.system_prompt_template
                    != original_persona_config.system_prompt_template
                    or persona.temperature != original_persona_config.temperature
                    or persona.max_tokens != original_persona_config.max_tokens
                ):
                    st.session_state.persona_changes_detected = True
                    break

    if st.session_state.persona_changes_detected:
        st.warning(
            "Unsaved changes detected in persona configurations. Please save as a custom framework or reset to persist them."
        )
        if st.button(
            "Reset All Personas for Current Framework",
            key="reset_all_personas_button",
            use_container_width=True,
            on_click=update_activity_timestamp,
        ):
            update_activity_timestamp()
            if st.session_state.persona_manager.reset_all_personas_for_current_framework(
                st.session_state.selected_persona_set
            ):
                st.toast("All personas for the current framework reset to default.")
                st.session_state.persona_changes_detected = False
                st.rerun()
            else:
                st.error("Could not reset all personas for the current framework.")

    sorted_persona_names = sorted(st.session_state.personas.keys())

    for p_name in sorted_persona_names:
        persona: PersonaConfig = st.session_state.persona_manager.all_personas.get(
            p_name
        )
        if not persona:
            st.warning(f"Persona '{p_name}' not found in manager. Skipping.")
            continue

        with st.expander(f"**{persona.name.replace('_', ' ')}**", expanded=False):
            st.markdown(f"**Description:** {persona.description}")

            # MODIFIED: Use system_prompt_template for value
            new_system_prompt = st.text_area(
                "System Prompt",
                value=persona.system_prompt_template,
                height=200,
                key=f"system_prompt_{p_name}",
                help="The core instructions for this persona.",
            )
            # MODIFIED: Compare against system_prompt_template
            if new_system_prompt != persona.system_prompt_template:
                # MODIFIED: Log change for system_prompt_template
                _log_persona_change(
                    p_name,
                    "system_prompt_template",
                    persona.system_prompt_template,
                    new_system_prompt,
                )
                # MODIFIED: Assign to system_prompt_template
                persona.system_prompt_template = new_system_prompt

            new_temperature = st.slider(
                "Temperature",
                min_value=0.0,
                max_value=1.0,
                value=persona.temperature,
                step=0.05,
                key=f"temperature_{p_name}",
                help="Controls the randomness of the output. Lower values mean less random.",
            )
            if new_temperature != persona.temperature:
                _log_persona_change(
                    p_name, "temperature", persona.temperature, new_temperature
                )
                persona.temperature = new_temperature

            new_max_tokens = st.number_input(
                "Max Output Tokens",
                min_value=1,
                max_value=MAX_TOKENS_LIMIT,
                value=persona.max_tokens,
                step=128,
                key=f"max_tokens_{p_name}",
                help="Maximum number of tokens the LLM can generate in response.",
            )
            if new_max_tokens != persona.max_tokens:
                _log_persona_change(
                    p_name, "max_tokens", persona.max_tokens, new_max_tokens
                )
                persona.max_tokens = new_max_tokens

            if st.button(
                f"Reset {p_name.replace('_', ' ')} to Default",
                key=f"reset_persona_{p_name}",
                on_click=update_activity_timestamp,
            ):
                update_activity_timestamp()
                if st.session_state.persona_manager.reset_persona_to_default(p_name):
                    st.toast(f"Persona '{p_name.replace('_', ' ')}' reset to default.")
                    st.session_state.persona_changes_detected = False
                    st.rerun()
                else:
                    st.error("Could not reset all personas for the current framework.")

st.markdown("---")
run_col, reset_col = st.columns([0.8, 0.2])
with run_col:
    run_button_clicked = st.button(
        "🚀 Run Socratic Debate",
        type="primary",
        use_container_width=True,
        disabled=not (
            st.session_state.api_key_input and st.session_state.api_key_functional
        ),
    )
with reset_col:
    st.button(
        "🔄 Reset All",
        on_click=lambda: reset_app_state(settings_instance, EXAMPLE_PROMPTS),
        use_container_width=True,
    )


def _run_socratic_debate_process():
    """Handles the execution of the Socratic debate process."""

    debate_instance: Optional[SocraticDebate] = None

    request_id = str(uuid.uuid4())[:8]

    is_self_analysis_prompt_detected = (
        st.session_state.persona_manager.prompt_analyzer.is_self_analysis_prompt(
            user_prompt
        )
    )

    if is_self_analysis_prompt_detected:
        st.info(
            "Self-analysis prompt detected. Loading Project Chimera's codebase context..."
        )
        try:
            scanner = get_codebase_scanner_instance()
            full_codebase_analysis = scanner.load_own_codebase_context()

            st.session_state.codebase_scanner.file_structure = (
                full_codebase_analysis.get("file_structure", {})
            )
            st.session_state.codebase_scanner.raw_file_contents = (
                full_codebase_analysis.get("raw_file_contents", {})
            )

            st.session_state.context_analyzer.raw_file_contents = (
                st.session_state.codebase_scanner.raw_file_contents
            )
            st.session_state.context_analyzer.compute_file_embeddings(
                st.session_state.context_analyzer.raw_file_contents
            )
            st.session_state.context_analyzer._last_raw_file_contents_hash = hash(
                frozenset(st.session_state.context_analyzer.raw_file_contents.items())
            )

            st.session_state.codebase_context = (
                st.session_state.codebase_scanner.file_structure
            )
            st.session_state.raw_file_contents = (
                st.session_state.codebase_scanner.raw_file_contents
            )

            logger.info(
                "Successfully loaded Project Chimera's codebase context for self-analysis into cached instances."
            )
            if not st.session_state.raw_file_contents or all(
                not content for content in st.session_state.raw_file_contents.values()
            ):
                st.warning(
                    "⚠️ Self-analysis: No codebase content available for analysis. "
                    "This might indicate an issue with the scanner's include/exclude patterns, "
                    "empty files, or an empty project directory. Self-analysis will proceed but may be limited."
                )
                logger.warning(
                    "Self-analysis: Codebase context is empty after loading."
                )
        except Exception as e:
            if st.session_state.context_analyzer:
                st.session_state.context_analyzer.file_embeddings = {}
                st.session_state.context_analyzer.raw_file_contents = {}
            st.error(
                f"❌ Error loading Project Chimera's codebase for self-analysis: {e}"
            )
            logger.error(f"Failed to load own codebase context: {e}", exc_info=True)
            return

    logger.info(
        "Starting Socratic Debate process.",
        extra={"request_id": request_id, "user_prompt": user_prompt},
    )

    st.session_state.token_tracker.reset()
    st.session_state.file_analysis_cache = None

    if not st.session_state.api_key_input.strip():
        st.error("Please enter your Gemini API Key in the sidebar to proceed.")
        logger.warning(
            "API key missing, debate process aborted.", extra={"request_id": request_id}
        )
        return
    elif not st.session_state.api_key_functional:
        st.error(
            "Your Gemini API Key is not functional. Please test it in the sidebar."
        )
        logger.warning(
            "API key not functional, debate process aborted.",
            extra={"request_id": request_id},
        )
        return
    elif not user_prompt.strip():
        st.error("Please enter a prompt.")
        logger.warning(
            "User prompt is empty, debate process aborted.",
            extra={"request_id": request_id},
        )
        return

    try:
        st.session_state.session_rate_limiter_instance(lambda: None)()
    except RateLimitExceededError as e:
        handle_debate_errors(e)
        return
    except Exception as e:
        handle_debate_errors(e)
        return

    sanitized_prompt = sanitize_user_input(user_prompt)
    if sanitized_prompt != user_prompt:
        st.warning("User prompt was sanitized to mitigate potential injection risks.")
        st.session_state.user_prompt_input = sanitized_prompt
        logger.info(
            "Prompt was sanitized.",
            extra={
                "request_id": request_id,
                "original_prompt": user_prompt,
                "sanitized_prompt": sanitized_prompt,
            },
        )
    else:
        logger.debug(
            "Prompt did not require sanitization.", extra={"request_id": request_id}
        )

    current_user_prompt_for_debate = sanitized_prompt

    st.session_state.debate_ran = False
    final_answer = {
        "ANALYSIS_SUMMARY": "Debate Failed - Unhandled Error",  # MODIFIED: Use ANALYSIS_SUMMARY for self-improvement domain
        "IMPACTFUL_SUGGESTIONS": [],  # MODIFIED: Use IMPACTFUL_SUGGESTIONS
        "malformed_blocks": [
            {
                "type": "UNHANDLED_ERROR_INIT",
                "message": "Debate failed during initialization or early phase.",
            }
        ],
    }

    with st.status("Socratic Debate in Progress", expanded=True) as status:
        main_progress_message = st.empty()
        main_progress_message.markdown("### Initializing debate...")

        overall_progress_bar = st.progress(0)

        active_persona_placeholder = st.empty()

        def update_status(
            message,
            state,
            current_total_tokens,
            current_total_cost,
            estimated_next_step_tokens=0,
            estimated_next_step_cost=0.0,
            progress_pct: float = None,
            current_persona_name: str = None,
        ):
            main_progress_message.markdown(f"### {message}")

            if current_persona_name:
                active_persona_placeholder.markdown(
                    f"Currently running: [bold]{current_persona_name}[/bold]..."
                )
            else:
                active_persona_placeholder.empty()

            if progress_pct is not None:
                st.session_state.debate_progress = max(0.0, min(1.0, progress_pct))
                overall_progress_bar.progress(st.session_state.debate_progress)
            else:
                st.session_state.debate_progress = min(
                    st.session_state.debate_progress + 0.01, 0.99
                )
                overall_progress_bar.progress(st.session_state.debate_progress)

        def update_status_with_realtime_metrics(
            message,
            state,
            current_total_tokens,
            current_total_cost,
            estimated_next_step_tokens=0,
            estimated_next_step_cost=0.0,
            progress_pct: float = None,
            current_persona_name: str = None,
        ):
            update_status(
                message,
                state,
                current_total_tokens,
                current_total_cost,
                estimated_next_step_tokens,
                estimated_next_step_cost,
                progress_pct,
                current_persona_name,
            )

            st.session_state.current_debate_tokens_used = current_total_tokens
            st.session_state.current_debate_cost_usd = current_total_cost

        @contextlib.contextmanager
        def capture_rich_output_and_get_console():
            """Captures rich output (like Streamlit elements) and returns the captured content."""
            buffer = io.StringIO()
            console_instance = Console(file=buffer, force_terminal=True, soft_wrap=True)
            yield buffer, console_instance

        with capture_rich_output_and_get_console() as (
            rich_output_buffer,
            rich_console_instance,
        ):
            try:
                domain_for_run = st.session_state.selected_persona_set

                if (
                    st.session_state.selected_example_name != CUSTOM_PROMPT_KEY
                    and st.session_state.active_example_framework_hint
                ):
                    domain_for_run = st.session_state.active_example_framework_hint
                    logger.debug(
                        f"Using active example framework hint: '{domain_for_run}' for example '{st.session_state.selected_example_name}'."
                    )
                elif st.session_state.selected_example_name == CUSTOM_PROMPT_KEY:
                    recommended_domain = st.session_state.persona_manager.prompt_analyzer.recommend_domain_from_keywords(
                        current_user_prompt_for_debate
                    )
                    if recommended_domain:
                        domain_for_run = recommended_domain
                        logger.debug(
                            f"Using recommended domain for custom prompt: '{domain_for_run}'."
                        )

                logger.info(f"Final domain selected for debate: {domain_for_run}")

                logger.debug(f"_run_socratic_debate_process started.")
                logger.debug(
                    f"Prompt at start of debate function: {current_user_prompt_for_debate[:100]}..."
                )
                logger.debug(
                    f"Domain selection logic - Initial domain_for_run: {st.session_state.selected_persona_set}"
                )
                logger.debug(
                    f"Domain selection logic - Selected example name: {st.session_state.selected_example_name}"
                )
                logger.debug(
                    f"Domain selection logic - Active example framework hint: {st.session_state.active_example_framework_hint}"
                )
                logger.debug(
                    f"Domain selection logic - Sidebar selected persona set: {st.session_state.selected_persona_set}"
                )
                logger.debug(
                    f"Domain selection logic - Final domain_for_run: {domain_for_run}"
                )

                if (
                    st.session_state.context_analyzer
                    and st.session_state.codebase_scanner
                    and st.session_state.codebase_scanner.raw_file_contents
                ):
                    current_files_hash = hash(
                        frozenset(
                            st.session_state.codebase_scanner.raw_file_contents.items()
                        )
                    )
                    if (
                        not hasattr(
                            st.session_state.context_analyzer,
                            "_last_raw_file_contents_hash",
                        )
                        or st.session_state.context_analyzer._last_raw_file_contents_hash
                        != current_files_hash
                    ):
                        st.session_state.context_analyzer.raw_file_contents = (
                            st.session_state.codebase_scanner.raw_file_contents
                        )
                        try:
                            st.session_state.context_analyzer.compute_file_embeddings(
                                st.session_state.codebase_scanner.raw_file_contents
                            )
                            st.session_state.context_analyzer._last_raw_file_contents_hash = current_files_hash
                            logger.info(
                                "Re-computed file embeddings for context analyzer in app.py before debate init."
                            )
                        except Exception as e:
                            st.error(f"❌ Error re-computing context embeddings: {e}")
                            logger.error(
                                f"Failed to re-compute context embeddings: {e}",
                                exc_info=True,
                            )
                            return

                debate_instance = SocraticDebate(
                    initial_prompt=current_user_prompt_for_debate,
                    # MODIFIED: Use settings_instance.GEMINI_API_KEY as fallback
                    api_key=st.session_state.api_key_input
                    or settings_instance.GEMINI_API_KEY,
                    model_name=st.session_state.selected_model_selectbox,
                    all_personas=st.session_state.all_personas,
                    persona_sets=st.session_state.persona_sets,
                    domain=domain_for_run,
                    status_callback=update_status_with_realtime_metrics,
                    rich_console=rich_console_instance,
                    context_analyzer=st.session_state.context_analyzer,
                    is_self_analysis=is_self_analysis_prompt_detected,
                    settings=settings_instance,
                    persona_manager=st.session_state.persona_manager,
                    token_tracker=st.session_state.token_tracker,
                    codebase_scanner=get_codebase_scanner_instance(),
                    summarizer_pipeline_instance=get_summarizer_pipeline_instance(),
                )

                logger.info(
                    "Executing Socratic Debate via core.SocraticDebate.",
                    extra={
                        "request_id": request_id,
                        "debate_instance_id": id(debate_instance),
                    },
                )

                final_answer, st.session_state.intermediate_steps_output = (
                    debate_instance.run_debate()
                )

                if (
                    hasattr(debate_instance, "file_analysis_cache")
                    and debate_instance.file_analysis_cache
                ):
                    st.session_state.file_analysis_cache = (
                        debate_instance.file_analysis_cache
                    )
                    logger.debug("Captured file_analysis_cache from debate instance.")
                else:
                    st.session_state.file_analysis_cache = None

                logger.info(
                    "Socratic Debate execution finished.",
                    extra={
                        "request_id": request_id,
                        "debate_instance_id": id(debate_instance),
                    },
                )

                st.session_state.process_log_output_text = rich_output_buffer.getvalue()
                st.session_state.final_answer_output = final_answer
                st.session_state.last_config_params = {
                    "max_tokens_budget": st.session_state.max_tokens_budget_input,
                    "model_name": st.session_state.selected_model_selectbox,
                    "show_intermediate_steps": st.session_state.show_intermediate_steps_checkbox,
                    "domain": domain_for_run,
                }
                st.session_state.debate_ran = True
                status.update(
                    label="Socratic Debate Complete!", state="complete", expanded=False
                )

            except (
                TokenBudgetExceededError,
                SchemaValidationError,
                ChimeraError,
                CircuitBreakerError,
                LLMProviderError,
                TypeError,
            ) as e:
                handle_debate_errors(e)
                status.update(
                    label=f"Socratic Debate Failed: {type(e).__name__}",
                    state="error",
                    expanded=True,
                )
                st.session_state.debate_ran = True
                if debate_instance:
                    st.session_state.intermediate_steps_output = (
                        debate_instance.intermediate_steps
                    )
            except Exception as e:
                handle_debate_errors(e)
                status.update(
                    label=f"Socratic Debate Failed: An unexpected error occurred",
                    state="error",
                    expanded=True,
                )
                st.session_state.debate_ran = True
                if debate_instance:
                    st.session_state.intermediate_steps_output = (
                        debate_instance.intermediate_steps
                    )
            finally:
                if debate_instance:
                    if hasattr(debate_instance, "close") and callable(
                        debate_instance.close
                    ):
                        logger.info(
                            f"Calling close() on SocraticDebate instance {id(debate_instance)}."
                        )
                        debate_instance.close()
                    del debate_instance
                gc.collect()
                logger.info(
                    "Explicit garbage collection triggered in app.py after debate process."
                )


if run_button_clicked:
    update_activity_timestamp()

    for attempt in range(MAX_DEBATE_RETRIES):
        try:
            _run_socratic_debate_process()
            break
        except RateLimitExceededError as e:
            if attempt < MAX_DEBATE_RETRIES - 1:
                wait_time = DEBATE_RETRY_DELAY_SECONDS * (attempt + 1)
                st.info(
                    f"Rate limit exceeded. Retrying Socratic Debate in {wait_time:.1f} seconds... (Attempt {attempt + 1}/{MAX_DEBATE_RETRIES})"
                )
                time.sleep(wait_time)
                update_activity_timestamp()
            else:
                st.error(
                    f"Max retries ({MAX_DEBATE_RETRIES}) for Socratic Debate reached due to rate limiting. Please try again later."
                )
                handle_debate_errors(e)
                break
        except LLMProviderError as e:
            provider_error_code = e.details.get("provider_error_code")
            if (
                isinstance(provider_error_code, int)
                and 500 <= provider_error_code < 600
            ):
                if attempt < MAX_DEBATE_RETRIES - 1:
                    wait_time = DEBATE_RETRY_DELAY_SECONDS * (attempt + 1)
                    st.info(
                        f"LLM Server Error ({provider_error_code}) detected. Retrying Socratic Debate in {wait_time:.1f} seconds... (Attempt {attempt + 1}/{MAX_DEBATE_RETRIES})"
                    )
                    time.sleep(wait_time)
                    update_activity_timestamp()
                    continue
                else:
                    st.error(
                        f"Max retries ({MAX_DEBATE_RETRIES}) for Socratic Debate reached due to LLM server error ({provider_error_code}). Please try again later."
                    )
                    handle_debate_errors(e)
                    break
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
        format_choice = st.radio(
            "Choose report format:",
            ["Complete Report (Markdown)", "Summary (Text)"],
            label_visibility="collapsed",
            key="report_format_radio",
            on_change=update_activity_timestamp,
        )

        report_content = (
            generate_markdown_report(
                user_prompt=user_prompt,
                final_answer=st.session_state.final_answer_output,
                intermediate_steps=st.session_state.intermediate_steps_output,
                process_log_output=st.session_state.process_log_output_text,
                config_params=st.session_state.last_config_params,
                persona_audit_log=st.session_state.persona_audit_log,
            )
            if "Complete" in format_choice
            else "This is a placeholder for the summary report. Implement summary generation logic here."
        )

        file_name = f"chimera_report_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}{'_full' if 'Complete' in format_choice else '_summary'}.{'md' if 'Complete' in format_choice else 'txt'}"

        st.download_button(
            label="⬇️ Download Selected Format",
            data=report_content,
            file_name=file_name,
            use_container_width=True,
            type="primary",
            on_click=update_activity_timestamp,
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
                    json.dumps(raw_output_data), LLMOutput
                )
                malformed_blocks_from_parser.extend(
                    parsed_llm_output_dict.get("malformed_blocks", [])
                )
            except Exception as e:
                st.error(
                    f"Failed to parse final LLMOutput for Software Engineering domain: {e}"
                )
                parsed_llm_output_dict = {
                    "COMMIT_MESSAGE": "Parsing Error",
                    "RATIONALE": f"Failed to parse final LLM output into expected structure. Error: {e}",
                    "CODE_CHANGES": [],
                    "malformed_blocks": [
                        {
                            "type": "UI_PARSING_ERROR",
                            "message": str(e),
                            "raw_string_snippet": str(raw_output_data)[:500],
                        }
                    ],
                }
        elif isinstance(raw_output_data, list) and raw_output_data:
            try:
                parsed_llm_output_dict = LLMOutputParser().parse_and_validate(
                    json.dumps(raw_output_data[0]), LLMOutput
                )
                malformed_blocks_from_parser.extend(
                    parsed_llm_output_dict.get("malformed_blocks", [])
                )
            except Exception as e:
                st.error(
                    f"Failed to parse list-based LLMOutput for Software Engineering domain: {e}"
                )
                parsed_llm_output_dict = {
                    "COMMIT_MESSAGE": "Parsing Error",
                    "RATIONALE": f"Failed to parse list-based LLM output into expected structure. Error: {e}",
                    "CODE_CHANGES": [],
                    "malformed_blocks": [
                        {
                            "type": "UI_PARSING_ERROR",
                            "message": str(e),
                            "raw_string_snippet": str(raw_output_data)[:500],
                        }
                    ],
                }
        else:
            st.error(
                f"Final answer for Software Engineering is not a structured dictionary or list. Raw output type: {type(raw_output_data).__name__}"
            )
            parsed_llm_output_dict = {
                "COMMIT_MESSAGE": "Error: Output not structured.",
                "RATIONALE": f"Error: Output not structured. Raw output type: {type(raw_output_data).__name__}",
                "CODE_CHANGES": [],
                "malformed_blocks": [
                    {
                        "type": "UI_PARSING_ERROR",
                        "message": f"Final answer was not a dictionary or list. Type: {type(raw_output_data).__name__}",
                        "raw_string_snippet": str(raw_output_data)[:500],
                    }
                ],
            }

        validation_results_by_file = validate_code_output_batch(
            parsed_llm_output_dict,
            st.session_state.get("raw_file_contents", {}),
            file_analysis_cache=st.session_state.get("file_analysis_cache", None),
        )

        all_issues = []
        if isinstance(validation_results_by_file, dict):
            for file_issues_list in validation_results_by_file.values():
                if isinstance(file_issues_list, list):
                    all_issues.extend(file_issues_list)

        all_malformed_blocks = malformed_blocks_from_parser
        if (
            isinstance(validation_results_by_file, dict)
            and "malformed_blocks" in validation_results_by_file
        ):
            all_malformed_blocks.extend(validation_results_by_file["malformed_blocks"])

        summary_col1, summary_col2 = st.columns(2, gap="medium")
        with summary_col1:
            st.markdown("**Commit Message Suggestion**")
            st.code(
                parsed_llm_output_dict.get("COMMIT_MESSAGE", "N/A"), language="text"
            )
        with summary_col2:
            st.markdown("**Token Usage**")
            total_tokens = st.session_state.intermediate_steps_output.get(
                "Total_Tokens_Used", 0
            )
            total_cost = st.session_state.intermediate_steps_output.get(
                "Total_Estimated_Cost_USD", 0.0
            )
            st.metric("Total Tokens Consumed", f"{total_tokens:,}")
            st.metric("Total Estimated Cost (USD)", f"${total_cost:.6f}")
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
                    st.error(
                        f"**Malformed Output Detected:** The LLM produced {len(all_malformed_blocks)} block(s) that could not be parsed or validated correctly. Raw output snippets are provided below."
                    )

                if all_malformed_blocks:
                    with st.expander("Malformed Output Details"):
                        for block_info in all_malformed_blocks:
                            st.error(
                                f"**Type:** {block_info.get('type', 'Unknown')}\n**Message:** {block_info.get('message', 'N/A')}"
                            )
                            raw_snippet = block_info.get("raw_string_snippet", "")
                            if not raw_snippet:
                                st.code("<No content available>", language="text")
                            else:
                                display_content = raw_snippet[:1000]
                                if len(raw_snippet) > 1000:
                                    display_content += "..."
                                st.code(display_content, language="text")
                            st.markdown("---")

                issues_by_file = defaultdict(list)
                for issue in all_issues:
                    issues_by_file[issue.get("file", "N/A")].append(issue)

                for file_path, file_issues in issues_by_file.items():
                    with st.expander(
                        f"File: `{file_path}` ({len(file_issues)} issues)",
                        expanded=False,
                    ):
                        issues_by_type = defaultdict(list)
                        for issue in file_issues:
                            issues_by_type[issue.get("type", "Unknown")].append(issue)

                        for issue_type, type_issues in sorted(issues_by_type.items()):
                            with st.expander(
                                f"**{issue_type}** ({len(type_issues)} issues)",
                                expanded=False,
                            ):
                                for issue in type_issues:
                                    line_info = (
                                        f" (Line: {issue.get('line', 'N/A')}, Col: {issue.get('column', 'N/A')})"
                                        if issue.get("line")
                                        else ""
                                    )
                                    st.markdown(
                                        f"- **{issue.get('code', '')}**: {issue['message']}{line_info}"
                                    )

        st.subheader("Proposed Code Changes")
        if not parsed_llm_output_dict.get("CODE_CHANGES") and not all_malformed_blocks:
            st.info("No code changes were proposed.")

        for change in parsed_llm_output_dict.get("CODE_CHANGES", []):
            with st.expander(
                f"📝 **{change.get('FILE_PATH', 'N/A')}** (`{change.get('ACTION', 'N/A')}`)",
                expanded=False,
            ):
                st.write(f"**Action:** {change.get('ACTION', 'N/A')}")
                st.write(f"**File Path:** {change.get('FILE_PATH', 'N/A')}")

                if change.get("ACTION") in [
                    "ADD",
                    "MODIFY",
                    "CREATE",
                    "CREATE_DIRECTORY",
                ]:
                    if change.get("ACTION") == "MODIFY":
                        original_content = st.session_state.raw_file_contents.get(
                            change.get("FILE_PATH", "N/A"), ""
                        )
                        if original_content:
                            diff_lines = difflib.unified_diff(
                                original_content.splitlines(keepends=True),
                                change.get("FULL_CONTENT", "").splitlines(
                                    keepends=True
                                ),
                                fromfile=f"a/{change.get('FILE_PATH', 'N/A')}",
                                tofile=f"b/{change.get('FILE_PATH', 'N/A')}",
                                lineterm="",
                            )
                            diff_output = "\n".join(diff_lines)
                            st.write("**Changes:**")
                            st.code(diff_output, language="diff")
                        else:
                            st.write("**New Content:**")
                            st.code(change.get("FULL_CONTENT", ""), language="python")
                    else:
                        st.write("**Content:**")
                        display_content = (
                            change.get("FULL_CONTENT", "")[:1500] + "..."
                            if len(change.get("FULL_CONTENT", "")) > 1500
                            else change.get("FULL_CONTENT", "")
                        )
                        st.code(display_content, language="python")

                    st.download_button(
                        label=f"Download {'File' if change.get('ACTION') == 'ADD' else 'New File Content'}",
                        data=change.get("FULL_CONTENT", ""),
                        file_name=change.get("FILE_PATH", "N/A"),
                        use_container_width=True,
                        type="secondary",
                        on_click=update_activity_timestamp,
                    )

                elif change.get("ACTION") == "REMOVE":
                    st.write("**Lines to Remove:**")
                    st.code("\n".join(change.get("LINES", [])), language="text")

    elif actual_debate_domain == "Self-Improvement":
        st.subheader("Final Synthesized Answer")
        final_analysis_output = st.session_state.final_answer_output
        malformed_blocks_from_parser = []

        analysis_summary = "Error: Output not structured for Self-Improvement."
        impactful_suggestions = []

        if isinstance(final_analysis_output, dict):
            malformed_blocks_from_parser.extend(
                final_analysis_output.get("malformed_blocks", [])
            )

            if (
                final_analysis_output.get("version") == "1.0"
                and "data" in final_analysis_output
            ):
                v1_data = final_analysis_output.get("data", {})
                analysis_summary = v1_data.get("ANALYSIS_SUMMARY", "N/A")
                impactful_suggestions = v1_data.get("IMPACTFUL_SUGGESTIONS", [])
                malformed_blocks_from_parser.extend(v1_data.get("malformed_blocks", []))
            else:
                analysis_summary = "Error: Unexpected SelfImprovementAnalysisOutput version or structure."
                malformed_blocks_from_parser.append(
                    {
                        "type": "UNEXPECTED_VERSION_OR_STRUCTURE",
                        "message": analysis_summary,
                        "raw_string_snippet": str(final_analysis_output)[:500],
                    }
                )
        else:
            st.error(
                f"Final answer for Self-Improvement is not a structured dictionary. Raw output type: {type(final_analysis_output).__name__}"
            )
            analysis_summary = f"Error: Output not structured for Self-Improvement. Raw output type: {type(final_analysis_output).__name__}"
            malformed_blocks_from_parser.append(
                {
                    "type": "UI_PARSING_ERROR",
                    "message": analysis_summary,
                    "raw_string_snippet": str(final_analysis_output)[:500],
                }
            )

        st.markdown("**Analysis Summary**")
        st.markdown(analysis_summary)

        if (
            "LLM returned a single suggestion item instead of the full analysis."
            in analysis_summary
        ):
            st.warning(
                "⚠️ The LLM provided only a partial analysis (a single suggestion) due to potential token limits or reasoning constraints. Consider refining your prompt or increasing the token budget for a more comprehensive report."
            )

        st.markdown("**Impactful Suggestions**")
        if not impactful_suggestions:
            st.info("No specific suggestions were provided in the analysis.")
        else:
            for suggestion_idx, suggestion in enumerate(impactful_suggestions):
                with st.expander(
                    f"💡 {suggestion.get('AREA', 'N/A')}: {suggestion.get('PROBLEM', 'N/A')[:80]}...",
                    expanded=False,
                ):
                    st.markdown(f"**Area:** {suggestion.get('AREA', 'N/A')}")
                    st.markdown(f"**Problem:** {suggestion.get('PROBLEM', 'N/A')}")
                    st.markdown(
                        f"**Proposed Solution:** {suggestion.get('PROPOSED_SOLUTION', 'N/A')}"
                    )
                    st.markdown(
                        f"**Expected Impact:** {suggestion.get('EXPECTED_IMPACT', 'N/A')}"
                    )

                    code_changes = suggestion.get("CODE_CHANGES_SUGGESTED", [])
                    if code_changes:
                        st.markdown("**Suggested Code Changes:**")
                        for change_idx, change in enumerate(code_changes):
                            with st.expander(
                                f"📝 {change.get('FILE_PATH', 'N/A')} (`{change.get('ACTION', 'N/A')}`)",
                                expanded=False,
                            ):
                                st.write(f"**Action:** {change.get('ACTION', 'N/A')}")
                                st.write(
                                    f"**File Path:** {change.get('FILE_PATH', 'N/A')}"
                                )

                                if change.get("ACTION") in [
                                    "ADD",
                                    "MODIFY",
                                    "CREATE",
                                    "CREATE_DIRECTORY",
                                ]:
                                    if change.get("DIFF_CONTENT"):
                                        st.write("**Changes (Unified Diff):**")
                                        st.code(
                                            change.get("DIFF_CONTENT", ""),
                                            language="diff",
                                        )
                                        st.download_button(
                                            label=f"Download Diff for {change.get('FILE_PATH', 'N/A')}",
                                            data=change.get("DIFF_CONTENT", ""),
                                            file_name=f"{Path(change.get('FILE_PATH', 'N/A')).name}.diff",
                                            use_container_width=True,
                                            type="secondary",
                                            on_click=update_activity_timestamp,
                                            key=f"diff_download_{suggestion_idx}_{change_idx}_{change.get('FILE_PATH', 'N/A').replace('/', '_')}",
                                        )
                                    elif change.get("FULL_CONTENT"):
                                        st.write("**Content:**")
                                        display_content = (
                                            change.get("FULL_CONTENT", "")[:1500]
                                            + "..."
                                            if len(change.get("FULL_CONTENT", ""))
                                            > 1500
                                            else change.get("FULL_CONTENT", "")
                                        )
                                        st.code(display_content, language="python")
                                        st.download_button(
                                            label=f"Download {'File' if change.get('ACTION') == 'ADD' else 'New File Content'}",
                                            data=change.get("FULL_CONTENT", ""),
                                            file_name=change.get("FILE_PATH", "N/A"),
                                            use_container_width=True,
                                            type="secondary",
                                            on_click=update_activity_timestamp,
                                            key=f"full_download_{suggestion_idx}_{change_idx}_{change.get('FILE_PATH', 'N/A').replace('/', '_')}",
                                        )
                                    else:
                                        st.info(
                                            "No content or diff provided for this change."
                                        )
                                elif change.get("ACTION") == "REMOVE":
                                    st.write("**Lines to Remove:**")
                                    st.code(
                                        "\n".join(change.get("LINES", [])),
                                        language="text",
                                    )
                    else:
                        st.info("No specific code changes suggested for this item.")

        if malformed_blocks_from_parser:
            with st.expander("Malformed Blocks (Self-Improvement Output)"):
                st.json(malformed_blocks_from_parser)

    else:
        st.subheader("Final Synthesized Answer")
        if (
            isinstance(st.session_state.final_answer_output, dict)
            and "general_output" in st.session_state.final_answer_output
        ):
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
            display_steps = {
                k: v
                for k, v in st.session_state.intermediate_steps_output.items()
                if not k.endswith("_Tokens_Used")
                and not k.endswith("_Estimated_Cost_USD")
                and k != "Total_Tokens_Used"
                and k != "Total_Estimated_Cost_USD"
                and k != "debate_history"
                and not k.startswith("malformed_blocks")
            }
            sorted_step_keys = sorted(
                display_steps.keys(),
                key=lambda x: (x.split("_")[0] if "_" in x else "", x),
            )
            for step_key in sorted_step_keys:
                persona_name = step_key.split("_")[0]
                display_name = (
                    step_key.replace("_Output", "")
                    .replace("_Critique", "")
                    .replace("_Feedback", "")
                    .replace("_", " ")
                    .title()
                )
                content = display_steps.get(step_key, "N/A")
                token_base_name = (
                    step_key.replace("_Output", "")
                    .replace("_Critique", "")
                    .replace("_Feedback", "")
                )
                token_count_key = f"{token_base_name}_Tokens_Used"
                tokens_used = st.session_state.intermediate_steps_output.get(
                    token_count_key, "N/A"
                )

                actual_temp = st.session_state.intermediate_steps_output.get(
                    f"{persona_name}_Actual_Temperature"
                )
                actual_max_tokens = st.session_state.intermediate_steps_output.get(
                    f"{persona_name}_Actual_Max_Tokens"
                )

                persona_params_info = ""
                if actual_temp is not None or actual_max_tokens is not None:
                    persona_params_info = " (Parameters: "
                    if actual_temp is not None:
                        persona_params_info += f"Temp={actual_temp:.2f}"
                    if actual_max_tokens is not None:
                        if actual_temp is not None:
                            persona_params_info += ", "
                        persona_params_info += f"MaxTokens={actual_max_tokens}"
                    persona_params_info += ")"

                with st.expander(
                    f"**{display_name}** (Tokens: {tokens_used}){persona_params_info}"
                ):
                    if isinstance(content, dict):
                        st.json(content)
                    else:
                        st.markdown(f"```markdown\n{content}\n```")
        st.subheader("Process Log")
        st.code(
            strip_ansi_codes(st.session_state.process_log_output_text), language="text"
        )
