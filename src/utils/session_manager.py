# src/utils/session_manager.py
import streamlit as st
import time
import os
import uuid
import logging
from typing import Dict, Any, Optional

from src.persona_manager import PersonaManager
from src.context.context_analyzer import ContextRelevanceAnalyzer, CodebaseScanner
from src.token_tracker import TokenUsageTracker
from src.config.settings import ChimeraSettings
from src.middleware.rate_limiter import RateLimiter
from src.utils.api_key_validator import validate_gemini_api_key_format

logger = logging.getLogger(__name__)

SESSION_TIMEOUT_SECONDS = 1800  # 30 minutes of inactivity

def update_activity_timestamp():
    st.session_state.last_activity_timestamp = time.time()
    logger.debug("Activity timestamp updated.")

def _initialize_session_state(app_config: ChimeraSettings, example_prompts: Dict[str, Any]):
    """Initializes or resets all session state variables to their default values."""
    # MODIFIED: Access total_budget directly from app_config (ChimeraSettings instance)
    MAX_TOKENS_LIMIT = app_config.total_budget
    CONTEXT_TOKEN_BUDGET_RATIO_FROM_CONFIG = app_config.context_token_budget_ratio
    DOMAIN_KEYWORDS = app_config.domain_keywords # NEW: Access domain_keywords from app_config

    defaults = {
        "initialized": True,
        "api_key_input": os.getenv("GEMINI_API_KEY", ""),
        "user_prompt_input": "",
        "max_tokens_budget_input": MAX_TOKENS_LIMIT,
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
        "codebase_context": {}, # This will now hold raw_file_contents for UI display/download
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
        "api_key_format_message": "Not provided",
        "api_key_functional_message": "Not tested",
        "current_debate_tokens_used": 0,
        "current_debate_cost_usd": 0.0,
        "last_activity_timestamp": time.time(),
        "structured_codebase_context": {}, # NEW: For structured analysis from CodebaseScanner
        "raw_file_contents": {}, # NEW: For raw file contents used by ContextRelevanceAnalyzer
        "context_ratio_user_modified": False,
    }

    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

    if "token_tracker" not in st.session_state:
        st.session_state.token_tracker = TokenUsageTracker(
            budget=st.session_state.max_tokens_budget_input
        )
    else:
        st.session_state.token_tracker.budget = st.session_state.max_tokens_budget_input
        st.session_state.token_tracker.reset()

    if "persona_manager" not in st.session_state:
        st.session_state.persona_manager = PersonaManager(
            DOMAIN_KEYWORDS, token_tracker=st.session_state.token_tracker
        )
        st.session_state.all_personas = st.session_state.persona_manager.all_personas
        st.session_state.persona_sets = st.session_state.persona_manager.persona_sets
        st.session_state.selected_persona_set = (
            st.session_state.persona_manager.available_domains[0]
            if st.session_state.persona_manager.available_domains
            else "General"
        )
        initial_framework_personas = (
            st.session_state.persona_manager.get_persona_sequence_for_framework(
                st.session_state.selected_persona_set
            )
        )
        st.session_state.personas = {
            name: st.session_state.persona_manager.all_personas.get(name)
            for name in initial_framework_personas
            if name in st.session_state.persona_manager.all_personas
        }

    if "context_analyzer" not in st.session_state:
        # NEW: Use cache dir from settings
        analyzer = ContextRelevanceAnalyzer(
            cache_dir=app_config.sentence_transformer_cache_dir,
            raw_file_contents=st.session_state.raw_file_contents,
        )
        if st.session_state.persona_manager.persona_router:
            analyzer.set_persona_router(st.session_state.persona_manager.persona_router)
        st.session_state.context_analyzer = analyzer

    if "session_rate_limiter_instance" not in st.session_state:
        st.session_state.session_rate_limiter_instance = RateLimiter(
            key_func=lambda: st.session_state._session_id, calls=10, period=60.0
        )

    # Initial API key validation
    if st.session_state.api_key_input:
        is_valid_format, message = validate_gemini_api_key_format(st.session_state.api_key_input)
        st.session_state.api_key_valid_format = is_valid_format
        st.session_state.api_key_format_message = message

    if (
        "user_prompt_input" not in st.session_state
        or not st.session_state.user_prompt_input
    ):
        if not example_prompts:
            # Fallback if no example prompts provided
            example_prompts = {
                "Coding & Implementation": {
                    "Implement Python API Endpoint": {
                        "prompt": "Implement a new FastAPI endpoint.",
                        "description": "Generate an API endpoint.",
                        "framework_hint": "Software Engineering",
                    }
                }
            }

        default_example_category = list(example_prompts.keys())[0]
        default_example_name = list(example_prompts[default_example_category].keys())[0]
        st.session_state.user_prompt_input = example_prompts[default_example_category][
            default_example_name
        ]["prompt"]
        st.session_state.selected_example_name = default_example_name
        st.session_state.selected_prompt_category = default_example_category
        st.session_state.active_example_framework_hint = example_prompts[
            default_example_category
        ][default_example_name].get("framework_hint")

def reset_app_state(app_config: ChimeraSettings, example_prompts: Dict[str, Any]):
    """Resets all session state variables to their default values."""
    _initialize_session_state(app_config, example_prompts)
    st.rerun()

def check_session_expiration(app_config: ChimeraSettings, example_prompts: Dict[str, Any]):
    """Checks for session expiration due to inactivity."""
    if "initialized" in st.session_state and st.session_state.initialized:
        if time.time() - st.session_state.last_activity_timestamp > SESSION_TIMEOUT_SECONDS:
            st.warning(
                "Your session has expired due to inactivity. Resetting the application."
            )
            _initialize_session_state(app_config, example_prompts)
            st.rerun()