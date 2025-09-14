# src/utils/session_manager.py
import streamlit as st
import time
import os
import uuid
import logging
from typing import Dict, Any, Optional, Callable  # Added Callable for type hint

from src.persona_manager import PersonaManager
from src.context.context_analyzer import ContextRelevanceAnalyzer, CodebaseScanner
from src.token_tracker import TokenUsageTracker
from src.config.settings import ChimeraSettings
from src.middleware.rate_limiter import RateLimiter
from src.utils.api_key_validator import validate_gemini_api_key_format

logger = logging.getLogger(__name__)

SESSION_TIMEOUT_SECONDS = 1800


def update_activity_timestamp():
    st.session_state.last_activity_timestamp = time.time()
    logger.debug("Activity timestamp updated.")


def _initialize_session_state(
    app_config: ChimeraSettings,
    example_prompts: Dict[str, Any],
    # NEW: Add these two parameters to the function signature
    get_context_relevance_analyzer_instance: Callable[
        [ChimeraSettings], ContextRelevanceAnalyzer
    ],
    get_codebase_scanner_instance: Callable[[], CodebaseScanner],
    get_summarizer_pipeline_instance: Callable[
        [], Any
    ],  # NEW: Add summarizer pipeline instance callable
):
    """Initializes or resets all session state variables to their default values."""
    MAX_TOKENS_LIMIT = app_config.total_budget
    CONTEXT_TOKEN_BUDGET_RATIO_FROM_CONFIG = app_config.context_token_budget_ratio
    DOMAIN_KEYWORDS = app_config.domain_keywords

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
        "api_key_format_message": "Not provided",
        "api_key_functional_message": "Not tested",
        "current_debate_tokens_used": 0,
        "current_debate_cost_usd": 0.0,
        "last_activity_timestamp": time.time(),
        "structured_codebase_context": {},
        "raw_file_contents": {},
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

    # --- MODIFIED: Use passed cached instances ---
    # The import from app is no longer needed here, as the functions are passed as arguments.
    # from app import get_context_relevance_analyzer_instance, get_codebase_scanner_instance

    if "context_analyzer" not in st.session_state:
        analyzer = get_context_relevance_analyzer_instance(
            _settings=app_config
        )  # Use passed function
        analyzer.raw_file_contents = (
            st.session_state.raw_file_contents
        )  # Update its raw_file_contents
        if st.session_state.persona_manager.persona_router:
            analyzer.set_persona_router(st.session_state.persona_manager.persona_router)
        st.session_state.context_analyzer = analyzer
    else:  # Ensure the cached instance is updated with current raw_file_contents
        st.session_state.context_analyzer.raw_file_contents = (
            st.session_state.raw_file_contents
        )

    if "codebase_scanner" not in st.session_state:
        st.session_state.codebase_scanner = (
            get_codebase_scanner_instance()
        )  # Use passed function

    # REMOVED: PromptOptimizer initialization from here. It's now handled in SocraticDebate.
    # --- END MODIFIED ---

    if "session_rate_limiter_instance" not in st.session_state:
        st.session_state.session_rate_limiter_instance = RateLimiter(
            key_func=lambda: st.session_state._session_id, calls=10, period=60.0
        )

    if st.session_state.api_key_input:
        is_valid_format, message = validate_gemini_api_key_format(
            st.session_state.api_key_input
        )
        st.session_state.api_key_valid_format = is_valid_format
        st.session_state.api_key_format_message = message


def reset_app_state(app_config: ChimeraSettings, example_prompts: Dict[str, Any]):
    """Resets all session state variables to their default values."""
    # MODIFIED: Pass the cached functions to _initialize_session_state
    # The import from app is no longer needed here, as the functions are passed as arguments.
    from app import (
        get_context_relevance_analyzer_instance,
        get_codebase_scanner_instance,
        get_summarizer_pipeline_instance, # NEW: Import the summarizer pipeline instance
    )

    _initialize_session_state(
        app_config=app_config,
        example_prompts=example_prompts,
        get_context_relevance_analyzer_instance=get_context_relevance_analyzer_instance,
        get_codebase_scanner_instance=get_codebase_scanner_instance,
        get_summarizer_pipeline_instance=get_summarizer_pipeline_instance, # NEW: Pass the summarizer pipeline instance
    )
    st.rerun()


def check_session_expiration(
    app_config: ChimeraSettings, example_prompts: Dict[str, Any]
):
    """Checks for session expiration due to inactivity."""
    if "initialized" in st.session_state and st.session_state.initialized:
        if (
            time.time() - st.session_state.last_activity_timestamp
            > SESSION_TIMEOUT_SECONDS
        ):
            st.warning(
                "Your session has expired due to inactivity. Resetting the application."
            )
            # MODIFIED: Pass the cached functions to _initialize_session_state
            # The import from app is no longer needed here, as the functions are passed as arguments.
            from app import (
                get_context_relevance_analyzer_instance,
                get_codebase_scanner_instance,
                get_summarizer_pipeline_instance, # NEW: Import the summarizer pipeline instance
            )

            _initialize_session_state(
                app_config=app_config,
                example_prompts=example_prompts,
                get_context_relevance_analyzer_instance=get_context_relevance_analyzer_instance,
                get_codebase_scanner_instance=get_codebase_scanner_instance,
                get_summarizer_pipeline_instance=get_summarizer_pipeline_instance, # NEW: Pass the summarizer pipeline instance
            )
            st.rerun()