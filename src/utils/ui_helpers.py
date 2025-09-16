import streamlit as st
import logging
from typing import Dict, Any
import os  # NEW: Import os
import signal  # NEW: Import signal
import sys  # NEW: Import sys

from src.utils.api_key_validator import (
    validate_gemini_api_key_format,
    test_gemini_api_key_functional,
)
from src.utils.session_manager import update_activity_timestamp

logger = logging.getLogger(__name__)


def on_api_key_change():
    """Callback to validate API key format and update activity timestamp."""
    if "api_key_input" not in st.session_state:
        st.session_state.api_key_input = ""
    if "api_key_valid_format" not in st.session_state:
        st.session_state.api_key_valid_format = False
    if "api_key_format_message" not in st.session_state:
        st.session_state.api_key_format_message = "Not provided"
    if "api_key_functional_message" not in st.session_state:
        st.session_state.api_key_functional_message = "Not tested"

    api_key_value = st.session_state.api_key_input.strip()
    is_valid_format, message = validate_gemini_api_key_format(api_key_value)
    st.session_state.api_key_valid_format = is_valid_format
    st.session_state.api_key_format_message = message
    update_activity_timestamp()
    if "token_tracker" in st.session_state:
        st.session_state.token_tracker.reset()
        st.session_state.current_debate_tokens_used = 0
        st.session_state.current_debate_cost_usd = 0.0


def display_key_status():
    """Display detailed API key status with appropriate icons"""
    if not st.session_state.api_key_input:
        st.caption("🔑 Key status: Not provided")
        return

    if not st.session_state.api_key_valid_format:
        st.caption(f"❌ Key status: {st.session_state.api_key_format_message}")
        return

    if not st.session_state.api_key_functional:
        st.caption(f"⚠️ Key status: {st.session_state.api_key_functional_message}")
        return

    st.caption("✅ Key status: Valid and functional")


def test_api_key():
    """Test the API key functionality with proper error handling"""
    api_key = st.session_state.api_key_input.strip()
    if not api_key:
        st.session_state.api_key_functional = False
        st.session_state.api_key_functional_message = "No API key provided to test"
        return

    is_functional, message = test_gemini_api_key_functional(api_key)
    st.session_state.api_key_functional = is_functional
    st.session_state.api_key_functional_message = message


# NEW FUNCTION: For proper process termination
def shutdown_streamlit():
    """Properly shutdown Streamlit application."""
    logger.info("Shutting down Streamlit application...")
    print("Shutting down Streamlit application...")
    # Force exit to ensure all processes terminate
    # This is a strong measure, but necessary for Streamlit's multi-process nature
    os._exit(0)
