"""
Sidebar UI Components for Project Chimera
Separated from app.py to reduce complexity
"""
from typing import Any

import streamlit as st


def create_sidebar() -> dict[str, Any]:
    """
    Create the sidebar UI components that were previously in app.py
    """
    # Placeholder for sidebar creation logic that would be moved from app.py
    # This is a simplified version to demonstrate the separation of concerns
    with st.sidebar:
        st.header("Settings")
        model_selection = st.selectbox("Select Model:", ["gemini-1.5-pro", "gemini-1.0-pro"])
        temperature = st.slider("Temperature:", 0.0, 1.0, 0.7)

        return {
            "model_selection": model_selection,
            "temperature": temperature
        }
