"""
UI Components for Project Chimera
Separated from app.py to reduce complexity
"""
from typing import Any

import streamlit as st


def create_main_ui() -> dict[str, Any]:
    """
    Create the main UI components that were previously in app.py
    """
    # Placeholder for UI creation logic that would be moved from app.py
    # This is a simplified version to demonstrate the separation of concerns
    st.title("Project Chimera")
    user_input = st.text_area("Enter your prompt:", height=150)
    persona_selection = st.selectbox("Select Persona:", ["Creative", "Analytical", "Technical"])

    return {
        "user_input": user_input,
        "persona_selection": persona_selection,
        "submit_button": st.button("Submit")
    }
