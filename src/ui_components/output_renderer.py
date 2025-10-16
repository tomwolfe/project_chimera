"""
Output Renderer for Project Chimera
Separated from app.py to reduce complexity
"""
from typing import Any

import streamlit as st


def render_output(output_data: dict[str, Any]) -> None:
    """
    Render the output that was previously handled directly in app.py
    """
    # Placeholder for output rendering logic that would be moved from app.py
    # This is a simplified version to demonstrate the separation of concerns
    if output_data:
        st.subheader("Response:")
        st.write(output_data.get("response", "No response generated"))
        st.json(output_data.get("metadata", {}))
