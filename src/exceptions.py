# src/exceptions.py
# Based on Iteration #3, Suggestion #1: Minimalist structure for 80/20 impact.

import datetime
from typing import Optional, Dict, Any
import streamlit as st # Import streamlit to access session state

class ChimeraError(Exception):
    """Base exception for all Chimera errors"""
    # This simplified version focuses on the core inheritance.
    # More detailed attributes (like details, severity, timestamp) are omitted
    # for minimal complexity as per the 80/20 principle.
    pass

class TokenBudgetExceededError(ChimeraError):
    """Raised when token usage exceeds budget"""
    # Simplified from original to just inherit from ChimeraError.
    # Specific parameters like current_tokens, budget, details are omitted
    # in this minimal version, assuming the calling code will handle context.
    pass

# --- MODIFIED CLASS START ---
class LLMResponseValidationError(ChimeraError):
    """Raised when LLM response fails schema validation, with code-specific guidance"""
    def __init__(self, message: str, invalid_response: Any = None, 
                 expected_schema: str = None, details: dict = None):
        
        # Add code-specific guidance when analyzing code
        # Check if Streamlit context is available and if the prompt context suggests code analysis
        prompt_context = ""
        if 'st' in globals(): # Check if Streamlit context is available
            prompt_context = st.session_state.get('initial_prompt', '')
        
        is_code_analysis = False
        if prompt_context:
            prompt_lower = prompt_context.lower()
            # Keywords indicating code analysis or self-analysis of Chimera
            if "code" in prompt_lower or "analyze" in prompt_lower or "refactor" in prompt_lower or "chimera" in prompt_lower or "self-analysis" in prompt_lower:
                is_code_analysis = True
        
        if is_code_analysis:
            message += "\n\nWhen analyzing codebases, ensure responses include:\n" \
                      "1) Complete file content (not diffs)\n" \
                      "2) PEP8-compliant code\n" \
                      "3) Structural elements (classes/functions) preserved\n" \
                      "4) Key imports maintained"
        
        super().__init__(message)
        self.invalid_response = invalid_response
        self.expected_schema = expected_schema
        self.details = details
# --- MODIFIED CLASS END ---

# Note: The 'ContextAnalysisError' class from the original code is omitted here
# as the Iteration #3 suggestion for error handling focused on a minimal set
# of core exceptions.