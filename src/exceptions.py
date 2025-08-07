# src/exceptions.py
# Based on Iteration #3, Suggestion #1: Minimalist structure for 80/20 impact.

import datetime
from typing import Optional, Dict, Any

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

class LLMResponseValidationError(ChimeraError):
    """Raised when LLM response fails schema validation"""
    # Simplified from original. Specific parameters like invalid_response,
    # expected_schema, and details are omitted for minimal complexity.
    pass

# Note: The 'ContextAnalysisError' class from the original code is omitted here
# as the Iteration #3 suggestion for error handling focused on a minimal set
# of core exceptions.