"""
Unified exception hierarchy for Project Chimera with standardized error handling patterns.
"""

import datetime
from typing import Optional, Dict, Any

class ChimeraError(Exception):
    """Base exception for all Project Chimera errors."""
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None, 
                 severity: str = "ERROR", recoverable: bool = False):
        super().__init__(message)
        self.details = details or {}
        self.severity = severity  # ERROR, WARNING, INFO
        self.recoverable = recoverable
        self.timestamp = datetime.datetime.now()

class ContextAnalysisError(ChimeraError):
    """Base exception for context analysis failures."""
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(
            message, 
            details=details,
            severity="ERROR",
            recoverable=True
        )

class LLMResponseValidationError(ChimeraError):
    """Raised when LLM response fails schema validation."""
    def __init__(self, 
                 message: str, 
                 invalid_response: Any,
                 expected_schema: str,
                 details: Optional[Dict[str, Any]] = None):
        error_details = {
            "invalid_response": str(invalid_response)[:500] + "..." if invalid_response and len(str(invalid_response)) > 500 else invalid_response,
            "expected_schema": expected_schema,
            **(details or {})
        }
        super().__init__(
            f"LLM response validation failed: {message}",
            details=error_details,
            severity="ERROR",
            recoverable=True
        )

class TokenBudgetExceededError(ChimeraError):
    """Raised when token usage exceeds budget."""
    def __init__(self, 
                 current_tokens: int, 
                 budget: int,
                 details: Optional[Dict[str, Any]] = None):
        error_details = {
            "current_tokens": current_tokens,
            "budget": budget,
            **(details or {})
        }
        super().__init__(
            f"Token budget exceeded: {current_tokens}/{budget} tokens used",
            details=error_details,
            severity="WARNING",
            recoverable=True
        )
