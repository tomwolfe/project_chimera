# src/exceptions.py
# Based on Iteration #3, Suggestion #1: Minimalist structure for 80/20 impact.

import datetime
from typing import Optional, Dict, Any
import streamlit as st # Import streamlit to access session state

# --- MODIFIED CLASS START ---
class ChimeraError(Exception):
    """Base exception for all Chimera errors with standardized structure."""
    def __init__(self, message: str, details: Optional[dict] = None):
        super().__init__(message)
        self.details = details or {}
        self.timestamp = datetime.datetime.now()

class LLMProviderError(ChimeraError):
    """Base exception for LLM provider errors."""
    def __init__(self, message: str, provider_error_code: Any = None, details: Optional[dict] = None):
        full_details = (details or {}).copy()
        full_details["provider_error_code"] = provider_error_code
        super().__init__(message, details=full_details)

class GeminiAPIError(LLMProviderError):
    """Specific exception for Gemini API errors."""
    def __init__(self, message: str, code: int = None, response_details: Any = None):
        super().__init__(message, provider_error_code=code, details={"response_details": response_details})

class LLMUnexpectedError(LLMProviderError):
    """Specific exception for unexpected LLM errors."""
    pass

class ValidationPhaseError(ChimeraError):
    """Base for errors occurring during response validation."""
    def __init__(self, message: str, invalid_response: Any = None, 
                 expected_schema: str = None, details: Optional[dict] = None):
        full_details = (details or {}).copy()
        full_details.update({
            "invalid_response": invalid_response,
            "expected_schema": expected_schema
        })
        super().__init__(message, full_details)

class SchemaValidationError(ValidationPhaseError):
    """Specific error when response fails schema validation."""
    def __init__(self, error_type: str, field_path: str, 
                 invalid_value: Any = None, details: Optional[dict] = None):
        message = f"Schema validation failed: {error_type} at '{field_path}'"
        full_details = (details or {}).copy()
        full_details.update({
            "error_type": error_type,
            "field_path": field_path,
            "invalid_value": invalid_value
        })
        super().__init__(message, details=full_details)

# The TokenBudgetExceededError is kept simple as per previous iterations,
# but could be enhanced to inherit from ChimeraError with details if needed.
class TokenBudgetExceededError(ChimeraError):
    """Raised when token usage exceeds budget"""
    def __init__(self, current_tokens: int, budget: int, details: Optional[Dict[str, Any]] = None):
        error_details = {
            "current_tokens": current_tokens,
            "budget": budget,
            "phase": details.get("phase", "N/A"), # Added phase
            "step_name": details.get("step_name", "N/A"), # Added step_name
            "tokens_needed": details.get("tokens_needed", "N/A"), # Added tokens_needed
            **(details or {})
        }
        message = f"Token budget exceeded: {current_tokens}/{budget} tokens used. Phase: {error_details['phase']}, Step: {error_details['step_name']}"
        super().__init__(message, details=error_details)

# --- ADDED CIRCUITBREAKERERROR START ---
class CircuitBreakerError(ChimeraError):
    """Exception raised when the circuit breaker is open and prevents execution."""
    def __init__(self, message: str, details: Optional[dict] = None):
        super().__init__(message, details)
# --- ADDED CIRCUITBREAKERERROR END ---

# LLMResponseValidationError is now superseded by SchemaValidationError for schema issues,
# but might be kept for other non-schema-related LLM errors if they arise.
class LLMResponseValidationError(ValidationPhaseError):
    """Raised when LLM response fails validation, with code-specific guidance."""
    # This class now inherits from ValidationPhaseError for better structure.
    # The specific logic for adding code-focused guidance is now handled within
    # core.py's _analyze_validation_error, which is more context-aware.
    # This class primarily serves as a structured wrapper for validation errors.
    pass
# --- MODIFIED CLASS END ---