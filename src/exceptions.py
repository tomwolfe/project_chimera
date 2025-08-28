# src/exceptions.py
import datetime
from typing import Optional, Dict, Any
import traceback # Import traceback

class ChimeraError(Exception):
    """Base exception for all Chimera errors with standardized structure."""
    def __init__(self, message: str, details: Optional[dict] = None, original_exception: Optional[Exception] = None):
        super().__init__(message)
        self.details = details or {}
        self.timestamp = datetime.datetime.now()
        self.original_exception = original_exception # Store original exception
        self.stack_trace = traceback.format_exc() if original_exception else None # Capture stack trace if original exception exists

    def to_dict(self) -> Dict[str, Any]:
        """Convert exception to structured dictionary for logging/reporting."""
        return {
            "message": str(self), # CHANGED THIS KEY BACK TO "message" for consistency
            "details": self.details,
            "timestamp": self.timestamp.isoformat(),
            "original_exception_type": type(self.original_exception).__name__ if self.original_exception else None,
            "original_exception_message": str(self.original_exception) if self.original_exception else None,
            "stack_trace": self.stack_trace,
            "type": self.__class__.__name__
        }

class LLMProviderError(ChimeraError):
    """Base exception for LLM provider errors."""
    def __init__(self, message: str, provider_error_code: Any = None, details: Optional[dict] = None, original_exception: Optional[Exception] = None):
        full_details = (details or {}).copy()
        full_details["provider_error_code"] = provider_error_code
        super().__init__(message, details=full_details, original_exception=original_exception)

class GeminiAPIError(LLMProviderError):
    """Specific exception for Gemini API errors."""
    def __init__(self, message: str, code: int = None, response_details: Any = None, original_exception: Optional[Exception] = None):
        super().__init__(message, provider_error_code=code, details={"response_details": response_details}, original_exception=original_exception)

class LLMUnexpectedError(LLMProviderError):
    """Specific exception for unexpected LLM errors."""
    pass

class ValidationPhaseError(ChimeraError):
    """Base for errors occurring during response validation."""
    def __init__(self, message: str, invalid_response: Any = None, 
                 expected_schema: str = None, details: Optional[dict] = None, original_exception: Optional[Exception] = None):
        full_details = (details or {}).copy()
        full_details.update({
            "invalid_response": invalid_response,
            "expected_schema": expected_schema
        })
        super().__init__(message, full_details, original_exception=original_exception)

class SchemaValidationError(ValidationPhaseError):
    """Specific error when response fails schema validation."""
    def __init__(self, error_type: str, field_path: str, 
                 invalid_value: Any = None, details: Optional[dict] = None, original_exception: Optional[Exception] = None):
        message = f"Schema validation failed: {error_type} at '{field_path}'"
        full_details = (details or {}).copy()
        full_details.update({
            "error_type": error_type,
            "field_path": field_path,
            "invalid_value": invalid_value
        })
        super().__init__(message, details=full_details, original_exception=original_exception)

class TokenBudgetExceededError(ChimeraError):
    """Raised when token usage exceeds budget"""
    def __init__(self, current_tokens: int, budget: int, details: Optional[Dict[str, Any]] = None, original_exception: Optional[Exception] = None):
        error_details = {
            "current_tokens": current_tokens,
            "budget": budget,
            "phase": details.get("phase", "N/A"),
            "step_name": details.get("step_name", "N/A"),
            "tokens_needed": details.get("tokens_needed", "N/A"),
            **(details or {})
        }
        message = f"Token budget exceeded: {current_tokens}/{budget} tokens used. Phase: {error_details['phase']}, Step: {error_details['step_name']}"
        super().__init__(message, details=error_details, original_exception=original_exception)

class CircuitBreakerError(ChimeraError):
    """Exception raised when the circuit breaker is open and prevents execution."""
    def __init__(self, message: str, details: Optional[dict] = None, original_exception: Optional[Exception] = None):
        super().__init__(message, details, original_exception=original_exception)

class LLMResponseValidationError(ValidationPhaseError):
    """Raised when LLM response fails validation, with code-specific guidance."""
    pass