# src/exceptions.py
"""Custom exceptions for the Project Chimera application."""

import datetime
import traceback  # Import traceback
from typing import Any, Optional


class ChimeraError(Exception):
    """Base exception for all Chimera errors with standardized structure."""

    # FIX: Added error_code parameter and explicitly stored message for __str__ and to_dict
    def __init__(
        self,
        message: str,
        error_code: str = "CHIMERA_ERROR",
        details: Optional[dict] = None,
        original_exception: Optional[Exception] = None,
    ):
        super().__init__(message)
        self.message = message  # Store message explicitly for __str__ and to_dict
        self.error_code = error_code
        self.details = details or {}
        self.timestamp = datetime.datetime.now()
        self.original_exception = original_exception
        # Capture stack trace only if an original_exception is provided, otherwise it's the current frame.
        # traceback.format_exc() captures the traceback of the *current* exception being handled.
        # If original_exception is None, it means this ChimeraError is the primary error,
        # so we capture its own creation stack.
        self.stack_trace = (
            traceback.format_exc() if original_exception else traceback.format_stack()
        )  # Use format_stack for current frame

    # FIX: Modified to_dict to include error_code and use self.message
    def to_dict(self) -> dict[str, Any]:
        """Convert exception to structured dictionary for logging/reporting."""
        return {
            "error_type": self.__class__.__name__,
            "error_code": self.error_code,
            "message": self.message,  # Use self.message as per fix
            "details": self.details,
            "timestamp": self.timestamp.isoformat(),
            "original_exception_type": type(self.original_exception).__name__
            if self.original_exception
            else None,
            "original_exception_message": str(self.original_exception)
            if self.original_exception
            else None,
            "stack_trace": self.stack_trace,
        }

    # FIX: Added __str__ method for robust string representation
    def __str__(self):
        """String representation of the error."""
        return f"{self.error_code}: {self.message}"


class CodebaseAccessError(ChimeraError):
    """Exception raised when codebase access is not available."""

    pass


class LLMProviderError(ChimeraError):
    """Base exception for LLM provider errors."""

    def __init__(
        self,
        message: str,
        provider_error_code: Any = None,
        details: Optional[dict] = None,
        original_exception: Optional[Exception] = None,
        error_code: str = "LLM_PROVIDER_ERROR",
    ):
        full_details = (details or {}).copy()
        if provider_error_code is not None:
            full_details["provider_error_code"] = provider_error_code
            # If provider_error_code is more specific, use it as the primary error_code
            # Otherwise, use the default "LLM_PROVIDER_ERROR"
            final_error_code = (
                provider_error_code
                if isinstance(provider_error_code, str)
                else error_code
            )
        else:
            final_error_code = error_code

        super().__init__(
            message,
            error_code=final_error_code,
            details=full_details,
            original_exception=original_exception,
        )


# NEW: Specific exceptions for LLM API request and response errors
class LLMProviderRequestError(LLMProviderError):
    """Exception for LLM API request errors (e.g., 4xx, 5xx HTTP errors)."""

    pass


class LLMProviderResponseError(LLMProviderError):
    """Exception for LLM API response parsing errors (e.g., JSONDecodeError)."""

    pass


class GeminiAPIError(LLMProviderError):
    """Specific exception for Gemini API errors."""

    def __init__(
        self,
        message: str,
        code: int = None,
        response_details: Any = None,
        original_exception: Optional[Exception] = None,
    ):
        super().__init__(
            message,
            provider_error_code=code,
            details={"response_details": response_details},
            original_exception=original_exception,
        )


class LLMUnexpectedError(LLMProviderError):
    """Specific exception for unexpected LLM errors."""

    def __init__(
        self,
        message: str,
        details: Optional[dict] = None,
        original_exception: Optional[Exception] = None,
    ):
        super().__init__(
            message,
            error_code="LLM_UNEXPECTED_ERROR",
            details=details,
            original_exception=original_exception,
        )


class ValidationPhaseError(ChimeraError):
    """Base for errors occurring during response validation."""

    def __init__(
        self,
        message: str,
        error_code: str = "VALIDATION_PHASE_ERROR",
        invalid_response: Any = None,
        expected_schema: str = None,
        details: Optional[dict] = None,
        original_exception: Optional[Exception] = None,
    ):
        full_details = (details or {}).copy()
        full_details.update(
            {"invalid_response": invalid_response, "expected_schema": expected_schema}
        )
        super().__init__(
            message,
            error_code=error_code,
            details=full_details,
            original_exception=original_exception,
        )


class SchemaValidationError(ValidationPhaseError):
    """Specific error when response fails schema validation."""

    def __init__(
        self,
        error_type: str,
        field_path: str,
        invalid_value: Any = None,
        details: Optional[dict] = None,
        original_exception: Optional[Exception] = None,
    ):
        message = f"Schema validation failed: {error_type} at '{field_path}'"
        full_details = (details or {}).copy()
        full_details.update(
            {
                "error_type": error_type,
                "field_path": field_path,
                "invalid_value": invalid_value,
            }
        )
        super().__init__(
            message,
            error_code="SCHEMA_VALIDATION_ERROR",
            details=full_details,
            original_exception=original_exception,
        )


class TokenBudgetExceededError(ChimeraError):
    """Raised when token usage exceeds budget."""

    def __init__(
        self,
        current_tokens: int,
        budget: int,
        details: Optional[dict[str, Any]] = None,
        original_exception: Optional[Exception] = None,
    ):
        details = details or {}
        error_details = {
            "current_tokens": current_tokens,
            "budget": budget,
            "phase": details.get("phase", "N/A"),
            "step_name": details.get("step_name", "N/A"),
            "tokens_needed": details.get("tokens_needed", "N/A"),
            **details,
        }
        message = f"Token budget exceeded: {current_tokens}/{budget} tokens used. Phase: {error_details['phase']}, Step: {error_details['step_name']}"
        super().__init__(
            message,
            error_code="TOKEN_BUDGET_EXCEEDED",
            details=error_details,
            original_exception=original_exception,
        )


class CircuitBreakerError(ChimeraError):
    """Exception raised when the circuit breaker is open and prevents execution."""

    def __init__(
        self,
        message: str,
        details: Optional[dict] = None,
        original_exception: Optional[Exception] = None,
    ):
        super().__init__(
            message,
            error_code="CIRCUIT_BREAKER_OPEN",
            details=details,
            original_exception=original_exception,
        )


class LLMResponseValidationError(ValidationPhaseError):
    """Raised when LLM response fails validation, with code-specific guidance."""

    def __init__(
        self,
        message: str,
        invalid_response: Any = None,
        expected_schema: str = None,
        details: Optional[dict] = None,
        original_exception: Optional[Exception] = None,
    ):
        super().__init__(
            message,
            invalid_response=invalid_response,
            expected_schema=expected_schema,
            error_code="LLM_RESPONSE_VALIDATION_ERROR",
            details=details,
            original_exception=original_exception,
        )
