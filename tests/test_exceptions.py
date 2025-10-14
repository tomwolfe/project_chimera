"""Tests for the exceptions module."""

import datetime

from src.exceptions import (
    ChimeraError,
    CircuitBreakerError,
    CodebaseAccessError,
    GeminiAPIError,
    LLMProviderError,
    LLMProviderRequestError,
    LLMProviderResponseError,
    LLMResponseValidationError,
    LLMUnexpectedError,
    SchemaValidationError,
    TokenBudgetExceededError,
    ValidationPhaseError,
)


class TestChimeraError:
    """Test suite for ChimeraError base exception."""

    def test_chimera_error_basic_initialization(self):
        """Test basic initialization of ChimeraError."""
        error = ChimeraError("Test error message")
        assert str(error) == "CHIMERA_ERROR: Test error message"
        assert error.message == "Test error message"
        assert error.error_code == "CHIMERA_ERROR"
        assert isinstance(error.timestamp, datetime.datetime)
        assert error.details == {}

    def test_chimera_error_with_error_code(self):
        """Test initialization with custom error code."""
        error = ChimeraError("Test message", error_code="CUSTOM_CODE")
        assert error.error_code == "CUSTOM_CODE"

    def test_chimera_error_with_details(self):
        """Test initialization with details."""
        details = {"key": "value"}
        error = ChimeraError("Test message", details=details)
        assert error.details == details

    def test_chimera_error_with_original_exception(self):
        """Test initialization with original exception."""
        original = ValueError("original error")
        error = ChimeraError("Test message", original_exception=original)
        assert error.original_exception == original

    def test_chimera_error_to_dict(self):
        """Test conversion to dictionary."""
        original = ValueError("original error")
        error = ChimeraError("Test message", original_exception=original)
        error_dict = error.to_dict()

        assert error_dict["error_type"] == "ChimeraError"
        assert error_dict["error_code"] == "CHIMERA_ERROR"
        assert error_dict["message"] == "Test message"
        assert error_dict["original_exception_type"] == "ValueError"
        assert error_dict["original_exception_message"] == "original error"

    def test_chimera_error_str(self):
        """Test string representation."""
        error = ChimeraError("Test message", error_code="TEST_CODE")
        assert str(error) == "TEST_CODE: Test message"


class TestCodebaseAccessError:
    """Test suite for CodebaseAccessError."""

    def test_codebase_access_error(self):
        """Test CodebaseAccessError initialization."""
        error = CodebaseAccessError("Codebase not accessible")
        assert isinstance(error, ChimeraError)
        assert str(error) == "CHIMERA_ERROR: Codebase not accessible"


class TestLLMProviderError:
    """Test suite for LLMProviderError."""

    def test_llm_provider_error_basic(self):
        """Test basic LLMProviderError initialization."""
        error = LLMProviderError("LLM error")
        assert str(error) == "LLM_PROVIDER_ERROR: LLM error"
        assert error.error_code == "LLM_PROVIDER_ERROR"

    def test_llm_provider_error_with_provider_code(self):
        """Test LLMProviderError with provider error code."""
        error = LLMProviderError("LLM error", provider_error_code="GEMINI_400")
        assert error.error_code == "GEMINI_400"  # Uses provider error code if string

    def test_llm_provider_error_with_numeric_provider_code(self):
        """Test LLMProviderError with numeric provider error code."""
        error = LLMProviderError("LLM error", provider_error_code=400)
        assert (
            error.error_code == "LLM_PROVIDER_ERROR"
        )  # Falls back to default when not string

    def test_llm_provider_error_with_details(self):
        """Test LLMProviderError with additional details."""
        error = LLMProviderError(
            "LLM error", provider_error_code=400, details={"status": 400}
        )
        assert error.details["provider_error_code"] == 400
        assert error.details["status"] == 400


class TestSpecificLLMErrorTypes:
    """Test suite for specific LLM error types."""

    def test_llm_provider_request_error(self):
        """Test LLMProviderRequestError."""
        error = LLMProviderRequestError("Request failed")
        assert isinstance(error, LLMProviderError)
        assert str(error) == "LLM_PROVIDER_ERROR: Request failed"

    def test_llm_provider_response_error(self):
        """Test LLMProviderResponseError."""
        error = LLMProviderResponseError("Response parsing failed")
        assert isinstance(error, LLMProviderError)
        assert str(error) == "LLM_PROVIDER_ERROR: Response parsing failed"

    def test_gemini_api_error(self):
        """Test GeminiAPIError."""
        error = GeminiAPIError(
            "API error", code=403, response_details={"error": "Forbidden"}
        )
        assert (
            error.error_code == "LLM_PROVIDER_ERROR"
        )  # Falls back to default when code is numeric
        assert error.details["response_details"] == {"error": "Forbidden"}

    def test_llm_unexpected_error(self):
        """Test LLMUnexpectedError."""
        error = LLMUnexpectedError("Unexpected error", details={"debug_info": "info"})
        assert error.error_code == "LLM_UNEXPECTED_ERROR"
        assert error.details["debug_info"] == "info"


class TestValidationPhaseError:
    """Test suite for ValidationPhaseError and subclasses."""

    def test_validation_phase_error(self):
        """Test ValidationPhaseError basic initialization."""
        error = ValidationPhaseError(
            "Validation failed",
            invalid_response={"bad": "data"},
            expected_schema="GoodSchema",
        )
        assert str(error) == "VALIDATION_PHASE_ERROR: Validation failed"
        assert error.details["invalid_response"] == {"bad": "data"}
        assert error.details["expected_schema"] == "GoodSchema"

    def test_schema_validation_error(self):
        """Test SchemaValidationError."""
        error = SchemaValidationError(
            error_type="required_field_missing",
            field_path="user.name",
            invalid_value=None,
        )
        assert (
            str(error)
            == "SCHEMA_VALIDATION_ERROR: Schema validation failed: required_field_missing at 'user.name'"
        )
        assert error.details["error_type"] == "required_field_missing"
        assert error.details["field_path"] == "user.name"
        assert error.details["invalid_value"] is None

    def test_schema_validation_error_with_details(self):
        """Test SchemaValidationError with additional details."""
        error = SchemaValidationError(
            error_type="type_mismatch",
            field_path="user.age",
            invalid_value="not_a_number",
            details={"suggestion": "Use an integer value"},
        )
        assert error.details["suggestion"] == "Use an integer value"


class TestTokenBudgetExceededError:
    """Test suite for TokenBudgetExceededError."""

    def test_token_budget_exceeded_error_basic(self):
        """Test basic TokenBudgetExceededError initialization."""
        error = TokenBudgetExceededError(current_tokens=1500, budget=1000)
        assert (
            str(error)
            == "TOKEN_BUDGET_EXCEEDED: Token budget exceeded: 1500/1000 tokens used. Phase: N/A, Step: N/A"
        )
        assert error.error_code == "TOKEN_BUDGET_EXCEEDED"
        assert error.details["current_tokens"] == 1500
        assert error.details["budget"] == 1000

    def test_token_budget_exceeded_error_with_phase_details(self):
        """Test TokenBudgetExceededError with phase details."""
        details = {
            "phase": "debate",
            "step_name": "Skeptical_Generator",
            "tokens_needed": 200,
        }
        error = TokenBudgetExceededError(
            current_tokens=1500, budget=1000, details=details
        )
        assert error.details["phase"] == "debate"
        assert error.details["step_name"] == "Skeptical_Generator"
        assert error.details["tokens_needed"] == 200


class TestCircuitBreakerError:
    """Test suite for CircuitBreakerError."""

    def test_circuit_breaker_error(self):
        """Test CircuitBreakerError initialization."""
        error = CircuitBreakerError("Circuit breaker is open")
        assert str(error) == "CIRCUIT_BREAKER_OPEN: Circuit breaker is open"
        assert error.error_code == "CIRCUIT_BREAKER_OPEN"


class TestLLMResponseValidationError:
    """Test suite for LLMResponseValidationError."""

    def test_llm_response_validation_error(self):
        """Test LLMResponseValidationError initialization."""
        error = LLMResponseValidationError(
            "Response validation failed",
            invalid_response={"bad": "response"},
            expected_schema="ExpectedSchema",
        )
        assert error.error_code == "LLM_RESPONSE_VALIDATION_ERROR"
        assert error.details["invalid_response"] == {"bad": "response"}
        assert error.details["expected_schema"] == "ExpectedSchema"
