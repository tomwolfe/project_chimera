# tests/test_llm_provider.py
import pytest
from unittest.mock import MagicMock, patch
from src.llm_provider import GeminiProvider, LLMProviderError
from src.models import PersonaConfig
from google.genai.errors import APIError

@pytest.fixture
def mock_genai_client():
    """Mocks the genai.Client and its methods for testing."""
    mock_client = MagicMock()
    # Mock token counting
    mock_client.models.count_tokens.return_value.total_tokens = 10
    # Mock content generation response
    mock_client.models.generate_content.return_value.candidates = [
        MagicMock(content=MagicMock(parts=[MagicMock(text="Mocked LLM response")]))
    ]
    return mock_client

@pytest.fixture
def gemini_provider_instance(mock_genai_client):
    """Provides an instance of GeminiProvider, mocking the genai.Client."""
    # Patch the genai.Client constructor to return our mock client
    with patch("google.genai.Client", return_value=mock_genai_client):
        # Instantiate GeminiProvider with a mock client
        provider = GeminiProvider(api_key="test_api_key", model_name="gemini-2.5-flash-lite")
        # Ensure the provider's internal client is the mock
        provider.client = mock_genai_client
        return provider

def test_gemini_provider_generate_success(gemini_provider_instance, mock_genai_client):
    """Tests successful content generation from the GeminiProvider."""
    # Arrange
    prompt = "Test prompt for successful generation."
    system_prompt = "You are a helpful test assistant."
    persona_config = PersonaConfig(name="TestPersona", system_prompt=system_prompt, temperature=0.5, max_tokens=100)

    # Act
    response_text, input_tokens, output_tokens = gemini_provider_instance.generate(
        prompt=prompt, system_prompt=system_prompt, temperature=0.5, max_tokens=100, persona_config=persona_config
    )

    # Assert
    assert response_text == "Mocked LLM response"
    assert input_tokens == 10  # From mock_genai_client.models.count_tokens
    assert output_tokens == 10 # From mock_genai_client.models.count_tokens
    mock_genai_client.models.generate_content.assert_called_once()
    # Optionally, check the arguments passed to generate_content
    call_args, call_kwargs = mock_genai_client.models.generate_content.call_args
    assert call_args[0] == "gemini-2.5-flash-lite" # Model name
    assert call_kwargs["contents"] == prompt
    assert call_kwargs["config"].temperature == 0.5
    assert call_kwargs["config"].max_output_tokens == 100
    assert call_kwargs["config"].system_instruction == system_prompt

def test_gemini_provider_generate_api_error_non_retryable(gemini_provider_instance, mock_genai_client):
    """Tests handling of a non-retryable API error (e.g., invalid API key)."""
    # Arrange: Simulate an APIError with a non-retryable code
    mock_genai_client.models.generate_content.side_effect = APIError("Invalid API Key", code=401)

    # Act & Assert
    # Expect a specific LLMProviderError subclass for invalid API keys
    with pytest.raises(LLMProviderError, match="Invalid API Key"):
        gemini_provider_instance.generate(
            prompt="Test with invalid key", system_prompt="", temperature=0.5, max_tokens=100
        )
    mock_genai_client.models.generate_content.assert_called_once()

def test_gemini_provider_generate_api_error_retryable(gemini_provider_instance, mock_genai_client):
    """Tests handling of a retryable API error (e.g., rate limit)."""
    # Arrange: Simulate an APIError with a retryable code
    mock_genai_client.models.generate_content.side_effect = APIError("Rate limit exceeded", code=429)

    # Act & Assert
    # The generate method itself should retry internally. If it exhausts retries,
    # it should raise an LLMProviderError or similar.
    # For this test, we'll check if it raises an error after retries (if implemented)
    # or if the underlying APIError is propagated.
    # The current implementation of GeminiProvider's generate doesn't explicitly show retry logic
    # within the method itself, but the CircuitBreaker decorator handles retries.
    # So, we expect the CircuitBreakerError if the APIError is caught and fails enough times.
    # For simplicity, we'll test that the APIError is at least caught and potentially wrapped.
    
    # If the CircuitBreaker is active and configured for retries, this call might succeed after a delay.
    # For a unit test of the provider's error handling, we might mock the circuit breaker or
    # test the error propagation.
    
    # Let's assume the CircuitBreaker is applied to `generate` and will eventually raise CircuitBreakerError
    # if the underlying error persists. For this test, we'll check if the APIError is caught and wrapped.
    
    # The `handle_errors` decorator on `generate` should wrap APIError.
    with pytest.raises(LLMProviderError, match="Rate limit exceeded"):
        gemini_provider_instance.generate(
            prompt="Test rate limit", system_prompt="", temperature=0.5, max_tokens=100
        )
    mock_genai_client.models.generate_content.assert_called_once()

def test_gemini_provider_token_counting(gemini_provider_instance, mock_genai_client):
    """Tests the token counting functionality."""
    # Arrange
    text_to_count = "This is a test string to count tokens."
    
    # Act
    token_count = gemini_provider_instance.tokenizer.count_tokens(text_to_count)
    
    # Assert
    assert token_count == 10 # Based on mock_genai_client.models.count_tokens return value
    mock_genai_client.models.count_tokens.assert_called_once_with(
        model="gemini-2.5-flash-lite",
        contents=text_to_count
    )

def test_gemini_provider_token_counting_empty_string(gemini_provider_instance):
    """Tests token counting for an empty string."""
    token_count = gemini_provider_instance.tokenizer.count_tokens("")
    assert token_count == 0

def test_gemini_provider_token_counting_with_error_fallback(gemini_provider_instance, mock_genai_client):
    """Tests token counting fallback mechanism when API call fails."""
    # Arrange: Make count_tokens fail
    mock_genai_client.models.count_tokens.side_effect = Exception("Simulated API error for token count")
    
    text_to_count = "This is a test string to count tokens."
    
    # Act
    token_count = gemini_provider_instance.tokenizer.count_tokens(text_to_count)
    
    # Assert: Check if fallback approximation is used
    # The fallback logic uses len(text) / 3.5 for code-like content or 4 for text.
    # This text is more like general text.
    expected_approx_tokens = max(1, int(len(text_to_count) / 4))
    assert token_count == expected_approx_tokens
    mock_genai_client.models.count_tokens.assert_called_once()

def test_gemini_provider_calculate_usd_cost(gemini_provider_instance):
    """Tests the USD cost calculation."""
    # Arrange
    input_tokens = 1000
    output_tokens = 500
    
    # Expected cost based on gemini-1.5-flash pricing (0.08/1M input, 0.24/1M output)
    expected_input_cost = (1000 / 1000) * 0.08
    expected_output_cost = (500 / 1000) * 0.24
    expected_total_cost = expected_input_cost + expected_output_cost
    
    # Act
    actual_cost = gemini_provider_instance.calculate_usd_cost(input_tokens, output_tokens)
    
    # Assert
    assert actual_cost == pytest.approx(expected_total_cost)

def test_gemini_provider_calculate_usd_cost_zero_tokens(gemini_provider_instance):
    """Tests cost calculation with zero tokens."""
    actual_cost = gemini_provider_instance.calculate_usd_cost(0, 0)
    assert actual_cost == 0.0

def test_gemini_provider_calculate_usd_cost_unknown_model(gemini_provider_instance):
    """Tests cost calculation for an unknown model name."""
    # Temporarily change the model name to something not in the pricing map
    original_model_name = gemini_provider_instance.model_name
    gemini_provider_instance.model_name = "unknown-model-v1"
    
    actual_cost = gemini_provider_instance.calculate_usd_cost(1000, 500)
    
    # Restore original model name
    gemini_provider_instance.model_name = original_model_name
    
    assert actual_cost == 0.0 # Expect 0 cost if model is unknown

def test_gemini_provider_trim_text_no_truncation(gemini_provider_instance):
    """Tests text trimming when no truncation is needed."""
    text = "Short text."
    max_tokens = 100
    trimmed = gemini_provider_instance.trim_text_to_tokens(text, max_tokens)
    assert trimmed == text

def test_gemini_provider_trim_text_with_truncation(gemini_provider_instance):
    """Tests text trimming when truncation is needed."""
    # Create a text that will definitely exceed the token limit
    long_text = "This is a very long string that will surely exceed the token limit. " * 500
    max_tokens = 50 # A small token limit
    
    # Mock token counting to simulate exceeding the limit
    original_count_tokens = gemini_provider_instance.tokenizer.count_tokens
    
    # Simulate token counts: first call returns > max_tokens, subsequent calls return less
    call_count = 0
    def mock_count_tokens_for_trim(text):
        nonlocal call_count
        if call_count == 0:
            call_count += 1
            return max_tokens + 50 # Simulate exceeding limit
        else:
            return max_tokens - 10 # Simulate fitting after trimming
    
    gemini_provider_instance.tokenizer.count_tokens = mock_count_tokens_for_trim
    
    trimmed_text = gemini_provider_instance.trim_text_to_tokens(long_text, max_tokens)
    
    # Assert that the text was trimmed and the indicator was added
    assert len(trimmed_text) < len(long_text)
    assert trimmed_text.endswith("...")
    assert gemini_provider_instance.tokenizer.count_tokens(trimmed_text) < max_tokens
    
    # Restore original method
    gemini_provider_instance.tokenizer.count_tokens = original_count_tokens

def test_gemini_provider_trim_text_with_truncation_and_indicator(gemini_provider_instance):
    """Tests text trimming with a specific truncation indicator."""
    long_text = "This is a very long string that will surely exceed the token limit. " * 500
    max_tokens = 50
    truncation_indicator = "[TRUNCATED]"
    
    original_count_tokens = gemini_provider_instance.tokenizer.count_tokens
    call_count = 0
    def mock_count_tokens_for_trim_with_indicator(text):
        nonlocal call_count
        if call_count == 0:
            call_count += 1
            return max_tokens + 50
        else:
            return max_tokens - 10
    
    gemini_provider_instance.tokenizer.count_tokens = mock_count_tokens_for_trim_with_indicator
    
    trimmed_text = gemini_provider_instance.trim_text_to_tokens(long_text, max_tokens, truncation_indicator)
    
    assert len(trimmed_text) < len(long_text)
    assert trimmed_text.endswith(truncation_indicator)
    assert gemini_provider_instance.tokenizer.count_tokens(trimmed_text) < max_tokens
    
    gemini_provider_instance.tokenizer.count_tokens = original_count_tokens

def test_gemini_provider_trim_text_no_indicator_needed(gemini_provider_instance):
    """Tests trimming when the indicator itself would exceed the limit."""
    long_text = "Short text."
    max_tokens = 10
    truncation_indicator = "[TRUNCATED]"
    
    original_count_tokens = gemini_provider_instance.tokenizer.count_tokens
    def mock_count_tokens_for_trim_no_indicator(text):
        return 5 # Simulate text that fits within limit
    
    gemini_provider_instance.tokenizer.count_tokens = mock_count_tokens_for_trim_no_indicator
    
    trimmed_text = gemini_provider_instance.trim_text_to_tokens(long_text, max_tokens, truncation_indicator)
    
    assert trimmed_text == long_text # Should not add indicator if it fits
    assert gemini_provider_instance.tokenizer.count_tokens(trimmed_text) < max_tokens
    
    gemini_provider_instance.tokenizer.count_tokens = original_count_tokens

def test_gemini_provider_trim_text_max_tokens_zero_or_negative(gemini_provider_instance):
    """Tests trimming with zero or negative max_tokens."""
    text = "Some text."
    assert gemini_provider_instance.trim_text_to_tokens(text, 0) == ""
    assert gemini_provider_instance.trim_text_to_tokens(text, -10) == ""

def test_gemini_provider_trim_text_empty_text(gemini_provider_instance):
    """Tests trimming an empty string."""
    assert gemini_provider_instance.trim_text_to_tokens("", 100) == ""
    assert gemini_provider_instance.trim_text_to_tokens("", 0) == ""