# tests/test_llm_provider.py

import pytest
import os
from unittest.mock import MagicMock, patch
from src.llm_provider import GeminiProvider, LLMProviderError, CircuitBreakerError
from src.tokenizers import GeminiTokenizer
from google.genai.errors import APIError
import google.genai as genai

# Fixture for a mock Gemini client
@pytest.fixture
def mock_genai_client():
    mock_client = MagicMock()
    mock_client.models.count_tokens.return_value.total_tokens = 10
    mock_client.models.generate_content.return_value.candidates = [
        MagicMock(content=MagicMock(parts=[MagicMock(text="Mocked response")]))
    ]
    return mock_client

# Fixture for a GeminiProvider instance with a mock client
@pytest.fixture
def gemini_provider(mock_genai_client):
    # Patch genai.Client to return our mock_genai_client
    with patch('google.genai.Client', return_value=mock_genai_client):
        return GeminiProvider(api_key="test_api_key", model_name="gemini-2.5-flash-lite")

def test_gemini_provider_initialization_success(mock_genai_client):
    """Test successful initialization of GeminiProvider."""
    with patch('google.genai.Client', return_value=mock_genai_client):
        provider = GeminiProvider(api_key="test_api_key")
        assert provider.model_name == "gemini-2.5-flash-lite"
        assert isinstance(provider.tokenizer, GeminiTokenizer)

def test_gemini_provider_initialization_invalid_api_key():
    """Test initialization failure with an invalid API key."""
    with patch('google.genai.Client', side_effect=APIError("API key not valid")):
        with pytest.raises(LLMProviderError, match="Invalid API Key"):
            GeminiProvider(api_key="invalid_key")

def test_gemini_provider_generate_success(gemini_provider, mock_genai_client):
    """Test successful content generation."""
    prompt = "Hello"
    system_prompt = "You are a helpful assistant."
    temperature = 0.5
    max_tokens = 100
    
    response_text, input_tokens, output_tokens = gemini_provider.generate(
        prompt=prompt,
        system_prompt=system_prompt,
        temperature=temperature,
        max_tokens=max_tokens
    )
    
    assert response_text == "Mocked response"
    assert input_tokens == 10 # From mock_genai_client.models.count_tokens
    assert output_tokens == 10 # From mock_genai_client.models.count_tokens
    mock_genai_client.models.generate_content.assert_called_once()

def test_gemini_provider_generate_api_error_retry(gemini_provider, mock_genai_client):
    """Test API error triggering retry mechanism."""
    mock_genai_client.models.generate_content.side_effect = [
        APIError("Transient error", code=500), # First call fails
        MagicMock(candidates=[MagicMock(content=MagicMock(parts=[MagicMock(text="Retry success")]))]) # Second call succeeds
    ]
    
    # Temporarily reduce MAX_RETRIES for faster test execution
    original_max_retries = gemini_provider.MAX_RETRIES
    gemini_provider.MAX_RETRIES = 2
    
    try:
        response_text, _, _ = gemini_provider.generate(
            prompt="Test retry", system_prompt="", temperature=0.5, max_tokens=100
        )
        assert response_text == "Retry success"
        assert mock_genai_client.models.generate_content.call_count == 2
    finally:
        gemini_provider.MAX_RETRIES = original_max_retries # Restore original
        mock_genai_client.models.generate_content.reset_mock() # Reset mock for other tests

def test_gemini_provider_generate_api_error_max_retries_exceeded(gemini_provider, mock_genai_client):
    """Test API error exceeding max retries."""
    mock_genai_client.models.generate_content.side_effect = APIError("Persistent error", code=500)
    
    original_max_retries = gemini_provider.MAX_RETRIES
    gemini_provider.MAX_RETRIES = 1 # Only one attempt
    
    with pytest.raises(LLMProviderError): # Expect LLMProviderError (or GeminiAPIError)
        gemini_provider.generate(prompt="Test max retry", system_prompt="", temperature=0.5, max_tokens=100)
    
    assert mock_genai_client.models.generate_content.call_count == 1 # Only one call made
    gemini_provider.MAX_RETRIES = original_max_retries # Restore original
    mock_genai_client.models.generate_content.reset_mock() # Reset mock for other tests

def test_gemini_provider_calculate_usd_cost(gemini_provider):
    """Test cost calculation for different models."""
    # Test flash model pricing
    gemini_provider.model_name = "gemini-2.5-flash-lite"
    cost = gemini_provider.calculate_usd_cost(1000, 2000)
    expected_cost_flash = (1000 / 1000) * 0.00008 + (2000 / 1000) * 0.00024
    assert cost == pytest.approx(expected_cost_flash)

    # Test pro model pricing
    gemini_provider.model_name = "gemini-2.5-pro"
    cost = gemini_provider.calculate_usd_cost(1000, 2000)
    expected_cost_pro = (1000 / 1000) * 0.0005 + (2000 / 1000) * 0.0015
    assert cost == pytest.approx(expected_cost_pro)

def test_gemini_provider_circuit_breaker_open(gemini_provider, mock_genai_client):
    """Test circuit breaker opening and preventing calls."""
    # Simulate failures to open the circuit
    gemini_provider.generate.circuit_breaker.failures = gemini_provider.generate.circuit_breaker.failure_threshold - 1
    mock_genai_client.models.generate_content.side_effect = APIError("Failure to open CB", code=500)
    
    with pytest.raises(APIError):
        gemini_provider.generate(prompt="Fail once", system_prompt="", temperature=0.5, max_tokens=100)
    
    assert gemini_provider.generate.circuit_breaker.state == "OPEN"
    
    # Next call should be blocked by circuit breaker
    with pytest.raises(CircuitBreakerError, match="Service unavailable: Circuit breaker is open"):
        gemini_provider.generate(prompt="Blocked by CB", system_prompt="", temperature=0.5, max_tokens=100)
    
    mock_genai_client.models.generate_content.reset_mock() # Reset mock for other tests
    gemini_provider.generate.circuit_breaker.state = "CLOSED" # Reset CB for other tests
    gemini_provider.generate.circuit_breaker.failures = 0

def test_gemini_provider_context_window_exceeded_error(gemini_provider, mock_genai_client):
    """Test handling of context window exceeded error."""
    mock_genai_client.models.generate_content.side_effect = APIError("Prompt too large", code=400)
    
    with pytest.raises(LLMProviderError, match="LLM context window exceeded"):
        gemini_provider.generate(prompt="Very long prompt", system_prompt="", temperature=0.5, max_tokens=100)
    
    mock_genai_client.models.generate_content.assert_called_once()
    mock_genai_client.models.generate_content.reset_mock() # Reset mock for other tests

# Add more tests as needed for other functionalities in llm_provider.py
