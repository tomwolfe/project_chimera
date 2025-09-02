import pytest
from unittest.mock import MagicMock
from src.llm_provider import GeminiProvider # Corrected to GeminiProvider

@pytest.fixture
def mock_gemini_provider(): # Renamed fixture for clarity
    # Mock the GeminiProvider instance
    provider = MagicMock(spec=GeminiProvider)
    # Mock the 'generate' method, which is the actual method in GeminiProvider
    # It returns (generated_text, input_tokens, output_tokens)
    provider.generate.return_value = ("Mocked response text", 10, 20)
    return provider

def test_gemini_provider_generate_success(mock_gemini_provider): # Renamed test for clarity
    # Test the 'generate' method for success
    generated_text, input_tokens, output_tokens = mock_gemini_provider.generate(
        prompt="Test prompt",
        system_prompt="System instruction",
        temperature=0.5,
        max_tokens=100
    )
    assert generated_text == "Mocked response text"
    assert input_tokens == 10
    assert output_tokens == 20
    # Verify that 'generate' was called once with the correct arguments
    mock_gemini_provider.generate.assert_called_once_with(
        prompt="Test prompt",
        system_prompt="System instruction",
        temperature=0.5,
        max_tokens=100
    )

def test_gemini_provider_handle_api_error(mock_gemini_provider): # Renamed test for clarity
    # Test error handling for the 'generate' method
    mock_gemini_provider.generate.side_effect = Exception("API Error")
    with pytest.raises(Exception, match="API Error"):
        mock_gemini_provider.generate(
            prompt="Prompt that causes error",
            system_prompt="System instruction",
            temperature=0.5,
            max_tokens=100
        )