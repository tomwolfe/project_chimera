import pytest
from unittest.mock import MagicMock, patch
from src.llm_provider import GeminiProvider, LLMProviderError
from src.models import PersonaConfig
from google.genai.errors import APIError

@pytest.fixture
def mock_genai_client():
    mock_client = MagicMock()
    mock_client.models.count_tokens.return_value.total_tokens = 10
    mock_client.models.generate_content.return_value.candidates = [
        MagicMock(content=MagicMock(parts=[MagicMock(text="Mocked LLM response")]))
    ]
    return mock_client

@pytest.fixture
def gemini_provider_instance(mock_genai_client):
    with patch("google.genai.Client", return_value=mock_genai_client):
        return GeminiProvider(api_key="test_api_key", model_name="gemini-2.5-flash-lite")

def test_gemini_provider_generate_success(gemini_provider_instance, mock_genai_client):
    # Arrange
    prompt = "Test prompt"
    system_prompt = "You are a test assistant."
    persona_config = PersonaConfig(name="TestPersona", system_prompt=system_prompt, temperature=0.5, max_tokens=100)

    # Act
    response_text, input_tokens, output_tokens = gemini_provider_instance.generate(
        prompt=prompt, system_prompt=system_prompt, temperature=0.5, max_tokens=100, persona_config=persona_config
    )

    # Assert
    assert response_text == "Mocked LLM response"
    assert input_tokens == 10 # From mock_genai_client.models.count_tokens
    assert output_tokens == 10 # From mock_genai_client.models.count_tokens
    mock_genai_client.models.generate_content.assert_called_once()

def test_gemini_provider_generate_api_error_non_retryable(gemini_provider_instance, mock_genai_client):
    # Arrange
    mock_genai_client.models.generate_content.side_effect = APIError("Invalid API Key", code=401)

    # Act & Assert
    with pytest.raises(LLMProviderError, match="Invalid API Key"): # Expect specific LLMProviderError
        gemini_provider_instance.generate(
            prompt="Invalid key test", system_prompt="", temperature=0.5, max_tokens=100
        )
    mock_genai_client.models.generate_content.assert_called_once()