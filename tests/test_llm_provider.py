import pytest
from unittest.mock import MagicMock, patch
from src.llm_provider import GeminiProvider
from src.tokenizers.gemini_tokenizer import GeminiTokenizer
import google.genai as genai

@pytest.fixture
def mock_genai_client():
    """Mocks the google.genai.Client."""
    mock_client = MagicMock(spec=genai.Client)
    mock_client.models.count_tokens.return_value = MagicMock(total_tokens=10)
    mock_client.models.generate_content.return_value = MagicMock(
        candidates=[MagicMock(content=MagicMock(parts=[MagicMock(text="Mocked LLM response")]))]
    )
    return mock_client

@pytest.fixture
def mock_tokenizer(mock_genai_client):
    """Mocks the GeminiTokenizer."""
    tokenizer = MagicMock(spec=GeminiTokenizer)
    tokenizer.model_name = "gemini-2.5-flash-lite"
    tokenizer.count_tokens.side_effect = lambda text: len(text) // 4 + 1 # Simple token estimation
    tokenizer.max_output_tokens = 65536
    return tokenizer

@pytest.fixture
def gemini_provider(mock_genai_client, mock_tokenizer):
    """Provides an instance of GeminiProvider with mocked dependencies."""
    with patch('src.llm_provider.genai.Client', return_value=mock_genai_client), \
         patch('src.llm_provider.GeminiTokenizer', return_value=mock_tokenizer):
        provider = GeminiProvider(api_key="test-api-key", model_name="gemini-2.5-flash-lite", tokenizer=mock_tokenizer)
        return provider

class TestGeminiProvider:
    def test_initialization(self, gemini_provider, mock_genai_client, mock_tokenizer):
        assert gemini_provider.client == mock_genai_client
        assert gemini_provider.tokenizer == mock_tokenizer
        assert gemini_provider.model_name == "gemini-2.5-flash-lite"

    def test_generate_success(self, gemini_provider, mock_genai_client, mock_tokenizer):
        prompt = "Test prompt"
        system_prompt = "System instruction"
        temperature = 0.5
        max_tokens = 100

        generated_text, input_tokens, output_tokens = gemini_provider.generate(
            prompt=prompt,
            system_prompt=system_prompt,
            temperature=temperature,
            max_tokens=max_tokens
        )

        assert generated_text == "Mocked LLM response"
        # Input tokens should be counted for combined prompt
        mock_tokenizer.count_tokens.assert_any_call(f"{system_prompt}\n\n{prompt}")
        # Output tokens should be counted for generated text
        mock_tokenizer.count_tokens.assert_any_call("Mocked LLM response")
        mock_genai_client.models.generate_content.assert_called_once()

    def test_generate_api_error(self, gemini_provider, mock_genai_client):
        mock_genai_client.models.generate_content.side_effect = genai.errors.APIError("API error", code=500)
        with pytest.raises(Exception, match="API error"):
            gemini_provider.generate(
                prompt="Error prompt",
                system_prompt="System instruction",
                temperature=0.5,
                max_tokens=100
            )