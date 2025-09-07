# tests/test_llm_provider_unit.py
import pytest
from unittest.mock import MagicMock, patch
import google.genai as genai
from google.genai.errors import APIError

from src.llm_provider import GeminiProvider
from src.tokenizers.gemini_tokenizer import GeminiTokenizer
from src.config.settings import ChimeraSettings

@pytest.fixture
def mock_llm_client_success():
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.candidates = [MagicMock()]
    mock_response.candidates[0].content = MagicMock()
    mock_response.candidates[0].content.parts = [MagicMock()]
    mock_response.candidates[0].content.parts[0].text = "This is a simulated LLM response."
    mock_client.models.generate_content.return_value = mock_response
    
    mock_count_tokens_response = MagicMock()
    mock_count_tokens_response.total_tokens = 10
    mock_client.models.count_tokens.return_value = mock_count_tokens_response
    return mock_client
    
@pytest.fixture
def mock_llm_client_api_error():
    mock_client = MagicMock()
    mock_client.models.generate_content.side_effect = APIError("Simulated API Error", code=500)
    mock_client.models.count_tokens.return_value.total_tokens = 0
    return mock_client

@pytest.fixture
def mock_llm_client_rate_limit():
    mock_client = MagicMock()
    mock_client.models.generate_content.side_effect = APIError("Rate limit exceeded", code=429)
    mock_client.models.count_tokens.return_value.total_tokens = 0
    return mock_client

@pytest.mark.parametrize(
    "prompt, expected_output_structure",
    [
        ("Generate a poem about AI.", str),
        ("Summarize this text: ...", str),
        ("Translate to French: Hello", str),
    ],
)
def test_llm_provider_generate_content_success(mock_llm_client_success, prompt, expected_output_structure):
    """Tests successful content generation from the LLM provider."""
    with patch('src.llm_provider.genai.Client', return_value=mock_llm_client_success):
        mock_tokenizer = GeminiTokenizer(model_name="mock-model", genai_client=mock_llm_client_success)
        provider = GeminiProvider(
            api_key="mock-key",
            model_name="mock-model",
            tokenizer=mock_tokenizer,
            settings=ChimeraSettings()
        )
        
        mock_llm_client_success.models.generate_content.return_value.candidates[0].content.parts[0].text = "Simulated response for " + prompt
        mock_llm_client_success.models.count_tokens.return_value.total_tokens = len(prompt) // 4 # Simulate token count

        response_text, input_tokens, output_tokens = provider.generate(
            prompt=prompt,
            system_prompt="You are a helpful assistant.",
            temperature=0.7,
            max_tokens=100,
        )

        assert response_text is not None
        assert isinstance(response_text, expected_output_structure)
        assert "Simulated response for" in response_text
        assert input_tokens > 0
        assert output_tokens > 0

        mock_llm_client_success.models.generate_content.assert_called_once()
        mock_llm_client_success.models.generate_content.reset_mock() # Reset for next iteration

def test_llm_provider_generate_api_error(mock_llm_client_api_error):
    """Tests handling of API errors during content generation."""
    with patch('src.llm_provider.genai.Client', return_value=mock_llm_client_api_error):
        mock_tokenizer = GeminiTokenizer(model_name="mock-model", genai_client=mock_llm_client_api_error)
        provider = GeminiProvider(
            api_key="mock-key",
            model_name="mock-model",
            tokenizer=mock_tokenizer,
            settings=ChimeraSettings()
        )
        
        with pytest.raises(APIError, match="Simulated API Error"):
            provider.generate(
                prompt="This prompt should cause an error.",
                system_prompt="You are a helpful assistant.",
                temperature=0.7,
                max_tokens=100,
            )
        mock_llm_client_api_error.models.generate_content.assert_called_once()

def test_llm_provider_generate_rate_limit_error(mock_llm_client_rate_limit):
    """Tests handling of rate limit errors during content generation."""
    with patch('src.llm_provider.genai.Client', return_value=mock_llm_client_rate_limit):
        mock_tokenizer = GeminiTokenizer(model_name="mock-model", genai_client=mock_llm_client_rate_limit)
        provider = GeminiProvider(
            api_key="mock-key",
            model_name="mock-model",
            tokenizer=mock_tokenizer,
            settings=ChimeraSettings(max_retries=1) # Set max_retries to 1 to quickly hit the limit
        )
        
        with pytest.raises(APIError, match="Rate limit exceeded"):
            provider.generate(
                prompt="This prompt should cause a rate limit error.",
                system_prompt="You are a helpful assistant.",
                temperature=0.7,
                max_tokens=100,
            )
        mock_llm_client_rate_limit.models.generate_content.assert_called_once()

def test_llm_provider_calculate_usd_cost(mock_llm_client_success):
    """Tests cost calculation."""
    with patch('src.llm_provider.genai.Client', return_value=mock_llm_client_success):
        mock_tokenizer = GeminiTokenizer(model_name="gemini-2.5-flash-lite", genai_client=mock_llm_client_success)
        provider = GeminiProvider(
            api_key="mock-key",
            model_name="gemini-2.5-flash-lite",
            tokenizer=mock_tokenizer,
            settings=ChimeraSettings()
        )
        input_tokens = 1000
        output_tokens = 500
        expected_cost = (1000/1000 * 0.00008) + (500/1000 * 0.00024)
        cost = provider.calculate_usd_cost(input_tokens, output_tokens)
        assert cost == pytest.approx(expected_cost)
    
def test_llm_provider_tokenizer_integration(mock_llm_client_success):
    """Tests that the tokenizer is correctly integrated and used."""
    with patch('src.llm_provider.genai.Client', return_value=mock_llm_client_success):
        mock_tokenizer = GeminiTokenizer(model_name="mock-model", genai_client=mock_llm_client_success)
        provider = GeminiProvider(
            api_key="mock-key",
            model_name="mock-model",
            tokenizer=mock_tokenizer,
            settings=ChimeraSettings()
        )
        
        test_prompt = "Hello world"
        mock_tokenizer.count_tokens.return_value = 2 # Mock a token count
        
        provider.generate(
            prompt=test_prompt,
            system_prompt="System",
            temperature=0.5,
            max_tokens=10,
        )
        
        mock_tokenizer.count_tokens.assert_called_with("System\n\nHello world")
