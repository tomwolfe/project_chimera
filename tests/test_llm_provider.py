import pytest
from unittest.mock import MagicMock, patch
import google.genai as genai
from google.genai.errors import APIError

# Assuming llm_provider.py contains a class or functions to interact with an LLM
# Replace with the actual import path if different
from src.llm_provider import LLMProvider
from src.tokenizers.gemini_tokenizer import GeminiTokenizer
from src.config.settings import ChimeraSettings


# Mocking the LLM API client to isolate tests
@pytest.fixture
def mock_llm_client():
    mock_client = MagicMock()
    # Mock the models.generate_content method
    mock_response = MagicMock()
    mock_response.candidates = [MagicMock()]
    mock_response.candidates[0].content = MagicMock()
    mock_response.candidates[0].content.parts = [MagicMock()]
    mock_response.candidates[0].content.parts[0].text = "This is a simulated LLM response."
    mock_client.models.generate_content.return_value = mock_response
    
    # Mock the models.count_tokens method
    mock_count_tokens_response = MagicMock()
    mock_count_tokens_response.total_tokens = 10
    mock_client.models.count_tokens.return_value = mock_count_tokens_response
    return mock_client


@pytest.mark.parametrize(
    "prompt, expected_output_structure",
    [
        ("Generate a poem about AI.", str),
        ("Summarize this text: ...", str),
        ("Translate to French: Hello", str),
    ],
)
def test_llm_provider_generate_content_success(
    mock_llm_client, prompt, expected_output_structure
):
    """Tests successful content generation from the LLM provider."""
    # Patch genai.Client to return our mock_llm_client
    with patch('src.llm_provider.genai.Client', return_value=mock_llm_client):
        # Create a mock tokenizer that uses the mock_llm_client
        mock_tokenizer = GeminiTokenizer(model_name="mock-model", genai_client=mock_llm_client)
        
        provider = LLMProvider(
            api_key="mock-key",
            model_name="mock-model",
            tokenizer=mock_tokenizer,
            settings=ChimeraSettings()
        )
        
        # Mock the generate method to return a tuple: (raw_output, input_tokens, output_tokens)
        mock_llm_client.models.generate_content.return_value.candidates[0].content.parts[0].text = "Simulated response for " + prompt
        mock_llm_client.models.count_tokens.return_value.total_tokens = len(prompt) // 4 # Simulate token count

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

        mock_llm_client.models.generate_content.assert_called_once()
        mock_llm_client.models.generate_content.reset_mock() # Reset for next iteration

def test_llm_provider_generate_api_error(mock_llm_client):
    """Tests handling of API errors during content generation."""
    with patch('src.llm_provider.genai.Client', return_value=mock_llm_client):
        mock_tokenizer = GeminiTokenizer(model_name="mock-model", genai_client=mock_llm_client)
        provider = LLMProvider(
            api_key="mock-key",
            model_name="mock-model",
            tokenizer=mock_tokenizer,
            settings=ChimeraSettings()
        )
        
        # Configure the mock client to raise an APIError
        mock_llm_client.models.generate_content.side_effect = APIError("Simulated API Error", code=500)

        with pytest.raises(APIError, match="Simulated API Error"):
            provider.generate(
                prompt="This prompt should cause an error.",
                system_prompt="You are a helpful assistant.",
                temperature=0.7,
                max_tokens=100,
            )
        mock_llm_client.models.generate_content.assert_called_once()

def test_llm_provider_calculate_usd_cost(mock_llm_client):
    """Tests cost calculation."""
    with patch('src.llm_provider.genai.Client', return_value=mock_llm_client):
        mock_tokenizer = GeminiTokenizer(model_name="gemini-2.5-flash-lite", genai_client=mock_llm_client)
        provider = LLMProvider(
            api_key="mock-key",
            model_name="gemini-2.5-flash-lite",
            tokenizer=mock_tokenizer,
            settings=ChimeraSettings()
        )
        # Using the costs defined in llm_provider.py for gemini-1.5-flash
        # input: 0.00008, output: 0.00024 per 1k tokens
        input_tokens = 1000
        output_tokens = 500
        expected_cost = (1000/1000 * 0.00008) + (500/1000 * 0.00024)
        cost = provider.calculate_usd_cost(input_tokens, output_tokens)
        assert cost == pytest.approx(expected_cost)

def test_llm_provider_tokenizer_integration(mock_llm_client):
    """Tests that the tokenizer is correctly integrated and used."""
    with patch('src.llm_provider.genai.Client', return_value=mock_llm_client):
        mock_tokenizer = GeminiTokenizer(model_name="mock-model", genai_client=mock_llm_client)
        provider = LLMProvider(
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
        
        # Assert that tokenizer.count_tokens was called for the input prompt
        mock_tokenizer.count_tokens.assert_called_with("System\n\nHello world")