# tests/test_llm_provider_unit.py
import pytest
from unittest.mock import MagicMock, patch
import google.genai as genai
from google.genai.errors import APIError
import socket  # NEW: Import socket for network error tests

from src.llm_provider import GeminiProvider
from src.llm_tokenizers.gemini_tokenizer import (
    GeminiTokenizer,
)  # MODIFIED: Updated import path
from src.config.settings import ChimeraSettings
from src.exceptions import (
    LLMUnexpectedError,
    SchemaValidationError,
    GeminiAPIError,
)  # NEW: Import GeminiAPIError
from src.models import (
    GeneralOutput,
)  # NEW: Import GeneralOutput for schema validation tests


@pytest.fixture
def mock_llm_client_success():
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.candidates = [MagicMock()]
    mock_response.candidates[0].content = MagicMock()
    mock_response.candidates[0].content.parts = [MagicMock()]
    mock_response.candidates[0].content.parts[
        0
    ].text = "This is a simulated LLM response."
    mock_client.models.generate_content.return_value = mock_response

    mock_count_tokens_response = MagicMock()
    mock_count_tokens_response.total_tokens = 10
    mock_client.models.count_tokens.return_value = mock_count_tokens_response
    return mock_client


@pytest.fixture
def mock_llm_client_api_error():
    mock_client = MagicMock()
    # FIX: Correct APIError constructor: code is a positional argument, response_json is a dict
    mock_client.models.generate_content.side_effect = genai.types.APIError(
        "Simulated API Error", 500, {"error": {"message": "Simulated API Error"}}
    )
    mock_client.models.count_tokens.return_value.total_tokens = 0
    return mock_client


@pytest.fixture
def mock_llm_client_rate_limit():
    mock_client = MagicMock()
    # FIX: Correct APIError constructor: code is a positional argument, response_json is a dict
    mock_client.models.generate_content.side_effect = genai.types.APIError(
        "Rate limit exceeded", 429, {"error": {"message": "Rate limit exceeded"}}
    )
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
def test_llm_provider_generate_content_success(
    mock_llm_client_success, prompt, expected_output_structure
):
    """Tests successful content generation from the LLM provider."""
    with patch("src.llm_provider.genai.Client", return_value=mock_llm_client_success):
        mock_tokenizer = GeminiTokenizer(
            model_name="mock-model", genai_client=mock_llm_client_success
        )
        provider = GeminiProvider(
            api_key="AIza_mock-key-for-testing-purposes-1234567890",  # FIX: Long enough API key
            model_name="mock-model",
            tokenizer=mock_tokenizer,
            settings=ChimeraSettings(),
        )

        mock_llm_client_success.models.generate_content.return_value.candidates[
            0
        ].content.parts[0].text = "Simulated response for " + prompt
        mock_llm_client_success.models.count_tokens.return_value.total_tokens = (
            len(prompt) // 4
        )  # Simulate token count

        response_text, input_tokens, output_tokens, is_truncated = provider.generate(
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
        assert is_truncated == False

        mock_llm_client_success.models.generate_content.assert_called_once()
        mock_llm_client_success.models.generate_content.reset_mock()  # Reset for next iteration


def test_llm_provider_generate_api_error(mock_llm_client_api_error):
    """Tests handling of API errors during content generation."""
    with patch("src.llm_provider.genai.Client", return_value=mock_llm_client_api_error):
        mock_tokenizer = GeminiTokenizer(
            model_name="mock-model", genai_client=mock_llm_client_api_error
        )
        provider = GeminiProvider(
            api_key="AIza_mock-key-for-testing-purposes-1234567890",  # FIX: Long enough API key
            model_name="mock-model",
            tokenizer=mock_tokenizer,
            settings=ChimeraSettings(),
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
    with patch(
        "src.llm_provider.genai.Client", return_value=mock_llm_client_rate_limit
    ):
        mock_tokenizer = GeminiTokenizer(
            model_name="mock-model", genai_client=mock_llm_client_rate_limit
        )
        provider = GeminiProvider(
            api_key="AIza_mock-key-for-testing-purposes-1234567890",  # FIX: Long enough API key
            model_name="mock-model",
            tokenizer=mock_tokenizer,
            settings=ChimeraSettings(
                max_retries=1
            ),  # Set max_retries to 1 to quickly hit the limit
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
    with patch("src.llm_provider.genai.Client", return_value=mock_llm_client_success):
        mock_tokenizer = GeminiTokenizer(
            model_name="gemini-2.5-flash-lite", genai_client=mock_llm_client_success
        )
        provider = GeminiProvider(
            api_key="AIza_mock-key-for-testing-purposes-1234567890",  # FIX: Long enough API key
            model_name="gemini-2.5-flash-lite",
            tokenizer=mock_tokenizer,
            settings=ChimeraSettings(),
        )
        input_tokens = 1000
        output_tokens = 500
        expected_cost = (1000 / 1000 * 0.00008) + (500 / 1000 * 0.00024)
        cost = provider.calculate_usd_cost(input_tokens, output_tokens)
        assert cost == pytest.approx(expected_cost)


def test_llm_provider_tokenizer_integration(mock_llm_client_success):
    """Tests that the tokenizer is correctly integrated and used."""
    with patch("src.llm_provider.genai.Client", return_value=mock_llm_client_success):
        mock_tokenizer = GeminiTokenizer(
            model_name="mock-model", genai_client=mock_llm_client_success
        )
        provider = GeminiProvider(
            api_key="AIza_mock-key-for-testing-purposes-1234567890",  # FIX: Long enough API key
            model_name="mock-model",
            tokenizer=mock_tokenizer,
            settings=ChimeraSettings(),
        )

        test_prompt = "Hello world"
        system_prompt = "System"
        combined_prompt = f"{system_prompt}\n\n{test_prompt}"  # The actual string passed to count_tokens

        provider.generate(
            prompt=test_prompt,
            system_prompt=system_prompt,
            temperature=0.5,
            max_tokens=10,
        )

        # Assert that the tokenizer's count_tokens method was called with the correct combined prompt
        mock_llm_client_success.models.count_tokens.assert_called_with(
            model="mock-model", contents=combined_prompt
        )  # FIX: Assert on the correct mock and argument


def test_llm_provider_generate_malformed_response_index_error():
    """
    Test that GeminiProvider.generate handles a malformed API response
    (e.g., empty candidates list) and raises LLMUnexpectedError.
    """
    mock_client = MagicMock()
    # Simulate an API response with no candidates
    mock_response_no_candidates = MagicMock()
    mock_response_no_candidates.candidates = []  # This will cause an IndexError when [0] is accessed
    mock_client.models.generate_content.return_value = mock_response_no_candidates

    mock_tokenizer = MagicMock(spec=GeminiTokenizer)
    mock_tokenizer.count_tokens.side_effect = lambda text: len(text) // 4 if text else 0
    mock_tokenizer.max_output_tokens = 8192

    with patch("src.llm_provider.genai.Client", return_value=mock_client):
        provider = GeminiProvider(
            api_key="AIza_mock-key-for-testing-purposes-1234567890",  # FIX: Long enough API key
            model_name="mock-model",
            tokenizer=mock_tokenizer,
            settings=ChimeraSettings(),
        )

        with pytest.raises(
            LLMUnexpectedError, match="No candidates in response"
        ):  # FIX: Match the specific error message
            provider.generate(
                prompt="Test malformed response",
                system_prompt="System prompt",
                temperature=0.7,
                max_tokens=100,
            )
        mock_client.models.generate_content.assert_called_once()


def test_llm_provider_generate_malformed_response_attribute_error():
    """
    Test that GeminiProvider.generate handles a malformed API response
    (e.g., missing content/parts) and raises LLMUnexpectedError.
    """
    mock_client = MagicMock()
    # Simulate an API response where content.parts is missing
    mock_response_missing_parts = MagicMock()
    mock_response_missing_parts.candidates = [MagicMock()]
    mock_response_missing_parts.candidates[0].content = MagicMock()
    mock_response_missing_parts.candidates[
        0
    ].content.parts = []  # This will cause an IndexError
    mock_client.models.generate_content.return_value = mock_response_missing_parts

    mock_tokenizer = MagicMock(spec=GeminiTokenizer)
    mock_tokenizer.count_tokens.side_effect = lambda text: len(text) // 4 if text else 0
    mock_tokenizer.max_output_tokens = 8192

    with patch("src.llm_provider.genai.Client", return_value=mock_client):
        provider = GeminiProvider(
            api_key="AIza_mock-key-for-testing-purposes-1234567890",  # FIX: Long enough API key
            model_name="mock-model",
            tokenizer=mock_tokenizer,
            settings=ChimeraSettings(),
        )

        with pytest.raises(
            LLMUnexpectedError, match="No content parts in response"
        ):  # FIX: Match the specific error message
            provider.generate(
                prompt="Test malformed response",
                system_prompt="System prompt",
                temperature=0.7,
                max_tokens=100,
            )
        mock_client.models.generate_content.assert_called_once()


def test_llm_provider_generate_with_schema_validation_success():
    """Tests successful generation and schema validation."""
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.candidates = [MagicMock()]
    mock_response.candidates[0].content = MagicMock()
    mock_response.candidates[0].content.parts = [MagicMock()]
    mock_response.candidates[0].content.parts[
        0
    ].text = '{"general_output": "Valid output"}'
    mock_client.models.generate_content.return_value = mock_response

    mock_tokenizer = MagicMock(spec=GeminiTokenizer)
    mock_tokenizer.count_tokens.side_effect = lambda text: len(text) // 4 if text else 0
    mock_tokenizer.max_output_tokens = 8192

    with patch("src.llm_provider.genai.Client", return_value=mock_client):
        provider = GeminiProvider(
            api_key="AIza_mock-key-for-testing-purposes-1234567890",  # FIX: Long enough API key
            model_name="mock-model",
            tokenizer=mock_tokenizer,
            settings=ChimeraSettings(),
        )

        response_text, _, _, _ = provider.generate(
            prompt="Generate valid JSON",
            system_prompt="You are a helpful assistant.",
            output_schema=GeneralOutput,  # Pass a valid schema
            temperature=0.7,
            max_tokens=100,
        )
        assert response_text == '{"general_output": "Valid output"}'
        mock_client.models.generate_content.assert_called_once()


def test_llm_provider_generate_with_schema_validation_failure():
    """Tests that schema validation failure raises SchemaValidationError."""
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.candidates = [MagicMock()]
    mock_response.candidates[0].content = MagicMock()
    mock_response.candidates[0].content.parts = [MagicMock()]
    mock_response.candidates[0].content.parts[
        0
    ].text = '{"invalid_field": "Malformed output"}'  # Does not match GeneralOutput
    mock_client.models.generate_content.return_value = mock_response

    mock_tokenizer = MagicMock(spec=GeminiTokenizer)
    mock_tokenizer.count_tokens.side_effect = lambda text: len(text) // 4 if text else 0
    mock_tokenizer.max_output_tokens = 8192

    with patch("src.llm_provider.genai.Client", return_value=mock_client):
        provider = GeminiProvider(
            api_key="AIza_mock-key-for-testing-purposes-1234567890",  # FIX: Long enough API key
            model_name="mock-model",
            tokenizer=mock_tokenizer,
            settings=ChimeraSettings(),
        )

        with pytest.raises(
            SchemaValidationError, match="LLM output failed schema validation"
        ):
            provider.generate(
                prompt="Generate malformed JSON",
                system_prompt="System prompt",
                output_schema=GeneralOutput,  # Pass a schema that won't match
                temperature=0.7,
                max_tokens=100,
            )
        mock_client.models.generate_content.assert_called_once()


# Add these new test functions to tests/test_llm_provider_unit.py


def test_llm_provider_generate_api_error_401(mock_llm_client_api_error):
    """Tests that a 401 APIError (Unauthorized) raises GeminiAPIError."""
    # FIX: Correct APIError constructor: code is a positional argument
    mock_llm_client_api_error.models.generate_content.side_effect = (
        genai.types.APIError(
            "Invalid API Key",
            401,
            {
                "error": {"message": "Invalid API Key"}
            },  # FIX: Pass dict for response_json
        )
    )
    with patch("src.llm_provider.genai.Client", return_value=mock_llm_client_api_error):
        mock_tokenizer = GeminiTokenizer(
            model_name="mock-model", genai_client=mock_llm_client_api_error
        )
        provider = GeminiProvider(
            api_key="AIza_mock-key-for-testing-purposes-1234567890",  # FIX: Long enough API key
            model_name="mock-model",
            tokenizer=mock_tokenizer,
            settings=ChimeraSettings(),
        )
        with pytest.raises(GeminiAPIError, match="Invalid API Key"):
            provider.generate(
                prompt="test", system_prompt="sys", temperature=0.7, max_tokens=100
            )
        mock_llm_client_api_error.models.generate_content.assert_called_once()


def test_llm_provider_generate_api_error_403(mock_llm_client_api_error):
    """Tests that a 403 APIError (Forbidden) raises GeminiAPIError."""
    # FIX: Correct APIError constructor: code is a positional argument
    mock_llm_client_api_error.models.generate_content.side_effect = (
        genai.types.APIError(
            "API Key lacks permissions",
            403,
            {
                "error": {"message": "API Key lacks permissions"}
            },  # FIX: Pass dict for response_json
        )
    )
    with patch("src.llm_provider.genai.Client", return_value=mock_llm_client_api_error):
        mock_tokenizer = GeminiTokenizer(
            model_name="mock-model", genai_client=mock_llm_client_api_error
        )
        provider = GeminiProvider(
            api_key="AIza_mock-key-for-testing-purposes-1234567890",  # FIX: Long enough API key
            model_name="mock-model",
            tokenizer=mock_tokenizer,
            settings=ChimeraSettings(),
        )
        with pytest.raises(GeminiAPIError, match="API Key lacks permissions"):
            provider.generate(
                prompt="test", system_prompt="sys", temperature=0.7, max_tokens=100
            )
        mock_llm_client_api_error.models.generate_content.assert_called_once()


def test_llm_provider_generate_network_error(mock_llm_client_success):
    """Tests that a network error (e.g., socket.gaierror) raises LLMUnexpectedError."""
    mock_llm_client_success.models.generate_content.side_effect = socket.gaierror(
        "Name or service not known"
    )
    with patch("src.llm_provider.genai.Client", return_value=mock_llm_client_success):
        mock_tokenizer = GeminiTokenizer(
            model_name="mock-model", genai_client=mock_llm_client_success
        )
        provider = GeminiProvider(
            api_key="AIza_mock-key-for-testing-purposes-1234567890",  # FIX: Long enough API key
            model_name="mock-model",
            tokenizer=mock_tokenizer,
            settings=ChimeraSettings(  # FIX: Use valid max_backoff_seconds
                max_retries=1, max_backoff_seconds=5
            ),
        )
        with pytest.raises(
            LLMUnexpectedError, match="Name or service not known"
        ):  # Match the specific error message
            provider.generate(
                prompt="Test network error",
                system_prompt="System prompt",
                temperature=0.7,
                max_tokens=100,
            )
        # Should be called max_retries + 1 times (initial + retries)
        assert (
            mock_llm_client_success.models.generate_content.call_count == 2
        )  # 1 initial + 1 retry
