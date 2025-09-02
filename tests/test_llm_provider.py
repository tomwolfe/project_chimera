import pytest
from src.llm_provider import LLMProvider


# Mocking the LLM API response for testing
class MockLLMResponse:
    def __init__(self, text):
        self.text = text

    def __str__(self):
        return self.text


class MockLLM:
    def __init__(self, response_text):
        self.response_text = response_text

    def generate_content(self, prompt):
        # Simulate a simple response based on prompt content
        if "hello" in prompt.lower():
            return MockLLMResponse(self.response_text)
        else:
            return MockLLMResponse("Default response.")


@pytest.fixture
def llm_provider():
    # Use a mock LLM for testing to avoid actual API calls
    mock_llm = MockLLM("This is a mocked LLM response.")
    # Assuming LLMProvider's constructor can accept a mock LLM client directly
    # or can be initialized with a dummy API key and then have its client mocked.
    # For this test, we'll assume it can take a 'model' argument that is our mock.
    # If LLMProvider requires a genai.Client, you'd mock that.
    # Given the structure of src/llm_provider.py, it expects an API key and creates its own client.
    # So, we need to mock the internal genai.Client or the generate method of GeminiProvider.

    # Let's adjust the fixture to mock the GeminiProvider's generate method
    with patch("src.llm_provider.GeminiProvider.generate") as mock_generate:
        mock_generate.return_value = (
            "This is a mocked LLM response.",
            100,
            50,
        )  # (text, input_tokens, output_tokens)
        # LLMProvider expects an API key and model name, but its generate method is mocked.
        # The actual LLMProvider instance will still be created, but its API call is intercepted.
        provider = LLMProvider(
            api_key="dummy-api-key", model_name="gemini-2.5-flash-lite"
        )
        yield provider


def test_llm_provider_generate_content_success(llm_provider):
    """Test successful content generation from LLM provider."""
    prompt = "Say hello to the LLM."
    response_text, _, _ = llm_provider.generate(
        prompt,
        system_prompt="You are a helpful assistant.",
        temperature=0.4,
        max_tokens=100,
    )
    assert "mocked LLM response" in response_text.lower()


def test_llm_provider_generate_content_empty_prompt(llm_provider):
    """Test content generation with an empty prompt."""
    prompt = ""
    response_text, _, _ = llm_provider.generate(
        prompt,
        system_prompt="You are a helpful assistant.",
        temperature=0.4,
        max_tokens=100,
    )
    assert "mocked LLM response" in response_text.lower()  # Mock always returns this


def test_llm_provider_generate_content_error_handling(llm_provider):
    """Test error handling for LLM content generation (simulated)."""
    with patch(
        "src.llm_provider.GeminiProvider.generate", side_effect=Exception("API Error")
    ):
        with pytest.raises(Exception, match="API Error"):
            llm_provider.generate(
                "This prompt should cause an error.",
                system_prompt="You are a helpful assistant.",
                temperature=0.4,
                max_tokens=100,
            )


# Add more tests for different scenarios and edge cases
