import pytest
from src.llm_provider import LLMProvider

# Mock the LLM API call for testing
class MockLLM:
    def __init__(self, response_text):
        self.response_text = response_text

    def generate_content(self, prompt):
        class MockResponse:
            def __init__(self, text):
                self.text = text
        return MockResponse(self.response_text)

@pytest.fixture
def mock_llm_provider():
    return LLMProvider(model=MockLLM("This is a mock LLM response."))

def test_llm_provider_initialization(mock_llm_provider):
    """Test that LLMProvider initializes correctly."""
    assert isinstance(mock_llm_provider, LLMProvider)

def test_llm_provider_generate_content(mock_llm_provider):
    """Test that LLMProvider correctly calls the model and returns content."""
    prompt = "What is the capital of France?"
    response = mock_llm_provider.generate_content(prompt)
    assert response == "This is a mock LLM response."

def test_llm_provider_handle_api_error(monkeypatch):
    """Test error handling in LLMProvider."""
    def mock_generate_error(prompt):
        raise Exception("API Error")
    
    monkeypatch.setattr("src.llm_provider.LLMProvider.generate_content", mock_generate_error)
    
    with pytest.raises(Exception) as excinfo:
        mock_llm_provider.generate_content("Test prompt")
    assert "API Error" in str(excinfo.value)