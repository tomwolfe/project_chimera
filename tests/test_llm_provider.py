# tests/test_llm_provider.py

import pytest
from unittest.mock import MagicMock
from src.llm_provider import LLMProvider

# Mocking the LLM client to avoid actual API calls during testing
class MockLLMClient:
    def __init__(self):
        self.chat = MagicMock()

    def generate_content(self, prompt):
        mock_response = MagicMock()
        mock_response.text = "Mocked LLM response."
        return mock_response

@pytest.fixture
def llm_provider():
    client = MockLLMClient()
    return LLMProvider(client=client)

def test_llm_provider_initialization(llm_provider):
    assert isinstance(llm_provider, LLMProvider)
    assert isinstance(llm_provider.client, MockLLMClient)

def test_llm_provider_generate_content(llm_provider):
    prompt = "Test prompt."
    response = llm_provider.generate_content(prompt)
    assert response == "Mocked LLM response."
    llm_provider.client.chat.assert_called_once()

def test_llm_provider_generate_content_with_history(llm_provider):
    prompt = "Test prompt with history."
    history = [{"role": "user", "content": "Previous message."}]
    response = llm_provider.generate_content(prompt, history=history)
    assert response == "Mocked LLM response."
    # Check if chat was called with the correct arguments, including history
    llm_provider.client.chat.assert_called_once()
    # Note: The exact arguments depend on how the MockLLMClient is designed to be called.
    # This is a placeholder assertion.

def test_llm_provider_handle_error(llm_provider):
    # Mocking an error scenario
    llm_provider.client.chat.side_effect = Exception("API Error")
    prompt = "Error test prompt."
    with pytest.raises(Exception, match="API Error"):
        llm_provider.generate_content(prompt)
    llm_provider.client.chat.assert_called_once()