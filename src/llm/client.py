import abc
from typing import Any


class LLMClient(abc.ABC):
    """Abstract Base Class for LLM clients.
    Defines the interface for interacting with various Large Language Models.
    """

    @abc.abstractmethod
    def generate_response(self, prompt: str, **kwargs) -> dict[str, Any]:
        """Generates a response from the LLM based on the given prompt."""
        pass

    @abc.abstractmethod
    def count_tokens(self, text: str) -> int:
        """Counts the number of tokens in the given text."""
        pass


class MockLLMClient(LLMClient):
    """A mock implementation of LLMClient for testing purposes.
    Returns predefined responses and token counts.
    """

    def __init__(self, mock_response: dict[str, Any] = None, mock_token_count: int = 0):
        self.mock_response = (
            mock_response
            if mock_response is not None
            else {"choices": [{"message": {"content": "This is a mock response."}}]}
        )
        self.mock_token_count = mock_token_count

    def generate_response(self, prompt: str, **kwargs) -> dict[str, Any]:
        """Simulates generating a response from the LLM.
        Prints the received prompt for debugging.
        """
        # print(f"MockLLMClient received prompt: {prompt[:100]}...") # For debugging
        return self.mock_response

    def count_tokens(self, text: str) -> int:
        """Simulates counting tokens."""
        return self.mock_token_count


# Placeholder for actual LLM client implementation (e.g., OpenAI, Anthropic)
# class OpenAICLient(LLMClient):
#     def __init__(self, api_key: str):
#         # Initialize OpenAI client
#         pass
#
#     def generate_response(self, prompt: str, **kwargs) -> Dict[str, Any]:
#         # Call OpenAI API
#         pass
#
#     def count_tokens(self, text: str) -> int:
#         # Use tiktoken or similar to count tokens
#         pass
