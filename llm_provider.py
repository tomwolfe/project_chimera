# llm_provider.py
import google.genai as genai
from google.genai import types
from google.genai.errors import APIError

# --- Custom Exceptions ---
class LLMProviderError(Exception):
    """Base exception for LLM provider errors."""
    pass

class GeminiAPIError(LLMProviderError):
    """Specific exception for Gemini API errors."""
    def __init__(self, message: str, code: int = None):
        super().__init__(message)
        self.code = code

class LLMUnexpectedError(LLMProviderError):
    """Specific exception for unexpected LLM errors."""
    pass

class GeminiProvider:
    def __init__(self, api_key: str, model_name: str = "gemini-2.5-flash-lite"):
        # Rationale: 'flash-lite' offers the best balance of speed, cost, and capability for this iterative, multi-call process.
        self.client = genai.Client(api_key=api_key)
        self.model_name = model_name

    def generate(self, prompt: str, system_prompt: str, temperature: float, max_tokens: int) -> str:
        """
        Generates content using the Gemini model.
        Returns a tuple of (generated_text: str, total_tokens_used: int).
        Raises custom exceptions on error.
        """
        config = types.GenerateContentConfig(
            system_instruction=system_prompt,
            temperature=temperature,
            max_output_tokens=max_tokens
        )
        try:
            # The 'contents' argument expects a list of parts, even for a single string.
            # The SDK automatically converts a string to [types.UserContent(parts=[types.Part.from_text(text=prompt)])]
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=prompt,
                config=config
            )
            total_tokens = response.usage_metadata.prompt_token_count + response.usage_metadata.candidates_token_count
            return response.text, total_tokens
        except APIError as e:
            # Handles API-level errors (e.g., invalid key, rate limits)
            raise GeminiAPIError(e.message, e.code) from e
        except Exception as e:
            # Handles other exceptions (e.g., network issues)
            raise LLMUnexpectedError(str(e)) from e

    def count_tokens(self, prompt: str, system_prompt: str) -> int:
        """
        Estimates the token count for a given prompt and system prompt.
        """
        try:
            response = self.client.models.count_tokens(
                model=self.model_name,
                contents=prompt,
                system_instruction=system_prompt
            )
            return response.total_tokens
        except Exception as e:
            # If counting tokens fails, it's likely an API issue (e.g., invalid key)
            # Re-raise as GeminiAPIError for consistent handling.
            raise GeminiAPIError(f"Failed to count tokens: {e}") from e