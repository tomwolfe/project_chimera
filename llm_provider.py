# llm_provider.py
import google.genai as genai
from google.genai import types
from google.genai.errors import APIError

class GeminiProvider:
    def __init__(self, api_key: str, model_name: str = "gemini-2.5-flash-lite-preview-06-17"):
        # Rationale: 'flash-lite' offers the best balance of speed, cost, and capability for this iterative, multi-call process.
        self.client = genai.Client(api_key=api_key)
        self.model_name = model_name

    def generate(self, prompt: str, system_prompt: str, temperature: float, max_tokens: int) -> str:
        config = types.GenerateContentConfig(
            system_instruction=system_prompt,
            temperature=temperature,
            max_output_tokens=max_tokens
        )
        try:
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=prompt,
                config=config
            )
            return response.text
        except APIError as e:
            # Handles API-level errors (e.g., invalid key, rate limits)
            return f"[ERROR] Gemini API Error: {e.message}"
        except Exception as e:
            # Handles other exceptions (e.g., network issues)
            return f"[ERROR] An unexpected error occurred: {e}"
