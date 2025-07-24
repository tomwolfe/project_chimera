# llm_provider.py
import google.genai as genai
from google.genai import types
import time
import random
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

# --- Token Cost Definitions (per 1,000 tokens) ---
# These are example costs and should be verified against Google's official pricing.
# As of late 2023/early 2024, Gemini 1.5 Flash is cheaper than 1.5 Pro.
# The project uses "gemini-2.5-flash-lite", "gemini-2.5-pro", "gemini-2.5-flash".
# We'll map "2.5-flash-lite" and "2.5-flash" to 1.5 Flash pricing, and "2.5-pro" to 1.5 Pro pricing.
TOKEN_COSTS_PER_1K_TOKENS = {
    "gemini-1.5-flash": { # Used for "gemini-2.5-flash-lite" and "gemini-2.5-flash"
        "input": 0.00008,
        "output": 0.00024,
    },
    "gemini-1.5-pro": { # Used for "gemini-2.5-pro"
        "input": 0.0005,
        "output": 0.0015,
    }
}

class GeminiProvider:
    # Retry parameters
    MAX_RETRIES = 5
    INITIAL_BACKOFF_SECONDS = 1
    BACKOFF_FACTOR = 2
    MAX_BACKOFF_SECONDS = 60
    # HTTP status codes that indicate a transient error and should be retried
    RETRYABLE_HTTP_CODES = {429, 500, 502, 503, 504}

    def __init__(self, api_key: str, model_name: str = "gemini-2.5-flash-lite", status_callback=None):
        self.client = genai.Client(api_key=api_key)
        self.model_name = model_name
        self.status_callback = status_callback # Callback for Streamlit status updates

    def _log_status(self, message: str, state: str = "running", expanded: bool = True,
                    current_total_tokens: int = 0, current_total_cost: float = 0.0,
                    estimated_next_step_tokens: int = 0, estimated_next_step_cost: float = 0.0):
        if self.status_callback:
            # Pass all relevant metrics to the callback
            self.status_callback(
                message=message,
                state=state,
                expanded=expanded,
                current_total_tokens=current_total_tokens,
                current_total_cost=current_total_cost,
                estimated_next_step_tokens=estimated_next_step_tokens,
                estimated_next_step_cost=estimated_next_step_cost
            )
        else:
            print(f"[LLM Provider] {message}") # Fallback to print if no callback

    def _get_pricing_model_name(self) -> str:
        """Maps the user-selected model name to a pricing tier name."""
        if "flash" in self.model_name:
            return "gemini-1.5-flash"
        elif "pro" in self.model_name:
            return "gemini-1.5-pro"
        return "gemini-1.5-flash" # Default to flash if unknown

    def calculate_usd_cost(self, input_tokens: int, output_tokens: int) -> float:
        """Calculates the estimated USD cost based on token usage and model pricing."""
        pricing_model = self._get_pricing_model_name()
        costs = TOKEN_COSTS_PER_1K_TOKENS.get(pricing_model)
        if not costs:
            self._log_status(f"Warning: No pricing information for model '{self.model_name}'. Cost estimation will be $0.", state="warning")
            return 0.0
        
        input_cost = (input_tokens / 1000) * costs["input"]
        output_cost = (output_tokens / 1000) * costs["output"]
        return input_cost + output_cost

    def generate(self, prompt: str, system_prompt: str, temperature: float, max_tokens: int) -> tuple[str, int, int]:
        """
        Generates content using the Gemini model.
        Returns a tuple of (generated_text: str, input_tokens_used: int, output_tokens_used: int).
        Raises custom exceptions on error.
        """
        config = types.GenerateContentConfig(
            system_instruction=system_prompt,
            temperature=temperature,
            max_output_tokens=max_tokens
        )

        for attempt in range(1, self.MAX_RETRIES + 1):
            try:
                # The 'contents' argument expects a list of parts, even for a single string.
                # The SDK automatically converts a string to [types.UserContent(parts=[types.Part.from_text(text=prompt)])]
                response = self.client.models.generate_content(
                    model=self.model_name,
                    contents=prompt,
                    config=config
                )
                input_tokens = response.usage_metadata.prompt_token_count
                output_tokens = response.usage_metadata.candidates_token_count
                return response.text, input_tokens, output_tokens
            except APIError as e:
                if e.code in self.RETRYABLE_HTTP_CODES and attempt < self.MAX_RETRIES:
                    backoff_time = min(self.INITIAL_BACKOFF_SECONDS * (self.BACKOFF_FACTOR ** (attempt - 1)), self.MAX_BACKOFF_SECONDS)
                    jitter = random.uniform(0, 0.5 * backoff_time) # Add jitter
                    sleep_time = backoff_time + jitter
                    self._log_status(f"Gemini API Error (Code: {e.code}). Retrying in {sleep_time:.2f} seconds... (Attempt {attempt}/{self.MAX_RETRIES})", state="running")
                    time.sleep(sleep_time)
                else:
                    # Non-retryable API error or last retry failed
                    raise GeminiAPIError(e.message, e.code) from e
            except Exception as e:
                # Catch-all for other unexpected errors (e.g., network issues)
                if attempt < self.MAX_RETRIES:
                    backoff_time = min(self.INITIAL_BACKOFF_SECONDS * (self.BACKOFF_FACTOR ** (attempt - 1)), self.MAX_BACKOFF_SECONDS)
                    jitter = random.uniform(0, 0.5 * backoff_time)
                    sleep_time = backoff_time + jitter
                    self._log_status(f"Unexpected error: {e}. Retrying in {sleep_time:.2f} seconds... (Attempt {attempt}/{self.MAX_RETRIES})", state="running")
                    time.sleep(sleep_time)
                else:
                    raise LLMUnexpectedError(str(e)) from e
        
        # This part should ideally not be reached if exceptions are always re-raised on last attempt
        raise LLMUnexpectedError("Max retries exceeded for generate call.")

    def count_tokens(self, prompt: str, system_prompt: str) -> int:
        """
        Estimates the token count for a given prompt and system prompt.
        """
        contents_for_counting = [
            types.Content(role='system', parts=[types.Part(text=system_prompt)]),
            types.Content(role='user', parts=[types.Part(text=prompt)])
        ]

        for attempt in range(1, self.MAX_RETRIES + 1):
            try:
                response = self.client.models.count_tokens(
                    model=self.model_name,
                    contents=contents_for_counting
                )
                return response.total_tokens
            except APIError as e:
                if e.code in self.RETRYABLE_HTTP_CODES and attempt < self.MAX_RETRIES:
                    backoff_time = min(self.INITIAL_BACKOFF_SECONDS * (self.BACKOFF_FACTOR ** (attempt - 1)), self.MAX_BACKOFF_SECONDS)
                    jitter = random.uniform(0, 0.5 * backoff_time)
                    sleep_time = backoff_time + jitter
                    self._log_status(f"Gemini API Error (Code: {e.code}) during token count. Retrying in {sleep_time:.2f} seconds... (Attempt {attempt}/{self.MAX_RETRIES})", state="running")
                    time.sleep(sleep_time)
                else:
                    raise GeminiAPIError(e.message, e.code) from e
            except Exception as e:
                if attempt < self.MAX_RETRIES:
                    backoff_time = min(self.INITIAL_BACKOFF_SECONDS * (self.BACKOFF_FACTOR ** (attempt - 1)), self.MAX_BACKOFF_SECONDS)
                    jitter = random.uniform(0, 0.5 * backoff_time)
                    sleep_time = backoff_time + jitter
                    self._log_status(f"Unexpected error: {e} during token count. Retrying in {sleep_time:.2f} seconds... (Attempt {attempt}/{self.MAX_RETRIES})", state="running")
                    time.sleep(sleep_time)
                else:
                    raise LLMUnexpectedError(str(e)) from e
        
        raise LLMUnexpectedError("Max retries exceeded for count_tokens call.")