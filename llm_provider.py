# llm_provider.py
import google.genai as genai
from google.genai import types
import time
import random
from google.genai.errors import APIError
import streamlit as st
from collections import defaultdict
import hashlib
import json
import re
import logging

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

logger = logging.getLogger(__name__)

# Apply st.cache_resource to the class itself
@st.cache_resource
class GeminiProvider:
    # Retry parameters
    MAX_RETRIES = 10
    INITIAL_BACKOFF_SECONDS = 1
    BACKOFF_FACTOR = 2
    MAX_BACKOFF_SECONDS = 60 # Maximum backoff time in seconds
    RETRYABLE_HTTP_CODES = {429, 500, 502, 503, 504}

    # Renamed status_callback to _status_callback to prevent hashing issues with Streamlit's cache_resource
    def __init__(self, api_key: str, model_name: str = "gemini-2.5-flash-lite", _status_callback=None):
        # Store these as instance attributes, but they are also used for caching.
        self._api_key = api_key # Store API key for hashing/equality
        self.model_name = model_name
        self._status_callback = _status_callback # Use the renamed parameter
        self.client = genai.Client(api_key=self._api_key) # Initialize client here

    # Define __hash__ and __eq__ for caching to work correctly
    def __hash__(self):
        # Hash based on model_name and a hash of the API key (not the key itself)
        return hash((self.model_name, hashlib.sha256(self._api_key.encode()).hexdigest()))

    def __eq__(self, other):
        if not isinstance(other, GeminiProvider):
            return NotImplemented
        return self.model_name == other.model_name and self._api_key == other._api_key

    def _log_status(self, message: str, state: str = "running", expanded: bool = True,
                    current_total_tokens: int = 0, current_total_cost: float = 0.0,
                    estimated_next_step_tokens: int = 0, estimated_next_step_cost: float = 0.0):
        # Use the renamed internal attribute
        if self._status_callback:
            self._status_callback(
                message=message,
                state=state,
                expanded=expanded,
                current_total_tokens=current_total_tokens,
                current_total_cost=current_total_cost,
                estimated_next_step_tokens=estimated_next_step_tokens,
                estimated_next_step_cost=estimated_next_step_cost
            )
        else:
            logger.info(f"[LLM Provider] {message}")

    def _get_pricing_model_name(self) -> str:
        if "flash" in self.model_name:
            return "gemini-1.5-flash"
        elif "pro" in self.model_name:
            return "gemini-1.5-pro"
        return "gemini-1.5-flash"

    def calculate_usd_cost(self, input_tokens: int, output_tokens: int) -> float:
        pricing_model = self._get_pricing_model_name()
        costs = TOKEN_COSTS_PER_1K_TOKENS.get(pricing_model)
        if not costs:
            self._log_status(f"Warning: No pricing information for model '{self.model_name}'. Cost estimation will be $0.", state="running")
            return 0.0

        input_cost = (input_tokens / 1000) * costs["input"]
        output_cost = (output_tokens / 1000) * costs["output"]
        return input_cost + output_cost

    # Use st.cache_data for methods that perform computations based on inputs
    # The 'self' argument is implicitly handled by Streamlit's caching for methods
    # when the class itself is cached with @st.cache_resource.
    @st.cache_data(ttl=3600, show_spinner=False)
    def generate(_self, prompt: str, system_prompt: str, temperature: float, max_tokens: int) -> tuple[str, int, int]:
        config = types.GenerateContentConfig(
            system_instruction=system_prompt,
            temperature=temperature,
            max_output_tokens=max_tokens
        )

        for attempt in range(1, _self.MAX_RETRIES + 1):
            try:
                response = _self.client.models.generate_content(
                    model=_self.model_name,
                    contents=prompt,
                    config=config
                )
                input_tokens = response.usage_metadata.prompt_token_count
                output_tokens = response.usage_metadata.candidates_token_count

                return response.text, input_tokens, output_tokens
            except APIError as e:
                error_msg = str(e).encode('utf-8', 'replace').decode('utf-8')
                http_status_code = getattr(e, 'response', None)
                if http_status_code: http_status_code = http_status_code.status_code

                if http_status_code is not None and http_status_code in _self.RETRYABLE_HTTP_CODES and attempt < _self.MAX_RETRIES:
                    backoff_time = min(_self.INITIAL_BACKOFF_SECONDS * (_self.BACKOFF_FACTOR ** (attempt - 1)), _self.MAX_BACKOFF_SECONDS)
                    jitter = random.uniform(0, 0.5 * backoff_time)
                    sleep_time = backoff_time + jitter
                    _self._log_status(f"Gemini API Error (Status: {http_status_code}, Message: {error_msg}). Retrying in {sleep_time:.2f} seconds... (Attempt {attempt}/{_self.MAX_RETRIES})", state="running")
                    time.sleep(sleep_time)
                else:
                    raise GeminiAPIError(error_msg, http_status_code if http_status_code is not None else getattr(e, 'code', None)) from e
            except Exception as e:
                error_msg = str(e).encode('utf-8', 'replace').decode('utf-8')
                if attempt < _self.MAX_RETRIES:
                    backoff_time = min(_self.INITIAL_BACKOFF_SECONDS * (_self.BACKOFF_FACTOR ** (attempt - 1)), _self.MAX_BACKOFF_SECONDS)
                    jitter = random.uniform(0, 0.5 * backoff_time)
                    sleep_time = backoff_time + jitter
                    _self._log_status(f"Unexpected error: {error_msg}. Retrying in {sleep_time:.2f} seconds... (Attempt {attempt}/{_self.MAX_RETRIES})", state="running")
                    time.sleep(sleep_time)
                else:
                    raise LLMUnexpectedError(error_msg) from e

        raise LLMUnexpectedError("Max retries exceeded for generate call.")

    @st.cache_data(ttl=3600, show_spinner=False)
    def count_tokens(_self, prompt: str, system_prompt: str) -> int:
        full_text_for_counting = f"{system_prompt}\n\n{prompt}"
        contents_for_counting = [
            types.Content(role='user', parts=[types.Part(text=full_text_for_counting)])
        ]

        for attempt in range(1, _self.MAX_RETRIES + 1):
            try:
                response = _self.client.models.count_tokens(
                    model=_self.model_name,
                    contents=contents_for_counting
                )
                return response.total_tokens
            except APIError as e:
                error_msg = str(e).encode('utf-8', 'replace').decode('utf-8')
                http_status_code = getattr(e, 'response', None)
                if http_status_code: http_status_code = http_status_code.status_code

                if http_status_code is not None and http_status_code in _self.RETRYABLE_HTTP_CODES and attempt < _self.MAX_RETRIES:
                    backoff_time = min(_self.INITIAL_BACKOFF_SECONDS * (_self.BACKOFF_FACTOR ** (attempt - 1)), _self.MAX_BACKOFF_SECONDS)
                    jitter = random.uniform(0, 0.5 * backoff_time)
                    sleep_time = backoff_time + jitter
                    _self._log_status(f"Gemini API Error (Status: {http_status_code}, Message: {error_msg}) during token count. Retrying in {sleep_time:.2f} seconds... (Attempt {attempt}/{_self.MAX_RETRIES})", state="running")
                    time.sleep(sleep_time)
                else:
                    raise GeminiAPIError(error_msg, http_status_code if http_status_code is not None else getattr(e, 'code', None)) from e
            except Exception as e:
                error_msg = str(e).encode('utf-8', 'replace').decode('utf-8')
                if attempt < _self.MAX_RETRIES:
                    backoff_time = min(_self.INITIAL_BACKOFF_SECONDS * (_self.BACKOFF_FACTOR ** (attempt - 1)), _self.MAX_BACKOFF_SECONDS)
                    jitter = random.uniform(0, 0.5 * backoff_time)
                    sleep_time = backoff_time + jitter
                    _self._log_status(f"Unexpected error: {error_msg} during token count. Retrying in {sleep_time:.2f} seconds... (Attempt {attempt}/{_self.MAX_RETRIES})", state="running")
                    time.sleep(sleep_time)
                else:
                    raise LLMUnexpectedError(error_msg) from e

        raise LLMUnexpectedError("Max retries exceeded for count_tokens call.")

    # recommend_domain is a static method, it doesn't need the cached instance.
    # It should be cached independently if needed.
    @staticmethod
    def recommend_domain(prompt: str, api_key: str, model_name: str = "gemini-2.5-flash-lite") -> str:
        if not prompt or not api_key:
            return "General"

        # Create a temporary provider instance for this static method call
        # This instance is NOT cached by the class-level @st.cache_resource
        # If this method itself needs caching, it should have its own @st.cache_data decorator.
        provider = GeminiProvider(api_key=api_key, model_name=model_name)
        try:
            response, _, _ = provider.generate(
                prompt=f"Analyze the following prompt and determine which domain it best fits into. Choose ONLY from these options: 'Science', 'Business', 'Creative', 'Software Engineering', or 'General' (if none clearly apply).\n\nPrompt: {prompt}\n\nRespond with ONLY the domain name, nothing else. Be concise.",
                system_prompt="You are an expert at categorizing problems into appropriate reasoning domains. Respond with a single word indicating the best domain match.",
                temperature=0.1,
                max_tokens=32
            )
            return response.strip()
        except Exception as e:
            error_msg = str(e).encode('utf-8', 'replace').decode('utf-8')
            logger.error(f"Error in domain recommendation LLM call: {error_msg}")
            return "General"