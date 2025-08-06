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
from pathlib import Path
import socket

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

    # MODIFIED: Removed _status_callback from __init__
    def __init__(self, api_key: str, model_name: str = "gemini-2.5-flash-lite"):
        self._api_key = api_key # Store API key for hashing/equality
        self.model_name = model_name # This is part of the cache key
        self.client = genai.Client(api_key=self._api_key) # Initialize client here
        
    # Define __hash__ and __eq__ for caching to work correctly
    def __hash__(self):
        # Hash based on model_name and a hash of the API key (not the key itself)
        return hash((self.model_name, hashlib.sha256(self._api_key.encode()).hexdigest()))

    def __eq__(self, other):
        if not isinstance(other, GeminiProvider):
            return NotImplemented
        return self.model_name == other.model_name and self._api_key == other._api_key

    # MODIFIED: Added _status_callback as an argument
    def _log_status(self, message: str, _status_callback=None, state: str = "running", expanded: bool = True,
                    current_total_tokens: int = 0, current_total_cost: float = 0.0,
                    estimated_next_step_tokens: int = 0, estimated_next_step_cost: float = 0.0):
        if _status_callback: # MODIFIED: Use the passed argument
            _status_callback( # MODIFIED: Call the passed argument
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

    @st.cache_data(ttl=3600, show_spinner=False)
    # MODIFIED: Added _status_callback as an argument
    def generate(_self, prompt: str, system_prompt: str, temperature: float, max_tokens: int, _status_callback=None) -> tuple[str, int, int]:
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
                    # MODIFIED: Pass _status_callback
                    _self._log_status(f"Gemini API Error (Status: {http_status_code}, Message: {error_msg}). Retrying in {sleep_time:.2f} seconds... (Attempt {attempt}/{_self.MAX_RETRIES})", _status_callback=_status_callback, state="running")
                    time.sleep(sleep_time)
                else:
                    raise GeminiAPIError(error_msg, http_status_code if http_status_code is not None else getattr(e, 'code', None)) from e
            except Exception as e:
                error_msg = str(e).encode('utf-8', 'replace').decode('utf-8')
                
                # Check if the error is a DNS resolution error (nodename nor servname provided)
                if isinstance(e, socket.gaierror):
                    user_friendly_error = (
                        f"Network error during API call: Could not resolve hostname. "
                        f"This might be due to DNS issues or proxy misconfiguration. "
                        f"Details: {error_msg}"
                    )
                    # Raise a specific error type with a more informative message
                    raise LLMProviderError(user_friendly_error) from e
                
                # Handle other exceptions as before
                if attempt < _self.MAX_RETRIES:
                    backoff_time = min(_self.INITIAL_BACKOFF_SECONDS * (_self.BACKOFF_FACTOR ** (attempt - 1)), _self.MAX_BACKOFF_SECONDS)
                    jitter = random.uniform(0, 0.5 * backoff_time)
                    sleep_time = backoff_time + jitter
                    # MODIFIED: Pass _status_callback
                    _self._log_status(f"Unexpected error: {error_msg}. Retrying in {sleep_time:.2f} seconds... (Attempt {attempt}/{_self.MAX_RETRIES})", _status_callback=_status_callback, state="running")
                    time.sleep(sleep_time)
                else:
                    raise LLMUnexpectedError(error_msg) from e

        raise LLMUnexpectedError("Max retries exceeded for generate call.")

    @st.cache_data(ttl=3600, show_spinner=False)
    # MODIFIED: Added _status_callback as an argument
    def count_tokens(_self, prompt: str, system_prompt: str, _status_callback=None) -> int:
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
                    # MODIFIED: Pass _status_callback
                    _self._log_status(f"Gemini API Error (Status: {http_status_code}, Message: {error_msg}) during token count. Retrying in {sleep_time:.2f} seconds... (Attempt {attempt}/{_self.MAX_RETRIES})", _status_callback=_status_callback, state="running")
                    time.sleep(sleep_time)
                else:
                    raise GeminiAPIError(error_msg, http_status_code if http_status_code is not None else getattr(e, 'code', None)) from e
            except Exception as e:
                error_msg = str(e).encode('utf-8', 'replace').decode('utf-8')
                
                # Check if the error is a DNS resolution error (nodename nor servname provided)
                if isinstance(e, socket.gaierror):
                    user_friendly_error = (
                        f"Network error during token count: Could not resolve hostname. "
                        f"This might be due to DNS issues or proxy misconfiguration. "
                        f"Details: {error_msg}"
                    )
                    # Raise a specific error type with a more informative message
                    raise LLMProviderError(user_friendly_error) from e
                
                # Handle other exceptions as before
                if attempt < _self.MAX_RETRIES:
                    backoff_time = min(_self.INITIAL_BACKOFF_SECONDS * (_self.BACKOFF_FACTOR ** (attempt - 1)), _self.MAX_BACKOFF_SECONDS)
                    jitter = random.uniform(0, 0.5 * backoff_time)
                    sleep_time = backoff_time + jitter
                    # MODIFIED: Pass _status_callback
                    _self._log_status(f"Unexpected error: {error_msg}. Retrying in {sleep_time:.2f} seconds... (Attempt {attempt}/{_self.MAX_RETRIES})", _status_callback=_status_callback, state="running")
                    time.sleep(sleep_time)
                else:
                    raise LLMUnexpectedError(error_msg) from e

        raise LLMUnexpectedError("Max retries exceeded for count_tokens call.")