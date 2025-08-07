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
from abc import ABC, abstractmethod # Import ABC and abstractmethod

# --- Tokenizer Interface and Implementation ---
# Import the Tokenizer ABC and GeminiTokenizer implementation
from src.tokenizers import Tokenizer, GeminiTokenizer 

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

    def __init__(self, api_key: str, model_name: str = "gemini-2.5-flash-lite", tokenizer: Tokenizer = None):
        self._api_key = api_key # Store API key for hashing/equality
        self.model_name = model_name # This is part of the cache key
        self.client = genai.Client(api_key=self._api_key) # Initialize client here
        
        # Use provided tokenizer or create a default GeminiTokenizer
        self.tokenizer = tokenizer or GeminiTokenizer(model_name=self.model_name)
        
    # Define __hash__ and __eq__ for caching to work correctly
    def __hash__(self):
        # Hash based on model_name, API key hash, and tokenizer type hash
        tokenizer_type_hash = hash(type(self.tokenizer))
        return hash((self.model_name, hashlib.sha256(self._api_key.encode()).hexdigest(), tokenizer_type_hash))

    def __eq__(self, other):
        if not isinstance(other, GeminiProvider):
            return NotImplemented
        return (self.model_name == other.model_name and 
                self._api_key == other._api_key and
                type(self.tokenizer) == type(other.tokenizer)) # Compare tokenizer types

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
            logger.warning(f"No pricing information for model '{self.model_name}'. Cost estimation will be $0.")
            return 0.0

        input_cost = (input_tokens / 1000) * costs["input"]
        output_cost = (output_tokens / 1000) * costs["output"]
        return input_cost + output_cost

    # Removed @st.cache_data from generate and count_tokens.
    # The GeminiProvider instance itself is cached via @st.cache_resource.
    # Caching individual method calls on a cached instance can be complex
    # if the instance's internal state (like the tokenizer) is not part of the cache key.
    # Relying on the cached instance is generally sufficient.

    def generate(self, prompt: str, system_prompt: str, temperature: float, max_tokens: int, _status_callback=None) -> tuple[str, int, int]:
        config = types.GenerateContentConfig(
            system_instruction=system_prompt,
            temperature=temperature,
            max_output_tokens=max_tokens
        )

        for attempt in range(1, self.MAX_RETRIES + 1):
            try:
                # Use the tokenizer for all token counts for consistency
                prompt_with_system = f"{system_prompt}\n\n{prompt}"
                input_tokens = self.tokenizer.count_tokens(prompt_with_system)
                
                response = self.client.models.generate_content(
                    model=self.model_name,
                    contents=prompt,
                    config=config
                )
                
                output_tokens = self.tokenizer.count_tokens(response.text)

                return response.text, input_tokens, output_tokens
                
            except APIError as e:
                error_msg = str(e).encode('utf-8', 'replace').decode('utf-8')
                http_status_code = getattr(e, 'response', None)
                if http_status_code: http_status_code = http_status_code.status_code

                if http_status_code is not None and http_status_code in self.RETRYABLE_HTTP_CODES and attempt < self.MAX_RETRIES:
                    backoff_time = min(self.INITIAL_BACKOFF_SECONDS * (self.BACKOFF_FACTOR ** (attempt - 1)), self.MAX_BACKOFF_SECONDS)
                    jitter = random.uniform(0, 0.5 * backoff_time)
                    sleep_time = backoff_time + jitter
                    if _status_callback:
                        _status_callback(message=f"Gemini API Error (Status: {http_status_code}, Message: {error_msg}). Retrying in {sleep_time:.2f} seconds... (Attempt {attempt}/{self.MAX_RETRIES})", state="running")
                    else:
                        logger.warning(f"Gemini API Error (Status: {http_status_code}, Message: {error_msg}). Retrying in {sleep_time:.2f} seconds... (Attempt {attempt}/{self.MAX_RETRIES})")
                    time.sleep(sleep_time)
                else:
                    raise GeminiAPIError(error_msg, http_status_code if http_status_code is not None else getattr(e, 'code', None)) from e
            except Exception as e:
                error_msg = str(e).encode('utf-8', 'replace').decode('utf-8')
                
                if isinstance(e, socket.gaierror):
                    user_friendly_error = (
                        f"Network error during API call: Could not resolve hostname. "
                        f"This might be due to DNS issues or proxy misconfiguration. "
                        f"Details: {error_msg}"
                    )
                    raise LLMProviderError(user_friendly_error) from e
                
                if attempt < self.MAX_RETRIES:
                    backoff_time = min(self.INITIAL_BACKOFF_SECONDS * (self.BACKOFF_FACTOR ** (attempt - 1)), self.MAX_BACKOFF_SECONDS)
                    jitter = random.uniform(0, 0.5 * backoff_time)
                    sleep_time = backoff_time + jitter
                    if _status_callback:
                        _status_callback(message=f"Unexpected error: {error_msg}. Retrying in {sleep_time:.2f} seconds... (Attempt {attempt}/{self.MAX_RETRIES})", state="running")
                    else:
                        logger.error(f"Unexpected error: {error_msg}. Retrying in {sleep_time:.2f} seconds... (Attempt {attempt}/{self.MAX_RETRIES})")
                    time.sleep(sleep_time)
                else:
                    raise LLMUnexpectedError(error_msg) from e

        raise LLMUnexpectedError("Max retries exceeded for generate call.")

    def count_tokens(self, prompt: str, system_prompt: str, _status_callback=None) -> int:
        full_text_for_counting = f"{system_prompt}\n\n{prompt}"
        
        # Use the tokenizer for counting
        try:
            return self.tokenizer.count_tokens(full_text_for_counting)
        except Exception as e:
            # Handle potential errors from the tokenizer itself
            error_msg = f"Error using tokenizer to count tokens: {str(e)}"
            logger.error(error_msg)
            if _status_callback:
                _status_callback(message=f"[red]{error_msg}[/red]", state="error")
            # Re-raise as a provider error
            raise LLMProviderError(error_msg) from e