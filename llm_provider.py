# src/llm_provider.py
import streamlit as st
import google.genai as genai
from google.genai import types
from google.genai.errors import APIError
import time
import hashlib
import re
import socket
import abc
from collections import defaultdict
from typing import List, Dict, Tuple, Any, Callable, Optional, Type
import logging
from pathlib import Path
import random # Needed for backoff jitter

# --- Tokenizer Interface and Implementation ---
# Import the Tokenizer ABC and GeminiTokenizer implementation
from src.tokenizers.base import Tokenizer
from src.tokenizers.gemini_tokenizer import GeminiTokenizer

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
    
    # Define retryable error codes and HTTP status codes at class level
    RETRYABLE_ERROR_CODES = {429, 500, 502, 503, 504} # For API errors that are retryable
    RETRYABLE_HTTP_CODES = {429, 500, 502, 503, 504} # For HTTP errors that are retryable

    def __init__(self, api_key: str, model_name: str = "gemini-2.5-flash-lite", tokenizer: Tokenizer = None):
        self._api_key = api_key # Store API key for hashing/equality
        self.model_name = model_name # This is part of the cache key
        # Initialize client using the correct SDK pattern
        self.client = genai.Client(api_key=self._api_key)
        
        # Use provided tokenizer or create a default GeminiTokenizer
        # FIX: Pass the genai_client instance to the GeminiTokenizer
        self.tokenizer = tokenizer or GeminiTokenizer(model_name=self.model_name, genai_client=self.client)
        
    # Define __hash__ and __eq__ for caching to work correctly
    def __hash__(self):
        # Hash based on model_name, API key hash, and tokenizer type hash
        tokenizer_type_hash = hash(type(self.tokenizer))
        return hash((self.model_name, hashlib.sha256(self._api_key.encode()).hexdigest(), tokenizer_type_hash))

    def __eq__(self, other):
        if not isinstance(other, GeminiProvider):
            return NotImplemented
        return (self.model_name == other.model_name and 
                self._api_key == other.api_key and
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

    # --- MODIFIED METHOD FOR #3 PRIORITY (EFFICIENCY) ---
    def generate(self, prompt: str, system_prompt: str, temperature: float, max_tokens: int, _status_callback=None) -> tuple[str, int, int]:
        """Generate content using the updated SDK pattern with retry logic and token tracking."""
        config = types.GenerateContentConfig(
            system_instruction=system_prompt,
            temperature=temperature,
            max_output_tokens=max_tokens
        )

        # Use the modified _generate_with_retry method
        return self._generate_with_retry(prompt, system_prompt, config, _status_callback)

    def _generate_with_retry(self, prompt: str, system_prompt: str, config: types.GenerateContentConfig, _status_callback=None) -> tuple[str, int, int]:
        """Handles content generation with retry logic."""
        for attempt in range(1, self.MAX_RETRIES + 1):
            try:
                # Use the tokenizer for all token counts for consistency.
                # Combine system prompt and prompt for input token calculation.
                prompt_with_system = f"{system_prompt}\n\n{prompt}" if system_prompt else prompt
                input_tokens = self.tokenizer.count_tokens(prompt_with_system)
                
                # CORRECTED: Use client.models.generate_content pattern
                response = self.client.models.generate_content(
                    model=self.model_name,
                    contents=prompt,
                    config=config
                )
                
                # Extract text response
                generated_text = ""
                if response.candidates and len(response.candidates) > 0:
                    content = response.candidates[0].content
                    if content and content.parts and len(content.parts) > 0:
                        generated_text = content.parts[0].text
                
                # Get output tokens using the tokenizer.
                output_tokens = self.tokenizer.count_tokens(generated_text)

                # Log token usage
                logger.debug(f"Generated response (input: {input_tokens}, output: {output_tokens} tokens)")
                
                return generated_text, input_tokens, output_tokens
                
            except Exception as e:
                error_msg = str(e).encode('utf-8', 'replace').decode('utf-8')
                
                # --- MODIFIED FOR #4 PRIORITY (MAINTAINABILITY) ---
                # Standardize retry decision logic and backoff calculation.
                should_retry = False
                # Check for specific retryable error codes from Gemini API errors
                if isinstance(e, APIError):
                    if e.code in self.RETRYABLE_ERROR_CODES:
                        should_retry = True
                    # Also check HTTP status code if available in the APIError response
                    http_status_code = getattr(e, 'response', None)
                    if http_status_code and http_status_code.status_code in self.RETRYABLE_HTTP_CODES:
                        should_retry = True
                # Handle network resolution errors (DNS issues)
                elif isinstance(e, socket.gaierror):
                    should_retry = True
                
                if should_retry and attempt < self.MAX_RETRIES:
                    # Calculate backoff time with jitter for exponential backoff
                    # Use 'attempt' directly for exponential backoff calculation
                    backoff_time = min(self.INITIAL_BACKOFF_SECONDS * (self.BACKOFF_FACTOR ** attempt), self.MAX_BACKOFF_SECONDS)
                    jitter = random.uniform(0, 0.5 * backoff_time) # Add random jitter
                    sleep_time = backoff_time + jitter
                    
                    log_message = f"Error: {error_msg}. Retrying in {sleep_time:.2f} seconds... (Attempt {attempt}/{self.MAX_RETRIES})"
                    if _status_callback:
                        _status_callback(message=log_message, state="running")
                    else:
                        logger.warning(log_message)
                    time.sleep(sleep_time)
                else:
                    # If not retryable or max retries exceeded, raise a specific error
                    if isinstance(e, APIError):
                        raise GeminiAPIError(error_msg, getattr(e, 'code', None)) from e
                    else:
                        raise LLMUnexpectedError(error_msg) from e
        
        # If loop finishes without returning, it means max retries were exceeded
        raise LLMUnexpectedError("Max retries exceeded for generate call.")
    # --- END MODIFIED METHOD ---

    # --- MODIFIED METHOD FOR #3 PRIORITY (EFFICIENCY) ---
    def count_tokens(self, prompt: str, system_prompt: str = "", _status_callback=None) -> int:
        """Counts tokens consistently and robustly using the tokenizer."""
        # Combine system prompt and prompt for accurate token counting.
        # Ensure consistent handling of empty system prompts.
        prompt_with_system = f"{system_prompt}\n\n{prompt}" if system_prompt else prompt
        
        # Use the tokenizer for counting.
        try:
            # --- ADDED FOR ROBUSTNESS: Ensure text is properly encoded for token counting ---
            # This prevents errors with special characters or unexpected byte sequences.
            try:
                # Encode to UTF-8, then decode back to ensure valid string for API.
                # 'replace' error handling ensures that any problematic characters are handled gracefully.
                text_encoded = prompt_with_system.encode('utf-8')
                text_for_tokenizer = text_encoded.decode('utf-8', errors='replace') 
            except UnicodeEncodeError:
                # Fallback if encoding itself fails, replace problematic chars
                text_for_tokenizer = prompt_with_system.encode('utf-8', errors='replace').decode('utf-8', errors='replace')
                logger.warning("Fixed encoding issues in text for token counting by replacing problematic characters.")
            # --- END ADDED FOR ROBUSTNESS ---
            
            return self.tokenizer.count_tokens(text_for_tokenizer)
        except Exception as e:
            # Handle potential errors from the tokenizer itself
            error_msg = f"Error using tokenizer to count tokens: {str(e)}"
            logger.error(error_msg)
            if _status_callback:
                _status_callback(message=f"[red]{error_msg}[/red]", state="error")
            # Re-raise as a provider error to be handled by core.py
            raise LLMProviderError(error_msg) from e
    # --- END MODIFIED METHOD ---