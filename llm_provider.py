# src/llm_provider.py
import time
from collections import defaultdict
import streamlit as st
import logging
from functools import wraps
from typing import Callable, Any, Dict, Optional
import google.genai as genai # ADDED: This was missing
from google.genai import types # ADDED: This was missing
from google.genai.errors import APIError # ADDED: This was missing
import hashlib # ADDED: This was missing
import random # ADDED: This was missing
import socket # ADDED: This was missing
from rich.console import Console # ADDED: This was missing

# --- Tokenizer Interface and Implementation ---
from src.tokenizers.base import Tokenizer
from src.tokenizers.gemini_tokenizer import GeminiTokenizer

# --- MODIFICATION: Import PersonaConfig from src.models ---
from src.models import PersonaConfig
# --- END MODIFICATION ---

# --- Custom Exceptions ---
from src.exceptions import ChimeraError, LLMProviderError, GeminiAPIError, LLMUnexpectedError, TokenBudgetExceededError, CircuitBreakerError # Corrected import, added LLMProviderError and CircuitBreakerError

# --- NEW IMPORT FOR CIRCUIT BREAKER ---
from src.resilience.circuit_breaker import CircuitBreaker
# --- END NEW IMPORT ---

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

class GeminiProvider:
    MAX_RETRIES = 10
    INITIAL_BACKOFF_SECONDS = 1
    BACKOFF_FACTOR = 2
    MAX_BACKOFF_SECONDS = 60 # Maximum backoff time in seconds
    
    RETRYABLE_ERROR_CODES = {429, 500, 502, 503, 504}
    RETRYABLE_HTTP_CODES = {429, 500, 502, 503, 504}
    
    def __init__(self, api_key: str, model_name: str = "gemini-2.5-flash-lite", tokenizer: Tokenizer = None, rich_console: Optional[Console] = None):
        self._api_key = api_key
        self.model_name = model_name
        self.rich_console = rich_console or Console(stderr=True)
        
        try:
            self.client = genai.Client(api_key=self._api_key)
        except Exception as e:
            error_msg = str(e)
            logger.error(f"Failed to initialize genai.Client: {error_msg}")
            error_msg_lower = error_msg.lower()
            if "api key not valid" in error_msg_lower or "invalid_argument" in error_msg_lower or "invalid_api_key" in error_msg_lower:
                raise LLMProviderError(f"Failed to initialize Gemini client: Invalid API Key. Please check your Gemini API Key.", provider_error_code="INVALID_API_KEY") from e
            else:
                raise LLMProviderError(f"Failed to initialize Gemini client: {error_msg}") from e
        
        try:
            self.tokenizer = tokenizer or GeminiTokenizer(model_name=self.model_name, genai_client=self.client)
        except Exception as e:
            logger.error(f"Failed to initialize GeminiTokenizer: {e}")
            raise LLMProviderError(f"Failed to initialize Gemini tokenizer: {e}") from e

    def __hash__(self):
        tokenizer_type_hash = hash(type(self.tokenizer))
        return hash((self.model_name, hashlib.sha256(self._api_key.encode()).hexdigest(), tokenizer_type_hash))

    def __eq__(self, other):
        if not isinstance(other, GeminiProvider):
            return NotImplemented
        return (self.model_name == other.model_name and 
                self._api_key == other.api_key and
                type(self.tokenizer) == type(other.tokenizer))

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

    # --- CIRCUIT BREAKER APPLIED HERE ---
    @CircuitBreaker(
        failure_threshold=3, 
        recovery_timeout=60, # Wait 60 seconds before trying again
        # Count API errors and previous circuit breaker rejections as failures
        expected_exception=(APIError, CircuitBreakerError) 
    )
    def generate(self, prompt: str, system_prompt: str, temperature: float, max_tokens: int, persona_config: PersonaConfig = None, intermediate_results: Dict[str, Any] = None, requested_model_name: str = None) -> tuple[str, int, int]:
        """
        Generates content using the Gemini API, protected by a circuit breaker.
        
        If the circuit breaker is OPEN, this method will raise CircuitBreakerError
        before the actual API call is made. If HALF-OPEN, it allows one attempt.
        If CLOSED, it proceeds with the API call.
        """
        
        final_model_to_use = requested_model_name
        
        if final_model_to_use and final_model_to_use != self.model_name:
            logger.debug(f"Requested model '{final_model_to_use}' differs from provider's initialized model '{self.model_name}'.")
            if not hasattr(self, 'tokenizer') or self.tokenizer.model_name != final_model_to_use:
                 logger.debug(f"Tokenizer model name mismatch. Requested: {final_model_to_use}, Current: {getattr(self.tokenizer, 'model_name', 'N/A')}. Re-initializing tokenizer.")
                 try:
                     self.tokenizer = GeminiTokenizer(model_name=final_model_to_use, genai_client=self.client)
                 except ValueError as e:
                     logger.error(f"Failed to re-initialize tokenizer for model '{final_model_to_use}': {e}")
                     # Fallback to the original model's tokenizer if re-initialization fails
                     self.tokenizer = GeminiTokenizer(model_name=self.model_name, genai_client=self.client)
                     final_model_to_use = self.model_name
                     logger.warning(f"Falling back to default model '{self.model_name}' due to tokenizer issue.")
            current_model_name = final_model_to_use
        else:
            current_model_name = self.model_name

        config = types.GenerateContentConfig(
            system_instruction=system_prompt,
            temperature=temperature,
            max_output_tokens=max_tokens
        )
        
        # This is the actual call that might fail and trigger the circuit breaker
        return self._generate_with_retry(prompt, system_prompt, config, current_model_name)
    # --- END CIRCUIT BREAKER APPLIED HERE ---

    def _generate_with_retry(self, prompt: str, system_prompt: str, config: types.GenerateContentConfig, model_name_to_use: str = None) -> tuple[str, int, int]:
        """Internal method to handle retries for API calls, called by the circuit breaker."""
        for attempt in range(1, self.MAX_RETRIES + 1):
            try:
                # Construct the full prompt including system instruction if provided
                prompt_with_system = f"{system_prompt}\n\n{prompt}" if system_prompt else prompt
                input_tokens = self.tokenizer.count_tokens(prompt_with_system) # USE self.tokenizer.count_tokens
                
                # Make the actual API call
                response = self.client.models.generate_content(
                    model=model_name_to_use or self.model_name,
                    contents=prompt,
                    config=config
                )
                
                generated_text = ""
                if response.candidates and len(response.candidates) > 0:
                    content = response.candidates[0].content
                    if content and content.parts and len(content.parts) > 0:
                        generated_text = content.parts[0].text
                
                output_tokens = self.tokenizer.count_tokens(generated_text) # USE self.tokenizer.count_tokens
                logger.debug(f"Generated response (model: {model_name_to_use}, input: {input_tokens}, output: {output_tokens} tokens)")
                
                return generated_text, input_tokens, output_tokens
                
            except Exception as e:
                # Capture error message, replacing potentially problematic characters for logging
                error_msg = str(e).encode('utf-8', 'replace').decode('utf-8')
                
                should_retry = False
                if isinstance(e, APIError):
                    # Check for specific API error codes that indicate transient issues
                    if e.code in self.RETRYABLE_ERROR_CODES:
                        should_retry = True
                    # Check for retryable HTTP status codes from the response object
                    http_status_code = getattr(e, 'response', None)
                    if http_status_code and http_status_code.status_code in self.RETRYABLE_HTTP_CODES:
                        should_retry = True
                elif isinstance(e, socket.gaierror): # Network-related errors
                    should_retry = True
                elif "access denied" in error_msg.lower() or "permission" in error_msg.lower(): # Permission issues might be transient
                    logger.warning(f"Access denied or permission error encountered: {error_msg}")
                    should_retry = True

                if should_retry and attempt < self.MAX_RETRIES:
                    # Calculate backoff time with jitter
                    backoff_time = min(self.INITIAL_BACKOFF_SECONDS * (self.BACKOFF_FACTOR ** attempt), self.MAX_BACKOFF_SECONDS)
                    jitter = random.uniform(0, 0.5 * min(self.INITIAL_BACKOFF_SECONDS * (self.BACKOFF_FACTOR ** attempt), self.MAX_BACKOFF_SECONDS))
                    sleep_time = backoff_time + jitter
                    
                    log_message = f"[yellow]Error: {error_msg}. Retrying in {sleep_time:.2f} seconds... (Attempt {attempt}/{self.MAX_RETRIES})[/yellow]"
                    if self.rich_console:
                        self.rich_console.print(log_message)
                    else:
                        logger.warning(log_message)
                    time.sleep(sleep_time)
                else:
                    # If not retrying or max retries reached, raise a specific error
                    if isinstance(e, APIError):
                        raise GeminiAPIError(error_msg, getattr(e, 'code', None)) from e
                    else:
                        raise LLMUnexpectedError(error_msg) from e
            
            # If loop finishes without returning or raising, it means max retries were exceeded
            raise LLMUnexpectedError("Max retries exceeded for generate call.")

    # REMOVE THIS METHOD: It's redundant and uses a simple dict cache.
    # The tokenizer object (self.tokenizer) should be the single source of truth for token counting.
    # def count_tokens(self, text: str) -> int:
    #     """Counts tokens in the given text using the Gemini API, with caching."""
    #     if not text:
    #         return 0
    #         
    #     # Use a hash of the text for cache key to avoid issues with identical content
    #     text_hash = hash(text)
    #     
    #     if text_hash in self._cache:
    #         logger.debug(f"Cache hit for token count (hash: {text_hash}).")
    #         return self._cache[text_hash]
    #     
    #     try:
    #         # Ensure text is properly encoded for the API call, replacing errors
    #         try:
    #             text_encoded = text.encode('utf-8')
    #             text_for_api = text_encoded.decode('utf-8', errors='replace')
    #         except UnicodeEncodeError:
    #             # Fallback if encoding fails, replace problematic characters
    #             text_for_api = text.encode('utf-8', errors='replace').decode('utf-8', errors='replace')
    #             logger.warning("Fixed encoding issues in text for token counting by replacing problematic characters.")
    #         
    #         # Use the count_tokens API to get token count
    #         response = self.genai_client.models.count_tokens(
    #             model=self.model_name,
    #             contents=text_for_api
    #         )
    #         tokens = response.total_tokens
    #         
    #         # Cache the result
    #         self._cache[text_hash] = tokens
    #         logger.debug(f"Token count for text (hash: {text_hash}) is {tokens}. Stored in cache.")
    #         return tokens
    #         
    #     except Exception as e:
    #         logger.error(f"Gemini token counting failed for model '{self.model_name}': {str(e)}")
    #         # Fallback to approximate count if API fails, to prevent crashing the budget calculation
    #         # IMPROVED FALLBACK: Use a more accurate approximation (e.g., 4 chars per token)
    #         approx_tokens = max(1, int(len(text) / 4))  # More accurate fallback
    #         logger.warning(f"Falling back to improved token approximation ({approx_tokens}) due to error: {str(e)}")
    #         return approx_tokens

    def estimate_tokens_for_context(self, context_str: str, prompt: str) -> int:
        """Estimates tokens for a context and prompt combination."""
        combined_text = f"{context_str}\n\n{prompt}"
        return self.tokenizer.count_tokens(combined_text) # USE self.tokenizer.count_tokens