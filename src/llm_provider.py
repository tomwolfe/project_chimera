# src/llm_provider.py
"""
Provides an interface for interacting with the Gemini LLM API,
including retry mechanisms, token counting, cost calculation,
and circuit breaker protection.
"""

import time
from collections import defaultdict
import streamlit as st
import logging
from functools import wraps
from typing import Callable, Any, Dict, Optional, Type
import google.genai as genai
from google.genai import types
from google.genai.errors import APIError
import hashlib
import random
import socket
import json # Added for structured logging helper

from rich.console import Console

# --- Tokenizer Interface and Implementation ---
from src.tokenizers.base import Tokenizer
from src.tokenizers.gemini_tokenizer import GeminiTokenizer

# --- MODIFICATION: Import PersonaConfig from src.models ---
from src.models import PersonaConfig
# --- END MODIFICATION ---

# --- Custom Exceptions ---
from src.exceptions import ChimeraError, LLMProviderError, GeminiAPIError, LLMUnexpectedError, TokenBudgetExceededError, CircuitBreakerError, SchemaValidationError # Corrected import, added LLMProviderError, CircuitBreakerError, and SchemaValidationError

# --- NEW IMPORT FOR CIRCUIT BREAKER ---
from src.resilience.circuit_breaker import CircuitBreaker
# --- END NEW IMPORT ---

# --- NEW IMPORT FOR ERROR HANDLER ---
from src.utils.error_handler import handle_errors
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
    
    def __init__(self, api_key: str, model_name: str = "gemini-2.5-flash-lite", tokenizer: Tokenizer = None, rich_console: Optional[Console] = None, request_id: Optional[str] = None): # ADD request_id parameter
        self._api_key = api_key
        self.model_name = model_name
        self.rich_console = rich_console or Console(stderr=True)
        self.request_id = request_id # Store request_id
        self._log_extra = {"request_id": self.request_id or "N/A"} # Prepare log extra data for this instance
        
        try:
            self.client = genai.Client(api_key=self._api_key)
        # --- MODIFICATION START ---
        # Catch specific errors related to API key initialization
        except (APIError, ValueError) as e: # Catch APIError for network/auth issues, ValueError for client-side validation
            error_msg = str(e)
            self._log_with_context("error", f"Failed to initialize genai.Client: {error_msg}", exc_info=True) # Use _log_with_context
            error_msg_lower = error_msg.lower()
            if "api key not valid" in error_msg_lower or "invalid_argument" in error_msg_lower or "invalid_api_key" in error_msg_lower:
                raise LLMProviderError(f"Failed to initialize Gemini client: Invalid API Key. Please check your Gemini API Key.", provider_error_code="INVALID_API_KEY", original_exception=e) from e # Pass original_exception
            else:
                raise LLMProviderError(f"Failed to initialize Gemini client: {error_msg}", original_exception=e) from e # Pass original_exception
        except Exception as e: # Catch any other unexpected errors
            self._log_with_context("error", f"An unexpected error occurred during genai.Client initialization: {e}", exc_info=True)
            raise LLMProviderError(f"Failed to initialize Gemini client unexpectedly: {e}", original_exception=e) from e
        # --- MODIFICATION END ---
        
        try:
            self.tokenizer = tokenizer or GeminiTokenizer(model_name=self.model_name, genai_client=self.client)
        except Exception as e:
            self._log_with_context("error", f"Failed to initialize GeminiTokenizer: {e}", exc_info=True) # Use _log_with_context
            raise LLMProviderError(f"Failed to initialize Gemini tokenizer: {e}", original_exception=e) from e # Pass original_exception

    def __hash__(self):
        tokenizer_type_hash = hash(type(self.tokenizer))
        return hash((self.model_name, hashlib.sha256(self._api_key.encode()).hexdigest(), tokenizer_type_hash))

    def __eq__(self, other):
        if not isinstance(other, GeminiProvider):
            return NotImplemented
        return (self.model_name == other.model_name and
                self._api_key == other.api_key and
                type(self.tokenizer) == type(other.tokenizer))

    def _log_with_context(self, level: str, message: str, **kwargs):
        """Helper to add request context to all logs from this instance."""
        exc_info = kwargs.pop('exc_info', None)
        log_data = {**self._log_extra, **kwargs}
        # Convert non-serializable objects to strings for logging to prevent errors
        for k, v in log_data.items():
            try:
                json.dumps({k: v}) 
            except TypeError:
                log_data[k] = str(v) 
        
        logger_method = getattr(logger, level) # Use the module-level logger
        if exc_info is not None:
            logger_method(message, exc_info=exc_info, extra=log_data)
        else:
            logger_method(message, extra=log_data)

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
            self._log_with_context("warning", f"No pricing information for model '{self.model_name}'. Cost estimation will be $0.")
            return 0.0

        input_cost = (input_tokens / 1000) * costs["input"]
        output_cost = (output_tokens / 1000) * costs["output"]
        return input_cost + output_cost

    # ---CIRCUIT BREAKER APPLIED HERE---
    @handle_errors(log_level="ERROR") # Apply the decorator here
    @CircuitBreaker(
        failure_threshold=3,
        recovery_timeout=60, # Wait 60 seconds before trying again
        # Count API errors, circuit breaker rejections, and schema validation errors as failures
        expected_exception=(APIError, CircuitBreakerError, SchemaValidationError, LLMUnexpectedError, GeminiAPIError) # Include custom exceptions
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
            self._log_with_context("debug", f"Requested model '{final_model_to_use}' differs from provider's initialized model '{self.model_name}'.")
            if not hasattr(self, 'tokenizer') or self.tokenizer.model_name != final_model_to_use:
                 self._log_with_context("debug", f"Tokenizer model name mismatch. Requested: {final_model_to_use}, Current: {getattr(self.tokenizer, 'model_name', 'N/A')}. Re-initializing tokenizer.")
                 try:
                     self.tokenizer = GeminiTokenizer(model_name=final_model_to_use, genai_client=self.client)
                 except ValueError as e:
                     self._log_with_context("error", f"Failed to re-initialize tokenizer for model '{final_model_to_use}': {e}", exc_info=True)
                     # Fallback to the original model's tokenizer if re-initialization fails
                     self.tokenizer = GeminiTokenizer(model_name=self.model_name, genai_client=self.client)
                     final_model_to_use = self.model_name
                     self._log_with_context("warning", f"Falling back to default model '{self.model_name}' due to tokenizer issue.")
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
                
                # --- NEW: Log LLM Input ---
                # Log a snippet of the prompt for quick overview
                self._log_with_context("debug", "LLM Prompt Snippet",
                                       model=model_name_to_use or self.model_name,
                                       input_tokens=input_tokens,
                                       prompt_snippet=prompt_with_system[:500] + "..." if len(prompt_with_system) > 500 else prompt_with_system)
                # Log the full system prompt and user prompt for detailed debugging if needed
                self._log_with_context("info", "LLM Prompt Sent",
                                       model=model_name_to_use or self.model_name,
                                       temperature=config.temperature,
                                       max_output_tokens=config.max_output_tokens,
                                       full_system_prompt=system_prompt,
                                       full_user_prompt=prompt,
                                       input_tokens=input_tokens)
                # --- END NEW ---
                
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
                self._log_with_context("debug", f"Generated response (model: {model_name_to_use}, input: {input_tokens}, output: {output_tokens} tokens)")
                
                # --- NEW: Log LLM Output ---
                # Log a snippet of the generated text for quick overview
                self._log_with_context("debug", "LLM Response Snippet",
                                       model=model_name_to_use or self.model_name,
                                       output_tokens=output_tokens,
                                       generated_text_snippet=generated_text[:500] + "..." if len(generated_text) > 500 else generated_text)
                # Log the full generated text for detailed analysis
                self._log_with_context("info", "LLM Response Received",
                                       model=model_name_to_use or self.model_name,
                                       output_tokens=output_tokens,
                                       full_generated_text=generated_text)
                # --- END NEW ---
                
                return generated_text, input_tokens, output_tokens
                
            except Exception as e:
                # Capture error message, replacing potentially problematic characters for logging
                error_msg = str(e).encode('utf-8', 'replace').decode('utf-8')
                
                should_retry = False
                error_details = {} # NEW: Dictionary to hold specific error details for logging

                if isinstance(e, APIError):
                    error_details['api_error_code'] = getattr(e, 'code', None)
                    # Check for specific API error codes that indicate transient issues
                    if e.code in self.RETRYABLE_ERROR_CODES:
                        should_retry = True
                    # Check for retryable HTTP status codes from the response object
                    http_status_code = getattr(e, 'response', None)
                    if http_status_code:
                        error_details['http_status_code'] = http_status_code.status_code
                        if http_status_code.status_code in self.RETRYABLE_HTTP_CODES:
                            should_retry = True
                elif isinstance(e, socket.gaierror): # Network-related errors
                    should_retry = True
                    error_details['network_error'] = "socket.gaierror"
                elif "access denied" in error_msg.lower() or "permission" in error_msg.lower(): # Permission issues might be transient
                    self._log_with_context("warning", f"Access denied or permission error encountered: {error_msg}", **error_details)
                    should_retry = True
                # NEW: Add check for context window exceeded or similar errors
                elif "context window exceeded" in error_msg.lower() or "prompt too large" in error_msg.lower() or "max_input_tokens" in error_msg.lower():
                    self._log_with_context("error", f"LLM context window exceeded: {error_msg}", **error_details)
                    # This is typically not retryable with the same prompt, so we should break
                    raise LLMUnexpectedError(f"LLM context window exceeded: {error_msg}", original_exception=e) from e


                if should_retry and attempt < self.MAX_RETRIES:
                    # Calculate backoff time with jitter
                    backoff_time = min(self.INITIAL_BACKOFF_SECONDS * (self.BACKOFF_FACTOR ** attempt), self.MAX_BACKOFF_SECONDS)
                    jitter = random.uniform(0, 0.5 * min(self.INITIAL_BACKOFF_SECONDS * (self.BACKOFF_FACTOR ** attempt), self.MAX_BACKOFF_SECONDS))
                    sleep_time = backoff_time + jitter
                    
                    log_message = f"Error: {error_msg}. Retrying in {sleep_time:.2f} seconds... (Attempt {attempt}/{self.MAX_RETRIES})"
                    if self.rich_console:
                        self.rich_console.print(f"[yellow]{log_message}[/yellow]")
                    else:
                        self._log_with_context("warning", log_message, **error_details) # Use _log_with_context and include error_details
                    time.sleep(sleep_time)
                else:
                    # If not retrying or max retries reached, raise a specific error
                    if isinstance(e, APIError):
                        raise GeminiAPIError(error_msg, getattr(e, 'code', None), original_exception=e) from e
                    else:
                        raise LLMUnexpectedError(error_msg, original_exception=e) from e
            
            # If loop finishes without returning or raising, it means max retries were exceeded
            raise LLMUnexpectedError("Max retries exceeded for generate call.")

    def estimate_tokens_for_context(self, context_str: str, prompt: str) -> int:
        """Estimates tokens for a context and prompt combination."""
        combined_text = f"{context_str}\n\n{prompt}"
        return self.tokenizer.count_tokens(combined_text)