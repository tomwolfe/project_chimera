# src/llm_provider.py
import streamlit as st
import google.genai as genai
from google.genai import types
from google.genai.errors import APIError # Import APIError
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
from src.tokenizers.base import Tokenizer
from src.tokenizers.gemini_tokenizer import GeminiTokenizer

# --- MODIFICATION: Import PersonaConfig from src.models ---
# This import was missing, causing the NameError in the GeminiProvider class definition.
from src.models import PersonaConfig
# --- END MODIFICATION ---

# --- Custom Exceptions ---
# Correcting the inheritance for LLMProviderError
from src.exceptions import ChimeraError # Import ChimeraError

class LLMProviderError(ChimeraError): # Inherit from ChimeraError
    """Base exception for LLM provider errors."""
    def __init__(self, message: str, provider_error_code: Any = None, details: Optional[dict] = None):
        full_details = (details or {}).copy()
        full_details["provider_error_code"] = provider_error_code
        super().__init__(message, details=full_details)

class GeminiAPIError(LLMProviderError):
    """Specific exception for Gemini API errors."""
    def __init__(self, message: str, code: int = None, response_details: Any = None):
        super().__init__(message, provider_error_code=code, details={"response_details": response_details})

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
# --- FIX START ---
# Removed @st.cache_resource decorator as genai.Client is not serializable/cacheable
# @st.cache_resource
class GeminiProvider:
# --- FIX END ---
    # Retry parameters
    MAX_RETRIES = 10
    INITIAL_BACKOFF_SECONDS = 1
    BACKOFF_FACTOR = 2
    MAX_BACKOFF_SECONDS = 60 # Maximum backoff time in seconds
    
    # Define retryable error codes and HTTP status codes at class level
    RETRYABLE_ERROR_CODES = {429, 500, 502, 503, 504} # For API errors that are retryable
    RETRYABLE_HTTP_CODES = {429, 500, 502, 503, 504} # For HTTP errors that are retryable

    def __init__(self, api_key: str, model_name: str = "gemini-2.5-flash-lite", tokenizer: Tokenizer = None):
        self._api_key = api_key
        self.model_name = model_name
        
        # --- FIX START ---
        try:
            # Attempt to initialize the genai client. This is where API key validation might fail.
            self.client = genai.Client(api_key=self._api_key)
        except Exception as e: # Catching a broad exception here to handle various potential init failures
            logger.error(f"Failed to initialize genai.Client: {e}")
            # Raise a specific LLMProviderError indicating the issue.
            # Check if the error message suggests an invalid API key.
            error_msg_lower = str(e).lower()
            if "api key not valid" in error_msg_lower or "invalid_argument" in error_msg_lower or "invalid_api_key" in error_msg_lower:
                raise LLMProviderError(f"Failed to initialize Gemini client: Invalid API Key. Please check your Gemini API Key.", provider_error_code="INVALID_API_KEY") from e
            else:
                raise LLMProviderError(f"Failed to initialize Gemini client: {e}") from e
        
        # Ensure tokenizer is initialized only if client is successful
        try:
            self.tokenizer = tokenizer or GeminiTokenizer(model_name=self.model_name, genai_client=self.client)
        except Exception as e:
            logger.error(f"Failed to initialize GeminiTokenizer: {e}")
            # Raise an error if tokenizer initialization fails
            raise LLMProviderError(f"Failed to initialize Gemini tokenizer: {e}") from e
        # --- FIX END ---

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

    def generate(self, prompt: str, system_prompt: str, temperature: float, max_tokens: int, _status_callback=None, persona_config: PersonaConfig = None, intermediate_results: Dict[str, Any] = None, requested_model_name: str = None) -> tuple[str, int, int]:
        """
        Generate content using the specified model, with retry logic and token tracking.
        Prioritizes requested_model_name, falling back to provider's default if needed.
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
        
        return self._generate_with_retry(prompt, system_prompt, config, _status_callback, current_model_name)

    def _generate_with_retry(self, prompt: str, system_prompt: str, config: types.GenerateContentConfig, _status_callback=None, model_name_to_use: str = None) -> tuple[str, int, int]:
        """Handles content generation with retry logic, using the specified model."""
        for attempt in range(1, self.MAX_RETRIES + 1):
            try:
                prompt_with_system = f"{system_prompt}\n\n{prompt}" if system_prompt else prompt
                input_tokens = self.tokenizer.count_tokens(prompt_with_system)
                
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
                
                output_tokens = self.tokenizer.count_tokens(generated_text)
                logger.debug(f"Generated response (model: {model_name_to_use}, input: {input_tokens}, output: {output_tokens} tokens)")
                
                return generated_text, input_tokens, output_tokens
                
            except Exception as e:
                error_msg = str(e).encode('utf-8', 'replace').decode('utf-8')
                
                should_retry = False
                if isinstance(e, APIError):
                    if e.code in self.RETRYABLE_ERROR_CODES:
                        should_retry = True
                    http_status_code = getattr(e, 'response', None)
                    if http_status_code and http_status_code.status_code in self.RETRYABLE_HTTP_CODES:
                        should_retry = True
                elif isinstance(e, socket.gaierror):
                    should_retry = True
                elif "access denied" in error_msg.lower() or "permission" in error_msg.lower():
                    logger.warning(f"Access denied or permission error encountered: {error_msg}")
                    should_retry = True

                if should_retry and attempt < self.MAX_RETRIES:
                    backoff_time = min(self.INITIAL_BACKOFF_SECONDS * (self.BACKOFF_FACTOR ** attempt), self.MAX_BACKOFF_SECONDS)
                    jitter = random.uniform(0, 0.5 * min(self.INITIAL_BACKOFF_SECONDS * (self.BACKOFF_FACTOR ** attempt), self.MAX_BACKOFF_SECONDS))
                    sleep_time = backoff_time + jitter
                    
                    log_message = f"Error: {error_msg}. Retrying in {sleep_time:.2f} seconds... (Attempt {attempt}/{self.MAX_RETRIES})"
                    if _status_callback:
                        _status_callback(message=log_message, state="running")
                    else:
                        logger.warning(log_message)
                    time.sleep(sleep_time)
                else:
                    if isinstance(e, APIError):
                        raise GeminiAPIError(error_msg, getattr(e, 'code', None)) from e
                    else:
                        raise LLMUnexpectedError(error_msg) from e
            
            raise LLMUnexpectedError("Max retries exceeded for generate call.")

    def count_tokens(self, prompt: str, system_prompt: str = "", _status_callback=None) -> int:
        """Counts tokens consistently and robustly using the tokenizer."""
        prompt_with_system = f"{system_prompt}\n\n{prompt}" if system_prompt else prompt
        
        try:
            try:
                text_encoded = prompt_with_system.encode('utf-8')
                text_for_tokenizer = text_encoded.decode('utf-8', errors='replace') 
            except UnicodeEncodeError:
                text_for_tokenizer = prompt_with_system.encode('utf-8', errors='replace').decode('utf-8', errors='replace')
                logger.warning("Fixed encoding issues in text for token counting by replacing problematic characters.")
            
            return self.tokenizer.count_tokens(text_for_tokenizer)
        except Exception as e:
            error_msg = f"Error using tokenizer to count tokens: {str(e)}"
            logger.error(error_msg)
            if _status_callback:
                _status_callback(message=f"[red]{error_msg}[/red]", state="error")
            raise LLMProviderError(error_msg) from e

    def _calculate_prompt_complexity(self, prompt: str) -> float:
        """Placeholder for calculating prompt complexity score (0.0 to 1.0)."""
        prompt_lower = prompt.lower()
        complexity = 0.0
        length_factor = min(1.0, len(prompt) / 2000.0)
        complexity += length_factor * 0.5
        technical_keywords = [
            "code", "analyze", "refactor", "algorithm", "architecture", "system",
            "science", "research", "business", "market", "creative", "art",
            "security", "test", "deploy", "optimize", "debug"
        ]
        keyword_count = sum(1 for kw in technical_keywords if kw in prompt_lower)
        complexity += (keyword_count / len(technical_keywords) if technical_keywords else 0) * 0.5
        return max(0.0, min(1.0, complexity))

    def _extract_quality_metrics_from_results(self, intermediate_results: Dict[str, Any]) -> Dict[str, float]:
        """Placeholder for extracting quality metrics from intermediate results."""
        metrics = {
            "code_quality": 0.5, "security_risk_score": 0.5, "maintainability_index": 0.5,
            "test_coverage_estimate": 0.5, "reasoning_depth": 0.5, "architectural_coherence": 0.5
        }
        if "Context_Aware_Assistant_Output" in intermediate_results:
            caa_output = intermediate_results["Context_Aware_Assistant_Output"]
            if isinstance(caa_output, dict) and "quality_metrics" in caa_output and isinstance(caa_output["quality_metrics"], dict):
                quality_metrics_from_caa = caa_output["quality_metrics"]
                for metric_name, value in quality_metrics_from_caa.items():
                    if metric_name in metrics:
                        metrics[metric_name] = max(metrics[metric_name], value)
        for key in metrics:
            metrics[key] = max(0.0, min(1.0, metrics[key]))
        logger.debug(f"Extracted quality metrics (placeholder): {metrics}")
        return metrics