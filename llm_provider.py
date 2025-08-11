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
# These are now imported from src.exceptions.py
# class LLMProviderError(Exception): ...
# class GeminiAPIError(LLMProviderError): ...
# class LLMUnexpectedError(LLMProviderError): ...

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

    # --- MODIFIED METHOD FOR #7 PRIORITY (EFFICIENCY) ---
    def generate(self, prompt: str, system_prompt: str, temperature: float, max_tokens: int, _status_callback=None, persona_config: Optional[PersonaConfig] = None, intermediate_results: Optional[Dict[str, Any]] = None) -> tuple[str, int, int]:
        """Generate content using the updated SDK pattern with retry logic and token tracking."""
        
        # --- NEW: Dynamic Model Selection ---
        use_pro = self._should_use_pro_model(prompt, persona_config, intermediate_results)
        model_to_use = "gemini-2.5-pro" if use_pro else "gemini-2.5-flash-lite"
        
        current_model_name = self.model_name
        if model_to_use != self.model_name:
            logger.warning(f"Dynamically switching to model '{model_to_use}' for this generation.")
            # Re-instantiate tokenizer if it's model-specific
            if not hasattr(self, 'tokenizer') or self.tokenizer.model_name != model_to_use:
                 self.tokenizer = GeminiTokenizer(model_name=model_to_use, genai_client=self.client)
            current_model_name = model_to_use
        
        config = types.GenerateContentConfig(
            system_instruction=system_prompt,
            temperature=temperature,
            max_output_tokens=max_tokens
        )

        # Use the modified _generate_with_retry method, passing the determined model name
        return self._generate_with__retry(prompt, system_prompt, config, _status_callback, current_model_name)

    def _generate_with_retry(self, prompt: str, system_prompt: str, config: types.GenerateContentConfig, _status_callback=None, model_name_to_use: str = None) -> tuple[str, int, int]:
        """Handles content generation with retry logic, using the specified model."""
        for attempt in range(1, self.MAX_RETRIES + 1):
            try:
                # Use the tokenizer for all token counts for consistency.
                prompt_with_system = f"{system_prompt}\n\n{prompt}" if system_prompt else prompt
                input_tokens = self.tokenizer.count_tokens(prompt_with_system)
                
                response = self.client.models.generate_content(
                    model=model_name_to_use or self.model_name, # Use specified model or default
                    contents=prompt,
                    config=config
                )
                
                generated_text = ""
                if response.candidates and len(response.candidates) > 0:
                    content = response.candidates[0].content
                    if content and content.parts and len(content.parts) > 0:
                        generated_text = content.parts[0].text
                
                output_tokens = self.tokenizer.count_tokens(generated_text)
                
                logger.debug(f"Generated response (input: {input_tokens}, output: {output_tokens} tokens)")
                
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
                
                if should_retry and attempt < self.MAX_RETRIES:
                    backoff_time = min(self.INITIAL_BACKOFF_SECONDS * (self.BACKOFF_FACTOR ** attempt), self.MAX_BACKOFF_SECONDS)
                    jitter = random.uniform(0, 0.5 * backoff_time)
                    sleep_time = backoff_time + jitter
                    
                    log_message = f"Error: {error_msg}. Retrying in {sleep_time:.2f} seconds... (Attempt {attempt}/{self.MAX_RETRIES})"
                    if _status_callback:
                        _status_callback(message=log_message, state="running")
                    else:
                        logger.warning(log_message)
                    time.sleep(sleep_time)
                else:
                    if isinstance(e, APIError):
                        raise GeminiAPIError(error_msg, getattr(e, 'code', None), getattr(e, 'response', None)) from e
                    else:
                        raise LLMUnexpectedError(error_msg) from e
        
        raise LLMUnexpectedError("Max retries exceeded for generate call.")
    # --- END MODIFIED METHOD ---

    # --- MODIFIED METHOD FOR #3 PRIORITY (EFFICIENCY) ---
    def count_tokens(self, prompt: str, system_prompt: str = "", _status_callback=None) -> int:
        """Counts tokens consistently and robustly using the tokenizer."""
        prompt_with_system = f"{system_prompt}\n\n{prompt}" if system_prompt else prompt
        
        try:
            # --- ADDED FOR ROBUSTNESS: Ensure text is properly encoded for token counting ---
            try:
                text_encoded = prompt_with_system.encode('utf-8')
                text_for_tokenizer = text_encoded.decode('utf-8', errors='replace') 
            except UnicodeEncodeError:
                text_for_tokenizer = prompt_with_system.encode('utf-8', errors='replace').decode('utf-8', errors='replace')
                logger.warning("Fixed encoding issues in text for token counting by replacing problematic characters.")
            # --- END ADDED FOR ROBUSTNESS ---
            
            return self.tokenizer.count_tokens(text_for_tokenizer)
        except Exception as e:
            error_msg = f"Error using tokenizer to count tokens: {str(e)}"
            logger.error(error_msg)
            if _status_callback:
                _status_callback(message=f"[red]{error_msg}[/red]", state="error")
            raise LLMProviderError(error_msg) from e
    # --- END MODIFIED METHOD ---

    # --- NEW HELPER METHODS FOR DYNAMIC MODEL SELECTION ---
    def _should_use_pro_model(self, prompt: str, persona_config: Optional[PersonaConfig], intermediate_results: Optional[Dict[str, Any]]) -> bool:
        """
        Determines if the Gemini Pro model should be used based on task complexity,
        persona requirements, or intermediate results.
        """
        if not persona_config: # If persona_config is not provided, use default logic
            persona_config = PersonaConfig(name="Default", system_prompt="", temperature=0.5, max_tokens=1024)
            
        prompt_lower = prompt.lower()
        complexity_score = self._calculate_prompt_complexity(prompt)
        
        # Persona-based preference for Pro model
        pro_personas = ["Impartial_Arbitrator", "Security_Auditor", "Code_Architect", "Constructive_Critic", "DevOps_Engineer", "Test_Engineer"]
        persona_prefers_pro = any(p in persona_config.name for p in pro_personas)
        
        # Quality metric-based preference for Pro model
        low_quality_detected = False
        if intermediate_results:
            quality_metrics = self._extract_quality_metrics_from_results(intermediate_results)
            if quality_metrics.get("reasoning_depth", 1.0) < 0.6 or \
               quality_metrics.get("security_risk_score", 0.0) > 0.7 or \
               quality_metrics.get("code_quality", 1.0) < 0.7:
                low_quality_detected = True
        
        # Use Pro model if complexity is high, persona prefers it, or low quality detected
        if complexity_score > 0.7 or persona_prefers_pro or low_quality_detected:
            return True
        
        return False

    def _calculate_prompt_complexity(self, prompt: str) -> float:
        """Placeholder for calculating prompt complexity score (0.0 to 1.0)."""
        prompt_lower = prompt.lower()
        complexity = 0.0
        length_factor = min(1.0, len(prompt) / 2000.0)
        complexity += length_factor * 0.5
        technical_keywords = ["code", "analyze", "refactor", "algorithm", "architecture", "system", "science", "research", "business", "market", "creative", "art", "security", "test", "deploy", "optimize", "debug"]
        keyword_count = sum(1 for kw in technical_keywords if kw in prompt_lower)
        complexity += (keyword_count / len(technical_keywords) if technical_keywords else 0) * 0.5
        return max(0.0, min(1.0, complexity))

    def _extract_quality_metrics_from_results(self, intermediate_results: Dict[str, Any]) -> Dict[str, float]:
        """Placeholder for extracting quality metrics from intermediate results."""
        # This logic would be similar to core.py's _extract_quality_metrics
        # and would need access to the same data structures or parsing logic.
        # For simplicity, returning default values here.
        return {
            "code_quality": 0.5, "security_risk_score": 0.5, "maintainability_index": 0.5,
            "test_coverage_estimate": 0.5, "reasoning_depth": 0.5, "architectural_coherence": 0.5
        }
    # --- END NEW HELPER METHODS ---