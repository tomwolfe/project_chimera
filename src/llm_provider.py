# src/llm_provider.py
"""
Provides an interface for interacting with the Gemini LLM API,
including retry mechanisms, token counting, cost calculation,
and circuit breaker protection.
"""

from pathlib import Path
import time
import logging
from functools import wraps
from typing import Callable, Any, Dict, Optional, Type, Tuple
import google.genai as genai
from google.genai import types
from google.genai.errors import APIError
import hashlib # Used in __hash__
import random # Used in _generate_with_retry
import socket # Used in _generate_with_retry
import json # Used for structured logging helper
from pydantic import BaseModel, ValidationError # NEW: Import BaseModel and ValidationError

from rich.console import Console

# --- Tokenizer Interface and Implementation ---
from src.tokenizers.base import Tokenizer
from src.tokenizers.gemini_tokenizer import GeminiTokenizer

# --- MODIFICATION: Import PersonaConfig from src.models ---
from src.models import PersonaConfig
# --- END MODIFICATION ---

# --- Custom Exceptions ---
from src.exceptions import (
    ChimeraError,
    LLMProviderError,
    GeminiAPIError,
    LLMUnexpectedError,
    TokenBudgetExceededError,
    CircuitBreakerError,
    SchemaValidationError, # NEW: Import SchemaValidationError
)

# --- NEW IMPORT FOR CIRCUIT BREAKER ---
from src.resilience.circuit_breaker import CircuitBreaker
# --- END NEW IMPORT ---

# --- NEW IMPORT FOR ERROR HANDLER ---
from src.utils.error_handler import handle_errors
# --- END NEW IMPORT ---

from src.config.model_registry import ModelRegistry, ModelSpecification # NEW: Import ModelRegistry
from src.config.settings import ChimeraSettings

# NEW IMPORTS: From src/utils/api_key_validator.py
from src.utils.api_key_validator import validate_gemini_api_key_format, test_gemini_api_key_functional, fetch_api_key # MODIFIED: Import fetch_api_key
import os # NEW: Import os for environment variables


# --- Token Cost Definitions (per 1,000 tokens) ---
TOKEN_COSTS_PER_1K_TOKENS = {
    "gemini-1.5-flash": {
        "input": 0.00008,
        "output": 0.00024,
    },
    "gemini-1.5-pro": {
        "input": 0.0005,
        "output": 0.0015,
    },
}

logger = logging.getLogger(__name__)


class GeminiProvider:
    MAX_RETRIES = 10
    INITIAL_BACKOFF_SECONDS = 1
    BACKOFF_FACTOR = 2
    # MAX_BACKOFF_SECONDS = 60 # REMOVED: Will be set from settings - MODIFIED LINE

    RETRYABLE_ERROR_CODES = {429, 500, 502, 503, 504}
    RETRYABLE_HTTP_CODES = {429, 500, 502, 503, 504}
    # NEW: Add SchemaValidationError to retryable exceptions for internal retry logic
    RETRYABLE_LLM_EXCEPTIONS = (SchemaValidationError,)

    def __init__(
        self,
        api_key: str,
        model_name: str = "gemini-2.5-flash-lite",
        max_retries: int = None,
        max_backoff_seconds: int = None,
        tokenizer: Tokenizer = None,
        rich_console: Optional[Console] = None,
        request_id: Optional[str] = None,
        settings: Optional[Any] = None,
    ):
        self.model_name = model_name
        self.model_registry = ModelRegistry() # Initialize ModelRegistry
        self.rich_console = rich_console or Console(stderr=True)
        self.request_id = request_id
        self._log_extra = {
            "request_id": self.request_id or "N/A"
        }
        self.settings = settings or ChimeraSettings()

        try:
            # Prioritize API key from fetch_api_key, which handles secrets manager and env var fallback
            resolved_api_key = api_key or fetch_api_key() # Use provided key first, then fetch
            if not resolved_api_key:
                logger.critical("GEMINI_API_KEY environment variable is not set. LLMProvider will not function.", extra={'event': 'startup_failure'})
                raise ValueError("LLM API Key is not configured.")
            
            self._api_key = resolved_api_key # Store the resolved API key

            # Validate API key format and functionality early
            is_valid_format, format_message = validate_gemini_api_key_format(self._api_key)
            if not is_valid_format:
                logger.critical(f"Invalid API key format: {format_message}", extra={'event': 'startup_failure'})
                raise ValueError(f"Invalid API key format: {format_message}")
            self.client = genai.Client(api_key=self._api_key)
        except (
            APIError,
            ValueError,
        ) as e:
            error_msg = str(e)
            self._log_with_context(
                "error",
                f"Failed to initialize genai.Client: {error_msg}",
                exc_info=True,
            )
            error_msg_lower = error_msg.lower()
            if (
                "api key not valid" in error_msg_lower
                or "invalid_argument" in error_msg_lower
                or "invalid_api_key" in error_msg_lower
            ):
                raise LLMProviderError(
                    f"Failed to initialize Gemini client: Invalid API Key. Please check your Gemini API Key.",
                    provider_error_code="INVALID_API_KEY",
                    original_exception=e,
                ) from e
            else:
                raise LLMProviderError(
                    f"Failed to initialize Gemini client: {error_msg}",
                    original_exception=e,
                ) from e
        except Exception as e:
            self._log_with_context(
                "error",
                f"An unexpected error occurred during genai.Client initialization: {e}",
                exc_info=True,
            )
            raise LLMProviderError(
                f"Failed to initialize Gemini client unexpectedly: {e}",
                original_exception=e,
            ) from e

        try:
            self.tokenizer = tokenizer or GeminiTokenizer(
                model_name=self.model_name, genai_client=self.client
            )
        except Exception as e:
            self._log_with_context(
                "error", f"Failed to initialize GeminiTokenizer: {e}", exc_info=True
            )
            raise LLMProviderError(
                f"Failed to initialize Gemini tokenizer: {e}", original_exception=e
            ) from e

        self.MAX_RETRIES = (
            max_retries if max_retries is not None else self.settings.max_retries
        )
        self.MAX_BACKOFF_SECONDS = (
            max_backoff_seconds
            if max_backoff_seconds is not None
            else self.settings.max_backoff_seconds
        )

    def __hash__(self):
        tokenizer_type_hash = hash(type(self.tokenizer))
        return hash(
            (
                self.model_name,
                hashlib.sha256(self._api_key.encode()).hexdigest(),
                tokenizer_type_hash,
            )
        )

    def __eq__(self, other):
        if not isinstance(other, GeminiProvider):
            return NotImplemented
        return (
            self.model_name == other.model_name
            and self._api_key == other.api_key
            and type(self.tokenizer) == type(other.tokenizer)
        )

    def get_model_specification(self, model_name: str) -> Optional[ModelSpecification]:
        """Retrieves model specification from the registry."""
        return self.model_registry.get_model(preferred_model_name=model_name)

    def _log_with_context(self, level: str, message: str, **kwargs):
        """Helper to add request context to all logs from this instance."""
        exc_info = kwargs.pop("exc_info", None)
        log_data = {**self._log_extra, **kwargs}
        for k, v in log_data.items():
            try:
                json.dumps({k: v})
            except TypeError:
                log_data[k] = str(v)

        logger_method = getattr(logger, level)
        if exc_info is not None:
            logger_method(message, exc_info=exc_info, extra=log_data)
        else:
            logger_method(message, extra=log_data)

    def _get_pricing_model_name(self) -> str:
        model_spec = self.get_model_specification(self.model_name)
        if model_spec:
            # Map to pricing model names if they differ from actual model names
            if "flash" in model_spec.name: return "gemini-1.5-flash"
            if "pro" in model_spec.name: return "gemini-1.5-pro"
        return "gemini-1.5-flash" # Fallback

    def calculate_usd_cost(self, input_tokens: int, output_tokens: int) -> float:
        pricing_model = self._get_pricing_model_name()
        costs = TOKEN_COSTS_PER_1K_TOKENS.get(pricing_model)
        if not costs:
            self._log_with_context(
                "warning",
                f"No pricing information for model '{self.model_name}'. Cost estimation will be $0.",
            )
            return 0.0

        input_cost = (input_tokens / 1000) * costs["input"]
        output_cost = (output_tokens / 1000) * costs["output"]
        return input_cost + output_cost

    @handle_errors(log_level="ERROR")
    @CircuitBreaker(
        failure_threshold=3,
        recovery_timeout=60,
        expected_exception=(
            APIError,
            CircuitBreakerError,
            SchemaValidationError,
            LLMUnexpectedError,
            GeminiAPIError,
        ),
    )
    def generate(
        self,
        prompt: str,
        system_prompt: str = "",
        output_schema: Optional[Type[BaseModel]] = None,
        temperature: float = 0.7,
        max_tokens: int = 100,
        persona_config: PersonaConfig = None,
        intermediate_results: Dict[str, Any] = None,
        requested_model_name: str = None,
    ) -> tuple[str, int, int, bool]:
        """
        Generates content using the Gemini API, protected by a circuit breaker.
        """

        final_model_to_use = requested_model_name

        # Get max_output_tokens from ModelRegistry for the current model
        current_model_spec = self.get_model_specification(final_model_to_use or self.model_name)
        if current_model_spec:
            self.tokenizer.max_output_tokens = current_model_spec.max_output_tokens

        if final_model_to_use and final_model_to_use != self.model_name:
            self._log_with_context(
                "debug",
                f"Requested model '{final_model_to_use}' differs from provider's initialized model '{self.model_name}'.",
            )
            if (
                not hasattr(self, "tokenizer")
                or self.tokenizer.model_name != final_model_to_use
            ):
                self._log_with_context(
                    "debug",
                    f"Tokenizer model name mismatch. Requested: {final_model_to_use}, Current: {getattr(self.tokenizer, 'model_name', 'N/A')}. Re-initializing tokenizer.",
                )
                try:
                    self.tokenizer = GeminiTokenizer(
                        model_name=final_model_to_use, genai_client=self.client
                    )
                except ValueError as e:
                    self._log_with_context(
                        "error",
                        f"Failed to re-initialize tokenizer for model '{final_model_to_use}': {e}",
                        exc_info=True,
                    )
                    self.tokenizer = GeminiTokenizer(
                        model_name=self.model_name, genai_client=self.client
                    )
                    final_model_to_use = self.model_name
                    self._log_with_context(
                        "warning",
                        f"Falling back to default model '{self.model_name}' due to tokenizer issue.",
                    )
            current_model_name = final_model_to_use
        else:
            current_model_name = self.model_name

        config = types.GenerateContentConfig(
            system_instruction=system_prompt,
            temperature=temperature,
            max_output_tokens=max_tokens,
        )

        generated_text, input_tokens, output_tokens, is_truncated = self._generate_with_retry(
            prompt, system_prompt, config, current_model_name
        )

        # NEW: Enforce schema compliance if a schema is provided
        if output_schema:
            try:
                # Attempt to parse the LLM output into the Pydantic model
                # Use model_validate_json for Pydantic v2, parse_raw for v1
                if hasattr(output_schema, 'model_validate_json'):
                    output_schema.model_validate_json(generated_text)
                else:
                    output_schema.parse_raw(generated_text)
            except ValidationError as ve:
                error_msg = f"LLM output failed schema validation: {ve}"
                self._log_with_context("warning", error_msg, llm_output_snippet=generated_text[:200])
                # Raise SchemaValidationError directly to trigger core.py's retry mechanism
                raise SchemaValidationError(
                    error_type="SCHEMA_VALIDATION_FAILED",
                    field_path="LLM_OUTPUT", # Generic field path for overall schema failure
                    invalid_value=generated_text[:500], # Provide a snippet of the invalid output
                    original_exception=ve
                )
        return generated_text, input_tokens, output_tokens, is_truncated

    def _generate_with_retry(
        self,
        prompt: str,
        system_prompt: str,
        config: types.GenerateContentConfig,
        model_name_to_use: str = None,
    ) -> tuple[str, int, int, bool]:
        """Internal method to handle retries for API calls, called by the circuit breaker."""
        for attempt in range(1, self.MAX_RETRIES + 1):
            try:
                prompt_with_system = (
                    f"{system_prompt}\n\n{prompt}" if system_prompt else prompt
                )
                input_tokens = self.tokenizer.count_tokens(
                    prompt_with_system
                )

                self._log_with_context(
                    "debug",
                    "LLM Prompt Snippet",
                    model=model_name_to_use or self.model_name,
                    input_tokens=input_tokens,
                    prompt_snippet=prompt_with_system[:500] + "..."
                    if len(prompt_with_system) > 500
                    else prompt_with_system,
                )
                self._log_with_context(
                    "info",
                    "LLM Prompt Sent",
                    model=model_name_to_use or self.model_name,
                    temperature=config.temperature,
                    max_output_tokens=config.max_output_tokens,
                    full_system_prompt=system_prompt,
                    full_user_prompt=prompt,
                    input_tokens=input_tokens,
                )

                response = self.client.models.generate_content(
                    model=model_name_to_use or self.model_name,
                    contents=prompt,
                    config=config,
                )

                generated_text = ""
                if response.candidates and len(response.candidates) > 0:
                    content = response.candidates[0].content
                    if content and content.parts and len(content.parts) > 0:
                        generated_text = content.parts[0].text

                output_tokens = self.tokenizer.count_tokens(
                    generated_text
                )
                # MODIFIED: Calculate is_truncated based on actual output tokens vs max_output_tokens
                is_truncated = output_tokens >= config.max_output_tokens * 0.95
                self._log_with_context(
                    "debug",
                    f"Generated response (model: {model_name_to_use}, input: {input_tokens}, output: {output_tokens} tokens)",
                )

                self._log_with_context(
                    "debug",
                    "LLM Response Snippet",
                    model=model_name_to_use or self.model_name,
                    output_tokens=output_tokens,
                    generated_text_snippet=generated_text[:500] + "..."
                    if len(generated_text) > 500
                    else generated_text,
                )
                self._log_with_context(
                    "info",
                    "LLM Response Received",
                    model=model_name_to_use or self.model_name,
                    output_tokens=output_tokens,
                    full_generated_text=generated_text,
                )

                return generated_text, input_tokens, output_tokens, is_truncated

            except self.RETRYABLE_LLM_EXCEPTIONS as e: # NEW: Explicitly catch retryable LLM exceptions
                error_msg = str(e)
                if attempt < self.MAX_RETRIES:
                    backoff_time = min(
                        self.INITIAL_BACKOFF_SECONDS * (self.BACKOFF_FACTOR**attempt),
                        self.MAX_BACKOFF_SECONDS,
                    )
                    jitter = random.uniform(
                        0,
                        0.5
                        * min(
                            self.INITIAL_BACKOFF_SECONDS
                            * (self.BACKOFF_FACTOR**attempt),
                            self.MAX_BACKOFF_SECONDS,
                        ),
                    )
                    sleep_time = backoff_time + jitter

                    log_message = f"Retryable LLM exception: {error_msg}. Retrying in {sleep_time:.2f} seconds... (Attempt {attempt}/{self.MAX_RETRIES})"
                    if self.rich_console:
                        self.rich_console.print(f"[yellow]{log_message}[/yellow]")
                    else:
                        self._log_with_context(
                            "warning", log_message, error_type=type(e).__name__
                        )
                    time.sleep(sleep_time)
                else:
                    self._log_with_context(
                        "error",
                        f"Max retries exceeded for retryable LLM exception: {error_msg}",
                        error_type=type(e).__name__,
                        exc_info=True,
                    )
                    raise e # Re-raise the original retryable exception if retries are exhausted

            except Exception as e:
                error_msg = str(e)
                
                should_retry = False
                error_details = {}

                # Specific handling for common critical errors
                if isinstance(e, APIError):
                    if e.code == 401: # Unauthorized - likely invalid API key
                        raise GeminiAPIError(f"Invalid API Key: {error_msg}", code=e.code, original_exception=e) from e
                    elif e.code == 403: # Forbidden - permissions issue
                        raise GeminiAPIError(f"API Key lacks permissions: {error_msg}", code=e.code, original_exception=e) from e
                    elif e.code == 429: # Rate Limit Exceeded
                        # This is typically handled by the retry logic, but if retries are exhausted, it should be a specific error
                        if attempt >= self.MAX_RETRIES:
                            raise GeminiAPIError(f"Rate Limit Exceeded after retries: {error_msg}", code=e.code, original_exception=e) from e
                        should_retry = True
                    elif e.code == 400 and ("context window exceeded" in error_msg.lower() or "prompt too large" in error_msg.lower()):
                        self._log_with_context(
                            "error",
                            f"LLM context window exceeded: {error_msg}",
                            **error_details,
                        )
                        raise LLMUnexpectedError(
                            f"LLM context window exceeded: {error_msg}",
                            original_exception=e,
                        ) from e
                
                if isinstance(e, APIError): # Existing APIError handling
                    error_details["api_error_code"] = getattr(e, "code", None)
                    if e.code in self.RETRYABLE_ERROR_CODES:
                        should_retry = True
                    http_status_code = getattr(e, "response", None)
                    if http_status_code:
                        error_details["http_status_code"] = http_status_code.status_code
                        if http_status_code.status_code in self.RETRYABLE_HTTP_CODES:
                            should_retry = True
                elif isinstance(e, socket.gaierror): # Existing network error handling
                    should_retry = True
                    error_details["network_error"] = "socket.gaierror"
                elif "context window exceeded" in error_msg.lower() or "prompt too large" in error_msg.lower(): # Existing context window exceeded check
                    self._log_with_context(
                        "error",
                        f"LLM context window exceeded: {error_msg}",
                        **error_details,
                    )
                    raise LLMUnexpectedError(
                        f"LLM context window exceeded: {error_msg}",
                        original_exception=e,
                    ) from e

                if should_retry and attempt < self.MAX_RETRIES:
                    backoff_time = min(
                        self.INITIAL_BACKOFF_SECONDS * (self.BACKOFF_FACTOR**attempt),
                        self.MAX_BACKOFF_SECONDS,
                    )
                    jitter = random.uniform(
                        0,
                        0.5
                        * min(
                            self.INITIAL_BACKOFF_SECONDS
                            * (self.BACKOFF_FACTOR**attempt),
                            self.MAX_BACKOFF_SECONDS,
                        ),
                    )
                    sleep_time = backoff_time + jitter

                    log_message = f"Error: {error_msg}. Retrying in {sleep_time:.2f} seconds... (Attempt {attempt}/{self.MAX_RETRIES})"
                    if self.rich_console:
                        self.rich_console.print(f"[yellow]{log_message}[/yellow]")
                    else:
                        self._log_with_context(
                            "warning", log_message, **error_details
                        )
                    time.sleep(sleep_time)
                else:
                    if isinstance(e, APIError):
                        raise GeminiAPIError(
                            error_msg, getattr(e, "code", None), original_exception=e
                        ) from e
                    else:
                        raise LLMUnexpectedError(error_msg, original_exception=e) from e

            raise LLMUnexpectedError("Max retries exceeded for generate call.")