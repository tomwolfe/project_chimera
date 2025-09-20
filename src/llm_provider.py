# src/llm_provider.py
"""Provides an interface for interacting with the Gemini LLM API,
including retry mechanisms, token counting, cost calculation,
and circuit breaker protection.
"""

import hashlib
import json
import logging
import socket
from typing import Any, Dict, Optional, Type

# --- Tokenizer Interface and Implementation ---
import tiktoken
from google import genai
from google.genai import types as genai_types
from google.genai.errors import APIError, ServerError
from pydantic import BaseModel, ValidationError
from rich.console import Console

# --- NEW IMPORTS FOR RETRY AND RATE LIMIT ---
from tenacity import (
    RetryError,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

# --- END NEW IMPORT ---
# FIX: Import ModelSpecification explicitly
from src.config.model_registry import ModelRegistry, ModelSpecification
from src.config.settings import ChimeraSettings

# --- Custom Exceptions ---
from src.exceptions import (
    CircuitBreakerError,
    GeminiAPIError,
    LLMProviderError,
    LLMProviderRequestError,
    LLMProviderResponseError,
    LLMUnexpectedError,
    SchemaValidationError,
    TokenBudgetExceededError,
)
from src.llm_tokenizers.base import Tokenizer
from src.llm_tokenizers.gemini_tokenizer import GeminiTokenizer
from src.middleware.rate_limiter import RateLimitExceededError

# NEW: Import LLMOutput and SelfImprovementAnalysisOutputV1 from src.models
from src.models import LLMOutput, SelfImprovementAnalysisOutputV1

# --- END NEW IMPORTS ---
# --- NEW IMPORT FOR CIRCUIT BREAKER ---
from src.resilience.circuit_breaker import CircuitBreaker

# --- END NEW IMPORT ---
# --- NEW IMPORT FOR ERROR HANDLER ---
from src.utils.core_helpers.error_handler import handle_errors  # Updated import

# NEW IMPORT: For PromptOptimizer
from src.utils.prompting.prompt_optimizer import PromptOptimizer  # Updated import
from src.utils.reporting.output_parser import LLMOutputParser  # Updated import

# NEW IMPORTS: From src/utils/validation/api_key_validator.py # Updated import
from src.utils.validation.api_key_validator import (  # Updated import
    validate_gemini_api_key_format,
)

# --- Token Cost Definitions (per 1,000 tokens) ---
TOKEN_COSTS_PER_1K_TOKENS = {
    "gemini-1.5-flash": {"input": 0.00008, "output": 0.00024},
    "gemini-1.5-pro": {"input": 0.0005, "output": 0.0015},
}

logger = logging.getLogger(__name__)

# Initialize tiktoken encoder globally or as a class attribute for efficiency
# Fallback to a common encoder if the specific model's encoder is not found.
try:
    _TIKTOKEN_ENCODING = tiktoken.encoding_for_model(
        "gpt-4"
    )  # Using gpt-4 as a common default
except KeyError:
    logger.warning("Could not find tiktoken encoding for 'gpt-4'. Using 'cl100k_base'.")
    _TIKTOKEN_ENCODING = tiktoken.get_encoding("cl100k_base")

# Define a generic max tokens per prompt if not specified by model or settings
GENERIC_MAX_TOKENS_PER_PROMPT = 4096


class GeminiProvider:
    def __init__(
        self,
        api_key: str,
        model_name: str = "gemini-2.5-flash-lite",
        max_retries: int = None,
        max_backoff_seconds: int = None,
        tokenizer: Tokenizer = None,
        rich_console: Optional[Console] = None,
        request_id: Optional[str] = None,
        settings: Optional[
            ChimeraSettings
        ] = None,  # MODIFIED: Type hint to ChimeraSettings
        summarizer_pipeline_instance: Any = None,
        prompt_optimizer_instance: Optional[PromptOptimizer] = None,
    ):
        self.model_name = model_name
        self.model_registry = ModelRegistry()
        self.rich_console = rich_console or Console(stderr=True)
        self.request_id = request_id
        self._log_extra = {"request_id": self.request_id or "N/A"}
        self.output_parser = LLMOutputParser()
        self.settings = settings or ChimeraSettings()  # Ensure settings is initialized
        self.summarizer_pipeline_instance = summarizer_pipeline_instance
        self.prompt_optimizer = prompt_optimizer_instance
        self._tiktoken_encoding = _TIKTOKEN_ENCODING

        try:
            # MODIFIED: Prioritize passed api_key, then settings.GEMINI_API_KEY
            resolved_api_key = api_key or self.settings.GEMINI_API_KEY
            if not resolved_api_key:
                logger.critical(
                    "GEMINI_API_KEY is not set. LLMProvider will not function.",
                    extra={"event": "startup_failure"},
                )
                raise ValueError("LLM API Key is not configured.")

            self._api_key = resolved_api_key

            is_valid_format, format_message = validate_gemini_api_key_format(
                self._api_key
            )
            if not is_valid_format:
                logger.critical(
                    f"Invalid API key format: {format_message}",
                    extra={"event": "startup_failure"},
                )
                raise ValueError(f"Invalid API key format: {format_message}")
            self.client = genai.Client(api_key=self._api_key)
        except (APIError, ValueError) as e:
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
                    "Failed to initialize Gemini client: Invalid API Key. Please check your Gemini API Key.",
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

        if not self.prompt_optimizer:
            logger.warning(
                "PromptOptimizer instance not provided to GeminiProvider. Prompt optimization will be limited."
            )
            self.prompt_optimizer = PromptOptimizer(
                tokenizer=self.tokenizer,
                settings=self.settings,
                summarizer_pipeline=self.summarizer_pipeline_instance,
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
            and self._api_key == other._api_key  # MODIFIED: Use _api_key
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
            if "flash" in model_spec.name:
                return "gemini-1.5-flash"
            if "pro" in model_spec.name:
                return "gemini-1.5-pro"
        return "gemini-1.5-flash"

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

    def _adjust_prompt_for_schema_validation(
        self, prompt: str, output_schema: Type[BaseModel]
    ) -> str:
        # Add more specific instructions about the required schema
        schema_info = self._get_schema_info(output_schema)
        new_prompt = (
            f"{prompt}\n\n"
            "CRITICAL: Your output MUST strictly adhere to the following JSON schema:\n"
            f"{schema_info}\n"
            "DO NOT include any conversational text or markdown fences outside the JSON. "
            "Focus solely on providing a correct, valid JSON response."
        )
        return new_prompt

    def _get_schema_info(self, output_schema: Type[BaseModel]) -> str:
        # Convert the schema to a string representation
        schema = output_schema.model_json_schema()
        return json.dumps(schema, indent=2)

    def _count_tokens_tiktoken(self, text: str, model: str = None) -> int:
        """Counts tokens in the given text using tiktoken, with fallback for unknown models."""
        if not text:
            return 0
        try:
            # Use the model name if provided, otherwise fallback to self.model_name
            effective_model = model if model else self.model_name
            # Attempt to get the encoder for the specific model
            encoding = tiktoken.encoding_for_model(effective_model)
        except KeyError:
            # Fallback to a common encoder if the specific model's encoder is not found.
            logger.warning(
                f"Could not find tiktoken encoding for model '{effective_model}'. Using 'cl100k_base'."
            )
            encoding = tiktoken.get_encoding("cl100k_base")
        return len(encoding.encode(text))

    def _truncate_prompt_tiktoken(
        self, prompt: str, max_tokens: int, model: str = None
    ) -> str:
        """Truncates a prompt to a maximum number of tokens using tiktoken."""
        if not prompt:
            return ""
        try:
            effective_model = model if model else self.model_name
            encoding = tiktoken.encoding_for_model(effective_model)
        except KeyError:
            logger.warning(
                f"Could not find tiktoken encoding for model '{effective_model}'. Using 'cl100k_base'."
            )
            encoding = tiktoken.get_encoding("cl100k_base")

        tokens = encoding.encode(prompt)
        if len(tokens) > max_tokens:
            truncated_tokens = tokens[:max_tokens]
            return encoding.decode(truncated_tokens)
        return prompt

    def _get_max_tokens_for_model(self, model: str = None) -> int:
        """Returns the maximum token limit for a given model, with a default from settings."""
        effective_model = model if model else self.model_name
        model_spec = self.get_model_specification(effective_model)
        if model_spec:
            return model_spec.max_input_tokens

        return self.settings.max_tokens_per_persona.get(
            effective_model, GENERIC_MAX_TOKENS_PER_PROMPT
        )

    @handle_errors(log_level="ERROR")
    @CircuitBreaker(
        failure_threshold=5,
        recovery_timeout=60,
        expected_exception=(
            APIError,
            CircuitBreakerError,
            LLMUnexpectedError,
            GeminiAPIError,
            RetryError,
            RateLimitExceededError,
            LLMProviderRequestError,
            LLMProviderResponseError,
        ),
    )
    @retry(
        wait=wait_exponential(multiplier=1, min=4, max=60),
        stop=stop_after_attempt(5),
        reraise=True,
        retry=(
            retry_if_exception_type(APIError)
            | retry_if_exception_type(socket.gaierror)
            | retry_if_exception_type(LLMProviderRequestError)
            | retry_if_exception_type(LLMProviderResponseError)
        ),
    )
    def generate(
        self,
        prompt: str,
        system_prompt: str = "",
        output_schema: Optional[Type[BaseModel]] = None,
        temperature: float = 0.7,
        max_tokens: int = 100,
        persona_config: Any = None,
        intermediate_results: Dict[str, Any] = None,
        requested_model_name: str = None,
    ) -> tuple[str, int, int, bool]:
        """Generates content using the Gemini API, protected by a circuit breaker and tenacity retries."""
        final_model_to_use = requested_model_name

        current_model_spec = self.get_model_specification(
            final_model_to_use or self.model_name
        )
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

        config = genai_types.GenerateContentConfig(
            system_instruction=system_prompt,
            temperature=temperature,
            max_output_tokens=max_tokens,
        )

        max_schema_retries = 3
        current_prompt_for_llm = prompt

        # NEW: Token counting and enforcement before API call
        full_prompt_text_for_token_check = (
            f"{system_prompt}\n\n{current_prompt_for_llm}"
            if system_prompt
            else current_prompt_for_llm
        )
        prompt_tokens_for_check = self._count_tokens_tiktoken(
            full_prompt_text_for_token_check, current_model_name
        )
        max_input_tokens_for_model = self._get_max_tokens_for_model(current_model_name)

        if prompt_tokens_for_check > max_input_tokens_for_model:
            self._log_with_context(
                "error",
                f"Prompt exceeds maximum input token limit for model '{current_model_name}': {prompt_tokens_for_check}/{max_input_tokens_for_model}",
            )
            raise TokenBudgetExceededError(
                current_tokens=prompt_tokens_for_check,
                budget=max_input_tokens_for_model,
                details={
                    "phase": "LLM_Call",
                    "step_name": "Prompt_Preparation",
                    "tokens_needed": prompt_tokens_for_check,
                },
            )
        self._log_with_context(
            "info",
            f"Prompt token count for model '{current_model_name}': {prompt_tokens_for_check}/{max_input_tokens_for_model}",
        )

        for i in range(max_schema_retries):
            try:
                optimized_prompt = current_prompt_for_llm

                prompt_with_system = (
                    f"{system_prompt}\n\n{optimized_prompt}"
                    if system_prompt
                    else optimized_prompt
                )
                # Use native Gemini tokenizer for actual input token count for cost tracking
                input_tokens = self.tokenizer.count_tokens(prompt_with_system)

                self._log_with_context(
                    "debug",
                    "LLM Prompt Snippet",
                    model=current_model_name,
                    input_tokens=input_tokens,
                    prompt_snippet=prompt_with_system[:500] + "..."
                    if len(prompt_with_system) > 500
                    else prompt_with_system,
                )
                self._log_with_context(
                    "info",
                    "LLM Prompt Sent",
                    model=current_model_name,
                    temperature=config.temperature,
                    max_output_tokens=config.max_output_tokens,
                    full_system_prompt=system_prompt,
                    full_user_prompt=optimized_prompt,
                    input_tokens=input_tokens,
                )

                response = self.client.models.generate_content(
                    model=current_model_name, contents=optimized_prompt, config=config
                )

                if not response.candidates:
                    raise LLMUnexpectedError(
                        "No candidates in response", details={"response": response}
                    )

                if not response.candidates[0].content.parts:
                    raise LLMUnexpectedError(
                        "No content parts in response", details={"response": response}
                    )

                generated_text = response.candidates[0].content.parts[0].text

                if output_schema:
                    try:
                        cleaned_generated_text = self.output_parser._clean_llm_output(
                            generated_text
                        )

                        # NEW: Apply diff header repair proactively if schema expects it
                        if output_schema in [
                            SelfImprovementAnalysisOutputV1,
                            LLMOutput,
                        ]:
                            try:
                                # Temporarily parse to check for CODE_CHANGES/CODE_CHANGES_SUGGESTED
                                temp_parsed = json.loads(cleaned_generated_text)
                                if isinstance(temp_parsed, dict):
                                    if "IMPACTFUL_SUGGESTIONS" in temp_parsed:
                                        for suggestion in temp_parsed[
                                            "IMPACTFUL_SUGGESTIONS"
                                        ]:
                                            if "CODE_CHANGES_SUGGESTED" in suggestion:
                                                for change in suggestion[
                                                    "CODE_CHANGES_SUGGESTED"
                                                ]:
                                                    if change.get(
                                                        "ACTION"
                                                    ) == "MODIFY" and change.get(
                                                        "DIFF_CONTENT"
                                                    ):
                                                        change["DIFF_CONTENT"] = (
                                                            LLMOutputParser._repair_diff_content_headers(
                                                                change["DIFF_CONTENT"]
                                                            )
                                                        )
                                        cleaned_generated_text = json.dumps(temp_parsed)
                                    elif "CODE_CHANGES" in temp_parsed:
                                        for change in temp_parsed["CODE_CHANGES"]:
                                            if change.get(
                                                "ACTION"
                                            ) == "MODIFY" and change.get(
                                                "DIFF_CONTENT"
                                            ):
                                                change["DIFF_CONTENT"] = (
                                                    LLMOutputParser._repair_diff_content_headers(
                                                        change["DIFF_CONTENT"]
                                                    )
                                                )
                                        cleaned_generated_text = json.dumps(temp_parsed)
                            except json.JSONDecodeError:
                                pass
                            except Exception as e:
                                self._log_with_context(
                                    "warning",
                                    f"Proactive diff header repair failed: {e}",
                                    exc_info=True,
                                )

                        if hasattr(output_schema, "model_validate_json"):
                            output_schema.model_validate_json(cleaned_generated_text)
                        else:
                            output_schema.parse_raw(cleaned_generated_text)
                    except ValidationError as ve:
                        error_msg = f"LLM output failed schema validation: {ve}"
                        self._log_with_context(
                            "warning",
                            error_msg,
                            llm_output_snippet=generated_text[:200],
                        )
                        raise SchemaValidationError(
                            error_type="EARLY_SCHEMA_VALIDATION_FAILED",
                            field_path="LLM_OUTPUT",
                            invalid_value=generated_text[:500],
                            original_exception=ve,
                        )

                # Use native Gemini tokenizer for actual output token count for cost tracking
                output_tokens = self.tokenizer.count_tokens(generated_text)
                is_truncated = output_tokens >= config.max_output_tokens * 0.95
                self._log_with_context(
                    "debug",
                    f"Generated response (model: {current_model_name}, input: {input_tokens}, output: {output_tokens} tokens)",
                )

                self._log_with_context(
                    "debug",
                    "LLM Response Snippet",
                    model=current_model_name,
                    output_tokens=output_tokens,
                    generated_text_snippet=generated_text[:500] + "..."
                    if len(generated_text) > 500
                    else generated_text,
                )
                self._log_with_context(
                    "info",
                    "LLM Response Received",
                    model=current_model_name,
                    output_tokens=output_tokens,
                    full_generated_text=generated_text,
                )

                final_output_to_return = generated_text
                if output_schema:
                    final_output_to_return = cleaned_generated_text
                return final_output_to_return, input_tokens, output_tokens, is_truncated

            except SchemaValidationError as e:
                if i < max_schema_retries - 1:
                    current_prompt_for_llm = self._adjust_prompt_for_schema_validation(
                        current_prompt_for_llm, output_schema
                    )
                    continue
                else:
                    raise e
            except ServerError as e:
                error_msg = str(e)
                self._log_with_context(
                    "error",
                    f"Gemini ServerError encountered: {error_msg}",
                    exc_info=True,
                )
                raise LLMProviderRequestError(
                    message=f"Gemini ServerError: {error_msg}",
                    provider_error_code=e.code if hasattr(e, "code") else 500,
                    original_exception=e,
                ) from e
            except APIError as e:
                error_msg = str(e)
                error_msg_lower = error_msg.lower()
                response_json = (
                    e.response_json
                    if hasattr(e, "response_json") and isinstance(e.response_json, dict)
                    else {
                        "error": {
                            "code": e.code if hasattr(e, "code") else 500,
                            "message": error_msg,
                            "raw_response": str(getattr(e, "response_json", None)),
                        }
                    }
                )

                if hasattr(e, "code") and e.code == 401:
                    raise GeminiAPIError(
                        f"Invalid API Key: {error_msg}",
                        code=e.code,
                        response_details=response_json,
                        original_exception=e,
                    ) from e
                elif hasattr(e, "code") and e.code == 403:
                    raise GeminiAPIError(
                        f"API Key lacks permissions: {error_msg}",
                        code=e.code,
                        response_details=response_json,
                        original_exception=e,
                    ) from e
                elif hasattr(e, "code") and e.code == 429:
                    raise RateLimitExceededError(
                        f"Rate limit exceeded: {error_msg}", original_exception=e
                    ) from e
                elif (
                    hasattr(e, "code")
                    and e.code == 400
                    and (
                        "context window exceeded" in error_msg_lower
                        or "prompt too large" in error_msg_lower
                    )
                ):
                    raise LLMUnexpectedError(
                        f"LLM context window exceeded: {error_msg}",
                        original_exception=e,
                    ) from e
                elif (
                    hasattr(e, "code")
                    and e.code == 400
                    and (
                        "invalid json" in error_msg_lower
                        or "invalid json format" in error_msg_lower
                        or "invalid response format" in error_msg_lower
                    )
                ):
                    raise LLMProviderResponseError(
                        error_type="API_INVALID_JSON",
                        field_path="LLM_OUTPUT",
                        invalid_value=error_msg[:500],
                        original_exception=e,
                    ) from e
                else:
                    raise LLMProviderError(error_msg, original_exception=e) from e
            except socket.gaierror as e:
                raise LLMUnexpectedError(
                    f"Network error: {e}", original_exception=e
                ) from e
            except Exception as e:
                error_msg = str(e)
                if (
                    "context window exceeded" in error_msg.lower()
                    or "prompt too large" in error_msg.lower()
                ):
                    raise LLMUnexpectedError(
                        f"LLM context window exceeded: {error_msg}",
                        original_exception=e,
                    ) from e
                raise LLMProviderError(error_msg, original_exception=e) from e

        raise LLMProviderError(
            "Failed to generate valid response after all schema retries."
        )

    def close(self):
        """Explicitly releases resources held by the GeminiProvider.
        For google.genai, this primarily involves setting the client to None
        to encourage garbage collection.
        """
        if hasattr(self, "client") and self.client:
            logger.info("Closing GeminiProvider: Releasing LLM client resources.")
            self.client = None
        else:
            logger.debug("GeminiProvider: No active client to close.")
