# src/llm/orchestrator.py
import logging
from typing import Any, Optional, Type

from pydantic import BaseModel
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from src.config.settings import ChimeraSettings
from src.exceptions import (
    CircuitBreakerError,
    LLMProviderRequestError,
    LLMProviderResponseError,
    SchemaValidationError,
    TokenBudgetExceededError,
)
from src.llm_provider import GeminiProvider
from src.resilience.circuit_breaker import CircuitBreaker
from src.token_tracker import TokenUsageTracker

logger = logging.getLogger(__name__)


class LLMOrchestrator:
    """
    Orchestrates LLM calls, applying resilience patterns like retries and circuit breaking,
    and handling token tracking and response formatting.
    """

    def __init__(
        self,
        llm_provider: GeminiProvider,
        token_tracker: TokenUsageTracker,
        settings: ChimeraSettings,
        request_id: str,
    ):
        self.llm_provider = llm_provider
        self.token_tracker = token_tracker
        self.settings = settings
        self.request_id = request_id
        self._log_extra = {"request_id": self.request_id or "N/A"}

        # Initialize CircuitBreaker
        self.circuit_breaker = CircuitBreaker(
            failure_threshold=self.settings.max_retries,  # Use settings for threshold
            recovery_timeout=self.settings.max_backoff_seconds,  # Use settings for timeout
            expected_exception=(
                LLMProviderRequestError,
                LLMProviderResponseError,
                TokenBudgetExceededError,
                SchemaValidationError,  # Schema validation failures can also open the circuit
            ),
        )
        logger.info(
            "LLMOrchestrator initialized with CircuitBreaker.", extra=self._log_extra
        )

    def _log_with_context(self, level: str, message: str, **kwargs):
        """Helper to add request context to all logs from this instance."""
        exc_info = kwargs.pop("exc_info", None)
        log_data = {**self._log_extra, **kwargs}
        logger_method = getattr(logger, level)
        if exc_info is not None:
            logger_method(message, exc_info=exc_info, extra=log_data)
        else:
            logger_method(message, extra=log_data)

    @retry(
        wait=wait_exponential(multiplier=1, min=4, max=10),  # Exponential backoff
        stop=stop_after_attempt(5),  # Max 5 attempts (initial + 4 retries)
        retry=retry_if_exception_type(
            (
                LLMProviderRequestError,
                LLMProviderResponseError,
                TokenBudgetExceededError,
                SchemaValidationError,
            )
        ),
        reraise=True,  # Re-raise the last exception if all retries fail
    )
    def _call_llm_with_resilience(
        self,
        prompt: str,
        system_prompt: str,
        output_schema: Type[BaseModel],
        temperature: float,
        max_tokens: int,
        persona_config: Any,
        requested_model_name: Optional[str],
        context: str,
    ) -> dict[str, Any]:
        """
        Internal method to make the actual LLM call with retry logic.
        The circuit breaker is applied externally to this retry loop.
        """
        self._log_with_context(
            "debug",
            "Attempting LLM call with resilience.",
            persona=persona_config.name,
            context=context,
        )

        # FIX: Set the current stage for semantic token tracking (Efficiency improvement)
        self.token_tracker.set_current_stage(context)

        # The llm_provider.generate method itself is now a thin wrapper around the API call,
        # without its own retry/circuit breaker.
        generated_text, input_tokens, output_tokens, is_truncated = (
            self.llm_provider.generate(
                prompt=prompt,
                system_prompt=system_prompt,
                output_schema=output_schema,
                temperature=temperature,
                max_tokens=max_tokens,
                persona_config=persona_config,
                requested_model_name=requested_model_name,
            )
        )

        # Record token usage after a successful generation
        self.token_tracker.record_usage(
            input_tokens, output_tokens, persona=persona_config.name
        )

        return {
            "text": generated_text,
            "usage": {
                "prompt_tokens": input_tokens,
                "completion_tokens": output_tokens,
                "total_tokens": input_tokens + output_tokens,
            },
            "is_truncated": is_truncated,
        }

    def call_llm(
        self,
        template_name: str,  # This parameter is not used here, but kept for compatibility with core.py's call
        prompt_params: dict[
            str, Any
        ],  # This parameter is not used here, but kept for compatibility with core.py's call
        system_prompt: str,
        output_schema: Type[BaseModel],
        temperature: float,
        max_tokens: int,
        persona_config: Any,
        requested_model_name: Optional[str],
        context: str,
    ) -> dict[str, Any]:
        """
        Centralized function to make LLM calls, applying circuit breaking and retries.
        """
        if not self.circuit_breaker.is_available():
            self._log_with_context(
                "warning",
                "Circuit breaker is OPEN. Rejecting LLM call.",
                persona=persona_config.name,
                context=context,
            )
            raise CircuitBreakerError(
                f"Service unavailable: Circuit breaker is open for {persona_config.name}."
            )

        try:
            # Call the internal method with retry logic
            response = self._call_llm_with_resilience(
                prompt=prompt_params[
                    "prompt"
                ],  # Extract actual prompt from prompt_params
                system_prompt=system_prompt,
                output_schema=output_schema,
                temperature=temperature,
                max_tokens=max_tokens,
                persona_config=persona_config,
                requested_model_name=requested_model_name,
                context=context,
            )
            self.circuit_breaker.record_success()
            return response
        except (
            LLMProviderRequestError,
            LLMProviderResponseError,
            TokenBudgetExceededError,
            SchemaValidationError,
        ) as e:
            self.circuit_breaker.record_failure()
            self._log_with_context(
                "error",
                f"LLM call failed after retries, recording failure in circuit breaker: {e}",
                persona=persona_config.name,
                context=context,
                exc_info=True,
            )
            raise e  # Re-raise the exception for core.py to handle
        except Exception as e:
            # Catch any other unexpected exceptions and record as failure
            self.circuit_breaker.record_failure()
            self._log_with_context(
                "critical",
                f"Unexpected error during LLM call, recording failure in circuit breaker: {e}",
                persona=persona_config.name,
                context=context,
                exc_info=True,
            )
            raise e

    def close(self):
        """Clean up resources if any."""
        # The llm_provider is closed by SocraticDebate.close, no need to close it here again.
        self._log_with_context("info", "LLMOrchestrator resources closed.")
