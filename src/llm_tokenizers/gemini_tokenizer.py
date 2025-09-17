# src/llm_tokenizers/gemini_tokenizer.py
"""Gemini-specific tokenizer implementation."""

import logging
from typing import Optional, Dict, Any, TYPE_CHECKING
from .base import Tokenizer  # This import is relative, so it remains the same
import hashlib
import re
import sys
from functools import lru_cache

logger = logging.getLogger(__name__)

# Use TYPE_CHECKING to avoid circular import at runtime
if TYPE_CHECKING:
    from src.config.model_registry import ModelRegistry


class GeminiTokenizer(Tokenizer):
    """Gemini-specific tokenizer that uses the google-genai library to count tokens."""

    def __init__(
        self, model_name: str = "gemini-2.5-flash-lite", genai_client: Any = None
    ):
        """Initializes the GeminiTokenizer.

        Args:
            model_name: The Gemini model name to use for token counting.
            genai_client: An initialized google.genai.Client instance.
        """
        if genai_client is None:
            raise ValueError(
                "genai_client must be provided to GeminiTokenizer for token counting."
            )

        self.genai_client = genai_client
        self.model_name = model_name
        self._max_output_tokens: int = 65536  # Default, updated to 65k output tokens

    @property
    def max_output_tokens(self) -> int:
        """Returns the maximum number of output tokens for the model."""
        return self._max_output_tokens

    @max_output_tokens.setter
    def max_output_tokens(self, value: int):
        self._max_output_tokens = value

    @lru_cache(maxsize=512)
    def count_tokens(self, text: str) -> int:
        """Counts tokens in the given text using the Gemini API, with caching."""
        if not text:
            return 0

        try:
            # Ensure text is properly encoded for the API call
            try:
                text_encoded = text.encode("utf-8")
                text_for_api = text_encoded.decode("utf-8", errors="replace")
            except UnicodeEncodeError:
                # Fallback if encoding fails, replace problematic characters
                text_for_api = text.encode("utf-8", errors="replace").decode(
                    "utf-8", errors="replace"
                )
                logger.warning(
                    "Fixed encoding issues in text for token counting by replacing problematic characters."
                )

            # Use the count_tokens API to get token count
            response = self.genai_client.models.count_tokens(
                model=self.model_name, contents=text_for_api
            )
            tokens = response.total_tokens
            logger.debug(f"Token count for text is {tokens}. Stored in cache.")
            return tokens

        except Exception as e:
            logger.error(
                f"Gemini token counting failed for model '{self.model_name}': {str(e)}"
            )
            # Improved fallback based on content type heuristic
            # Check for common code indicators in the first 200 characters
            if any(
                indicator in text[:200]
                for indicator in [
                    "def ",
                    "class ",
                    "import ",
                    "{",
                    "}",
                    "func",
                    "var ",
                    "const ",
                ]
            ):
                # Code content is denser - use ~3.5 characters per token
                approx_tokens = max(1, int(len(text) / 3.5))
            else:
                # Standard text content - use ~4 characters per token
                approx_tokens = max(1, int(len(text) / 4))
            logger.warning(
                f"Falling back to improved token approximation ({approx_tokens}) due to error: {str(e)}"
            )
            return approx_tokens

    def estimate_tokens_for_context(self, context_str: str, prompt: str) -> int:
        """Estimates tokens for a context and prompt combination."""
        combined_text = f"{context_str}\n\n{prompt}"
        return self.count_tokens(combined_text)

    def truncate_to_token_limit(  # Renamed from trim_text_to_tokens
        self, text: str, max_tokens: int, truncation_indicator: str = ""
    ) -> str:
        """
        Trim text to fit within the specified token limit.
        Uses a binary search approach for efficiency.
        """
        if max_tokens < 1:
            return ""

        # If the text already fits, return it as is
        current_tokens = self.count_tokens(text)
        if current_tokens <= max_tokens:
            return text

        # Adjust max_tokens for the truncation indicator if it's used
        effective_max_tokens = max(1, max_tokens)  # Ensure at least 1 token for content
        if truncation_indicator:
            indicator_tokens = self.count_tokens(truncation_indicator)
            effective_max_tokens = max(1, max_tokens - indicator_tokens)

        # Use binary search to find the longest prefix that fits within effective_max_tokens
        low = 0
        high = len(text)
        best_char_limit = 0

        while low <= high:
            mid = (low + high) // 2
            if mid == 0:
                current_tokens_at_mid = 0
            else:
                current_tokens_at_mid = self.count_tokens(text[:mid])

            if current_tokens_at_mid <= effective_max_tokens:
                best_char_limit = mid
                low = mid + 1
            else:
                high = mid - 1

        trimmed_text = text[:best_char_limit]

        # Append truncation indicator if truncation actually occurred
        # Check if the trimmed text is actually shorter than the original text
        if truncation_indicator and self.count_tokens(trimmed_text) < current_tokens:
            return trimmed_text + truncation_indicator

        return trimmed_text
