# src/tokenizers/gemini_tokenizer.py
"""Gemini-specific tokenizer implementation."""

import logging
from typing import Optional, Dict, Any
from .base import Tokenizer
import hashlib
import re
import sys
from functools import lru_cache

logger = logging.getLogger(__name__)


class GeminiTokenizer(Tokenizer):
    """Gemini-specific tokenizer that uses the google-genai library to count tokens."""

    # Map model names to their max output tokens.
    # Values are based on common Gemini model specifications and the provided code's usage.
    # Note: Official Gemini API docs might not always list exact max output tokens for every variant.
    # These are generally accepted values.
    MODEL_MAX_OUTPUT_TOKENS = {
        # Gemini 2.5 models (explicitly listed as per app's selectbox)
        "gemini-2.5-flash-lite": 65536,
        "gemini-2.5-flash": 65536,
        "gemini-2.5-pro": 65536,
        # Default fallback for any other models or versions
        "default": 65536,
    }

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

    @property
    def max_output_tokens(self) -> int:
        """Returns the maximum number of output tokens for the model."""
        # Try to find the exact model name in the map
        if self.model_name in self.MODEL_MAX_OUTPUT_TOKENS:
            return self.MODEL_MAX_OUTPUT_TOKENS[self.model_name]

        # Attempt to match with common patterns if exact name not found
        # This is a heuristic and might need adjustment if model naming conventions change.
        for model_pattern, token_limit in self.MODEL_MAX_OUTPUT_TOKENS.items():
            # Simple check for common prefixes or substrings
            if model_pattern in self.model_name:
                logger.debug(
                    f"Matched model '{self.model_name}' with pattern '{model_pattern}', using limit {token_limit}"
                )
                return token_limit

        # Return default value if no match found
        logger.warning(
            f"Unknown model '{self.model_name}', using default max output tokens ({self.MODEL_MAX_OUTPUT_TOKENS['default']})"
        )
        return self.MODEL_MAX_OUTPUT_TOKENS["default"]

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

    def truncate_to_token_limit( # Renamed from trim_text_to_tokens
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
        effective_max_tokens = max(1, max_tokens) # Ensure at least 1 token for content
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