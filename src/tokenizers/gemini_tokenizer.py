# src/tokenizers/gemini_tokenizer.py
"""Gemini-specific tokenizer implementation."""
import logging
from typing import Optional, Dict, Any
from .base import Tokenizer
import hashlib
import re
import sys # Import sys for sys.stderr
from functools import lru_cache # ADD THIS IMPORT

logger = logging.getLogger(__name__)

class GeminiTokenizer(Tokenizer):
    """Gemini-specific tokenizer that uses the google-genai library to count tokens."""
    
    # Map model names to their max output tokens.
    # Values are based on common Gemini model specifications and the provided code's usage.
    # Note: Official Gemini API docs might not always list exact max output tokens for every variant.
    # These are generally accepted values.
    MODEL_MAX_OUTPUT_TOKENS = {

        # Gemini 2.5 models (explicitly listed as per app's selectbox)
        "gemini-2.5-flash-lite": 8192,
        "gemini-2.5-flash": 8192,
        "gemini-2.5-pro": 32768,
        
        # Default fallback for any other models or versions
        "default": 8192
    }
    
    def __init__(self, model_name: str = "gemini-2.5-flash-lite", genai_client: Any = None):
        """Initializes the GeminiTokenizer.
        
        Args:
            model_name: The Gemini model name to use for token counting.
            genai_client: An initialized google.genai.Client instance.
        """
        if genai_client is None:
            raise ValueError("genai_client must be provided to GeminiTokenizer for token counting.")
        
        self.genai_client = genai_client
        self.model_name = model_name
        # self._cache = {}  # REMOVE THIS LINE - cache is now handled by lru_cache

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
                logger.debug(f"Matched model '{self.model_name}' with pattern '{model_pattern}', using limit {token_limit}")
                return token_limit
        
        # Return default value if no match found
        logger.warning(f"Unknown model '{self.model_name}', using default max output tokens (8192)")
        return self.MODEL_MAX_OUTPUT_TOKENS["default"]
    
    @lru_cache(maxsize=512) # ADD THIS DECORATOR
    def count_tokens(self, text: str) -> int:
        """Counts tokens in the given text using the Gemini API, with caching."""
        if not text:
            return 0
            
        # Use a hash of the text for cache key
        # text_hash = hash(text) # REMOVE THIS LINE - lru_cache handles hashing
        
        # if text_hash in self._cache: # REMOVE THIS BLOCK - lru_cache handles caching
        #     logger.debug(f"Cache hit for token count (hash: {text_hash}).")
        #     return self._cache[text_hash]
        
        try:
            # Ensure text is properly encoded for the API call
            try:
                text_encoded = text.encode('utf-8')
                text_for_api = text_encoded.decode('utf-8', errors='replace')
            except UnicodeEncodeError:
                # Fallback if encoding fails, replace problematic characters
                text_for_api = text.encode('utf-8', errors='replace').decode('utf-8', errors='replace')
                logger.warning("Fixed encoding issues in text for token counting by replacing problematic characters.")
            
            # Use the count_tokens API to get token count
            response = self.genai_client.models.count_tokens(
                model=self.model_name,
                contents=text_for_api
            )
            tokens = response.total_tokens
            
            # Cache the result # REMOVE THIS LINE - lru_cache handles caching
            # self._cache[text_hash] = tokens # REMOVE THIS LINE
            logger.debug(f"Token count for text is {tokens}. Stored in cache.") # Modified log message
            return tokens
            
        except Exception as e:
            logger.error(f"Gemini token counting failed for model '{self.model_name}': {str(e)}")
            # Fallback to approximate count if API fails, to prevent crashing the budget calculation
            # IMPROVED FALLBACK: Use a more accurate approximation (e.g., 4 chars per token)
            approx_tokens = max(1, int(len(text) / 4))  # More accurate fallback
            logger.warning(f"Falling back to improved token approximation ({approx_tokens}) due to error: {str(e)}")
            return approx_tokens

    def estimate_tokens_for_context(self, context_str: str, prompt: str) -> int:
        """Estimates tokens for a context and prompt combination."""
        combined_text = f"{context_str}\n\n{prompt}"
        return self.count_tokens(combined_text)