# src/tokenizers/gemini_tokenizer.py
"""Gemini-specific tokenizer implementation."""
import logging
from typing import Optional, Dict, Any # Added Dict, Any for genai_client type hint

# Import the Tokenizer ABC from base.py
from .base import Tokenizer

logger = logging.getLogger(__name__)

class GeminiTokenizer(Tokenizer):
    """Gemini-specific tokenizer that uses the google-genai library to count tokens."""
    
    def __init__(self, model_name: str = "gemini-2.5-flash-lite", genai_client: Any = None):
        """Initializes the GeminiTokenizer.
        
        Args:
            model_name: The Gemini model name to use for token counting.
            genai_client: An initialized google.genai.Client instance.
        """
        if genai_client is None:
            raise ValueError("genai_client must be provided to GeminiTokenizer for token counting.")
        self.genai_client = genai_client  # Store the client instance
        self.model_name = model_name
        self._cache = {} # Initialize cache for token counts

    def count_tokens(self, text: str) -> int:
        """Counts tokens in the given text using the Gemini API, with caching.
        
        Args:
            text: The input string to count tokens for.
            
        Returns:
            The total number of tokens.
            
        Raises:
            Exception: If token counting fails.
        """
        if not text:
            return 0
        
        # Use hash for cache key to handle long strings efficiently
        # Ensure text is hashable and consistent (e.g., normalize whitespace if needed, but hash() handles strings well)
        text_hash = hash(text)
        
        if text_hash in self._cache:
            logger.debug(f"Cache hit for token count (hash: {text_hash}).")
            return self._cache[text_hash]
        
        # If not in cache, perform the actual token count
        try:
            # --- MODIFICATION START ---
            # Ensure text is properly encoded for token counting to prevent errors
            # with special characters or unexpected byte sequences.
            try:
                text_encoded = text.encode('utf-8')
                # Re-decode to ensure it's a valid string for the API call
                text_for_api = text_encoded.decode('utf-8', errors='replace') 
            except UnicodeEncodeError:
                # Fallback if encoding itself fails, replace problematic chars
                text_for_api = text.encode('utf-8', errors='replace').decode('utf-8', errors='replace')
                logger.warning("Fixed encoding issues in text for token counting by replacing problematic characters.")
            # --- MODIFICATION END ---
            
            response = self.genai_client.models.count_tokens(model=self.model_name, contents=text_for_api)
            tokens = response.total_tokens
            
            # Store in cache
            self._cache[text_hash] = tokens
            
            # Trim cache to prevent unbounded growth
            if len(self._cache) > 1000:
                # Keep most recent 500 entries (as per suggestion)
                # This is a simple LRU-like eviction strategy.
                self._cache = dict(list(self._cache.items())[-500:])
            
            logger.debug(f"Token count for text (hash: {text_hash}) is {tokens}. Stored in cache.")
            return tokens
            
        except Exception as e:
            # Catch potential API errors or network issues during token counting
            logger.error(f"Gemini token counting failed for model '{self.model_name}': {e}")
            # Re-raise to allow higher-level error handling (e.g., in core.py)
            raise

    def estimate_tokens_for_context(self, context_str: str, prompt: str) -> int:
        """Estimates tokens for a context and prompt combination.
        
        Args:
            context_str: The context string to include.
            prompt: The user's prompt.
        
        Returns:
            The estimated total number of tokens.
        """
        # Combine context and prompt with a separator.
        # This method is called to estimate the total tokens for a combined input.
        combined_text = f"{context_str}\n\n{prompt}"
        # Use the count_tokens method (which is now cached) to get the actual token count.
        return self.count_tokens(combined_text)