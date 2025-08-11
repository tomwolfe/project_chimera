# src/tokenizers/gemini_tokenizer.py
"""Gemini-specific tokenizer implementation."""
import logging
from typing import Optional, Dict, Any
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
        self.genai_client = genai_client
        self.model_name = model_name
        self._cache = {} # Initialize cache for token counts

    def count_tokens(self, text: str) -> int:
        """Counts tokens in the given text using the Gemini API, with caching.
        """
        if not text:
            return 0
        
        text_hash = hash(text)
        
        if text_hash in self._cache:
            logger.debug(f"Cache hit for token count (hash: {text_hash}).")
            return self._cache[text_hash]
        
        try:
            try:
                text_encoded = text.encode('utf-8')
                text_for_api = text_encoded.decode('utf-8', errors='replace') 
            except UnicodeEncodeError:
                text_for_api = text.encode('utf-8', errors='replace').decode('utf-8', errors='replace')
                logger.warning("Fixed encoding issues in text for token counting by replacing problematic characters.")
            
            response = self.genai_client.models.count_tokens(model=self.model_name, contents=text_for_api)
            tokens = response.total_tokens
            
            # --- MODIFICATION FOR IMPROVEMENT 3.2 ---
            # Store in cache and implement bounded cache (LRU-like)
            self._cache[text_hash] = tokens
            if len(self._cache) > 1000: # Limit cache size
                # Keep the most recent 500 entries
                self._cache = dict(list(self._cache.items())[-500:])
            # --- END MODIFICATION ---
            
            logger.debug(f"Token count for text (hash: {text_hash}) is {tokens}. Stored in cache.")
            return tokens
            
        except Exception as e:
            logger.error(f"Gemini token counting failed for model '{self.model_name}': {e}")
            raise

    def estimate_tokens_for_context(self, context_str: str, prompt: str) -> int:
        """Estimates tokens for a context and prompt combination."""
        combined_text = f"{context_str}\n\n{prompt}"
        return self.count_tokens(combined_text)