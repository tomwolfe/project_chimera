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

    def count_tokens(self, text: str) -> int:
        """Counts tokens in the given text using the Gemini API.
        
        Args:
            text: The input string to count tokens for.
            
        Returns:
            The total number of tokens.
            
        Raises:
            Exception: If token counting fails (e.g., API error, network issue).
        """
        if not text:
            return 0
        try:
            # As per Google Gen AI SDK documentation, count_tokens accepts raw text
            response = self.genai_client.models.count_tokens(model=self.model_name, contents=text)
            return response.total_tokens
        except Exception as e:
            # Catch potential API errors or network issues during token counting
            logger.error(f"Gemini token counting failed for model '{self.model_name}': {e}")
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
        # Use the count_tokens method to get the actual token count.
        return self.count_tokens(combined_text)