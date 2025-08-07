# src/tokenizers/gemini_tokenizer.py
"""Gemini-specific tokenizer implementation."""
import google.genai as genai
from google.genai import types
import logging
# Import the Tokenizer ABC from base.py instead of src.tokenizers
from .base import Tokenizer

logger = logging.getLogger(__name__)

class GeminiTokenizer(Tokenizer):
    """Gemini-specific tokenizer that uses the google-genai library to count tokens."""    
    # FIX: Added genai_client parameter and validation
    def __init__(self, model_name: str = "gemini-2.5-flash-lite", genai_client=None):
        """
        Initializes the GeminiTokenizer.
        
        Args:
            model_name: The Gemini model name to use for token counting.
            genai_client: An initialized google.genai.Client instance.
        """
        if genai_client is None:
            raise ValueError("genai_client must be provided to GeminiTokenizer for token counting.")
        self.genai_client = genai_client # Store the client instance
        self.model_name = model_name
        
    def count_tokens(self, text: str) -> int:
        """
        Counts tokens in the given text using the Gemini API.
        
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
            # FIX: Use the client instance to call models.count_tokens
            # FIX: Use types.Part.from_text for clarity in contents argument
            response = self.genai_client.models.count_tokens(
                model=self.model_name,
                contents=[types.Part.from_text(text)]
            )
            return response.total_tokens
        except Exception as e:
            # Catch potential API errors or network issues during token counting.
            logger.error(f"Gemini token counting failed for model '{self.model_name}': {e}")
            # Re-raise the exception to be handled by the caller (e.g., GeminiProvider)
            raise

    def estimate_tokens_for_context(self, context_str: str, prompt: str) -> int:
        """
        Estimates the total tokens required for a given context and prompt.
        This is a helper for budget calculation.
        """
        try:
            context_tokens = self.count_tokens(context_str) if context_str else 0
            prompt_tokens = self.count_tokens(prompt) if prompt else 0
            return context_tokens + prompt_tokens
        except Exception as e:
            logger.error(f"Error estimating tokens for context/prompt: {e}")
            # FIX: Re-raise the exception as it's critical for budget calculation
            raise