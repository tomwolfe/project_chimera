# src/tokenizers/gemini_tokenizer.py
"""Gemini-specific tokenizer implementation."""
from google import genai
from google.genai import types
import logging
# Import the Tokenizer ABC from base.py instead of src.tokenizers
from .base import Tokenizer  # CHANGED THIS LINE

logger = logging.getLogger(__name__)

class GeminiTokenizer(Tokenizer):
    """Gemini-specific tokenizer that uses the google-genai library to count tokens."""    
    def __init__(self, model_name: str = "gemini-2.5-flash-lite"):
        """
        Initializes the GeminiTokenizer.
        
        Args:
            model_name: The Gemini model name to use for token counting.
                        This should match the model used by the GeminiProvider.
        """
        self.model_name = model_name
        # The genai library handles client initialization internally or
        # expects it to be managed by the caller (e.g., GeminiProvider).
        # This tokenizer relies on the genai library being configured with an API key,
        # which is assumed to be handled by the GeminiProvider.
        
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
            # The genai.count_tokens function requires a model name and content.
            # The content should be structured as a list of parts.
            # We'll assume a single user part for simplicity.
            response = genai.count_tokens(
                model=self.model_name,
                contents=[{"parts": [{"text": text}]}]
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
            # Return a conservative estimate or re-raise
            return 0 # Or raise an error if this is critical for budget calculation
