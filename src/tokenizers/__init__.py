# src/tokenizers/__init__.py
"""
Tokenizer module for abstracting token counting across different LLM providers.
"""

from abc import ABC, abstractmethod

# Import the GeminiTokenizer class from its specific file
from .gemini_tokenizer import GeminiTokenizer

class Tokenizer(ABC):
    """
    Abstract Base Class for tokenizers.
    Defines the interface for counting tokens for different LLM providers.
    """
    
    @abstractmethod
    def count_tokens(self, text: str) -> int:
        """
        Counts tokens in the given text.
        
        Args:
            text: The input string to count tokens for.
            
        Returns:
            The total number of tokens.
            
        Raises:
            Exception: If token counting fails.
        """
        pass
    
    @abstractmethod
    def estimate_tokens_for_context(self, context_str: str, prompt: str) -> int:
        """
        Estimates the total tokens required for a given context and prompt.
        This is a helper for budget calculation.
        
        Args:
            context_str: The context string.
            prompt: The prompt string.
            
        Returns:
            The estimated total number of tokens.
        """
        pass

# Export both the ABC and the concrete implementation
__all__ = ["Tokenizer", "GeminiTokenizer"]