from abc import ABC, abstractmethod

class Tokenizer(ABC):
    """Abstract Base Class for tokenizers.
    Defines the interface for counting tokens for different LLM providers."""
    
    @abstractmethod
    def count_tokens(self, text: str) -> int:
        """Counts tokens in the given text.
        
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

    @abstractmethod
    def trim_text_to_tokens(self, text: str, max_tokens: int, truncation_indicator: str = "") -> str:
        """
        Trims the given text to fit within the specified token limit.
        
        Args:
            text: The input string to trim.
            max_tokens: The maximum number of tokens allowed.
            truncation_indicator: An optional string to append if truncation occurs.
        
        Returns:
            The trimmed string.
        """
        pass