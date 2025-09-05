# src/utils/prompt_optimizer.py
import logging
from typing import Dict, Any, List, Optional
from src.tokenizers import Tokenizer
from src.config.settings import ChimeraSettings # Import ChimeraSettings

logger = logging.getLogger(__name__)

class PromptOptimizer:
    """Optimizes prompts for various personas based on context and token limits."""

    def __init__(self, tokenizer: Tokenizer, settings: ChimeraSettings):
        self.tokenizer = tokenizer
        self.settings = settings

    def optimize_prompt(
        self,
        prompt: str,
        persona_name: str,
        max_output_tokens_for_turn: int,
    ) -> str:
        """
        Optimizes a prompt for a specific persona based on actual token usage and persona-specific limits.
        This method aims to reduce the input prompt size if it, combined with the expected output,
        exceeds a reasonable threshold for the persona.
        """
        # Calculate current prompt tokens
        prompt_tokens = self.tokenizer.count_tokens(prompt)
        
        # Get persona-specific token limits from settings
        # Use a default if not explicitly defined for the persona
        max_input_tokens_for_persona = self.settings.max_tokens_per_persona.get(
            persona_name, self.settings.default_max_input_tokens_per_persona
        )
        
        # Total tokens for the turn (input + output) should ideally not exceed a certain limit
        # This is a heuristic: we want to keep the total conversation within a manageable window.
        # A simple approach is to ensure the input prompt doesn't exceed its allocated budget.
        
        if prompt_tokens > max_input_tokens_for_persona:
            logger.warning(
                f"{persona_name} prompt exceeds input token limit ({prompt_tokens}/{max_input_tokens_for_persona}). Optimizing..."
            )
            
            # Truncate the prompt to fit the input token limit
            # The truncation indicator is added by the tokenizer's method
            optimized_prompt = self.tokenizer.truncate_to_token_limit(
                prompt, max_input_tokens_for_persona, 
                truncation_indicator="\n\n[TRUNCATED - focusing on most critical aspects]"
            )
            
            logger.info(
                f"Prompt for {persona_name} truncated from {prompt_tokens} to {self.tokenizer.count_tokens(optimized_prompt)} tokens."
            )
            return optimized_prompt
        
        return prompt

    def _optimize_self_improvement_prompt(self, prompt: str, max_tokens: int) -> str:
        """
        Specific optimization for Self-Improvement Analyst prompts.
        Prioritizes keeping critical sections like metrics and suggestions.
        """
        # This is a placeholder for more intelligent, structured truncation
        # For now, it uses the general truncation but can be enhanced.
        return self.tokenizer.truncate_to_token_limit(
            prompt, max_tokens, truncation_indicator="\n\n[TRUNCATED - focusing on most critical aspects]"
        )

    def _optimize_security_prompt(self, prompt: str, max_tokens: int) -> str:
        """
        Specific optimization for Security Auditor prompts.
        Prioritizes keeping high-severity issues and relevant code snippets.
        """
        # This is a placeholder for more intelligent, structured truncation
        # For now, it uses the general truncation but can be enhanced.
        return self.tokenizer.truncate_to_token_limit(
            prompt, max_tokens, truncation_indicator="\n\n[TRUNCATED - focusing on high-severity issues]"
        )
