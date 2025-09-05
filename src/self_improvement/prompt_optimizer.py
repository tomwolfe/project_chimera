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
        exceeds a reasonable threshold for the persona, or if the overall token budget is constrained.
        """
        # Calculate current prompt tokens
        prompt_tokens = self.tokenizer.count_tokens(prompt)
        
        # Get persona-specific token limits from settings
        # Use a default if not explicitly defined for the persona
        # Note: max_output_tokens_for_turn is already passed, so we use that as the target output.
        # The persona's max_tokens from PersonaConfig is the hard limit for output.
        # Here, we're concerned with the *input* prompt's size.
        persona_input_token_limit = self.settings.max_tokens_per_persona.get(
            persona_name, self.settings.default_max_input_tokens_per_persona
        )

        # Heuristic: If the current prompt tokens are already very high,
        # or if the combined input + expected output tokens exceed a certain percentage
        # of the persona's total capacity (e.g., 80% of its max_tokens from PersonaConfig),
        # then we should aggressively optimize the input.
        # The `max_output_tokens_for_turn` is the *actual* budget for the output,
        # so we should consider it.
        
        # Calculate a soft limit for the input prompt based on the persona's overall capacity
        # and the expected output size.
        # A persona's total capacity is its max_tokens (from PersonaConfig, which is passed to _execute_llm_turn)
        # We want to ensure input + output fits within that.
        # Let's assume `max_output_tokens_for_turn` is the effective max output.
        # So, the input should ideally be `persona_total_capacity - max_output_tokens_for_turn`.
        # However, `persona_input_token_limit` from settings is a more direct control for input.
        
        # Prioritize `persona_input_token_limit` from settings for input control.
        effective_input_limit = persona_input_token_limit

        if prompt_tokens > effective_input_limit:
            logger.warning(
                f"{persona_name} prompt exceeds effective input token limit ({prompt_tokens}/{effective_input_limit}). Optimizing..."
            )
            
            optimized_prompt = self.tokenizer.truncate_to_token_limit(
                prompt, effective_input_limit, 
                truncation_indicator="\n\n[TRUNCATED - focusing on most critical aspects]"
            )
            
            logger.info(
                f"Prompt for {persona_name} truncated from {prompt_tokens} to {self.tokenizer.count_tokens(optimized_prompt)} tokens."
            )
            return optimized_prompt
        
        return prompt