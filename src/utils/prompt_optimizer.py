# src/utils/prompt_optimizer.py
import logging
from typing import Dict, Any, List, Optional
from src.llm_tokenizers.base import Tokenizer # MODIFIED: Updated import path
from src.config.settings import ChimeraSettings # Import ChimeraSettings
import re # Added for prompt section parsing
import json # NEW: Import json for debate history optimization

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
        # The `max_output_tokens_for_turn` is the *actual` budget for the output,
        # so we should consider it.
        
        # Calculate a soft limit for the input prompt based on the persona's overall capacity
        # and the expected output size.
        # A persona's total capacity is its max_tokens (from PersonaConfig, which is passed to _execute_llm_turn)
        # We want to ensure input + output fits within that.
        # Let's assume `max_output_tokens_for_turn` is the effective max output.
        # So, the input should ideally be `persona_total_capacity - max_output_tokens_for_turn`.
        
        # Prioritize `persona_input_token_limit` from settings for input control.
        effective_input_limit = persona_input_token_limit

        # --- REMOVED: Enhanced logic for Self_Improvement_Analyst ---
        # The complex regex-based optimization for Self_Improvement_Analyst is removed.
        # After cleaning personas.yaml, the system prompt will be concise enough
        # that generic truncation is sufficient and more robust.
        # --- End REMOVED logic ---

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
        
        return prompt # Return original prompt if no truncation needed

    def optimize_debate_history(self, debate_history_json_str: str, max_tokens: int) -> str:
        """
        Dynamically optimizes debate history by summarizing or prioritizing turns.
        This is a conceptual implementation that would involve an LLM call for summarization
        or more advanced heuristics.
        """
        current_tokens = self.tokenizer.count_tokens(debate_history_json_str)
        if current_tokens <= max_tokens:
            return debate_history_json_str

        # Placeholder for actual intelligent summarization logic
        # In a real scenario, this would involve:
        # 1. Sending the debate_history_json_str to a smaller LLM for summarization.
        # 2. Using semantic search to pick the most relevant turns.
        # 3. Prioritizing turns with 'conflict_found' or 'CODE_CHANGES_SUGGESTED'.

        # For now, a simple truncation as a fallback
        logger.warning(f"Debate history too long ({current_tokens} tokens). Applying aggressive truncation to fit {max_tokens} tokens.")
        return self.tokenizer.truncate_to_token_limit(
            debate_history_json_str,
            max_tokens,
            truncation_indicator="\n[...debate history further summarized/truncated...]\\n"
        )