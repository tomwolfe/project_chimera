# src/utils/prompt_optimizer.py
import logging
from typing import Dict, Any, List, Optional, Tuple
from src.llm_tokenizers.base import Tokenizer
from src.config.settings import ChimeraSettings
import re
import json

# REMOVED: from transformers import pipeline # Removed as pipeline will be passed in
# REMOVED: _summarizer = None # Removed global variable
# REMOVED: get_summarizer() function # Removed global getter function

logger = logging.getLogger(__name__)


class PromptOptimizer:
    """Optimizes prompts for various personas based on context and token limits."""

    def __init__(self, tokenizer: Tokenizer, settings: ChimeraSettings, summarizer_pipeline: Any):
        """
        Initializes the PromptOptimizer.

        Args:
            tokenizer: An instance of a Tokenizer.
            settings: An instance of ChimeraSettings.
            summarizer_pipeline: An instance of the Hugging Face summarization pipeline.
        """
        self.tokenizer = tokenizer
        self.settings = settings
        self.summarizer_pipeline = summarizer_pipeline # Store the passed pipeline instance

    def _summarize_text(self, text: str, target_tokens: int) -> str:
        """Summarizes text to a target token count using a pre-trained model."""
        if not self.summarizer_pipeline:
            logger.error("Summarizer pipeline is not initialized. Cannot summarize text.")
            # Fallback to simple truncation if summarizer is not available
            return self.tokenizer.truncate_to_token_limit(text, target_tokens)

        try:
            # Use the stored pipeline instance
            # Determine max_length and min_length for the summarizer based on target_tokens
            # The summarizer model has its own max input length (e.g., 1024 for distilbart-cnn-6-6).
            # We need to ensure the input to the summarizer is within its limits, and the output
            # is within the target_tokens.

            # Max input tokens for the summarizer model itself.
            # This is a common limit for many summarization models.
            max_input_tokens_for_summarizer_model = 1024

            # Pre-truncate the input text if it exceeds the summarizer's input limit.
            if self.tokenizer.count_tokens(text) > max_input_tokens_for_summarizer_model:
                logger.warning(
                    f"Input text for summarizer is too long ({self.tokenizer.count_tokens(text)} tokens). Pre-truncating to {max_input_tokens_for_summarizer_model} tokens."
                )
                text = self.tokenizer.truncate_to_token_limit(
                    text, max_input_tokens_for_summarizer_model
                )

            # Determine output length for the summarizer.
            # The summarizer's max_length is in tokens for its output.
            # We want the *final* summary to be <= target_tokens.
            # So, we ask the summarizer for a length up to its own max, and then truncate its output.
            summarizer_max_output_tokens = min(
                target_tokens, 142
            )  # Model's typical max output (e.g., 142 for distilbart-cnn-6-6)
            summarizer_min_output_tokens = max(
                5, int(target_tokens * 0.5)
            )  # Ensure a minimum, at least 5

            # Ensure max is at least min
            summarizer_max_output_tokens = max(
                summarizer_max_output_tokens, summarizer_min_output_tokens
            )

            summary_result = self.summarizer_pipeline(
                text,
                max_length=summarizer_max_output_tokens,
                min_length=summarizer_min_output_tokens,
                do_sample=False,  # For deterministic output
            )
            summary = summary_result[0]["summary_text"]

            # Use the tokenizer to ensure the final summary fits the target_tokens
            final_summary = self.tokenizer.truncate_to_token_limit(
                summary, target_tokens
            )
            if not final_summary and text.strip():  # If final_summary is empty but original text wasn't
                return "[...summary truncated due to token limits...]"  # Return a placeholder
            return final_summary
        except Exception as e:
            logger.error(
                f"Summarization failed: {e}. Falling back to truncation.", exc_info=True
            )
            # Fallback to simple truncation if summarization fails
            return self.tokenizer.truncate_to_token_limit(text, target_tokens)

    def optimize_prompt(
        self, prompt: str, persona_name: str, max_output_tokens_for_turn: int
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
        persona_input_token_limit = self.settings.max_tokens_per_persona.get(
            persona_name, self.settings.default_max_input_tokens_per_persona
        )

        # Prioritize `persona_input_token_limit` from settings for input control.
        effective_input_limit = persona_input_token_limit

        # Ensure a minimum effective_input_limit to prevent accidental truncation to 0 tokens
        MIN_EFFECTIVE_INPUT_LIMIT = 50  # tokens
        effective_input_limit = max(effective_input_limit, MIN_EFFECTIVE_INPUT_LIMIT)

        if prompt_tokens > effective_input_limit:
            logger.warning(
                f"{persona_name} prompt exceeds effective input token limit ({prompt_tokens}/{effective_input_limit}). Optimizing..."
            )

            # Use summarization for optimization
            optimized_prompt = self._summarize_text(prompt, effective_input_limit)

            logger.info(
                f"Prompt for {persona_name} optimized from {prompt_tokens} to {self.tokenizer.count_tokens(optimized_prompt)} tokens."
            )
            return optimized_prompt

        return prompt

    def optimize_debate_history(
        self, debate_history_json_str: str, max_tokens: int
    ) -> str:
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
        logger.warning(
            "Debate history too long (%s tokens). Applying aggressive summarization/truncation to fit %s tokens.",
            current_tokens,
            max_tokens,
        )
        return self.tokenizer.truncate_to_token_limit(
            debate_history_json_str,
            max_tokens,
            truncation_indicator="\n[...debate history further summarized/truncated...]\n",
        )