# src/utils/prompt_optimizer.py
import logging
from typing import Dict, Any, List, Optional, Tuple
from src.llm_tokenizers.base import Tokenizer
from src.config.settings import ChimeraSettings
import re
import json

import tiktoken
from transformers import pipeline

logger = logging.getLogger(__name__)

# Initialize summarization pipeline once to avoid repeated loading
_summarizer = None


def get_summarizer():
    """Initializes and returns the summarization pipeline, cached globally."""
    global _summarizer
    if _summarizer is None:
        logger.info(
            "Initializing Hugging Face summarization pipeline (sshleifer/distilbart-cnn-6-6). This is a one-time load."
        )
        # Using a smaller, faster model for summarization
        _summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-6-6")
    return _summarizer


class PromptOptimizer:
    """Optimizes prompts for various personas based on context and token limits."""

    def __init__(self, tokenizer: Tokenizer, settings: ChimeraSettings):
        self.tokenizer = tokenizer
        self.settings = settings

    def _summarize_text(self, text: str, target_tokens: int) -> str:
        """Summarizes text to a target token count using a pre-trained model."""
        try:
            summarizer = get_summarizer()
            # Pre-truncate input text to a manageable size for the summarizer model
            # distilbart-cnn-6-6 has a max input length of 1024 tokens.
            max_input_tokens_for_summarizer = 1024
            if self.tokenizer.count_tokens(text) > max_input_tokens_for_summarizer:
                logger.warning(
                    f"Input text for summarizer is too long ({self.tokenizer.count_tokens(text)} tokens). Pre-truncating to {max_input_tokens_for_summarizer} tokens."
                )
                text = self.tokenizer.truncate_to_token_limit(
                    text, max_input_tokens_for_summarizer
                )

            # distilbart-cnn-6-6 typically has a max output length around 142 tokens.
            # We cap the requested summary length to avoid over-generation and memory issues.
            # Target tokens is the budget for the summary.
            max_summary_length_tokens = min(target_tokens, 142)  # Cap at model's typical max output
            min_summary_length_tokens = max(5, int(target_tokens * 0.5))  # Ensure a minimum length, at least 5 tokens
            max_summary_length_tokens = max(max_summary_length_tokens, min_summary_length_tokens)  # Ensure max is at least min

            summary_result = summarizer(
                text,
                max_length=max_summary_length_tokens,
                min_length=min_summary_length_tokens,
                do_sample=False,  # For deterministic output
            )
            summary = summary_result[0]["summary_text"]

            # Use the tokenizer to ensure the final summary fits the target_tokens
            final_summary = self.tokenizer.truncate_to_token_limit(summary, target_tokens)
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
            f"Debate history too long ({current_tokens} tokens). Applying aggressive summarization/truncation to fit {max_tokens} tokens."
        )
        return self.tokenizer.truncate_to_token_limit(
            debate_history_json_str,
            max_tokens,
            truncation_indicator="\n[...debate history further summarized/truncated...]\n",
        )