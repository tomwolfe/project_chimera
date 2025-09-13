# src/utils/prompt_optimizer.py
import logging
from typing import Dict, Any, List, Optional, Tuple
from src.llm_tokenizers.base import Tokenizer
from src.config.settings import ChimeraSettings
import re
import json

logger = logging.getLogger(__name__)


class PromptOptimizer:
    """Optimizes prompts for various personas based on context and token limits."""

    def __init__(self, tokenizer: Tokenizer, settings: ChimeraSettings, summarizer_pipeline: Any):
        """
        Initializes the PromptOptimizer.

        Args:
            tokenizer: An instance of a Tokenizer (GeminiTokenizer).
            settings: An instance of ChimeraSettings.
            summarizer_pipeline: An instance of the Hugging Face summarization pipeline.
        """
        self.tokenizer = tokenizer # This is the Gemini tokenizer
        self.settings = settings
        self.summarizer_pipeline = summarizer_pipeline # Store the passed pipeline instance

        # NEW: Extract the tokenizer from the summarizer pipeline
        # This is crucial for consistent token counting for the summarizer's input.
        if hasattr(self.summarizer_pipeline, 'tokenizer'):
            self.summarizer_tokenizer = self.summarizer_pipeline.tokenizer
            # The model_max_length attribute gives the maximum input sequence length for the model's tokenizer.
            self.summarizer_model_max_input_tokens = self.summarizer_tokenizer.model_max_length
            logger.info(f"Summarizer tokenizer initialized with max input length: {self.summarizer_model_max_input_tokens}")
        else:
            logger.warning("Summarizer pipeline does not have a 'tokenizer' attribute. Falling back to a default max input length for summarizer.")
            self.summarizer_tokenizer = None # Fallback if not found
            self.summarizer_model_max_input_tokens = 1024 # Common default for many summarization models (e.g., distilbart)

    def _summarize_text(self, text: str, target_tokens: int) -> str:
        """Summarizes text to a target token count using a pre-trained model."""
        if not self.summarizer_pipeline:
            logger.error("Summarizer pipeline is not initialized. Cannot summarize text.")
            # Fallback to simple truncation if summarizer is not available
            return self.tokenizer.truncate_to_token_limit(text, target_tokens)

        try:
            # Use the summarizer's own tokenizer for accurate pre-truncation of input
            pre_truncated_text = text
            if self.summarizer_tokenizer:
                # Tokenize the text using the summarizer's tokenizer, applying truncation
                tokenized_input = self.summarizer_tokenizer(
                    text,
                    truncation=True,
                    max_length=self.summarizer_model_max_input_tokens,
                    return_tensors="pt" # Return PyTorch tensors
                )
                # Decode back to string to get the pre-truncated text
                pre_truncated_text = self.summarizer_tokenizer.decode(
                    tokenized_input['input_ids'][0],
                    skip_special_tokens=True # Remove special tokens like [CLS], [SEP]
                )
                
                # Log if pre-truncation occurred
                if len(tokenized_input['input_ids'][0]) > self.summarizer_model_max_input_tokens:
                    logger.warning(
                        f"Input text for summarizer was pre-truncated using summarizer's tokenizer. "
                        f"Original tokens (approx): {self.tokenizer.count_tokens(text)}, "
                        f"Summarizer's max input: {self.summarizer_model_max_input_tokens}."
                    )
            else:
                # Fallback to Gemini tokenizer if summarizer_tokenizer is not available
                # This path is less ideal but provides a safeguard.
                pre_truncated_text = self.tokenizer.truncate_to_token_limit(
                    text, self.summarizer_model_max_input_tokens
                )
                if self.tokenizer.count_tokens(text) > self.summarizer_model_max_input_tokens:
                    logger.warning(
                        f"Input text for summarizer is too long ({self.tokenizer.count_tokens(text)} tokens). "
                        f"Pre-truncating to {self.summarizer_model_max_input_tokens} tokens using Gemini tokenizer (fallback)."
                    )
            
            # Determine output length for the summarizer.
            # The summarizer's max_length is in tokens for its output.
            # We want the *final* summary (measured in Gemini tokens) to be <= target_tokens.
            # The distilbart-cnn-6-6 model has a typical max output of 142 tokens.
            # We set the summarizer's internal max_length to a reasonable value, then re-truncate
            # its output using the Gemini tokenizer to fit the overall budget.
            
            # Use a conservative upper bound for the summarizer's internal output,
            # typically around 1/3 to 1/2 of the input length or the model's default.
            # For distilbart-cnn-6-6, 142 is a common max output.
            summarizer_internal_max_output_tokens = min(target_tokens, self.summarizer_model_max_input_tokens) 
            summarizer_internal_min_output_tokens = max(5, int(target_tokens * 0.2)) # Ensure a minimum, at least 5

            # Ensure max is at least min
            summarizer_internal_max_output_tokens = max(
                summarizer_internal_max_output_tokens, summarizer_internal_min_output_tokens
            )

            summary_result = self.summarizer_pipeline(
                pre_truncated_text, # Use the pre-truncated text here
                max_length=summarizer_internal_max_output_tokens,
                min_length=summarizer_internal_min_output_tokens,
                do_sample=False,  # For deterministic output
            )
            summary = summary_result[0]["summary_text"]

            # Use the Gemini tokenizer to ensure the final summary fits the overall target_tokens
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
        # Calculate current prompt tokens using the Gemini tokenizer
        prompt_tokens = self.tokenizer.count_tokens(prompt)

        # Get persona-specific token limits from settings
        persona_input_token_limit = self.settings.max_tokens_per_persona.get(
            persona_name, self.settings.default_max_input_tokens_per_persona
        )

        effective_input_limit = persona_input_token_limit

        # Ensure a minimum effective_input_limit to prevent accidental truncation to 0 tokens
        MIN_EFFECTIVE_INPUT_LIMIT = 50  # tokens
        effective_input_limit = max(effective_input_limit, MIN_EFFECTIVE_INPUT_LIMIT)

        if prompt_tokens > effective_input_limit:
            logger.warning(
                f"{persona_name} prompt exceeds effective input token limit ({prompt_tokens}/{effective_input_limit}). Optimizing..."
            )

            # Use summarization for optimization
            # The target_tokens for summarization should be `effective_input_limit`
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
        """
        current_tokens = self.tokenizer.count_tokens(debate_history_json_str)
        if current_tokens <= max_tokens:
            return debate_history_json_str

        logger.warning(
            "Debate history too long (%s tokens). Applying aggressive summarization/truncation to fit %s tokens.",
            current_tokens,
            max_tokens,
        )
        # Use the summarizer pipeline for debate history as well
        return self._summarize_text(debate_history_json_str, max_tokens)