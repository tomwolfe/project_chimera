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

    def __init__(
        self, tokenizer: Tokenizer, settings: ChimeraSettings, summarizer_pipeline: Any
    ):
        """
        Initializes the PromptOptimizer.

        Args:
            tokenizer: An instance of a Tokenizer (GeminiTokenizer).
            settings: An instance of ChimeraSettings.
            summarizer_pipeline: An instance of the Hugging Face summarization pipeline.
        """
        self.tokenizer = tokenizer
        self.settings = settings
        self.summarizer_pipeline = summarizer_pipeline

        if hasattr(self.summarizer_pipeline, "tokenizer"):
            self.summarizer_tokenizer = self.summarizer_pipeline.tokenizer
            self.summarizer_model_max_input_tokens = (
                self.summarizer_tokenizer.model_max_length
            )
            logger.info(
                f"Summarizer tokenizer initialized with max input length: {self.summarizer_model_max_input_tokens}"
            )
        else:
            logger.warning(
                "Summarizer pipeline does not have a 'tokenizer' attribute. Falling back to a default max input length for summarizer."
            )
            self.summarizer_tokenizer = None
            self.summarizer_model_max_input_tokens = 1024

    def _summarize_text(
        self, text: str, target_tokens: int, truncation_indicator: str = ""
    ) -> str:
        """Summarizes text to a target token count using a pre-trained model."""
        if not self.summarizer_pipeline:
            logger.error(
                "Summarizer pipeline is not initialized. Cannot summarize text."
            )
            return self.tokenizer.truncate_to_token_limit(
                text, target_tokens, truncation_indicator
            )

        try:
            pre_truncated_text = text
            if self.summarizer_tokenizer:
                MAX_CHARS_FOR_SUMMARIZER_INPUT = (
                    self.summarizer_model_max_input_tokens * 4
                )
                if len(text) > MAX_CHARS_FOR_SUMMARIZER_INPUT:
                    logger.warning(
                        f"Input text for summarizer is extremely long ({len(text)} chars). Pre-truncating to {MAX_CHARS_FOR_SUMMARIZER_INPUT} chars."
                    )
                    pre_truncated_text = text[:MAX_CHARS_FOR_SUMMARIZER_INPUT]

                tokenized_input = self.summarizer_tokenizer(
                    pre_truncated_text,
                    truncation=True,
                    max_length=self.summarizer_model_max_input_tokens,
                    return_tensors="pt",
                )
                pre_truncated_text = self.summarizer_tokenizer.decode(
                    tokenized_input["input_ids"][0],
                    skip_special_tokens=True,
                )

                if (
                    self.tokenizer.count_tokens(text)
                    > self.summarizer_model_max_input_tokens
                ):
                    logger.warning(
                        f"Input text for summarizer was pre-truncated using summarizer's tokenizer. "
                        f"Original tokens (approx): {self.tokenizer.count_tokens(text)}, "
                        f"Summarizer's max input: {self.summarizer_model_max_input_tokens}."
                    )
            else:
                pre_truncated_text = self.tokenizer.truncate_to_token_limit(
                    text, self.summarizer_model_max_input_tokens
                )
                if (
                    self.tokenizer.count_tokens(text)
                    > self.summarizer_model_max_input_tokens
                ):
                    logger.warning(
                        f"Input text for summarizer is too long ({self.tokenizer.count_tokens(text)} tokens). "
                        f"Pre-truncating to {self.summarizer_model_max_input_tokens} tokens using Gemini tokenizer (fallback)."
                    )

            DISTILBART_MAX_OUTPUT_TOKENS = 256
            summarizer_internal_max_output_tokens = min(
                target_tokens, DISTILBART_MAX_OUTPUT_TOKENS
            )
            summarizer_internal_min_output_tokens = max(
                5, int(target_tokens * 0.2)
            )

            summarizer_internal_max_output_tokens = max(
                summarizer_internal_max_output_tokens,
                summarizer_internal_min_output_tokens,
            )

            summary_result = self.summarizer_pipeline(
                pre_truncated_text,
                max_length=summarizer_internal_max_output_tokens,
                min_length=summarizer_internal_min_output_tokens,
                do_sample=False,
            )
            logger.debug(
                f"Summarizer pipeline raw output length: {len(summary_result[0]['summary_text'])} chars."
            )
            summary = summary_result[0]["summary_text"]

            del summary_result

            final_summary = self.tokenizer.truncate_to_token_limit(
                summary, target_tokens, truncation_indicator
            )
            if not final_summary and text.strip():
                return "[...summary could not be generated or was too short, original content truncated...]"
            return final_summary
        except Exception as e:
            logger.error(
                f"Summarization failed: {e}. Falling back to truncation.", exc_info=True
            )
            return self.tokenizer.truncate_to_token_limit(
                text, target_tokens, truncation_indicator
            )

    def optimize_prompt(
        self, prompt: str, persona_name: str, max_output_tokens_for_turn: int
    ) -> str:
        """
        Optimizes a prompt for a specific persona based on actual token usage and persona-specific limits.
        This method aims to reduce the input prompt size if it, combined with the expected output,
        exceeds a reasonable threshold for the persona, or if the overall token budget is constrained.
        """
        # --- NEW: Apply basic text optimization (redundant phrases, sentence simplification) ---
        optimized_text_pre_truncation = prompt
        
        # Remove redundant phrases
        redundant_phrases = [
            "please", "kindly", "could you", "would you", "i would like to", 
            "i need you to", "i want you to", "i am asking you to", 
            "i am requesting you to", "i am hoping you can", "i am hoping that you can"
        ]
        for phrase in redundant_phrases:
            optimized_text_pre_truncation = optimized_text_pre_truncation.replace(phrase, "")
        
        # Simplify complex sentences
        complex_structures = [
            "in order to", "due to the fact that", "it is important to note that",
            "it should be noted that", "it is worth mentioning that", "it is clear that",
            "it is evident that", "it is apparent that", "it is obvious that"
        ]
        replacements = [
            "to", "because", "note that", "note that", "mention that", "clearly",
            "evidently", "apparently", "obviously"
        ]
        for i, structure in enumerate(complex_structures):
            optimized_text_pre_truncation = optimized_text_pre_truncation.replace(structure, replacements[i])
        
        # Remove unnecessary whitespace
        optimized_text_pre_truncation = " ".join(optimized_text_pre_truncation.split())
        # --- END NEW ---

        # Calculate current prompt tokens using the Gemini tokenizer on the pre-optimized text
        prompt_tokens = self.tokenizer.count_tokens(optimized_text_pre_truncation)

        # Get persona-specific token limits from settings
        persona_input_token_limit = self.settings.max_tokens_per_persona.get(
            persona_name, self.settings.default_max_input_tokens_per_persona
        )

        effective_input_limit = min(
            persona_input_token_limit, self.summarizer_model_max_input_tokens
        )

        MIN_EFFECTIVE_INPUT_LIMIT = 50
        effective_input_limit = max(effective_input_limit, MIN_EFFECTIVE_INPUT_LIMIT)

        if prompt_tokens > effective_input_limit:
            logger.warning(
                f"{persona_name} prompt exceeds effective input token limit ({prompt_tokens}/{effective_input_limit}). Optimizing..."
            )

            optimized_prompt = self._summarize_text(
                optimized_text_pre_truncation, # Use the pre-optimized text here
                effective_input_limit,
                truncation_indicator="\n\n[TRUNCATED - focusing on most critical aspects]",
            )

            logger.info(
                f"Prompt for {persona_name} optimized from {prompt_tokens} to {self.tokenizer.count_tokens(optimized_prompt)} tokens."
            )
            return optimized_prompt

        return optimized_text_pre_truncation # Return the pre-optimized text if no further truncation/summarization needed

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
        return self._summarize_text(
            debate_history_json_str,
            max_tokens,
            truncation_indicator="... (debate history further summarized/truncated...)",
        )