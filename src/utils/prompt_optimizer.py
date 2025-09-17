# src/utils/prompt_optimizer.py
import logging
from typing import Dict, Any, List, Optional, Tuple, Union
from jinja2 import Environment, FileSystemLoader  # ADDED
from src.llm_tokenizers.base import Tokenizer
from src.llm_tokenizers.gemini_tokenizer import GeminiTokenizer
from src.config.settings import ChimeraSettings
import re
import json


logger = logging.getLogger(__name__)


class PromptOptimizer:
    """Optimizes prompts for various personas based on context and token limits."""

    def __init__(
        self, tokenizer: Tokenizer, settings: ChimeraSettings, summarizer_pipeline: Any
    ):
        """Initializes the PromptOptimizer."""
        self.tokenizer = tokenizer
        self.settings = settings
        self.summarizer_pipeline = summarizer_pipeline
        self.env = Environment(
            loader=FileSystemLoader("templates")
        )  # NEW: Initialize Jinja2 environment
        logger.info(
            "PromptOptimizer initialized with template directory: templates"
        )  # MODIFIED to use existing logger

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

    def _count_tokens_robustly(self, text: str) -> int:
        """Robustly counts tokens using available tokenizer methods."""
        if hasattr(self.tokenizer, "count_tokens"):
            return self.tokenizer.count_tokens(text)
        elif hasattr(self.tokenizer, "encode"):
            return len(self.tokenizer.encode(text))
        else:
            # Fallback for unknown tokenizer types
            logger.warning(
                f"Unknown tokenizer type for {type(self.tokenizer).__name__}. Falling back to character count / 4 estimate."
            )
            return len(text) // 4  # Rough estimate

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
                    tokenized_input["input_ids"][0], skip_special_tokens=True
                )

                if (
                    self._count_tokens_robustly(text)
                    > self.summarizer_model_max_input_tokens
                ):
                    logger.warning(
                        f"Input text for summarizer was pre-truncated using summarizer's tokenizer. "
                        f"Original tokens (approx): {self._count_tokens_robustly(text)}, "
                        f"Summarizer's max input: {self.summarizer_model_max_input_tokens}."
                    )
            else:
                pre_truncated_text = self.tokenizer.truncate_to_token_limit(
                    text, self.summarizer_model_max_input_tokens
                )
                if (
                    self._count_tokens_robustly(text)
                    > self.summarizer_model_max_input_tokens
                ):
                    logger.warning(
                        f"Input text for summarizer is too long ({self._count_tokens_robustly(text)} tokens). "
                        f"Pre-truncating to {self.summarizer_model_max_input_tokens} tokens using Gemini tokenizer (fallback)."
                    )

            DISTILBART_MAX_OUTPUT_TOKENS = 256
            summarizer_internal_max_output_tokens = min(
                target_tokens, DISTILBART_MAX_OUTPUT_TOKENS
            )
            summarizer_internal_min_output_tokens = max(5, int(target_tokens * 0.2))

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

    def get_prompt(self, template_name: str, context: dict) -> str:  # ADDED
        """Get a formatted prompt from a template."""
        try:
            template = self.env.get_template(f"{template_name}.j2")
            return template.render(context)
        except Exception as e:
            logger.error(f"Error loading template {template_name}: {str(e)}")
            raise

    def optimize_prompt(
        self,
        user_prompt_text: str,
        persona_name: str,
        max_output_tokens_for_turn: int,
        system_message_for_token_count: str = "",
    ) -> str:
        """
        Optimizes a user prompt string for a specific persona based on context and token limits.
        This method aims to reduce the input prompt size if it, combined with the expected output,
        exceeds a reasonable threshold for the persona, or if the overall token budget is constrained.
        It returns an optimized user prompt string.
        """
        # Calculate current prompt tokens (including system message for accurate total)
        # This is the total input tokens that will be sent to the LLM
        full_input_tokens = self._count_tokens_robustly(
            system_message_for_token_count + user_prompt_text
        )

        # Get persona-specific token limits from settings
        persona_input_token_limit = self.settings.max_tokens_per_persona.get(
            persona_name, self.settings.default_max_input_tokens_per_persona
        )

        # The effective input limit is the persona's limit, but also consider the model's overall max input tokens
        # and the expected output tokens.
        # The `max_output_tokens_for_turn` is already a budget for the output.
        # So, the `effective_input_limit` is the maximum tokens allowed for the *input* part of the prompt.
        effective_input_limit = persona_input_token_limit

        MIN_EFFECTIVE_INPUT_LIMIT = 50
        effective_input_limit = max(effective_input_limit, MIN_EFFECTIVE_INPUT_LIMIT)

        # If the full input (system + user) exceeds the persona's input limit, we need to optimize.
        if full_input_tokens > effective_input_limit:
            logger.warning(
                f"{persona_name} prompt (total input tokens: {full_input_tokens}) exceeds effective input token limit ({effective_input_limit}). Optimizing..."
            )

            # Calculate how many tokens are available for the user_prompt_text
            # after accounting for the system_message.
            system_message_tokens = self._count_tokens_robustly(
                system_message_for_token_count
            )
            available_for_user_prompt = effective_input_limit - system_message_tokens

            if available_for_user_prompt <= MIN_EFFECTIVE_INPUT_LIMIT:
                # If very little space, return a very short summary or indicator for the user prompt
                return self.tokenizer.truncate_to_token_limit(
                    user_prompt_text,
                    MIN_EFFECTIVE_INPUT_LIMIT,
                    truncation_indicator="... (user prompt too long)",
                )

            # Truncate the user_prompt_text itself
            optimized_user_prompt_text = self.tokenizer.truncate_to_token_limit(
                user_prompt_text,
                available_for_user_prompt,
                truncation_indicator="\n... (user prompt truncated)",
            )
            logger.info(
                f"User prompt for {persona_name} optimized from {self._count_tokens_robustly(user_prompt_text)} to {self._count_tokens_robustly(optimized_user_prompt_text)} tokens."
            )
            return optimized_user_prompt_text

        return user_prompt_text  # No optimization needed

    def optimize_debate_history(
        self, debate_history_json_str: str, max_tokens: int
    ) -> str:
        """
        Dynamically optimizes debate history by summarizing or prioritizing turns.
        """
        current_tokens = self._count_tokens_robustly(debate_history_json_str)
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

    def optimize_persona_system_prompt(self, persona_config_data: Dict) -> Dict:
        """
        Optimizes a persona's system prompt by removing redundant generic instructions
        and adding specific token optimization directives for high-token personas.
        """
        persona_name = persona_config_data.get("name")
        if persona_name in [
            "Security_Auditor",
            "Self_Improvement_Analyst",
            "Code_Architect",
        ]:
            system_prompt = persona_config_data["system_prompt"]

            # Remove generic instructions that aren't specific to the persona
            # These are examples, adjust based on actual common redundancies
            system_prompt = re.sub(
                r"You are a highly analytical AI assistant\.", "", system_prompt
            )
            system_prompt = re.sub(
                r"Provide clear and concise responses\.", "", system_prompt
            )

            # Add specific token optimization instructions
            token_optimization_directives = """
            **Token Optimization Instructions:**
            - Be concise but thorough
            - Avoid repeating information
            - Use bullet points for clear structure
            - Prioritize the most critical information first
            - Limit your response to the most essential points
            """
            # Only add if not already present to avoid duplication on retries
            if token_optimization_directives.strip() not in system_prompt:
                system_prompt += token_optimization_directives

            persona_config_data["system_prompt"] = system_prompt
            logger.info(
                f"Optimized system prompt for high-token persona: {persona_name}"
            )
        return persona_config_data
