# src/utils/prompt_optimizer.py
import logging
from typing import Dict, Any, List, Optional, Tuple
from src.llm_tokenizers.base import Tokenizer
from src.llm_tokenizers.gemini_tokenizer import GeminiTokenizer # NEW: Import GeminiTokenizer
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
        self, conversation_history: List[Dict[str, str]], persona_name: str, max_output_tokens_for_turn: int, system_message: str = ""
    ) -> List[Dict[str, str]]:
        """
        Optimizes a list of messages (conversation history + optional system message)
        for a specific persona based on actual token usage and persona-specific limits.
        This method aims to reduce the input prompt size if it, combined with the expected output,
        exceeds a reasonable threshold for the persona, or if the overall token budget is constrained.
        It returns an optimized list of message dictionaries.
        """
        # Combine system message and conversation history into a single list of messages
        messages_for_optimization = []
        if system_message:
            messages_for_optimization.append({"role": "system", "content": system_message})
        messages_for_optimization.extend(conversation_history)

        # Calculate current prompt tokens using the Gemini tokenizer on the list of messages
        prompt_tokens = self.tokenizer.count_tokens_from_messages(messages_for_optimization)

        # Get persona-specific token limits from settings
        persona_input_token_limit = self.settings.max_tokens_per_persona.get(
            persona_name, self.settings.default_max_input_tokens_per_persona
        )

        # The effective input limit should also consider the model's overall max input tokens
        # and the summarizer's max input tokens if summarization is to be used.
        # For now, we'll use the persona's configured limit.
        effective_input_limit = persona_input_token_limit

        MIN_EFFECTIVE_INPUT_LIMIT = 50
        effective_input_limit = max(effective_input_limit, MIN_EFFECTIVE_INPUT_LIMIT)

        if prompt_tokens > effective_input_limit:
            logger.warning(
                f"{persona_name} prompt exceeds effective input token limit ({prompt_tokens}/{effective_input_limit}). Optimizing..."
            )

            # Truncate conversation history to fit the budget.
            # Prioritize keeping the system message and the most recent messages.
            optimized_messages = []
            current_tokens = 0

            # Add system message first (if present)
            if system_message:
                system_msg_tokens = self.tokenizer.count_tokens(system_message)
                if system_msg_tokens <= effective_input_limit:
                    optimized_messages.append({"role": "system", "content": system_message})
                    current_tokens += system_msg_tokens
                else:
                    # If system message alone exceeds limit, truncate it
                    truncated_system_message = self.tokenizer.truncate_to_token_limit(
                        system_message, effective_input_limit - 50, # Reserve some for truncation indicator
                        truncation_indicator="\n... (system prompt truncated)"
                    )
                    optimized_messages.append({"role": "system", "content": truncated_system_message})
                    current_tokens += self.tokenizer.count_tokens(truncated_system_message)
                    logger.warning(f"System message for {persona_name} was truncated.")
                    return optimized_messages # Return early if system message alone fills budget

            # Iterate through conversation history in reverse to keep recent messages
            history_to_process = list(conversation_history) # Make a copy
            temp_history_messages = [] # To build up messages in reverse order

            for message in reversed(history_to_process):
                message_content = message.get("content", "")
                # Add a buffer for role and other metadata (e.g., 10 tokens)
                message_tokens = self.tokenizer.count_tokens(message_content) + 10

                if current_tokens + message_tokens <= effective_input_limit:
                    temp_history_messages.insert(0, message) # Insert at beginning to maintain original order
                    current_tokens += message_tokens
                else:
                    # If the current message is too large to fit, try to summarize it
                    remaining_budget_for_message = effective_input_limit - current_tokens
                    if remaining_budget_for_message > MIN_EFFECTIVE_INPUT_LIMIT: # Only summarize if meaningful space
                        summarized_content = self._summarize_text(
                            message_content,
                            remaining_budget_for_message - 20, # Reserve tokens for indicator
                            truncation_indicator="... (summarized)"
                        )
                        if summarized_content:
                            temp_history_messages.insert(0, {"role": message["role"], "content": summarized_content})
                            current_tokens += self.tokenizer.count_tokens(summarized_content) + 10
                            logger.debug(f"Summarized a message for {persona_name}.")
                    break # Stop adding messages once limit is reached

            optimized_messages.extend(temp_history_messages)

            logger.info(
                f"Prompt for {persona_name} optimized from {prompt_tokens} to {self.tokenizer.count_tokens_from_messages(optimized_messages)} tokens."
            )
            return optimized_messages

        # If no optimization needed, return the original messages (with system message prepended if applicable)
        return messages_for_optimization

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

    def optimize_persona_system_prompt(self, persona_config_data: Dict) -> Dict:
        """
        Optimizes a persona's system prompt by removing redundant generic instructions
        and adding specific token optimization directives for high-token personas.
        """
        persona_name = persona_config_data.get("name")
        if persona_name in ["Security_Auditor", "Self_Improvement_Analyst", "Code_Architect"]:
            system_prompt = persona_config_data["system_prompt"]

            # Remove generic instructions that aren't specific to the persona
            system_prompt = system_prompt.replace("You are a highly analytical AI assistant.", "")
            system_prompt = system_prompt.replace("Provide clear and concise responses.", "")

            # Add specific token optimization instructions
            system_prompt += """
            **Token Optimization Instructions:**
            - Be concise but thorough
            - Avoid repeating information
            - Use bullet points for clear structure
            - Prioritize the most critical information first
            - Limit your response to the most essential points
            """

            persona_config_data["system_prompt"] = system_prompt
            logger.info(f"Optimized system prompt for high-token persona: {persona_name}")
        return persona_config_data