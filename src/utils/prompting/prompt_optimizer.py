# src/utils/prompting/prompt_optimizer.py
import logging
from typing import Dict, Any, Optional, Tuple, TYPE_CHECKING  # ADDED TYPE_CHECKING
from jinja2 import Environment, FileSystemLoader, Template
from src.llm_tokenizers.base import Tokenizer
from src.llm_tokenizers.gemini_tokenizer import GeminiTokenizer
from src.config.settings import ChimeraSettings
import re
from collections import defaultdict
from pathlib import Path

# NEW IMPORTS: For internal optimization decisions
from src.models import PersonaConfig

# Use TYPE_CHECKING to avoid circular import at runtime for PersonaManager
if TYPE_CHECKING:
    from src.persona_manager import PersonaManager
    from src.token_tracker import TokenUsageTracker


logger = logging.getLogger(__name__)


class PromptOptimizer:
    """Optimizes prompts for various personas based on context and token limits."""

    def __init__(
        self,
        tokenizer: Tokenizer,
        settings: ChimeraSettings,
        summarizer_pipeline: Any,
        persona_manager: Optional[
            "PersonaManager"
        ] = None,  # MODIFIED: Type hint with quotes
        token_tracker: Optional[
            "TokenTracker"
        ] = None,  # MODIFIED: Type hint with quotes
    ):
        """Initializes the PromptOptimizer."""
        self.tokenizer = tokenizer
        self.settings = settings
        self.summarizer_pipeline = summarizer_pipeline
        self.env = Environment(
            loader=FileSystemLoader("prompts")
        )  # Load Jinja2 templates from 'prompts' directory
        logger.info("PromptOptimizer initialized with template directory: prompts")

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

        self.token_cache = defaultdict(dict)
        self.prompt_templates = {}
        self._load_prompt_templates()

        self.persona_manager = persona_manager
        self.token_tracker = token_tracker

    def _load_prompt_templates(self, template_dir: str = "prompts"):
        """Loads all Jinja2 templates from the specified directory."""
        template_path = Path(template_dir)
        if not template_path.exists():
            logger.warning(
                f"Prompt templates directory '{template_dir}' not found. No templates loaded."
            )
            return

        for template_file in template_path.glob("*.j2"):
            with open(template_file, "r", encoding="utf-8") as f:
                self.prompt_templates[template_file.stem] = f.read()
        logger.info(
            f"Loaded {len(self.prompt_templates)} prompt templates from '{template_dir}'."
        )

    def _count_tokens_robustly(self, text: str) -> int:
        """Robustly counts tokens using available tokenizer methods."""
        if hasattr(self.tokenizer, "count_tokens"):
            return self.tokenizer.count_tokens(text)
        elif hasattr(self.tokenizer, "encode"):
            return len(self.tokenizer.encode(text))
        else:
            logger.warning(
                f"Unknown tokenizer type for {type(self.tokenizer).__name__}. Falling back to character count / 4 estimate."
            )
            return len(text) // 4

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
                # The previous manual character-based pre-truncation is removed.
                # We now rely on the summarizer's tokenizer for truncation.
                # The tokenizer needs to know the model's max length to truncate properly.
                max_len = self.summarizer_model_max_input_tokens
                tokenized_input = self.summarizer_tokenizer(
                    pre_truncated_text,
                    truncation=True,
                    max_length=max_len,
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

    def generate_prompt(
        self, template_name: str, context: Optional[Dict[str, Any]] = None
    ) -> str:
        """Generates a formatted prompt from a template, rendering it with Jinja2."""
        # FIX: The template name should be the key, not the content.
        # The system_prompt_template field in PersonaConfig now holds the template *name*.
        template_content = self.prompt_templates.get(template_name)
        if not template_content:
            logger.error(
                f"Prompt template '{template_name}' not found in loaded templates."
            )
            # Return a clear error message instead of raising an unhandled exception
            return f"Error: Prompt template '{template_name}' could not be loaded."
        try:
            template = self.env.get_template(
                f"{template_name}.j2"
            )  # Use Jinja environment to get template
            return template.render(context=context or {})
        except Exception as e:
            logger.error(f"Error rendering template {template_name}: {str(e)}")
            raise

    def optimize_prompt(
        self,
        user_prompt_text: str,
        persona_config: PersonaConfig,
        max_output_tokens_for_turn: int,
        system_message_for_token_count: str = "",
        is_self_analysis_prompt: bool = False,
    ) -> str:
        """
        Optimizes a user prompt string for a specific persona based on context and token limits.
        """
        persona_name = persona_config.name

        # Check cache first
        if (
            persona_name in self.token_cache
            and user_prompt_text in self.token_cache[persona_name]
        ):
            return self.token_cache[persona_name][user_prompt_text]

        optimized_prompt_text = user_prompt_text

        # 1. Redundant phrase removal (as per report)
        optimized_prompt_text = re.sub(r"\s+", " ", optimized_prompt_text.strip())
        optimized_prompt_text = re.sub(
            r"\b(a|an|the)\b", "", optimized_prompt_text, flags=re.IGNORECASE
        )

        # 2. Ensure critical instructions are preserved/added (as per report, for self-analysis)
        if (
            is_self_analysis_prompt
            and "80/20" not in optimized_prompt_text
            and "pareto" not in optimized_prompt_text.lower()
        ):
            optimized_prompt_text += " Prioritize changes using the 80/20 principle."

        # Determine if aggressive optimization is needed internally
        aggressive_optimization_flag = False
        if (
            self.token_tracker and self.persona_manager
        ):  # Ensure dependencies are available
            persona_efficiency_score = (
                persona_config.token_efficiency_score
                if isinstance(persona_config.token_efficiency_score, (int, float))
                else 0.0
            )
            if (
                self.token_tracker.get_consumption_rate()
                > self.settings.GLOBAL_TOKEN_CONSUMPTION_THRESHOLD
            ):
                if persona_efficiency_score < getattr(
                    self.settings, "TOKEN_EFFICIENCY_SCORE_THRESHOLD", 0.7
                ):
                    aggressive_optimization_flag = True
                    logger.info(
                        f"PromptOptimizer: Aggressive token optimization triggered for {persona_name} due to high global consumption and low persona efficiency.",
                        persona=persona_name,
                        global_consumption_rate=self.token_tracker.get_consumption_rate(),
                        persona_efficiency_score=persona_efficiency_score,
                    )

        # Calculate current prompt tokens (including system message for accurate total)
        full_input_tokens = self._count_tokens_robustly(
            system_message_for_token_count + optimized_prompt_text
        )

        # Get persona-specific token limits from settings
        persona_input_token_limit = self.settings.max_tokens_per_persona.get(
            persona_name, self.settings.default_max_input_tokens_per_persona
        )

        effective_input_limit = persona_input_token_limit
        if aggressive_optimization_flag:
            # Further reduce the effective input limit if aggressive optimization is on
            effective_input_limit = max(50, int(effective_input_limit * 0.75))

        MIN_EFFECTIVE_INPUT_LIMIT = 50
        effective_input_limit = max(effective_input_limit, MIN_EFFECTIVE_INPUT_LIMIT)

        # If the full input (system + user) exceeds the persona's input limit, we need to optimize.
        if full_input_tokens > effective_input_limit:
            logger.warning(
                f"{persona_name} prompt (total input tokens: {full_input_tokens}) exceeds effective input token limit ({effective_input_limit}). Optimizing..."
            )

            system_message_tokens = self._count_tokens_robustly(
                system_message_for_token_count
            )
            available_for_user_prompt = effective_input_limit - system_message_tokens

            if available_for_user_prompt <= MIN_EFFECTIVE_INPUT_LIMIT:
                optimized_prompt_text = self.tokenizer.truncate_to_token_limit(
                    optimized_prompt_text,
                    MIN_EFFECTIVE_INPUT_LIMIT,
                    truncation_indicator="... (user prompt too long)",
                )
            else:
                optimized_prompt_text = self.tokenizer.truncate_to_token_limit(
                    optimized_prompt_text,
                    available_for_user_prompt,
                    truncation_indicator="\n... (user prompt truncated)",
                )
            logger.info(
                f"User prompt for {persona_name} optimized from {self._count_tokens_robustly(user_prompt_text)} to {self._count_tokens_robustly(optimized_prompt_text)} tokens."
            )

        # Cache result
        self.token_cache[persona_name][user_prompt_text] = optimized_prompt_text
        return optimized_prompt_text

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

        if "system_prompt" in persona_config_data and persona_name in [
            "Security_Auditor",
            "Self_Improvement_Analyst",
            "Code_Architect",
        ]:
            system_prompt = persona_config_data["system_prompt"]

            system_prompt = re.sub(
                r"You are a highly analytical AI assistant\.", "", system_prompt
            )
            system_prompt = re.sub(
                r"Provide clear and concise responses\.", "", system_prompt
            )

            token_optimization_directives = """
            **Token Optimization Instructions:**
            - Be concise but thorough
            - Avoid repeating information
            - Use bullet points for clear structure
            - Prioritize the most critical information first
            - Limit your response to the most essential points
            """
            if token_optimization_directives.strip() not in system_prompt:
                system_prompt += token_optimization_directives

            persona_config_data["system_prompt"] = system_prompt
            logger.info(
                f"Optimized system prompt for high-token persona: {persona_name}"
            )
        return persona_config_data
