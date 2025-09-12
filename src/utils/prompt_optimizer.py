import logging
from typing import Dict, Any, List, Optional, Tuple
from src.llm_tokenizers.base import Tokenizer
from src.config.settings import ChimeraSettings
import re
import json

# NEW IMPORTS FOR SUMMARIZATION
import tiktoken
from transformers import pipeline

logger = logging.getLogger(__name__)

# Initialize summarization pipeline once to avoid repeated loading
_summarizer = None


def get_summarizer():
    global _summarizer
    if _summarizer is None:
        # Using a smaller, faster model for summarization
        _summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-6-6")
    return _summarizer


def summarize_text(text: str, target_tokens: int) -> str:
    """Summarizes text to a target token count using a pre-trained model."""
    # Fallback to truncation if summarization model fails or is not loaded
    try:
        summarizer = get_summarizer()
        # Pre-truncate input text to a manageable size for the summarizer model
        # distilbart-cnn-6-6 has a max input length of 1024 tokens.
        # We'll use a slightly larger buffer for safety, but still well within typical model limits.
        # Heuristic: 1 token ~ 4 characters. So 1024 tokens is ~4096 characters.
        max_input_chars_for_summarizer = (
            4096 * 2
        )  # Allow a bit more, as tokenizers vary
        if len(text) > max_input_chars_for_summarizer:
            logger.warning(
                f"Input text for summarizer is too long ({len(text)} chars). Pre-truncating to {max_input_chars_for_summarizer} chars."
            )
            text = text[:max_input_chars_for_summarizer]

        # Estimate max_length for summarizer based on target_tokens (heuristic: 1 token ~ 4 chars)
        max_summary_length_words = target_tokens * 2  # Rough estimate
        min_summary_length_words = min(
            30, int(target_tokens * 0.5)
        )  # Ensure a minimum length

        # Ensure max_length is not excessively large for the summarizer model
        # distilbart-cnn-6-6 typically has a max output length around 142.
        # We need to be careful not to ask for too much.
        max_summary_length_words = min(
            max_summary_length_words, 256
        )  # Cap output length for distilbart

        summary_result = summarizer(
            text,
            max_length=max_summary_length_words,
            min_length=min_summary_length_words,
            do_sample=False,  # For deterministic output
        )
        summary = summary_result[0]["summary_text"]

        # Ensure the summary itself doesn't exceed target_tokens after generation
        encoding = tiktoken.get_encoding("cl100k_base")  # Use tiktoken for final check
        summary_tokens = len(encoding.encode(summary))

        if summary_tokens > target_tokens:
            # If summarizer still produced too much, truncate it
            return encoding.decode(encoding.encode(summary)[:target_tokens])
        return summary
    except Exception as e:
        logger.error(
            f"Summarization failed: {e}. Falling back to truncation.", exc_info=True
        )
        # Fallback to simple truncation if summarization fails
        encoding = tiktoken.get_encoding("cl100k_base")
        return encoding.decode(encoding.encode(text)[:target_tokens])


class PromptOptimizer:
    """Optimizes prompts for various personas based on context and token limits."""

    def __init__(self, tokenizer: Tokenizer, settings: ChimeraSettings):
        self.tokenizer = tokenizer
        self.settings = settings

    def _get_persona_specific_optimization_config(
        self, persona_name: str
    ) -> Tuple[Dict[str, str], List[str]]:
        """
        Returns persona-specific regex patterns for sections and their ordered keys
        for truncation priority.
        """
        # Default sections and their regex patterns (most comprehensive)
        default_sections_to_optimize = {
            "core_mission": r"(You are Project Chimera's Self-Improvement Analyst.*?)(?=\n---)",
            "critical_instruction_absolute_adherence": r"(\*\*CRITICAL INSTRUCTION: ABSOLUTE ADHERENCE TO CONFLICT RESOLUTION\*\*.*?)(?=\*\*CRITICAL INSTRUCTION:\*\*|\*\*SECURITY ANALYSIS:\*\*|---)",
            "critical_instruction_general": r"(\*\*CRITICAL INSTRUCTION:\*\*.*?)(?=\*\*SECURITY ANALYSIS:\*\*|\*\*TOKEN OPTIMIZATION \(AI Efficiency\):\*\*|---)",
            "security_analysis": r"(\*\*SECURITY ANALYSIS:\*\*.*?)(?=\*\*TOKEN OPTIMIZATION \(AI Efficiency\):\*\*|\*\*TESTING STRATEGY \(AI Robustness\):\*\*|\*\*AI REASONING QUALITY & DEBATE PROCESS IMPROVEMENT:\*\*|---)",
            "token_optimization": r"(\*\*TOKEN OPTIMIZATION \(AI Efficiency\):\*\*.*?)(?=\*\*TESTING STRATEGY \(AI Robustness\):\*\*|\*\*AI REASONING QUALITY & DEBATE PROCESS IMPROVEMENT:\*\*|---)",
            "testing_strategy": r"(\*\*TESTING STRATEGY \(AI Robustness\):\*\*.*?)(?=\*\*AI REASONING QUALITY & DEBATE PROCESS IMPROVEMENT:\*\*|---)",
            "ai_reasoning_quality": r"(\*\*AI REASONING QUALITY & DEBATE PROCESS IMPROVEMENT:\*\*.*?)(?=\n---|\Z)",
            "initial_problem": r"(Initial Problem:.*?)(?=\n\nRelevant Code Context:|\n\nDebate History:|\n\nObjective Metrics and Analysis:|\n\n---|\Z)",
            "relevant_code_context": r"(Relevant Code Context:.*?)(?=\n\nDebate History:|\n\nObjective Metrics and Analysis:|\n\n---|\Z)",
            "debate_history": r"(Debate History:.*?)(?=\n\nObjective Metrics and Analysis:|\n\n---|\Z)",
            "objective_metrics": r"(Objective Metrics and Analysis:.*?)(?=\n\n---|\Z)",
            "previous_debate_output_summary": r"(Previous Debate Output Summary \(with issues\):.*?)(?=\n\n---|\Z)",
            "previous_debate_output": r"(Previous Debate Output:.*?)(?=\n\n---|\Z)",
            "conflict_resolution_summary": r"(Conflict Resolution Summary:.*?)(?=\n\n---|\Z)",
            "unresolved_conflict": r"(Unresolved Conflict:.*?)(?=\n\n---|\Z)",
            "json_instructions": r"(---\n\s*\*\*CRITICAL JSON OUTPUT INSTRUCTIONS: ABSOLUTELY MUST BE FOLLOWED\*\*.*?)(?=\*\*CRITICAL DIFF FORMAT INSTRUCTION:\*\*|\*\*JSON Schema for .*?:\*\*|\Z)",
            "diff_instructions": r"(\*\*CRITICAL DIFF FORMAT INSTRUCTION:\*\*.*?)(?=\*\*CRITICAL REMOVE FORMAT INSTRUCTION:\*\*|\*\*JSON Schema for .*?:\*\*|\Z)",
            "remove_instructions": r"(\*\*CRITICAL REMOVE FORMAT INSTRUCTION:\*\*.*?)(?=\*\*JSON Schema for .*?:\*\*|\Z)",
            "json_schema": r"(\*\*JSON Schema for .*?:\*\*.*?)(?=\*\*Synthesize the following feedback into the specified JSON format:\*\*|\Z)",
            "synthesis_feedback_instruction": r"(\*\*Synthesize the following feedback into the specified JSON format:\*\*.*?)(?=\Z)",
        }

        # Default order of keys for reconstruction and truncation priority (later keys are truncated more aggressively)
        default_ordered_keys = [
            "core_mission",
            "critical_instruction_absolute_adherence",
            "critical_instruction_general",
            "json_instructions",
            "diff_instructions",
            "remove_instructions",
            "json_schema",
            "synthesis_feedback_instruction",
            "initial_problem",
            "conflict_resolution_summary",
            "unresolved_conflict",
            "previous_debate_output_summary",
            "previous_debate_output",
            "objective_metrics",
            "security_analysis",
            "token_optimization",
            "testing_strategy",
            "ai_reasoning_quality",
            "debate_history",
            "relevant_code_context",
        ]

        # Persona-specific adjustments
        if persona_name == "Self_Improvement_Analyst":
            # Prioritize critical instructions, schema, and self-improvement specific sections
            # Order: Core Mission, Critical Instructions, JSON Schema, Synthesis Instruction,
            # then Objective Metrics, AI Reasoning Quality, Security, Token Opt, Testing,
            # then Problem, Conflict, Debate History, Code Context.
            ordered_keys = [
                "core_mission",
                "critical_instruction_absolute_adherence",
                "critical_instruction_general",
                "json_instructions",
                "diff_instructions",
                "remove_instructions",
                "json_schema",
                "synthesis_feedback_instruction",
                "objective_metrics",  # Higher priority for Self-Improvement Analyst
                "ai_reasoning_quality",
                "security_analysis",
                "token_optimization",
                "testing_strategy",
                "initial_problem",
                "conflict_resolution_summary",
                "unresolved_conflict",
                "previous_debate_output_summary",
                "previous_debate_output",
                "debate_history",
                "relevant_code_context",
            ]
            return default_sections_to_optimize, ordered_keys
        elif persona_name == "Security_Auditor":
            # Prioritize security-related sections, then general instructions
            ordered_keys = [
                "core_mission",
                "critical_instruction_general",
                "json_instructions",
                "diff_instructions",
                "remove_instructions",
                "json_schema",
                "security_analysis",  # Higher priority for Security Auditor
                "initial_problem",
                "relevant_code_context",
                "debate_history",
                "objective_metrics",
                "token_optimization",
                "testing_strategy",
                "ai_reasoning_quality",
                "previous_debate_output_summary",
                "previous_debate_output",
                "conflict_resolution_summary",
                "unresolved_conflict",
                "synthesis_feedback_instruction",  # Lower priority for non-synthesis persona
            ]
            return default_sections_to_optimize, ordered_keys
        # Add more persona-specific configurations here if needed
        # elif persona_name == "Code_Architect":
        #     # Example: Prioritize architectural context and solutions
        #     ordered_keys = [
        #         "core_mission",
        #         "critical_instruction_general",
        #         "json_instructions",
        #         "diff_instructions",
        #         "remove_instructions",
        #         "json_schema",
        #         "relevant_code_context", # High priority for code architect
        #         "initial_problem",
        #         "debate_history",
        #         "objective_metrics",
        #         # ... other sections with lower priority
        #     ]
        #     return default_sections_to_optimize, ordered_keys
        else:
            # For other personas, use the default configuration
            return default_sections_to_optimize, default_ordered_keys

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
        persona_input_token_limit = self.settings.max_tokens_per_persona.get(
            persona_name, self.settings.default_max_input_tokens_per_persona
        )

        # Prioritize `persona_input_token_limit` from settings for input control.
        effective_input_limit = persona_input_token_limit

        # If the prompt is already within limits, no optimization needed
        if prompt_tokens <= effective_input_limit:
            return prompt

        logger.warning(
            f"{persona_name} prompt exceeds effective input token limit ({prompt_tokens}/{effective_input_limit}). Optimizing..."
        )

        # Use the new summarization function if the prompt is too long
        optimized_prompt = summarize_text(prompt, effective_input_limit)

        logger.info(
            f"Prompt for {persona_name} optimized from {prompt_tokens} to {self.tokenizer.count_tokens(optimized_prompt)} tokens."
        )
        return optimized_prompt

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
            f"Debate history too long ({current_tokens} tokens). Applying aggressive truncation to fit {max_tokens} tokens."
        )
        return self.tokenizer.truncate_to_token_limit(
            debate_history_json_str,
            max_tokens,
            truncation_indicator="\n[...debate history further summarized/truncated...]\\n",
        )
