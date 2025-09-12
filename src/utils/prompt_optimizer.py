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

    def __init__(self, tokenizer: Tokenizer, settings: ChimeraSettings):
        self.tokenizer = tokenizer
        self.settings = settings

    def _get_persona_specific_optimization_config(self, persona_name: str) -> Tuple[Dict[str, str], List[str]]:
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
                "objective_metrics", # Higher priority for Self-Improvement Analyst
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
                "security_analysis", # Higher priority for Security Auditor
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
                "synthesis_feedback_instruction", # Lower priority for non-synthesis persona
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

        # Get persona-specific optimization configuration
        sections_to_optimize_patterns, ordered_keys = self._get_persona_specific_optimization_config(persona_name)

        optimized_prompt_parts = []
        current_tokens_used = 0

        # Extract sections in the defined order of patterns
        extracted_sections: Dict[str, str] = {}
        temp_prompt = prompt
        for key, pattern in sections_to_optimize_patterns.items():
            match = re.search(pattern, temp_prompt, re.DOTALL)
            if match:
                extracted_sections[key] = match.group(0).strip()
                temp_prompt = temp_prompt.replace(match.group(0), f"__PLACEHOLDER_{key.upper()}__", 1)

        # Reconstruct the prompt based on ordered_keys, applying truncation
        remaining_budget = effective_input_limit

        for key in ordered_keys:
            if key in extracted_sections:
                section_content = extracted_sections[key]
                section_tokens = self.tokenizer.count_tokens(section_content)

                if current_tokens_used + section_tokens <= remaining_budget:
                    optimized_prompt_parts.append(section_content)
                    current_tokens_used += section_tokens
                else:
                    tokens_for_this_section = max(0, remaining_budget - current_tokens_used)
                    if tokens_for_this_section > 0:
                        if key == "debate_history":
                            truncated_section = self.optimize_debate_history(
                                section_content, tokens_for_this_section
                            )
                        else:
                            truncated_section = self.tokenizer.truncate_to_token_limit(
                                section_content,
                                tokens_for_this_section,
                                truncation_indicator="\n[...section truncated...]",
                            )
                        optimized_prompt_parts.append(truncated_section)
                        current_tokens_used += self.tokenizer.count_tokens(truncated_section)
                    else:
                        optimized_prompt_parts.append("\n[...further content truncated...]")
                    break # Stop processing further sections if budget is exhausted

        final_optimized_prompt = "\n\n".join(optimized_prompt_parts)
        final_optimized_prompt_tokens = self.tokenizer.count_tokens(final_optimized_prompt)

        if final_optimized_prompt_tokens > effective_input_limit:
            logger.warning(
                f"Prompt for {persona_name} still exceeds limit after structured optimization ({final_optimized_prompt_tokens}/{effective_input_limit}). Applying final aggressive truncation."
            )
            final_optimized_prompt = self.tokenizer.truncate_to_token_limit(
                final_optimized_prompt,
                effective_input_limit,
                truncation_indicator="\n\n[TRUNCATED - focusing on most critical aspects]",
            )

        logger.info(
            f"Prompt for {persona_name} truncated from {prompt_tokens} to {self.tokenizer.count_tokens(final_optimized_prompt)} tokens."
        )
        return final_optimized_prompt

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