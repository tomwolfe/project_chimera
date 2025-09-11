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

        # If the prompt is already within limits, no optimization needed
        if prompt_tokens <= effective_input_limit:
            return prompt

        logger.warning(
            f"{persona_name} prompt exceeds effective input token limit ({prompt_tokens}/{effective_input_limit}). Optimizing..."
        )

        # Strategy: Aggressively truncate less critical sections first.
        # This requires parsing the prompt structure.
        # For self-improvement prompts, the structure is well-defined.
        # For other prompts, a more general approach is needed.

        optimized_prompt_parts = []
        current_tokens_after_optimization = 0

        # Define sections and their priority for truncation (lower index = higher priority to keep)
        # This is a heuristic and can be refined.
        sections_to_optimize = {
            "core_mission": r"(You are Project Chimera's Self-Improvement Analyst.*?)\n---",
            "critical_instruction_absolute_adherence": r"(\*\*CRITICAL INSTRUCTION: ABSOLUTE ADHERENCE TO CONFLICT RESOLUTION\*\*.*?)(?=\*\*CRITICAL INSTRUCTION:\*\*|\*\*SECURITY ANALYSIS:\*\*|\*\*TOKEN OPTIMIZATION \(AI Efficiency\):\*\*|\*\*TESTING STRATEGY \(AI Robustness\):\*\*|\*\*AI REASONING QUALITY & DEBATE PROCESS IMPROVEMENT:\*\*|---)",
            "critical_instruction_general": r"(\*\*CRITICAL INSTRUCTION:\*\*.*?)(?=\*\*SECURITY ANALYSIS:\*\*|\*\*TOKEN OPTIMIZATION \(AI Efficiency\):\*\*|\*\*TESTING STRATEGY \(AI Robustness\):\*\*|\*\*AI REASONING QUALITY & DEBATE PROCESS IMPROVEMENT:\*\*|---)",
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
            "json_instructions": r"(---\n\s*\*\*CRITICAL JSON OUTPUT INSTRUCTIONS: ABSOLUTELY MUST BE FOLLOWED\*\*.*?)(?=\*\*CRITICAL DIFF FORMAT INSTRUCTION:\*\*|\*\*JSON Schema for SelfImprovementAnalysisOutputV1:\*\*|\Z)",
            "diff_instructions": r"(\*\*CRITICAL DIFF FORMAT INSTRUCTION:\*\*.*?)(?=\*\*CRITICAL REMOVE FORMAT INSTRUCTION:\*\*|\*\*JSON Schema for SelfImprovementAnalysisOutputV1:\*\*|\Z)",
            "remove_instructions": r"(\*\*CRITICAL REMOVE FORMAT INSTRUCTION:\*\*.*?)(?=\*\*JSON Schema for SelfImprovementAnalysisOutputV1:\*\*|\Z)",
            "json_schema": r"(\*\*JSON Schema for SelfImprovementAnalysisOutputV1:\*\*.*?)(?=\*\*Synthesize the following feedback into the specified JSON format:\*\*|\Z)",
            "synthesis_feedback_instruction": r"(\*\*Synthesize the following feedback into the specified JSON format:\*\*.*?)(?=\Z)",
        }

        # Extract sections in a defined order (higher priority first)
        extracted_sections: Dict[str, str] = {}
        temp_prompt = prompt # Use a temporary variable to extract sections
        for key, pattern in sections_to_optimize.items():
            match = re.search(pattern, temp_prompt, re.DOTALL)
            if match:
                extracted_sections[key] = match.group(0).strip() # Capture the full matched section
                # Replace the matched section with a placeholder to avoid re-matching
                temp_prompt = temp_prompt.replace(match.group(0), f"__PLACEHOLDER_{key.upper()}__", 1)

        # Define the order of sections to reconstruct the prompt, with truncation priority
        # Core instructions and schema are highest priority.
        # Then initial problem, then conflict resolution, then previous output, then metrics, then debate history, then code context.
        ordered_keys = [
            "core_mission",
            "critical_instruction_absolute_adherence",
            "critical_instruction_general",
            "security_analysis",
            "token_optimization",
            "testing_strategy",
            "ai_reasoning_quality",
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
            "debate_history",
            "relevant_code_context",
        ]

        final_optimized_prompt_parts = []
        current_tokens_used = 0

        # Calculate remaining budget for dynamic truncation
        remaining_budget = effective_input_limit

        # First, add high-priority, non-truncatable sections (e.g., core instructions, schema)
        for key in ["core_mission", "critical_instruction_absolute_adherence", "critical_instruction_general", "json_instructions", "diff_instructions", "remove_instructions", "json_schema", "synthesis_feedback_instruction"]:
            if key in extracted_sections:
                section_content = extracted_sections[key]
                section_tokens = self.tokenizer.count_tokens(section_content)
                if current_tokens_used + section_tokens <= remaining_budget:
                    final_optimized_prompt_parts.append(section_content)
                    current_tokens_used += section_tokens
                else:
                    # This should ideally not happen for critical instructions if budget is reasonable
                    logger.warning(f"Critical section '{key}' could not fit in prompt. Truncating.")
                    truncated_section = self.tokenizer.truncate_to_token_limit(section_content, remaining_budget - current_tokens_used, truncation_indicator="\n[...critical section truncated...]")
                    final_optimized_prompt_parts.append(truncated_section)
                    current_tokens_used += self.tokenizer.count_tokens(truncated_section)
                    break # Stop adding if critical section itself was truncated

        # Then, add other sections with dynamic truncation based on remaining budget
        # Order of keys here determines truncation priority (later keys are truncated more aggressively)
        dynamic_truncation_keys = [
            "initial_problem",
            "conflict_resolution_summary",
            "unresolved_conflict",
            "previous_debate_output_summary",
            "previous_debate_output",
            "objective_metrics",
            "security_analysis", # Specific analysis sections
            "token_optimization",
            "testing_strategy",
            "ai_reasoning_quality",
            "debate_history",
            "relevant_code_context",
        ]

        for key in dynamic_truncation_keys:
            if key in extracted_sections:
                section_content = extracted_sections[key]
                section_tokens = self.tokenizer.count_tokens(section_content)

                if current_tokens_used + section_tokens <= remaining_budget:
                    final_optimized_prompt_parts.append(section_content)
                    current_tokens_used += section_tokens
                else:
                    # Calculate how many tokens are left for this section
                    tokens_for_this_section = max(0, remaining_budget - current_tokens_used)
                    if tokens_for_this_section > 0:
                        if key == "debate_history":
                            truncated_section = self.optimize_debate_history(section_content, tokens_for_this_section)
                        else:
                            truncated_section = self.tokenizer.truncate_to_token_limit(
                                section_content, tokens_for_this_section,
                                truncation_indicator="\n[...section truncated...]"
                            )
                        final_optimized_prompt_parts.append(truncated_section)
                        current_tokens_used += self.tokenizer.count_tokens(truncated_section)
                    else:
                        # If no tokens left, just add a truncation indicator
                        final_optimized_prompt_parts.append("\n[...further content truncated...]")
                    break # Stop processing further sections if budget is exhausted

        final_optimized_prompt = "\n\n".join(final_optimized_prompt_parts)
        final_optimized_prompt_tokens = self.tokenizer.count_tokens(final_optimized_prompt)

        if final_optimized_prompt_tokens > effective_input_limit:
            # If still too long after structured optimization, apply a final aggressive truncation
            logger.warning(
                f"Prompt for {persona_name} still exceeds limit after structured optimization ({final_optimized_prompt_tokens}/{effective_input_limit}). Applying final aggressive truncation."
            )
            final_optimized_prompt = self.tokenizer.truncate_to_token_limit(
                final_optimized_prompt, effective_input_limit,
                truncation_indicator="\n\n[TRUNCATED - focusing on most critical aspects]"
            )

        logger.info(
            f"Prompt for {persona_name} truncated from {prompt_tokens} to {self.tokenizer.count_tokens(final_optimized_prompt)} tokens."
        )
        return final_optimized_prompt

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