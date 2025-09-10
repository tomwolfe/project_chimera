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
            "initial_problem": r"(Initial Problem:.*?)(?=\n\nRelevant Code Context:|\n\nDebate History:|\n\nObjective Metrics and Analysis:|\n\n---|\Z)",
            "relevant_code_context": r"(Relevant Code Context:.*?)(?=\n\nDebate History:|\n\nObjective Metrics and Analysis:|\n\n---|\Z)",
            "debate_history": r"(Debate History:.*?)(?=\n\nObjective Metrics and Analysis:|\n\n---|\Z)",
            "objective_metrics": r"(Objective Metrics and Analysis:.*?)(?=\n\n---|\Z)",
            "previous_debate_output_summary": r"(Previous Debate Output Summary \(with issues\):.*?)(?=\n\n---|\Z)",
            "previous_debate_output": r"(Previous Debate Output:.*?)(?=\n\n---|\Z)",
            "conflict_resolution_summary": r"(Conflict Resolution Summary:.*?)(?=\n\n---|\Z)",
            "unresolved_conflict": r"(Unresolved Conflict:.*?)(?=\n\n---|\Z)",
        }

        # Extract sections in a defined order (e.g., keep initial problem and core instructions, truncate context/history)
        extracted_sections: Dict[str, str] = {}
        for key, pattern in sections_to_optimize.items():
            match = re.search(pattern, prompt, re.DOTALL)
            if match:
                extracted_sections[key] = match.group(1).strip()
                # Remove extracted part from prompt to avoid re-matching
                prompt = prompt.replace(match.group(1), "", 1)

        # Reconstruct the prompt, prioritizing critical parts
        # 1. Initial Problem (always keep as much as possible)
        if "initial_problem" in extracted_sections:
            optimized_prompt_parts.append(extracted_sections["initial_problem"])

        # 2. Conflict Resolution (important for self-correction)
        if "conflict_resolution_summary" in extracted_sections:
            optimized_prompt_parts.append(extracted_sections["conflict_resolution_summary"])
        if "unresolved_conflict" in extracted_sections:
            optimized_prompt_parts.append(extracted_sections["unresolved_conflict"])

        # 3. Previous Debate Output (summarized if possible)
        if "previous_debate_output_summary" in extracted_sections:
            optimized_prompt_parts.append(extracted_sections["previous_debate_output_summary"])
        elif "previous_debate_output" in extracted_sections:
            # Attempt to summarize the previous output if it's too long
            prev_output_content = extracted_sections["previous_debate_output"]
            # Heuristic: allocate a small portion of the remaining budget for this
            remaining_budget_for_prev_output = max(500, effective_input_limit // 5)
            optimized_prev_output = self.tokenizer.truncate_to_token_limit(
                prev_output_content, remaining_budget_for_prev_output,
                truncation_indicator="\n[...previous output truncated...]"
            )
            optimized_prompt_parts.append(optimized_prev_output)

        # 4. Objective Metrics (summarized if possible)
        if "objective_metrics" in extracted_sections:
            metrics_content = extracted_sections["objective_metrics"]
            remaining_budget_for_metrics = max(500, effective_input_limit // 4)
            optimized_metrics = self.tokenizer.truncate_to_token_limit(
                metrics_content, remaining_budget_for_metrics,
                truncation_indicator="\n[...metrics truncated...]"
            )
            optimized_prompt_parts.append(optimized_metrics)

        # 5. Debate History (most aggressive truncation)
        if "debate_history" in extracted_sections:
            history_content = extracted_sections["debate_history"]
            remaining_budget_for_history = max(200, effective_input_limit // 8)
            optimized_history = self.optimize_debate_history(
                history_content, remaining_budget_for_history
            )
            optimized_prompt_parts.append(optimized_history)

        # 6. Relevant Code Context (truncate if still too long)
        if "relevant_code_context" in extracted_sections:
            context_content = extracted_sections["relevant_code_context"]
            remaining_budget_for_context = max(1000, effective_input_limit // 3)
            optimized_context = self.tokenizer.truncate_to_token_limit(
                context_content, remaining_budget_for_context,
                truncation_indicator="\n[...code context truncated...]"
            )
            optimized_prompt_parts.append(optimized_context)

        # Combine and re-evaluate
        final_optimized_prompt = "\n\n".join(optimized_prompt_parts)
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