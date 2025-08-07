# src/token_manager.py
import logging
from typing import Dict, Any, List
# Assuming Tokenizer ABC is available in src.tokenizers.base
from src.tokenizers.base import Tokenizer
from src.exceptions import TokenBudgetExceededError # Assuming this exception exists

logger = logging.getLogger(__name__)

class TokenManager:
    """Centralized token management for the ISAL process."""
    
    def __init__(self, llm_provider, max_total_tokens: int = 300000):
        self.llm_provider = llm_provider # Assumes llm_provider has count_tokens and estimate_tokens_for_context methods
        self.max_total_tokens = max_total_tokens
        self.used_tokens = {"context": 0, "debate": 0, "synthesis": 0}
        self.phase_budgets = {"context": 0, "debate": 0, "synthesis": 0}
        self.initial_input_tokens = 0 # To store tokens for initial prompt + context

    def _get_adaptive_phase_ratios(self, prompt: str, context_present: bool) -> Dict[str, float]:
        """Dynamically adjust phase ratios based on prompt content and context presence."""
        ratios = {"context": 0.15, "debate": 0.75, "synthesis": 0.10} # Default ratios
        prompt_lower = prompt.lower()

        # Increase context budget for code analysis or self-analysis prompts
        if "code" in prompt_lower or "analyze" in prompt_lower or "refactor" in prompt_lower or "chimera" in prompt_lower or "self-analysis" in prompt_lower:
            ratios["context"] = 0.25
            ratios["debate"] = 0.65
            ratios["synthesis"] = 0.10
        
        # If context is explicitly provided and relevant, ensure it gets a minimum budget
        if context_present and ratios["context"] < 0.10:
            ratios["context"] = 0.10

        # Normalize ratios to sum to 1.0
        total = sum(ratios.values())
        if total > 0: # Avoid division by zero
            for phase in ratios:
                ratios[phase] /= total
        else: # Fallback if total is zero (should not happen with default ratios)
            ratios = {"context": 0.15, "debate": 0.75, "synthesis": 0.10} # Reset to defaults

        return ratios

    def calculate_phase_budgets(self, context_str: str, prompt: str):
        """Calculate token budgets for each phase with adaptive ratios."""
        try:
            # Estimate tokens for initial input (context + prompt)
            # Ensure llm_provider has estimate_tokens_for_context method
            self.initial_input_tokens = self.llm_provider.estimate_tokens_for_context(context_str, prompt)
            
            # Calculate available tokens for the debate/synthesis phases
            available_tokens = max(0, self.max_total_tokens - self.initial_input_tokens)
            
            # Get adaptive ratios
            phase_ratios = self._get_adaptive_phase_ratios(prompt, bool(context_str))
            
            # Calculate phase budgets, ensuring minimums for critical phases
            self.phase_budgets["context"] = max(200, int(available_tokens * phase_ratios["context"]))
            self.phase_budgets["debate"] = max(500, int(available_tokens * phase_ratios["debate"]))
            self.phase_budgets["synthesis"] = max(400, int(available_tokens * phase_ratios["synthesis"])) # Increased min for synthesis

            logger.info(f"Token budgets calculated: Initial Input={self.initial_input_tokens}, "
                       f"Context={self.phase_budgets['context']}, Debate={self.phase_budgets['debate']}, "
                       f"Synthesis={self.phase_budgets['synthesis']}")
        except AttributeError:
            logger.error("LLM Provider is missing 'estimate_tokens_for_context' or 'count_tokens' method.")
            # Fallback to a safe, fixed budget if calculation fails due to missing methods
            self.phase_budgets = {"context": 500, "debate": 15000, "synthesis": 1000}
            logger.warning("Using fallback token budgets due to LLM provider method issues.")
        except Exception as e:
            logger.error(f"Error calculating phase budgets: {e}")
            # Fallback to a safe, fixed budget if calculation fails
            self.phase_budgets = {"context": 500, "debate": 15000, "synthesis": 1000}
            logger.warning("Using fallback token budgets due to calculation error.")

    def track_usage(self, phase: str, tokens: int):
        """Track token usage for a specific phase."""
        if phase in self.used_tokens:
            self.used_tokens[phase] += tokens
        else:
            logger.warning(f"Unknown phase '{phase}' for token tracking.")
    
    def get_remaining_tokens(self, phase: str) -> int:
        """Get remaining tokens for a phase."""
        if phase not in self.phase_budgets:
            return 0
        return max(0, self.phase_budgets[phase] - self.used_tokens[phase])
    
    def check_budget(self, phase: str, tokens_needed: int, step_name: str) -> int:
        """Check if tokens_needed exceed remaining budget for the phase."""
        remaining = self.get_remaining_tokens(phase)
        if tokens_needed > remaining:
            raise TokenBudgetExceededError(
                current_tokens=self.used_tokens[phase],
                budget=self.phase_budgets[phase],
                details={"phase": phase, "step": step_name, "tokens_requested": tokens_needed}
            )
        return tokens_needed # Return tokens needed if within budget

    def get_total_used_tokens(self) -> int:
        """Returns the sum of tokens used across all phases."""
        return sum(self.used_tokens.values())
