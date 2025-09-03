# src/token_tracker.py
import time
from typing import Dict, List, Optional


class TokenUsageTracker:
    """Tracks token usage across the reasoning process with predictive capabilities."""

    def __init__(self, budget: int = 128000):
        self.budget = budget
        self.current_usage = 0  # Total tokens used (for compatibility with older logic)
        self.usage_history = []  # (timestamp, total_tokens_for_event)
        self.persona_token_map = {}  # persona_name: total_tokens

        # NEW: For semantic token weighting and granular tracking
        self.input_tokens = 0
        self.output_tokens = 0
        self.total_tokens = 0  # Sum of input_tokens and output_tokens
        self.high_value_tokens = 0
        self.low_value_tokens = 0
        self._current_stage: Optional[str] = None # To be set by the orchestrator (e.g., core.py)

    def track_usage(self, input_tokens: int, output_tokens: int, persona: Optional[str] = None):
        """
        Record token usage with optional persona attribution and semantic weighting.
        This method replaces the previous 'record_usage'.
        """
        self.input_tokens += input_tokens
        self.output_tokens += output_tokens
        self.total_tokens = self.input_tokens + self.output_tokens

        # Update current_usage for compatibility with existing methods that might rely on it
        self.current_usage = self.total_tokens

        # Update usage history (tracks total tokens for this specific event)
        self.usage_history.append((time.time(), input_tokens + output_tokens))

        if persona:
            if persona not in self.persona_token_map:
                self.persona_token_map[persona] = 0
            self.persona_token_map[persona] += (input_tokens + output_tokens)

        # NEW: Semantic token weighting logic from the diff
        if hasattr(self, '_current_stage') and self._current_stage is not None:
            if self._current_stage == 'final_synthesis':
                self.high_value_tokens += output_tokens
            elif self._current_stage == 'intermediate_reasoning':
                self.low_value_tokens += output_tokens

    def get_consumption_rate(self) -> float:
        """Calculate current token consumption rate as percentage of budget."""
        if self.total_tokens == 0 or self.budget == 0:
            return 0.0
        return min(1.0, self.total_tokens / self.budget)

    def get_high_consumption_personas(self, threshold: float = 0.15) -> Dict[str, int]:
        """Identify personas consuming disproportionate tokens."""
        total = sum(self.persona_token_map.values())
        if total == 0:
            return {}

        return {
            persona: tokens
            for persona, tokens in self.persona_token_map.items()
            if (tokens / total) > threshold
        }

    def reset(self):
        """Resets the tracker's state."""
        self.budget = 128000 # Reset budget to default or keep current? Assuming default for full reset.
        self.current_usage = 0
        self.usage_history = []
        self.persona_token_map = {}
        
        # NEW: Reset for semantic token weighting and granular tracking
        self.input_tokens = 0
        self.output_tokens = 0
        self.total_tokens = 0
        self.high_value_tokens = 0
        self.low_value_tokens = 0
        self._current_stage = None