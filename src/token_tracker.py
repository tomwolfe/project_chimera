import logging
import time
from collections import defaultdict
from typing import Any, Optional

logger = logging.getLogger(__name__)


class TokenUsageTracker:
    """Tracks token usage across the reasoning process."""

    def __init__(
        self,
        budget: int = 128000,
        max_history_items: int = 100,  # NEW: Add max_history_items parameter
    ):
        """Initializes the token tracker.

        Args:
            budget: The total token budget for the process.
            max_history_items: The maximum number of historical usage entries to store. # NEW: Docstring update

        """
        self.budget = budget
        self.current_usage = 0
        self.usage_history: list[
            tuple[float, int]
        ] = []  # Stores (timestamp, total_tokens_used) tuples

        # NEW: Granular tracking for cost calculation and metrics
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self.persona_granular_map: dict[str, dict[str, Any]] = defaultdict(
            lambda: {"total": 0, "prompt": 0, "completion": 0, "successful_turns": 0}
        )

        self.persona_token_map: dict[str, int] = defaultdict(
            int
        )  # Stores total tokens used per persona: persona_name: total_tokens
        self.max_history_items = max_history_items  # NEW: Store max_history_items

        # NEW: Attributes for semantic token weighting
        self.high_value_tokens = 0
        self.low_value_tokens = 0
        self._current_stage = None  # To be set by the orchestrator (e.g., core.py) to indicate the current phase (e.g., 'intermediate_reasoning', 'final_synthesis')

    def record_usage(self, prompt_tokens: int, completion_tokens: int, persona: Optional[str] = None, is_successful_turn: bool = True):
        """Records token usage, optionally attributing it to a persona.
        Also applies semantic token weighting based on the current stage.

        Args:
            prompt_tokens: The number of prompt tokens used in this interaction.
            completion_tokens: The number of completion tokens used in this interaction.
            persona: The name of the persona involved, if applicable.
            is_successful_turn: Whether the turn resulted in a successful, valid output.

        """
        tokens = prompt_tokens + completion_tokens
        self.current_usage += tokens
        self.usage_history.append((time.time(), tokens))

        self.total_prompt_tokens += prompt_tokens
        self.total_completion_tokens += completion_tokens

        # NEW: Keep only the most recent max_history_items items
        if len(self.usage_history) > self.max_history_items:
            self.usage_history = self.usage_history[-self.max_history_items :]
            logger.debug(
                f"Token usage history truncated to {self.max_history_items} items."
            )

        # Attribute tokens to persona if provided
        if persona:
            # Using defaultdict means we don't need the if persona not in self.persona_token_map check
            self.persona_granular_map[persona]["total"] += tokens
            self.persona_granular_map[persona]["prompt"] += prompt_tokens
            self.persona_granular_map[persona]["completion"] += completion_tokens
            if is_successful_turn:
                self.persona_granular_map[persona]["successful_turns"] += 1

            self.persona_token_map[persona] += tokens

        # NEW: Semantic token weighting logic
        # This logic categorizes tokens based on the current stage of processing.
        # 'final_synthesis' stage tokens are considered high-value.
        # 'intermediate_reasoning' stage tokens are considered lower-value.
        if self._current_stage == "final_synthesis":
            self.high_value_tokens += tokens  # High value tokens are those directly contributing to final output
        elif self._current_stage == "debate":  # Debate turns are intermediate reasoning
            self.low_value_tokens += (
                tokens  # Low value tokens are for intermediate steps
            )
        elif self._current_stage == "context":  # Context analysis is also intermediate
            self.low_value_tokens += tokens
        # Other stages (e.g., context analysis, initial prompt) are not explicitly categorized here
        # but contribute to current_usage and potentially persona_token_map.

    def get_consumption_rate(self) -> float:
        """Calculates the current token consumption rate as a percentage of the budget."""
        if not self.usage_history or self.budget == 0:
            return 0.0
        # Ensure division by zero is avoided and rate is capped at 1.0 (100%)
        return min(1.0, self.current_usage / self.budget)

    def get_high_consumption_personas(self, threshold: float = 0.15) -> dict[str, int]:
        """Identifies personas that are consuming a disproportionate amount of tokens.

        Args:
            threshold: The proportion of total tokens used by a persona to be considered high consumption.

        Returns:
            A dictionary mapping persona names to their token counts for high consumers.

        """
        total = sum(self.persona_token_map.values())
        if total == 0:
            return {}

        return {
            persona: tokens
            for persona, tokens in self.persona_token_map.items()
            if (tokens / total) > threshold
        }

    def reset(self):
        """Resets the tracker's state to initial values."""
        self.current_usage = 0
        self.usage_history = []
        self.persona_token_map.clear()  # Use clear for defaultdict
        # NEW: Reset semantic token counters and stage
        self.high_value_tokens = 0
        self.low_value_tokens = 0
        self._current_stage = None
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self.persona_granular_map.clear()

    def get_granular_usage_summary(self) -> dict[str, Any]:
        """Returns a detailed breakdown of token usage for cost calculation and metrics."""
        return {
            "total_tokens": self.current_usage,
            "total_prompt_tokens": self.total_prompt_tokens,
            "total_completion_tokens": self.total_completion_tokens,
            "persona_breakdown": dict(self.persona_granular_map),
            "semantic_breakdown": {
                "high_value_tokens": self.high_value_tokens,
                "low_value_tokens": self.low_value_tokens,
            },
            "current_stage": self._current_stage,
        }

    def set_current_stage(self, stage: Optional[str]):
        """Sets the current processing stage. This is crucial for semantic token weighting
        as it informs the `record_usage` method about the context of token consumption.

        Args:
            stage: The name of the current processing stage (e.g., 'intermediate_reasoning', 'final_synthesis').

        """
        self._current_stage = stage