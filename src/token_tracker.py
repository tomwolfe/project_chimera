# src/token_tracker.py
import time
from typing import Dict, List, Optional

class TokenUsageTracker:
    """Tracks token usage across the reasoning process with predictive capabilities."""
    
    def __init__(self, budget: int = 128000):
        self.budget = budget
        self.current_usage = 0
        self.usage_history = []  # (timestamp, tokens_used)
        self.persona_token_map = {}  # persona_name: total_tokens
        
    def record_usage(self, tokens: int, persona: Optional[str] = None):
        """Record token usage with optional persona attribution."""
        self.current_usage += tokens
        self.usage_history.append((time.time(), tokens))
        
        if persona:
            if persona not in self.persona_token_map:
                self.persona_token_map[persona] = 0
            self.persona_token_map[persona] += tokens
            
    def get_consumption_rate(self) -> float:
        """Calculate current token consumption rate as percentage of budget."""
        if not self.usage_history or self.budget == 0:
            return 0.0
        return min(1.0, self.current_usage / self.budget)
    
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
        self.current_usage = 0
        self.usage_history = []
        self.persona_token_map = {}
