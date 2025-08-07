# src/config/settings.py
from pydantic import BaseModel, Field, validator, model_validator
from typing import Self # Import Self for Python 3.11+ type hinting

class ChimeraSettings(BaseModel):
    max_retries: int = Field(default=2, ge=1, le=5)
    max_backoff_seconds: int = Field(default=30, ge=5, le=60)
    context_token_budget_ratio: float = Field(default=0.2, ge=0.05, le=0.5)
    debate_token_budget_ratio: float = Field(default=0.8, ge=0.5, le=0.95)

    # Use model_validator for cross-field validation and modification
    @model_validator(mode='after')
    def normalize_token_budget_ratios(self) -> Self:
        """
        Auto-normalizes token budget ratios to sum to 1.0 while preserving
        user intent and respecting defined boundaries.
        """
        context_ratio = self.context_token_budget_ratio
        debate_ratio = self.debate_token_budget_ratio
        
        # Check if the ratios are already close enough to 1.0
        # Use a small tolerance for floating-point comparisons
        if abs(context_ratio + debate_ratio - 1.0) < 0.001:
            # Ratios are already valid, return self without modification
            return self
        
        # If they don't sum to 1.0, preserve the user's relative preference
        # by normalizing them based on their current proportion.
        total = context_ratio + debate_ratio
        
        # Calculate the normalized proportion for the context ratio
        # Ensure total is not zero to avoid division by zero.
        normalized_context_ratio = 0.0
        if total > 0.0001: # Use a small threshold to avoid division by near-zero
            normalized_context_ratio = context_ratio / total
        
        # Apply the boundaries to the normalized ratios
        # Context ratio should be between 0.05 and 0.5
        final_context_ratio = max(0.05, min(0.5, normalized_context_ratio))
        
        # Debate ratio is derived from the context ratio to ensure they sum to 1.0
        # Debate ratio should be between 0.5 and 0.95
        final_debate_ratio = max(0.5, min(0.95, 1.0 - final_context_ratio))
        
        # Re-normalize if the debate ratio adjustment pushed the context ratio out of bounds
        # This ensures the sum is exactly 1.0 and boundaries are respected.
        if abs(final_context_ratio + final_debate_ratio - 1.0) > 0.001:
             # Recalculate context ratio based on the debate ratio's boundary
             final_context_ratio = max(0.05, min(0.5, 1.0 - final_debate_ratio))
             # Ensure final_debate_ratio is also updated if context ratio changed
             final_debate_ratio = 1.0 - final_context_ratio

        # Update the model's fields with the normalized and boundary-checked values
        self.context_token_budget_ratio = final_context_ratio
        self.debate_token_budget_ratio = final_debate_ratio
        
        return self