# src/config/settings.py
from pydantic import BaseModel, Field, validator, model_validator
from typing import Self # Import Self for Python 3.11+ type hinting

class ChimeraSettings(BaseModel):
    max_retries: int = Field(default=2, ge=1, le=5)
    max_backoff_seconds: int = Field(default=30, ge=5, le=60)
    context_token_budget_ratio: float = Field(default=0.2, ge=0.05, le=0.5)
    debate_token_budget_ratio: float = Field(default=0.8, ge=0.5, le=0.95)
    self_analysis_context_ratio: float = Field(default=0.35, ge=0.1, le=0.6)
    self_analysis_debate_ratio: float = Field(default=0.65, ge=0.4, le=0.9)
    # ADD THIS LINE: Define the total_budget field
    total_budget: int = Field(default=10000, ge=1000, le=1000000) # Default to 10k, but allow up to 1M

    @model_validator(mode='after')
    def normalize_token_budget_ratios(self) -> Self:
        # Calculate sum of ratios
        total_ratio = (
            self.context_token_budget_ratio + 
            self.debate_token_budget_ratio
        )
        
        # Handle self-analysis ratios separately
        self_analysis_total = (
            self.self_analysis_context_ratio +
            self.self_analysis_debate_ratio
        )
        
        # Normalize ratios to sum to 1.0 while maintaining proportions
        if total_ratio > 0:
            ratio_factor = 1.0 / total_ratio
            final_context_ratio = self.context_token_budget_ratio * ratio_factor
            final_debate_ratio = self.debate_token_budget_ratio * ratio_factor
        else:
            final_context_ratio = 0.2
            final_debate_ratio = 0.8
        
        if self_analysis_total > 0:
            self_analysis_ratio_factor = 1.0 / self_analysis_total
            final_self_analysis_context_ratio = self.self_analysis_context_ratio * self_analysis_ratio_factor
            final_self_analysis_debate_ratio = self.self_analysis_debate_ratio * self_analysis_ratio_factor
        else:
            final_self_analysis_context_ratio = 0.35
            final_self_analysis_debate_ratio = 0.65
        
        # Update the model's fields
        self.context_token_budget_ratio = final_context_ratio
        self.debate_token_budget_ratio = final_debate_ratio
        self.self_analysis_context_ratio = final_self_analysis_context_ratio
        self.self_analysis_debate_ratio = final_self_analysis_debate_ratio
        
        return self