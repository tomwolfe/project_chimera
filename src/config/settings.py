# src/config/settings.py
from pydantic import BaseModel, Field, validator, model_validator
from typing import Self # Import Self for Python 3.11+ type hinting

class ChimeraSettings(BaseModel):
    """
    Configuration settings for Project Chimera, including token budgeting and retry parameters.
    """
    max_retries: int = Field(default=2, ge=1, le=5, description="Maximum number of retries for LLM API calls.")
    max_backoff_seconds: int = Field(default=30, ge=5, le=60, description="Maximum backoff time in seconds for retries.")
    
    # Ratios for allocating the total token budget across different phases.
    # These are normalized to sum to 1.0 in the model_validator.
    context_token_budget_ratio: float = Field(default=0.2, ge=0.05, le=0.5, description="Proportion of total budget for context analysis.")
    debate_token_budget_ratio: float = Field(default=0.7, ge=0.5, le=0.95, description="Proportion of total budget for debate turns.")
    synthesis_token_budget_ratio: float = Field(default=0.1, ge=0.05, le=0.2, description="Proportion of total budget for final synthesis.") # ADD THIS LINE
    
    # Specific ratios for self-analysis prompts, also normalized.
    self_analysis_context_ratio: float = Field(default=0.35, ge=0.1, le=0.6, description="Proportion of total budget for context analysis during self-analysis.")
    self_analysis_debate_ratio: float = Field(default=0.65, ge=0.4, le=0.9, description="Proportion of total budget for debate turns during self-analysis.")
    
    # Total token budget for a single Socratic debate run.
    # Changed default to 1000000 to match the UI's maximum and the likely intended default.
    total_budget: int = Field(default=1000000, ge=1000, le=1000000, description="Maximum total tokens allowed for a single Socratic debate run.")

    @model_validator(mode='after')
    def normalize_token_budget_ratios(self) -> Self:
        """
        Normalizes the token budget ratios to ensure they sum to 1.0,
        maintaining their relative proportions. This is crucial for
        correctly allocating the total budget across phases.
        """
        # Normalize general ratios (context, debate, synthesis)
        total_general_ratio = (
            self.context_token_budget_ratio +
            self.debate_token_budget_ratio +
            self.synthesis_token_budget_ratio # INCLUDE SYNTHESIS
        )
        
        # Apply normalization factors
        if total_general_ratio > 0:
            ratio_factor = 1.0 / total_general_ratio
            self.context_token_budget_ratio *= ratio_factor
            self.debate_token_budget_ratio *= ratio_factor
            self.synthesis_token_budget_ratio *= ratio_factor # NORMALIZE SYNTHESIS
        else:
            # Fallback if ratios are zero or invalid
            self.context_token_budget_ratio = 0.2
            self.debate_token_budget_ratio = 0.7
            self.synthesis_token_budget_ratio = 0.1 # DEFAULT SYNTHESIS
        
        # Normalize self-analysis ratios (context, debate)
        self_analysis_total = (
            self.self_analysis_context_ratio +
            self.self_analysis_debate_ratio
        )
        
        if self_analysis_total > 0:
            self_analysis_ratio_factor = 1.0 / self_analysis_total
            self.self_analysis_context_ratio *= self_analysis_ratio_factor
            self.self_analysis_debate_ratio *= self_analysis_ratio_factor
        else:
            # Fallback for self-analysis ratios
            self.self_analysis_context_ratio = 0.35
            self.self_analysis_debate_ratio = 0.65
        
        return self