# src/config/settings.py - New file
from pydantic import BaseModel, Field, validator

class ChimeraSettings(BaseModel):
    max_retries: int = Field(default=2, ge=1, le=5)
    max_backoff_seconds: int = Field(default=30, ge=5, le=60)
    context_token_budget_ratio: float = Field(default=0.2, ge=0.05, le=0.5)
    debate_token_budget_ratio: float = Field(default=0.8, ge=0.5, le=0.95)
    
    @validator('debate_token_budget_ratio')
    def validate_ratios(cls, v, values):
        context_ratio = values.get('context_token_budget_ratio', 0.2)
        if context_ratio + v != 1.0:
            raise ValueError("Token budget ratios must sum to 1.0")
        return v

# In src/core.py
from src.config.settings import ChimeraSettings

def __init__(self, settings: ChimeraSettings = None, ...):
    self.settings = settings or ChimeraSettings()
    
    # Use validated settings throughout
    self.max_retries = self.settings.max_retries
    self.max_backoff_seconds = self.settings.max_backoff_seconds
    # etc.

# In app.py - Add configuration validation
try:
    settings = ChimeraSettings(
        context_token_budget_ratio=st.session_state.context_token_budget_ratio,
        # other parameters
    )
except ValidationError as e:
    st.error(f"Invalid configuration: {e}")
    # Provide helpful guidance
