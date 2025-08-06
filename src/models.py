# src/models.py
from typing import Dict, Any, Optional, List
from pydantic import BaseModel, Field, validator, model_validator

# --- Pydantic Models for Schema Validation ---
class PersonaConfig(BaseModel):
    name: str
    description: Optional[str] = None
    system_prompt: str
    temperature: float = Field(..., ge=0.0, le=1.0)
    max_tokens: int = Field(..., gt=0)

class ReasoningFrameworkConfig(BaseModel):
    framework_name: str
    personas: Dict[str, PersonaConfig]
    persona_sets: Dict[str, List[str]]
    version: int = 1 # Application's current framework schema version

    @model_validator(mode='after')
    def validate_persona_sets_references(self):
        # This validation is more relevant in core.py where all_personas are known
        return self
