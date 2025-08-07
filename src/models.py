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

# NEW: Pydantic model for Context_Aware_Assistant's output
class ContextAnalysisOutput(BaseModel):
    key_modules: List[str] = Field(..., alias="key_modules", description="List of important modules/files and their purpose.")
    security_concerns: List[str] = Field(..., alias="security_concerns", description="List of potential security issues or patterns.")
    architectural_patterns: List[str] = Field(..., alias="architectural_patterns", description="List of observed architectural patterns or design principles.")
    performance_bottlenecks: List[str] = Field(..., alias="performance_bottlenecks", description="List of potential performance issues or areas for optimization.")

# Existing models for LLMOutput and CodeChange (assuming they are already defined in output_parser.py
# but for clarity and central definition, they should ideally be here too.
# For this exercise, I'll assume they are imported from output_parser.py as per the original code,
# but if they were defined here, output_parser.py would import them.)
# Let's move them here for better architecture.

class CodeChange(BaseModel):
    file_path: str = Field(..., alias="FILE_PATH")
    action: str = Field(..., alias="ACTION")
    full_content: Optional[str] = Field(None, alias="FULL_CONTENT")
    lines: List[str] = Field(default_factory=list, alias="LINES")

    @validator('file_path')
    def validate_file_path(cls, v):
        """Validates and sanitizes the file path."""
        # This validator relies on src.utils.path_utils.sanitize_and_validate_file_path
        # which is imported in output_parser.py. For a clean model, this dependency
        # should be handled, e.g., by passing a callable or making it a method of a class
        # that holds this logic. For now, we'll keep it as is, assuming the context.
        from src.utils.path_utils import sanitize_and_validate_file_path # Local import to avoid circular
        return sanitize_and_validate_file_path(v)
        
    @validator('action')
    def validate_action(cls, v):
        """Validates the action type."""
        valid_actions = ["ADD", "MODIFY", "REMOVE"]
        if v not in valid_actions:
            raise ValueError(f"Invalid action: '{v}'. Must be one of {valid_actions}.")
        return v

    @model_validator(mode='after')
    def check_content_based_on_action(self) -> 'CodeChange':
        """Ensures that full_content is provided for ADD/MODIFY and lines for REMOVE."""
        if self.action in ["ADD", "MODIFY"] and self.full_content is None:
            raise ValueError(f"full_content is required for action '{self.action}' on file '{self.file_path}'.")
        if self.action == "REMOVE" and not isinstance(self.lines, list):
            raise ValueError(f"lines must be a list for action 'REMOVE' on file '{self.file_path}'. Found type: {type(self.lines).__name__}.")
        return self

class LLMOutput(BaseModel):
    commit_message: str = Field(alias="COMMIT_MESSAGE")
    rationale: str = Field(alias="RATIONALE")
    code_changes: List[CodeChange] = Field(alias="CODE_CHANGES")
    conflict_resolution: Optional[str] = Field(None, alias="CONFLICT_RESOLUTION")
    unresolved_conflict: Optional[str] = Field(None, alias="UNRESOLVED_CONFLICT")