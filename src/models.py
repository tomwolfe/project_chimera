# src/models.py
from typing import Dict, Any, Optional, List
from pydantic import BaseModel, Field, validator, model_validator
import logging # Added for logger
import re # Added for regex in file_path validation

# --- MODIFICATION FOR IMPROVEMENT 4.3 ---
# Import models from src.models
# from src.models import PersonaConfig # Already imported implicitly by being in the same file
# --- END MODIFICATION ---

logger = logging.getLogger(__name__) # Initialize logger

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
    version: int = 1

    @model_validator(mode='after')
    def validate_persona_sets_references(self):
        # This validation is more relevant in core.py where all_personas are known
        return self

# NEW: Pydantic model for Context_Aware_Assistant's output
class ContextAnalysisOutput(BaseModel):
    key_modules: List[Dict[str, Any]] = Field(..., alias="key_modules", description="List of important modules/files and their purpose.")
    security_concerns: List[str] = Field(..., alias="security_concerns", description="List of potential security issues or patterns.")
    architectural_patterns: List[str] = Field(..., alias="architectural_patterns", description="List of observed architectural patterns or design principles.")
    performance_bottlenecks: List[str] = Field(..., alias="performance_bottlenecks", description="List of potential performance issues or areas for optimization.")

# --- MODIFICATION FOR IMPROVEMENT 4.3 ---
# Moved LLMOutput and CodeChange definitions here for centralization.
class CodeChange(BaseModel):
    file_path: str = Field(..., alias="FILE_PATH")
    action: str = Field(..., alias="ACTION")
    full_content: Optional[str] = Field(None, alias="FULL_CONTENT")
    lines: List[str] = Field(default_factory=list, alias="LINES")

    @validator('file_path')
    def validate_file_path(cls, v):
        """Validates and sanitizes the file path."""
        # This validator relies on src.utils.path_utils.sanitize_and_validate_file_path
        # For better separation, this validation might be better handled outside the model
        # or by passing a validation callable. For now, keeping the import here.
        try:
            from src.utils.path_utils import sanitize_and_validate_file_path
            return sanitize_and_validate_file_path(v)
        except ImportError:
            # Fallback if path_utils is not available during model definition
            logger.warning("Could not import path_utils for file_path validation. Proceeding without strict validation.")
            return v
        except ValueError as ve:
            raise ValueError(f"Invalid file path: {ve}") from ve
        
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
    # Add malformed_blocks field for parser feedback (as per Improvement 2.2)
    malformed_blocks: List[Dict[str, Any]] = Field(default_factory=list, alias="malformed_blocks")

# NEW: Pydantic model for general critique output
class CritiqueOutput(BaseModel):
    critique_summary: str = Field(..., alias="CRITIQUE_SUMMARY", description="A concise summary of the critique.")
    critique_points: List[Dict[str, Any]] = Field(..., alias="CRITIQUE_POINTS", description="Detailed points of critique.")
    suggestions: List[str] = Field(default_factory=list, alias="SUGGESTIONS", description="Actionable suggestions for improvement.")
    malformed_blocks: List[Dict[str, Any]] = Field(default_factory=list, alias="malformed_blocks")
# --- END MODIFICATION ---

# --- ADDED DEFINITION FOR CHIMERASETTINGS ---
class ChimeraSettings(BaseModel):
    """
    Default configuration settings for the Chimera application.
    These can be overridden by explicit arguments or configuration files.
    """
    # These fields are based on common parameters used in core.py and app.py
    # and are assumed to be part of default settings if not provided otherwise.
    default_model_name: str = "gemini-2.5-flash-lite"
    default_max_tokens_budget: int = 10000
    default_context_token_budget_ratio: float = 0.25
    # Add any other default settings that ChimeraSettings is intended to manage.
# --- END ADDED DEFINITION ---