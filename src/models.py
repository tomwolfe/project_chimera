# src/models.py
from typing import Dict, Any, Optional, List, Literal # Added Literal for ConflictReport
from pydantic import BaseModel, Field, validator, model_validator
import logging
import re
from pathlib import Path

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
    
    # NEW: Fields for persona-specific summaries (Improvement 3)
    security_summary: Optional[Dict[str, Any]] = Field(None, alias="security_summary", description="Summary tailored for Security_Auditor.")
    architecture_summary: Optional[Dict[str, Any]] = Field(None, alias="architecture_summary", description="Summary tailored for Code_Architect.")
    devops_summary: Optional[Dict[str, Any]] = Field(None, alias="devops_summary", description="Summary tailored for DevOps_Engineer.")
    testing_summary: Optional[Dict[str, Any]] = Field(None, alias="testing_summary", description="Summary tailored for Test_Engineer.")
    general_overview: Optional[str] = Field(None, alias="general_overview", description="A general high-level overview of the codebase context.")


    @model_validator(mode='after')
    def validate_paths_in_context_output(self) -> 'ContextAnalysisOutput':
        """Validates file paths within key_modules for security."""
        # Import sanitize_and_validate_file_path locally here
        try:
            from src.utils.path_utils import sanitize_and_validate_file_path
        except ImportError as e:
            logger.warning(f"Could not import 'sanitize_and_validate_file_path' for ContextAnalysisOutput validation: {e}. Skipping path validation.")
            return self # Skip validation if import fails

        for module in self.key_modules:
            if 'name' in module and isinstance(module['name'], str):
                try:
                    # Apply sanitization and validation to the file path string
                    module['name'] = sanitize_and_validate_file_path(module['name'])
                except ValueError as e:
                    logger.warning(f"Invalid file path detected in ContextAnalysisOutput.key_modules: '{module['name']}' - {e}")
                    # Optionally, replace with a safe placeholder or remove the entry
                    module['name'] = f"INVALID_PATH_DETECTED:{module['name']}"
        return self

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
        # Import sanitize_and_validate_file_path locally here
        try:
            from src.utils.path_utils import sanitize_and_validate_file_path
        except ImportError as e:
            logger.warning(f"Could not import 'sanitize_and_validate_file_path' for CodeChange validation: {e}. Proceeding without strict validation.")
            return v # Proceed without strict validation if import fails
        
        try:
            return sanitize_and_validate_file_path(v)
        except ValueError as ve:
            raise ValueError(f"Invalid file path: {ve}") from ve
        
    @validator('action')
    def validate_action(cls, v):
        """Validates the action type."""
        valid_actions = ["ADD", "MODIFY", "REMOVE"] # Ensure 'APPEND' is NOT here
        if v not in valid_actions:
            raise ValueError(f"Invalid action: '{v}'. Must be one of {valid_actions}.")
        return v

    @model_validator(mode='after')
    def check_content_based_on_action(self) -> 'CodeChange':
        """Ensures that full_content is provided for ADD/MODIFY and lines for REMOVE."""
        if self.action in ["ADD", "MODIFY"] and self.full_content is None:
            raise ValueError(f"full_content is required for action '{self.action}' on file '{self.file_path}'.")
        # FIX: Ensure lines is a non-empty list for REMOVE
        if self.action == "REMOVE" and (not isinstance(self.lines, list) or not self.lines):
            raise ValueError(f"lines must be a non-empty list for action 'REMOVE' on file '{self.file_path}'. Found type: {type(self.lines).__name__} or empty.")
        return self

class LLMOutput(BaseModel):
    commit_message: str = Field(alias="COMMIT_MESSAGE")
    rationale: str = Field(alias="RATIONALE")
    code_changes: List[CodeChange] = Field(alias="CODE_CHANGES")
    conflict_resolution: Optional[str] = Field(None, alias="CONFLICT_RESOLUTION")
    unresolved_conflict: Optional[str] = Field(None, alias="UNRESOLVED_CONFLICT")
    # Add malformed_blocks field for parser feedback (as per Improvement 2.2)
    malformed_blocks: List[Dict[str, Any]] = Field(default_factory=list, alias="malformed_blocks")
    # NEW: Dedicated field for malformed items within CODE_CHANGES
    malformed_code_change_items: List[Dict[str, Any]] = Field(default_factory=list, alias="malformed_code_change_items")

# NEW: Pydantic model for general critique output
class CritiqueOutput(BaseModel):
    critique_summary: str = Field(..., alias="CRITIQUE_SUMMARY", description="A concise summary of the critique.")
    critique_points: List[Dict[str, Any]] = Field(..., alias="CRITIQUE_POINTS", description="Detailed points of critique.")
    suggestions: List[str] = Field(default_factory=list, alias="SUGGESTIONS", description="Actionable suggestions for improvement.")
    malformed_blocks: List[Dict[str, Any]] = Field(default_factory=list, alias="malformed_blocks") # Ensure malformed_blocks is present

    @model_validator(mode='after')
    def validate_paths_in_suggestions(self) -> 'CritiqueOutput':
        """Validates potential file paths within suggestions for security."""
        # Import sanitize_and_validate_file_path locally here
        try:
            from src.utils.path_utils import sanitize_and_validate_file_path
        except ImportError as e:
            logger.warning(f"Could not import 'sanitize_and_validate_file_path' for CritiqueOutput validation: {e}. Skipping path validation.")
            return self # Skip validation if import fails

        sanitized_suggestions = []
        for suggestion in self.suggestions:
            # Example: simple regex to find strings that look like file paths
            potential_paths = re.findall(r'\b(?:src|data|tests|config|custom_frameworks)[/\w.-]+\.py\b', suggestion, re.IGNORECASE)
            for path in potential_paths:
                try:
                    sanitized_path = sanitize_and_validate_file_path(path)
                    suggestion = suggestion.replace(path, sanitized_path) # Replace with sanitized version
                except ValueError as e:
                    logger.warning(f"Invalid file path detected in CritiqueOutput.suggestions: '{path}' - {e}")
                    suggestion = suggestion.replace(path, f"INVALID_PATH_DETECTED:{path}")
            sanitized_suggestions.append(suggestion)
        self.suggestions = sanitized_suggestions
        return self

# NEW: Pydantic model for General_Synthesizer's output
class GeneralOutput(BaseModel):
    general_output: str = Field(..., alias="general_output", description="The synthesized general output.")
    malformed_blocks: List[Dict[str, Any]] = Field(default_factory=list, alias="malformed_blocks") # Ensure malformed_blocks is present

    @model_validator(mode='after')
    def validate_paths_in_general_output(self) -> 'GeneralOutput':
        """Validates potential file paths within general output for security."""
        # Import sanitize_and_validate_file_path locally here
        try:
            from src.utils.path_utils import sanitize_and_validate_file_path
        except ImportError as e:
            logger.warning(f"Could not import 'sanitize_and_validate_file_path' for GeneralOutput validation: {e}. Skipping path validation.")
            return self # Skip validation if import fails

        sanitized_output = self.general_output
        potential_paths = re.findall(r'\b(?:src|data|tests|config|custom_frameworks)[/\w.-]+\.py\b', sanitized_output, re.IGNORECASE)
        for path in potential_paths:
            try:
                sanitized_path = sanitize_and_validate_file_path(path)
                sanitized_output = sanitized_output.replace(path, sanitized_path)
            except ValueError as e:
                logger.warning(f"Invalid file path detected in GeneralOutput.general_output: '{path}' - {e}")
                sanitized_output = sanitized_output.replace(path, f"INVALID_PATH_DETECTED:{path}")
        self.general_output = sanitized_output
        return self

# NEW: Pydantic model for Conflict Report (Improvement 1)
class ConflictReport(BaseModel):
    conflict_type: Literal["LOGICAL_INCONSISTENCY", "DATA_DISCREPANCY", "METHODOLOGY_DISAGREEMENT", "RESOURCE_CONSTRAINT", "SECURITY_VS_PERFORMANCE"] = Field(..., description="Type of conflict identified.")
    summary: str = Field(..., description="A concise summary of the conflict.")
    involved_personas: List[str] = Field(..., description="Names of personas whose outputs are in conflict.")
    conflicting_outputs_snippet: str = Field(..., description="A brief snippet or reference to the conflicting parts of the debate history.")
    proposed_resolution_paths: List[str] = Field(default_factory=list, description="2-3 high-level suggestions for resolving this conflict.")
    malformed_blocks: List[Dict[str, Any]] = Field(default_factory=list, alias="malformed_blocks") # For parser feedback

# NEW: Pydantic model for SelfImprovementAnalysisOutput (Improvement 4)
class SelfImprovementAnalysisOutput(BaseModel):
    analysis_summary: str = Field(..., alias="ANALYSIS_SUMMARY", description="Overall summary of the self-improvement analysis.")
    impactful_suggestions: List[Dict[str, Any]] = Field(..., alias="IMPACTFUL_SUGGESTIONS", description="List of structured suggestions for improvement.")
    malformed_blocks: List[Dict[str, Any]] = Field(default_factory=list, alias="malformed_blocks")

    @model_validator(mode='after')
    def validate_suggestion_structure(self) -> 'SelfImprovementAnalysisOutput':
        for suggestion in self.impactful_suggestions:
            if not all(k in suggestion for k in ["AREA", "PROBLEM", "PROPOSED_SOLUTION", "EXPECTED_IMPACT"]):
                raise ValueError("Each suggestion must contain 'AREA', 'PROBLEM', 'PROPOSED_SOLUTION', 'EXPECTED_IMPACT'.")
            if "CODE_CHANGES_SUGGESTED" in suggestion:
                # Validate each code change using the CodeChange model
                validated_code_changes = []
                for cc_data in suggestion["CODE_CHANGES_SUGGESTED"]:
                    try:
                        validated_code_changes.append(CodeChange.model_validate(cc_data).model_dump(by_alias=True))
                    except ValidationError as e:
                        logger.warning(f"Malformed CodeChange in SelfImprovementAnalysisOutput: {e}. Skipping this change.")
                        # Optionally, add to malformed_blocks or a dedicated field
                suggestion["CODE_CHANGES_SUGGESTED"] = validated_code_changes
        return self