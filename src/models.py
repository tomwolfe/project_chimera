# src/models.py
from typing import Dict, Any, Optional, List, Literal
from pydantic import BaseModel, Field, validator, model_validator, ConfigDict, ValidationError, field_validator
import logging
import re
from pathlib import Path

logger = logging.getLogger(__name__)

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
    diff_content: Optional[str] = Field(None, alias="DIFF_CONTENT", description="Unified diff format for MODIFY actions (for larger files).") # ADD THIS LINE

    @field_validator('file_path') # CHANGED: @validator to @field_validator
    @classmethod # ADDED: @classmethod
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
        
    @field_validator('action') # CHANGED: @validator to @field_validator
    @classmethod # ADDED: @classmethod
    def validate_action(cls, v):
        """Validates the action type."""
        valid_actions = ["ADD", "MODIFY", "REMOVE"]
        if v not in valid_actions:
            raise ValueError(f"Invalid action: '{v}'. Must be one of {valid_actions}.")
        return v

    @model_validator(mode='after')
    def check_content_based_on_action(self) -> 'CodeChange':
        """Ensures content is provided based on action type and prioritizes diff_content for MODIFY."""
        if self.action == "ADD":
            if self.full_content is None:
                raise ValueError(f"FULL_CONTENT is required for action 'ADD' on file '{self.file_path}'.")
            if self.diff_content is not None: # Ensure no diff for ADD
                raise ValueError(f"DIFF_CONTENT should not be provided for action 'ADD' on file '{self.file_path}'.")
        elif self.action == "MODIFY":
            # --- MODIFICATION START ---
            if self.diff_content is not None:
                # If diff_content is provided, it takes precedence. Clear full_content if present.
                if self.full_content is not None:
                    logger.warning(f"Both FULL_CONTENT and DIFF_CONTENT provided for MODIFY on {self.file_path}. Prioritizing DIFF_CONTENT.")
                    self.full_content = None
            elif self.full_content is None:
                # If neither diff_content nor full_content is provided, raise error.
                raise ValueError(f"Either FULL_CONTENT or DIFF_CONTENT is required for action 'MODIFY' on file '{self.file_path}'.")
            # --- MODIFICATION END ---
        elif self.action == "REMOVE":
            if not isinstance(self.lines, list) or not self.lines:
                raise ValueError(f"LINES must be a non-empty list for action 'REMOVE' on file '{self.file_path}'.")
            if self.full_content is not None or self.diff_content is not None: # Ensure no full/diff content for REMOVE
                raise ValueError(f"FULL_CONTENT or DIFF_CONTENT should not be provided for action 'REMOVE' on file '{self.file_path}'.")
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
    conflict_type: Literal["LOGICAL_INCONSISTENCY", "DATA_DISCREPANCY", "METHODOLOGY_DISAGREEMENT", "RESOURCE_CONSTRAINT", "SECURITY_VS_PERFORMANCE", "NO_CONFLICT"] = Field(..., description="Type of conflict identified.")
    summary: str = Field(..., description="A concise summary of the conflict.")
    involved_personas: List[str] = Field(..., description="Names of personas whose outputs are in conflict.")
    conflicting_outputs_snippet: str = Field(..., description="A brief snippet or reference to the conflicting parts of the debate history.")
    proposed_resolution_paths: List[str] = Field(default_factory=list, description="2-3 high-level suggestions for resolving this conflict.")
    conflict_found: bool = Field(..., description="True if a conflict was identified, False otherwise.") # ADDED THIS LINE
    malformed_blocks: List[Dict[str, Any]] = Field(default_factory=list, alias="malformed_blocks") # For parser feedback

# NEW: Pydantic model for SelfImprovementAnalysisOutputV1 (Original structure, now versioned)
class SelfImprovementAnalysisOutputV1(BaseModel):
    """Version 1 of the self-improvement analysis output schema."""
    analysis_summary: str = Field(..., alias="ANALYSIS_SUMMARY", description="Overall summary of the self-improvement analysis.")
    impactful_suggestions: List[Dict[str, Any]] = Field(..., alias="IMPACTFUL_SUGGESTIONS", description="List of structured suggestions for improvement.")
    malformed_blocks: List[Dict[str, Any]] = Field(default_factory=list, alias="malformed_blocks") # Ensure malformed_blocks is present
    
    model_config = ConfigDict(populate_by_name=True)

    @model_validator(mode='after')
    def validate_suggestion_structure(self) -> 'SelfImprovementAnalysisOutputV1':
        processed_suggestions = []
        for suggestion in self.impactful_suggestions:
            # Check for fundamental structure of a suggestion item
            required_fields = ["AREA", "PROBLEM", "PROPOSED_SOLUTION", "EXPECTED_IMPACT"]
            if not all(k in suggestion for k in required_fields):
                self.malformed_blocks.append({
                    "type": "MALFORMED_SUGGESTION_STRUCTURE",
                    "message": f"A suggestion item is missing required fields: {', '.join(required_fields)}. Skipping this suggestion.",
                    "raw_string_snippet": str(suggestion)[:500]
                })
                continue # Skip this malformed suggestion

            if "CODE_CHANGES_SUGGESTED" in suggestion:
                validated_code_changes = []
                for cc_data in suggestion["CODE_CHANGES_SUGGESTED"]:
                    try:
                        validated_code_changes.append(CodeChange.model_validate(cc_data).model_dump(by_alias=True))
                    except ValidationError as e:
                        logger.warning(f"Malformed CodeChange in SelfImprovementAnalysisOutputV1: {e}. Skipping this change.")
                        # Capture the inner validation error in malformed_blocks
                        self.malformed_blocks.append({
                            "type": "CODE_CHANGE_SCHEMA_VALIDATION_ERROR",
                            "message": f"A suggested code change item failed validation: {e}",
                            "raw_string_snippet": str(cc_data)[:500]
                        })
                suggestion["CODE_CHANGES_SUGGESTED"] = validated_code_changes
            processed_suggestions.append(suggestion)
        self.impactful_suggestions = processed_suggestions # Update the list after processing
        return self

# NEW: Pydantic model for SelfImprovementAnalysisOutput (Versioned wrapper)
class SelfImprovementAnalysisOutput(BaseModel):
    """Current version of the self-improvement analysis output schema with versioning."""
    version: str = Field(default="1.0", description="Schema version")
    data: Dict = Field(..., description="Actual analysis data following version-specific schema")
    metadata: Dict = Field(default_factory=dict, description="Additional metadata about the analysis")
    malformed_blocks: List[Dict[str, Any]] = Field(default_factory=list, alias="malformed_blocks") # For parser feedback

    @model_validator(mode='after')
    def validate_data_structure(self) -> 'SelfImprovementAnalysisOutput':
        if self.version == "1.0":
            # Validate against V1 schema
            try:
                # Pass malformed_blocks from the wrapper to the inner V1 model
                # This allows the V1 model to collect its own malformed blocks
                v1_data = SelfImprovementAnalysisOutputV1.model_validate(self.data)
                # Merge any malformed blocks from the inner V1 model back into the wrapper's malformed_blocks
                self.malformed_blocks.extend(v1_data.malformed_blocks)
                # Update data with the validated and potentially modified V1 data
                self.data = v1_data.model_dump(by_alias=True)
            except ValidationError as e:
                raise ValueError(f"Data does not match schema version {self.version}: {str(e)}")
        # Future versions would be handled here
        else:
            raise ValueError(f"Unsupported schema version: {self.version}")
        return self
    
    def to_v1(self) -> Dict:
        """Convert to version 1 format for backward compatibility."""
        if self.version == "1.0":
            return self.data
        # Conversion logic for future versions would go here
        raise NotImplementedError("Conversion to V1 not implemented for this version")

# NEW: LLMResponseModel for general LLM outputs that need validation
class LLMResponseModel(BaseModel):
    """
    A generic Pydantic model for validating LLM responses that are not
    specifically tied to a persona's structured output schema.
    This can be used for general LLM calls where a simple, consistent
    output structure is expected (e.g., a result string and confidence).
    """
    result: str = Field(..., description="The main result or answer from the LLM.")
    confidence: float = Field(default=0.0, ge=0.0, le=1.0, description="Confidence score of the LLM's response.")
    # Add other common fields you expect from general LLM responses
    # For example, if it often returns a rationale:
    rationale: Optional[str] = Field(None, description="Explanation or reasoning behind the result.")
    # Or if it returns code snippets:
    code_snippet: Optional[str] = Field(None, description="A code snippet if the response involves code generation.")