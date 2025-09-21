import re
from enum import Enum
from typing import Any, Dict, List, Literal, Optional

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    field_validator,
    model_validator,
)


# --- Pydantic Models for Schema Validation ---
class PersonaConfig(BaseModel):
    name: str
    description: Optional[str] = None
    system_prompt_template: (
        str  # MODIFIED: Replaced system_prompt with system_prompt_template
    )
    output_schema: str = Field(
        "GeneralOutput",
        description="The Pydantic model name for the persona's output schema.",
    )
    temperature: float = Field(..., ge=0.0, le=1.0)
    max_tokens: int = Field(..., gt=0)
    token_efficiency_score: Optional[float] = Field(None, ge=0.0, le=1.0)  # ADDED


class ReasoningFrameworkConfig(BaseModel):
    framework_name: str
    personas: Dict[str, PersonaConfig]
    persona_sets: Dict[str, List[str]]
    version: int = 1

    @model_validator(mode="after")
    def validate_persona_sets_references(self):
        # This validation is more relevant in core.py where all_personas are known
        return self


# NEW: Pydantic models for detailed configuration analysis
class CiWorkflowStep(BaseModel):
    name: str
    uses: Optional[str] = None
    runs_commands: Optional[List[str]] = None
    code_snippet: Optional[str] = None


class CiWorkflowJob(BaseModel):
    steps_summary: List[CiWorkflowStep]


class CiWorkflowConfig(BaseModel):
    name: Optional[str] = None
    on_triggers: Optional[Dict[str, Any]] = None
    jobs: Dict[str, CiWorkflowJob] = Field(default_factory=dict)


class PreCommitHook(BaseModel):
    repo: str
    rev: str
    id: str
    args: List[str] = Field(default_factory=list)
    code_snippet: Optional[str] = None


class RuffConfig(BaseModel):
    line_length: Optional[int] = None
    target_version: Optional[str] = None
    lint_select: Optional[List[str]] = None
    lint_ignore: Optional[List[str]] = None
    format_settings: Optional[Dict[str, Any]] = None
    config_snippet: Optional[str] = None


class BanditConfig(BaseModel):
    exclude_dirs: Optional[List[str]] = None
    severity_level: Optional[str] = None
    confidence_level: Optional[str] = None
    skip_checks: Optional[List[str]] = None
    config_snippet: Optional[str] = None


class PydanticSettingsConfig(BaseModel):
    env_file: Optional[str] = None


class PyprojectTomlConfig(BaseModel):
    ruff: Optional[RuffConfig] = None
    bandit: Optional[BanditConfig] = None
    pydantic_settings: Optional[PydanticSettingsConfig] = None


class ConfigurationAnalysisOutput(BaseModel):
    """Structured output for analyzing project configuration files."""

    ci_workflow: Optional[CiWorkflowConfig] = None
    pre_commit_hooks: List[PreCommitHook] = Field(default_factory=list)
    pyproject_toml: Optional[PyprojectTomlConfig] = None
    malformed_blocks: List[Dict[str, Any]] = Field(
        default_factory=list, alias="malformed_blocks"
    )


# NEW MODEL: DeploymentAnalysisOutput
class DeploymentAnalysisOutput(BaseModel):
    """Structured output for analyzing deployment robustness."""

    dockerfile_present: bool = False
    dockerfile_healthcheck_present: bool = False
    dockerfile_non_root_user: bool = False
    dockerfile_exposed_ports: List[int] = Field(default_factory=list)
    dockerfile_multi_stage_build: bool = False
    dockerfile_problem_snippets: List[str] = Field(default_factory=list)
    prod_requirements_present: bool = False
    prod_dependency_count: int = 0
    dev_dependency_overlap_count: int = 0
    unpinned_prod_dependencies: List[str] = Field(default_factory=list)
    malformed_blocks: List[Dict[str, Any]] = Field(
        default_factory=list, alias="malformed_blocks"
    )


# NEW: Pydantic model for Context_Aware_Assistant's output
class ContextAnalysisOutput(BaseModel):
    key_modules: List[Dict[str, Any]] = Field(
        ...,
        alias="key_modules",
        description="List of important modules/files and their purpose.",
    )
    security_concerns: List[str] = Field(
        ...,
        alias="security_concerns",
        description="List of potential security issues or patterns.",
    )
    architectural_patterns: List[str] = Field(
        ...,
        alias="architectural_patterns",
        description="List of observed architectural patterns or design principles.",
    )
    performance_bottlenecks: List[str] = Field(
        ...,
        alias="performance_bottlenecks",
        description="List of potential performance issues or areas for optimization.",
    )

    # NEW: Fields for persona-specific summaries (Improvement 3)
    security_summary: Optional[Dict[str, Any]] = Field(
        None,
        alias="security_summary",
        description="Summary tailored for Security_Auditor, including problem snippets.",
    )
    architecture_summary: Optional[Dict[str, Any]] = Field(
        None,
        alias="architecture_summary",
        description="Summary tailored for Code_Architect, including problem snippets.",
    )
    devops_summary: Optional[Dict[str, Any]] = Field(
        None,
        alias="devops_summary",
        description="Summary tailored for DevOps_Engineer, including problem snippets.",
    )
    testing_summary: Optional[Dict[str, Any]] = Field(
        None,
        alias="testing_summary",
        description="Summary tailored for Test_Engineer, including problem snippets.",
    )
    general_overview: Optional[str] = Field(
        None,
        alias="general_overview",
        description="A general high-level overview of the codebase context.",
    )
    configuration_summary: Optional[ConfigurationAnalysisOutput] = Field(
        None,
        alias="configuration_summary",
        description="Structured summary of project configuration files.",
    )
    deployment_summary: Optional[DeploymentAnalysisOutput] = Field(
        None,
        alias="deployment_summary",
        description="Structured summary of deployment robustness.",
    )

    @model_validator(mode="after")
    def validate_paths_in_context_output(self) -> "ContextAnalysisOutput":
        """Validates file paths within key_modules for security."""
        # Import sanitize_and_validate_file_path locally here
        try:
            from src.utils.core_helpers.path_utils import (
                sanitize_and_validate_file_path,
            )
        except ImportError:
            # logger.warning(
            #     f"Could not import 'sanitize_and_validate_file_path' for ContextAnalysisOutput validation: {e}. Skipping path validation."
            # )
            return self

        for module in self.key_modules:
            if "name" in module and isinstance(module["name"], str):
                try:
                    module["name"] = sanitize_and_validate_file_path(module["name"])
                except ValueError:
                    # logger.warning(
                    #     f"Invalid file path detected in ContextAnalysisOutput.key_modules: '{module['name']}' - {e}"
                    # )
                    module["name"] = f"INVALID_PATH_DETECTED:{module['name']}"
        return self


# --- MODIFICATION FOR IMPROVEMENT 4.3 ---
# Moved LLMOutput and CodeChange definitions here for centralization.
class CodeChange(BaseModel):
    file_path: str = Field(..., alias="FILE_PATH")
    action: Literal["ADD", "MODIFY", "REMOVE", "CREATE", "CREATE_DIRECTORY"] = Field(
        ..., alias="ACTION"
    )
    full_content: Optional[str] = Field(None, alias="FULL_CONTENT")
    # MODIFIED: Allow 'lines' to be Optional[List[str]] to accept 'null' from LLM
    lines: Optional[List[str]] = Field(
        None,
        alias="LINES",
        description="List of line numbers or content for REMOVE action",
    )
    diff_content: Optional[str] = Field(
        None,
        alias="DIFF_CONTENT",
        description="Unified diff format for MODIFY actions (for larger files).",
    )

    @field_validator("file_path")
    @classmethod
    def validate_file_path(cls, v):
        """Validates and sanitizes the file path."""
        try:
            from src.utils.core_helpers.path_utils import (
                sanitize_and_validate_file_path,
            )
        except ImportError:
            # logger.warning(
            #     f"Could not import 'sanitize_and_validate_file_path' for CodeChange validation: {e}. Proceeding without strict validation."
            # )
            return v

        try:
            return sanitize_and_validate_file_path(v)
        except ValueError as ve:
            raise ValueError(f"Invalid file path: {ve}") from ve

    @field_validator("action")
    @classmethod
    def validate_action(cls, v):
        """Validates the action type."""
        valid_actions = ["ADD", "MODIFY", "REMOVE", "CREATE", "CREATE_DIRECTORY"]
        if v not in valid_actions:
            raise ValueError(f"Invalid action: '{v}'. Must be one of {valid_actions}.")
        return v

    @field_validator("diff_content")
    @classmethod
    def validate_diff_content_format(cls, v: Optional[str]) -> Optional[str]:
        """Validates that diff_content, if present, looks like a unified diff."""
        if v is None:
            return v
        if not re.search(r"^--- a/.*?\n\+\+\+ b/.*?\n", v, re.MULTILINE):
            raise ValueError(
                "DIFF_CONTENT does not appear to be in a standard unified diff format (missing '--- a/' and '+++ b/' headers)."
            )
        if not re.search(r"^[ +-@].*$", v, re.MULTILINE):
            raise ValueError(
                "DIFF_CONTENT does not contain typical diff line prefixes ('+', '-', ' ', '@')."
            )
        return v

    @model_validator(mode="after")
    def check_content_based_on_action(self) -> "CodeChange":
        """Ensures content is provided based on action type and prioritizes diff_content for MODIFY."""
        if self.action in ["ADD", "CREATE", "CREATE_DIRECTORY"]:
            if self.full_content is None and not self.lines:
                raise ValueError(
                    f"FULL_CONTENT or LINES is required for action '{self.action}' on file '{self.file_path}'."
                )
            if self.diff_content is not None:
                raise ValueError(
                    f"DIFF_CONTENT should not be provided for action '{self.action}' on file '{self.file_path}'."
                )
        elif self.action == "MODIFY":
            if self.diff_content is not None:
                if self.full_content is not None:
                    # logger.warning(
                    #     f"Both FULL_CONTENT and DIFF_CONTENT provided for MODIFY on {self.file_path}. Prioritizing DIFF_CONTENT."
                    # )
                    self.full_content = None
            elif self.full_content is None:
                raise ValueError(
                    f"Either FULL_CONTENT or DIFF_CONTENT is required for action 'MODIFY' on file '{self.file_path}'."
                )
        elif self.action == "REMOVE":
            # MODIFIED: Ensure self.lines is a non-empty list of strings for REMOVE action
            if not isinstance(self.lines, list) or not self.lines:
                raise ValueError(
                    f"LINES must be a non-empty list for action 'REMOVE' on file '{self.file_path}'."
                )
            if not all(
                isinstance(x, str) for x in self.lines
            ):  # NEW: Explicitly check for string type
                raise ValueError(
                    f"LINES must contain only strings for action 'REMOVE' on file '{self.file_path}'."
                )
            if self.full_content is not None or self.diff_content is not None:
                raise ValueError(
                    f"FULL_CONTENT or DIFF_CONTENT should not be provided for action 'REMOVE' on file '{self.file_path}'."
                )
        return self


class LLMOutput(BaseModel):
    commit_message: str = Field(alias="COMMIT_MESSAGE")
    rationale: str = Field(alias="RATIONALE")
    code_changes: List[CodeChange] = Field(alias="CODE_CHANGES")
    conflict_resolution: Optional[str] = Field(None, alias="CONFLICT_RESOLUTION")
    unresolved_conflict: Optional[str] = Field(None, alias="UNRESOLVED_CONFLICT")
    malformed_blocks: List[Dict[str, Any]] = Field(
        default_factory=list, alias="malformed_blocks"
    )
    malformed_code_change_items: List[Dict[str, Any]] = Field(
        default_factory=list, alias="malformed_code_change_items"
    )


class ImprovementArea(str, Enum):
    """Defines categories for self-improvement findings."""

    REASONING_QUALITY = "Reasoning Quality"
    ROBUSTNESS = "Robustness"
    EFFICIENCY = "Efficiency"
    MAINTAINABILITY = "Maintainability"
    SECURITY = "Security"


class CodeChangeDescription(BaseModel):
    """Describes a suggested code change."""

    action: str
    file_path: str
    description: str
    impact: str


class QuantitativeImpactMetrics(BaseModel):
    """Quantitative metrics for improvement impact assessment."""

    estimated_effort: int = Field(
        ge=1, le=10, description="Estimated effort on a scale of 1-10"
    )
    expected_quality_improvement: float = Field(
        ge=0.0, le=1.0, description="Expected improvement in reasoning quality (0-1)"
    )
    token_savings_percent: Optional[float] = Field(
        None, ge=0.0, le=1.0, description="Expected token usage reduction"
    )
    error_reduction_percent: Optional[float] = Field(
        None, ge=0.0, le=1.0, description="Expected error reduction"
    )


class SelfImprovementFinding(BaseModel):
    """A single self-improvement finding with problem, solution, and impact."""

    area: ImprovementArea
    problem: str
    solution: str
    impact: str
    priority_score: float = Field(ge=0.0, le=1.0)
    code_changes: List[CodeChange]
    metrics: Optional[QuantitativeImpactMetrics] = None
    pareto_score: float = Field(
        ge=0.0, le=1.0, description="80/20 Pareto principle score (impact/effort)"
    )


# NEW: Model for structured suggestions within CritiqueOutput
class SuggestionItem(BaseModel):
    area: str = Field(
        ...,
        alias="AREA",
        description="Category of the suggestion (e.g., Reasoning Quality, Robustness).",
    )
    problem: str = Field(..., alias="PROBLEM", description="Specific issue identified.")
    proposed_solution: str = Field(
        ...,
        alias="PROPOSED_SOLUTION",
        description="Concrete solution to the identified problem.",
    )
    expected_impact: str = Field(
        ...,
        alias="EXPECTED_IMPACT",
        description="Expected benefits of implementing the solution.",
    )
    code_changes_suggested: List[CodeChange] = Field(
        default_factory=list,
        alias="CODE_CHANGES_SUGGESTED",
        description="Details of suggested code modifications.",
    )
    rationale: Optional[str] = Field(
        None, alias="RATIONALE", description="Detailed rationale for the suggestion."
    )


# NEW: Pydantic model for general critique output
class CritiqueOutput(BaseModel):
    critique_summary: str = Field(
        ..., alias="CRITIQUE_SUMMARY", description="A concise summary of the critique."
    )
    critique_points: List[Dict[str, Any]] = Field(
        ..., alias="CRITIQUE_POINTS", description="Detailed points of critique."
    )
    suggestions: List[SuggestionItem] = Field(
        default_factory=list,
        alias="SUGGESTIONS",
        description="Actionable suggestions for improvement.",
    )
    malformed_blocks: List[Dict[str, Any]] = Field(
        default_factory=list, alias="malformed_blocks"
    )


# NEW: Pydantic model for General_Synthesizer's output
class GeneralOutput(BaseModel):
    general_output: str = Field(
        ..., alias="general_output", description="The synthesized general output."
    )
    malformed_blocks: List[Dict[str, Any]] = Field(
        default_factory=list, alias="malformed_blocks"
    )

    @model_validator(mode="after")
    def validate_paths_in_general_output(self) -> "GeneralOutput":
        """Validates potential file paths within general output for security."""
        try:
            from src.utils.core_helpers.path_utils import (
                sanitize_and_validate_file_path,
            )
        except ImportError:
            # logger.warning(
            #     f"Could not import 'sanitize_and_validate_file_path' for GeneralOutput validation: {e}. Skipping path validation."
            # )
            return self

        sanitized_output = self.general_output
        potential_paths = re.findall(
            r"\b(?:src|data|tests|config|custom_frameworks)[/\w.-]+\.py\b",
            sanitized_output,
            re.IGNORECASE,
        )
        for path in potential_paths:
            try:
                sanitized_path = sanitize_and_validate_file_path(path)
                sanitized_output = sanitized_output.replace(path, sanitized_path)
            except ValueError:
                # logger.warning(
                #     f"Invalid file path detected in GeneralOutput.general_output: '{path}' - {e}"
                # )
                sanitized_output = sanitized_output.replace(
                    path, f"INVALID_PATH_DETECTED:{path}"
                )
        self.general_output = sanitized_output
        return self


# NEW: Pydantic model for Conflict Report (Improvement 1)
class ConflictReport(BaseModel):
    conflict_type: Literal[
        "LOGICAL_INCONSISTENCY",
        "DATA_DISCREPANCY",
        "METHODOLOGY_DISAGREEMENT",
        "RESOURCE_CONSTRAINT",
        "SECURITY_VS_PERFORMANCE",
        "NO_CONFLICT",
    ] = Field(..., description="Type of conflict identified.")
    summary: str = Field(..., description="A concise summary of the conflict.")
    involved_personas: List[str] = Field(
        ..., description="Names of personas whose outputs are in conflict."
    )
    conflicting_outputs_snippet: str = Field(
        ...,
        description="A brief snippet or reference to the conflicting parts of the debate history.",
    )
    proposed_resolution_paths: List[str] = Field(
        default_factory=list,
        description="2-3 high-level suggestions for resolving this conflict.",
    )
    conflict_found: bool = Field(
        ..., description="True if a conflict was identified, False otherwise."
    )
    malformed_blocks: List[Dict[str, Any]] = Field(
        default_factory=list, alias="malformed_blocks"
    )


# NEW: Pydantic model for SelfImprovementAnalysisOutputV1 (Original structure, now versioned)
class SelfImprovementAnalysisOutputV1(BaseModel):
    """Version 1 of the self-improvement analysis output schema."""

    analysis_summary: str = Field(
        ...,
        alias="ANALYSIS_SUMMARY",
        description="Overall summary of the self-improvement analysis.",
    )
    # MODIFIED: Changed impactful_suggestions to be a list of SuggestionItem objects
    impactful_suggestions: List[SuggestionItem] = Field(
        ...,
        alias="IMPACTFUL_SUGGESTIONS",
        description="List of structured suggestions for improvement.",
    )
    estimated_impact_score: Optional[float] = Field(
        None,
        ge=0.0,
        le=1.0,
        description="AI's self-estimated impact score for the overall analysis (0.0 to 1.0).",
    )
    malformed_blocks: List[Dict[str, Any]] = Field(
        default_factory=list, alias="malformed_blocks"
    )

    model_config = ConfigDict(populate_by_name=True)

    @model_validator(mode="after")
    def validate_suggestion_structure(self) -> "SelfImprovementAnalysisOutputV1":
        processed_suggestions = []
        for suggestion in self.impactful_suggestions:
            if not isinstance(suggestion, SuggestionItem):
                try:
                    validated_suggestion = SuggestionItem.model_validate(suggestion)
                    processed_suggestions.append(validated_suggestion)
                except ValidationError as e:
                    self.malformed_blocks.append(
                        {
                            "type": "MALFORMED_SUGGESTION_STRUCTURE",
                            "message": f"A suggestion item failed validation: {e}. Skipping this suggestion.",
                            "raw_string_snippet": str(suggestion)[:500],
                        }
                    )
            else:
                processed_suggestions.append(suggestion)
        self.impactful_suggestions = processed_suggestions
        return self


# NEW: Pydantic model for SelfImprovementAnalysisOutput (Versioned wrapper)
class SelfImprovementAnalysisOutput(BaseModel):
    """Current version of the self-improvement analysis output schema with versioning."""

    version: str = Field(default="1.0", description="Schema version")
    data: Dict = Field(
        ..., description="Actual analysis data following version-specific schema"
    )
    metadata: Dict = Field(
        default_factory=dict, description="Additional metadata about the analysis"
    )
    malformed_blocks: List[Dict[str, Any]] = Field(
        default_factory=list, alias="malformed_blocks"
    )

    @model_validator(mode="after")
    def validate_data_structure(self) -> "SelfImprovementAnalysisOutput":
        if self.version == "1.0":
            try:
                v1_data = SelfImprovementAnalysisOutputV1.model_validate(self.data)
                self.malformed_blocks.extend(v1_data.malformed_blocks)
                self.data = v1_data.model_dump(by_alias=True)
            except ValidationError as e:
                raise ValueError(
                    f"Data does not match schema version {self.version}: {str(e)}"
                )
            except Exception as e:  # Catch other potential errors during validation
                raise ValueError(
                    f"An unexpected error occurred during V1 data validation: {str(e)}"
                )
        else:
            raise ValueError(f"Unsupported schema version: {self.version}")
        return self

    def to_v1(self) -> Dict:
        """Convert to version 1 format for backward compatibility."""
        if self.version == "1.0":
            return self.data
        raise NotImplementedError("Conversion to V1 not implemented for this version")


# NEW: LLMResponseModel for general LLM outputs that need validation
class LLMResponseModel(BaseModel):
    """A generic Pydantic model for validating LLM responses that are not
    specifically tied to a persona's structured output schema.
    """

    result: str = Field(..., description="The main result or answer from the LLM.")
    confidence: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Confidence score of the LLM's response.",
    )
    rationale: Optional[str] = Field(
        None, description="Explanation or reasoning behind the result."
    )
    code_snippet: Optional[str] = Field(
        None, description="A code snippet if the response involves code generation."
    )
