"""Tests for the models module with Pydantic models."""

import pytest
from pydantic import ValidationError

from src.models import (
    CodeChange,
    ConfigurationAnalysisOutput,
    ConflictReport,
    ContextAnalysisOutput,
    CritiqueOutput,
    DeploymentAnalysisOutput,
    GeneralOutput,
    LLMOutput,
    PersonaConfig,
    SelfImprovementAnalysisOutput,
    SelfImprovementAnalysisOutputV1,
    SuggestionItem,
)


class TestPersonaConfig:
    """Test suite for PersonaConfig model."""

    def test_persona_config_valid(self):
        """Test valid PersonaConfig creation."""
        persona = PersonaConfig(
            name="TestPersona",
            description="A test persona",
            system_prompt_template="You are a helpful assistant.",
            output_schema="GeneralOutput",
            temperature=0.5,
            max_tokens=1024,
            token_efficiency_score=0.8,
        )
        assert persona.name == "TestPersona"
        assert persona.temperature == 0.5
        assert persona.max_tokens == 1024

    def test_persona_config_required_fields(self):
        """Test that required fields are validated."""
        with pytest.raises(ValidationError):
            PersonaConfig(
                name="TestPersona",
                # Missing system_prompt_template
                output_schema="GeneralOutput",
                temperature=0.5,
                max_tokens=1024,
            )

    def test_persona_config_temperature_range(self):
        """Test temperature validation."""
        with pytest.raises(ValidationError):
            PersonaConfig(
                name="TestPersona",
                system_prompt_template="You are a helpful assistant.",
                output_schema="GeneralOutput",
                temperature=1.5,  # Should be between 0.0 and 1.0
                max_tokens=1024,
            )

    def test_persona_config_max_tokens_positive(self):
        """Test max_tokens validation."""
        with pytest.raises(ValidationError):
            PersonaConfig(
                name="TestPersona",
                system_prompt_template="You are a helpful assistant.",
                output_schema="GeneralOutput",
                temperature=0.5,
                max_tokens=0,  # Should be > 0
            )


class TestCodeChange:
    """Test suite for CodeChange model."""

    def test_code_change_valid_add(self):
        """Test valid CodeChange for ADD action."""
        code_change = CodeChange(
            FILE_PATH="test.py", ACTION="ADD", FULL_CONTENT="print('hello')"
        )
        assert code_change.file_path == "test.py"
        assert code_change.action == "ADD"
        assert code_change.full_content == "print('hello')"

    def test_code_change_valid_modify_with_content(self):
        """Test valid CodeChange for MODIFY action with full content."""
        code_change = CodeChange(
            FILE_PATH="test.py", ACTION="MODIFY", FULL_CONTENT="print('modified')"
        )
        assert code_change.file_path == "test.py"
        assert code_change.action == "MODIFY"

    def test_code_change_valid_modify_with_diff(self):
        """Test valid CodeChange for MODIFY action with diff content."""
        code_change = CodeChange(
            FILE_PATH="test.py",
            ACTION="MODIFY",
            DIFF_CONTENT="--- a/test.py\n+++ b/test.py\n@@ -1,1 +1,1 @@\n- old\n+ new",
        )
        assert code_change.file_path == "test.py"
        assert code_change.action == "MODIFY"
        assert "old" in code_change.diff_content

    def test_code_change_valid_remove(self):
        """Test valid CodeChange for REMOVE action."""
        code_change = CodeChange(
            FILE_PATH="test.py", ACTION="REMOVE", LINES=["line1", "line2"]
        )
        assert code_change.file_path == "test.py"
        assert code_change.action == "REMOVE"
        assert code_change.lines == ["line1", "line2"]

    def test_code_change_invalid_action(self):
        """Test validation for invalid action."""
        with pytest.raises(ValidationError):
            CodeChange(
                FILE_PATH="test.py",
                ACTION="INVALID_ACTION",
                FULL_CONTENT="some content",
            )

    def test_code_change_required_fields_for_add(self):
        """Test that ADD action requires full_content or lines."""
        with pytest.raises(ValidationError):
            CodeChange(
                FILE_PATH="test.py",
                ACTION="ADD",
                # Missing FULL_CONTENT or LINES
            )

    def test_code_change_remove_requires_lines(self):
        """Test that REMOVE action requires lines."""
        with pytest.raises(ValidationError):
            CodeChange(
                FILE_PATH="test.py",
                ACTION="REMOVE",
                # Missing LINES
            )

    def test_code_change_remove_lines_must_be_strings(self):
        """Test that REMOVE action lines must be strings."""
        with pytest.raises(ValidationError):
            CodeChange(
                FILE_PATH="test.py",
                ACTION="REMOVE",
                LINES=["valid_line", 123],  # 123 is not a string
            )


class TestLLMOutput:
    """Test suite for LLMOutput model."""

    def test_llm_output_valid(self):
        """Test valid LLMOutput creation."""
        output = LLMOutput(
            COMMIT_MESSAGE="Add new feature",
            RATIONALE="Implemented requested feature",
            CODE_CHANGES=[
                {
                    "FILE_PATH": "test.py",
                    "ACTION": "ADD",
                    "FULL_CONTENT": "print('hello')",
                }
            ],
        )
        assert output.commit_message == "Add new feature"
        assert len(output.code_changes) == 1

    def test_llm_output_minimal_valid(self):
        """Test minimal valid LLMOutput creation."""
        output = LLMOutput(
            COMMIT_MESSAGE="Fix bug", RATIONALE="Fixed a bug", CODE_CHANGES=[]
        )
        assert output.commit_message == "Fix bug"
        assert output.code_changes == []

    def test_llm_output_with_conflict_resolution(self):
        """Test LLMOutput with conflict resolution."""
        output = LLMOutput(
            COMMIT_MESSAGE="Resolve conflict",
            RATIONALE="Addressed conflict between approaches",
            CODE_CHANGES=[],
            CONFLICT_RESOLUTION="Conflict resolved by choosing approach A",
        )
        assert output.conflict_resolution == "Conflict resolved by choosing approach A"


class TestSuggestionItem:
    """Test suite for SuggestionItem model."""

    def test_suggestion_item_valid(self):
        """Test valid SuggestionItem creation."""
        suggestion = SuggestionItem(
            AREA="Reasoning Quality",
            PROBLEM="Inconsistent logic",
            PROPOSED_SOLUTION="Add validation",
            EXPECTED_IMPACT="Improved accuracy",
            PARETO_SCORE=0.8,
            VALIDATION_METHOD="Test with sample data",
            CODE_CHANGES_SUGGESTED=[
                {
                    "FILE_PATH": "test.py",
                    "ACTION": "MODIFY",
                    "FULL_CONTENT": "# Added validation",
                }
            ],
        )
        assert suggestion.area == "Reasoning Quality"
        assert suggestion.pareto_score == 0.8

    def test_suggestion_item_pareto_score_range(self):
        """Test pareto score validation."""
        with pytest.raises(ValidationError):
            SuggestionItem(
                AREA="Testing",
                PROBLEM="Missing tests",
                PROPOSED_SOLUTION="Add unit tests",
                EXPECTED_IMPACT="Better coverage",
                PARETO_SCORE=1.5,  # Should be between 0.0 and 1.0
                VALIDATION_METHOD="Run coverage",
                CODE_CHANGES_SUGGESTED=[],
            )

    def test_suggestion_item_required_fields(self):
        """Test that required fields are validated."""
        with pytest.raises(ValidationError):
            SuggestionItem(
                # Missing AREA
                PROBLEM="Missing tests",
                PROPOSED_SOLUTION="Add unit tests",
                EXPECTED_IMPACT="Better coverage",
                PARETO_SCORE=0.5,
                VALIDATION_METHOD="Run coverage",
                CODE_CHANGES_SUGGESTED=[],
            )


class TestSelfImprovementAnalysisOutputV1:
    """Test suite for SelfImprovementAnalysisOutputV1 model."""

    def test_self_improvement_v1_valid(self):
        """Test valid SelfImprovementAnalysisOutputV1 creation."""
        output = SelfImprovementAnalysisOutputV1(
            ANALYSIS_SUMMARY="Code quality improvements needed",
            IMPACTFUL_SUGGESTIONS=[
                {
                    "AREA": "Maintainability",
                    "PROBLEM": "Complex functions",
                    "PROPOSED_SOLUTION": "Break into smaller functions",
                    "EXPECTED_IMPACT": "Easier to maintain",
                    "PARETO_SCORE": 0.7,
                    "VALIDATION_METHOD": "Code review",
                    "CODE_CHANGES_SUGGESTED": [],
                }
            ],
            ESTIMATED_IMPACT_SCORE=0.85,
        )
        assert output.analysis_summary == "Code quality improvements needed"
        assert len(output.impactful_suggestions) == 1

    def test_self_improvement_v1_minimal(self):
        """Test minimal valid SelfImprovementAnalysisOutputV1 creation."""
        output = SelfImprovementAnalysisOutputV1(
            ANALYSIS_SUMMARY="Basic analysis", IMPACTFUL_SUGGESTIONS=[]
        )
        assert output.analysis_summary == "Basic analysis"
        assert output.impactful_suggestions == []

    def test_self_improvement_v1_invalid_suggestion_structure(self):
        """Test handling of invalid suggestion structure."""
        with pytest.raises(ValidationError):
            SelfImprovementAnalysisOutputV1(
                ANALYSIS_SUMMARY="Analysis with invalid suggestion",
                IMPACTFUL_SUGGESTIONS=[
                    {
                        "AREA": "Testing"  # Missing required fields
                        # Missing PROBLEM, PROPOSED_SOLUTION, etc.
                    }
                ],
            )


class TestSelfImprovementAnalysisOutput:
    """Test suite for SelfImprovementAnalysisOutput model."""

    def test_self_improvement_wrapper_valid_v1(self):
        """Test valid SelfImprovementAnalysisOutput with V1 data."""
        output = SelfImprovementAnalysisOutput(
            version="1.0",
            data={
                "ANALYSIS_SUMMARY": "Code improvements needed",
                "IMPACTFUL_SUGGESTIONS": [],
                "ESTIMATED_IMPACT_SCORE": 0.8,
            },
        )
        assert output.version == "1.0"
        assert output.data["ANALYSIS_SUMMARY"] == "Code improvements needed"

    def test_self_improvement_wrapper_invalid_version(self):
        """Test validation for unsupported version."""
        with pytest.raises(ValueError, match="Unsupported schema version"):
            SelfImprovementAnalysisOutput(
                version="2.0",  # Unsupported version
                data={},
            )

    def test_self_improvement_wrapper_invalid_data(self):
        """Test validation for invalid data structure."""
        with pytest.raises(ValueError, match="Data does not match schema version"):
            SelfImprovementAnalysisOutput(
                version="1.0",
                data={
                    "INVALID_FIELD": "invalid data"  # Missing required fields for V1
                },
            )


class TestContextAnalysisOutput:
    """Test suite for ContextAnalysisOutput model."""

    def test_context_analysis_output_valid(self):
        """Test valid ContextAnalysisOutput creation."""
        output = ContextAnalysisOutput(
            key_modules=[{"name": "core.py", "purpose": "Main logic"}],
            security_concerns=["SQL injection"],
            architectural_patterns=["MVC"],
            performance_bottlenecks=["Database queries"],
        )
        assert len(output.key_modules) == 1
        assert "SQL injection" in output.security_concerns

    def test_context_analysis_output_optional_fields(self):
        """Test ContextAnalysisOutput with optional fields."""
        output = ContextAnalysisOutput(
            key_modules=[],
            security_concerns=[],
            architectural_patterns=[],
            performance_bottlenecks=[],
            security_summary={"high_risk": True},
        )
        assert output.security_summary == {"high_risk": True}


class TestCritiqueOutput:
    """Test suite for CritiqueOutput model."""

    def test_critique_output_valid(self):
        """Test valid CritiqueOutput creation."""
        output = CritiqueOutput(
            CRITIQUE_SUMMARY="Good implementation with minor issues",
            CRITIQUE_POINTS=[{"type": "style", "detail": "Inconsistent naming"}],
            SUGGESTIONS=[
                {
                    "AREA": "Style",
                    "PROBLEM": "Naming",
                    "PROPOSED_SOLUTION": "Follow PEP8",
                    "EXPECTED_IMPACT": "Better readability",
                    "PARETO_SCORE": 0.6,
                    "VALIDATION_METHOD": "Review",
                    "CODE_CHANGES_SUGGESTED": [],
                }
            ],
        )
        assert output.critique_summary == "Good implementation with minor issues"
        assert len(output.suggestions) == 1

    def test_critique_output_minimal(self):
        """Test minimal valid CritiqueOutput creation."""
        output = CritiqueOutput(
            CRITIQUE_SUMMARY="Basic critique", CRITIQUE_POINTS=[], SUGGESTIONS=[]
        )
        assert output.critique_summary == "Basic critique"
        assert output.critique_points == []
        assert output.suggestions == []


class TestConflictReport:
    """Test suite for ConflictReport model."""

    def test_conflict_report_valid_types(self):
        """Test valid ConflictReport creation with different conflict types."""
        # Test various conflict types
        for conflict_type in [
            "LOGICAL_INCONSISTENCY",
            "DATA_DISCREPANCY",
            "METHODOLOGY_DISAGREEMENT",
            "RESOURCE_CONSTRAINT",
            "SECURITY_VS_PERFORMANCE",
        ]:
            report = ConflictReport(
                conflict_type=conflict_type,
                summary="Conflict detected",
                involved_personas=["PersonaA", "PersonaB"],
                conflicting_outputs_snippet="The outputs disagree",
                conflict_found=True,
            )
            assert report.conflict_type == conflict_type

    def test_conflict_report_no_conflict(self):
        """Test ConflictReport with no conflict."""
        report = ConflictReport(
            conflict_type="NO_CONFLICT",
            summary="No conflict found",
            involved_personas=["PersonaA"],
            conflicting_outputs_snippet="",
            conflict_found=False,
        )
        assert report.conflict_found is False
        assert report.conflicting_outputs_snippet == ""

    def test_conflict_report_invalid_type(self):
        """Test validation for invalid conflict type."""
        with pytest.raises(ValidationError):
            ConflictReport(
                conflict_type="INVALID_TYPE",
                summary="Invalid conflict type",
                involved_personas=["PersonaA"],
                conflicting_outputs_snippet="",
                conflict_found=False,
            )


class TestGeneralOutput:
    """Test suite for GeneralOutput model."""

    def test_general_output_valid(self):
        """Test valid GeneralOutput creation."""
        output = GeneralOutput(general_output="This is a general output")
        assert output.general_output == "This is a general output"

    def test_general_output_required_field(self):
        """Test validation for required general_output field."""
        with pytest.raises(ValidationError):
            GeneralOutput(
                # Missing general_output field
            )


class TestConfigurationAnalysisOutput:
    """Test suite for ConfigurationAnalysisOutput model."""

    def test_config_analysis_output_valid(self):
        """Test valid ConfigurationAnalysisOutput creation."""
        output = ConfigurationAnalysisOutput(pre_commit_hooks=[], malformed_blocks=[])
        assert output.pre_commit_hooks == []
        assert output.malformed_blocks == []

    def test_config_analysis_output_with_data(self):
        """Test ConfigurationAnalysisOutput with configuration data."""
        from src.models import PreCommitHook

        hook = PreCommitHook(
            repo="https://github.com/psf/black", rev="22.0.0", id="black"
        )

        output = ConfigurationAnalysisOutput(
            pre_commit_hooks=[hook], malformed_blocks=[]
        )
        assert len(output.pre_commit_hooks) == 1
        assert output.pre_commit_hooks[0].repo == "https://github.com/psf/black"


class TestDeploymentAnalysisOutput:
    """Test suite for DeploymentAnalysisOutput model."""

    def test_deployment_analysis_output_valid(self):
        """Test valid DeploymentAnalysisOutput creation."""
        output = DeploymentAnalysisOutput(
            dockerfile_present=True, prod_dependency_count=5, malformed_blocks=[]
        )
        assert output.dockerfile_present is True
        assert output.prod_dependency_count == 5

    def test_deployment_analysis_output_defaults(self):
        """Test DeploymentAnalysisOutput with default values."""
        output = DeploymentAnalysisOutput()
        assert output.dockerfile_present is False
        assert output.prod_dependency_count == 0
        assert output.unpinned_prod_dependencies == []
        assert output.malformed_blocks == []
