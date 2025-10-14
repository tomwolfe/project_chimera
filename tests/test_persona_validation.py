"""Test cases for persona-specific reliability and validation."""

import json

from src.models import (
    ConflictReport,
    CritiqueOutput,
    GeneralOutput,
    LLMOutput,
    SelfImprovementAnalysisOutputV1,
)
from src.utils.reporting.output_parser import LLMOutputParser


class TestPersonaReliability:
    """Test persona-specific output validation and reliability."""

    def setup_method(self):
        """Set up test fixtures."""
        self.parser = LLMOutputParser()

    def test_devils_advocate_conflict_report_validation(self):
        """Test Devils_Advocate's ConflictReport validation."""
        # Valid ConflictReport example
        valid_conflict_report = {
            "conflict_type": "LOGICAL_INCONSISTENCY",
            "summary": "Detected logical inconsistency between approach A and approach B",
            "involved_personas": ["Visionary_Generator", "Skeptical_Generator"],
            "conflicting_outputs_snippet": "Output A says X, output B says Y",
            "proposed_resolution_paths": [
                "Evaluate both approaches against success metrics",
                "Identify assumptions in each approach",
                "Seek synthesis of both approaches",
            ],
            "conflict_found": True,
        }

        result = self.parser.parse_and_validate(
            json.dumps(valid_conflict_report), ConflictReport
        )

        assert "conflict_type" in result
        assert result["conflict_type"] == "LOGICAL_INCONSISTENCY"
        assert "malformed_blocks" in result

    def test_devils_advocate_malformed_conflict_report(self):
        """Test Devils_Advocate's malformed ConflictReport handling."""
        # Malformed ConflictReport example with missing required fields
        malformed_conflict_report = {
            "summary": "Detected issue",  # Missing required conflict_type
            "involved_personas": ["Visionary_Generator"],
            "conflicting_outputs_snippet": "Some snippet",
            "conflict_found": True,
        }

        result = self.parser.parse_and_validate(
            json.dumps(malformed_conflict_report), ConflictReport
        )

        # Should have schema correction for missing required field
        assert "malformed_blocks" in result
        assert "conflict_type" in result  # Should have been added via correction
        assert result["conflict_type"] == "NO_CONFLICT"  # Default fallback

    def test_devils_advocate_invalid_conflict_type(self):
        """Test Devils_Advocate with invalid conflict type."""
        invalid_conflict_report = {
            "conflict_type": "INVALID_TYPE",
            "summary": "Test invalid type",
            "involved_personas": ["Visionary_Generator"],
            "conflicting_outputs_snippet": "Some snippet",
            "conflict_found": True,
        }

        result = self.parser.parse_and_validate(
            json.dumps(invalid_conflict_report), ConflictReport
        )

        # Should handle invalid enum value
        assert "malformed_blocks" in result

    def test_constructive_critic_critique_output_validation(self):
        """Test Constructive_Critic's CritiqueOutput validation."""
        valid_critique_output = {
            "CRITIQUE_SUMMARY": "Analysis of code quality and improvement suggestions",
            "CRITIQUE_POINTS": [
                {
                    "point_summary": "Potential performance issue",
                    "details": "The current implementation might be inefficient",
                    "recommendation": "Consider using a more efficient algorithm",
                }
            ],
            "SUGGESTIONS": [
                {
                    "AREA": "Efficiency",
                    "PROBLEM": "Inefficient algorithm implementation",
                    "PROPOSED_SOLUTION": "Use optimized algorithm",
                    "EXPECTED_IMPACT": "Significant performance improvement",
                    "PARETO_SCORE": 0.85,
                    "VALIDATION_METHOD": "Performance benchmarking",
                }
            ],
        }

        result = self.parser.parse_and_validate(
            json.dumps(valid_critique_output), CritiqueOutput
        )

        assert "CRITIQUE_SUMMARY" in result
        assert "CRITIQUE_POINTS" in result
        assert "SUGGESTIONS" in result
        assert len(result["SUGGESTIONS"]) == 1

    def test_constructive_critic_malformed_suggestions(self):
        """Test Constructive_Critic with malformed suggestions."""
        malformed_critique_output = {
            "CRITIQUE_SUMMARY": "Analysis with malformed suggestions",
            "CRITIQUE_POINTS": [],
            "SUGGESTIONS": [
                {
                    "AREA": "Efficiency"  # Missing other required fields
                },
                "just a string",  # Invalid suggestion format
                {
                    "AREA": "Maintainability",
                    "PROBLEM": "Poor code organization",
                    "PROPOSED_SOLUTION": "Reorganize code structure",
                    "EXPECTED_IMPACT": "Better maintainability",
                    "PARETO_SCORE": "invalid",  # Wrong type
                    "VALIDATION_METHOD": "Code review",
                },
            ],
        }

        result = self.parser.parse_and_validate(
            json.dumps(malformed_critique_output), CritiqueOutput
        )

        # Should handle malformed suggestions gracefully
        assert "malformed_blocks" in result

    def test_test_engineer_validation(self):
        """Test Test_Engineer's output validation."""
        test_engineer_output = {
            "CRITIQUE_SUMMARY": "Test coverage analysis",
            "CRITIQUE_POINTS": [
                {
                    "point_summary": "Insufficient test coverage",
                    "details": "Current tests don't cover edge cases",
                    "recommendation": "Add more comprehensive test cases",
                }
            ],
            "SUGGESTIONS": [
                {
                    "AREA": "Test Coverage",
                    "PROBLEM": "Missing edge case tests",
                    "PROPOSED_SOLUTION": "Add parameterized tests with edge case inputs",
                    "EXPECTED_IMPACT": "Improved test reliability",
                    "PARETO_SCORE": 0.9,
                    "VALIDATION_METHOD": "Run comprehensive test suite",
                }
            ],
        }

        result = self.parser.parse_and_validate(
            json.dumps(test_engineer_output), CritiqueOutput
        )

        assert "CRITIQUE_SUMMARY" in result
        EXPECTED_PARETO_SCORE = 0.9

        assert len(result["SUGGESTIONS"]) == 1
        assert result["SUGGESTIONS"][0]["PARETO_SCORE"] == EXPECTED_PARETO_SCORE

    def test_self_improvement_analyst_validation(self):
        """Test Self_Improvement_Analyst's output validation."""
        self_improvement_output = {
            "ANALYSIS_SUMMARY": "Analysis of Project Chimera's current state and improvement opportunities",
            "IMPACTFUL_SUGGESTIONS": [
                {
                    "AREA": "Robustness",
                    "PROBLEM": "Insufficient error handling in LLM output parsing",
                    "PROPOSED_SOLUTION": "Add comprehensive validation and fallback mechanisms",
                    "EXPECTED_IMPACT": "Improved reliability and error recovery",
                    "PARETO_SCORE": 0.95,
                    "VALIDATION_METHOD": "Monitor error rates and recovery success",
                },
                {
                    "AREA": "Maintainability",
                    "PROBLEM": "Complex validation logic scattered across modules",
                    "PROPOSED_SOLUTION": "Consolidate validation logic into utility modules",
                    "EXPECTED_IMPACT": "Easier maintenance and updates",
                    "PARETO_SCORE": 0.8,
                    "VALIDATION_METHOD": "Code review and refactoring metrics",
                },
            ],
        }

        result = self.parser.parse_and_validate(
            json.dumps(self_improvement_output), SelfImprovementAnalysisOutputV1
        )

        assert "ANALYSIS_SUMMARY" in result
        assert "IMPACTFUL_SUGGESTIONS" in result
        EXPECTED_SUGGESTION_COUNT = 2

        assert len(result["IMPACTFUL_SUGGESTIONS"]) == EXPECTED_SUGGESTION_COUNT
        assert all("PARETO_SCORE" in sug for sug in result["IMPACTFUL_SUGGESTIONS"])

    def test_self_improvement_analyst_empty_analysis(self):
        """Test Self_Improvement_Analyst with empty or minimal output."""
        empty_output = {"ANALYSIS_SUMMARY": "", "IMPACTFUL_SUGGESTIONS": []}

        result = self.parser.parse_and_validate(
            json.dumps(empty_output), SelfImprovementAnalysisOutputV1
        )

        # Should handle empty analysis (though with validation errors)
        assert "malformed_blocks" in result

    def test_impartial_arbitrator_llm_output_validation(self):
        """Test Impartial_Arbitrator's LLMOutput validation."""
        llm_output = {
            "COMMIT_MESSAGE": "Implement recommended changes from debate",
            "RATIONALE": "Synthesizing the best aspects of different approaches discussed by personas",
            "CODE_CHANGES": [
                {
                    "FILE_PATH": "src/core.py",
                    "ACTION": "MODIFY",
                    "DIFF_CONTENT": "--- a/src/core.py\n+++ b/src/core.py\n@@ -1,5 +1,7 @@\n+import logging\n+\n def process_data():\n     return 'processed data'\n",
                }
            ],
        }

        result = self.parser.parse_and_validate(json.dumps(llm_output), LLMOutput)

        assert "COMMIT_MESSAGE" in result
        assert "RATIONALE" in result
        assert "CODE_CHANGES" in result
        assert len(result["CODE_CHANGES"]) == 1
        assert result["CODE_CHANGES"][0]["ACTION"] == "MODIFY"

    def test_impartial_arbitrator_invalid_action(self):
        """Test Impartial_Arbitrator with invalid action in code change."""
        llm_output = {
            "COMMIT_MESSAGE": "Fix issue",
            "RATIONALE": "Implementing fix",
            "CODE_CHANGES": [
                {
                    "FILE_PATH": "src/core.py",
                    "ACTION": "INVALID_ACTION",  # Invalid action
                    "FULL_CONTENT": "content",
                }
            ],
        }

        result = self.parser.parse_and_validate(json.dumps(llm_output), LLMOutput)

        # Should handle invalid action
        assert "malformed_blocks" in result

    def test_general_output_persona_validation(self):
        """Test General_Synthesizer and other general output personas."""
        general_output = {
            "general_output": "This is a synthesized general output from the persona",
            "malformed_blocks": [],
        }

        result = self.parser.parse_and_validate(
            json.dumps(general_output), GeneralOutput
        )

        assert "general_output" in result
        assert (
            result["general_output"]
            == "This is a synthesized general output from the persona"
        )

    def test_general_output_empty_string(self):
        """Test General_Synthesizer with empty string output."""
        empty_output = {
            "general_output": ""  # Empty string - should fail validation
        }

        result = self.parser.parse_and_validate(json.dumps(empty_output), GeneralOutput)

        # Should have validation error for empty required field
        assert "malformed_blocks" in result

    def test_multiple_personas_same_schema_conflicts(self):
        """Test how different personas with same schema interact."""
        # Two different persona outputs that should both conform to CritiqueOutput schema
        code_architect_output = {
            "CRITIQUE_SUMMARY": "Architecture review findings",
            "CRITIQUE_POINTS": [
                {
                    "point_summary": "Architectural design issue",
                    "details": "The current architecture is not scalable",
                    "recommendation": "Implement microservices architecture",
                }
            ],
            "SUGGESTIONS": [],
        }

        security_auditor_output = {
            "CRITIQUE_SUMMARY": "Security vulnerabilities detected",
            "CRITIQUE_POINTS": [
                {
                    "point_summary": "Potential injection vulnerability",
                    "details": "Unsanitized input could lead to injection",
                    "recommendation": "Implement input validation and sanitization",
                }
            ],
            "SUGGESTIONS": [
                {
                    "AREA": "Security",
                    "PROBLEM": "Input validation missing",
                    "PROPOSED_SOLUTION": "Add comprehensive input sanitization",
                    "EXPECTED_IMPACT": "Reduced security risk",
                    "PARETO_SCORE": 0.9,
                    "VALIDATION_METHOD": "Security testing",
                }
            ],
        }

        # Both should parse successfully with the same schema
        arch_result = self.parser.parse_and_validate(
            json.dumps(code_architect_output), CritiqueOutput
        )
        sec_result = self.parser.parse_and_validate(
            json.dumps(security_auditor_output), CritiqueOutput
        )

        assert "CRITIQUE_SUMMARY" in arch_result
        assert "CRITIQUE_SUMMARY" in sec_result
        assert arch_result["CRITIQUE_SUMMARY"] == "Architecture review findings"
        assert sec_result["CRITIQUE_SUMMARY"] == "Security vulnerabilities detected"
