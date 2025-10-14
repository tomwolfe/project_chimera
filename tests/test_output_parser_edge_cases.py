"""Test cases for LLM output parser edge cases and failure modes."""

import json

from src.models import (
    ConflictReport,
    CritiqueOutput,
    GeneralOutput,
    LLMOutput,
    SelfImprovementAnalysisOutputV1,
)
from src.utils.reporting.output_parser import LLMOutputParser


class TestOutputParserEdgeCases:
    """Test edge cases and failure modes for LLM output parsing."""

    def setup_method(self):
        """Set up test fixtures."""
        self.parser = LLMOutputParser()

    def test_parse_truncated_json(self):
        """Test parsing of truncated JSON responses."""
        # Test with missing closing brace
        raw_output = '{"general_output": "test", "other_field": "value"'
        result = self.parser.parse_and_validate(raw_output, GeneralOutput)

        # Should have malformed_blocks indicating transformation was needed
        assert "malformed_blocks" in result
        assert len(result["malformed_blocks"]) > 0
        assert any(
            block.get("type") == "JSON_REPAIR_ATTEMPTED"
            for block in result["malformed_blocks"]
        )

    def test_parse_non_json_response(self):
        """Test parsing of non-JSON responses."""
        raw_output = "This is not JSON at all"
        result = self.parser.parse_and_validate(raw_output, GeneralOutput)

        # Should have fallback output
        assert "malformed_blocks" in result
        assert any(
            block.get("type") == "JSON_EXTRACTION_FAILED"
            for block in result["malformed_blocks"]
        )
        assert "general_output" in result

    def test_parse_json_with_extra_fields(self):
        """Test parsing of JSON with extra fields not in schema."""
        raw_output = json.dumps(
            {
                "general_output": "test output",
                "extra_field": "this should be ignored",
                "another_extra": "also ignored",
            }
        )
        result = self.parser.parse_and_validate(raw_output, GeneralOutput)

        # Should parse successfully and ignore extra fields
        assert "general_output" in result
        assert result["general_output"] == "test output"
        # Extra fields should not be present in the result
        assert "extra_field" not in result
        assert "another_extra" not in result

    def test_parse_json_with_missing_required_fields(self):
        """Test parsing when required fields are missing."""
        raw_output = json.dumps(
            {"malformed_blocks": []}
        )  # Missing required 'general_output'
        result = self.parser.parse_and_validate(raw_output, GeneralOutput)

        # Should have schema correction attempts
        assert "malformed_blocks" in result
        assert any(
            block.get("type") in ["SCHEMA_CORRECTION_SUCCESS", "JSON_REPAIR_ATTEMPTED"]
            for block in result["malformed_blocks"]
        )

    def test_parse_markdown_wrapped_json(self):
        """Test parsing of JSON wrapped in markdown code blocks."""
        raw_output = '```\n{\n  "general_output": "test markdown wrapped",\n  "malformed_blocks": []\n}\n```'
        result = self.parser.parse_and_validate(raw_output, GeneralOutput)

        assert "general_output" in result
        assert result["general_output"] == "test markdown wrapped"
        assert "malformed_blocks" in result

    def test_parse_markdown_wrapped_json_with_language_spec(self):
        """Test parsing of JSON with language specification in markdown."""
        raw_output = '```json\n{\n  "general_output": "test language specific",\n  "malformed_blocks": []\n}\n```'
        result = self.parser.parse_and_validate(raw_output, GeneralOutput)

        assert "general_output" in result
        assert result["general_output"] == "test language specific"

    def test_parse_empty_response(self):
        """Test parsing of empty response."""
        raw_output = ""
        result = self.parser.parse_and_validate(raw_output, GeneralOutput)

        assert "malformed_blocks" in result
        assert any(
            block.get("type") == "JSON_EXTRACTION_FAILED"
            for block in result["malformed_blocks"]
        )

    def test_parse_only_null_response(self):
        """Test parsing of 'null' response."""
        raw_output = "null"
        result = self.parser.parse_and_validate(raw_output, GeneralOutput)

        assert "malformed_blocks" in result
        assert any(
            block.get("type") == "JSON_EXTRACTION_FAILED"
            for block in result["malformed_blocks"]
        )

    def test_parse_single_string_response(self):
        """Test parsing of single string response instead of object."""
        raw_output = '"just a string"'
        result = self.parser.parse_and_validate(raw_output, GeneralOutput)

        assert "malformed_blocks" in result
        assert any(
            block.get("type") == "JSON_DECODE_ERROR" or "JSON_EXTRACTION_FAILED"
            for block in result["malformed_blocks"]
        )

    def test_parse_json_with_single_quotes(self):
        """Test parsing of JSON with single quotes instead of double quotes."""
        raw_output = (
            "{'general_output': 'test with single quotes', 'malformed_blocks': []}"
        )
        result = self.parser.parse_and_validate(raw_output, GeneralOutput)

        # Parser should handle single quotes
        assert "malformed_blocks" in result

    def test_parse_json_with_trailing_commas(self):
        """Test parsing of JSON with trailing commas."""
        raw_output = (
            '{"general_output": "test trailing comma", "malformed_blocks": [],}'
        )
        result = self.parser.parse_and_validate(raw_output, GeneralOutput)

        assert "general_output" in result
        assert result["general_output"] == "test trailing comma"

    def test_parse_malformed_code_changes(self):
        """Test parsing of malformed code changes in LLM output."""
        raw_output = json.dumps(
            {
                "COMMIT_MESSAGE": "Fix issue",
                "RATIONALE": "Testing malformed code changes",
                "CODE_CHANGES": [
                    {
                        "FILE_PATH": "test.py",
                        "ACTION": "INVALID_ACTION",
                        "FULL_CONTENT": "content",
                    },
                    {
                        "ACTION": "CREATE",
                        "FULL_CONTENT": "content",
                    },  # Missing FILE_PATH
                    {"FILE_PATH": "test2.py", "ACTION": "CREATE"},  # Missing content
                    {
                        "FILE_PATH": "test3.py",
                        "ACTION": "MODIFY",
                        "FULL_CONTENT": "content",
                        "DIFF_CONTENT": "diff",
                    },  # Both contents
                ],
            }
        )
        result = self.parser.parse_and_validate(raw_output, LLMOutput)

        # Should have malformed code change items
        assert "malformed_code_change_items" in result
        assert len(result["malformed_code_change_items"]) > 0

    def test_parse_llm_output_top_level_list(self):
        """Test parsing when LLM returns a list instead of object for LLMOutput."""
        raw_output = json.dumps(
            [
                {
                    "FILE_PATH": "test.py",
                    "ACTION": "CREATE",
                    "FULL_CONTENT": "print('hello')",
                }
            ]
        )
        result = self.parser.parse_and_validate(raw_output, LLMOutput)

        # Should handle the list appropriately and wrap it
        assert "malformed_blocks" in result
        assert any(
            block.get("type") == "TOP_LEVEL_LIST_WRAPPING"
            for block in result["malformed_blocks"]
        )

    def test_parse_self_improvement_top_level_list(self):
        """Test parsing when LLM returns a list instead of object for SelfImprovementAnalysisOutputV1."""
        suggestion = {
            "AREA": "Reasoning Quality",
            "PROBLEM": "Low reasoning quality detected",
            "PROPOSED_SOLUTION": "Improve reasoning algorithms",
            "EXPECTED_IMPACT": "High improvement potential",
            "PARETO_SCORE": 0.8,
            "VALIDATION_METHOD": "A/B testing",
        }
        raw_output = json.dumps([suggestion])
        result = self.parser.parse_and_validate(
            raw_output, SelfImprovementAnalysisOutputV1
        )

        # Should handle the list appropriately
        assert "malformed_blocks" in result
        assert any(
            block.get("type") == "TOP_LEVEL_LIST_WRAPPING"
            for block in result["malformed_blocks"]
        )

    def test_parse_self_improvement_single_suggestion_dict(self):
        """Test parsing when LLM returns a single suggestion dict instead of full object."""
        suggestion = {
            "AREA": "Reasoning Quality",
            "PROBLEM": "Low reasoning quality detected",
            "PROPOSED_SOLUTION": "Improve reasoning algorithms",
            "EXPECTED_IMPACT": "High improvement potential",
            "PARETO_SCORE": 0.8,
            "VALIDATION_METHOD": "A/B testing",
        }
        raw_output = json.dumps(suggestion)
        result = self.parser.parse_and_validate(
            raw_output, SelfImprovementAnalysisOutputV1
        )

        # Should detect and wrap the single suggestion
        assert "malformed_blocks" in result
        assert any(
            block.get("type") == "LLM_OUTPUT_MALFORMED"
            for block in result["malformed_blocks"]
        )

    def test_parse_circular_reference_json(self):
        """Test parsing of JSON that might have circular references."""
        # This is more about handling deeply nested structures
        deeply_nested = {
            "level1": {"level2": {"level3": {"level4": {"level5": "deep_value"}}}}
        }
        raw_output = json.dumps(deeply_nested)
        result = self.parser.parse_and_validate(raw_output, GeneralOutput)

        # Should handle nested objects
        assert result["general_output"] == str(deeply_nested)

    def test_parse_json_with_invalid_control_characters(self):
        """Test parsing of JSON with invalid control characters."""
        raw_output = '{"general_output": "test\\x00control\\x01chars"}'
        result = self.parser.parse_and_validate(raw_output, GeneralOutput)

        # Should handle control characters
        assert "malformed_blocks" in result

    def test_parse_json_with_unescaped_newlines(self):
        """Test parsing of JSON with unescaped newlines in strings."""
        raw_output = """{"general_output": "test line 1
        line 2"}"""
        result = self.parser.parse_and_validate(raw_output, GeneralOutput)

        # Should handle unescaped newlines during repair
        assert "malformed_blocks" in result

    def test_parse_json_with_unquoted_keys(self):
        """Test parsing of JSON with unquoted keys."""
        raw_output = "{general_output: 'test unquoted keys'}"
        result = self.parser.parse_and_validate(raw_output, GeneralOutput)

        # Should attempt to add quotes to unquoted keys
        assert "malformed_blocks" in result

    def test_parse_malformed_suggestion_items(self):
        """Test parsing of CritiqueOutput with malformed suggestion items."""
        raw_output = json.dumps(
            {
                "CRITIQUE_SUMMARY": "Test summary",
                "CRITIQUE_POINTS": [],
                "SUGGESTIONS": [
                    {"AREA": "Test Area"},  # Missing required fields
                    {"PROBLEM": "Test problem"},  # Missing required fields
                    {
                        "AREA": "Test Area",
                        "PROBLEM": "Test problem",
                        "PROPOSED_SOLUTION": "Solution",
                        "EXPECTED_IMPACT": "Impact",
                        "PARETO_SCORE": 0.8,
                        "VALIDATION_METHOD": "Method",
                    },
                    "just a string",  # Invalid format
                ],
                "malformed_blocks": [],
            }
        )
        result = self.parser.parse_and_validate(raw_output, CritiqueOutput)

        # Should handle malformed suggestions
        assert "malformed_blocks" in result
        assert (
            len(result["suggestions"]) >= 1
        )  # Should have at least one valid suggestion

    def test_parse_xml_wrapped_json(self):
        """Test parsing of JSON wrapped in XML-like tags."""
        raw_output = '<json_output>{"general_output": "test xml wrapped", "malformed_blocks": []}</json_output>'
        result = self.parser.parse_and_validate(raw_output, GeneralOutput)

        assert "general_output" in result
        assert result["general_output"] == "test xml wrapped"

    def test_parse_missing_end_xml_tag(self):
        """Test parsing of JSON with missing end XML tag."""
        raw_output = '<json_output>{"general_output": "missing end tag"}'
        result = self.parser.parse_and_validate(raw_output, GeneralOutput)

        # Should still be able to extract the JSON content
        assert "malformed_blocks" in result

    def test_parse_json_with_wrong_array_types(self):
        """Test parsing when arrays contain wrong types."""
        raw_output = json.dumps(
            {
                "ANALYSIS_SUMMARY": "Test summary",
                "IMPACTFUL_SUGGESTIONS": [
                    {
                        "AREA": "Test Area",
                        "PROBLEM": "Test problem",
                        "PROPOSED_SOLUTION": "Solution",
                        "EXPECTED_IMPACT": "Impact",
                        "PARETO_SCORE": 0.8,
                        "VALIDATION_METHOD": "Method",
                    },
                    "just a string instead of object",  # Wrong type
                    {
                        "AREA": "Another area",
                        "PROBLEM": "Another problem",
                        "PROPOSED_SOLUTION": "Another solution",
                        "EXPECTED_IMPACT": "Another impact",
                        "PARETO_SCORE": 0.9,
                        "VALIDATION_METHOD": "Another method",
                    },
                ],
            }
        )
        result = self.parser.parse_and_validate(
            raw_output, SelfImprovementAnalysisOutputV1
        )

        # Should handle malformed suggestion items
        assert "malformed_blocks" in result

    def test_parse_json_with_invalid_enum_values(self):
        """Test parsing when using invalid values for enum fields."""
        raw_output = json.dumps(
            {
                "conflict_type": "INVALID_CONFLICT_TYPE",  # Not a valid enum value
                "summary": "Test conflict",
                "involved_personas": ["Test Persona"],
                "conflicting_outputs_snippet": "Test snippet",
                "conflict_found": True,
            }
        )
        result = self.parser.parse_and_validate(raw_output, ConflictReport)

        # Should have validation errors for invalid enum
        assert "malformed_blocks" in result

    def test_parse_json_with_wrong_number_types(self):
        """Test parsing when numeric fields have wrong types."""
        raw_output = json.dumps(
            {
                "ANALYSIS_SUMMARY": "Test summary",
                "IMPACTFUL_SUGGESTIONS": [
                    {
                        "AREA": "Test Area",
                        "PROBLEM": "Test problem",
                        "PROPOSED_SOLUTION": "Solution",
                        "EXPECTED_IMPACT": "Impact",
                        "PARETO_SCORE": "0.8",  # Should be float, not string
                        "VALIDATION_METHOD": "Method",
                    }
                ],
            }
        )
        result = self.parser.parse_and_validate(
            raw_output, SelfImprovementAnalysisOutputV1
        )

        # Should handle type coercion
        assert "malformed_blocks" in result

    def test_parse_json_with_null_for_required_fields(self):
        """Test parsing when required fields are explicitly set to null."""
        raw_output = json.dumps(
            {
                "general_output": None,  # Explicitly null
                "malformed_blocks": [],
            }
        )
        result = self.parser.parse_and_validate(raw_output, GeneralOutput)

        # Should have schema correction for null required fields
        assert "malformed_blocks" in result

    def test_parse_json_with_extra_array_items(self):
        """Test parsing of arrays with invalid object structures."""
        raw_output = json.dumps(
            {
                "CRITIQUE_SUMMARY": "Test summary",
                "CRITIQUE_POINTS": [
                    {
                        "point_summary": "Valid point",
                        "details": "Details",
                        "recommendation": "Recommendation",
                    },
                    "invalid string item",  # Invalid item in array
                    {
                        "point_summary": "Another valid point",
                        "details": "More details",
                        "recommendation": "Another recommendation",
                    },
                ],
                "SUGGESTIONS": [
                    {
                        "AREA": "Test Area",
                        "PROBLEM": "Test problem",
                        "PROPOSED_SOLUTION": "Solution",
                        "EXPECTED_IMPACT": "Impact",
                        "PARETO_SCORE": 0.8,
                        "VALIDATION_METHOD": "Method",
                    }
                ],
            }
        )
        result = self.parser.parse_and_validate(raw_output, CritiqueOutput)

        # Should handle the invalid array item
        assert "malformed_blocks" in result

    def test_fallback_with_completely_invalid_input(self):
        """Test fallback behavior with completely invalid input."""
        raw_output = "This is not even remotely valid JSON or structured data"
        result = self.parser.parse_and_validate(raw_output, GeneralOutput)

        # Should return fallback with error information
        assert "malformed_blocks" in result
        assert any(
            block.get("type") == "LLM_OUTPUT_MALFORMED"
            for block in result["malformed_blocks"]
        )

    def test_parse_json_with_infinity_nan_values(self):
        """Test parsing of JSON with Infinity or NaN values."""
        # Note: JSON standard doesn't allow Infinity/NaN, but some LLMs might output them
        raw_output = (
            '{"general_output": "test", "malformed_blocks": [], "test_number": NaN}'
        )
        result = self.parser.parse_and_validate(raw_output, GeneralOutput)

        # Should fail to parse NaN but attempt repair
        assert "malformed_blocks" in result
