import pytest
import json
from pydantic import ValidationError
import re # For checking raw string snippets

# Import all models and the parser
from src.utils.output_parser import LLMOutputParser
from src.models import (
    LLMOutput, CodeChange, ContextAnalysisOutput, CritiqueOutput, GeneralOutput,
    ConflictReport, SelfImprovementAnalysisOutput, SelfImprovementAnalysisOutputV1
)

# Fixture for the parser
@pytest.fixture
def parser():
    return LLMOutputParser()

# Helper to check for malformed_blocks
def assert_malformed_block_present(result, block_type_substring):
    """Asserts that a malformed_block of a specific type is present in the result."""
    assert "malformed_blocks" in result
    assert any(block_type_substring in block.get("type", "") for block in result["malformed_blocks"]), \
        f"Expected malformed block type '{block_type_substring}' not found in {result['malformed_blocks']}"

# --- Test Cases for LLMOutput ---

def test_llm_output_valid_json(parser):
    raw_output = """
    {
      "COMMIT_MESSAGE": "Feat: Add new user authentication module",
      "RATIONALE": "Implemented a secure authentication flow.",
      "CODE_CHANGES": [
        {
          "FILE_PATH": "src/auth/auth_service.py",
          "ACTION": "ADD",
          "FULL_CONTENT": "def login(): pass"
        }
      ],
      "malformed_blocks": []
    }
    """
    result = parser.parse_and_validate(raw_output, LLMOutput)
    assert result["COMMIT_MESSAGE"] == "Feat: Add new user authentication module"
    assert len(result["CODE_CHANGES"]) == 1
    assert result["CODE_CHANGES"][0]["FILE_PATH"] == "src/auth/auth_service.py" # Assert relative path
    assert not result["malformed_blocks"]

def test_llm_output_markdown_json(parser):
    raw_output = """
    Here is the proposed change:
    ```json
    {
      "COMMIT_MESSAGE": "Fix: Typo in README",
      "RATIONALE": "Corrected a spelling mistake.",
      "CODE_CHANGES": [],
      "malformed_blocks": []
    }
    ```
    Please review.
    """
    result = parser.parse_and_validate(raw_output, LLMOutput)
    assert result["COMMIT_MESSAGE"] == "Fix: Typo in README"
    assert not result["malformed_blocks"]

def test_llm_output_extra_text_and_markers(parser):
    raw_output = """
    Some introductory text.
    START_JSON_OUTPUT
    {
      "COMMIT_MESSAGE": "Refactor: Improve logging",
      "RATIONALE": "Standardized logging format.",
      "CODE_CHANGES": [],
      "malformed_blocks": []
    }
    END_JSON_OUTPUT
    Some concluding remarks.
    """
    result = parser.parse_and_validate(raw_output, LLMOutput)
    assert result["COMMIT_MESSAGE"] == "Refactor: Improve logging"
    assert not result["malformed_blocks"]

def test_llm_output_malformed_json_repair(parser):
    raw_output = """
    {
      'COMMIT_MESSAGE': "Feat: Add feature",
      'RATIONALE': "Added a new feature.",
      'CODE_CHANGES': [
        {
          'FILE_PATH': "src/new_feature.py",
          'ACTION': "ADD",
          'FULL_CONTENT': "print('hello')"
        },
      ] # Trailing comma here
    }
    """
    result = parser.parse_and_validate(raw_output, LLMOutput)
    assert result["COMMIT_MESSAGE"] == "Feat: Add feature"
    assert len(result["CODE_CHANGES"]) == 1
    assert result["CODE_CHANGES"][0]["FILE_PATH"] == "src/new_feature.py" # Assert relative path
    assert_malformed_block_present(result, "JSON_REPAIR_ATTEMPTED")
    assert_malformed_block_present(result, "LLM_OUTPUT_MALFORMED") # Should be present due to repairs

def test_llm_output_handles_list_of_code_changes(parser):
    raw_output = """
    [
      {
        "FILE_PATH": "src/app.py",
        "ACTION": "MODIFY",
        "FULL_CONTENT": "new content"
      },
      {
        "FILE_PATH": "src/core.py",
        "ACTION": "ADD",
        "FULL_CONTENT": "new file content"
      }
    ]
    """
    result = parser.parse_and_validate(raw_output, LLMOutput)
    assert "COMMIT_MESSAGE" in result
    assert "LLM returned multiple code changes directly" in result["RATIONALE"]
    assert len(result["CODE_CHANGES"]) == 2
    assert result["CODE_CHANGES"][0]["FILE_PATH"] == "src/app.py" # Assert relative path
    assert_malformed_block_present(result, "LLM_OUTPUT_MALFORMED") # Should be present due to list wrapping

def test_llm_output_handles_raw_code_change_dict(parser):
    raw_output = """
    {
      "FILE_PATH": "src/config.py",
      "ACTION": "MODIFY",
      "FULL_CONTENT": "updated config"
    }
    """
    result = parser.parse_and_validate(raw_output, LLMOutput)
    assert "COMMIT_MESSAGE" in result
    assert "LLM generated a direct code change" in result["RATIONALE"]
    assert len(result["CODE_CHANGES"]) == 1
    assert result["CODE_CHANGES"][0]["FILE_PATH"] == "src/config.py" # Assert relative path
    assert_malformed_block_present(result, "LLM_OUTPUT_MALFORMED") # Should be present due to wrapping

def test_llm_output_empty_string_fallback(parser):
    raw_output = ""
    result = parser.parse_and_validate(raw_output, LLMOutput)
    assert "LLM_OUTPUT_ERROR" in result["COMMIT_MESSAGE"]
    assert "No valid JSON data could be extracted or parsed" in result["RATIONALE"]
    assert not result["CODE_CHANGES"]
    assert_malformed_block_present(result, "JSON_EXTRACTION_FAILED")
    assert_malformed_block_present(result, "LLM_OUTPUT_MALFORMED") # Should be present due to fallback

def test_llm_output_non_json_string_fallback(parser):
    raw_output = "This is just a plain text response, not JSON at all."
    result = parser.parse_and_validate(raw_output, LLMOutput)
    assert "LLM_OUTPUT_ERROR" in result["COMMIT_MESSAGE"]
    assert "No valid JSON data could be extracted or parsed" in result["RATIONALE"]
    assert not result["CODE_CHANGES"]
    assert_malformed_block_present(result, "JSON_EXTRACTION_FAILED")
    assert_malformed_block_present(result, "LLM_OUTPUT_MALFORMED") # Should be present due to fallback

# --- Test Cases for CritiqueOutput ---

def test_critique_output_valid_json(parser):
    raw_output = """
    {
      "CRITIQUE_SUMMARY": "Overall good, but needs more tests.",
      "CRITIQUE_POINTS": [
        {"point_summary": "Low test coverage", "details": "Many functions lack unit tests."}
      ],
      "SUGGESTIONS": ["Add unit tests for data_processor.py"],
      "malformed_blocks": []
    }
    """
    result = parser.parse_and_validate(raw_output, CritiqueOutput)
    assert result["CRITIQUE_SUMMARY"] == "Overall good, but needs more tests."
    assert len(result["CRITIQUE_POINTS"]) == 1
    assert len(result["SUGGESTIONS"]) == 1
    assert not result["malformed_blocks"]

def test_critique_output_handles_list_of_strings_as_suggestions(parser):
    raw_output = """
    [
      "Suggestion 1: Improve error handling.",
      "Suggestion 2: Add more unit tests."
    ]
    """
    result = parser.parse_and_validate(raw_output, CritiqueOutput)
    assert "CRITIQUE_SUMMARY" in result
    assert "LLM returned a list of strings as suggestions" in result["CRITIQUE_SUMMARY"]
    assert len(result["SUGGESTIONS"]) == 2
    assert result["SUGGESTIONS"][0] == "Suggestion 1: Improve error handling."
    assert_malformed_block_present(result, "LLM_OUTPUT_MALFORMED") # Should be present due to list wrapping

def test_critique_output_handles_list_of_critique_points(parser):
    raw_output = """
    [
      {"point_summary": "Missing validation", "details": "Input is not validated."},
      {"point_summary": "Poor testability", "details": "Hard to write tests."}
    ]
    """
    result = parser.parse_and_validate(raw_output, CritiqueOutput)
    assert "CRITIQUE_SUMMARY" in result
    assert "LLM returned a list of dicts as critique points" in result["CRITIQUE_SUMMARY"]
    assert len(result["CRITIQUE_POINTS"]) == 2
    assert result["CRITIQUE_POINTS"][0]["point_summary"] == "Missing validation"
    assert_malformed_block_present(result, "LLM_OUTPUT_MALFORMED") # Should be present due to list wrapping

def test_critique_output_handles_mixed_list_fallback(parser):
    raw_output = """
    [
      "This is a string suggestion.",
      {"point_summary": "This is a dict point", "details": "Details here."},
      "Another string."
    ]
    """
    result = parser.parse_and_validate(raw_output, CritiqueOutput)
    assert "CRITIQUE_SUMMARY" in result
    assert "LLM returned a mixed/unexpected list" in result["CRITIQUE_SUMMARY"]
    assert not result["CRITIQUE_POINTS"]
    assert not result["SUGGESTIONS"]
    assert_malformed_block_present(result, "LLM_OUTPUT_MALFORMED") # Should be present due to list wrapping

# --- Test Cases for ContextAnalysisOutput ---

def test_context_analysis_valid_json(parser):
    raw_output = """
    {
      "key_modules": [{"name": "src/core.py", "purpose": "Main debate logic"}],
      "security_concerns": ["Potential API key exposure"],
      "architectural_patterns": ["Socratic Debate"],
      "performance_bottlenecks": [],
      "security_summary": {"vulnerabilities": ["None identified"], "critical_files": []},
      "architecture_summary": {"patterns": ["Modular"], "structural_issues": []},
      "devops_summary": {"deployment_issues": [], "ci_cd_issues": []},
      "testing_summary": {"coverage_gaps": [], "recommended_tests": []},
      "general_overview": "A high-level overview of the project.",
      "malformed_blocks": []
    }
    """
    result = parser.parse_and_validate(raw_output, ContextAnalysisOutput)
    assert len(result["key_modules"]) == 1
    assert result["key_modules"][0]["name"] == "src/core.py" # Assert relative path
    assert result["general_overview"] == "A high-level overview of the project."
    assert not result["malformed_blocks"]

def test_context_analysis_empty_list_fields(parser):
    raw_output = """
    {
      "key_modules": [],
      "security_concerns": [],
      "architectural_patterns": [],
      "performance_bottlenecks": [],
      "security_summary": {"vulnerabilities": [], "critical_files": []},
      "architecture_summary": {"patterns": [], "structural_issues": []},
      "devops_summary": {"deployment_issues": [], "ci_cd_issues": []},
      "testing_summary": {"coverage_gaps": [], "recommended_tests": []},
      "general_overview": "Minimal context.",
      "malformed_blocks": []
    }
    """
    result = parser.parse_and_validate(raw_output, ContextAnalysisOutput)
    assert not result["key_modules"]
    assert result["general_overview"] == "Minimal context."
    assert not result["malformed_blocks"]

# --- Test Cases for ConflictReport ---

def test_conflict_report_valid_json(parser):
    raw_output = """
    {
      "conflict_type": "METHODOLOGY_DISAGREEMENT",
      "summary": "Personas disagree on implementation approach.",
      "involved_personas": ["Visionary_Generator", "Skeptical_Generator"],
      "conflicting_outputs_snippet": "Gen1: Use microservices. Gen2: Use monolith.",
      "proposed_resolution_paths": ["Evaluate trade-offs", "Seek arbitrator input"],
      "conflict_found": true,
      "malformed_blocks": []
    }
    """
    result = parser.parse_and_validate(raw_output, ConflictReport)
    assert result["conflict_type"] == "METHODOLOGY_DISAGREEMENT"
    assert result["conflict_found"] is True
    assert len(result["involved_personas"]) == 2
    assert not result["malformed_blocks"]

def test_conflict_report_handles_list_of_dicts(parser):
    raw_output = """
    [
      {
        "conflict_type": "METHODOLOGY_DISAGREEMENT",
        "summary": "First conflict",
        "involved_personas": ["A"],
        "conflicting_outputs_snippet": "snippet1",
        "proposed_resolution_paths": [],
        "conflict_found": true
      },
      {
        "conflict_type": "DATA_DISCREPANCY",
        "summary": "Second conflict",
        "involved_personas": ["B"],
        "conflicting_outputs_snippet": "snippet2",
        "proposed_resolution_paths": [],
        "conflict_found": true
      }
    ]
    """
    result = parser.parse_and_validate(raw_output, ConflictReport)
    assert result["conflict_type"] == "METHODOLOGY_DISAGREEMENT" # Should pick the first valid dict's type
    assert "First conflict" in result["summary"] # Should contain the summary from the first dict
    assert_malformed_block_present(result, "LLM_OUTPUT_MALFORMED") # Should be present due to list wrapping

def test_conflict_report_handles_list_of_non_dicts_fallback(parser):
    raw_output = """
    [
      "Conflict summary 1",
      "Conflict summary 2"
    ]
    """
    result = parser.parse_and_validate(raw_output, ConflictReport)
    assert result["conflict_type"] == "METHODOLOGY_DISAGREEMENT"
    assert "LLM returned a list instead of a single ConflictReport object" in result["summary"]
    assert result["conflict_found"] is True
    assert_malformed_block_present(result, "LLM_OUTPUT_MALFORMED") # Should be present due to list wrapping

# --- Test Cases for GeneralOutput ---

def test_general_output_valid_json(parser):
    raw_output = """
    {
      "general_output": "This is a general synthesized response.",
      "malformed_blocks": []
    }
    """
    result = parser.parse_and_validate(raw_output, GeneralOutput)
    assert result["general_output"] == "This is a general synthesized response."
    assert not result["malformed_blocks"]

def test_general_output_handles_list_of_strings(parser):
    raw_output = """
    [
      "First point of general output.",
      "Second point of general output."
    ]
    """
    result = parser.parse_and_validate(raw_output, GeneralOutput)
    assert "general_output" in result
    assert "First point of general output.\nSecond point of general output." in result["general_output"]
    assert_malformed_block_present(result, "LLM_OUTPUT_MALFORMED") # Should be present due to list wrapping

def test_general_output_handles_list_of_dicts(parser):
    raw_output = """
    [
      {"item": "value1"},
      {"item": "value2"}
    ]
    """
    result = parser.parse_and_validate(raw_output, GeneralOutput)
    assert "general_output" in result
    assert "LLM returned a list of objects. Summarized content:" in result["general_output"]
    assert_malformed_block_present(result, "LLM_OUTPUT_MALFORMED") # Should be present due to list wrapping

# --- Test Cases for SelfImprovementAnalysisOutput ---

def test_self_improvement_parser_valid_v1_json(parser):
    raw_output = """
    {
      "ANALYSIS_SUMMARY": "Codebase needs refactoring.",
      "IMPACTFUL_SUGGESTIONS": [
        {
          "AREA": "Maintainability",
          "PROBLEM": "High cyclomatic complexity",
          "PROPOSED_SOLUTION": "Refactor complex functions",
          "EXPECTED_IMPACT": "Improved readability",
          "CODE_CHANGES_SUGGESTED": []
        }
      ],
      "malformed_blocks": []
    }
    """
    result = parser.parse_and_validate(raw_output, SelfImprovementAnalysisOutput)
    assert result["version"] == "1.0"
    assert result["data"]["ANALYSIS_SUMMARY"] == "Codebase needs refactoring."
    assert len(result["data"]["IMPACTFUL_SUGGESTIONS"]) == 1
    assert not result["malformed_blocks"]

def test_self_improvement_parser_valid_versioned_json(parser):
    raw_output = """
    {
      "version": "1.0",
      "data": {
        "ANALYSIS_SUMMARY": "Overall healthy codebase.",
        "IMPACTFUL_SUGGESTIONS": [],
        "malformed_blocks": []
      },
      "metadata": {"timestamp": "2023-01-01"},
      "malformed_blocks": []
    }
    """
    result = parser.parse_and_validate(raw_output, SelfImprovementAnalysisOutput)
    assert result["version"] == "1.0"
    assert result["data"]["ANALYSIS_SUMMARY"] == "Overall healthy codebase."
    assert not result["data"]["IMPACTFUL_SUGGESTIONS"]
    assert not result["malformed_blocks"]

def test_self_improvement_parser_handles_list_of_suggestions(parser):
    raw_output = """
    [
      {
        "AREA": "Robustness",
        "PROBLEM": "Schema validation failures",
        "PROPOSED_SOLUTION": "Implement stricter input validation",
        "EXPECTED_IMPACT": "Reduced errors",
        "CODE_CHANGES_SUGGESTED": []
      },
      {
        "AREA": "Maintainability",
        "PROBLEM": "PEP8 violations",
        "PROPOSED_SOLUTION": "Automate linting",
        "EXPECTED_IMPACT": "Cleaner code",
        "CODE_CHANGES_SUGGESTED": []
      }
    ]
    """
    result = parser.parse_and_validate(raw_output, SelfImprovementAnalysisOutput)
    assert result["version"] == "1.0"
    assert "ANALYSIS_SUMMARY" in result["data"]
    assert "LLM returned an array of suggestions" in result["data"]["ANALYSIS_SUMMARY"]
    assert len(result["data"]["IMPACTFUL_SUGGESTIONS"]) == 2
    assert result["data"]["IMPACTFUL_SUGGESTIONS"][0]["AREA"] == "Robustness"
    assert_malformed_block_present(result, "LLM_OUTPUT_MALFORMED") # Should be present due to list wrapping

def test_self_improvement_parser_handles_single_suggestion_dict(parser):
    raw_output = """
    {
      "AREA": "Efficiency",
      "PROBLEM": "High token usage",
      "PROPOSED_SOLUTION": "Optimize prompt engineering",
      "EXPECTED_IMPACT": "Reduced costs",
      "CODE_CHANGES_SUGGESTED": []
    }
    """
    result = parser.parse_and_validate(raw_output, SelfImprovementAnalysisOutput)
    assert result["version"] == "1.0"
    assert "ANALYSIS_SUMMARY" in result["data"]
    assert "LLM returned a single suggestion item" in result["data"]["ANALYSIS_SUMMARY"]
    assert len(result["data"]["IMPACTFUL_SUGGESTIONS"]) == 1
    assert result["data"]["IMPACTFUL_SUGGESTIONS"][0]["AREA"] == "Efficiency"
    assert_malformed_block_present(result, "LLM_OUTPUT_MALFORMED") # Should be present due to wrapping

def test_self_improvement_parser_handles_malformed_suggestion_in_list(parser):
    # List contains a malformed CodeChange item (invalid ACTION)
    raw_output = """
    [
      {
        "AREA": "Robustness",
        "PROBLEM": "Schema validation failures",
        "PROPOSED_SOLUTION": "Implement stricter input validation",
        "EXPECTED_IMPACT": "Reduced errors",
        "CODE_CHANGES_SUGGESTED": [
            {"FILE_PATH": "path/to/file.py", "ACTION": "INVALID_ACTION", "FULL_CONTENT": "content"}
        ]
      }
    ]
    """
    result = parser.parse_and_validate(raw_output, SelfImprovementAnalysisOutput)
    assert result["version"] == "1.0"
    assert "ANALYSIS_SUMMARY" in result["data"]
    assert len(result["data"]["IMPACTFUL_SUGGESTIONS"]) == 1
    # The invalid CodeChange should be filtered out or marked as malformed by the model_validator
    assert not result["data"]["IMPACTFUL_SUGGESTIONS"][0]["CODE_CHANGES_SUGGESTED"]
    assert_malformed_block_present(result, "LLM_OUTPUT_MALFORMED") # Due to overall transformation/repair
    assert_malformed_block_present(result, "CODE_CHANGE_SCHEMA_VALIDATION_ERROR") # Due to inner validation error

def test_self_improvement_parser_empty_suggestions_list(parser):
    raw_output = """
    {
      "ANALYSIS_SUMMARY": "No issues found.",
      "IMPACTFUL_SUGGESTIONS": [],
      "malformed_blocks": []
    }
    """
    result = parser.parse_and_validate(raw_output, SelfImprovementAnalysisOutput)
    assert result["version"] == "1.0"
    assert result["data"]["ANALYSIS_SUMMARY"] == "No issues found."
    assert not result["data"]["IMPACTFUL_SUGGESTIONS"]
    assert not result["malformed_blocks"]

# --- General Robustness Tests ---

def test_parser_handles_json_with_start_marker_only(parser):
    raw_output = """
    Some text.
    START_JSON_OUTPUT
    {"general_output": "This is valid JSON after the start marker."}
    """
    result = parser.parse_and_validate(raw_output, GeneralOutput)
    assert result["general_output"] == 'This is valid JSON after the start marker.'
    assert_malformed_block_present(result, "MISSING_END_MARKER")
    assert_malformed_block_present(result, "LLM_OUTPUT_MALFORMED") # Should be present due to missing end marker

def test_parser_handles_json_with_unquoted_keys(parser):
    raw_output = """
    {
      key: "value",
      "another_key": "another_value"
    }
    """
    result = parser.parse_and_validate(raw_output, GeneralOutput)
    assert result["general_output"] == '{"key": "value", "another_key": "another_value"}'
    assert_malformed_block_present(result, "JSON_REPAIR_ATTEMPTED")
    assert_malformed_block_present(result, "LLM_OUTPUT_MALFORMED") # Should be present due to repairs
    assert any("Added quotes to unquoted keys" in d.get("details", "") for block in result["malformed_blocks"] for d in block.get("details", []))

def test_parser_handles_json_with_trailing_commas(parser):
    raw_output = """
    {
      "key": "value",
      "another_key": "another_value",
    }
    """
    result = parser.parse_and_validate(raw_output, GeneralOutput)
    assert result["general_output"] == '{"key": "value", "another_key": "another_value"}'
    assert_malformed_block_present(result, "JSON_REPAIR_ATTEMPTED")
    assert_malformed_block_present(result, "LLM_OUTPUT_MALFORMED") # Should be present due to repairs
    assert any("Removed trailing commas" in d.get("details", "") for block in result["malformed_blocks"] for d in block.get("details", []))

def test_parser_handles_json_with_numbered_array_elements(parser):
    raw_output = """
    {
      "items": [
        0: {"id": 1, "name": "Item 1"},
        1: {"id": 2, "name": "Item 2"}
      ]
    }
    """
    result = parser.parse_and_validate(raw_output, GeneralOutput)
    assert result["general_output"] == '{"items": [{"id": 1, "name": "Item 1"}, {"id": 2, "name": "Item 2"}]}'
    assert_malformed_block_present(result, "JSON_REPAIR_ATTEMPTED")
    assert_malformed_block_present(result, "LLM_OUTPUT_MALFORMED") # Should be present due to repairs
    assert any("Fixed numbered array elements" in d.get("details", "") for block in result["malformed_blocks"] for d in block.get("details", []))

def test_parser_handles_json_with_incorrectly_wrapped_array(parser):
    raw_output = """
    {
      "data": "[{\\"key\\": \\"value\\"}]"
    }
    """
    result = parser.parse_and_validate(raw_output, GeneralOutput)
    assert result["general_output"] == '{"data": [{"key": "value"}]}'
    assert_malformed_block_present(result, "JSON_REPAIR_ATTEMPTED")
    assert_malformed_block_present(result, "LLM_OUTPUT_MALFORMED") # Should be present due to repairs
    assert any("Fixed array incorrectly wrapped in quotes" in d.get("details", "") for block in result["malformed_blocks"] for d in block.get("details", []))

def test_parser_raises_validation_error_for_invalid_content(parser):
    raw_output = """
    {
      "COMMIT_MESSAGE": "Invalid action test",
      "RATIONALE": "Testing invalid action.",
      "CODE_CHANGES": [
        {
          "FILE_PATH": "src/test.py",
          "ACTION": "INVALID_ACTION",
          "FULL_CONTENT": "print('hello')"
        }
      ],
      "malformed_blocks": []
    }
    """
    result = parser.parse_and_validate(raw_output, LLMOutput)
    assert result["COMMIT_MESSAGE"] == "LLM_OUTPUT_ERROR" # Fallback message
    assert "Schema validation failed: value_error, Invalid action: 'INVALID_ACTION'" in result["RATIONALE"]
    assert not result["CODE_CHANGES"]
    assert_malformed_block_present(result, "SCHEMA_VALIDATION_ERROR")
    assert_malformed_block_present(result, "LLM_OUTPUT_MALFORMED") # Should be present due to fallback

def test_parser_handles_empty_json_object(parser):
    raw_output = "{}"
    result = parser.parse_and_validate(raw_output, GeneralOutput)
    assert result["general_output"] == "{}"
    assert not result["malformed_blocks"] # Empty object is valid JSON, no repair needed

def test_parser_handles_empty_json_array(parser):
    raw_output = "[]"
    result = parser.parse_and_validate(raw_output, GeneralOutput)
    assert result["general_output"] == "[]"
    assert_malformed_block_present(result, "LLM_OUTPUT_MALFORMED") # Should be present due to list wrapping

def test_parser_handles_json_with_start_marker_and_no_end_marker_but_valid_json_after(parser):
    raw_output = """
    Some preamble.
    START_JSON_OUTPUT
    {
      "general_output": "This is valid JSON after the start marker."
    }
    More text.
    """
    result = parser.parse_and_validate(raw_output, GeneralOutput)
    assert result["general_output"] == "This is valid JSON after the start marker."
    assert_malformed_block_present(result, "MISSING_END_MARKER")
    assert_malformed_block_present(result, "LLM_OUTPUT_MALFORMED") # Should be present due to missing end marker

def test_parser_handles_json_with_start_marker_and_no_end_marker_and_malformed_json_after(parser):
    raw_output = """
    Some preamble.
    START_JSON_OUTPUT
    {
      "general_output": "This is malformed JSON after the start marker.",
      "key": "value",
      "trailing_comma":
    }
    More text.
    """
    result = parser.parse_and_validate(raw_output, GeneralOutput)
    assert "general_output" in result
    assert "LLM output could not be fully parsed or validated" in result["malformed_blocks"][0]["message"]
    assert_malformed_block_present(result, "MISSING_END_MARKER")
    assert_malformed_block_present(result, "JSON_DECODE_ERROR")
    assert_malformed_block_present(result, "LLM_OUTPUT_MALFORMED") # Should be present due to fallback