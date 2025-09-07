import pytest
from src.utils.output_parser import LLMOutputParser
from src.models import LLMOutput, CritiqueOutput, GeneralOutput, SelfImprovementAnalysisOutputV1
from pydantic import ValidationError

@pytest.fixture
def parser():
    return LLMOutputParser()

def test_parse_and_validate_valid_llm_output(parser):
    """Tests parsing and validation of a valid LLMOutput."""
    raw_output = """
    {
      "COMMIT_MESSAGE": "feat: Add new feature",
      "RATIONALE": "Implemented new feature as requested.",
      "CODE_CHANGES": []
    }
    """
    result = parser.parse_and_validate(raw_output, LLMOutput)
    assert result["COMMIT_MESSAGE"] == "feat: Add new feature"
    assert not result["malformed_blocks"]

def test_parse_and_validate_valid_critique_output(parser):
    """Tests parsing and validation of a valid CritiqueOutput."""
    raw_output = """
    {
      "CRITIQUE_SUMMARY": "Good overall, but needs tests.",
      "CRITIQUE_POINTS": [{"point_summary": "Missing tests", "details": "No unit tests found."}],
      "SUGGESTIONS": ["Add unit tests for core logic"],
      "malformed_blocks": []
    }
    """
    result = parser.parse_and_validate(raw_output, CritiqueOutput)
    assert result["CRITIQUE_SUMMARY"] == "Good overall, but needs tests."
    assert len(result["CRITIQUE_POINTS"]) == 1
    assert not result["malformed_blocks"]

def test_parse_and_validate_invalid_json_string(parser):
    """Tests handling of an invalid JSON string."""
    raw_output = "This is not JSON"
    result = parser.parse_and_validate(raw_output, GeneralOutput)
    assert "JSON_EXTRACTION_FAILED" in result["malformed_blocks"][0]["type"]
    assert "general_output" in result
    assert "No valid JSON data could be extracted or parsed." in result["general_output"]

def test_parse_and_validate_json_with_markdown_fences(parser):
    """Tests parsing JSON embedded in markdown fences."""
    raw_output = """
    ```json
    {
      "general_output": "This is a valid JSON output from markdown."
    }
    ```
    """
    result = parser.parse_and_validate(raw_output, GeneralOutput)
    assert result["general_output"] == "This is a valid JSON output from markdown."
    assert not result["malformed_blocks"]

def test_parse_and_validate_json_with_conversational_filler(parser):
    """Tests parsing JSON with conversational filler."""
    raw_output = """
    Here is the JSON output:
    {
      "general_output": "Hello from LLM."
    }
    Thanks!
    """
    result = parser.parse_and_validate(raw_output, GeneralOutput)
    assert result["general_output"] == "Hello from LLM."
    assert not result["malformed_blocks"]

def test_parse_and_validate_malformed_json_with_repair(parser):
    """Tests parsing malformed JSON that can be repaired."""
    raw_output = """
    {
      "key": "value",
      "another_key": "another_value",
    }
    """ # Trailing comma
    result = parser.parse_and_validate(raw_output, GeneralOutput)
    assert result["general_output"] == '{"key": "value", "another_key": "another_value"}'
    assert any("JSON_REPAIR_ATTEMPTED" == block["type"] for block in result["malformed_blocks"])

def test_parse_and_validate_schema_mismatch(parser):
    """Tests validation failure when JSON structure doesn't match schema."""
    raw_output = """
    {
      "not_a_commit_message": "Invalid field",
      "RATIONALE": "Some rationale",
      "CODE_CHANGES": []
    }
    """
    result = parser.parse_and_validate(raw_output, LLMOutput)
    assert any("SCHEMA_VALIDATION_ERROR" == block["type"] for block in result["malformed_blocks"])
    assert "COMMIT_MESSAGE" in result # Fallback should still provide the field

def test_parse_and_validate_top_level_list_for_dict_schema(parser):
    """Tests handling of a top-level JSON array when a dict is expected."""
    raw_output = """
    [
      {"AREA": "Robustness", "PROBLEM": "Issue 1", "PROPOSED_SOLUTION": "Fix 1", "EXPECTED_IMPACT": "Impact 1", "CODE_CHANGES_SUGGESTED": []},
      {"AREA": "Efficiency", "PROBLEM": "Issue 2", "PROPOSED_SOLUTION": "Fix 2", "EXPECTED_IMPACT": "Impact 2", "CODE_CHANGES_SUGGESTED": []}
    ]
    """
    result = parser.parse_and_validate(raw_output, SelfImprovementAnalysisOutputV1)
    assert result["ANALYSIS_SUMMARY"] == "LLM returned an array of suggestions instead of the full analysis. Review the 'IMPACTFUL_SUGGESTIONS' section."
    assert len(result["IMPACTFUL_SUGGESTIONS"]) == 2
    assert any("TOP_LEVEL_LIST_WRAPPING" == block["type"] for block in result["malformed_blocks"])

def test_parse_and_validate_single_suggestion_for_self_improvement_schema(parser):
    """Tests handling of a single suggestion dict when full SelfImprovementAnalysisOutput is expected."""
    raw_output = """
    {
      "AREA": "Robustness",
      "PROBLEM": "Issue 1",
      "PROPOSED_SOLUTION": "Fix 1",
      "EXPECTED_IMPACT": "Impact 1",
      "CODE_CHANGES_SUGGESTED": []
    }
    """
    result = parser.parse_and_validate(raw_output, SelfImprovementAnalysisOutputV1)
    assert "LLM returned a single suggestion item" in result["ANALYSIS_SUMMARY"]
    assert len(result["IMPACTFUL_SUGGESTIONS"]) == 1
    assert result["IMPACTFUL_SUGGESTIONS"][0]["AREA"] == "Robustness"

def test_parse_and_validate_empty_string(parser):
    """Tests handling of an empty string input."""
    raw_output = ""
    result = parser.parse_and_validate(raw_output, GeneralOutput)
    assert "JSON_EXTRACTION_FAILED" in result["malformed_blocks"][0]["type"]
    assert "No valid JSON data could be extracted or parsed." in result["general_output"]

# Add more tests for _extract_first_outermost_json, _repair_json_string, _force_close_truncated_json