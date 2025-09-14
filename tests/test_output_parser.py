import pytest
from src.utils.output_parser import LLMOutputParser
from src.models import (
    LLMOutput,
    CritiqueOutput,
    GeneralOutput,
    SelfImprovementAnalysisOutputV1,
    SuggestionItem,
)  # NEW: Import SuggestionItem
from pydantic import ValidationError
import json  # NEW: Import json for direct json.loads calls in tests


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
      "CRITIQUE_POINTS": [{"point_summary": "Missing tests", "details": "No unit tests found.", "recommendation": "Add tests"}],
      "SUGGESTIONS": [],
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
    assert (
        "No valid JSON data could be extracted or parsed." in result["general_output"]
    )  # FIX: Check for substring
    assert any(
        block["type"] == "JSON_EXTRACTION_FAILED"
        for block in result["malformed_blocks"]
    )


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
    """  # Trailing comma
    result = parser.parse_and_validate(raw_output, GeneralOutput)
    assert json.loads(result["general_output"]) == {
        "key": "value",
        "another_key": "another_value",
    }  # FIX: Parse and compare dicts
    assert any(
        "JSON_REPAIR_ATTEMPTED" == block["type"] for block in result["malformed_blocks"]
    )


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
    assert any(
        "SCHEMA_VALIDATION_ERROR" == block["type"]
        for block in result["malformed_blocks"]
    )
    assert "COMMIT_MESSAGE" in result  # Fallback should still provide the field


def test_parse_and_validate_top_level_list_for_dict_schema(parser):
    """Tests handling of a top-level JSON array when a dict is expected."""
    raw_output = """
    [
      {"AREA": "Robustness", "PROBLEM": "Issue 1", "PROPOSED_SOLUTION": "Fix 1", "EXPECTED_IMPACT": "Impact 1", "CODE_CHANGES_SUGGESTED": []},
      {"AREA": "Efficiency", "PROBLEM": "Issue 2", "PROPOSED_SOLUTION": "Fix 2", "EXPECTED_IMPACT": "Impact 2", "CODE_CHANGES_SUGGESTED": []}
    ]
    """
    result = parser.parse_and_validate(raw_output, SelfImprovementAnalysisOutputV1)
    assert (
        result["ANALYSIS_SUMMARY"]
        == "LLM returned an array of suggestions instead of the full analysis. Review the 'IMPACTFUL_SUGGESTIONS' section."
    )
    assert len(result["IMPACTFUL_SUGGESTIONS"]) == 2
    assert result["IMPACTFUL_SUGGESTIONS"][0]["AREA"] == "Robustness"
    assert any(
        "TOP_LEVEL_LIST_WRAPPING" == block["type"]
        for block in result["malformed_blocks"]
    )


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
    assert (
        "No valid JSON data could be extracted or parsed." in result["general_output"]
    )
    assert any(
        block["type"] == "JSON_EXTRACTION_FAILED"
        for block in result["malformed_blocks"]
    )


def test_parse_and_validate_unbalanced_quotes_repair(parser):
    """Tests repair of unbalanced quotes."""
    raw_output = '{"key": "value}'
    result = parser.parse_and_validate(raw_output, GeneralOutput)
    assert json.loads(result["general_output"]) == {
        "key": "value"
    }  # FIX: Parse and compare dicts
    assert any(b["type"] == "JSON_REPAIR_ATTEMPTED" for b in result["malformed_blocks"])
    assert any(
        "Added missing closing braces." in d["details"]
        for b in result["malformed_blocks"]
        for d in b.get("details", [])
    )


def test_parse_and_validate_unescaped_newline_repair(parser):
    """Tests repair of unescaped newlines within a string."""
    raw_output = '{"key": "value with\nnewline"}'
    result = parser.parse_and_validate(raw_output, GeneralOutput)
    assert json.loads(result["general_output"]) == {
        "key": "value with\\nnewline"
    }  # FIX: Parse and compare dicts
    assert any(b["type"] == "JSON_REPAIR_ATTEMPTED" for b in result["malformed_blocks"])
    assert any(
        "Escaped unescaped newlines within strings." in d["details"]
        for b in result["malformed_blocks"]
        for d in b.get("details", [])
    )


def test_parse_and_validate_top_level_list_of_strings_critique_output(parser):
    """Tests handling of a top-level list of strings for CritiqueOutput."""
    raw_output = '["Suggestion 1", "Suggestion 2"]'
    result = parser.parse_and_validate(raw_output, CritiqueOutput)
    assert (
        result["CRITIQUE_SUMMARY"] == "LLM returned a list of strings as suggestions."
    )
    assert len(result["SUGGESTIONS"]) == 2
    assert result["SUGGESTIONS"][0]["AREA"] == "General"
    assert result["SUGGESTIONS"][0]["PROBLEM"] == "Suggestion 1"
    assert result["SUGGESTIONS"][0]["PROPOSED_SOLUTION"] == "N/A"
    assert result["SUGGESTIONS"][0]["EXPECTED_IMPACT"] == "N/A"
    assert result["SUGGESTIONS"][0]["CODE_CHANGES_SUGGESTED"] == []
    assert result["SUGGESTIONS"][0]["RATIONALE"] is None
    assert any(
        "TOP_LEVEL_LIST_WRAPPING" == block["type"]
        for block in result["malformed_blocks"]
    )


def test_parse_and_validate_top_level_list_of_dicts_critique_output(parser):
    """Tests handling of a top-level list of dicts for CritiqueOutput."""
    raw_output = '[{"point_summary": "P1", "details": "D1", "recommendation": "R1"}, {"point_summary": "P2", "details": "D2", "recommendation": "R2"}]'
    result = parser.parse_and_validate(raw_output, CritiqueOutput)
    assert (
        result["CRITIQUE_SUMMARY"] == "LLM returned a list of dicts as critique points."
    )
    assert len(result["CRITIQUE_POINTS"]) == 2
    assert result["CRITIQUE_POINTS"][0]["point_summary"] == "P1"
    assert any(
        "TOP_LEVEL_LIST_WRAPPING" == block["type"]
        for block in result["malformed_blocks"]
    )


def test_parse_and_validate_top_level_list_of_code_changes_llm_output(parser):
    """Tests handling of a top-level list of CodeChange objects for LLMOutput."""
    raw_output = '[{"FILE_PATH": "file1.py", "ACTION": "ADD", "FULL_CONTENT": "print(\'hello\')"}]'
    result = parser.parse_and_validate(raw_output, LLMOutput)
    assert result["COMMIT_MESSAGE"] == "LLM returned multiple code changes directly."
    assert len(result["CODE_CHANGES"]) == 1
    assert result["CODE_CHANGES"][0]["FILE_PATH"] == "file1.py"
    assert any(
        "TOP_LEVEL_LIST_WRAPPING" == block["type"]
        for block in result["malformed_blocks"]
    )


def test_parse_and_validate_top_level_list_general_output(parser):
    """Tests handling of a top-level list for GeneralOutput."""
    raw_output = '["item1", "item2"]'
    result = parser.parse_and_validate(raw_output, GeneralOutput)
    assert result["general_output"] == "item1\nitem2"
    assert any(
        "TOP_LEVEL_LIST_WRAPPING" == block["type"]
        for block in result["malformed_blocks"]
    )


def test_parse_and_validate_empty_list_general_output(parser):
    """Tests handling of an empty list for GeneralOutput."""
    raw_output = "[]"
    result = parser.parse_and_validate(raw_output, GeneralOutput)
    assert result["general_output"] == "[]"
    assert any(
        "EMPTY_JSON_LIST" == block["type"] for block in result["malformed_blocks"]
    )


def test_parse_and_validate_salvaged_fragment(parser):
    """Tests that a salvaged JSON fragment is recorded in malformed_blocks."""
    raw_output = 'Some text before. {"key": "value", "partial": "data" Some text after.'
    result = parser.parse_and_validate(raw_output, GeneralOutput)
    # FIX: The salvaged fragment is now added to malformed_blocks, and the general_output is a fallback message.
    assert (
        "No valid JSON data could be extracted or parsed." in result["general_output"]
    )
    assert any(
        b["type"] == "SALVAGED_JSON_FRAGMENT" for b in result["malformed_blocks"]
    )
    salvaged_block = next(
        b for b in result["malformed_blocks"] if b["type"] == "SALVAGED_JSON_FRAGMENT"
    )
    assert (
        '{"key": "value", "partial": "data"}' in salvaged_block["raw_string_snippet"]
    )  # Should be the force-closed version


def test_parse_and_validate_unknown_schema_model_string(parser):
    """Tests handling of an unknown schema model provided as a string."""
    raw_output = '{"general_output": "test"}'
    with pytest.raises(ValueError, match="Unknown schema model: UnknownSchema"):
        parser.parse_and_validate(raw_output, "UnknownSchema")


def test_parse_and_validate_malformed_code_change_in_list(parser):
    """Tests handling of a malformed CodeChange item within a list of suggestions."""
    raw_output = """
    {
      "ANALYSIS_SUMMARY": "Summary",
      "IMPACTFUL_SUGGESTIONS": [
        {
          "AREA": "Robustness",
          "PROBLEM": "P1",
          "PROPOSED_SOLUTION": "S1",
          "EXPECTED_IMPACT": "I1",
          "CODE_CHANGES_SUGGESTED": [
            {"FILE_PATH": "file1.py", "ACTION": "INVALID_ACTION"}
          ]
        }
      ]
    }
    """
    result = parser.parse_and_validate(raw_output, SelfImprovementAnalysisOutputV1)
    assert result["ANALYSIS_SUMMARY"] == "Summary"
    assert len(result["IMPACTFUL_SUGGESTIONS"]) == 1
    assert any(
        b["type"] == "MALFORMED_SUGGESTION_STRUCTURE"
        for b in result["malformed_blocks"]
    )
    assert "INVALID_ACTION" in result["malformed_blocks"][0]["message"]
    # The malformed suggestion should still be present, but its CODE_CHANGES_SUGGESTED might be empty or contain the problematic item depending on Pydantic's behavior.
    # For this test, we check that the malformed_blocks are correctly populated.
    assert (
        len(result["IMPACTFUL_SUGGESTIONS"][0]["CODE_CHANGES_SUGGESTED"]) == 0
    )  # FIX: Pydantic will drop invalid items
