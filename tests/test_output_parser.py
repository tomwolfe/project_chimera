import pytest
from src.utils.output_parser import parse_llm_output

def test_parse_llm_output_valid_json():
    """Tests parsing of a valid JSON string."""
    llm_response = '{"key": "value", "number": 123}'
    expected_output = {"key": "value", "number": 123}
    parsed_output = parse_llm_output(llm_response)
    assert parsed_output == expected_output

def test_parse_llm_output_invalid_json():
    """Tests handling of invalid JSON strings."""
    llm_response = 'This is not JSON'
    # Expecting an error or a specific return value for invalid input
    with pytest.raises(ValueError, match="Could not parse LLM output as JSON"):
        parse_llm_output(llm_response)

def test_parse_llm_output_empty_string():
    """Tests handling of an empty string input."""
    llm_response = ''
    with pytest.raises(ValueError, match="Could not parse LLM output as JSON"):
        parse_llm_output(llm_response)

def test_parse_llm_output_with_extra_text():
    """Tests parsing when JSON is embedded within other text."""
    llm_response = 'Here is the JSON: {\"status\": \"success\"} End of response.'
    expected_output = {"status": "success"}
    parsed_output = parse_llm_output(llm_response)
    assert parsed_output == expected_output

def test_parse_llm_output_list_json():
    """Tests parsing of a JSON array."""
    llm_response = '[{"item": 1}, {"item": 2}]'
    expected_output = [{"item": 1}, {"item": 2}]
    parsed_output = parse_llm_output(llm_response)
    assert parsed_output == expected_output

# Add more tests for different edge cases and expected output formats if applicable.
