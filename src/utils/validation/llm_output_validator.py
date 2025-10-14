"""Shared validation utilities for LLM output parsing and validation."""

import json
import logging
import re
from typing import Any, Dict, List, Optional, Tuple, Type

from pydantic import BaseModel, ValidationError

logger = logging.getLogger(__name__)


class LLMOutputValidationError(Exception):
    """Custom exception for LLM output validation errors."""

    pass


def validate_and_extract_json(text: str) -> Optional[Dict[str, Any]]:
    """
    Extract and validate JSON from text with multiple extraction strategies.

    Args:
        text: Raw text that may contain JSON

    Returns:
        Parsed JSON dictionary if successful, None otherwise
    """
    # Try direct JSON parsing first
    try:
        return json.loads(text.strip())
    except json.JSONDecodeError:
        pass

    # Try extracting from markdown code blocks
    json_from_markdown = _extract_json_from_markdown(text)
    if json_from_markdown:
        return json_from_markdown

    # Try extracting first outermost JSON object
    json_from_extraction = _extract_first_outermost_json(text)
    if json_from_extraction:
        return json_from_extraction

    # Try extracting from XML-like tags
    json_from_xml = _extract_from_xml_tags(text, "json_output")
    if json_from_xml:
        return json_from_xml

    # Try extracting with JSON delimiters
    marker_result = _extract_json_with_markers(text)
    if marker_result:
        json_str, _ = marker_result
        if json_str:
            return json.loads(json_str)

    return None


def _extract_json_from_markdown(text: str) -> Optional[Dict[str, Any]]:
    """Extract JSON content from markdown code blocks."""
    # Pattern to match any code block, optionally with a language specifier
    markdown_block_pattern = r"```(?:json|python|text|yaml|yml)?\s*(.*?)```"

    matches = list(re.finditer(markdown_block_pattern, text, re.DOTALL | re.MULTILINE))

    for match in matches:
        block_content = match.group(1).strip()
        if block_content:
            # Try the robust JSON extraction on this block content
            extracted_json = _extract_first_outermost_json(block_content)
            if extracted_json:
                try:
                    return json.loads(extracted_json)
                except json.JSONDecodeError:
                    continue

    return None


def _extract_first_outermost_json(text: str) -> Optional[str]:
    """Extract the first outermost balanced JSON object or array from text."""
    balance = 0
    expected_closers_stack = []
    start_index = -1

    for i, char in enumerate(text):
        if char == "{":
            if start_index == -1:  # First opening brace
                start_index = i
                balance = 1
                expected_closers_stack.append("}")
            else:
                balance += 1
                expected_closers_stack.append("}")
        elif char == "[":
            if start_index == -1:  # First opening bracket
                start_index = i
                balance = 1
                expected_closers_stack.append("]")
            else:
                balance += 1
                expected_closers_stack.append("]")
        elif char == "}":
            if start_index != -1:
                balance -= 1
                if expected_closers_stack and expected_closers_stack[-1] == "}":
                    expected_closers_stack.pop()
        elif char == "]" and start_index != -1:
            balance -= 1
            if expected_closers_stack and expected_closers_stack[-1] == "]":
                expected_closers_stack.pop()

        if start_index != -1 and balance == 0 and not expected_closers_stack:
            potential_json_str = text[start_index : i + 1]
            try:
                json.loads(potential_json_str)
                return potential_json_str.strip()
            except json.JSONDecodeError:
                # If it's not valid JSON despite being balanced, reset and continue
                start_index = -1
                balance = 0
                expected_closers_stack = []
                continue

    return None


def _extract_from_xml_tags(text: str, tag: str) -> Optional[str]:
    """Extract content from within specific XML-like tags."""
    start_tag = f"<{tag}>"
    end_tag = f"</{tag}>"
    start_match = text.find(start_tag)
    if start_match == -1:
        return None

    end_match = text.find(end_tag, start_match + len(start_tag))
    if end_match == -1:
        # If end tag is missing, try to extract up to the end of the string
        return text[start_match + len(start_tag) :].strip()

    return text[start_match + len(start_tag) : end_match].strip()


def _extract_json_with_markers(
    text: str,
    start_marker: str = "START_JSON_OUTPUT",
    end_marker: str = "END_JSON_OUTPUT",
) -> Optional[Tuple[str, bool]]:
    """Extracts JSON content explicitly delimited by start and end markers."""
    start_match = re.search(re.escape(start_marker), text)
    if not start_match:
        return None

    text_after_start_marker = text[start_match.end() :]
    end_match = re.search(re.escape(end_marker), text_after_start_marker)

    json_content_raw = ""
    end_marker_found = False

    if end_match:
        json_content_raw = text_after_start_marker[: end_match.start()].strip()
        end_marker_found = True
    else:
        # Try to extract what looks like JSON from remaining content
        json_content_raw = _extract_first_outermost_json(text_after_start_marker) or ""

    if json_content_raw:
        # Basic sanitization for trailing commas before closing braces/brackets
        json_content_raw = re.sub(r",\s*([\}\]])", r"\1", json_content_raw)
        return json_content_raw, end_marker_found

    return None


def clean_llm_output(raw_output: str) -> str:
    """Clean common LLM output artifacts like markdown fences and conversational filler."""
    cleaned = raw_output

    # First attempt to extract from specific XML-like tags if present
    json_from_xml_tags = _extract_from_xml_tags(cleaned, "json_output")
    if json_from_xml_tags:
        return json_from_xml_tags

    # Remove markdown code blocks
    cleaned = re.sub(
        r"^\s*```(?:json|python|text|yaml|yml)?\s*\n(.*?)\n\s*```\s*$",
        r"\1",
        cleaned,
        flags=re.DOTALL | re.MULTILINE | re.IGNORECASE,
    )

    # Remove common conversational prefixes
    cleaned = re.sub(
        r"^\s*(?:Here is the JSON output|Here's the|As a [^:]+?:|```(?:json|python|text|yaml|yml)?|```).*?\n",
        "",
        cleaned,
        flags=re.DOTALL | re.IGNORECASE,
    )
    cleaned = re.sub(
        r"\n(?:```(?:json|python|text|yaml|yml)?|```).*?\Z",
        "",
        cleaned,
        flags=re.DOTALL | re.MULTILINE | re.IGNORECASE,
    )

    # Extract the first JSON-like object/array if it exists
    json_start = -1
    json_end = -1

    for i, char in enumerate(cleaned):
        if char in {"{", "["}:
            json_start = i
            break

    for i in range(len(cleaned) - 1, -1, -1):
        if cleaned[i] == "}" or cleaned[i] == "]":
            json_end = i + 1
            break

    if json_start != -1 and json_end != -1 and json_end > json_start:
        cleaned = cleaned[json_start:json_end]

    return cleaned.strip()


def validate_output_against_schema(
    data: Dict[str, Any], schema_model: Type[BaseModel]
) -> Dict[str, Any]:
    """
    Validate extracted data against a Pydantic schema with error handling.

    Args:
        data: Dictionary of data to validate
        schema_model: Pydantic model class to validate against

    Returns:
        Validated data as dictionary
    """
    try:
        validated_output = schema_model.model_validate(data)
        return validated_output.model_dump(by_alias=True)
    except ValidationError as ve:
        # Attempt schema correction for common issues
        corrected_data = attempt_schema_correction(data, ve, schema_model)
        if corrected_data:
            try:
                validated_output = schema_model.model_validate(corrected_data)
                return validated_output.model_dump(by_alias=True)
            except ValidationError as second_ve:
                logger.warning(f"Schema correction failed: {second_ve}")
                raise LLMOutputValidationError(
                    f"Schema validation failed after correction: {second_ve}"
                ) from second_ve
        else:
            logger.warning(f"Schema validation failed: {ve}")
            raise LLMOutputValidationError(f"Schema validation failed: {ve}") from ve


def attempt_schema_correction(
    output: Dict[str, Any], error: ValidationError, schema_model: Type[BaseModel]
) -> Optional[Dict[str, Any]]:
    """
    Attempt automatic correction of common schema validation failures by adding default values.

    Args:
        output: The output dict that failed validation
        error: The ValidationError that occurred
        schema_model: The schema model that was used

    Returns:
        Corrected output dict if corrections were made, None otherwise
    """

    corrected_output = output.copy()
    correction_made = False

    for err in error.errors():
        field_path = err["loc"]
        error_msg = err["msg"]

        if "missing required field" in error_msg.lower():
            # For SelfImprovementAnalysisOutputV1
            if field_path == ("ANALYSIS_SUMMARY",):
                corrected_output["ANALYSIS_SUMMARY"] = (
                    "Analysis summary not provided by LLM, auto-filled."
                )
                correction_made = True
            elif field_path == ("IMPACTFUL_SUGGESTIONS",):
                corrected_output["IMPACTFUL_SUGGESTIONS"] = []
                correction_made = True
            # For CritiqueOutput
            elif field_path == ("CRITIQUE_SUMMARY",):
                corrected_output["CRITIQUE_SUMMARY"] = (
                    "Critique summary not provided by LLM, auto-filled."
                )
                correction_made = True
            elif field_path == ("CRITIQUE_POINTS",):
                corrected_output["CRITIQUE_POINTS"] = []
                correction_made = True
            elif field_path == ("SUGGESTIONS",):
                corrected_output["SUGGESTIONS"] = []
                correction_made = True
            # For LLMOutput
            elif field_path == ("COMMIT_MESSAGE",):
                corrected_output["COMMIT_MESSAGE"] = (
                    "Commit message not provided by LLM, auto-filled."
                )
                correction_made = True
            elif field_path == ("RATIONALE",):
                corrected_output["RATIONALE"] = (
                    "Rationale not provided by LLM, auto-filled."
                )
                correction_made = True
            elif field_path == ("CODE_CHANGES",):
                corrected_output["CODE_CHANGES"] = []
                correction_made = True
            # For GeneralOutput
            elif field_path == ("general_output",):
                corrected_output["general_output"] = (
                    "General output not provided by LLM, auto-filled."
                )
                correction_made = True
            # For ConflictReport
            elif field_path == ("conflict_type",):
                corrected_output["conflict_type"] = "NO_CONFLICT"
                correction_made = True
            elif field_path == ("summary",):
                corrected_output["summary"] = (
                    "Conflict summary not provided by LLM, auto-filled."
                )
                correction_made = True
            elif field_path == ("involved_personas",):
                corrected_output["involved_personas"] = []
                correction_made = True
            elif field_path == ("conflicting_outputs_snippet",):
                corrected_output["conflicting_outputs_snippet"] = ""
                correction_made = True
            elif field_path == ("conflict_found",):
                corrected_output["conflict_found"] = False
                correction_made = True
            # For SuggestionItem (new fields)
            elif field_path == ("PARETO_SCORE",):
                corrected_output["PARETO_SCORE"] = 0.0
                correction_made = True
            elif field_path == ("VALIDATION_METHOD",):
                corrected_output["VALIDATION_METHOD"] = "N/A"
                correction_made = True

    if correction_made:
        from logging import getLogger

        logger = getLogger(__name__)
        logger.info(f"Attempted schema correction: {error_msg}. Corrected output.")
        return corrected_output
    return None


def validate_code_change_structure(data: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """
    Validate the specific structure of code change data.

    Args:
        data: Code change data to validate

    Returns:
        Tuple of (is_valid, list_of_errors)
    """
    errors = []

    required_fields = ["FILE_PATH", "ACTION"]
    for field in required_fields:
        if field not in data:
            errors.append(f"Missing required field: {field}")

    if "FILE_PATH" in data:
        file_path = data["FILE_PATH"]
        if not isinstance(file_path, str) or not file_path.strip():
            errors.append("FILE_PATH must be a non-empty string")

        # Basic security check to prevent directory traversal
        if isinstance(file_path, str) and (
            ".." in file_path or file_path.startswith("/") or ":/" in file_path
        ):
            errors.append(
                f"FILE_PATH contains invalid characters preventing directory traversal: {file_path}"
            )

    if "ACTION" in data:
        action = data["ACTION"]
        valid_actions = ["ADD", "MODIFY", "REMOVE", "CREATE", "CREATE_DIRECTORY"]
        if action not in valid_actions:
            errors.append(
                f"Invalid ACTION: '{action}'. Must be one of {valid_actions}."
            )

    # Validate content based on action type
    if "ACTION" in data and data["ACTION"] in ["ADD", "CREATE", "CREATE_DIRECTORY"]:
        has_content = "FULL_CONTENT" in data and data["FULL_CONTENT"] is not None
        has_lines = "LINES" in data and data["LINES"] is not None
        has_diff = "DIFF_CONTENT" in data and data["DIFF_CONTENT"] is not None

        if not (has_content or has_lines):
            errors.append(
                f"FULL_CONTENT or LINES is required for action '{data['ACTION']}'"
            )

        if has_diff:
            errors.append(
                f"DIFF_CONTENT should not be provided for action '{data['ACTION']}'"
            )

    elif "ACTION" in data and data["ACTION"] == "MODIFY":
        has_full_content = "FULL_CONTENT" in data and data["FULL_CONTENT"] is not None
        has_diff_content = "DIFF_CONTENT" in data and data["DIFF_CONTENT"] is not None

        if not (has_full_content or has_diff_content):
            errors.append(
                "Either FULL_CONTENT or DIFF_CONTENT is required for action 'MODIFY'"
            )

    elif "ACTION" in data and data["ACTION"] == "REMOVE":
        has_lines = "LINES" in data and data["LINES"] is not None
        has_full_content = "FULL_CONTENT" in data and data["FULL_CONTENT"] is not None
        has_diff_content = "DIFF_CONTENT" in data and data["DIFF_CONTENT"] is not None

        if (
            not has_lines
            or not isinstance(data["LINES"], list)
            or len(data["LINES"]) == 0
        ):
            errors.append("LINES must be a non-empty list for action 'REMOVE'")
        elif has_full_content or has_diff_content:
            errors.append(
                "FULL_CONTENT or DIFF_CONTENT should not be provided for action 'REMOVE'"
            )

    return len(errors) == 0, errors


def repair_json_string(json_str: str) -> Tuple[str, List[Dict[str, str]]]:
    """
    Apply common JSON repair heuristics to fix common LLM output issues.

    Args:
        json_str: JSON string to repair

    Returns:
        Tuple of (repaired_json_string, repair_log)
    """
    repair_log = []
    original_str = json_str

    # Heuristic 1: Remove trailing commas
    temp_str = re.sub(r",\s*([\}\]])", r"\1", json_str)
    if temp_str != json_str:
        repair_log.append(
            {"action": "remove_trailing_commas", "details": "Removed trailing commas."}
        )
        json_str = temp_str

    # Heuristic 2: Add missing closing braces/brackets
    open_braces = json_str.count("{")
    close_braces = json_str.count("}")
    if open_braces > close_braces:
        json_str += "}" * (open_braces - close_braces)
        repair_log.append(
            {
                "action": "add_closing_braces",
                "details": f"Added {open_braces - close_braces} missing closing braces.",
            }
        )

    open_brackets = json_str.count("[")
    close_brackets = json_str.count("]")
    if open_brackets > close_brackets:
        json_str += "]" * (open_brackets - close_brackets)
        repair_log.append(
            {
                "action": "add_closing_brackets",
                "details": f"Added {open_brackets - close_brackets} missing closing brackets.",
            }
        )

    # Heuristic 3: Replace single quotes with double quotes
    temp_str = re.sub(r"(?<!\\)'", '"', json_str)
    if temp_str != json_str:
        repair_log.append(
            {
                "action": "quote_replacement",
                "details": "Replaced single quotes with double quotes.",
            }
        )
        json_str = temp_str

    # Heuristic 4: Handle unquoted keys (simple heuristic)
    temp_str = re.sub(r"([{,]\s*)([a-zA-Z_][a-zA-Z0-9_]*)\s*:", r'\1"\2":', json_str)
    if temp_str != json_str:
        repair_log.append(
            {"action": "unquoted_key_fix", "details": "Added quotes to unquoted keys."}
        )
        json_str = temp_str

    # Heuristic 5: Handle unescaped newlines within string values
    temp_str = re.sub(r"(?<!\\)\n", r"\\n", json_str)
    if temp_str != json_str:
        repair_log.append(
            {
                "action": "newline_escaping",
                "details": "Escaped unescaped newlines within strings.",
            }
        )
        json_str = temp_str

    # Heuristic 6: Remove invalid control characters
    temp_str = re.sub(r"[\x00-\x1F\x7F]", "", json_str)
    if temp_str != json_str:
        repair_log.append(
            {
                "action": "control_char_removal",
                "details": "Removed invalid control characters.",
            }
        )
        json_str = temp_str

    if original_str != json_str:
        logger.info(
            f"Repaired JSON string. Repairs: {', '.join([r['details'] for r in repair_log])}"
        )

    return json_str, repair_log


def force_close_truncated_json(json_str: str) -> str:
    """Attempt to heuristically close a truncated JSON string."""
    cleaned_str = json_str.strip()
    if not cleaned_str:
        return ""

    if not (cleaned_str.startswith("{") or cleaned_str.startswith("[")):
        return cleaned_str

    open_braces = cleaned_str.count("{")
    close_braces = cleaned_str.count("}")
    open_brackets = cleaned_str.count("[")
    close_brackets = cleaned_str.count("]")

    if open_braces > close_braces and not cleaned_str.endswith("}"):
        logger.debug(f"Force-closing with {open_braces - close_braces} braces.")
        cleaned_str += "}" * (open_braces - close_braces)
    if open_brackets > close_brackets and not cleaned_str.endswith("]"):
        logger.debug(f"Force-closing with {open_brackets - close_brackets} brackets.")
        cleaned_str += "]" * (open_brackets - close_brackets)

    return cleaned_str
