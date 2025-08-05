# llm_output_validator.py

import json
import logging
from jsonschema import validate, ValidationError
import re # Added import for regex operations
import os # Added import for path operations
import contextlib # Added import for context manager

# Define the expected JSON schema for the Impartial_Arbitrator's output
LLM_OUTPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "COMMIT_MESSAGE": {"type": "string", "minLength": 1},
        "RATIONALE": {"type": "string", "minLength": 1},
        "CODE_CHANGES": {
            "type": "array",
            "description": "A list of code changes proposed.",
            "items": {
                "type": "object",
                "properties": {
                    "file_path": {"type": "string", "minLength": 1},
                    "action": {"type": "string", "enum": ["ADD", "MODIFY", "REMOVE"]},
                    "full_content": {"type": "string"},
                    "lines": {"type": "array", "items": {"type": "string"}}
                },
                "required": ["file_path", "action"],
                "oneOf": [
                    {
                        "properties": {"action": {"const": "ADD"}, "full_content": {"type": "string"}},
                        "required": ["full_content"],
                        "additionalProperties": False
                    },
                    {
                        "properties": {"action": {"const": "MODIFY"}, "full_content": {"type": "string"}},
                        "required": ["full_content"],
                        "additionalProperties": False
                    },
                    {
                        "properties": {"action": {"const": "REMOVE"}, "lines": {"type": "array", "items": {"type": "string"}}},
                        "required": ["lines"],
                        "additionalProperties": False
                    }
                ],
                "additionalProperties": False
            }
        }
    },
    "required": ["COMMIT_MESSAGE", "RATIONALE", "CODE_CHANGES"],
    "additionalProperties": False
}

def _repair_json_string(json_str: str) -> str:
    """
    Attempts to repair common LLM-induced JSON errors.
    NOTE: This is a heuristic fallback and should be used cautiously.
    Focus should be on improving LLM output generation.
    """
    repaired_str = json_str

    # Fix missing commas between JSON objects (e.g., } { -> }, {)
    repaired_str = re.sub(r'}\s*{', '}, {', repaired_str)
    # Fix missing commas after array elements (e.g., "item1" "item2" -> "item1", "item2")
    repaired_str = re.sub(r'"\s*"', '", "', repaired_str)
    # Fix missing commas after numbers/booleans in arrays
    repaired_str = re.sub(r'(\d|true|false)\s*(?=[\"{\\[\\]])', r'\1, ', repaired_str)
    
    # Fix unquoted keys (e.g., {key: "value"} -> {"key": "value"})
    # This regex looks for a word character sequence followed by a colon, not preceded by a quote.
    repaired_str = re.sub(r'([{,]\s*)([a-zA-Z_][a-zA-Z0-9_]*)\s*:', r'\1"\2":', repaired_str)
    
    # Fix trailing commas in objects and arrays (e.g., {"key": "value",} -> {"key": "value"})
    repaired_str = re.sub(r',\s*([\]}])', r'\1', repaired_str)

    return repaired_str

def _sanitize_file_path(file_path: str, base_dir: str = ".") -> str:
    """
    Sanitizes a file path to prevent traversal attacks.
    Ensures the path remains within the specified base directory.
    """
    if not file_path:
        raise PathTraversalError("File path cannot be empty.")

    # Normalize path to resolve '..', '.', etc.
    normalized_path = os.path.normpath(os.path.join(base_dir, file_path))

    # Check if the normalized path is still within the base directory
    if not normalized_path.startswith(os.path.abspath(base_dir)):
        raise PathTraversalError(
            f"File path traversal detected: '{file_path}' resolved to "
            f"'{normalized_path}', which is outside the base directory."
        )
    return normalized_path

class PathTraversalError(Exception): # Define PathTraversalError if not already defined elsewhere
    """Exception raised for detected path traversal attempts."""
    pass

class LLMOutputParsingError(Exception): # Define LLMOutputParsingError if not already defined elsewhere
    """Custom exception for errors during LLM output parsing."""
    pass

class InvalidSchemaError(LLMOutputParsingError): # Define InvalidSchemaError if not already defined elsewhere
    """Exception raised when LLM output does not conform to the expected schema."""
    pass


def validate_llm_output(raw_output: str, base_dir: str = ".") -> dict:
    """
    Parses the raw LLM output string and validates it against the expected schema.
    Returns the validated dictionary if successful, raises ValidationError otherwise.
    Includes basic security checks for file paths.
    """
    logging.info("Starting LLM output validation.")
    
    # Attempt to extract JSON from markdown code blocks or fallback
    llm_output_cleaned = ""
    json_block_match = re.search(r'```json\s*(.*?)\s*```', raw_output, re.DOTALL)
    if json_block_match:
        llm_output_cleaned = json_block_match.group(1).strip()
        logging.debug("Extracted JSON from markdown block.")
    else:
        generic_block_match = re.search(r'```\s*(.*?)\s*```', raw_output, re.DOTALL)
        if generic_block_match:
            llm_output_cleaned = generic_block_match.group(1).strip()
            logging.warning("No JSON markdown block found, attempting to parse generic code block.")
        else:
            first_brace = raw_output.find('{')
            last_brace = raw_output.rfind('}')
            if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
                llm_output_cleaned = raw_output[first_brace : last_brace + 1]
                logging.warning("No markdown fences found, attempting to parse content between first '{' and last '}'.")
            else:
                llm_output_cleaned = raw_output.strip()
                logging.warning("No JSON structure detected, attempting to parse raw output.")

    if not llm_output_cleaned:
        logging.error("LLM output is empty after cleaning.")
        raise LLMOutputParsingError("LLM output was empty or contained only whitespace.")

    try:
        # Attempt to parse JSON, with heuristic repair for common LLM errors
        repaired_output = _repair_json_string(llm_output_cleaned)
        try:
            data = json.loads(repaired_output)
            logging.debug("Successfully parsed LLM output (potentially after repair).")
        except json.JSONDecodeError as e:
            logging.error(f"Failed to decode LLM JSON output even after repair: {e}")
            raise LLMOutputParsingError(f"Failed to parse LLM output after heuristic repair. Error: {e}") from e

        # Validate against the formal JSON schema
        validate(instance=data, schema=LLM_OUTPUT_SCHEMA)
        logging.info("LLM output schema validation passed.")

        # Perform additional security and content-specific validations
        for i, change in enumerate(data.get("CODE_CHANGES", [])):
            file_path = change.get("file_path", "")
            action = change.get("action", "")

            # Security Check: Path Traversal Prevention
            try:
                sanitized_path = _sanitize_file_path(file_path, base_dir)
                # Update the path in the data structure if it was modified by sanitization
                if sanitized_path != file_path:
                    change["file_path"] = sanitized_path
                    logging.debug(f"Sanitized file path: '{file_path}' -> '{sanitized_path}'")
            except PathTraversalError as pte:
                logging.error(f"LLM Output Validation Error: {pte}")
                raise pte # Re-raise to be caught by the main handler

            # Security Check: Content Sanitization (basic placeholder)
            if action in ["ADD", "MODIFY"]:
                content = change.get("full_content", "")
                if not content: # Should be caught by schema, but double-check
                    raise ValidationError(f"Missing 'full_content' for {action} action on file '{file_path}'.")
                # Placeholder for content sanitization logic
                # sanitized_content = sanitize_code(content)
                # if not is_safe(sanitized_content):
                #    raise ValidationError(f"Unsafe content detected in '{file_path}'.")

        logging.info("LLM output successfully validated against schema and security checks.")
        return data

    except json.JSONDecodeError as e:
        logging.error(f"LLM Output Validation Error: Invalid JSON format - {e}")
        raise LLMOutputParsingError(f"Invalid JSON format: {e}") from e
    except ValidationError as e:
        logging.error(f"LLM Output Validation Error: Schema validation failed - {e.message}")
        # Provide more detailed error information from jsonschema
        error_details = f"Message: {e.message}\nPath: {list(e.path)}\nSchema Path: {list(e.schema_path)}\nValidator: {e.validator} ({e.validator_value})"
        raise InvalidSchemaError(f"Schema validation failed: {error_details}") from e
    except PathTraversalError as e:
        logging.error(f"LLM Output Validation Error: Path Traversal detected - {e}")
        raise e # Re-raise the specific error
    except Exception as e:
        # Catch any other unexpected errors during validation
        logging.error(f"LLM Output Validation Error: An unexpected error occurred - {e}")
        raise LLMOutputParsingError(f"Unexpected validation error: {e}") from e

# --- Placeholder Security Functions (to be implemented robustly) ---
def sanitize_code(content: str) -> str:
    # TODO: Implement robust code sanitization (e.g., AST-based checks for malicious patterns)
    logging.debug("Sanitizing code content (placeholder)...")
    return content

def is_safe(content: str) -> bool:
    # TODO: Implement robust safety checks (e.g., disallow specific imports/functions)
    logging.debug("Performing safety check on content (placeholder)...")
    return True
