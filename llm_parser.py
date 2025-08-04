# llm_parser.py

import json
import re
import os
import io
import contextlib
from typing import Dict, Any, Optional, List, Tuple
import jsonschema

# Define a custom exception for parsing errors
class LLMOutputParsingError(Exception):
    """Custom exception for errors during LLM output parsing."""
    pass

class InvalidSchemaError(LLMOutputParsingError):
    """Exception raised when LLM output does not conform to the expected schema."""
    pass

class PathTraversalError(LLMOutputParsingError):
    """Exception raised for detected path traversal attempts."""
    pass

# Define the expected schema for LLM output
# This schema reflects the structure expected by the Impartial_Arbitrator persona.
LLM_OUTPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "COMMIT_MESSAGE": {"type": "string"},
        "RATIONALE": {"type": "string"},
        "CODE_CHANGES": {
            "type": "object", # Changed to object (dictionary)
            "description": "A dictionary where keys are file paths and values describe the changes.",
            "patternProperties": {
                "^.*$": { # Matches any file path key
                    "type": "object",
                    "properties": {
                        "file_path": {"type": "string", "minLength": 1}, # This property is now part of the nested object
                        "action": {"type": "string", "enum": ["ADD", "MODIFY", "REMOVE"]},
                        "full_content": {"type": "string"}, # For ADD/MODIFY
                        "lines": {"type": "array", "items": {"type": "string"}} # For REMOVE
                    },
                    "required": ["file_path", "action"],
                    "oneOf": [
                        {"properties": {"action": {"const": "ADD"}, "full_content": {"type": "string"}}, "required": ["full_content"]},
                        {"properties": {"action": {"const": "MODIFY"}, "full_content": {"type": "string"}}, "required": ["full_content"]},
                        {"properties": {"action": {"const": "REMOVE"}, "lines": {"type": "array", "items": {"type": "string"}}}, "required": ["lines"]}
                    ]
                }
            },
            "additionalProperties": False # Disallow properties not matching the patternProperties
        }
    },
    "required": ["COMMIT_MESSAGE", "RATIONALE", "CODE_CHANGES"],
    "additionalProperties": False # Disallow unexpected top-level keys
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

    # Attempt to fix incorrectly escaped backslashes (e.g., \\" -> \")
    # This is tricky and might have false positives, use with caution.
    # A more robust solution would involve a proper JSON parser that can handle errors.
    # For now, we'll focus on common LLM output issues.
    # repaired_str = repaired_str.replace('\\\\"', '\\"') # Example: Fix \\" to \"
    # repaired_str = repaired_str.replace('\\\\n', '\\n') # Example: Fix \\n to \n

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


@contextlib.contextmanager
def _handle_parsing_errors(raw_output: str, context: str = "parsing"):
    """Context manager to handle JSON decoding and schema validation errors."""
    try:
        yield raw_output # Yield the raw output for processing within the 'with' block
    except json.JSONDecodeError as e:
        # Attempt heuristic repair
        repaired_output = _repair_json_string(raw_output)
        try:
            # Validate repair by attempting to load it
            json.loads(repaired_output)
            yield repaired_output # Return repaired string if successful
        except json.JSONDecodeError as repair_e:
            raise LLMOutputParsingError(
                f"Failed to parse LLM output after heuristic repair. "
                f"Original error: {e}. Repair error: {repair_e}. "
                f"Raw output snippet: '{raw_output[:200]}...'.") from repair_e
        except Exception as repair_e: # Catch other potential errors during repair validation
             raise LLMOutputParsingError(
                f"Unexpected error during heuristic repair validation. "
                f"Original error: {e}. Repair error: {repair_e}. "
                f"Raw output snippet: '{raw_output[:200]}...'.") from repair_e

    except ValueError as e: # Handles non-dict JSONs
         raise LLMOutputParsingError(f"LLM output is not a JSON object: {e}") from e
    except jsonschema.ValidationError as e:
        # Provide more detailed error information from jsonschema
        error_details = f"Message: {e.message}\nPath: {list(e.path)}\nSchema Path: {list(e.schema_path)}\nValidator: {e.validator} ({e.validator_value})"
        raise InvalidSchemaError(f"LLM output schema validation failed: {error_details}") from e
    except PathTraversalError as e:
        raise
    except Exception as e: # Catch any other unexpected errors during initial parsing
        raise LLMOutputParsingError(f"An unexpected error occurred during {context}: {e}") from e

def parse_llm_code_output(llm_output: str, base_dir: str = ".") -> Dict[str, Any]:
    """Parses the structured JSON output from the LLM into a dictionary.

    Args:
        llm_output: The raw string output from the LLM.
        base_dir: The base directory for validating file paths.

    Returns:
        A dictionary containing parsed commit information.

    Raises:
        LLMOutputParsingError: If parsing fails even after heuristic repair.
        InvalidSchemaError: If the parsed JSON does not conform to the expected schema.
        PathTraversalError: If a file path attempts to traverse directories.
    """
    output = {
        'summary': {'commit_message': '', 'rationale': '', 'conflict_resolution': '', 'unresolved_conflict': ''},
        'changes': {}, # Changed to dictionary keyed by file_path
        'malformed_blocks': [], # Keep for non-critical warnings
    }

    llm_output_stripped = llm_output.strip()
    llm_output_cleaned = ""

    # Try to extract content from ```json ... ``` block
    json_block_match = re.search(r'```json\s*(.*?)\s*```', llm_output_stripped, re.DOTALL)
    if json_block_match:
        llm_output_cleaned = json_block_match.group(1).strip()
    else:
        # Try to extract content from generic ``` ... ``` block
        generic_block_match = re.search(r'```\s*(.*?)\s*```', llm_output_stripped, re.DOTALL)
        if generic_block_match:
            llm_output_cleaned = generic_block_match.group(1).strip()
        else:
            # Fallback: find the first '{' and last '}'
            first_brace = llm_output_stripped.find('{')
            last_brace = llm_output_stripped.rfind('}')
            
            if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
                llm_output_cleaned = llm_output_stripped[first_brace : last_brace + 1]
                output['malformed_blocks'].append(
                    f"Warning: No markdown code block fences found. Attempting to parse content between first '{{' and last '}}'."
                )
            else:
                llm_output_cleaned = llm_output_stripped
                output['malformed_blocks'].append(
                    f"No valid JSON structure (braces or markdown fences) detected. Attempting to parse raw output."
                )

    if not llm_output_cleaned:
        output['malformed_blocks'].append(
            f"LLM output was empty or contained only whitespace after stripping."
        )
        # Return early if no content to parse, but log the warning
        return output

    try:
        # Use the context manager to handle parsing and repair attempts
        with _handle_parsing_errors(llm_output_cleaned) as processed_output:
            # processed_output is either the original cleaned output or the repaired version
            parsed_data = json.loads(processed_output)

        # Validate against the schema
        jsonschema.validate(instance=parsed_data, schema=LLM_OUTPUT_SCHEMA)

        # Extract and sanitize data
        output['summary']['commit_message'] = parsed_data.get('COMMIT_MESSAGE', '').strip()
        rationale_content = parsed_data.get('RATIONALE', '').strip()
        output['summary']['rationale'] = rationale_content

        # Extract conflict resolution and unresolved conflict from rationale
        conflict_res_match = re.search(r"CONFLICT RESOLUTION:\s*(.*?)(?=\nUNRESOLVED CONFLICT:|\n\n|$)", rationale_content, re.DOTALL)
        if conflict_res_match:
            output['summary']['conflict_resolution'] = conflict_res_match.group(1).strip()
        
        unresolved_conflict_match = re.search(r"UNRESOLVED CONFLICT:\s*(.*?)(?=\n\n|$)", rationale_content, re.DOTALL)
        if unresolved_conflict_match:
            output['summary']['unresolved_conflict'] = unresolved_conflict_match.group(1).strip()

        code_changes_dict_from_llm = parsed_data.get('CODE_CHANGES', {})
        
        # Sanitize file paths within code changes and structure for app.py
        sanitized_changes_dict = {}
        for file_path_key, change_item in code_changes_dict_from_llm.items():
            try:
                # The schema expects keys to be file paths, but the LLM might not always provide them correctly.
                # We'll use the 'file_path' property within the change_item for robustness.
                file_path = change_item.get('file_path')
                if not file_path:
                    raise KeyError("Missing 'file_path' property within a CODE_CHANGES item.")
                
                action = change_item.get('action')
                if not action:
                    raise KeyError(f"Missing 'action' property for file '{file_path}'.")
                
                sanitized_path = _sanitize_file_path(file_path, base_dir)
                
                change_data = {
                    'file_path': sanitized_path,
                    'action': action,
                    'type': action # Add 'type' key for app.py display
                }

                if action in ['ADD', 'MODIFY']:
                    # LLM output should provide 'full_content' as per persona prompt
                    if 'full_content' not in change_item:
                        raise KeyError(f"'full_content' missing for {action} action in file '{file_path}'.")
                    change_data['content'] = change_item['full_content'] # For app.py display
                    change_data['new_content'] = change_item['full_content'] # For diff generation in app.py
                elif action == 'REMOVE':
                    if 'lines' not in change_item or not isinstance(change_item['lines'], list):
                        raise KeyError(f"'lines' missing or not a list for REMOVE action in file '{file_path}'.")
                    change_data['lines'] = change_item['lines']
                else:
                    # Should not happen due to enum in schema, but good for robustness
                    output['malformed_blocks'].append(f"Unknown action type '{action}' encountered for file '{file_path}'.")
                
                # Add to the dictionary, keyed by the sanitized file path
                sanitized_changes_dict[sanitized_path] = change_data

            except KeyError as ke:
                 output['malformed_blocks'].append(f"Error processing CODE_CHANGES item: {ke}")
            except PathTraversalError as pte:
                # Log path traversal attempts as warnings, but don't stop processing
                output['malformed_blocks'].append(f"Path Traversal Warning: {pte}")
            except Exception as e:
                 output['malformed_blocks'].append(f"Error processing change item for file '{file_path_key}': {e}")

        output['changes'] = sanitized_changes_dict # Assign the dictionary

    except LLMOutputParsingError as e:
        # Provide more context for parsing errors
        output['malformed_blocks'].append(f"Critical Parsing Error: {e}. "
                                          f"Raw output snippet: '{llm_output_cleaned[:200]}...'")
    except InvalidSchemaError as e:
        # Provide more detailed error information from jsonschema
        output['malformed_blocks'].append(f"Schema Validation Error: {e}")
    except PathTraversalError as e:
         output['malformed_blocks'].append(f"Path Traversal Error: {e}")
    except Exception as e: # Catch-all for unexpected errors during the process
        # Provide more context for unexpected errors
        output['malformed_blocks'].append(f"Unexpected error during parsing: {e}. "
                                          f"Raw output snippet: '{llm_output_cleaned[:200]}...'")

    return output