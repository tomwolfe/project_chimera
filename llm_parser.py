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
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "file_path": {"type": "string", "minLength": 1},
                    "action": {"type": "string", "enum": ["ADD", "MODIFY", "DELETE"]},
                    # 'content' is used here as a general key, but the parser will map it
                    # to 'full_content' or 'lines' based on the 'action'.
                    "content": {"type": "string"} 
                },
                "required": ["file_path", "action", "content"]
            }
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

    # Basic fixes for Python keywords potentially concatenated
    keywords = ['import', 'def', 'class', 'from', 'return', 'raise', 'assert']
    for keyword in keywords:
        # Add space after keyword if followed immediately by an identifier
        repaired_str = re.sub(rf'{keyword}([a-zA-Z_])', rf'{keyword} \\1', repaired_str)

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
                f"Raw output snippet: '{raw_output[:100]}...'.") from repair_e
        except Exception as repair_e: # Catch other potential errors during repair validation
             raise LLMOutputParsingError(
                f"Unexpected error during heuristic repair validation. "
                f"Original error: {e}. Repair error: {repair_e}. "
                f"Raw output snippet: '{raw_output[:100]}...'.") from repair_e

    except ValueError as e: # Handles non-dict JSONs
         raise LLMOutputParsingError(f"LLM output is not a JSON object: {e}") from e
    except jsonschema.ValidationError as e:
        raise InvalidSchemaError(f"LLM output schema validation failed: {e.message}") from e
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
        'changes': [], # Changed to list to match app.py's expectation
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

        code_changes_list = parsed_data.get('CODE_CHANGES', [])
        
        # Sanitize file paths within code changes
        sanitized_changes = []
        for change_item in code_changes_list:
            try:
                # The schema currently expects 'content', but the arbitrator prompt specifies 'full_content' or 'lines'.
                # We need to map these correctly.
                file_path = change_item['file_path']
                action = change_item['action']
                
                # Map 'content' from schema to 'full_content' or 'lines' based on action
                if action in ['ADD', 'MODIFY']:
                    # The arbitrator prompt specifies 'full_content' for ADD/MODIFY
                    content_key = 'full_content' if 'full_content' in change_item else 'content' # Fallback to 'content' if 'full_content' is missing
                    if content_key not in change_item:
                        raise KeyError(f"'{content_key}' or 'content' missing for {action} action.")
                    
                    sanitized_path = _sanitize_file_path(file_path, base_dir)
                    sanitized_changes.append({
                        'file_path': sanitized_path,
                        'action': action,
                        'content': change_item[content_key] # Use the mapped content key
                    })
                elif action == 'REMOVE':
                    # The arbitrator prompt specifies 'lines' for REMOVE
                    lines_key = 'lines'
                    if lines_key not in change_item or not isinstance(change_item[lines_key], list):
                        raise KeyError(f"'{lines_key}' missing or not a list for REMOVE action.")
                    
                    sanitized_path = _sanitize_file_path(file_path, base_dir)
                    sanitized_changes.append({
                        'file_path': sanitized_path,
                        'action': action,
                        'lines': change_item[lines_key]
                    })
                else:
                    # Should not happen due to enum in schema, but good for robustness
                    output['malformed_blocks'].append(f"Unknown action type '{action}' encountered.")

            except PathTraversalError as pte:
                # Log path traversal attempts as warnings, but don't stop processing
                output['malformed_blocks'].append(f"Path Traversal Warning: {pte}")
                # Optionally, skip this change or add it with a flag
                # For now, we skip adding it to sanitized_changes
            except KeyError as ke:
                 output['malformed_blocks'].append(f"Missing required key in CODE_CHANGES item: {ke}")
            except Exception as e:
                 output['malformed_blocks'].append(f"Error processing change item '{change_item.get('file_path', 'N/A')}': {e}")

        output['changes'] = sanitized_changes

    except LLMOutputParsingError as e:
        output['malformed_blocks'].append(f"Critical Parsing Error: {e}")
        # Depending on requirements, you might want to re-raise or return partial data
        # For now, we log it and return the structure with the error message
    except InvalidSchemaError as e:
        output['malformed_blocks'].append(f"Schema Validation Error: {e}")
        # Log schema errors, potentially return partial data or raise
    except PathTraversalError as e:
         output['malformed_blocks'].append(f"Path Traversal Error: {e}")
         # Log path traversal errors
    except Exception as e: # Catch-all for unexpected errors during the process
        output['malformed_blocks'].append(f"Unexpected error during parsing: {e}")

    return output
