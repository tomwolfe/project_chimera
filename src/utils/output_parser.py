# src/utils/output_parser.py

import json
import logging
import re
import sys
import traceback
from typing import Dict, Any, List, Optional, Type
from pathlib import Path
from pydantic import BaseModel, Field, validator, model_validator, ValidationError

# Import models from src.models
from src.models import CodeChange, LLMOutput, ContextAnalysisOutput

logger = logging.getLogger(__name__)

class InvalidSchemaError(Exception):
    """Exception raised when the LLM output does not match the expected schema."""
    pass

class LLMOutputParser:
    def __init__(self):
        self.logger = logger

    # REMOVED: _escape_json_string_value is problematic and not needed.
    # json.dumps handles all necessary escaping when serializing Python objects to JSON strings.

    def _extract_and_sanitize_json_string(self, text: str) -> Optional[str]:
        """
        Attempts to extract the outermost valid JSON object or array from text.
        This method prioritizes finding a structurally sound JSON block using
        brace/bracket counting, then applies basic sanitization.
        It correctly handles escaped characters within strings and avoids
        incorrectly pairing delimiters. This implementation refines the LLM's
        own suggested approach for more robust extraction.
        """
        self.logger.debug("Attempting robust JSON extraction and sanitization...")

        # 1. Remove markdown code block fences if present
        text_cleaned = re.sub(r'```json\s*', '', text, flags=re.MULTILINE)
        text_cleaned = re.sub(r'\s*```', '', text_cleaned, flags=re.MULTILINE)
        
        # Find potential start indices of JSON objects or arrays
        potential_starts = []
        for i, char in enumerate(text_cleaned):
            if char == '{' or char == '[':
                potential_starts.append(i)
        
        if not potential_starts:
            self.logger.debug("No JSON start delimiters found.")
            return None

        # Iterate through potential start points to find a valid JSON block
        for start_index in potential_starts:
            balance = 0
            # Stack to keep track of expected closing delimiters for nested structures
            expected_closers_stack = [] 

            # Determine the type of the starting delimiter
            if text_cleaned[start_index] == '{':
                balance = 1
                expected_closers_stack.append('}')
            elif text_cleaned[start_index] == '[':
                balance = 1
                expected_closers_stack.append(']')
            else:
                continue # Should not happen if potential_starts is correctly populated

            end_index = -1
            # Iterate from the character *after* the start_index
            for i in range(start_index + 1, len(text_cleaned)):
                char = text_cleaned[i]

                # Handle nested structures
                if char == '{':
                    balance += 1
                    expected_closers_stack.append('}')
                elif char == '[':
                    balance += 1
                    expected_closers_stack.append(']')
                elif char == '}':
                    balance -= 1
                    if expected_closers_stack and expected_closers_stack[-1] == '}':
                        expected_closers_stack.pop()
                    else:
                        # Mismatched closing brace, this path is invalid
                        balance = -999 # Force invalid balance
                        break # Exit inner loop for this start_index
                elif char == ']':
                    balance -= 1
                    if expected_closers_stack and expected_closers_stack[-1] == ']':
                        expected_closers_stack.pop()
                    else:
                        # Mismatched closing bracket, this path is invalid
                        balance = -999 # Force invalid balance
                        break # Exit inner loop for this start_index
                
                # If balance is zero and stack is empty, we found a complete outermost structure
                if balance == 0 and not expected_closers_stack:
                    end_index = i + 1 # +1 to include the closing delimiter
                    potential_json_str = text_cleaned[start_index:end_index]
                    try:
                        # Attempt to parse to validate the extracted string
                        json.loads(potential_json_str)
                        self.logger.debug(f"Successfully extracted valid JSON block: {potential_json_str[:100]}...")
                        
                        # Apply basic sanitization like removing trailing commas before closing braces/brackets
                        # This helps with minor LLM quirks.
                        potential_json_str = re.sub(r',\s*([\}\]])', r'\1', potential_json_str)
                        return potential_json_str.strip()
                    except json.JSONDecodeError:
                        self.logger.debug("Extracted block is not valid JSON, continuing search for a longer valid one.")
                        # Do NOT break here. Continue the inner loop to find a larger, valid block
                        # that might encompass the current invalid one, or a different valid block.
                        continue 
            # If the inner loop finishes and balance is not 0 or stack not empty,
            # this start_index did not lead to a valid, complete JSON block.
            # The outer loop will try the next potential_start.

        self.logger.debug("Failed to extract a valid JSON block after all attempts.")
        return None

    def parse_and_validate(self, raw_output: str, schema_model: Type[BaseModel]) -> Dict[str, Any]:
        """
        Parse and validate the raw LLM output against a given Pydantic schema.
        Returns a dictionary representation of the validated model, or a dictionary
        containing 'malformed_blocks' if parsing/validation fails.
        This output is then used by app.py to display errors or processed results.
        """
        self.logger.debug(f"Attempting to parse raw output: {raw_output[:500]}...")

        malformed_blocks_list = []

        # Apply robust extraction and sanitization
        sanitized_json_str = self._extract_and_sanitize_json_string(raw_output)
        
        parsed_data = {}
        if not sanitized_json_str:
            self.logger.error("Failed to extract any JSON structure from the output.")
            malformed_blocks_list.append({
                "type": "JSON_EXTRACTION_FAILED",
                "message": "Could not find or extract a valid JSON structure.",
                "raw_string_snippet": raw_output[:1000] + ("..." if len(raw_output) > 1000 else "") # Use raw_output directly
            })
            # Return a structured error dictionary that app.py can interpret
            return {
                "COMMIT_MESSAGE": "Parsing error",
                "RATIONALE": f"Failed to parse LLM output as JSON. Error: Could not extract valid JSON structure.\nAttempted raw output: {raw_output[:500]}...",
                "CODE_CHANGES": [], # Default empty list for CODE_CHANGES
                "CONFLICT_RESOLUTION": None,
                "UNRESOLVED_CONFLICT": None,
                "malformed_blocks": malformed_blocks_list # Crucial for app.py to know parsing failed
            }

        try:
            # Attempt to parse the sanitized string as JSON
            parsed_data = json.loads(sanitized_json_str)
        except json.JSONDecodeError as e:
            self.logger.error(f"JSON decoding failed after extraction: {e}")
            malformed_blocks_list.append({
                "type": "JSON_DECODE_ERROR",
                "message": str(e),
                "raw_string_snippet": sanitized_json_str[:1000] + ("..." if len(sanitized_json_str) > 1000 else "") # Use sanitized_json_str directly
            })
            # Return a structured error dictionary that app.py can interpret
            return {
                "COMMIT_MESSAGE": "Parsing error",
                "RATIONALE": f"Failed to parse LLM output as JSON. Error: {e}\nAttempted JSON string: {sanitized_json_str[:500]}...",
                "CODE_CHANGES": [], # Default empty list for CODE_CHANGES
                "CONFLICT_RESOLUTION": None,
                "UNRESOLVED_CONFLICT": None,
                "malformed_blocks": malformed_blocks_list # Crucial for app.py to know parsing failed
            }

        # Validate against the provided schema_model
        try:
            validated_output = schema_model(**parsed_data)
            self.logger.info(f"LLM output successfully validated against {schema_model.__name__} schema.")
            result_dict = validated_output.model_dump(by_alias=True) # Use model_dump for Pydantic v2+
            # Include any malformed blocks found during extraction (e.g., if markdown was stripped)
            result_dict["malformed_blocks"] = malformed_blocks_list 
            return result_dict
        except ValidationError as validation_e:
            self.logger.error(f"Schema validation failed for {schema_model.__name__}: {validation_e}")
            malformed_blocks_list.append({
                "type": "SCHEMA_VALIDATION_ERROR",
                "message": str(validation_e),
                "raw_string_snippet": raw_output # Use raw_output directly
            })
            
            # Attempt to salvage partial data for LLMOutput if it was the target schema
            # This allows the system to report *what* was wrong, even if the full structure failed.
            fallback_output = {
                "COMMIT_MESSAGE": "Schema validation failed",
                "RATIONALE": f"Original output: {raw_output[:500] + ('...' if len(raw_output) > 500 else '')}\nValidation Error: {str(validation_e)}",
                "CODE_CHANGES": [],
                "CONFLICT_RESOLUTION": None,
                "UNRESOLVED_CONFLICT": None,
                "malformed_blocks": malformed_blocks_list # Pass validation errors as malformed blocks
            }
            
            # If the target schema is LLMOutput, try to populate fields from parsed_data
            if schema_model == LLMOutput and isinstance(parsed_data, dict):
                fallback_output["COMMIT_MESSAGE"] = parsed_data.get("COMMIT_MESSAGE", fallback_output["COMMIT_MESSAGE"])
                if "RATIONALE" in parsed_data:
                    fallback_output["RATIONALE"] = parsed_data["RATIONALE"] # Do not escape here
                
                original_code_changes = parsed_data.get("CODE_CHANGES")
                if isinstance(original_code_changes, list):
                    processed_code_changes = []
                    for index, item in enumerate(original_code_changes):
                        if isinstance(item, dict):
                            try:
                                valid_item = CodeChange(**item)
                                processed_code_changes.append(valid_item.model_dump(by_alias=True)) # Use model_dump
                            except ValidationError as inner_val_e:
                                self.logger.warning(f"Fallback: Malformed dictionary item in CODE_CHANGES at index {index} skipped. Error: {inner_val_e}")
                                malformed_blocks_list.append({
                                    "type": "MALFORMED_CODE_CHANGE_ITEM",
                                    "index": index,
                                    "message": str(inner_val_e),
                                    "raw_item": str(item) # Use raw item directly
                                })
                                # Add a placeholder for the malformed item to keep list structure
                                processed_code_changes.append({
                                    "FILE_PATH": f"malformed_entry_{index}",
                                    "ACTION": "ADD", # Default to ADD for malformed
                                    "FULL_CONTENT": f"LLM provided a malformed dictionary entry in CODE_CHANGES at index {index}. Validation error: {inner_val_e}", # Do not escape here
                                    "LINES": []
                                })
                        else:
                            self.logger.warning(f"Fallback: Non-dictionary item in CODE_CHANGES at index {index} skipped.")
                            malformed_blocks_list.append({
                                "type": "NON_DICT_CODE_CHANGE_ITEM",
                                "index": index,
                                "message": "Item is not a dictionary.",
                                "raw_item": str(item) # Use raw item directly
                            })
                            # Add a placeholder for the malformed item
                            processed_code_changes.append({
                                "FILE_PATH": f"malformed_entry_{index}",
                                "ACTION": "ADD", # Default to ADD for malformed
                                "FULL_CONTENT": f"LLM provided a non-dictionary item in CODE_CHANGES at index {index}: {item}", # Do not escape here
                                "LINES": []
                            })
                    fallback_output["CODE_CHANGES"] = processed_code_changes
                else:
                    self.logger.warning(f"Fallback: CODE_CHANGES field was not a list or was missing in LLM output.")
                    malformed_blocks_list.append({
                        "type": "MALFORMED_CODE_CHANGES_FIELD",
                        "message": "CODE_CHANGES field was not a list or was missing.",
                        "raw_value": str(original_code_changes) # Use raw value directly
                    })
                    fallback_output["CODE_CHANGES"] = []

            return fallback_output
        except Exception as general_e:
            self.logger.error(f"An unexpected error occurred during schema validation: {general_e}")
            malformed_blocks_list.append({
                "type": "UNEXPECTED_VALIDATION_ERROR",
                "message": str(general_e),
                "raw_string_snippet": raw_output # Use raw_output directly
            })
            return {
                "COMMIT_MESSAGE": "Unexpected validation error",
                "RATIONALE": f"An unexpected error occurred during schema validation: {general_e}\nRaw output: {raw_output[:500]}...",
                "CODE_CHANGES": [],
                "CONFLICT_RESOLUTION": None,
                "UNRESOLVED_CONFLICT": None,
                "malformed_blocks": malformed_blocks_list
            }