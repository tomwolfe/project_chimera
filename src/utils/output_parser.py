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

    def _escape_json_string_value(self, value: str) -> str:
        """Escapes a string to be safely embedded as a JSON string value.
        This uses json.dumps to handle all necessary escaping (quotes, backslashes, newlines, etc.)
        and then removes the outer quotes added by json.dumps.
        """
        # Handle None explicitly to avoid "null" string in error messages
        if value is None:
            return ""
        # json.dumps will add outer quotes, so we slice them off.
        # It also handles unicode characters.
        return json.dumps(value)[1:-1]

    def _sanitize_json_string(self, json_str: str) -> str:
        """Applies deterministic sanitization rules to LLM output to fix common JSON errors."""
        original_json_str = json_str # Keep original for fallback if sanitization fails

        # 1. Remove markdown code block fences if present, but keep content.
        # This helps ensure that json.loads receives a clean JSON string.
        json_str = re.sub(r'```json\s*', '', json_str, flags=re.MULTILINE)
        json_str = re.sub(r'\s*```', '', json_str, flags=re.MULTILINE)
        
        # 2. Remove C-style comments (// and /* */)
        json_str = re.sub(r"//.*", "", json_str)
        json_str = re.sub(r"/\*.*?\*/", "", json_str, flags=re.DOTALL)

        # 3. Replace single quotes with double quotes for keys and string values
        json_str = json_str.replace("'", '"')

        # 4. Remove trailing commas before closing braces/brackets
        json_str = re.sub(r',\s*([\}\]])', r'\1', json_str)
        
        # Trim leading/trailing whitespace
        return json_str.strip()

    def parse_and_validate(self, raw_output: str, schema_model: Type[BaseModel]) -> Dict[str, Any]:
        """
        Parse and validate the raw LLM output against a given Pydantic schema.
        Returns a dictionary representation of the validated model, or a dictionary
        containing 'malformed_blocks' if parsing/validation fails.
        This output is then used by app.py to display errors or processed results.
        """
        self.logger.debug(f"Attempting to parse raw output: {raw_output[:500]}...")

        malformed_blocks_list = []

        # Apply sanitization heuristics
        sanitized_json_str = self._sanitize_json_string(raw_output)
        
        parsed_data = {}
        try:
            # Attempt to parse the sanitized string as JSON
            # json.loads will raise JSONDecodeError if the string is not valid JSON,
            # even after sanitization. This is where "Extra data" errors might occur
            # if the LLM output is truncated or has trailing text.
            parsed_data = json.loads(sanitized_json_str)
        except json.JSONDecodeError as e:
            self.logger.error(f"JSON decoding failed after sanitization: {e}")
            malformed_blocks_list.append({
                "type": "JSON_DECODE_ERROR",
                "message": str(e),
                "raw_string_snippet": self._escape_json_string_value(sanitized_json_str[:1000] + ("..." if len(sanitized_json_str) > 1000 else ""))
            })
            # Return a structured error dictionary that app.py can interpret
            return {
                "COMMIT_MESSAGE": "Parsing error",
                "RATIONALE": self._escape_json_string_value(f"Failed to parse LLM output as JSON. Error: {e}\nAttempted JSON string: {sanitized_json_str[:500]}..."),
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
            # Include any malformed blocks found during sanitization (e.g., if markdown was stripped)
            result_dict["malformed_blocks"] = malformed_blocks_list 
            return result_dict
        except ValidationError as validation_e:
            self.logger.error(f"Schema validation failed for {schema_model.__name__}: {validation_e}")
            malformed_blocks_list.append({
                "type": "SCHEMA_VALIDATION_ERROR",
                "message": str(validation_e),
                "raw_string_snippet": self._escape_json_string_value(raw_output[:500] + ("..." if len(raw_output) > 500 else ""))
            })
            
            # Attempt to salvage partial data for LLMOutput if it was the target schema
            # This allows the system to report *what* was wrong, even if the full structure failed.
            fallback_output = {
                "COMMIT_MESSAGE": "Schema validation failed",
                "RATIONALE": self._escape_json_string_value(f"Original output: {self._escape_json_string_value(raw_output[:500] + ('...' if len(raw_output) > 500 else ''))}\nValidation Error: {self._escape_json_string_value(str(validation_e))}"),
                "CODE_CHANGES": [],
                "CONFLICT_RESOLUTION": None,
                "UNRESOLVED_CONFLICT": None,
                "malformed_blocks": malformed_blocks_list # Pass validation errors as malformed blocks
            }
            
            # If the target schema is LLMOutput, try to populate fields from parsed_data
            if schema_model == LLMOutput and isinstance(parsed_data, dict):
                fallback_output["COMMIT_MESSAGE"] = parsed_data.get("COMMIT_MESSAGE", fallback_output["COMMIT_MESSAGE"])
                if "RATIONALE" in parsed_data:
                    fallback_output["RATIONALE"] = self._escape_json_string_value(parsed_data["RATIONALE"])
                
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
                                    "raw_item": self._escape_json_string_value(str(item))
                                })
                                # Add a placeholder for the malformed item to keep list structure
                                processed_code_changes.append({
                                    "FILE_PATH": f"malformed_entry_{index}",
                                    "ACTION": "ADD", # Default to ADD for malformed
                                    "FULL_CONTENT": self._escape_json_string_value(f"LLM provided a malformed dictionary entry in CODE_CHANGES at index {index}. Validation error: {inner_val_e}"),
                                    "LINES": []
                                })
                        else:
                            self.logger.warning(f"Fallback: Non-dictionary item in CODE_CHANGES at index {index} skipped.")
                            malformed_blocks_list.append({
                                "type": "NON_DICT_CODE_CHANGE_ITEM",
                                "index": index,
                                "message": "Item is not a dictionary.",
                                "raw_item": self._escape_json_string_value(str(item))
                            })
                            # Add a placeholder for the malformed item
                            processed_code_changes.append({
                                "FILE_PATH": f"malformed_entry_{index}",
                                "ACTION": "ADD", # Default to ADD for malformed
                                "FULL_CONTENT": self._escape_json_string_value(f"LLM provided a non-dictionary item in CODE_CHANGES at index {index}: {item}"),
                                "LINES": []
                            })
                    fallback_output["CODE_CHANGES"] = processed_code_changes
                else:
                    self.logger.warning(f"Fallback: CODE_CHANGES field was not a list or was missing in LLM output.")
                    malformed_blocks_list.append({
                        "type": "MALFORMED_CODE_CHANGES_FIELD",
                        "message": "CODE_CHANGES field was not a list or was missing.",
                        "raw_value": self._escape_json_string_value(str(original_code_changes))
                    })
                    fallback_output["CODE_CHANGES"] = []

            return fallback_output
        except Exception as general_e:
            self.logger.error(f"An unexpected error occurred during schema validation: {general_e}")
            malformed_blocks_list.append({
                "type": "UNEXPECTED_VALIDATION_ERROR",
                "message": str(general_e),
                "raw_string_snippet": self._escape_json_string_value(raw_output)
            })
            return {
                "COMMIT_MESSAGE": "Unexpected validation error",
                "RATIONALE": self._escape_json_string_value(f"An unexpected error occurred during schema validation: {general_e}\nRaw output: {raw_output[:500]}..."),
                "CODE_CHANGES": [],
                "CONFLICT_RESOLUTION": None,
                "UNRESOLVED_CONFLICT": None,
                "malformed_blocks": malformed_blocks_list
            }