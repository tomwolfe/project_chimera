# src/utils/output_parser.py

import json
import logging
import re
import sys
import traceback
from typing import Dict, Any, List, Optional
from pathlib import Path
from pydantic import BaseModel, Field, validator, model_validator, ValidationError

from src.utils.path_utils import sanitize_and_validate_file_path

logger = logging.getLogger(__name__)

class InvalidSchemaError(Exception):
    """Exception raised when the LLM output does not match the expected schema."""
    pass

class CodeChange(BaseModel):
    file_path: str = Field(..., alias="FILE_PATH")
    action: str = Field(..., alias="ACTION")
    full_content: Optional[str] = Field(None, alias="FULL_CONTENT")
    lines: List[str] = Field(default_factory=list, alias="LINES")

    @validator('file_path')
    def validate_file_path(cls, v):
        """Validates and sanitizes the file path."""
        return sanitize_and_validate_file_path(v)
        
    @validator('action')
    def validate_action(cls, v):
        """Validates the action type."""
        valid_actions = ["ADD", "MODIFY", "REMOVE"]
        if v not in valid_actions:
            raise ValueError(f"Invalid action: '{v}'. Must be one of {valid_actions}.")
        return v

    @model_validator(mode='after')
    def check_content_based_on_action(self) -> 'CodeChange':
        """Ensures that full_content is provided for ADD/MODIFY and lines for REMOVE."""
        if self.action in ["ADD", "MODIFY"] and self.full_content is None:
            raise ValueError(f"full_content is required for action '{self.action}' on file '{self.file_path}'.")
        if self.action == "REMOVE" and not isinstance(self.lines, list):
            raise ValueError(f"lines must be a list for action 'REMOVE' on file '{self.file_path}'. Found type: {type(self.lines).__name__}.")
        return self

class LLMOutput(BaseModel):
    commit_message: str = Field(alias="COMMIT_MESSAGE")
    rationale: str = Field(alias="RATIONALE")
    code_changes: List[CodeChange] = Field(alias="CODE_CHANGES")
    conflict_resolution: Optional[str] = Field(None, alias="CONFLICT_RESOLUTION")
    unresolved_conflict: Optional[str] = Field(None, alias="UNRESOLVED_CONFLICT")

class LLMOutputParser:
    def __init__(self, provider):
        self.provider = provider
        self.logger = logger

    def _escape_json_string_value(self, value: str) -> str:
        """Escapes a string to be safely embedded as a JSON string value.
        This uses json.dumps to handle all necessary escaping (quotes, backslashes, newlines, etc.)
        and then removes the outer quotes added by json.dumps.
        """
        return json.dumps(value)[1:-1]

    def parse_and_validate(self, raw_output: str) -> Dict[str, Any]:
        """
        Parse and validate the raw LLM output.

        This method attempts to extract structured data from the LLM's response,
        handling various potential issues with the output format.
        """
        self.logger.debug(f"Attempting to parse raw output: {raw_output[:500]}...")

        malformed_blocks_list = []

        # --- IMPROVED JSON EXTRACTION ---
        # Try to find JSON within markdown code blocks first, then fall back to raw JSON.
        json_str = None
        json_block_match = re.search(r'```json\s*(\{[\s\S]*\})\s*```', raw_output, re.MULTILINE)
        if json_block_match:
            json_str = json_block_match.group(1)
            self.logger.debug("Extracted JSON from markdown code block.")
        else:
            raw_json_match = re.search(r'\{[\s\S]*\}', raw_output)
            if raw_json_match:
                json_str = raw_json_match.group(0)
                self.logger.debug("Extracted raw JSON from response.")
            else:
                json_str = raw_output
                self.logger.debug("No JSON structure found, using full response for parsing attempt.")

        if json_str is None:
            json_str = raw_output
            self.logger.error("JSON string extraction failed unexpectedly. Using raw output.")

        # Clean up common JSON formatting issues before parsing
        json_str = json_str.replace("'", '"')
        json_str = re.sub(r',\s*}', '}', json_str)
        json_str = re.sub(r',\s*]', ']', json_str)
        json_str = re.sub(r'([{,]\s*)([a-zA-Z_][a-zA-Z0-9_]*)\s*:', r'\1"\2":', json_str)
        
        # NEW HEURISTICS: Attempt to fix common LLM JSON formatting errors
        # 1. Missing comma between a string value and a new key (e.g., "val""key":)
        json_str = re.sub(r'("[^"]*")\s*("([A-Z_]+)":)', r'\1,\n\2', json_str)
        # 2. Missing comma between an object/array/boolean/number/null and a new key (e.g., }{ "key": or 123"key":)
        # This is more general and covers cases where the value isn't a string.
        json_str = re.sub(r'([}\]"\d]|true|false|null)\s*("([A-Z_]+)":)', r'\1,\n\2', json_str, flags=re.IGNORECASE)
        
        parsed = {}
        try:
            parsed = json.loads(json_str)
        except json.JSONDecodeError as e:
            self.logger.error(f"JSON decoding failed: {e}")
            # MODIFIED: Add structured error to malformed_blocks_list
            malformed_blocks_list.append({
                "type": "JSON_DECODE_ERROR",
                "message": str(e),
                "raw_string_snippet": json_str[:1000] + ("..." if len(json_str) > 1000 else "")
            })
            
            return {
                "COMMIT_MESSAGE": "Parsing error",
                "RATIONALE": self._escape_json_string_value(f"Failed to parse LLM output as JSON. Error: {e}\nAttempted JSON string: {json_str[:500]}..."),
                "CODE_CHANGES": [],
                "CONFLICT_RESOLUTION": None,
                "UNRESOLVED_CONFLICT": None,
                "malformed_blocks": malformed_blocks_list
            }

        # Validate against schema
        try:
            validated_output = LLMOutput(**parsed)
            self.logger.info("LLM output successfully validated against schema.")
            result_dict = validated_output.dict(by_alias=True)
            result_dict["malformed_blocks"] = malformed_blocks_list
            return result_dict
        except ValidationError as validation_e:
            self.logger.error(f"Schema validation failed: {validation_e}")
            # MODIFIED: Add structured error to malformed_blocks_list
            malformed_blocks_list.append({
                "type": "SCHEMA_VALIDATION_ERROR",
                "message": str(validation_e),
                "raw_string_snippet": raw_output[:500] + ("..." if len(raw_output) > 500 else "")
            })
            
            # Escape the raw output for the rationale
            escaped_raw_output_for_rationale = self._escape_json_string_value(raw_output[:500] + ("..." if len(raw_output) > 500 else ""))
            
            fallback_output = {
                "COMMIT_MESSAGE": "Schema validation failed",
                "RATIONALE": f"Original output: {escaped_raw_output_for_rationale}\nValidation Error: {self._escape_json_string_value(str(validation_e))}",
                "CODE_CHANGES": [],
                "CONFLICT_RESOLUTION": None,
                "UNRESOLVED_CONFLICT": None,
                "malformed_blocks": malformed_blocks_list
            }
            
            if isinstance(parsed, dict):
                fallback_output["COMMIT_MESSAGE"] = parsed.get("COMMIT_MESSAGE", fallback_output["COMMIT_MESSAGE"])
                
                if "RATIONALE" in parsed:
                    fallback_output["RATIONALE"] = self._escape_json_string_value(parsed["RATIONALE"])
                else:
                    fallback_output["RATIONALE"] = self._escape_json_string_value(fallback_output["RATIONALE"])
                
                original_code_changes = parsed.get("CODE_CHANGES")
                if isinstance(original_code_changes, list):
                    processed_code_changes = []
                    for index, item in enumerate(original_code_changes):
                        if isinstance(item, dict):
                            try:
                                valid_item = CodeChange(**item)
                                processed_code_changes.append(valid_item.dict(by_alias=True))
                            except ValidationError as inner_val_e:
                                self.logger.warning(f"Fallback: Malformed dictionary item in CODE_CHANGES at index {index} skipped. Error: {inner_val_e}")
                                # MODIFIED: Add structured error for malformed CODE_CHANGES item
                                malformed_blocks_list.append({
                                    "type": "MALFORMED_CODE_CHANGE_ITEM",
                                    "index": index,
                                    "message": str(inner_val_e),
                                    "raw_item": self._escape_json_string_value(str(item))
                                })
                                processed_code_changes.append({
                                    "FILE_PATH": "malformed_dict_entry",
                                    "ACTION": "ADD",
                                    "FULL_CONTENT": self._escape_json_string_value(f"LLM provided a malformed dictionary entry in CODE_CHANGES at index {index}. Validation error: {inner_val_e}"),
                                    "LINES": []
                                })
                        else:
                            self.logger.warning(f"Fallback: Non-dictionary item in CODE_CHANGES at index {index} skipped.")
                            # MODIFIED: Add structured error for non-dictionary CODE_CHANGES item
                            malformed_blocks_list.append({
                                "type": "NON_DICT_CODE_CHANGE_ITEM",
                                "index": index,
                                "message": "Item is not a dictionary.",
                                "raw_item": self._escape_json_string_value(str(item))
                            })
                            processed_code_changes.append({
                                "FILE_PATH": "malformed_non_dict_entry",
                                "ACTION": "ADD",
                                "FULL_CONTENT": self._escape_json_string_value(f"LLM provided a non-dictionary item in CODE_CHANGES at index {index}: {item}"),
                                "LINES": []
                            })
                    fallback_output["CODE_CHANGES"] = processed_code_changes
                else:
                    self.logger.warning(f"Fallback: CODE_CHANGES field was not a list or was missing in LLM output.")
                    # MODIFIED: Add structured error for malformed CODE_CHANGES field
                    malformed_blocks_list.append({
                        "type": "MALFORMED_CODE_CHANGES_FIELD",
                        "message": "CODE_CHANGES field was not a list or was missing.",
                        "raw_value": self._escape_json_string_value(str(original_code_changes))
                    })
                    fallback_output["CODE_CHANGES"] = []

            # The raw output is already captured in a structured block if it caused schema validation failure.
            # No need for this check anymore: if not any(raw_output[:50] in block for block in malformed_blocks_list):
            #    malformed_blocks_list.append(self._escape_json_string_value(f"Original raw output that caused schema validation failure:\n{raw_output}"))

            fallback_output["malformed_blocks"] = malformed_blocks_list
            return fallback_output
        except Exception as general_e:
            self.logger.error(f"An unexpected error occurred during schema validation: {general_e}")
            # MODIFIED: Add structured error for general unexpected errors
            malformed_blocks_list.append({
                "type": "UNEXPECTED_VALIDATION_ERROR",
                "message": str(general_e),
                "raw_string_snippet": raw_output
            })
            return {
                "COMMIT_MESSAGE": "Unexpected validation error",
                "RATIONALE": self._escape_json_string_value(f"An unexpected error occurred during schema validation: {general_e}\nRaw output: {raw_output[:500]}..."),
                "CODE_CHANGES": [],
                "CONFLICT_RESOLUTION": None,
                "UNRESOLVED_CONFLICT": None,
                "malformed_blocks": malformed_blocks_list
            }