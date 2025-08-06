# src/utils/output_parser.py
import json
import logging
import re
import sys
import traceback
from typing import Dict, Any, List, Optional
from pathlib import Path
from pydantic import BaseModel, Field, validator, model_validator

# Import the centralized path utility functions
from src.utils.path_utils import sanitize_and_validate_file_path

logger = logging.getLogger(__name__)

class InvalidSchemaError(Exception):
    """Exception raised when the LLM output does not match the expected schema."""
    pass

class CodeChange(BaseModel):
    file_path: str = Field(..., alias="FILE_PATH") # Original uses FILE_PATH
    action: str = Field(..., alias="ACTION") # Original uses ACTION
    full_content: Optional[str] = Field(None, alias="FULL_CONTENT") # Required for ADD/MODIFY
    lines: List[str] = Field(default_factory=list, alias="LINES") # Required for REMOVE, can be empty

    @validator('file_path')
    def validate_file_path(cls, v):
        """Validates and sanitizes the file path."""
        # This ensures the path is absolute and normalized.
        # It also ensures the path is within the project's base directory.
        return sanitize_and_validate_file_path(v)
        
    @validator('action')
    def validate_action(cls, v):
        """Validates the action type."""
        valid_actions = ["ADD", "MODIFY", "REMOVE"]
        if v not in valid_actions:
            raise ValueError(f"Invalid action: '{v}'. Must be one of {valid_actions}.")
        return v

    # Model validator to check content requirements based on action
    @model_validator(mode='after')
    def check_content_based_on_action(self) -> 'CodeChange':
        """Ensures that full_content is provided for ADD/MODIFY and lines for REMOVE."""
        if self.action in ["ADD", "MODIFY"] and self.full_content is None:
            raise ValueError(f"full_content is required for action '{self.action}' on file '{self.file_path}'.")
        if self.action == "REMOVE" and not isinstance(self.lines, list): # Check if lines is not a list
            raise ValueError(f"lines must be a list for action 'REMOVE' on file '{self.file_path}'. Found type: {type(self.lines).__name__}.")
        return self

class LLMOutput(BaseModel):
    commit_message: str = Field(alias="COMMIT_MESSAGE")
    rationale: str = Field(alias="RATIONALE")
    code_changes: List[CodeChange] = Field(alias="CODE_CHANGES")
    # These fields are optional as per the schema, but their presence/absence should be handled.
    conflict_resolution: Optional[str] = Field(None, alias="CONFLICT_RESOLUTION") # Optional
    unresolved_conflict: Optional[str] = Field(None, alias="UNRESOLVED_CONFLICT") # Optional
    malformed_blocks: Optional[List[str]] = Field(None) # Added to capture parsing errors

    @model_validator(mode='after')
    def check_code_changes_content(self) -> 'LLMOutput':
        """Ensures that required content fields within CodeChange are present based on action."""
        for i, change in enumerate(self.code_changes):
            # Pydantic's model_validator on CodeChange already handles this,
            # but this provides an extra layer of validation at the LLMOutput level
            # and allows for more specific error messages referencing the index.
            if change.action in ["ADD", "MODIFY"] and change.full_content is None: # Check for None explicitly
                raise ValueError(f"LLMOutput validation failed: full_content is required for action '{change.action}' on file '{change.file_path}' (index {i}).")
            if change.action == "REMOVE" and not isinstance(change.lines, list): # Check if lines is not a list
                raise ValueError(f"LLMOutput validation failed: lines must be a list for action 'REMOVE' on file '{change.file_path}' (index {i}). Found type: {type(change.lines).__name__}.")
        return self

class LLMOutputParser:
    def __init__(self, provider):
        self.provider = provider
        self.logger = logger

    def parse_and_validate(self, raw_output: str) -> Dict[str, Any]:
        """
        Parse and validate the raw LLM output.

        This method attempts to extract structured data from the LLM's response,
        handling various potential issues with the output format.
        """
        self.logger.debug(f"Attempting to parse raw output: {raw_output[:500]}...")

        # First, try to find JSON within the response (LLM might add extra text)
        # This regex looks for a JSON object starting with '{' and ending with '}'
        # It handles nested structures and escaped characters within the JSON.
        json_match = re.search(r'\{[\s\S]*\}', raw_output)
        if json_match:
            json_str = json_match.group(0)
            self.logger.debug("Extracted JSON from potentially verbose LLM response")
        else:
            json_str = raw_output
            self.logger.debug("No JSON object found in expected location, using full response")

        # Clean up common JSON formatting issues before parsing
        # 1. Fix single quotes to double quotes
        json_str = json_str.replace("'", '"')
        # 2. Fix trailing commas before closing braces/brackets
        json_str = re.sub(r',\s*}', '}', json_str)
        json_str = re.sub(r',\s*]', ']', json_str)
        # 3. Fix missing quotes around keys (basic attempt) - more robust than just wrapping
        json_str = re.sub(r'([{,]\s*)([a-zA-Z_][a-zA-Z0-9_]*)\s*:', r'\1"\2":', json_str)
        
        # Try to parse the JSON
        try:
            parsed = json.loads(json_str)
        except json.JSONDecodeError as e:
            self.logger.error(f"JSON decoding failed: {e}")
            # Instead of aggressive recovery, return a structured error with malformed blocks
            return {
                "commit_message": "Parsing error",
                "rationale": f"Failed to parse LLM output as JSON. Error: {e}\nRaw output: {raw_output[:500]}...",
                "code_changes": [],
                "conflict_resolution": None,
                "unresolved_conflict": None,
                "malformed_blocks": [f"JSONDecodeError: {e}", f"Raw output:\n{raw_output}"]
            }

        # Validate against schema
        try:
            # Use Pydantic model for validation
            validated_output = LLMOutput(**parsed)
            self.logger.info("LLM output successfully validated against schema.")
            # Return as a dictionary for broader compatibility, including malformed_blocks if any
            result_dict = validated_output.dict(by_alias=True)
            # Ensure malformed_blocks is always a list, even if empty
            if "malformed_blocks" not in result_dict or result_dict["malformed_blocks"] is None:
                result_dict["malformed_blocks"] = []
            return result_dict
        except Exception as validation_e: # Catch Pydantic validation errors and others
            self.logger.error(f"Schema validation failed: {validation_e}")
            # Attempt to provide a fallback if validation fails but parsing succeeded
            # This might happen if required fields are missing or malformed.
            # We'll try to construct a minimal valid output.
            fallback_output = {
                "commit_message": "Schema validation failed",
                "rationale": f"Original output: {raw_output[:500]}...\nValidation Error: {validation_e}",
                "code_changes": [], # Default to empty list
                "conflict_resolution": None,
                "unresolved_conflict": None,
                "malformed_blocks": [f"Schema validation error: {validation_e}"] # Add validation error to malformed blocks
            }
            # If the original parsed data had some structure, try to incorporate it
            if isinstance(parsed, dict):
                fallback_output["commit_message"] = parsed.get("COMMIT_MESSAGE", fallback_output["commit_message"])
                fallback_output["rationale"] = parsed.get("RATIONALE", fallback_output["rationale"])
                # Try to salvage code_changes if they exist and are list-like
                original_code_changes = parsed.get("CODE_CHANGES")
                if isinstance(original_code_changes, list):
                    processed_code_changes = []
                    for index, item in enumerate(original_code_changes):
                        if isinstance(item, dict):
                            # It's a dictionary, but might still be invalid according to CodeChange schema.
                            # Try to validate it as a CodeChange.
                            try:
                                valid_item = CodeChange(**item)
                                processed_code_changes.append(valid_item.dict(by_alias=True)) # Use by_alias=True to keep original keys like FILE_PATH
                            except ValidationError as inner_val_e:
                                self.logger.warning(f"Fallback: Malformed dictionary item in CODE_CHANGES at index {index} skipped. Error: {inner_val_e}")
                                # Add a placeholder for this malformed dictionary item and log the error
                                fallback_output["malformed_blocks"].append(f"Malformed CODE_CHANGES item at index {index}: {item}. Error: {inner_val_e}")
                                processed_code_changes.append({
                                    "FILE_PATH": "malformed_dict_entry",
                                    "ACTION": "ADD", # Default action for malformed
                                    "FULL_CONTENT": f"LLM provided a malformed dictionary entry in CODE_CHANGES at index {index}. Validation error: {inner_val_e}",
                                    "LINES": []
                                })
                        else:
                            # It's not a dictionary (e.g., a string), so create a placeholder.
                            self.logger.warning(f"Fallback: Non-dictionary item in CODE_CHANGES at index {index} skipped.")
                            fallback_output["malformed_blocks"].append(f"Non-dictionary item in CODE_CHANGES at index {index}: {item}")
                            processed_code_changes.append({
                                "FILE_PATH": "malformed_non_dict_entry", # Use a placeholder file path
                                "ACTION": "ADD", # Default action for malformed
                                "FULL_CONTENT": f"LLM provided a non-dictionary item in CODE_CHANGES at index {index}: {item}",
                                "LINES": []
                            })
                    fallback_output["code_changes"] = processed_code_changes
                else:
                    # CODE_CHANGES was not a list, or was missing.
                    self.logger.warning(f"Fallback: CODE_CHANGES field was not a list or was missing in LLM output.")
                    # Add a specific malformed block entry for this case.
                    if "malformed_blocks" not in fallback_output:
                        fallback_output["malformed_blocks"] = []
                    fallback_output["malformed_blocks"].append(f"CODE_CHANGES field was not a list or was missing in LLM output. Raw value: {original_code_changes}")
                    # Ensure code_changes is an empty list if it was malformed.
                    fallback_output["code_changes"] = []

            # Ensure the raw output is captured if it wasn't already part of a specific malformed_blocks entry.
            if not any(raw_output[:50] in block for block in fallback_output["malformed_blocks"]):
                fallback_output["malformed_blocks"].append(f"Original raw output that caused schema validation failure:\n{raw_output}")

            # Ensure malformed_blocks is always a list, even if empty
            if "malformed_blocks" not in fallback_output or fallback_output["malformed_blocks"] is None:
                fallback_output["malformed_blocks"] = []

            return fallback_output

    def _attempt_json_recovery(self, json_str: str) -> str:
        """Attempt to fix common JSON formatting issues in the response."""
        # Fix missing quotes around keys
        json_str = re.sub(r'([{,]\s*)([a-zA-Z_][a-zA-Z0-9_]*)\s*:', r'\1"\2":', json_str)
        
        # Fix single quotes to double quotes
        json_str = json_str.replace("'", '"')
        
        # Fix trailing commas
        json_str = re.sub(r',\s*}', '}', json_str)
        json_str = re.sub(r',\s*]', ']', json_str)
        
        # Ensure it starts and ends with a JSON object structure if it looks like one
        # This is a heuristic and might not always be correct.
        # Only wrap if it's not already a valid JSON object or array.
        # This prevents wrapping a valid JSON array into an invalid object.
        if not (json_str.strip().startswith('{') and json_str.strip().endswith('}')) and \
           not (json_str.strip().startswith('[') and json_str.strip().endswith(']')):
            try: # Try to parse it as is first
                json.loads(json_str)
            except json.JSONDecodeError: # If it's not valid JSON, then try wrapping it as an object
                json_str = '{' + json_str + '}'
        
        return json_str

    # Removed unused _validate_schema method
    # def _validate_schema(self, data: Dict) -> bool:
    #     """Validates the parsed data against the LLMOutput schema."""
    #     try:
    #         LLMOutput(**data)
    #         return True
    #     except Exception as e:
    #         self.logger.warning(f"Schema validation failed: {e}")
    #         return False