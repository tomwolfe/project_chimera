# src/utils/output_parser.py
import json
import logging
import re
import sys
import traceback
from typing import Dict, Any, List, Optional
from pathlib import Path
from pydantic import BaseModel, Field, validator, model_validator, ValidationError # ADDED: Import ValidationError explicitly

# Import the centralized path utility functions
from src.utils.path_utils import sanitize_and_validate_file_path

logger = logging.getLogger(__name__)

class InvalidSchemaError(Exception):
    """Exception raised when the LLM output does not match the expected schema."""
    pass

class CodeChange(BaseModel):
    # MODIFIED: Kept aliases to expect uppercase keys from LLM, consistent with top-level keys
    file_path: str = Field(..., alias="FILE_PATH")
    action: str = Field(..., alias="ACTION")
    full_content: Optional[str] = Field(None, alias="FULL_CONTENT")
    lines: List[str] = Field(default_factory=list, alias="LINES")

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
    # REMOVED: malformed_blocks is NOT part of the expected LLM output schema, it's for internal error reporting.

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

        malformed_blocks_list = [] # ADDED: Initialize list for malformed blocks

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
        
        parsed = {} # Initialize parsed dict
        # Try to parse the JSON
        try:
            parsed = json.loads(json_str)
        except json.JSONDecodeError as e:
            self.logger.error(f"JSON decoding failed: {e}")
            malformed_blocks_list.append(f"JSONDecodeError: {e}") # ADDED
            malformed_blocks_list.append(f"Raw output:\n{raw_output}") # ADDED
            return {
                "commit_message": "Parsing error",
                "rationale": f"Failed to parse LLM output as JSON. Error: {e}\nRaw output: {raw_output[:500]}...",
                "code_changes": [],
                "conflict_resolution": None,
                "unresolved_conflict": None,
                "malformed_blocks": malformed_blocks_list # ADDED
            }

        # Validate against schema
        try:
            # Use Pydantic model for validation
            validated_output = LLMOutput(**parsed)
            self.logger.info("LLM output successfully validated against schema.")
            # Return as a dictionary for broader compatibility
            result_dict = validated_output.dict(by_alias=True)
            result_dict["malformed_blocks"] = malformed_blocks_list # ADDED: Add the collected malformed blocks
            return result_dict
        except ValidationError as validation_e: # Catch Pydantic validation errors
            self.logger.error(f"Schema validation failed: {validation_e}")
            malformed_blocks_list.append(f"Schema validation error: {validation_e}") # ADDED
            
            fallback_output = {
                "commit_message": "Schema validation failed",
                "rationale": f"Original output: {raw_output[:500]}...\nValidation Error: {validation_e}",
                "code_changes": [], # Default to empty list
                "conflict_resolution": None,
                "unresolved_conflict": None,
                "malformed_blocks": malformed_blocks_list # ADDED: Add validation error to malformed blocks
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
                                malformed_blocks_list.append(f"Malformed CODE_CHANGES item at index {index}: {item}. Error: {inner_val_e}") # ADDED
                                processed_code_changes.append({
                                    "FILE_PATH": "malformed_dict_entry", # Use uppercase to match Pydantic model alias
                                    "ACTION": "ADD", # Default action for malformed
                                    "FULL_CONTENT": f"LLM provided a malformed dictionary entry in CODE_CHANGES at index {index}. Validation error: {inner_val_e}",
                                    "LINES": []
                                })
                        else:
                            # It's not a dictionary (e.g., a string), so create a placeholder.
                            self.logger.warning(f"Fallback: Non-dictionary item in CODE_CHANGES at index {index} skipped.")
                            malformed_blocks_list.append(f"Non-dictionary item in CODE_CHANGES at index {index}: {item}") # ADDED
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
                    malformed_blocks_list.append(f"CODE_CHANGES field was not a list or was missing in LLM output. Raw value: {original_code_changes}") # ADDED
                    # Ensure code_changes is an empty list if it was malformed.
                    fallback_output["code_changes"] = []

            # Ensure the raw output is captured if it wasn't already part of a specific malformed_blocks entry.
            if not any(raw_output[:50] in block for block in malformed_blocks_list):
                malformed_blocks_list.append(f"Original raw output that caused schema validation failure:\n{raw_output}")

            fallback_output["malformed_blocks"] = malformed_blocks_list # Ensure this is updated
            return fallback_output
        except Exception as general_e: # Catch any other unexpected errors during validation
            self.logger.error(f"An unexpected error occurred during schema validation: {general_e}")
            malformed_blocks_list.append(f"Unexpected error during schema validation: {general_e}") # ADDED
            malformed_blocks_list.append(f"Original raw output:\n{raw_output}") # ADDED
            return {
                "commit_message": "Unexpected validation error",
                "rationale": f"An unexpected error occurred during schema validation: {general_e}\nRaw output: {raw_output[:500]}...",
                "code_changes": [],
                "conflict_resolution": None,
                "unresolved_conflict": None,
                "malformed_blocks": malformed_blocks_list # ADDED
            }

    # REMOVED: _attempt_json_recovery method (unused)
    # REMOVED: _validate_schema method (unused)