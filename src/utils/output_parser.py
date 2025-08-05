# src/utils/output_parser.py
import json
import logging
import re
from typing import Dict, Any, List
from pathlib import Path
from pydantic import BaseModel, Field, validator, model_validator

# Import the centralized path utility functions - CHANGED FROM 'utils.path_utils'
from src.utils.path_utils import sanitize_and_validate_file_path

logger = logging.getLogger(__name__)

class CodeChange(BaseModel):
    file_path: str
    action: str
    full_content: str | None = Field(None, alias="full_content") # Optional for REMOVE
    lines: List[str] | None = None # Optional for ADD/MODIFY

    @validator('file_path', pre=True)
    def validate_file_path(cls, v):
        """Validates file path to prevent traversal and ensure it's within the project's src directory."""
        # The sanitize_and_validate_file_path function handles empty checks, 
        # character sanitization, absolute path prevention, and containment within PROJECT_BASE_DIR.
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
    def check_content_based_on_action(self) -> 'CodeChange': # Changed return type hint
        """Ensures that full_content is provided for ADD/MODIFY and lines for REMOVE."""
        if self.action in ["ADD", "MODIFY"] and not self.full_content:
            raise ValueError(
                f"full_content is required for action '{self.action}' on file '{self.file_path}'.")
        # For REMOVE action, 'lines' should be provided, even if it's an empty list.
        if self.action == "REMOVE" and self.lines is None:
             raise ValueError(
                 f"lines must be provided (can be an empty list) for action 'REMOVE' on file '{self.file_path}'.")
        # Ensure 'lines' is a list if provided for REMOVE action
        if self.action == "REMOVE" and self.lines is not None and not isinstance(self.lines, list):
             raise ValueError(f"lines must be a list for action 'REMOVE' on file '{self.file_path}'. Found type: {type(self.lines).__name__}.")
        return self

class LLMOutput(BaseModel):
    commit_message: str = Field(alias="COMMIT_MESSAGE")
    rationale: str = Field(alias="RATIONALE")
    code_changes: List[CodeChange] = Field(alias="CODE_CHANGES")
    # These fields are optional as per the schema, but their presence/absence should be handled.
    conflict_resolution: str | None = Field(None, alias="CONFLICT_RESOLUTION")
    unresolved_conflict: str | None = Field(None, alias="UNRESOLVED_CONFLICT")

    @model_validator(mode='after')
    def check_code_changes_content(self) -> 'LLMOutput':
        """Ensures that required content fields within CodeChange are present based on action."""
        for i, change in enumerate(self.code_changes):
            # Pydantic's model_validator on CodeChange already handles this, 
            # but this provides an extra layer of validation at the LLMOutput level
            # and allows for more specific error messages referencing the index.
            if change.action in ["ADD", "MODIFY"] and not change.full_content:
                raise ValueError(
                    f"LLMOutput validation failed: full_content is required for action '{change.action}' on file '{change.file_path}' (index {i}).")
            if change.action == "REMOVE" and change.lines is None:
                 raise ValueError(
                     f"LLMOutput validation failed: lines must be provided for action 'REMOVE' on file '{change.file_path}' (index {i}).")
            if change.action == "REMOVE" and change.lines is not None and not isinstance(change.lines, list):
                 raise ValueError(f"LLMOutput validation failed: lines must be a list for action 'REMOVE' on file '{change.file_path}' (index {i}). Found type: {type(change.lines).__name__}.")
        return self

class LLMOutputParser:
    def __init__(self, llm_provider):
        self.llm_provider = llm_provider
        self.logger = logging.getLogger(self.__class__.__name__)

    def parse_and_validate(self, raw_output: str) -> Dict[str, Any]: # Return Dict for broader compatibility, Pydantic validation is internal
        json_content_str = None
        extracted_json_block = None

        try:
            # Regex to find JSON within ```json ... ``` or ``` ... ``` blocks.
            # This prioritizes ```json blocks but falls back to generic ``` blocks.
            match_json_block = re.search(r"```json\n(.*?)\n```|```\n(.*?)\n```", raw_output, re.DOTALL)
            
            if match_json_block:
                extracted_json_block = match_json_block.group(1) or match_json_block.group(2)
                self.logger.info("Extracted JSON block from markdown.")
            else:
                self.logger.warning("No markdown JSON block found. Attempting to parse entire output as JSON.")
                # Fallback: If no markdown block is found, attempt to parse the entire raw output.
                extracted_json_block = raw_output

            if not extracted_json_block or not extracted_json_block.strip():
                self.logger.error("No JSON content found or extracted after attempting markdown parsing.")
                raise ValueError("No JSON content found in the LLM output.")

            # --- NEW CHECK ---
            # Ensure the extracted block looks like a JSON object or array before proceeding.
            # This helps filter out plain text that might have been mistakenly captured,
            # preventing errors in normalization or json.loads.
            cleaned_block_for_check = extracted_json_block.strip()
            if not (cleaned_block_for_check.startswith('{') and cleaned_block_for_check.endswith('}')) and \
               not (cleaned_block_for_check.startswith('[') and cleaned_block_for_check.endswith(']')):
                self.logger.error(f"Extracted content does not appear to be valid JSON structure (missing {{}} or []). Content: {cleaned_block_for_check[:200]}...")
                raise ValueError("Extracted content does not appear to be valid JSON structure.")
            # --- END NEW CHECK ---

            # Attempt to normalize potential formatting issues before parsing
            normalized_json_str = self.normalize_json_string(extracted_json_block)
            self.logger.debug(f"Normalized JSON string: {normalized_json_str[:200]}...")

            # Parse the normalized content as JSON
            try:
                parsed_json_data = json.loads(normalized_json_str)
                self.logger.info("Successfully parsed normalized JSON content.")
            except json.JSONDecodeError as e:
                self.logger.error(f"Failed to decode JSON after normalization: {e}. Problematic content snippet: {normalized_json_str[:200]}...")
                # Provide more context in the error message for debugging
                raise ValueError(f"Malformed JSON received from LLM even after normalization: {e}") from e

            # Validate the parsed JSON data against the LLMOutput Pydantic model
            try:
                validated_output = LLMOutput(**parsed_json_data)
                self.logger.info("LLM output successfully validated against schema.")
                # Return as a dictionary for broader compatibility, as the Pydantic model itself is validated
                return validated_output.dict(by_alias=True)
            except ValidationError as e:
                self.logger.error(f"Pydantic validation failed: {e}. Problematic data snippet: {parsed_json_data}")
                # Raise a specific error indicating schema violation
                raise ValueError(f"LLM output schema validation failed: {e}") from e

        except ValueError as e:
            # Catch ValueErrors raised from JSON parsing, Pydantic validation, or structural checks
            self.logger.error(f"Error during LLM output processing: {e}")
            # Re-raise to allow higher levels to handle the failure
            raise
        except Exception as e:
            # Catch any other unexpected errors during the process
            self.logger.exception(f"An unexpected error occurred in parse_and_validate: {e}")
            # Provide a generic but informative error message for unexpected issues
            raise RuntimeError(f"An unexpected error occurred while processing LLM output: {e}") from e

    def normalize_json_string(self, json_string: str) -> str:
        """Attempts to normalize common LLM-generated JSON formatting issues."""
        normalized = json_string

        # 1. Escape newline characters (\n) within string values to be valid JSON.
        # This regex looks for a newline that is NOT already preceded by a backslash.
        # We replace it with \\n. The lookbehind `(?<!\\)` ensures we don't escape already escaped newlines.
        normalized = re.sub(r'(?<!\\)\n', r'\\n', normalized)
        
        # 2. Replace common incorrect quote escaping with correct \"
        # This regex looks for a quote that might be escaped incorrectly or is simply unescaped.
        # It aims to replace sequences like `\"` or `\\\"` within string literals with `\\\\\\\"`.
        # The lookbehind `(?<!\\\\)` ensures we don't escape already escaped quotes.
        # The replacement `\\\\\\\"` correctly becomes `\"` in the final JSON string.
        normalized = re.sub(r'(?<!\\\\)\"', r'\\\\\\\"', normalized)
        
        # 3. Remove extraneous whitespace around colons and commas for compactness
        normalized = re.sub(r'\s*:\s*', ':', normalized)
        normalized = re.sub(r'\s*,\s*', ',', normalized)

        # 4. Handle trailing commas in objects and arrays (common LLM error)
        # Remove comma before closing brace '}'
        normalized = re.sub(r',\s*}', '}', normalized)
        # Remove comma before closing bracket ']'
        normalized = re.sub(r',\s*]', ']', normalized)
        
        # Note: Fixing missing commas between elements or unquoted keys with regex is highly fragile
        # and prone to errors, so we defer those to the LLM prompt and rely on Pydantic for strict validation.

        return normalized

    # The validate_code_output_batch function is not provided, 
    # but its integration point would be here or called after parse_and_validate.
    # Enhancements would focus on robust error handling, type checking, and potentially 
    # batching logic for efficiency if applicable.
    def validate_code_output_batch(self, outputs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Placeholder for batch validation logic. 
        Actual implementation needs to be provided.
        Focus on robust error handling and type checking.
        """
        self.logger.info(f"Starting batch validation for {len(outputs)} outputs.")
        # Example: Iterate and apply individual validation or batch checks
        validated_outputs = []
        for i, output in enumerate(outputs):
            try:
                # Assuming each 'output' is already parsed and validated by parse_and_validate
                # This function might perform additional checks on the list of CodeChanges, etc.
                # For now, we'll just pass through validated outputs.
                # A real implementation might re-validate or perform cross-output checks.
                validated_outputs.append(output)
                self.logger.debug(f"Output {i} passed batch validation.")
            except Exception as e:
                self.logger.error(f"Batch validation failed for output {i}: {e}")
                # Decide on a strategy: skip, log, or raise?
                # For now, we log and skip.
        self.logger.info(f"Batch validation completed. {len(validated_outputs)} outputs passed.")
        return validated_outputs