# src/utils/output_parser.py
import json
import logging
import re
from typing import Dict, Any, List
from pathlib import Path
from pydantic import BaseModel, Field, validator, model_validator

# Import the centralized path utility functions
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
                # Return a structured error indicating the problem
                return {"malformed_blocks": ["Error: No JSON content found or extracted from LLM output."]}

            # --- NEW CHECK ---
            # Ensure the extracted block looks like a JSON object or array before proceeding.
            # This helps filter out plain text that might have been mistakenly captured,
            # preventing errors in normalization or json.loads.
            cleaned_block_for_check = extracted_json_block.strip()
            if not (cleaned_block_for_check.startswith('{') and cleaned_block_for_check.endswith('}')) and \
               not (cleaned_block_for_check.startswith('[') and cleaned_block_for_check.endswith(']')):
                self.logger.error(f"Extracted content does not appear to be valid JSON structure (missing {{}} or []). Content: {cleaned_block_for_check[:200]}...")
                # Return a structured error indicating the structural issue
                return {"malformed_blocks": [f"Error: Extracted content does not appear to be valid JSON structure. Content snippet: {cleaned_block_for_check[:200]}..."]}
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
                # Return a structured error indicating JSON decoding failure
                return {"malformed_blocks": [f"Error: Malformed JSON received from LLM even after normalization: {e}. Snippet: {normalized_json_str[:200]}..."]}

            # Validate the parsed JSON data against the LLMOutput Pydantic model
            try:
                validated_output = LLMOutput(**parsed_json_data)
                self.logger.info("LLM output successfully validated against schema.")
                # Return as a dictionary for broader compatibility, as the Pydantic model itself is validated
                return validated_output.dict(by_alias=True)
            except ValidationError as e:
                self.logger.error(f"Pydantic validation failed: {e}. Problematic data snippet: {parsed_json_data}")
                # Extract detailed error information for better debugging
                detailed_errors = []
                for error in e.errors():
                    loc = " -> ".join(map(str, error['loc']))
                    detailed_errors.append(f"Field '{loc}': {error['msg']} (Type: {error['type']})")

                error_message = "LLM output schema validation failed:\n" + "\n".join(detailed_errors)
                # Return a structured error indicating schema validation failure
                return {"malformed_blocks": [error_message]}

        except ValueError as e: # Catch ValueErrors from initial checks
            self.logger.error(f"Error during LLM output processing (ValueError): {e}")
            return {"malformed_blocks": [f"Error: {e}"]}
        except Exception as e: # Catch any other unexpected errors during the process
            self.logger.exception(f"An unexpected error occurred in parse_and_validate: {e}")
            # Return a structured error for unexpected issues
            return {"malformed_blocks": [f"An unexpected error occurred while processing LLM output: {e}"]}

    def normalize_json_string(self, json_string: str) -> str:
        """Attempts to normalize common LLM-generated JSON formatting issues."""
        normalized = json_string

        # 1. Escape newline characters (\n) within string values to be valid JSON.
        # This regex looks for a newline that is NOT already preceded by a backslash.
        # We replace it with \\n. The lookbehind `(?<!\\)` ensures we don't escape already escaped newlines.
        normalized = re.sub(r'(?<!\\)\n', r'\\n', normalized)

        # 2. Escape double quotes within string values to be valid JSON.
        # This regex looks for a double quote that is NOT already preceded by a backslash.
        # We replace it with \\\\\". The lookbehind `(?<!\\\\)` ensures we don't escape already escaped quotes.
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

    def validate_code_output_batch(self, outputs: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
        """Validates a batch of code changes and aggregates issues per file.
        This function is called after parse_and_validate has succeeded.
        """
        self.logger.info(f"Starting batch validation for {len(outputs)} code change entries.")
        all_validation_results = {}
        for i, change_entry in enumerate(outputs):
            file_path = change_entry.get('file_path')
            if file_path:
                try:
                    # validate_code_output expects a single change dict and original content (if available)
                    # We pass original_contents which maps file_path to its content.
                    original_content = None # Placeholder: original_contents are not directly passed to this parser method.
                    # If original_contents were needed for REMOVE/MODIFY checks, they'd need to be passed here.
                    # For now, we rely on the validator's internal checks or assume it doesn't need original content.
                    validation_result = validate_code_output(change_entry, original_content=original_content)

                    # Store issues per file path
                    all_validation_results[file_path] = validation_result.get('issues', [])
                    self.logger.debug(f"Validation for {file_path} completed with {len(validation_result.get('issues', []))} issues.")
                except Exception as e:
                    self.logger.error(f"Error during validation of change entry {i} for file {file_path}: {e}")
                    # Add an error issue if validation itself fails
                    if file_path not in all_validation_results:
                        all_validation_results[file_path] = []
                    all_validation_results[file_path].append({'type': 'Validation Tool Error', 'file': file_path, 'message': f'Failed to validate: {e}'})
            else:
                # Handle changes without file_path if necessary
                self.logger.warning(f"Encountered a code change without a 'file_path' in output {i}. Skipping validation for this item.")
                # Optionally, add a generic issue for the batch if such items are critical.
                # all_validation_results['<unspecified_file>'] = [{'type': 'Validation Error', 'message': 'Change item missing file_path'}]

        self.logger.info(f"Batch validation completed. Aggregated issues for {len(all_validation_results)} files.")
        return all_validation_results