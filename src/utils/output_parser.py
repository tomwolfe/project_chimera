# src/utils/output_parser.py
import json
import logging
import re
from typing import Dict, Any, List
from pathlib import Path
from pydantic import BaseModel, Field, ValidationError, validator, model_validator

# Define a base directory for safety. Adjust this path as needed for your project structure.
# For example, if your project root is where app.py is, and src is a subdirectory.
# If BASE_PROJECT_DIR is meant to be the root of the repository, adjust accordingly.
# For now, assuming 'src' is the intended safe directory for code changes.
BASE_PROJECT_DIR = Path("./src").resolve()

class CodeChange(BaseModel):
    file_path: str
    action: str
    full_content: str | None = Field(None, alias="full_content") # Optional for REMOVE
    lines: List[str] | None = None # Optional for ADD/MODIFY

    @validator('file_path')
    def validate_file_path(cls, v):
        """Validates file path to prevent traversal and ensure it's within the project's src directory."""
        try:
            # Normalize path and resolve against a base directory
            # This prevents traversal like '../../etc/passwd'
            normalized_path = Path(v).resolve()
            
            # Ensure the path is within the allowed base directory
            # This check will raise ValueError if normalized_path is not a subdirectory of base_dir
            normalized_path.relative_to(BASE_PROJECT_DIR)

            # Basic check for potentially malicious characters. Less critical after path normalization,
            # but can still catch unusual filenames or prevent specific OS-level issues.
            if re.search(r'[<>:"|?*]', v):
                 raise ValueError("Invalid file_path: contains forbidden characters.")

            return str(normalized_path) # Return the validated and normalized path
        except ValueError as e:
            # Catch errors from relative_to or Path operations
            raise ValueError(f"Invalid file_path '{v}': {e}") from e
        except Exception as e: # Catch any other unexpected errors during path processing
            raise ValueError(f"Error validating file_path '{v}': {e}") from e

    @validator('action')
    def validate_action(cls, v):
        valid_actions = ["ADD", "MODIFY", "REMOVE"]
        if v not in valid_actions:
            raise ValueError(f"Invalid action: '{v}'. Must be one of {valid_actions}.")
        return v

class LLMOutput(BaseModel):
    commit_message: str = Field(alias="COMMIT_MESSAGE")
    rationale: str = Field(alias="RATIONALE")
    code_changes: List[CodeChange] = Field(alias="CODE_CHANGES")
    # Optional fields for conflict resolution
    conflict_resolution: str | None = Field(None, alias="CONFLICT_RESOLUTION")
    unresolved_conflict: str | None = Field(None, alias="UNRESOLVED_CONFLICT")

    @model_validator(mode='after')
    def check_code_changes_content(self) -> 'LLMOutput':
        """Validates that full_content is present for ADD/MODIFY and lines for REMOVE."""
        for change in self.code_changes:
            if change.action in ["ADD", "MODIFY"] and not change.full_content:
                raise ValueError(
                    f"full_content is required for action '{change.action}' on file '{change.file_path}'."
                )
            if change.action == "REMOVE" and not change.lines:
                raise ValueError(
                    f"lines are required for action 'REMOVE' on file '{change.file_path}'."
                )
            if change.action == "REMOVE" and change.lines is not None and not isinstance(change.lines, list):
                 raise ValueError(f"lines must be a list for action 'REMOVE' on file '{change.file_path}'.")
        return self


class LLMOutputParser:
    def __init__(self, llm_provider):
        self.llm_provider = llm_provider
        self.logger = logging.getLogger(self.__class__.__name__)

    def parse_and_validate(self, raw_output: str) -> LLMOutput: # Return type changed to Pydantic model
        """Parses raw LLM output, validates JSON structure using Pydantic models."""
        json_content_str = None
        parsed_json_data = None

        try:
            # Attempt to extract JSON from markdown code blocks first
            # The `(?s)` flag makes '.' match newlines.
            # The `.*?` makes the match non-greedy.
            match_json_block = re.search(r"```json\n(.*?)\n```|```\n(.*?)\n```", raw_output, re.DOTALL)
            
            if match_json_block:
                json_content_str = match_json_block.group(1) or match_json_block.group(2)
                self.logger.info("Extracted JSON block from markdown.")
            else:
                # If no markdown block is found, attempt to parse the entire output as JSON.
                # This is a fallback and should be logged as potentially less reliable.
                self.logger.warning("No markdown JSON block found. Attempting to parse entire output as JSON.")
                json_content_str = raw_output

            if not json_content_str or not json_content_str.strip():
                self.logger.error("No JSON content found or extracted from the LLM output.")
                raise ValueError("No JSON content found in the LLM output.")
                
            # Attempt to parse the extracted JSON string
            try:
                parsed_json_data = json.loads(json_content_str)
                self.logger.info("Successfully parsed extracted JSON string.")
            except json.JSONDecodeError as e:
                self.logger.error(
                    f"Failed to decode JSON: {e}. Extracted JSON snippet: {json_content_str[:500]}..."
                )
                # Provide a more specific error message for JSON decoding failures
                raise ValueError(f"Extracted LLM output is not valid JSON. Error: {e}") from e

            # Validate the parsed JSON structure using Pydantic
            validated_data = LLMOutput(**parsed_json_data)
            self.logger.info("Pydantic validation successful.")
            return validated_data # Return Pydantic model instance

        except ValidationError as e:
            self.logger.error(f"Pydantic validation failed: {e}")
            # Provide more specific error messages from Pydantic
            error_details = []
            for error in e.errors():
                field_name = ".".join(map(str, error['loc']))
                error_details.append(f"Field '{field_name}': {error['msg']}")
            raise ValueError(f"LLM output schema validation failed. Details: {'; '.join(error_details)}") from e
        except Exception as e: # Catch any other unexpected errors during processing
            self.logger.error(f"An unexpected error occurred during LLM output processing: {e}")
            # Re-raise as a RuntimeError for unexpected issues
            raise RuntimeError(f"Failed to process LLM output: {e}") from e

    def get_llm_response(self, prompt: str) -> str:
        """Placeholder for calling the LLM provider and handling potential retries."""
        # TODO: Implement actual LLM call with retry logic using self.llm_provider
        # Example: 
        # retries = 3
        # for i in range(retries):
        #     try:
        #         response = self.llm_provider.generate(prompt)
        #         return response
        #     except Exception as e:
        #         self.logger.warning(f"LLM call failed (attempt {i+1}/{retries}): {e}")
        #         if i == retries - 1:
        #             raise RuntimeError("LLM call failed after multiple retries.") from e
        #         time.sleep(2**i) # Exponential backoff
        self.logger.warning("LLM provider integration for response generation is not yet implemented.")
        return "{ \"COMMIT_MESSAGE\": \"Placeholder Commit\", \"RATIONALE\": \"Placeholder Rationale\", \"CODE_CHANGES\": [ { \"file_path\": \"placeholder.py\", \"action\": \"ADD\", \"full_content\": \"# Placeholder file\\nprint(\\\"Hello, World!\\\")\\n\" } ] }"