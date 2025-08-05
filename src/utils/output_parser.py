# src/utils/output_parser.py
import json
import logging
import re
from typing import Dict, Any, List
from pydantic import BaseModel, Field, ValidationError, validator

# Define Pydantic models for better validation
class CodeChange(BaseModel):
    file_path: str
    action: str
    full_content: str | None = Field(None, alias="full_content") # Optional for REMOVE
    lines: List[str] | None = None # Optional for ADD/MODIFY

    @validator('file_path')
    def validate_file_path(cls, v):
        # Prevent path traversal vulnerabilities
        if '..' in v or v.startswith('/') or v.startswith('\\'):
            raise ValueError("Invalid file_path: contains '..' or is an absolute path.")
        # Basic check for potentially malicious characters, though not exhaustive
        if re.search(r'[<>:"|?*]', v):
             raise ValueError("Invalid file_path: contains forbidden characters.")
        return v

    @validator('action')
    def validate_action(cls, v):
        valid_actions = ["ADD", "MODIFY", "REMOVE"]
        if v not in valid_actions:
            raise ValueError(f"Invalid action: '{v}'. Must be one of {valid_actions}.")
        return v

    @validator('full_content', always=True) # Always check if required based on action
    def check_full_content_if_needed(cls, v, values):
        action = values.get('action')
        if action in ["ADD", "MODIFY"] and not v:
            raise ValueError("full_content is required for ADD or MODIFY actions.")
        return v

    @validator('lines', always=True) # Always check if required based on action
    def check_lines_if_needed(cls, v, values):
        action = values.get('action')
        if action == "REMOVE" and not v:
            raise ValueError("lines is required for REMOVE actions.")
        if action == "REMOVE" and v is not None and not isinstance(v, list):
             raise ValueError("lines must be a list for REMOVE actions.")
        return v

class LLMOutput(BaseModel):
    commit_message: str = Field(alias="COMMIT_MESSAGE")
    rationale: str = Field(alias="RATIONALE")
    code_changes: List[CodeChange] = Field(alias="CODE_CHANGES")
    # Optional fields for conflict resolution
    conflict_resolution: str | None = Field(None, alias="CONFLICT_RESOLUTION")
    unresolved_conflict: str | None = Field(None, alias="UNRESOLVED_CONFLICT")

class LLMOutputParser:
    def __init__(self, llm_provider):
        self.llm_provider = llm_provider
        self.logger = logging.getLogger(self.__class__.__name__)

    def parse_and_validate(self, raw_output: str) -> Dict[str, Any]:
        """Parses raw LLM output, validates JSON structure using Pydantic models."""
        try:
            parsed_json = json.loads(raw_output)
            self.logger.info("Successfully parsed LLM output as JSON.")
        except json.JSONDecodeError as e:
            self.logger.error(
                f"Failed to decode JSON: {e}. Raw output snippet: {raw_output[:500]}..."
            )
            raise ValueError(f"LLM output is not valid JSON. Error: {e}") from e

        try:
            # Validate the entire structure using Pydantic
            validated_data = LLMOutput(**parsed_json)
            # Convert Pydantic model back to dict for consistency with original proposal
            return validated_data.model_dump()
        except ValidationError as e:
            self.logger.error(f"Pydantic validation failed: {e}")
            # Provide more specific error messages from Pydantic
            error_details = []
            for error in e.errors():
                field_name = ".".join(map(str, error['loc']))
                error_details.append(f"Field '{field_name}': {error['msg']}")
            raise ValueError(f"LLM output schema validation failed. Details: {'; '.join(error_details)}") from e
        except Exception as e:
            self.logger.error(f"An unexpected error occurred during LLM output validation: {e}")
            raise RuntimeError(f"Failed to process LLM output: {e}") from e