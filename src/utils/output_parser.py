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

    def _extract_json_from_markdown(self, text: str) -> Optional[str]:
        """
        Extracts the outermost valid JSON object or array from markdown code blocks.
        Handles various markdown formats and potential LLM quirks like missing closing fences.
        """
        self.logger.debug("Attempting to extract JSON from markdown code blocks...")

        # Patterns to match JSON within markdown code blocks.
        patterns = [
            r'```json\s*({.*?})\s*```',  # Standard ```json block
            r'```\s*({.*?})\s*```',      # Standard ``` block
            r'```json\s*(\[.*?\])\s*```', # Standard ```json array block
            r'```\s*(\[.*?\])\s*```',      # Standard ``` array block
            r'```json\s*({.*?})',         # ```json block without closing fence
            r'```\s*({.*?})',             # ``` block without closing fence
            r'({.*?})\s*```',             # Block ending with ```
            r'```json\s*(\[.*?\])',       # ```json array block without closing fence
            r'```\s*(\[.*?\])',           # ``` array block without closing fence
            r'(\[.*?\])\s*```'            # Array block ending with ```
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text, re.DOTALL | re.MULTILINE)
            if match:
                json_str = match.group(1).strip()
                self.logger.debug(f"Extracted potential JSON string: {json_str[:100]}...")
                
                # Basic sanitization: remove trailing commas before closing braces/brackets
                json_str = re.sub(r',\s*([\}\]])', r'\1', json_str)
                
                try:
                    json.loads(json_str)
                    self.logger.debug("Successfully extracted and validated JSON block.")
                    return json_str
                except json.JSONDecodeError:
                    self.logger.debug("Extracted string is not valid JSON, trying next pattern.")
                    continue
        
        self.logger.debug("No valid JSON block found in markdown code blocks.")
        return None

    def _extract_and_sanitize_json_string(self, text: str) -> Optional[str]:
        """
        Attempts to extract the outermost valid JSON object or array from text.
        This method refines the LLM's own suggested approach for more robust extraction,
        handling nested delimiters and basic sanitization.
        """
        self.logger.debug("Attempting robust JSON extraction and sanitization...")

        # 1. Remove markdown code block fences if present
        text_cleaned = re.sub(r'```(?:json)?\s*', '', text, flags=re.MULTILINE)
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
            expected_closers_stack = [] 

            if text_cleaned[start_index] == '{':
                balance = 1
                expected_closers_stack.append('}')
            elif text_cleaned[start_index] == '[':
                balance = 1
                expected_closers_stack.append(']')
            else:
                continue 

            end_index = -1
            for i in range(start_index + 1, len(text_cleaned)):
                char = text_cleaned[i]

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
                        balance = -999 # Mismatched closer
                        break
                elif char == ']':
                    balance -= 1
                    if expected_closers_stack and expected_closers_stack[-1] == ']':
                        expected_closers_stack.pop()
                    else:
                        balance = -999 # Mismatched closer
                        break
                
                if balance == 0 and not expected_closers_stack:
                    end_index = i + 1
                    potential_json_str = text_cleaned[start_index:end_index]
                    try:
                        json.loads(potential_json_str)
                        self.logger.debug(f"Successfully extracted valid JSON block: {potential_json_str[:100]}...")
                        
                        # Apply basic sanitization: remove trailing commas before closing braces/brackets
                        potential_json_str = re.sub(r',\s*([\}\]])', r'\1', potential_json_str)
                        return potential_json_str.strip()
                    except json.JSONDecodeError:
                        self.logger.debug("Extracted block is not valid JSON, continuing search.")
                        continue 
            
        self.logger.debug("Failed to extract a valid JSON block after all attempts.")
        return None

    def parse_and_validate(self, raw_output: str, schema_model: Type[BaseModel]) -> Dict[str, Any]:
        """
        Parse and validate the raw LLM output against a given Pydantic schema.
        Returns a dictionary representation of the validated model, or a dictionary
        containing 'malformed_blocks' if parsing/validation fails.
        """
        self.logger.debug(f"Attempting to parse raw output: {raw_output[:500]}...")

        malformed_blocks_list = []

        # First, try extracting JSON from markdown code blocks using the new helper
        extracted_json_str = self._extract_json_from_markdown(raw_output)
        
        # If not found in markdown, try the more general extraction
        if not extracted_json_str:
            extracted_json_str = self._extract_and_sanitize_json_string(raw_output)
        
        parsed_data = {}
        if not extracted_json_str:
            self.logger.error("Failed to extract any JSON structure from the output.")
            malformed_blocks_list.append({
                "type": "JSON_EXTRACTION_FAILED",
                "message": "Could not find or extract a valid JSON structure from the output.",
                "raw_string_snippet": raw_output[:1000] + ("..." if len(raw_output) > 1000 else "")
            })
            return {
                "COMMIT_MESSAGE": "Parsing error",
                "RATIONALE": f"Failed to parse LLM output as JSON. Error: Could not extract valid JSON structure.\nAttempted raw output: {raw_output[:500]}...",
                "CODE_CHANGES": [],
                "malformed_blocks": malformed_blocks_list
            }

        try:
            parsed_data = json.loads(extracted_json_str)
        except json.JSONDecodeError as e:
            self.logger.error(f"JSON decoding failed after extraction: {e}")
            malformed_blocks_list.append({
                "type": "JSON_DECODE_ERROR",
                "message": str(e),
                "raw_string_snippet": extracted_json_str[:1000] + ("..." if len(extracted_json_str) > 1000 else "")
            })
            return {
                "COMMIT_MESSAGE": "Parsing error",
                "RATIONALE": f"Failed to parse LLM output as JSON. Error: {e}\nAttempted JSON string: {extracted_json_str[:500]}...",
                "CODE_CHANGES": [],
                "malformed_blocks": malformed_blocks_list
            }

        # Validate against the provided schema_model
        try:
            validated_output = schema_model(**parsed_data)
            self.logger.info(f"LLM output successfully validated against {schema_model.__name__} schema.")
            result_dict = validated_output.model_dump(by_alias=True)
            result_dict["malformed_blocks"] = malformed_blocks_list # Include any extraction errors
            return result_dict
        except ValidationError as validation_e:
            self.logger.error(f"Schema validation failed for {schema_model.__name__}: {validation_e}")
            malformed_blocks_list.append({
                "type": "SCHEMA_VALIDATION_ERROR",
                "message": str(validation_e),
                "raw_string_snippet": raw_output
            })
            
            # --- ENHANCED FALLBACK LOGIC ---
            # Attempt to salvage partial data for LLMOutput if it was the target schema
            fallback_output = {
                "COMMIT_MESSAGE": "Schema validation failed",
                "RATIONALE": f"Original output: {raw_output[:500]}...\nValidation Error: {str(validation_e)}",
                "CODE_CHANGES": [],
                "malformed_blocks": malformed_blocks_list
            }
            
            if schema_model == LLMOutput and isinstance(parsed_data, dict):
                fallback_output["COMMIT_MESSAGE"] = parsed_data.get("COMMIT_MESSAGE", fallback_output["COMMIT_MESSAGE"])
                if "RATIONALE" in parsed_data:
                    fallback_output["RATIONALE"] = parsed_data["RATIONALE"]
                
                original_code_changes = parsed_data.get("CODE_CHANGES")
                processed_code_changes = []
                if isinstance(original_code_changes, list):
                    for index, item in enumerate(original_code_changes):
                        if isinstance(item, dict):
                            try:
                                valid_item = CodeChange(**item)
                                processed_code_changes.append(valid_item.model_dump(by_alias=True))
                            except ValidationError as inner_val_e:
                                self.logger.warning(f"Fallback: Malformed dictionary item in CODE_CHANGES at index {index} skipped. Error: {inner_val_e}")
                                malformed_blocks_list.append({
                                    "type": "MALFORMED_CODE_CHANGE_ITEM", "index": index, "message": str(inner_val_e), "raw_item": str(item)
                                })
                                # Add a placeholder for the malformed item
                                processed_code_changes.append({
                                    "FILE_PATH": f"malformed_entry_{index}", "ACTION": "ADD", 
                                    "FULL_CONTENT": f"LLM provided a malformed dictionary entry in CODE_CHANGES at index {index}. Validation error: {inner_val_e}", "LINES": []
                                })
                        else:
                            self.logger.warning(f"Fallback: Non-dictionary item in CODE_CHANGES at index {index} skipped.")
                            malformed_blocks_list.append({
                                "type": "NON_DICT_CODE_CHANGE_ITEM", "index": index, "message": "Item is not a dictionary.", "raw_item": str(item)
                            })
                            # Add a placeholder for the malformed item
                            processed_code_changes.append({
                                "FILE_PATH": f"malformed_entry_{index}", "ACTION": "ADD", 
                                "FULL_CONTENT": f"LLM provided a non-dictionary item in CODE_CHANGES at index {index}: {item}", "LINES": []
                            })
                    fallback_output["CODE_CHANGES"] = processed_code_changes
                else:
                    self.logger.warning(f"Fallback: CODE_CHANGES field was not a list or was missing.")
                    malformed_blocks_list.append({
                        "type": "MALFORMED_CODE_CHANGES_FIELD", "message": "CODE_CHANGES field was not a list or was missing.", "raw_value": str(original_code_changes)
                    })
                    fallback_output["CODE_CHANGES"] = []
            
            return fallback_output
        except Exception as general_e:
            self.logger.error(f"An unexpected error occurred during schema validation: {general_e}")
            malformed_blocks_list.append({
                "type": "UNEXPECTED_VALIDATION_ERROR",
                "message": str(general_e),
                "raw_string_snippet": raw_output
            })
            return {
                "COMMIT_MESSAGE": "Unexpected validation error",
                "RATIONALE": f"An unexpected error occurred during schema validation: {general_e}\nRaw output: {raw_output[:500]}...",
                "CODE_CHANGES": [],
                "malformed_blocks": malformed_blocks_list
            }