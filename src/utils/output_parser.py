# src/utils/output_parser.py

import json
import logging
import re
import sys
import traceback
from typing import Dict, Any, List, Optional, Type
from pathlib import Path
from pydantic import BaseModel, Field, validator, model_validator, ValidationError

# --- MODIFICATION FOR IMPROVEMENT 4.3 ---
# Import models from src.models
from src.models import CodeChange, LLMOutput, ContextAnalysisOutput, CritiqueOutput # Ensure CritiqueOutput is imported
# --- END MODIFICATION ---

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

        patterns = [
            r'```json\s*({.*?})\s*```',
            r'```\s*({.*?})\s*```',
            r'```json\s*(\[.*?\])\s*```',
            r'```\s*(\[.*?\])\s*```',
            r'```json\s*({.*?})',
            r'```\s*({.*?})',
            r'({.*?})\s*```',
            r'```json\s*(\[.*?\])',
            r'```\s*(\[.*?\])',
            r'(\[.*?\])\s*```'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text, re.DOTALL | re.MULTILINE)
            if match:
                json_str = match.group(1).strip()
                self.logger.debug(f"Extracted potential JSON string: {json_str[:100]}...")
                
                # Basic sanitization for trailing commas before closing braces/brackets
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

        # Remove markdown code block fences first
        text_cleaned = re.sub(r'```(?:json)?\s*', '', text, flags=re.MULTILINE)
        text_cleaned = re.sub(r'\s*```', '', text_cleaned, flags=re.MULTILINE)
        
        # Find potential start of JSON (either '{' or '[')
        potential_starts = []
        for i, char in enumerate(text_cleaned):
            if char == '{' or char == '[':
                potential_starts.append(i)
        
        if not potential_starts:
            self.logger.debug("No JSON start delimiters found.")
            return None

        # Iterate through potential start points and try to find a balanced JSON structure
        for start_index in potential_starts:
            balance = 0
            expected_closers_stack = [] 

            if text_cleaned[start_index] == '{':
                balance = 1
                expected_closers_stack.append('}')
            elif text_cleaned[start_index] == '[':
                balance = 1
                expected_closers_stack.append(']')
            else: # Should not happen due to previous check, but for safety
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
                    else: # Mismatched closer or unbalanced
                        balance = -999 # Mark as invalid
                        break
                elif char == ']':
                    balance -= 1
                    if expected_closers_stack and expected_closers_stack[-1] == ']':
                        expected_closers_stack.pop()
                    else: # Mismatched closer or unbalanced
                        balance = -999 # Mark as invalid
                        break
                
                # Check for balanced state and empty stack
                if balance == 0 and not expected_closers_stack:
                    end_index = i + 1
                    potential_json_str = text_cleaned[start_index:end_index]
                    
                    # Basic sanitization for trailing commas before closing braces/brackets
                    potential_json_str = re.sub(r',\s*([\}\]])', r'\1', potential_json_str)
                    
                    try:
                        json.loads(potential_json_str) # Validate if it's parseable JSON
                        self.logger.debug(f"Successfully extracted valid JSON block: {potential_json_str[:100]}...")
                        return potential_json_str.strip()
                    except json.JSONDecodeError:
                        self.logger.debug("Extracted block is not valid JSON, continuing search.")
                        continue # Try next potential start or pattern
            
        self.logger.debug("Failed to extract a valid JSON block after all attempts.")
        return None

    def parse_and_validate(self, raw_output: str, schema_model: Type[BaseModel]) -> Dict[str, Any]:
        """
        Parse and validate the raw LLM output against a given Pydantic schema.
        Handles JSON extraction, parsing, and schema validation.
        Ensures the returned structure is a dictionary, even if LLM output is a JSON array.
        """
        self.logger.debug(f"Attempting to parse raw output: {raw_output[:500]}...")

        malformed_blocks_list = []
        extracted_json_str = None
        parsed_data = None
        data_to_validate = None

        # 1. Extract JSON string
        extracted_json_str = self._extract_json_from_markdown(raw_output)
        if not extracted_json_str:
            extracted_json_str = self._extract_and_sanitize_json_string(raw_output)
        
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

        # 2. Parse JSON string
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

        # 3. Handle cases where JSON is a list (e.g., LLM returns an array of outputs)
        if isinstance(parsed_data, list):
            self.logger.debug(f"Parsed JSON is a list. Attempting to process the first element.")
            if not parsed_data:
                self.logger.error("Parsed JSON list is empty.")
                malformed_blocks_list.append({
                    "type": "EMPTY_JSON_LIST",
                    "message": "The LLM returned an empty JSON list.",
                    "raw_string_snippet": extracted_json_str[:1000] + ("..." if len(extracted_json_str) > 1000 else "")
                })
                return {
                    "COMMIT_MESSAGE": "Parsing error",
                    "RATIONALE": "Failed to parse LLM output: Returned an empty JSON list.",
                    "CODE_CHANGES": [],
                    "malformed_blocks": malformed_blocks_list
                }
            
            # Take the first element for validation
            data_to_validate = parsed_data[0]
            self.logger.debug(f"Using first element of the list for validation.")
        elif isinstance(parsed_data, dict):
            data_to_validate = parsed_data
        else:
            # If parsed_data is neither a list nor a dict, it's malformed.
            self.logger.error(f"Parsed JSON is neither a list nor a dictionary: {type(parsed_data).__name__}")
            malformed_blocks_list.append({
                "type": "INVALID_JSON_STRUCTURE",
                "message": f"Expected JSON object or array, but got {type(parsed_data).__name__}.",
                "raw_string_snippet": extracted_json_str[:1000] + ("..." if len(extracted_json_str) > 1000 else "")
            })
            return {
                "COMMIT_MESSAGE": "Parsing error",
                "RATIONALE": f"Failed to parse LLM output into a valid JSON structure.",
                "CODE_CHANGES": [],
                "malformed_blocks": malformed_blocks_list
            }

        # Ensure data_to_validate is not None (e.g., if list was empty, though handled above)
        if data_to_validate is None:
             self.logger.error("No data available for validation after JSON processing.")
             malformed_blocks_list.append({
                "type": "NO_DATA_FOR_VALIDATION",
                "message": "No valid data could be extracted for schema validation.",
                "raw_string_snippet": extracted_json_str[:1000] + ("..." if len(extracted_json_str) > 1000 else "")
            })
             return {
                "COMMIT_MESSAGE": "Parsing error",
                "RATIONALE": "Failed to extract valid data for validation.",
                "CODE_CHANGES": [],
                "malformed_blocks": malformed_blocks_list
            }

        # 4. Validate against schema
        try:
            # data_to_validate should now be a dictionary
            validated_output = schema_model(**data_to_validate)
            self.logger.info(f"LLM output successfully validated against {schema_model.__name__} schema.")
            result_dict = validated_output.model_dump(by_alias=True)
            # Add any malformed blocks found during extraction/parsing
            result_dict["malformed_blocks"] = malformed_blocks_list 
            return result_dict
        except ValidationError as validation_e:
            self.logger.error(f"Schema validation failed for {schema_model.__name__}: {validation_e}")
            malformed_blocks_list.append({
                "type": "SCHEMA_VALIDATION_ERROR",
                "message": str(validation_e),
                "raw_string_snippet": extracted_json_str[:1000] + ("..." if len(extracted_json_str) > 1000 else "") # Use extracted JSON for context
            })
            
            # Construct a fallback output dictionary based on the expected schema type
            fallback_output: Dict[str, Any] = {
                "error_type": "SCHEMA_VALIDATION_FAILED",
                "error_message": f"Schema validation failed for {schema_model.__name__}: {str(validation_e)}",
                "raw_llm_output_snippet": extracted_json_str[:500],
                "malformed_blocks": malformed_blocks_list
            }

            if schema_model == LLMOutput and isinstance(data_to_validate, dict):
                # Specific salvage logic for LLMOutput
                fallback_output["COMMIT_MESSAGE"] = data_to_validate.get("COMMIT_MESSAGE", "Schema validation failed")
                fallback_output["RATIONALE"] = data_to_validate.get("RATIONALE", f"Original output: {extracted_json_str[:500]}...\nValidation Error: {str(validation_e)}")
                
                original_code_changes = data_to_validate.get("CODE_CHANGES")
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
                                # Add a placeholder for the malformed item to indicate the issue
                                processed_code_changes.append({
                                    "FILE_PATH": f"malformed_entry_{index}", "ACTION": "ADD", 
                                    "FULL_CONTENT": f"LLM provided a malformed dictionary entry in CODE_CHANGES at index {index}. Validation error: {inner_val_e}", "LINES": []
                                })
                        else:
                            self.logger.warning(f"Fallback: Non-dictionary item in CODE_CHANGES at index {index} skipped.")
                            malformed_blocks_list.append({
                                "type": "NON_DICT_CODE_CHANGE_ITEM", "index": index, "message": "Item is not a dictionary.", "raw_item": str(item)
                            })
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
            elif schema_model == CritiqueOutput and isinstance(data_to_validate, dict):
                fallback_output["CRITIQUE_SUMMARY"] = data_to_validate.get("CRITIQUE_SUMMARY", "Schema validation failed for critique.")
                fallback_output["CRITIQUE_POINTS"] = data_to_validate.get("CRITIQUE_POINTS", [])
                fallback_output["SUGGESTIONS"] = data_to_validate.get("SUGGESTIONS", [])
            elif schema_model == ContextAnalysisOutput and isinstance(data_to_validate, dict):
                fallback_output["key_modules"] = data_to_validate.get("key_modules", [])
                fallback_output["security_concerns"] = data_to_validate.get("security_concerns", [])
                fallback_output["architectural_patterns"] = data_to_validate.get("architectural_patterns", [])
                fallback_output["performance_bottlenecks"] = data_to_validate.get("performance_bottlenecks", [])

            return fallback_output
        except Exception as general_e:
            self.logger.error(f"An unexpected error occurred during schema validation: {general_e}")
            malformed_blocks_list.append({
                "type": "UNEXPECTED_VALIDATION_ERROR",
                "message": str(general_e),
                "raw_string_snippet": extracted_json_str[:1000] + ("..." if len(extracted_json_str) > 1000 else "")
            })
            return {
                "COMMIT_MESSAGE": "Unexpected validation error",
                "RATIONALE": f"An unexpected error occurred during schema validation: {general_e}\nRaw output: {extracted_json_str[:500]}...",
                "CODE_CHANGES": [],
                "malformed_blocks": malformed_blocks_list
            }
        except Exception as general_e:
            self.logger.error(f"An unexpected error occurred during schema validation: {general_e}")
            malformed_blocks_list.append({
                "type": "UNEXPECTED_VALIDATION_ERROR",
                "message": str(general_e),
                "raw_string_snippet": extracted_json_str[:1000] + ("..." if len(extracted_json_str) > 1000 else "")
            })
            return {
                "COMMIT_MESSAGE": "Unexpected validation error",
                "RATIONALE": f"An unexpected error occurred during schema validation: {general_e}\nRaw output: {extracted_json_str[:500]}...",
                "CODE_CHANGES": [],
                "malformed_blocks": malformed_blocks_list
            }