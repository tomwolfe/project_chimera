# src/utils/output_parser.py

import json
import logging
import re
import sys
import traceback
from typing import Dict, Any, List, Optional, Type
from pathlib import Path # Import Path for file_name extraction
from pydantic import BaseModel, ValidationError # Keep Pydantic ValidationError
from jsonschema import validate, ValidationError as JsonSchemaValidationError # Import jsonschema's ValidationError

# --- MODIFICATION FOR IMPROVEMENT 4.3 ---
# Import models from src.models
# Ensure GeneralOutput is imported here
from src.models import CodeChange, LLMOutput, ContextAnalysisOutput, CritiqueOutput, GeneralOutput
# --- END MODIFICATION ---

logger = logging.getLogger(__name__)

class LLMOutputParser:
    def __init__(self):
        self.logger = logger

    def _extract_json_with_markers(self, text: str, start_marker: str = "START_JSON_OUTPUT", end_marker: str = "END_JSON_OUTPUT") -> Optional[str]:
        """
        Extracts JSON content explicitly delimited by start and end markers.
        """
        self.logger.debug(f"Attempting to extract JSON using markers '{start_marker}' and '{end_marker}'...")
        start_match = re.search(re.escape(start_marker), text)
        if not start_match:
            self.logger.debug("Start marker not found.")
            return None

        # Search for the end marker *after* the start marker
        end_match = re.search(re.escape(end_marker), text[start_match.end():])
        if not end_match:
            self.logger.debug("End marker not found after start marker.")
            return None
        
        # Extract the content between the end of the start marker and the start of the end marker
        json_content_raw = text[start_match.end() : start_match.end() + end_match.start()].strip()
        
        # Attempt to find the actual JSON object/array within the raw content
        json_match_within_markers = re.search(r'(\{.*\}|\[.*\])', json_content_raw, re.DOTALL)
        if json_match_within_markers:
            json_str = json_match_within_markers.group(0).strip()
            
            # Basic sanitization for trailing commas before closing braces/brackets
            json_str = re.sub(r',\s*([\}\]])', r'\1', json_str)
            
            try:
                json.loads(json_str) # Validate if it's parseable JSON
                self.logger.debug("Successfully extracted and validated JSON block using markers.")
                return json_str
            except json.JSONDecodeError:
                self.logger.debug("Content within markers is not valid JSON.")
                return None
        
        self.logger.debug("No JSON object/array found within the markers.")
        return None
        
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

    def _repair_json_string(self, json_str: str) -> str:
        """Applies common JSON repair heuristics."""
        # Remove trailing commas before closing braces/brackets
        json_str = re.sub(r',\s*([\}\]])', r'\1', json_str)
        # Replace single quotes with double quotes (if not escaped)
        json_str = re.sub(r"(?<!\\)'", '"', json_str)
        # Handle unquoted keys (simple heuristic, might fail on complex cases)
        json_str = re.sub(r'([{,]\s*)([a-zA-Z_][a-zA-Z0-9_]*)\s*:', r'\1"\2":', json_str)
        return json_str

    def parse_and_validate(self, raw_output: str, schema_model: Type[BaseModel]) -> Dict[str, Any]:
        """
        Parse and validate the raw LLM output against a given Pydantic schema.
        Handles JSON extraction, parsing, and schema validation.
        Ensures the returned structure is a dictionary, even if LLM output is a JSON array.
        Populates 'malformed_blocks' field on failure.
        """
        self.logger.debug(f"Attempting to parse raw output: {raw_output[:500]}...")

        malformed_blocks_list = []
        extracted_json_str = None
        parsed_data = None
        data_to_validate = None

        # 1. Extract JSON string using a layered approach
        extracted_json_str = self._extract_json_with_markers(raw_output) # Try with specific markers first
        if not extracted_json_str:
            extracted_json_str = self._extract_json_from_markdown(raw_output) # Then try markdown blocks
        if not extracted_json_str:
            extracted_json_str = self._extract_and_sanitize_json_string(raw_output) # Finally, use the robust general extractor
        
        if not extracted_json_str:
            malformed_blocks_list.append({
                "type": "JSON_EXTRACTION_FAILED",
                "message": "Could not find or extract a valid JSON structure from the output.",
                "raw_string_snippet": raw_output[:1000] + ("..." if len(raw_output) > 1000 else "")
            })
            return self._create_fallback_output(schema_model, malformed_blocks_list, raw_output)

        # 2. Parse JSON string with repair attempts
        try:
            parsed_data = json.loads(extracted_json_str)
        except json.JSONDecodeError:
            repaired_json_str = self._repair_json_string(extracted_json_str)
            try:
                parsed_data = json.loads(repaired_json_str)
                self.logger.warning("Successfully repaired JSON string during decoding.")
            except json.JSONDecodeError as e:
                malformed_blocks_list.append({
                    "type": "JSON_DECODE_ERROR",
                    "message": str(e),
                    "raw_string_snippet": extracted_json_str[:1000] + ("..." if len(extracted_json_str) > 1000 else "")
                })
                return self._create_fallback_output(schema_model, malformed_blocks_list, raw_output)

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
                return self._create_fallback_output(schema_model, malformed_blocks_list, raw_output)
            
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
            return self._create_fallback_output(schema_model, malformed_blocks_list, raw_output)

        # Ensure data_to_validate is not None (e.g., if list was empty, though handled above)
        if data_to_validate is None:
             self.logger.error("No data available for validation after JSON processing.")
             malformed_blocks_list.append({
                "type": "NO_DATA_FOR_VALIDATION",
                "message": "No valid data could be extracted for schema validation.",
                "raw_string_snippet": extracted_json_str[:1000] + ("..." if len(extracted_json_str) > 1000 else "")
            })
             return self._create_fallback_output(schema_model, malformed_blocks_list, raw_output)

        # --- START FIX: Handle raw CodeChange output when LLMOutput is expected ---
        if schema_model == LLMOutput and isinstance(data_to_validate, dict):
            # Check if it looks like a CodeChange object directly (has FILE_PATH, ACTION, and one of FULL_CONTENT/LINES)
            is_code_change_like = all(k in data_to_validate for k in ["FILE_PATH", "ACTION"]) and \
                                  ("FULL_CONTENT" in data_to_validate or "LINES" in data_to_validate)
            
            if is_code_change_like:
                self.logger.warning("LLM output was a raw CodeChange object, but LLMOutput schema was expected. Wrapping it.")
                file_name = Path(data_to_validate.get("FILE_PATH", "unknown_file")).name
                action = data_to_validate.get("ACTION", "change").lower() # Extract action for better message
                
                # Construct the expected LLMOutput structure
                wrapped_data = {
                    "COMMIT_MESSAGE": f"Feat: {action.capitalize()} {file_name}", # More specific commit message
                    "RATIONALE": f"The LLM generated a direct code change for {file_name} as part of the solution. This was wrapped into the expected LLMOutput format.",
                    "CODE_CHANGES": [data_to_validate], # Wrap the single CodeChange object in a list
                    "malformed_blocks": malformed_blocks_list # Include any existing malformed blocks
                }
                data_to_validate = wrapped_data
        # --- END FIX ---

        # 4. Validate against schema
        try:
            # data_to_validate should now be a dictionary
            validated_output = schema_model(**data_to_validate)
            self.logger.info(f"LLM output successfully validated against {schema_model.__name__} schema.")
            result_dict = validated_output.model_dump(by_alias=True)
            # Add any malformed blocks found during extraction/parsing
            result_dict["malformed_blocks"] = malformed_blocks_list 
            return result_dict
        except ValidationError as validation_e: # Pydantic ValidationError
            self.logger.error(f"Schema validation failed for {schema_model.__name__}: {validation_e}")
            malformed_blocks_list.append({
                "type": "SCHEMA_VALIDATION_ERROR",
                "message": str(validation_e),
                "raw_string_snippet": extracted_json_str[:1000] + ("..." if len(extracted_json_str) > 1000 else "") # Use extracted JSON for context
            })
            
            # Construct a fallback output dictionary based on the expected schema type
            return self._create_fallback_output(schema_model, malformed_blocks_list, raw_output, data_to_validate)
        except Exception as general_e:
            self.logger.error(f"An unexpected error occurred during schema validation: {general_e}")
            malformed_blocks_list.append({
                "type": "UNEXPECTED_VALIDATION_ERROR",
                "message": str(general_e),
                "raw_string_snippet": extracted_json_str[:1000] + ("..." if len(extracted_json_str) > 1000 else "")
            })
            return self._create_fallback_output(schema_model, malformed_blocks_list, raw_output)

    def _create_fallback_output(self, schema_model: Type[BaseModel], malformed_blocks: List[Dict[str, Any]], raw_output_snippet: str, partial_data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Creates a structured fallback output based on the schema model."""
        partial_data = partial_data or {}
        fallback_output: Dict[str, Any] = {
            "malformed_blocks": malformed_blocks,
            "error_type": "LLM_OUTPUT_MALFORMED",
            "error_message": f"LLM output could not be fully parsed or validated. Raw snippet: {raw_output_snippet[:500]}..."
        }

        # Populate schema-specific fields with defaults or partial data
        if schema_model == LLMOutput:
            fallback_output["COMMIT_MESSAGE"] = partial_data.get("COMMIT_MESSAGE", "LLM_OUTPUT_ERROR")
            fallback_output["RATIONALE"] = partial_data.get("RATIONALE", "Failed to generate valid structured output.")
            fallback_output["CODE_CHANGES"] = partial_data.get("CODE_CHANGES", [])
            fallback_output["conflict_resolution"] = partial_data.get("CONFLICT_RESOLUTION")
            fallback_output["unresolved_conflict"] = partial_data.get("UNRESOLVED_CONFLICT")
        elif schema_model == CritiqueOutput:
            fallback_output["CRITIQUE_SUMMARY"] = partial_data.get("CRITIQUE_SUMMARY", "Critique output malformed.")
            fallback_output["CRITIQUE_POINTS"] = partial_data.get("CRITIQUE_POINTS", [])
            fallback_output["SUGGESTIONS"] = partial_data.get("SUGGESTIONS", [])
        elif schema_model == ContextAnalysisOutput:
            fallback_output["key_modules"] = partial_data.get("key_modules", [])
            fallback_output["security_concerns"] = partial_data.get("security_concerns", [])
            fallback_output["architectural_patterns"] = partial_data.get("architectural_patterns", [])
            fallback_output["performance_bottlenecks"] = partial_data.get("performance_bottlenecks", [])
        elif schema_model == GeneralOutput:
            fallback_output["general_output"] = partial_data.get("general_output", "General output malformed.")
        
        return fallback_output