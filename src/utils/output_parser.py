# src/utils/output_parser.py

import json
import logging
import re
import sys
import traceback
from typing import Dict, Any, List, Optional, Type, Tuple
from pathlib import Path
from pydantic import BaseModel, ValidationError

# Ensure all relevant models are imported
from src.models import CodeChange, LLMOutput, ContextAnalysisOutput, CritiqueOutput, GeneralOutput, ConflictReport, SelfImprovementAnalysisOutput

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
        # Use non-greedy matching to avoid capturing too much
        json_match_within_markers = re.search(r'(\{.*?\}|\[.*?\])', json_content_raw, re.DOTALL)
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

        # Use non-greedy matching for the content within the code block
        patterns = [
            r'```json\s*(\{.*?})\s*```',
            r'```\s*(\{.*?})\s*```',
            r'```json\s*(\[.*?])\s*```',
            r'```\s*(\[.*?])\s*```',
            r'```json\s*(\{.*?})', # Missing closing fence
            r'```\s*(\{.*?})',     # Missing closing fence
            r'(\{.*?})\s*```',     # Missing opening fence
            r'```json\s*(\[.*?])', # Missing closing fence
            r'```\s*(\[.*?])',     # Missing closing fence
            r'(\[.*?])\s*```'      # Missing opening fence
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
        Uses a robust, stack-based approach to handle nested delimiters.
        """
        self.logger.debug("Attempting robust JSON extraction and sanitization...")

        # Remove markdown code block fences and common conversational filler
        text_cleaned = re.sub(r'```(?:json|python|text)?\s*', '', text, flags=re.MULTILINE)
        text_cleaned = re.sub(r'\s*```', '', text_cleaned, flags=re.MULTILINE)
        # Remove common leading/trailing conversational filler
        text_cleaned = re.sub(r'^(?:Here is the JSON output|```json|```|```python|```text|```).*?\n', '', text_cleaned, flags=re.DOTALL | re.IGNORECASE)
        text_cleaned = re.sub(r'\n(?:```|```json|```python|```text|```).*?$', '', text_cleaned, flags=re.DOTALL | re.IGNORECASE)
        text_cleaned = text_cleaned.strip()

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
            else:
                continue 

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
                        # Mismatched closer or unbalanced, break and try next start_index
                        balance = -999 
                        break
                elif char == ']':
                    balance -= 1
                    if expected_closers_stack and expected_closers_stack[-1] == ']':
                        expected_closers_stack.pop()
                    else:
                        # Mismatched closer or unbalanced, break and try next start_index
                        balance = -999 
                        break
                
                if balance == 0 and not expected_closers_stack:
                    potential_json_str = text_cleaned[start_index:i+1]
                    
                    # Attempt to parse to confirm validity
                    try:
                        json.loads(potential_json_str)
                        self.logger.debug(f"Successfully extracted valid JSON block: {potential_json_str[:100]}...")
                        return potential_json_str.strip()
                    except json.JSONDecodeError:
                        self.logger.debug("Extracted block is not valid JSON, continuing search.")
                        continue 
            
            self.logger.debug("Failed to extract a valid JSON block after all attempts.")
            return None

    def _repair_json_string(self, json_str: str) -> Tuple[str, List[str]]:
        """Applies common JSON repair heuristics and logs repairs."""
        repair_log = []
        
        # 1. Remove trailing commas before closing braces/brackets
        original_json_str = json_str
        json_str = re.sub(r',\s*([\}\]])', r'\1', json_str)
        if original_json_str != json_str:
            repair_log.append("Removed trailing commas.")
        
        # 2. Replace single quotes with double quotes (careful not to break escaped quotes)
        # This is a complex problem for regex. A simpler, less aggressive approach:
        # Only replace single quotes that are likely delimiters, not within values.
        # This regex attempts to target single quotes that are not part of a word.
        original_json_str = json_str
        json_str = re.sub(r"(?<![a-zA-Z0-9_])'(?![a-zA-Z0-9_])", '"', json_str)
        if original_json_str != json_str:
            repair_log.append("Replaced likely single quotes with double quotes.")

        # 3. Handle unquoted keys (simple heuristic, might fail on complex cases)
        original_json_str = json_str
        json_str = re.sub(r'([{,]\s*)([a-zA-Z_][a-zA-Z0-9_]*)\s*:', r'\1"\2":', json_str)
        if original_json_str != json_str:
            repair_log.append("Added quotes to unquoted keys.")

        # 4. Attempt to balance braces and brackets (add missing closers at the end)
        open_braces = json_str.count('{')
        close_braces = json_str.count('}')
        if open_braces > close_braces:
            json_str += '}' * (open_braces - close_braces)
            repair_log.append(f"Added {open_braces - close_braces} missing closing braces.")
        
        open_brackets = json_str.count('[')
        close_brackets = json_str.count(']')
        if open_brackets > close_brackets:
            json_str += ']' * (open_brackets - close_brackets)
            repair_log.append(f"Added {open_brackets - close_brackets} missing closing brackets.")

        return json_str, repair_log

    def _parse_with_incremental_repair(self, json_str: str) -> Tuple[Optional[Dict[str, Any]], List[Dict[str, str]]]:
        """Attempts to parse JSON with incremental repair strategies."""
        repair_log = []
        
        # Attempt 1: Standard repair
        repaired_text, current_repairs = self._repair_json_string(json_str)
        repair_log.extend([{"action": "initial_repair", "details": r} for r in current_repairs])
        try:
            result = json.loads(repaired_text)
            return result, repair_log
        except json.JSONDecodeError:
            pass # Continue to next attempt

        # Attempt 2: Extract largest valid sub-object (DeepSeek's idea)
        largest_sub_object_str = self._extract_largest_valid_subobject(json_str)
        if largest_sub_object_str:
            repair_log.append({"action": "extracted_largest_subobject", "details": "Attempting to parse largest valid JSON fragment."})
            try:
                result = json.loads(largest_sub_object_str)
                return result, repair_log
            except json.JSONDecodeError:
                pass

        # Attempt 3: Treat as JSON lines (DeepSeek's idea)
        json_lines_str = self._convert_to_json_lines(json_str)
        if json_lines_str != json_str: # Only if conversion actually happened
            repair_log.append({"action": "converted_to_json_lines", "details": "Attempting to parse as JSON lines."})
            try:
                result = json.loads(json_lines_str)
                return result, repair_log
            except json.JSONDecodeError:
                pass

        return None, repair_log

    def _extract_largest_valid_subobject(self, json_str: str) -> str:
        """Extracts the largest potentially valid JSON object or array from malformed text."""
        # Use non-greedy matching for the content within the braces/brackets
        matches = list(re.finditer(r'(\{.*?\}|\[.*?\])', json_str, re.DOTALL))
        
        longest_valid_match = ""
        for match in matches:
            potential_json = match.group(0)
            try:
                json.loads(potential_json)
                if len(potential_json) > len(longest_valid_match):
                    longest_valid_match = potential_json
            except json.JSONDecodeError:
                continue
        return longest_valid_match

    def _convert_to_json_lines(self, json_str: str) -> str:
        """Converts potential JSON lines format to array format."""
        lines = json_str.strip().split('\n')
        json_objects = []
        
        for line in lines:
            line = line.strip()
            if line.startswith('{') and line.endswith('}'):
                try:
                    json.loads(line)  # Validate it's parseable
                    json_objects.append(line)
                except json.JSONDecodeError:
                    continue
                    
        if json_objects:
            return f'[{",".join(json_objects)}]'
            
        return json_str # Return original if no valid JSON lines found

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
        extracted_json_str = self._extract_json_with_markers(raw_output)
        if not extracted_json_str:
            extracted_json_str = self._extract_json_from_markdown(raw_output)
        if not extracted_json_str:
            extracted_json_str = self._extract_and_sanitize_json_string(raw_output)
        
        if not extracted_json_str:
            malformed_blocks_list.append({
                "type": "JSON_EXTRACTION_FAILED",
                "message": "Could not find or extract a valid JSON structure from the output.",
                "raw_string_snippet": raw_output[:1000] + ("..." if len(raw_output) > 1000 else "")
            })
            return self._create_fallback_output(schema_model, malformed_blocks_list, raw_output)

        # 2. Parse JSON string with incremental repair attempts
        parsed_data, repair_log = self._parse_with_incremental_repair(extracted_json_str)
        if repair_log:
            malformed_blocks_list.append({
                "type": "JSON_REPAIR_ATTEMPTED",
                "message": "JSON repair heuristics applied.",
                "details": repair_log,
                "raw_string_snippet": extracted_json_str[:1000] + ("..." if len(extracted_json_str) > 1000 else "")
            })

        if parsed_data is None:
            malformed_blocks_list.append({
                "type": "JSON_DECODE_ERROR",
                "message": "Failed to decode JSON even after repair attempts.",
                "raw_string_snippet": extracted_json_str[:1000] + ("..." if len(extracted_json_str) > 1000 else "")
            })
            return self._create_fallback_output(schema_model, malformed_blocks_list, raw_output)

        # 3. Handle cases where JSON is a list
        if isinstance(parsed_data, list):
            if not parsed_data:
                malformed_blocks_list.append({"type": "EMPTY_JSON_LIST", "message": "The LLM returned an empty JSON list."})
                return self._create_fallback_output(schema_model, malformed_blocks_list, raw_output)
            data_to_validate = parsed_data[0]
        elif isinstance(parsed_data, dict):
            data_to_validate = parsed_data
        else:
            malformed_blocks_list.append({"type": "INVALID_JSON_STRUCTURE", "message": f"Expected JSON object or array, but got {type(parsed_data).__name__}."})
            return self._create_fallback_output(schema_model, malformed_blocks_list, raw_output)

        # Handle raw CodeChange output when LLMOutput is expected
        if schema_model == LLMOutput and isinstance(data_to_validate, dict):
            is_code_change_like = all(k in data_to_validate for k in ["FILE_PATH", "ACTION"]) and \
                                  ("FULL_CONTENT" in data_to_validate or "LINES" in data_to_validate)
            if is_code_change_like:
                self.logger.warning("LLM output was a raw CodeChange object, but LLMOutput schema was expected. Wrapping it.")
                file_name = Path(data_to_validate.get("FILE_PATH", "unknown_file")).name
                action = data_to_validate.get("ACTION", "change").lower()
                wrapped_data = {
                    "COMMIT_MESSAGE": f"Feat: {action.capitalize()} {file_name}",
                    "RATIONALE": f"The LLM generated a direct code change for {file_name} as part of the solution. This was wrapped into the expected LLMOutput format.",
                    "CODE_CHANGES": [data_to_validate],
                    "malformed_blocks": malformed_blocks_list
                }
                data_to_validate = wrapped_data

        # 4. Validate against schema
        try:
            validated_output = schema_model(**data_to_validate)
            result_dict = validated_output.model_dump(by_alias=True)
            result_dict["malformed_blocks"] = malformed_blocks_list
            return result_dict
        except ValidationError as validation_e:
            malformed_blocks_list.append({
                "type": "SCHEMA_VALIDATION_ERROR",
                "message": str(validation_e),
                "raw_string_snippet": extracted_json_str[:1000] + ("..." if len(extracted_json_str) > 1000 else "")
            })
            return self._create_fallback_output(schema_model, malformed_blocks_list, raw_output, data_to_validate)
        except Exception as general_e:
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
            fallback_output["CONFLICT_RESOLUTION"] = partial_data.get("CONFLICT_RESOLUTION")
            fallback_output["UNRESOLVED_CONFLICT"] = partial_data.get("UNRESOLVED_CONFLICT")
        elif schema_model == CritiqueOutput:
            fallback_output["CRITIQUE_SUMMARY"] = partial_data.get("CRITIQUE_SUMMARY", "Critique output malformed.")
            fallback_output["CRITIQUE_POINTS"] = partial_data.get("CRITIQUE_POINTS", [])
            fallback_output["SUGGESTIONS"] = partial_data.get("SUGGESTIONS", [])
        elif schema_model == ContextAnalysisOutput:
            fallback_output["key_modules"] = partial_data.get("key_modules", [])
            fallback_output["security_concerns"] = partial_data.get("security_concerns", [])
            fallback_output["architectural_patterns"] = partial_data.get("architectural_patterns", [])
            fallback_output["performance_bottlenecks"] = partial_data.get("performance_bottlenecks", [])
            fallback_output["security_summary"] = partial_data.get("security_summary", {})
            fallback_output["architecture_summary"] = partial_data.get("architecture_summary", {})
            fallback_output["devops_summary"] = partial_data.get("devops_summary", {})
            fallback_output["testing_summary"] = partial_data.get("testing_summary", {})
            fallback_output["general_overview"] = partial_data.get("general_overview", "")
        elif schema_model == GeneralOutput:
            fallback_output["general_output"] = partial_data.get("general_output", "General output malformed.")
        elif schema_model == ConflictReport:
            fallback_output["conflict_type"] = partial_data.get("conflict_type", "UNKNOWN")
            fallback_output["summary"] = partial_data.get("summary", "Conflict report malformed.")
            fallback_output["involved_personas"] = partial_data.get("involved_personas", [])
            fallback_output["conflicting_outputs_snippet"] = partial_data.get("conflicting_outputs_snippet", "")
            fallback_output["proposed_resolution_paths"] = partial_data.get("proposed_resolution_paths", [])
        elif schema_model == SelfImprovementAnalysisOutput:
            fallback_output["ANALYSIS_SUMMARY"] = partial_data.get("ANALYSIS_SUMMARY", "Self-improvement analysis malformed.")
            fallback_output["IMPACTFUL_SUGGESTIONS"] = partial_data.get("IMPACTFUL_SUGGESTIONS", [])
        
        return fallback_output