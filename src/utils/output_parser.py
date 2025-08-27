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
from src.models import CodeChange, LLMOutput, ContextAnalysisOutput, CritiqueOutput, GeneralOutput, ConflictReport, SelfImprovementAnalysisOutput, SelfImprovementAnalysisOutputV1 # ADDED SelfImprovementAnalysisOutputV1

logger = logging.getLogger(__name__)

class LLMOutputParser:
    def __init__(self):
        self.logger = logger

    def _extract_json_with_markers(self, text: str, start_marker: str = "START_JSON_OUTPUT", end_marker: str = "END_JSON_OUTPUT") -> Optional[Tuple[str, bool]]:
        """
        Extracts JSON content explicitly delimited by start and end markers.
        Returns (json_string_content, end_marker_found).
        If only start_marker is found, returns (content_after_start_marker, False).
        If no valid JSON can be extracted after a missing end marker, returns None.
        """
        self.logger.debug(f"Attempting to extract JSON using markers '{start_marker}' and '{end_marker}'...")
        start_match = re.search(re.escape(start_marker), text)
        if not start_match:
            self.logger.debug("Start marker not found.")
            return None

        text_after_start_marker = text[start_match.end():]
        end_match = re.search(re.escape(end_marker), text_after_start_marker)
        
        json_content_raw = ""
        end_marker_found = False

        if end_match:
            json_content_raw = text_after_start_marker[:end_match.start()].strip()
            end_marker_found = True
            self.logger.debug("Both markers found. Extracted content between them.")
        else:
            self.logger.warning("Start marker found, but end marker is missing. Attempting to extract first outermost JSON object from content after start marker.")
            # If end marker is missing, try to robustly extract a single JSON object
            # from the content that follows the start marker.
            json_content_raw = self._extract_first_outermost_json(text_after_start_marker)
            if json_content_raw:
                self.logger.debug("Successfully extracted a single JSON object after missing end marker.")
            else:
                self.logger.warning("Could not extract a single JSON object after missing end marker.")
                return None # Return None if no single JSON object can be robustly extracted

        if json_content_raw:
            # Basic sanitization for trailing commas before closing braces/brackets
            json_content_raw = re.sub(r',\s*([\}\]])', r'\1', json_content_raw)
            return json_content_raw, end_marker_found
        
        self.logger.debug("No content found after start marker.")
        return None
        
    def _extract_first_outermost_json(self, text: str) -> Optional[str]:
        """
        Extracts the first outermost balanced JSON object or array from text.
        This is a robust, stack-based approach to handle nested delimiters.
        """
        self.logger.debug("Attempting to extract first outermost JSON block...")

        balance = 0
        expected_closers_stack = []
        start_index = -1

        for i, char in enumerate(text):
            if char == '{':
                if start_index == -1: # First opening brace
                    start_index = i
                    balance = 1
                    expected_closers_stack.append('}')
                else:
                    balance += 1
                    expected_closers_stack.append('}')
            elif char == '[':
                if start_index == -1: # First opening bracket
                    start_index = i
                    balance = 1
                    expected_closers_stack.append(']')
                else:
                    balance += 1
                    expected_closers_stack.append(']')
            elif char == '}':
                if start_index != -1:
                    balance -= 1
                    if expected_closers_stack and expected_closers_stack[-1] == '}':
                        expected_closers_stack.pop()
            elif char == ']':
                if start_index != -1:
                    balance -= 1
                    if expected_closers_stack and expected_closers_stack[-1] == ']':
                        expected_closers_stack.pop()
            
            if start_index != -1 and balance == 0 and not expected_closers_stack:
                potential_json_str = text[start_index:i+1]
                try:
                    json.loads(potential_json_str)
                    self.logger.debug(f"Successfully extracted first outermost valid JSON block: {potential_json_str[:100]}...")
                    return potential_json_str.strip()
                except json.JSONDecodeError:
                    # If it's not valid JSON despite being balanced, reset and continue
                    start_index = -1
                    balance = 0
                    expected_closers_stack = []
                    continue
        
        self.logger.debug("No first outermost valid JSON block found.")
        return None

    def _extract_json_from_markdown(self, text: str) -> Optional[str]:
        """
        Extracts content from markdown code blocks and then uses robust JSON extraction
        on that content.
        """
        self.logger.debug("Attempting to extract JSON from markdown code blocks...")

        # First, extract the raw content within any markdown code block
        # This regex is designed to capture the content between ``` fences,
        # regardless of the language specifier (json, python, etc.)
        # It also handles cases where the closing fence might be missing at the very end of the string.
        markdown_block_pattern = r'```(?:json|python|text|)\s*(.*?)(?:```|\Z)'
        
        matches = list(re.finditer(markdown_block_pattern, text, re.DOTALL | re.MULTILINE))
        
        for match in matches:
            block_content = match.group(1).strip()
            if block_content:
                self.logger.debug(f"Extracted raw content from markdown block: {block_content[:100]}...")
                
                # Now, apply the robust JSON extraction on this block content
                # This will find the first balanced and valid JSON object/array within the block
                extracted_json = self._extract_and_sanitize_json_string(block_content, remove_markdown_fences=False) # Pass False
                if extracted_json:
                    self.logger.debug("Successfully extracted and validated JSON from markdown block content.")
                    return extracted_json
        
        self.logger.debug("No valid JSON block found in markdown code blocks.")
        return None

    def _extract_and_sanitize_json_string(self, text: str, remove_markdown_fences: bool = True) -> Optional[str]:
        """
        Attempts to extract the outermost valid JSON object or array from text.
        Uses a robust, stack-based approach to handle nested delimiters.
        
        Args:
            text: The input string potentially containing JSON.
            remove_markdown_fences: If True, attempts to remove markdown fences and conversational filler.
                                    Set to False if `text` is already content from inside a markdown block.
        """
        self.logger.debug("Attempting robust JSON extraction and sanitization...")

        text_cleaned = text
        if remove_markdown_fences:
            # Remove markdown code block fences and common conversational filler
            text_cleaned = re.sub(r'```(?:json|python|text)?\s*', '', text_cleaned, flags=re.MULTILINE)
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

    def _extract_largest_valid_subobject(self, json_str: str) -> Optional[str]:
        """
        Extracts the largest potentially valid JSON object or array from malformed text.
        Prioritizes the longest valid JSON block found.
        """
        # Use non-greedy matching for the content within the braces/brackets
        matches = list(re.finditer(r'(\{.*?\}|\[.*?\])', json_str, re.DOTALL))
        
        longest_valid_match = "" 
        
        for match in matches:
            potential_json = match.group(0)
            try:
                json.loads(potential_json)
                # Always keep track of the longest valid match
                if len(potential_json) > len(longest_valid_match):
                    longest_valid_match = potential_json
            except json.JSONDecodeError:
                continue
        
        if longest_valid_match:
            return longest_valid_match
        return None # Return None if no valid block was found

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

    # NEW: Helper to clean LLM output from markdown fences and conversational filler
    def _clean_llm_output(self, raw_output: str) -> str:
        """Clean common LLM output artifacts like markdown fences and conversational filler."""
        cleaned = raw_output
        
        # Remove Markdown code blocks (```json, ```python, ```, etc.)
        # This regex is more robust, handling optional language specifiers and ensuring it's at start/end
        cleaned = re.sub(r'^\s*```(?:json|python|text)?\s*\n', '', cleaned, flags=re.MULTILINE)
        cleaned = re.sub(r'\n\s*```\s*$', '', cleaned, flags=re.MULTILINE)
        
        # Remove common leading/trailing conversational filler
        cleaned = re.sub(r'^(?:Here is the JSON output|```json|```|```python|```text|```).*?\n', '', cleaned, flags=re.DOTALL | re.IGNORECASE)
        cleaned = re.sub(r'\n(?:```|```json|```python|```text|```).*?$', '', cleaned, flags=re.DOTALL | re.IGNORECASE)
        
        # Attempt to find the outermost JSON structure and trim anything outside it
        json_start = -1
        json_end = -1
        
        # Find first '{' or '['
        for i, char in enumerate(cleaned):
            if char == '{' or char == '[':
                json_start = i
                break
        
        # Find last '}' or ']'
        for i in range(len(cleaned) - 1, -1, -1):
            if cleaned[i] == '}' or cleaned[i] == ']':
                json_end = i + 1
                break
        
        if json_start != -1 and json_end != -1 and json_end > json_start:
            cleaned = cleaned[json_start:json_end]
        
        return cleaned.strip()

    # NEW: Helper to detect if text contains what looks like a suggestion item at top level
    def _detect_potential_suggestion_item(self, text: str) -> Optional[Dict]:
        """
        Detects if the text contains what looks like a single 'IMPACTFUL_SUGGESTIONS' item
        at the top level, rather than the full SelfImprovementAnalysisOutput.
        """
        # Look for patterns that match a suggestion item's key fields
        # This is a heuristic, but targets the specific structure of an IMPACTFUL_SUGGESTIONS item
        area_match = re.search(r'"AREA"\s*:\s*"(Reasoning Quality|Robustness|Efficiency|Maintainability|Security)"', text)
        problem_match = re.search(r'"PROBLEM"\s*:\s*"[^"]+"', text)
        proposed_solution_match = re.search(r'"PROPOSED_SOLUTION"\s*:\s*"[^"]+"', text)
        
        if area_match and problem_match and proposed_solution_match:
            # Try to extract the object that contains these fields
            try:
                # Find the enclosing braces for this potential object
                start_idx = text.rfind('{', 0, area_match.start())
                if start_idx >= 0:
                    brace_count = 1
                    for i in range(start_idx + 1, len(text)):
                        if text[i] == '{':
                            brace_count += 1
                        elif text[i] == '}':
                            brace_count -= 1
                            if brace_count == 0:
                                obj_str = text[start_idx:i+1]
                                # Attempt to load as JSON to ensure it's a valid object
                                return json.loads(obj_str)
            except Exception as e:
                self.logger.debug(f"Error extracting potential suggestion item: {e}")
        
        return None

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

        # NEW: Clean raw output first to remove markdown fences and conversational filler
        cleaned_raw_output = self._clean_llm_output(raw_output)
        
        # 1. Attempt to extract JSON string using markers from the cleaned output
        marker_extraction_result = self._extract_json_with_markers(cleaned_raw_output)
        if marker_extraction_result:
            extracted_json_str, end_marker_found = marker_extraction_result
            if not end_marker_found:
                malformed_blocks_list.append({
                    "type": "MISSING_END_MARKER",
                    "message": "START_JSON_OUTPUT was found, but END_JSON_OUTPUT was missing. Attempted to parse content after START_JSON_OUTPUT.",
                    "raw_string_snippet": cleaned_raw_output[:1000] + ("..." if len(cleaned_raw_output) > 1000 else "")
                })
        else:
            # No start marker found, try markdown blocks on the full cleaned_raw_output
            extracted_json_str = self._extract_json_from_markdown(cleaned_raw_output)
            if not extracted_json_str:
                # No markdown blocks, try robust extraction on the whole cleaned_raw_output
                extracted_json_str = self._extract_and_sanitize_json_string(cleaned_raw_output, remove_markdown_fences=True)
        
        if not extracted_json_str:
            malformed_blocks_list.append({
                "type": "JSON_EXTRACTION_FAILED",
                "message": "Could not find or extract a valid JSON structure from the output.",
                "raw_string_snippet": cleaned_raw_output[:1000] + ("..." if len(cleaned_raw_output) > 1000 else "")
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
            # If the schema model is the versioned SelfImprovementAnalysisOutput,
            # we need to handle the versioning logic here.
            if schema_model == SelfImprovementAnalysisOutput:
                # First, try to validate as the versioned wrapper
                try:
                    validated_output = schema_model.model_validate(data_to_validate)
                except ValidationError:
                    # If it fails, assume it's a V1 data structure and wrap it
                    v1_data = SelfImprovementAnalysisOutputV1.model_validate(data_to_validate)
                    validated_output = SelfImprovementAnalysisOutput(
                        version="1.0",
                        data=v1_data.model_dump(by_alias=True)
                    )
            else:
                validated_output = schema_model.model_validate(data_to_validate)
            
            result_dict = validated_output.model_dump(by_alias=True)
            # Ensure malformed_blocks from parsing are added to the final result
            result_dict.setdefault("malformed_blocks", []).extend(malformed_blocks_list)
            return result_dict
        except ValidationError as validation_e:
            malformed_blocks_list.append({
                "type": "SCHEMA_VALIDATION_ERROR",
                "message": str(validation_e),
                "raw_string_snippet": extracted_json_str[:1000] + ("..." if len(extracted_json_str) > 1000 else "")
            })
            # If schema validation fails, return the fallback output directly.
            # The fallback output itself is now guaranteed to be schema-valid.
            return self._create_fallback_output(schema_model, malformed_blocks_list, raw_output, data_to_validate)
        except Exception as general_e:
            malformed_blocks_list.append({
                "type": "UNEXPECTED_VALIDATION_ERROR",
                "message": str(general_e),
                "raw_string_snippet": extracted_json_str[:1000] + ("..." if len(extracted_json_str) > 1000 else "")
            })
            # If any other exception occurs, return the fallback output directly.
            return self._create_fallback_output(schema_model, malformed_blocks_list, raw_output)

    def _create_fallback_output(self, schema_model: Type[BaseModel], malformed_blocks: List[Dict[str, Any]], raw_output_snippet: str, partial_data: Optional[Any] = None) -> Dict[str, Any]:
        """Creates a structured fallback output based on the schema model."""
        # Ensure partial_data is always a dictionary for safe .get() calls
        # If partial_data is a string, it means the LLM returned a raw string instead of JSON.
        # We should use this string as a general error message.
        if isinstance(partial_data, str):
            error_message_from_partial = partial_data
            partial_data = {} # Reset to empty dict for .get() calls
        else:
            error_message_from_partial = "Failed to generate valid structured output."
            if not isinstance(partial_data, dict):
                partial_data = {} # Ensure it's a dict for safe .get()

        # Add a general error block if not already present
        if not any(block.get("type") == "LLM_OUTPUT_MALFORMED" for block in malformed_blocks):
            malformed_blocks.insert(0, { # Insert at beginning to be prominent
                "type": "LLM_OUTPUT_MALFORMED",
                "message": f"LLM output could not be fully parsed or validated. Raw snippet: {raw_output_snippet[:500]}...",
                "raw_string_snippet": raw_output_snippet[:1000] + ("..." if len(raw_output_snippet) > 1000 else "")
            })

        # Initialize a dictionary that will strictly conform to the schema_model
        fallback_data_for_model: Dict[str, Any] = {}

        # Determine if partial_data is a single suggestion dict (for SelfImprovementAnalysisOutput)
        # NEW: Use the new _detect_potential_suggestion_item helper
        is_single_suggestion_dict = False
        if schema_model in [SelfImprovementAnalysisOutput, SelfImprovementAnalysisOutputV1]:
            detected_suggestion = self._detect_potential_suggestion_item(raw_output_snippet)
            if detected_suggestion:
                is_single_suggestion_dict = True
                partial_data = detected_suggestion # Use the detected suggestion as partial data

        # Populate schema-specific fields with defaults or partial data
        if schema_model == LLMOutput:
            fallback_data_for_model["COMMIT_MESSAGE"] = partial_data.get("COMMIT_MESSAGE", "LLM_OUTPUT_ERROR")
            fallback_data_for_model["RATIONALE"] = partial_data.get("RATIONALE", error_message_from_partial) # Use error_message_from_partial
            fallback_data_for_model["CODE_CHANGES"] = partial_data.get("CODE_CHANGES", [])
            fallback_data_for_model["CONFLICT_RESOLUTION"] = partial_data.get("CONFLICT_RESOLUTION")
            fallback_data_for_model["UNRESOLVED_CONFLICT"] = partial_data.get("UNRESOLVED_CONFLICT")
            fallback_data_for_model["malformed_blocks"] = malformed_blocks
            fallback_data_for_model["malformed_code_change_items"] = partial_data.get("malformed_code_change_items", [])
        elif schema_model == CritiqueOutput:
            fallback_data_for_model["CRITIQUE_SUMMARY"] = partial_data.get("CRITIQUE_SUMMARY", error_message_from_partial) # Use error_message_from_partial
            fallback_data_for_model["CRITIQUE_POINTS"] = partial_data.get("CRITIQUE_POINTS", [])
            fallback_data_for_model["SUGGESTIONS"] = partial_data.get("SUGGESTIONS", [])
            fallback_data_for_model["malformed_blocks"] = malformed_blocks
        elif schema_model == ContextAnalysisOutput:
            fallback_data_for_model["key_modules"] = partial_data.get("key_modules", [])
            fallback_data_for_model["security_concerns"] = partial_data.get("security_concerns", [])
            fallback_data_for_model["architectural_patterns"] = partial_data.get("architectural_patterns", [])
            fallback_data_for_model["performance_bottlenecks"] = partial_data.get("performance_bottlenecks", [])
            fallback_data_for_model["security_summary"] = partial_data.get("security_summary", {})
            fallback_data_for_model["architecture_summary"] = partial_data.get("architecture_summary", {})
            fallback_data_for_model["devops_summary"] = partial_data.get("devops_summary", {})
            fallback_data_for_model["testing_summary"] = partial_data.get("testing_summary", {})
            fallback_data_for_model["general_overview"] = partial_data.get("general_overview", error_message_from_partial) # Use error_message_from_partial
            fallback_data_for_model["malformed_blocks"] = malformed_blocks
        elif schema_model == GeneralOutput:
            fallback_data_for_model["general_output"] = partial_data.get("general_output", error_message_from_partial) # Use error_message_from_partial
            fallback_data_for_model["malformed_blocks"] = malformed_blocks
        elif schema_model == ConflictReport:
            fallback_data_for_model["conflict_type"] = partial_data.get("conflict_type", "UNKNOWN")
            fallback_data_for_model["summary"] = partial_data.get("summary", error_message_from_partial) # Use error_message_from_partial
            fallback_data_for_model["involved_personas"] = partial_data.get("involved_personas", [])
            fallback_data_for_model["conflicting_outputs_snippet"] = partial_data.get("conflicting_outputs_snippet", "")
            fallback_data_for_model["proposed_resolution_paths"] = partial_data.get("proposed_resolution_paths", [])
            fallback_data_for_model["conflict_found"] = partial_data.get("conflict_found", True) # Assume conflict if malformed
            fallback_data_for_model["malformed_blocks"] = malformed_blocks
        elif schema_model == SelfImprovementAnalysisOutput or schema_model == SelfImprovementAnalysisOutputV1: # Handle both old and new schema
            fallback_data_for_model["ANALYSIS_SUMMARY"] = partial_data.get("ANALYSIS_SUMMARY", error_message_from_partial) # Use error_message_from_partial
            fallback_data_for_model["IMPACTFUL_SUGGESTIONS"] = partial_data.get("IMPACTFUL_SUGGESTIONS", [])
            fallback_data_for_model["malformed_blocks"] = malformed_blocks
            # Special handling if the LLM returned a single suggestion dict instead of the full analysis
            if is_single_suggestion_dict:
                self.logger.warning("LLM returned a single suggestion dict instead of full SelfImprovementAnalysisOutput. Wrapping it.")
                fallback_data_for_model["ANALYSIS_SUMMARY"] = f"LLM returned a single suggestion item instead of the full analysis. Original error: {error_message_from_partial}"
                fallback_data_for_model["IMPACTFUL_SUGGESTIONS"] = [partial_data] # Wrap the single dict in a list
        
        # Now, attempt to validate this constructed fallback_data_for_model against the schema.
        # This ensures that what we return is always a valid Pydantic model instance.
        try:
            # If the target schema is the versioned wrapper, ensure the data is wrapped correctly
            if schema_model == SelfImprovementAnalysisOutput:
                # If we have a V1-like structure, wrap it
                if "ANALYSIS_SUMMARY" in fallback_data_for_model and "IMPACTFUL_SUGGESTIONS" in fallback_data_for_model:
                    v1_data = SelfImprovementAnalysisOutputV1.model_validate(fallback_data_for_model)
                    validated_fallback = SelfImprovementAnalysisOutput(
                        version="1.0",
                        data=v1_data.model_dump(by_alias=True),
                        malformed_blocks=malformed_blocks # Pass malformed blocks to the wrapper
                    )
                else: # If it's not even V1-like, create a minimal wrapper
                    validated_fallback = SelfImprovementAnalysisOutput(
                        version="1.0",
                        data={
                            "ANALYSIS_SUMMARY": error_message_from_partial,
                            "IMPACTFUL_SUGGESTIONS": [],
                            "malformed_blocks": malformed_blocks
                        },
                        malformed_blocks=malformed_blocks
                    )
            else:
                validated_fallback = schema_model.model_validate(fallback_data_for_model)
            
            return validated_fallback.model_dump(by_alias=True)
        except ValidationError as e:
            # If even the fallback construction fails validation, it means our fallback logic
            # for that schema_model is incorrect. Log this critical error.
            self.logger.critical(f"CRITICAL ERROR: Fallback output for schema {schema_model.__name__} is itself invalid: {e}. Returning raw error dict.", exc_info=True)
            # As a last resort, return a generic error dict. This should ideally not happen.
            return {
                "general_output": f"CRITICAL PARSING ERROR: Fallback for {schema_model.__name__} is invalid. {str(e)}",
                "malformed_blocks": malformed_blocks + [{"type": "CRITICAL_FALLBACK_ERROR", "message": str(e)}]
            }