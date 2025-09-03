# src/utils/output_parser.py

import json
import logging
import re
import sys
import traceback
from typing import Dict, Any, List, Optional, Type, Tuple
from pathlib import Path
from pydantic import (
    BaseModel,
    ValidationError,
    field_validator,
    model_validator,
    ConfigDict,
)  # Ensure BaseModel, ValidationError, field_validator, model_validator, ConfigDict are imported

# Ensure all relevant models are imported
from src.models import (
    CodeChange,
    LLMOutput,
    ContextAnalysisOutput,
    CritiqueOutput,
    GeneralOutput,
    ConflictReport,
    SelfImprovementAnalysisOutput,
    SelfImprovementAnalysisOutputV1,
    ConfigurationAnalysisOutput,
    DeploymentAnalysisOutput,
)

logger = logging.getLogger(__name__)


class LLMOutputParser:
    def __init__(self):
        self.logger = logger

    def _extract_json_with_markers(
        self,
        text: str,
        start_marker: str = "START_JSON_OUTPUT",
        end_marker: str = "END_JSON_OUTPUT",
    ) -> Optional[Tuple[str, bool]]:
        """
        Extracts JSON content explicitly delimited by start and end markers.
        Returns (json_string_content, end_marker_found).
        If only start_marker is found, returns (content_after_start_marker, False).
        If no valid JSON can be extracted after a missing end marker, returns None.
        """
        self.logger.debug(
            f"Attempting to extract JSON using markers '{start_marker}' and '{end_marker}'..."
        )
        start_match = re.search(re.escape(start_marker), text)
        if not start_match:
            self.logger.debug("Start marker not found.")
            return None

        text_after_start_marker = text[start_match.end() :]
        end_match = re.search(re.escape(end_marker), text_after_start_marker)

        json_content_raw = ""
        end_marker_found = False

        if end_match:
            json_content_raw = text_after_start_marker[: end_match.start()].strip()
            end_marker_found = True
            self.logger.debug("Both markers found. Extracted content between them.")
        else:
            self.logger.warning(
                "Start marker found, but end marker is missing. Attempting to extract first outermost JSON object from content after start marker."
            )
            # If end marker is missing, try to robustly extract a single JSON object
            # from the content that follows the start marker.
            json_content_raw = self._extract_first_outermost_json(
                text_after_start_marker
            )
            if json_content_raw:
                self.logger.debug(
                    "Successfully extracted a single JSON object after missing end marker."
                )
            else:
                self.logger.warning(
                    "Could not extract a single JSON object after missing end marker."
                )
                return None  # Return None if no single JSON object can be robustly extracted

        if json_content_raw:
            # Basic sanitization for trailing commas before closing braces/brackets
            json_content_raw = re.sub(r",\s*([\}\]])", r"\1", json_content_raw)
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
            if char == "{":
                if start_index == -1:  # First opening brace
                    start_index = i
                    balance = 1
                    expected_closers_stack.append("}")
                else:
                    balance += 1
                    expected_closers_stack.append("}")
            elif char == "[":
                if start_index == -1:  # First opening bracket
                    start_index = i
                    balance = 1
                    expected_closers_stack.append("]")
                else:
                    balance += 1
                    expected_closers_stack.append("]")
            elif char == "}":
                if start_index != -1:
                    balance -= 1
                    if expected_closers_stack and expected_closers_stack[-1] == "}":
                        expected_closers_stack.pop()
            elif char == "]":
                if start_index != -1:
                    balance -= 1
                    if expected_closers_stack and expected_closers_stack[-1] == "]":
                        expected_closers_stack.pop()

            if start_index != -1 and balance == 0 and not expected_closers_stack:
                potential_json_str = text[start_index : i + 1]
                try:
                    json.loads(potential_json_str)
                    self.logger.debug(
                        f"Successfully extracted first outermost valid JSON block: {potential_json_str[:100]}..."
                    )
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

        markdown_block_pattern = r"```(?:json|python|text|)\s*(.*?)(?:```|\Z)"

        matches = list(
            re.finditer(markdown_block_pattern, text, re.DOTALL | re.MULTILINE)
        )

        for match in matches:
            block_content = match.group(1).strip()
            if block_content:
                self.logger.debug(
                    f"Extracted raw content from markdown block: {block_content[:100]}..."
                )

                # Now, apply the robust JSON extraction on this block content
                # This will find the first balanced and valid JSON object/array within the block
                extracted_json = self._extract_first_outermost_json(
                    block_content
                )  # Use the robust extractor
                if extracted_json:
                    self.logger.debug(
                        "Successfully extracted and validated JSON from markdown block content."
                    )
                    return extracted_json

        self.logger.debug("No valid JSON block found in markdown code blocks.")
        return None

    def _repair_json_string(self, json_str: str) -> Tuple[str, List[str]]:
        """Applies common JSON repair heuristics and logs repairs."""
        repair_log = []

        # 1. Handle the specific numbered array element issue (e.g., "0:{")
        temp_str = re.sub(r"\d+\s*:\s*{", "{", json_str)
        temp_str = re.sub(r",\s*\d+\s*:\s*{", ", {", temp_str)
        if temp_str != json_str:
            repair_log.append("Fixed numbered array elements (e.g., '0:{' -> '{').")
            json_str = temp_str

        # 2. Handle cases where entire array is wrapped in quotes
        temp_str = re.sub(r'"\[\s*{', "[{", json_str)
        temp_str = re.sub(r'}\s*\]"', "}]", temp_str)
        if temp_str != json_str:
            repair_log.append("Fixed array incorrectly wrapped in quotes.")
            json_str = temp_str

        # 3. Replace single quotes with double quotes (careful not to break escaped quotes)
        temp_str = re.sub(r"(?<!\\)\'", '"', json_str)
        if temp_str != json_str:
            repair_log.append("Replaced single quotes with double quotes.")
            json_str = temp_str

        # 4. Handle unquoted keys (simple heuristic, might fail on complex cases)
        temp_str = re.sub(
            r"([{,]\s*)([a-zA-Z_][a-zA-Z0-9_]*)\s*:", r'\1"\2":', json_str
        )
        if temp_str != json_str:
            repair_log.append("Added quotes to unquoted keys.")
            json_str = temp_str

        # 5. Handle trailing commas in objects/arrays
        temp_str = re.sub(r",\s*([\}\]])", r"\1", json_str)
        if temp_str != json_str:
            repair_log.append("Removed trailing commas in objects/arrays.")
            json_str = temp_str

        # 6. Balance braces and brackets (add missing closers at the end)
        open_braces = json_str.count("{")
        close_braces = json_str.count("}")
        if open_braces > close_braces:
            json_str += "}" * (open_braces - close_braces)
            repair_log.append(
                f"Added {open_braces - close_braces} missing closing braces."
            )

        open_brackets = json_str.count("[")
        close_brackets = json_str.count("]")
        if open_brackets > close_brackets:
            json_str += "]" * (open_brackets - close_brackets)
            repair_log.append(
                f"Added {open_brackets - close_brackets} missing closing brackets."
            )

        return json_str, repair_log

    def _force_close_truncated_json(self, json_str: str) -> str:
        """
        Attempts to heuristically force-close a JSON string that appears to be truncated.
        This is a last-ditch effort and might not always produce perfectly valid JSON,
        but aims to salvage a parseable fragment.
        """
        cleaned_str = json_str.strip()
        if not cleaned_str:
            return ""

        # Check if it starts like a JSON object or array
        if not (cleaned_str.startswith("{") or cleaned_str.startswith("[")):
            return cleaned_str  # Not a JSON string or already malformed at start

        # Count open/close braces/brackets
        open_braces = cleaned_str.count("{")
        close_braces = cleaned_str.count("}")
        open_brackets = cleaned_str.count("[")
        close_brackets = cleaned_str.count("]")

        # If it's clearly unbalanced and truncated at the end, try to add closers
        # Only add if the string doesn't already end with the correct closer
        if open_braces > close_braces and not cleaned_str.endswith("}"):
            self.logger.debug(
                f"Force-closing with {open_braces - close_braces} braces."
            )
            cleaned_str += "}" * (open_braces - close_braces)
        if open_brackets > close_brackets and not cleaned_str.endswith("]"):
            self.logger.debug(
                f"Force-closing with {open_brackets - close_brackets} brackets."
            )
            cleaned_str += "]" * (open_brackets - close_brackets)

        return cleaned_str

    def _parse_with_incremental_repair(
        self, json_str: str
    ) -> Tuple[Optional[Dict[str, Any]], List[Dict[str, str]]]:
        """Attempts to parse JSON with incremental repair strategies."""
        repair_log = []
        current_json_str = json_str

        # Attempt 1: Apply all standard repairs and try to load
        repaired_text, current_repairs = self._repair_json_string(current_json_str)
        repair_log.extend(
            [{"action": "initial_repair", "details": r} for r in current_repairs]
        )
        try:
            result = json.loads(repaired_text)
            return result, repair_log
        except json.JSONDecodeError:
            self.logger.debug(
                f"Initial repair failed to parse. Trying further heuristics. Snippet: {repaired_text[:100]}"
            )
            current_json_str = repaired_text  # Update for next steps

        # Attempt 2: Extract largest valid sub-object from the (already repaired) text
        largest_sub_object_str = self._extract_largest_valid_subobject(current_json_str)
        if largest_sub_object_str:
            repair_log.append(
                {
                    "action": "extracted_largest_subobject",
                    "details": "Attempting to parse largest valid JSON fragment.",
                }
            )
            try:
                result = json.loads(largest_sub_object_str)
                return result, repair_log
            except json.JSONDecodeError:
                self.logger.debug(
                    f"Largest sub-object extraction failed to parse. Snippet: {largest_sub_object_str[:100]}"
                )
                pass  # Continue to next attempt

        # Attempt 3: Treat as JSON lines (DeepSeek's idea)
        json_lines_str = self._convert_to_json_lines(current_json_str)
        if json_lines_str != current_json_str:  # Only if conversion actually happened
            repair_log.append(
                {
                    "action": "converted_to_json_lines",
                    "details": "Attempting to parse as JSON lines.",
                }
            )
            try:
                result = json.loads(json_lines_str)
                return result, repair_log
            except json.JSONDecodeError:
                self.logger.debug(
                    f"JSON lines conversion failed to parse. Snippet: {json_lines_str[:100]}"
                )
                pass

        # NEW Attempt 4: Force-close truncated JSON as a last resort
        force_closed_json_str = self._force_close_truncated_json(current_json_str)
        if (
            force_closed_json_str != current_json_str
        ):  # Only if actual changes were made
            repair_log.append(
                {
                    "action": "force_closed_truncated_json",
                    "details": "Attempting to force-close truncated JSON.",
                }
            )
            try:
                result = json.loads(force_closed_json_str)
                return result, repair_log
            except json.JSONDecodeError:
                self.logger.debug(
                    f"Force-closed JSON failed to parse. Snippet: {force_closed_json_str[:100]}"
                )
                pass

        return None, repair_log

    def _extract_largest_valid_subobject(self, text: str) -> Optional[str]:
        """
        Extracts the largest potentially valid JSON object or array from malformed text.
        Leverages _extract_first_outermost_json for robustness.
        """
        self.logger.debug(
            "Attempting to extract largest valid sub-object using robust method."
        )

        longest_valid_match = ""
        current_search_text = text

        # Iterate, extracting one outermost JSON block at a time, and removing it from the search text
        # to find subsequent blocks.
        while True:
            extracted_json_str = self._extract_first_outermost_json(current_search_text)
            if extracted_json_str:
                if len(extracted_json_str) > len(longest_valid_match):
                    longest_valid_match = extracted_json_str

                # Remove the extracted part to find other potential JSON blocks
                # This is a simple removal; a more complex approach might handle overlapping.
                current_search_text = current_search_text.replace(
                    extracted_json_str, "", 1
                )
            else:
                break  # No more valid JSON blocks found

        if longest_valid_match:
            self.logger.debug(
                f"Successfully extracted largest valid sub-object: {longest_valid_match[:100]}..."
            )
            return longest_valid_match

        self.logger.debug("No largest valid sub-object found.")
        return None

    def _convert_to_json_lines(self, json_str: str) -> str:
        """Converts potential JSON lines format to array format."""
        lines = json_str.strip().split("\n")
        json_objects = []

        for line in lines:
            line = line.strip()
            if line.startswith("{") and line.endswith("}"):
                try:
                    json.loads(line)  # Validate it's parseable
                    json_objects.append(line)
                except json.JSONDecodeError:
                    continue

        if json_objects:
            return f"[{','.join(json_objects)}]"

        return json_str

    def _clean_llm_output(self, raw_output: str) -> str:
        """Clean common LLM output artifacts like markdown fences and conversational filler."""
        cleaned = raw_output

        # Remove Markdown code blocks (```json, ```python, ```, etc.)
        cleaned = re.sub(
            r"^\s*```(?:json|python|text|)\s*\n", "", cleaned, flags=re.MULTILINE
        )
        cleaned = re.sub(r"\n\s*```\s*$", "", cleaned, flags=re.MULTILINE)

        # Remove common leading/trailing conversational filler
        cleaned = re.sub(
            r"^(?:Here is the JSON output|```json|```|```python|```text|```).*?\n",
            "",
            cleaned,
            flags=re.DOTALL | re.IGNORECASE,
        )
        cleaned = re.sub(
            r"\n(?:```|```json|```python|```text|```).*?$",
            "",
            cleaned,
            flags=re.DOTALL | re.IGNORECASE,
        )

        # Attempt to find the outermost JSON structure and trim anything outside it
        json_start = -1
        json_end = -1

        # Find first '{' or '['
        for i, char in enumerate(cleaned):
            if char == "{" or char == "[":
                json_start = i
                break

        # Find last '}' or ']'
        for i in range(len(cleaned) - 1, -1, -1):
            if cleaned[i] == "}" or cleaned[i] == "]":
                json_end = i + 1
                break

        if json_start != -1 and json_end != -1 and json_end > json_start:
            cleaned = cleaned[json_start:json_end]

        return cleaned.strip()

    def _detect_potential_suggestion_item(self, text: str) -> Optional[Dict]:
        """
        Detects if the text contains what looks like a single 'IMPACTFUL_SUGGESTIONS' item
        at the top level, rather than the full SelfImprovementAnalysisOutput.
        """
        area_match = re.search(
            r'"AREA"\s*:\s*"(Reasoning Quality|Robustness|Efficiency|Maintainability|Security)"',
            text,
        )
        problem_match = re.search(r'"PROBLEM"\s*:\s*"[^"]+"', text)
        proposed_solution_match = re.search(r'"PROPOSED_SOLUTION"\s*:\s*"[^"]+"', text)

        if area_match and problem_match and proposed_solution_match:
            try:
                start_idx = text.rfind("{", 0, area_match.start())
                if start_idx >= 0:
                    brace_count = 1
                    for i in range(start_idx + 1, len(text)):
                        if text[i] == "{":
                            brace_count += 1
                        elif text[i] == "}":
                            brace_count -= 1
                            if brace_count == 0:
                                obj_str = text[start_idx : i + 1]
                                return json.loads(obj_str)
            except Exception as e:
                self.logger.debug(f"Error extracting potential suggestion item: {e}")

        return None

    def parse_and_validate(
        self, raw_output: str, schema_model: Type[BaseModel]
    ) -> Dict[str, Any]:
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
        transformation_needed = False

        # FIX: Handle case where schema_model is a string (model name) instead of the model class
        if isinstance(schema_model, str):
            # Dynamically import the model class if it's provided as a string name
            model_map = {
                "SelfImprovementAnalysisOutputV1": SelfImprovementAnalysisOutputV1,
                "ContextAnalysisOutput": ContextAnalysisOutput,
                "CritiqueOutput": CritiqueOutput,
                "GeneralOutput": GeneralOutput,
                "ConflictReport": ConflictReport,
                "ConfigurationAnalysisOutput": ConfigurationAnalysisOutput,
                "DeploymentAnalysisOutput": DeploymentAnalysisOutput,
                "SelfImprovementAnalysisOutput": SelfImprovementAnalysisOutput,
                "LLMOutput": LLMOutput,  # Ensure LLMOutput is also mapped
            }

            if schema_model in model_map:
                schema_model = model_map[schema_model]
            else:
                raise ValueError(f"Unknown schema model: {schema_model}")
        # End FIX

        # 1. Attempt to extract JSON string using markers from the cleaned output
        cleaned_raw_output = self._clean_llm_output(raw_output)

        marker_extraction_result = self._extract_json_with_markers(cleaned_raw_output)
        if marker_extraction_result:
            extracted_json_str, end_marker_found = marker_extraction_result
            if not end_marker_found:
                malformed_blocks_list.append(
                    {
                        "type": "MISSING_END_MARKER",
                        "message": "START_JSON_OUTPUT was found, but END_JSON_OUTPUT was missing. Attempted to parse content after START_JSON_OUTPUT.",
                        "raw_string_snippet": cleaned_raw_output[:1000]
                        + ("..." if len(cleaned_raw_output) > 1000 else ""),
                    }
                )
                transformation_needed = True
        else:
            # If no markers, try direct parsing of the cleaned output first.
            # If it works, no further extraction/repair is needed for this step.
            try:
                parsed_data = json.loads(cleaned_raw_output)
                extracted_json_str = cleaned_raw_output
            except json.JSONDecodeError:
                # No markdown blocks, try robust extraction on the whole cleaned_raw_output
                extracted_json_str = self._extract_json_from_markdown(
                    cleaned_raw_output
                )
                if not extracted_json_str:
                    extracted_json_str = self._extract_first_outermost_json(
                        cleaned_raw_output
                    )
                    if extracted_json_str:
                        transformation_needed = True  # If this succeeded, it means initial direct parsing failed

        if not extracted_json_str:
            malformed_blocks_list.append(
                {
                    "type": "JSON_EXTRACTION_FAILED",
                    "message": "Could not find or extract a valid JSON structure from the output.",
                    "raw_string_snippet": cleaned_raw_output[:1000]
                    + ("..." if len(cleaned_raw_output) > 1000 else ""),
                }
            )
            return self._create_fallback_output(
                schema_model, malformed_blocks_list, raw_output
            )

        # If parsed_data is still None, it means direct parsing of cleaned_raw_output failed,
        # and we need to proceed with incremental repair on extracted_json_str.
        if parsed_data is None:
            # 2. Parse JSON string with incremental repair attempts
            parsed_data, repair_log = self._parse_with_incremental_repair(
                extracted_json_str
            )
            if repair_log:
                malformed_blocks_list.append(
                    {
                        "type": "JSON_REPAIR_ATTEMPTED",
                        "message": "JSON repair heuristics applied.",
                        "details": repair_log,
                        "raw_string_snippet": extracted_json_str[:1000]
                        + ("..." if len(extracted_json_str) > 1000 else ""),
                    }
                )
                transformation_needed = True

        if parsed_data is None:
            malformed_blocks_list.append(
                {
                    "type": "JSON_DECODE_ERROR",
                    "message": "Failed to decode JSON even after repair attempts.",
                    "raw_string_snippet": extracted_json_str[:1000]
                    + ("..." if len(extracted_json_str) > 1000 else ""),
                }
            )
            return self._create_fallback_output(
                schema_model, malformed_blocks_list, raw_output
            )

        # 3. Handle cases where JSON is a list or a single dict
        if isinstance(parsed_data, list):
            transformation_needed = (
                True  # A list was returned when a dict was likely expected
            )
            malformed_blocks_list.append(
                {
                    "type": "TOP_LEVEL_LIST_WRAPPING",
                    "message": f"LLM returned a top-level JSON array, which was wrapped into an object for schema {schema_model.__name__}.",
                }
            )
            if not parsed_data:  # Handle empty list specifically
                malformed_blocks_list.append(
                    {
                        "type": "EMPTY_JSON_LIST",
                        "message": "The LLM returned an empty JSON list.",
                    }
                )
                return self._create_fallback_output(
                    schema_model, malformed_blocks_list, raw_output
                )

            # --- NEW: Handle specific schema types that might return lists ---
            if schema_model in [
                SelfImprovementAnalysisOutput,
                SelfImprovementAnalysisOutputV1,
            ]:
                self.logger.warning(
                    "LLM returned an array of suggestions instead of full SelfImprovementAnalysisOutput. Wrapping it."
                )
                data_to_validate = {
                    "ANALYSIS_SUMMARY": "LLM returned an array of suggestions instead of the full analysis. Review the 'IMPACTFUL_SUGGESTIONS' section.",
                    "IMPACTFUL_SUGGESTIONS": parsed_data,
                    "malformed_blocks": malformed_blocks_list,  # Pass malformed blocks to the wrapped data
                }
            elif schema_model == CritiqueOutput:
                if all(isinstance(item, str) for item in parsed_data):
                    self.logger.warning(
                        "LLM returned a list of strings for CritiqueOutput. Assuming they are suggestions."
                    )
                    data_to_validate = {
                        "CRITIQUE_SUMMARY": "LLM returned a list of strings as suggestions.",
                        "CRITIQUE_POINTS": [],
                        "SUGGESTIONS": parsed_data,
                        "malformed_blocks": malformed_blocks_list,
                    }
                elif all(isinstance(item, dict) for item in parsed_data):
                    self.logger.warning(
                        "LLM returned a list of dicts for CritiqueOutput. Assuming they are critique points."
                    )
                    data_to_validate = {
                        "CRITIQUE_SUMMARY": "LLM returned a list of dicts as critique points.",
                        "CRITIQUE_POINTS": parsed_data,
                        "SUGGESTIONS": [],
                        "malformed_blocks": malformed_blocks_list,
                    }
                else:
                    self.logger.warning(
                        "LLM returned a mixed/unexpected list for CritiqueOutput. Creating generic fallback."
                    )
                    data_to_validate = {
                        "CRITIQUE_SUMMARY": f"LLM returned a mixed/unexpected list. First item: {str(parsed_data[0])[:100]}...",
                        "CRITIQUE_POINTS": [],
                        "SUGGESTIONS": [],
                        "malformed_blocks": malformed_blocks_list,
                    }
            elif schema_model == LLMOutput:
                if all(
                    isinstance(item, dict)
                    and all(k in item for k in ["FILE_PATH", "ACTION"])
                    for item in parsed_data
                ):
                    self.logger.warning(
                        "LLM returned a list of CodeChange objects for LLMOutput. Wrapping them."
                    )
                    data_to_validate = {
                        "COMMIT_MESSAGE": "LLM returned multiple code changes directly.",
                        "RATIONALE": "The LLM generated a list of code changes directly, which were wrapped into the expected LLMOutput format.",
                        "CODE_CHANGES": parsed_data,
                        "malformed_blocks": malformed_blocks_list,
                    }
                else:
                    self.logger.warning(
                        "LLM returned an unexpected list for LLMOutput. Creating generic fallback."
                    )
                    data_to_validate = {
                        "COMMIT_MESSAGE": "LLM_OUTPUT_ERROR: Unexpected list format",
                        "RATIONALE": f"LLM returned an unexpected list for LLMOutput. First item: {str(parsed_data[0])[:100]}...",
                        "CODE_CHANGES": [],
                        "malformed_blocks": malformed_blocks_list,
                    }
            elif schema_model == ConflictReport:
                if any(isinstance(item, dict) for item in parsed_data):
                    self.logger.warning(
                        "LLM returned a list for ConflictReport. Attempting to use first dict item."
                    )
                    first_dict_item = next(
                        (item for item in parsed_data if isinstance(item, dict)), {}
                    )
                    data_to_validate = first_dict_item
                else:
                    self.logger.warning(
                        "LLM returned a list of non-dict items for ConflictReport. Creating generic fallback."
                    )
                    data_to_validate = {
                        "conflict_type": "METHODOLOGY_DISAGREEMENT",
                        "summary": f"LLM returned a list instead of a single ConflictReport object. First item: {str(parsed_data[0])[:100]}...",
                        "involved_personas": [],
                        "conflicting_outputs_snippet": "",
                        "proposed_resolution_paths": [],
                        "conflict_found": True,
                        "malformed_blocks": malformed_blocks_list,
                    }
            elif (
                schema_model == GeneralOutput
            ):  # MODIFIED: Explicit handling for GeneralOutput
                if not parsed_data:  # Handle empty list specifically
                    data_to_validate = {
                        "general_output": "[]",  # Represent empty list as string
                        "malformed_blocks": malformed_blocks_list,
                    }
                elif all(isinstance(item, str) for item in parsed_data):
                    self.logger.warning(
                        "LLM returned a list of strings for GeneralOutput. Concatenating them."
                    )
                    data_to_validate = {
                        "general_output": "\n".join(parsed_data),
                        "malformed_blocks": malformed_blocks_list,
                    }
                elif all(isinstance(item, dict) for item in parsed_data):
                    self.logger.warning(
                        "LLM returned a list of dicts for GeneralOutput. Summarizing content."
                    )
                    # Summarize the dicts into a string
                    summarized_content = (
                        "LLM returned a list of objects. Summarized content: "
                        + json.dumps(parsed_data[:3])
                        + ("..." if len(parsed_data) > 3 else "")
                    )
                    data_to_validate = {
                        "general_output": summarized_content,
                        "malformed_blocks": malformed_blocks_list,
                    }
                else:
                    self.logger.warning(
                        "LLM returned a mixed/unexpected list for GeneralOutput. Creating generic fallback."
                    )
                    data_to_validate = {
                        "general_output": f"LLM returned a mixed/unexpected list. First item: {str(parsed_data[0])[:100]}...",
                        "malformed_blocks": malformed_blocks_list,
                    }
            else:
                # For other schemas, if it's a list, wrap it in a generic dict structure
                data_to_validate = {
                    "general_output": f"LLM returned a list instead of a {schema_model.__name__}. Content: {str(parsed_data)[:500]}...",
                    "malformed_blocks": malformed_blocks_list,
                }
        elif isinstance(parsed_data, dict):
            data_to_validate = parsed_data
            if schema_model in [
                SelfImprovementAnalysisOutput,
                SelfImprovementAnalysisOutputV1,
            ]:
                # Use the extracted_json_str for detecting potential single suggestion items
                detected_suggestion = self._detect_potential_suggestion_item(
                    extracted_json_str
                )
                if (
                    detected_suggestion
                    and "IMPACTFUL_SUGGESTIONS" not in data_to_validate
                    and "ANALYSIS_SUMMARY" not in data_to_validate
                ):
                    self.logger.warning(
                        "LLM returned a single suggestion dict instead of full SelfImprovementAnalysisOutput. Pre-wrapping it."
                    )
                    data_to_validate = {
                        "ANALYSIS_SUMMARY": "LLM returned a single suggestion item instead of the full analysis. This was wrapped into the expected format.",
                        "IMPACTFUL_SUGGESTIONS": [detected_suggestion],
                        "malformed_blocks": malformed_blocks_list,
                    }
        else:
            malformed_blocks_list.append(
                {
                    "type": "INVALID_JSON_STRUCTURE",
                    "message": f"Expected JSON object or array, but got {type(parsed_data).__name__}.",
                }
            )
            return self._create_fallback_output(
                schema_model, malformed_blocks_list, raw_output
            )

        # 4. Validate against schema
        try:
            # Use model_validate for Pydantic v2, fallback to parse_obj for v1
            if hasattr(schema_model, "model_validate"):
                validated_output = schema_model.model_validate(data_to_validate)
            else:
                validated_output = schema_model.parse_obj(data_to_validate)

            result_dict = validated_output.model_dump(by_alias=True)
            # Ensure malformed_blocks from parsing are added to the final result
            # Add a general LLM_OUTPUT_MALFORMED block if any transformation was needed
            if transformation_needed and not any(
                block.get("type") == "LLM_OUTPUT_MALFORMED"
                for block in malformed_blocks_list
            ):
                malformed_blocks_list.insert(
                    0,
                    {
                        "type": "LLM_OUTPUT_MALFORMED",
                        "message": "LLM output required structural transformation or repair to conform to schema.",
                        "raw_string_snippet": raw_output[:500]
                        + ("..." if len(raw_output) > 500 else ""),
                    },
                )
            result_dict.setdefault("malformed_blocks", []).extend(malformed_blocks_list)
            return result_dict
        except ValidationError as validation_e:
            malformed_blocks_list.append(
                {
                    "type": "SCHEMA_VALIDATION_ERROR",
                    "message": str(validation_e),
                    "raw_string_snippet": extracted_json_str[:1000]
                    + ("..." if len(extracted_json_str) > 1000 else ""),
                }
            )
            return self._create_fallback_output(
                schema_model, malformed_blocks_list, raw_output, data_to_validate
            )
        except Exception as general_e:
            malformed_blocks_list.append(
                {
                    "type": "UNEXPECTED_VALIDATION_ERROR",
                    "message": str(general_e),
                    "raw_string_snippet": extracted_json_str[:1000]
                    + ("..." if len(extracted_json_str) > 1000 else ""),
                }
            )
            # FIX: Corrected argument order here
            return self._create_fallback_output(
                schema_model, malformed_blocks_list, raw_output, data_to_validate
            )

    def _create_fallback_output(
        self,
        schema_model: Type[BaseModel],
        malformed_blocks: List[Dict[str, Any]],
        raw_output_snippet: str,
        partial_data: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """Creates a structured fallback output based on the schema model."""

        # Create a copy to avoid modifying the original list passed from parse_and_validate
        current_malformed_blocks = malformed_blocks.copy()

        error_message_from_partial = "Failed to generate valid structured output."

        # Ensure partial_data is a dictionary for consistent access
        partial_data_as_dict: Dict[str, Any]
        if partial_data is None:
            partial_data_as_dict = {}
            error_message_from_partial = (
                "No valid JSON data could be extracted or parsed."
            )
        elif isinstance(partial_data, str):
            partial_data_as_dict = {}  # Cannot reliably convert string to dict for key checks
            error_message_from_partial = (
                f"LLM returned raw string: '{partial_data[:100]}...'"
                if len(partial_data) > 100
                else f"LLM returned raw string: '{partial_data}'"
            )
        elif not isinstance(partial_data, dict):
            partial_data_as_dict = {}
            error_message_from_partial = f"Unexpected partial data type: {type(partial_data).__name__}. Value: {str(partial_data)[:100]}"
        else:  # partial_data is already a dict
            partial_data_as_dict = partial_data
            error_message_from_partial = "Failed to generate valid structured output."  # Default message if no specific error from partial_data

        # Add a general error block if not already present
        if not any(
            block.get("type") == "LLM_OUTPUT_MALFORMED"
            for block in current_malformed_blocks
        ):
            current_malformed_blocks.insert(
                0,
                {
                    "type": "LLM_OUTPUT_MALFORMED",
                    "message": "LLM output could not be fully parsed or validated. Raw snippet: {raw_output_snippet[:500]}...",
                    "raw_string_snippet": raw_output_snippet[:1000]
                    + ("..." if len(raw_output_snippet) > 1000 else ""),
                },
            )

        # NEW: Attempt to find any valid JSON fragment in the raw output for better debugging
        salvaged_json_fragment = self._extract_largest_valid_subobject(
            raw_output_snippet
        )
        if salvaged_json_fragment:
            current_malformed_blocks.append(
                {
                    "type": "SALVAGED_JSON_FRAGMENT",
                    "message": "A valid JSON fragment was found in the raw output, but it did not match the expected schema or was not the primary output.",
                    "raw_string_snippet": salvaged_json_fragment,
                }
            )

        fallback_data_for_model: Dict[str, Any] = {}

        is_single_suggestion_dict = False
        if schema_model in [
            SelfImprovementAnalysisOutput,
            SelfImprovementAnalysisOutputV1,
        ]:
            detected_suggestion = self._detect_potential_suggestion_item(
                extracted_json_str
            )
            if (
                detected_suggestion
                and "IMPACTFUL_SUGGESTIONS" not in partial_data_as_dict
                and "ANALYSIS_SUMMARY" not in partial_data_as_dict
            ):
                is_single_suggestion_dict = True
                partial_data_as_dict = (
                    detected_suggestion if detected_suggestion is not None else {}
                )

        if schema_model == LLMOutput:
            fallback_data_for_model["COMMIT_MESSAGE"] = partial_data_as_dict.get(
                "COMMIT_MESSAGE", "LLM_OUTPUT_ERROR"
            )
            fallback_data_for_model["RATIONALE"] = partial_data_as_dict.get(
                "RATIONALE", error_message_from_partial
            )
            fallback_data_for_model["CODE_CHANGES"] = partial_data_as_dict.get(
                "CODE_CHANGES", []
            )
            fallback_data_for_model["CONFLICT_RESOLUTION"] = partial_data_as_dict.get(
                "CONFLICT_RESOLUTION"
            )
            fallback_data_for_model["UNRESOLVED_CONFLICT"] = partial_data_as_dict.get(
                "UNRESOLVED_CONFLICT"
            )
            fallback_data_for_model["malformed_blocks"] = current_malformed_blocks
            fallback_data_for_model["malformed_code_change_items"] = (
                partial_data_as_dict.get("malformed_code_change_items", [])
            )
        elif schema_model == CritiqueOutput:
            fallback_data_for_model["CRITIQUE_SUMMARY"] = partial_data_as_dict.get(
                "CRITIQUE_SUMMARY", error_message_from_partial
            )
            fallback_data_for_model["CRITIQUE_POINTS"] = partial_data_as_dict.get(
                "CRITIQUE_POINTS", []
            )
            fallback_data_for_model["SUGGESTIONS"] = partial_data_as_dict.get(
                "SUGGESTIONS", []
            )
            fallback_data_for_model["malformed_blocks"] = current_malformed_blocks
        elif schema_model == ContextAnalysisOutput:
            fallback_data_for_model["key_modules"] = partial_data_as_dict.get(
                "key_modules", []
            )
            fallback_data_for_model["security_concerns"] = partial_data_as_dict.get(
                "security_concerns", []
            )
            fallback_data_for_model["architectural_patterns"] = (
                partial_data_as_dict.get("architectural_patterns", [])
            )
            fallback_data_for_model["performance_bottlenecks"] = (
                partial_data_as_dict.get("performance_bottlenecks", [])
            )
            fallback_data_for_model["security_summary"] = partial_data_as_dict.get(
                "security_summary", {}
            )
            fallback_data_for_model["architecture_summary"] = partial_data_as_dict.get(
                "architecture_summary", {}
            )
            fallback_data_for_model["devops_summary"] = partial_data_as_dict.get(
                "devops_summary", {}
            )
            fallback_data_for_model["testing_summary"] = partial_data_as_dict.get(
                "testing_summary", {}
            )
            fallback_data_for_model["general_overview"] = partial_data_as_dict.get(
                "general_overview", error_message_from_partial
            )
            fallback_data_for_model["configuration_summary"] = partial_data_as_dict.get(
                "configuration_summary", {}
            )
            fallback_data_for_model["deployment_summary"] = partial_data_as_dict.get(
                "deployment_summary", {}
            )
            fallback_data_for_model["malformed_blocks"] = current_malformed_blocks
        elif schema_model == ConflictReport:
            fallback_data_for_model["conflict_type"] = partial_data_as_dict.get(
                "conflict_type", "METHODOLOGY_DISAGREEMENT"
            )
            fallback_data_for_model["summary"] = partial_data_as_dict.get(
                "summary", error_message_from_partial
            )
            fallback_data_for_model["involved_personas"] = partial_data_as_dict.get(
                "involved_personas", []
            )
            fallback_data_for_model["conflicting_outputs_snippet"] = (
                partial_data_as_dict.get("conflicting_outputs_snippet", "")
            )
            fallback_data_for_model["proposed_resolution_paths"] = (
                partial_data_as_dict.get("proposed_resolution_paths", [])
            )
            fallback_data_for_model["conflict_found"] = partial_data_as_dict.get(
                "conflict_found", True
            )
            fallback_data_for_model["malformed_blocks"] = current_malformed_blocks
        elif (
            schema_model == SelfImprovementAnalysisOutput
            or schema_model == SelfImprovementAnalysisOutputV1
        ):
            fallback_data_for_model["ANALYSIS_SUMMARY"] = partial_data_as_dict.get(
                "ANALYSIS_SUMMARY", error_message_from_partial
            )
            fallback_data_for_model["IMPACTFUL_SUGGESTIONS"] = partial_data_as_dict.get(
                "IMPACTFUL_SUGGESTIONS", []
            )
            fallback_data_for_model["malformed_blocks"] = current_malformed_blocks
            if is_single_suggestion_dict:
                self.logger.warning(
                    "LLM returned a single suggestion dict instead of full SelfImprovementAnalysisOutput. Wrapping it."
                )
                fallback_data_for_model["ANALYSIS_SUMMARY"] = (
                    f"LLM returned a single suggestion item instead of the full analysis. Original error: {error_message_from_partial}"
                )
                fallback_data_for_model["IMPACTFUL_SUGGESTIONS"] = [
                    partial_data_as_dict
                ]
        elif schema_model == GeneralOutput:
            fallback_data_for_model["general_output"] = partial_data_as_dict.get(
                "general_output", error_message_from_partial
            )
            fallback_data_for_model["malformed_blocks"] = current_malformed_blocks
        else:
            # For other schemas, if it's a list, wrap it in a generic dict structure
            fallback_data_for_model = {
                "general_output": f"LLM returned a list instead of a {schema_model.__name__}. Content: {str(partial_data_as_dict)[:500]}...",
                "malformed_blocks": current_malformed_blocks,
            }

        try:
            if schema_model == SelfImprovementAnalysisOutput:
                if (
                    "ANALYSIS_SUMMARY" in fallback_data_for_model
                    and "IMPACTFUL_SUGGESTIONS" in fallback_data_for_model
                ):
                    # Use model_validate for Pydantic v2, fallback to parse_obj for v1
                    if hasattr(SelfImprovementAnalysisOutputV1, "model_validate"):
                        v1_data = SelfImprovementAnalysisOutputV1.model_validate(
                            fallback_data_for_model
                        )
                    else:
                        v1_data = SelfImprovementAnalysisOutputV1.parse_obj(
                            fallback_data_for_model
                        )

                    validated_fallback = SelfImprovementAnalysisOutput(
                        version="1.0",
                        data=v1_data.model_dump(by_alias=True),
                        malformed_blocks=current_malformed_blocks,
                    )
                else:
                    validated_fallback = SelfImprovementAnalysisOutput(
                        version="1.0",
                        data={
                            "ANALYSIS_SUMMARY": error_message_from_partial,
                            "IMPACTFUL_SUGGESTIONS": [],
                            "malformed_blocks": current_malformed_blocks,
                        },
                        malformed_blocks=current_malformed_blocks,
                    )
            else:
                # Use model_validate for Pydantic v2, fallback to parse_obj for v1
                if hasattr(schema_model, "model_validate"):
                    validated_fallback = schema_model.model_validate(
                        fallback_data_for_model
                    )
                else:
                    validated_fallback = schema_model.parse_obj(fallback_data_for_model)

            return validated_fallback.model_dump(by_alias=True)
        except ValidationError as e:
            self.logger.critical(
                f"CRITICAL ERROR: Fallback output for schema {schema_model.__name__} is itself invalid: {e}. Returning raw error dict.",
                exc_info=True,
            )
            return {
                "general_output": f"CRITICAL PARSING ERROR: Fallback for {schema_model.__name__} is invalid. {str(e)}",
                "malformed_blocks": current_malformed_blocks
                + [{"type": "CRITICAL_FALLBACK_ERROR", "message": str(e)}],
            }