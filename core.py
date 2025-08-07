# src/core.py
import yaml
import time
import hashlib
import sys
import re
import ast
import pycodestyle
import difflib
import subprocess
import tempfile
import os
import json
import logging
from rich.console import Console
from pydantic import BaseModel, Field, validator, model_validator, ValidationError
import streamlit as st
from typing import List, Dict, Tuple, Any, Callable, Optional, Type
from llm_provider import GeminiProvider, LLMProviderError, GeminiAPIError, LLMUnexpectedError
from src.utils.output_parser import LLMOutputParser
from src.utils.code_validator import validate_code_output_batch
# Import all necessary models
from src.models import PersonaConfig, ReasoningFrameworkConfig, LLMOutput, ContextAnalysisOutput

# --- Custom Exceptions for Project Chimera ---
class ChimeraError(Exception):
    """Base exception for all Project Chimera errors."""
    pass

class ContextAnalysisError(ChimeraError):
    """Base exception for context analysis failures."""
    pass

class MalformedContextError(ContextAnalysisError):
    """Raised when context analysis response contains malformed blocks."""
    def __init__(self, message, details=None):
        super().__init__(message)
        self.details = details

# Note: The original code imports TokenBudgetExceededError from llm_provider.
# The suggestion is to use it more specifically and potentially redefine it.
# For clarity and adherence to the suggestion, we define it here inheriting from ContextAnalysisError.
# In a production scenario, careful consideration would be given to potential conflicts or
# merging this with the imported exception if they represent the same concept.
class TokenBudgetExceededError(ContextAnalysisError):
    """Raised when context exceeds token budget constraints."""
    def __init__(self, message, required_tokens, available_tokens):
        super().__init__(message)
        self.required_tokens = required_tokens
        self.available_tokens = available_tokens

class SocraticDebate:
    DEFAULT_MAX_RETRIES = 2
    MAX_BACKOFF_SECONDS = 30
    def __init__(self,
                 initial_prompt: str,
                 api_key: str,
                 max_total_tokens_budget: int,
                 model_name: str,
                 personas: Dict[str, PersonaConfig], # Personas active in the current domain
                 all_personas: Dict[str, PersonaConfig], # All loaded personas
                 persona_sets: Dict[str, List[str]],
                 persona_sequence: List[str],
                 gemini_provider: Optional[GeminiProvider] = None,
                 domain: str = "General",
                 status_callback: Callable = None,
                 rich_console: Optional[Console] = None,
                 codebase_context: Optional[Dict[str, str]] = None,
                 context_token_budget_ratio: float = 0.25):
        self.initial_prompt = initial_prompt
        self.max_total_tokens_budget = max_total_tokens_budget
        self.model_name = model_name
        self.personas = personas
        self.domain = domain
        self.all_personas = all_personas
        self.persona_sets = persona_sets
        self.persona_sequence = persona_sequence # Default sequence if domain doesn't override
        self.status_callback = status_callback
        self.context_token_budget_ratio = context_token_budget_ratio
        if gemini_provider:
            self.gemini_provider = gemini_provider
        else:
            self.gemini_provider = GeminiProvider(api_key=api_key, model_name=model_name)
        self.parser = LLMOutputParser() # Initialize without a specific schema model
        self.cumulative_token_usage = 0
        self.cumulative_usd_cost = 0.0
        self.intermediate_steps: Dict[str, Any] = {}
        self.rich_console = rich_console if rich_console else Console()
        self.current_thought = initial_prompt
        self.final_answer = "Process did not complete."
        self.codebase_context = codebase_context

    def _update_status(self, message: str, **kwargs):
        """Helper to print to console and call Streamlit callback."""
        self.rich_console.print(message)
        if self.status_callback:
            self.status_callback(message=message, **kwargs)

    def _get_persona(self, name: str) -> PersonaConfig:
        """Retrieves a persona by name, checking both specific and all personas."""
        persona = self.personas.get(name) or self.all_personas.get(name)
        if not persona:
            raise ValueError(f"Persona '{name}' not found.")
        return persona

    @staticmethod
    @st.cache_data(ttl=3600)
    def _prioritize_python_code(_gemini_provider: GeminiProvider, content: str, max_tokens: int) -> str:
        """
        Prioritizes imports, class/function definitions for Python code.
        Truncates the content to fit within max_tokens.
        """
        lines = content.splitlines()
        priority_lines = []
        other_lines = []
        try:
            tree = ast.parse(content)
            priority_line_numbers = set()
            for node in tree.body:
                if isinstance(node, (ast.Import, ast.ImportFrom, ast.ClassDef, ast.FunctionDef)):
                    start_lineno = node.lineno - 1
                    end_lineno = node.end_lineno if hasattr(node, 'end_lineno') else start_lineno + 1
                    for i in range(start_lineno, end_lineno):
                        if i < len(lines):
                            priority_line_numbers.add(i)
            for i, line in enumerate(lines):
                if i in priority_line_numbers:
                    priority_lines.append(line)
                else:
                    other_lines.append(line)
        except SyntaxError:
            logging.warning(f"[yellow]Warning: Syntax error in Python context file, falling back to simple truncation.[/yellow]")
            return SocraticDebate._truncate_text_by_tokens(_gemini_provider, content, max_tokens, _status_callback=None)
        combined_content = "\n".join(priority_lines + other_lines)
        return SocraticDebate._truncate_text_by_tokens(_gemini_provider, combined_content, max_tokens, _status_callback=None)

    @staticmethod
    def _truncate_text_by_tokens(_gemini_provider: GeminiProvider, text: str, max_tokens: int, _status_callback: Callable = None) -> str:
        """Truncates text to fit within max_tokens using the GeminiProvider's token counting."""
        if not text:
            return ""
        current_tokens = _gemini_provider.count_tokens(text, "", _status_callback=_status_callback)
        if current_tokens <= max_tokens:
            return text
        chars_per_token_estimate = 4
        target_chars = max_tokens * chars_per_token_estimate
        truncated_text = text
        if len(truncated_text) > target_chars:
            truncated_text = truncated_text[:target_chars]
        while _gemini_provider.count_tokens(truncated_text, "", _status_callback=_status_callback) > max_tokens and len(truncated_text) > 0:
            chars_to_remove = max(1, len(truncated_text) // 20)
            truncated_text = truncated_text[:-chars_to_remove]
            if len(truncated_text) == 0:
                break
        if _gemini_provider.count_tokens(text, "", _status_callback=_status_callback) > max_tokens:
            return truncated_text.strip() + "\n... (truncated)"
        return truncated_text

    # --- NEW METHOD FOR TOKEN BUDGET EXCEEDED ---
    def _handle_token_budget_exceeded(self) -> Dict[str, Any]:
        """Placeholder for handling token budget exceeded errors during context analysis."""
        # In a real scenario, this might try to reduce context, warn the user, or return a default.
        # For now, returning a default summary as per the original error handling.
        self._update_status("[red]Context token budget exceeded. Cannot perform detailed analysis. Returning default summary.[/red]")
        return self._get_default_analysis_summary()

    # --- MODIFIED METHOD: prepare_context ---
    def prepare_context(self) -> str:
        """Intelligently prioritizes and truncates codebase context based on semantic relevance."""
        if not self.codebase_context:
            return "No codebase context provided."
        
        context_budget = int(self.max_total_tokens_budget * self.context_token_budget_ratio)
        
        # Identify most relevant files based on prompt content
        relevant_files = self._identify_relevant_files(context_budget)
        
        # Build context with prioritized files
        context_str_parts = []
        current_tokens = 0
        
        for path in relevant_files:
            content = self.codebase_context[path]
            header = f"\n=== {path} ===\n"
            header_tokens = self.gemini_provider.count_tokens(header, "", _status_callback=self.status_callback)
            
            # Check if we have enough budget for this file
            remaining_budget = context_budget - current_tokens - header_tokens
            if remaining_budget <= 0:
                self._update_status(f"Skipping file '{path}' due to context token budget exhaustion.")
                break
                
            # Strategically truncate file content to preserve important sections
            file_content = self._truncate_file_content(content, remaining_budget)
            
            context_str_parts.append(header + file_content)
            current_tokens += header_tokens + self.gemini_provider.count_tokens(file_content, "", _status_callback=self.status_callback)
        
        # Add summary of excluded files for transparency
        excluded_files = set(self.codebase_context.keys()) - set(relevant_files)
        if excluded_files and len(context_str_parts) > 0:
            excluded_summary = f"\n\n=== EXCLUDED FILES ({len(excluded_files)} total) ===\n"
            excluded_summary += "The following relevant files were not included due to token constraints:\n"
            excluded_summary += ", ".join(list(excluded_files)[:10])  # Show first 10
            if len(excluded_files) > 10:
                excluded_summary += f", and {len(excluded_files) - 10} more..."
            
            # Only add if we have tokens for it
            if self.gemini_provider.count_tokens(excluded_summary, "", _status_callback=self.status_callback) <= (context_budget - current_tokens):
                context_str_parts.append(excluded_summary)
        
        final_context_string = "".join(context_str_parts)
        self._update_status(
            f"Prepared codebase context using {self.gemini_provider.count_tokens(final_context_string, '', _status_callback=self.status_callback)} "
            f"tokens from {len(relevant_files)} relevant files (budget: {context_budget} tokens)."
        )
        return final_context_string

    # --- NEW HELPER METHOD FOR prepare_context ---
    def _identify_relevant_files(self, context_budget: int) -> List[str]:
        """Uses semantic analysis to identify which files are most relevant to the current prompt."""
        # Extract meaningful keywords from the prompt
        prompt_keywords = self._extract_keywords_from_prompt()
        
        # Calculate relevance score for each file
        file_scores = []
        for path, content in self.codebase_context.items():
            score = 0
            
            # Check path relevance (higher weight for path matches)
            path_lower = path.lower()
            for keyword in prompt_keywords:
                if keyword in path_lower:
                    score += 3  # Path match is highly relevant
            
            # Check content relevance (sample first part to avoid processing entire large files)
            content_sample = content[:1000].lower()  # First 1000 chars for better accuracy
            for keyword in prompt_keywords:
                if keyword in content_sample:
                    score += 2  # Content match is relevant
            
            file_scores.append((path, score))
        
        # Sort by score (highest first) and return only relevant files
        file_scores.sort(key=lambda x: x[1], reverse=True)
        return [path for path, score in file_scores if score > 0] or list(self.codebase_context.keys())[:5]

    # --- NEW HELPER METHOD FOR prepare_context ---
    def _extract_keywords_from_prompt(self) -> List[str]:
        """Extracts meaningful keywords from the user prompt using simple NLP techniques."""
        prompt = self.initial_prompt.lower()
        
        # Remove common stop words
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'is', 'are', 'was', 'were', 'in', 'on', 
                     'at', 'to', 'for', 'with', 'by', 'of', 'as', 'it', 'this', 'that', 'code', 'file'}
        words = [word.strip('.,!?()[]{}') for word in prompt.split() 
                 if word.lower() not in stop_words and len(word) > 2]
        
        # Return unique, meaningful keywords
        return list(set(words))

    # --- NEW HELPER METHOD FOR prepare_context ---
    def _truncate_file_content(self, content: str, max_tokens: int) -> str:
        """Strategically truncates file content to preserve the most important sections."""
        # For Python files, prioritize the beginning (imports, class definitions) and key methods
        if content.endswith('.py'):
            # Keep the first part (typically imports and class definitions)
            first_part = content[:max_tokens//2]
            
            # If possible, preserve some key methods from the middle/end
            if len(content) > max_tokens//2:
                # Look for function/method definitions in the entire content
                method_pattern = r'def\s+\w+\s*\('
                method_matches = list(re.finditer(method_pattern, content)) # Use 'content' to find all methods
                
                if method_matches:
                    # Take content from up to 3 key methods
                    selected_methods = []
                    for i, match in enumerate(method_matches[:3]):
                        start_idx = match.start()
                        # Find end of method (next 'def' or end of file)
                        end_idx = len(content)
                        if i < len(method_matches) - 1:
                            end_idx = method_matches[i+1].start()
                        
                        method_content = content[start_idx:end_idx].strip()
                        # Limit each method to reasonable size
                        if len(method_content) > 500:
                            method_content = method_content[:500] + "\n    # [Truncated for token budget]"
                        selected_methods.append(method_content)
                    
                    # Combine first part with selected methods
                    return first_part + "\n\n# Key methods:\n" + "\n\n".join(selected_methods)
            
            # If no methods found or content is short, just return the first part
            return first_part
        
        # For config/markup files, prioritize the beginning which often contains critical settings
        return content[:max_tokens]


    # --- MODIFIED METHOD: _analyze_codebase_context ---
    def _analyze_codebase_context(self) -> Dict[str, Any]:
        """
        Analyzes the codebase context using a dedicated persona.
        """
        if not self.codebase_context:
            self._update_status("[yellow]No codebase context provided for analysis.[/yellow]")
            return self._get_default_analysis_summary()
        if "Context_Aware_Assistant" not in self.personas and "Context_Aware_Assistant" not in self.all_personas:
            self._update_status("[yellow]Context_Aware_Assistant persona not found. Skipping context analysis.[/yellow]")
            return self._get_default_analysis_summary()
        self._update_status("Analyzing codebase context with Context_Aware_Assistant...")
        
        # Prepare context string using the new strategic method
        context_string_for_analysis = self.prepare_context()
        
        def analysis_prompt_gen():
            # Ensure analysis_str is properly formatted if summary is available
            # Check if analysis was already run and stored in intermediate_steps
            analysis_data = self.intermediate_steps.get("Context_Analysis_Output")
            if analysis_data:
                analysis_str = json.dumps(analysis_data, indent=2)
            else:
                # If analysis failed or was skipped, use a default summary
                analysis_str = json.dumps(self._get_default_analysis_summary(), indent=2)
            
            return (f"CODEBASE CONTEXT:\n{context_string_for_analysis}\n\n"
                    f"INSTRUCTIONS:\n"
                    f"Analyze the provided codebase context thoroughly. Understand its structure, style, patterns, dependencies, and overall logic.\n"
                    f"Provide a concise summary of the codebase in the specified JSON format.")

        analysis_output_key = "Context_Analysis_Output"
        try:
            # Execute analysis with validation
            analysis_response = self._execute_persona_step(
                "Context_Aware_Assistant",
                analysis_prompt_gen,
                analysis_output_key,
                schema_model_for_validation=ContextAnalysisOutput, # Pass the schema model
                update_current_thought=False,
                is_final_answer_step=False # This is not the final answer for the whole debate
            )
            
            # Check for specific error conditions
            if analysis_response and analysis_response.get("malformed_blocks"):
                raise MalformedContextError( # Use the new custom exception
                    "Context analysis response contained malformed blocks",
                    details=analysis_response["malformed_blocks"]
                )
                
            if isinstance(analysis_response, dict):
                self._update_status("Codebase context analysis successful.")
                # Store the successful analysis response for potential later use (e.g., in visionary_prompt_gen)
                self.intermediate_steps[analysis_output_key] = analysis_response
                return analysis_response
            else:
                raise ContextAnalysisError("Context analysis persona returned non-JSON or empty response.") # Use the new custom exception
                
        except MalformedContextError as e: # Catch the specific new exception
            self._update_status(f"[yellow]Warning: {str(e)}. Using default summary.[/yellow]")
            logger.warning(f"Malformed context: {str(e)} - Details: {e.details}")
            return self._get_default_analysis_summary()
            
        except TokenBudgetExceededError as e: # Catch the specific new exception
            self._update_status(
                f"[red]Token budget exceeded: Need {e.required_tokens} tokens but only {e.available_tokens} available. "
                "Consider increasing token budget.[/red]"
            )
            logger.getLogger(__name__).error(f"Token budget exceeded during context analysis: {str(e)}") # Use logger directly
            # Call the placeholder method
            return self._handle_token_budget_exceeded()
            
        except ContextAnalysisError as e: # Catch the specific new exception
            self._update_status(f"[yellow]Context analysis failed: {str(e)}. Using default summary.[/yellow]")
            logger.exception("Context analysis error")
            return self._get_default_analysis_summary()
            
        except Exception as e: # Catch-all for unexpected errors with detailed logging
            self._update_status(f"[red]Unexpected error: {str(e)}. Using default summary.[/red]")
            logger.exception("Unexpected error during context analysis")
            return self._get_default_analysis_summary()

    def _execute_persona_step(self, persona_name: str, step_prompt_generator: Callable[[], str], output_key: str, max_retries_on_fail: int = 1, update_current_thought: bool = False, is_final_answer_step: bool = False, schema_model_for_validation: Optional[Type[BaseModel]] = None) -> Any:
        """Executes a single persona step, handling token budget, status updates, and parsing/validation errors."""
        persona = self._get_persona(persona_name)
        step_prompt = step_prompt_generator()
        
        json_repair_attempts = 0
        MAX_JSON_REPAIR_ATTEMPTS = 2 # Limit attempts to avoid infinite loops for JSON repair

        # This loop handles API errors and general retries for a persona step.
        for api_retry_attempt in range(max_retries_on_fail + 1):
            current_persona_name = persona_name
            current_persona = persona
            current_step_prompt = step_prompt # Start with original prompt for this API attempt
            current_output_key = output_key
            
            # If this is a fallback attempt due to API error, adjust prompt/persona
            if api_retry_attempt > 0:
                current_persona_name = "Generalist_Assistant"
                if "Generalist_Assistant" not in self.all_personas:
                    self._update_status(f"[red]Error: Generalist_Assistant not found for fallback. Aborting.[/red]", state="error")
                    raise LLMUnexpectedError("Generalist_Assistant persona not found for fallback.")
                current_persona = self._get_persona(current_persona_name)
                
                # Modify prompt for fallback: ask for plain text summary if it was a final answer step
                if is_final_answer_step and persona_name == "Impartial_Arbitrator":
                    current_step_prompt = (
                        f"The previous attempt to synthesize the final answer failed due to an API error. "
                        f"Please provide a concise, plain-text summary of the debate and the proposed solution. "
                        f"Do NOT attempt to generate JSON. Focus on the key takeaways and any unresolved issues. "
                        f"Original prompt for final answer:\n{step_prompt}"
                    )
                else:
                    current_step_prompt = (f"The previous attempt to process the following prompt with persona '{persona_name}' failed due to an API error. "
                                            f"Please provide a general, concise summary or attempt to answer the original prompt given the context. "
                                            f"Original prompt:\n{step_prompt}")
                current_output_key = f"{output_key}_API_Fallback_Attempt_{api_retry_attempt}"
                self._update_status(f"[yellow]Warning: Persona '{persona_name}' failed due to API error. Attempting fallback to '{current_persona_name}' (Attempt {api_retry_attempt}/{max_retries_on_fail}).[/yellow]", state="running")
            
            # Calculate tokens for the current prompt
            estimated_input_tokens = self.gemini_provider.count_tokens(prompt=current_step_prompt, system_prompt=current_persona.system_prompt, _status_callback=self.status_callback)
            remaining_budget = self.max_total_tokens_budget - self.cumulative_token_usage
            max_output_for_request = max(0, min(current_persona.max_tokens, remaining_budget - estimated_input_tokens))
            
            if estimated_input_tokens >= remaining_budget:
                 raise TokenBudgetExceededError(f"Prompt for '{persona_name}' ({estimated_input_tokens} tokens) exceeds remaining budget ({remaining_budget} tokens).")
            
            estimated_next_step_cost = self.gemini_provider.calculate_usd_cost(estimated_input_tokens, max_output_for_request)
            self._update_status(f"Running persona: {current_persona_name} (Input: {estimated_input_tokens} tokens, Output: {max_output_for_request} tokens)...",
                                current_total_tokens=self.cumulative_token_usage,
                                current_total_cost=self.cumulative_usd_cost,
                                estimated_next_step_tokens=estimated_input_tokens + max_output_for_request,
                                estimated_next_step_cost=estimated_next_step_cost)
            
            try:
                raw_response_text, input_tokens, output_tokens = self.gemini_provider.generate(
                    prompt=current_step_prompt,
                    system_prompt=current_persona.system_prompt,
                    temperature=current_persona.temperature,
                    max_tokens=max_output_for_request,
                    _status_callback=self.status_callback
                )
                tokens_used_in_step = input_tokens + output_tokens
                cost_this_step = self.gemini_provider.calculate_usd_cost(input_tokens, output_tokens)
                
                # --- START JSON SELF-CORRECTION LOOP (if schema_model_for_validation is provided) ---
                if schema_model_for_validation:
                    # Loop for JSON repair attempts
                    while json_repair_attempts <= MAX_JSON_REPAIR_ATTEMPTS:
                        parsed_data = None
                        parsing_error_details = None
                        
                        if raw_response_text:
                            try:
                                # Attempt to parse and validate the LLM's output against the provided schema
                                parsed_data = self.parser.parse_and_validate(raw_response_text, schema_model_for_validation)
                                
                                # If parse_and_validate returns malformed_blocks, it means it couldn't fully fix it.
                                # We still want the LLM to try and fix it itself.
                                if parsed_data.get("malformed_blocks"):
                                    # Take the first malformed block for the error message
                                    first_malformed_block = parsed_data["malformed_blocks"][0]
                                    if isinstance(first_malformed_block, dict):
                                        parsing_error_details = first_malformed_block
                                    else: # Fallback if malformed_blocks contains strings (shouldn't happen with new parser)
                                        parsing_error_details = {"type": "UNKNOWN_MALFORMED_BLOCK", "message": str(first_malformed_block), "raw_string_snippet": str(first_malformed_block)[:500]}
                                    raise ValueError(f"Malformed blocks detected by parser: {parsing_error_details['message']}")
                                # If no malformed blocks, parsing was successful, break loop
                                break 
                            except Exception as parse_err: # Catch any error during parsing/validation
                                # Capture details of the parsing error
                                parsing_error_details = {
                                    "type": "JSON_PARSE_ERROR",
                                    "message": str(parse_err),
                                    "raw_string_snippet": self.parser._escape_json_string_value(raw_response_text[:1000] + ("..." if len(raw_response_text) > 1000 else ""))
                                }
                                # Corrected: Call self._update_status, not self.self._update_status
                                self._update_status(f"[yellow]Warning: JSON parsing failed for '{current_persona_name}'. Attempting self-correction (Attempt {json_repair_attempts + 1}/{MAX_JSON_REPAIR_ATTEMPTS + 1}). Error: {parse_err}[/yellow]", state="running")
                        else:
                            # Handle empty response case
                            parsing_error_details = {
                                "type": "EMPTY_RESPONSE",
                                "message": "LLM returned empty response for final answer.",
                                "raw_string_snippet": ""
                            }
                            # Corrected: Call self._update_status, not self.self._update_status
                            self._update_status(f"[yellow]Warning: LLM returned empty response for '{current_persona_name}'. Attempting self-correction (Attempt {json_repair_attempts + 1}/{MAX_JSON_REPAIR_ATTEMPTS + 1}).[/yellow]", state="running")

                        json_repair_attempts += 1
                        if json_repair_attempts > MAX_JSON_REPAIR_ATTEMPTS:
                            # Max attempts reached, return the best effort or error
                            error_message = f"LLM output parsing/validation failed for persona '{current_persona_name}' after {MAX_JSON_REPAIR_ATTEMPTS} self-correction attempts. See malformed_blocks for details."
                            self.intermediate_steps[f"{current_output_key}_Error"] = error_message
                            self._update_status(error_message, state="error")
                            # Construct a final error dictionary that is valid JSON
                            # The parse_and_validate method already returns a structured error dict.
                            # We just need to ensure it's the last one.
                            if parsed_data is None: # If no data was ever parsed
                                parsed_data = {
                                    "COMMIT_MESSAGE": "Parsing error (Max retries)",
                                    "RATIONALE": self.parser._escape_json_string_value(f"Failed to parse LLM output as JSON after multiple attempts. Error: {parsing_error_details['message']}\nRaw output: {raw_response_text[:500]}..."),
                                    "CODE_CHANGES": [],
                                    "CONFLICT_RESOLUTION": None,
                                    "UNRESOLVED_CONFLICT": None,
                                    "malformed_blocks": [parsing_error_details]
                                }
                            break # Exit the while loop and return the error dict
                        
                        # Prepare self-correction prompt
                        # Dynamically get the schema definition for the prompt
                        schema_json_str = json.dumps(schema_model_for_validation.model_json_schema(), indent=2)

                        # --- MODIFIED CORRECTION PROMPT ---
                        correction_prompt = (
                            f"Your previous output was malformed and could not be parsed as valid JSON. "
                            f"The specific error was: '{parsing_error_details['message']}'. "
                            f"This indicates a syntax issue, likely a missing comma between JSON elements or incorrect structure. "
                            f"Please review the expected JSON schema and regenerate the *entire* JSON object. "
                            f"Pay extreme attention to ensuring all key-value pairs and array elements are correctly separated by commas and that the overall structure is valid JSON.\n\n"
                            f"The expected JSON schema is:\n```json\n{schema_json_str}\n```\n\n"
                            f"Your previous malformed output snippet:\n```\n{raw_response_text}\n```\n\n" # Pass raw_response_text as is
                            f"Please regenerate the JSON object ensuring strict adherence to the schema and valid JSON syntax. Do NOT include any conversational text or markdown outside the JSON block."
                        )
                        # --- END MODIFIED CORRECTION PROMPT ---
                        
                        # Store the correction prompt for debugging/intermediate steps
                        self.intermediate_steps[f"{current_output_key}_Correction_Prompt_Attempt_{json_repair_attempts}"] = correction_prompt

                        # Call LLM again with correction prompt
                        raw_response_text, input_tokens_corr, output_tokens_corr = self.gemini_provider.generate(
                            prompt=correction_prompt,
                            system_prompt=current_persona.system_prompt,
                            temperature=current_persona.temperature, # Keep original temperature
                            max_tokens=max_output_for_request, # Keep original max tokens
                            _status_callback=self.status_callback
                        )
                        tokens_used_in_step += (input_tokens_corr + output_tokens_corr) # Add tokens from correction attempt
                        cost_this_step += self.gemini_provider.calculate_usd_cost(input_tokens_corr, output_tokens_corr) # Add cost
                        self.intermediate_steps[f"{current_output_key}_Correction_Output_Attempt_{json_repair_attempts}"] = raw_response_text
                        self._update_status(f"Self-correction attempt {json_repair_attempts} for '{current_persona_name}' completed. Used {input_tokens_corr + output_tokens_corr} tokens in this sub-step.",
                                            current_total_tokens=self.cumulative_token_usage + tokens_used_in_step, # Show cumulative including this sub-step
                                            current_total_cost=self.cumulative_usd_cost + cost_this_step) # Show cumulative including this sub-step
                    
                    # After the loop, parsed_data holds the successfully parsed data or the final error dict
                    self.intermediate_steps[current_output_key] = parsed_data
                    self.intermediate_steps[f"{current_persona_name}_Tokens_Used"] = tokens_used_in_step # Store tokens for UI display
                    self.cumulative_token_usage += tokens_used_in_step
                    self.cumulative_usd_cost += cost_this_step
                    if is_final_answer_step: self.final_answer = parsed_data
                    self._update_status(f"{current_persona_name} completed. Used {tokens_used_in_step} tokens.",
                                        current_total_tokens=self.cumulative_token_usage,
                                        current_total_cost=self.cumulative_usd_cost)
                    return parsed_data
                # --- END JSON SELF-CORRECTION LOOP ---
                else: # No schema_model_for_validation provided, just return raw text
                    if raw_response_text:
                        self.intermediate_steps[current_output_key] = raw_response_text
                    else:
                        self.intermediate_steps[current_output_key] = "[INFO] LLM returned empty response."
                        tokens_used_in_step = input_tokens # If response is empty, only count input tokens
                        cost_this_step = self.gemini_provider.calculate_usd_cost(input_tokens, 0)
                    
                    self.intermediate_steps[f"{current_persona_name}_Tokens_Used"] = tokens_used_in_step # Store tokens for UI display
                    self.cumulative_token_usage += tokens_used_in_step
                    self.cumulative_usd_cost += cost_this_step
                    if update_current_thought: self.current_thought = raw_response_text
                    self._update_status(f"{current_persona_name} completed. Used {tokens_used_in_step} tokens.",
                                        current_total_tokens=self.cumulative_token_usage,
                                        current_total_cost=self.cumulative_usd_cost)
                    return raw_response_text
            
            except LLMProviderError as e: # Catch API errors and other LLM provider issues
                error_msg = f"[ERROR] Persona '{current_persona_name}' failed: {e}"
                self.intermediate_steps[current_output_key] = error_msg
                self._update_status(error_msg, state="error")
                if api_retry_attempt == max_retries_on_fail:
                    raise # Re-raise if max retries are exceeded
            except Exception as e: # Catch any other unexpected errors
                error_msg = f"[ERROR] Unexpected error during '{current_persona_name}' execution: {e}"
                self.intermediate_steps[current_output_key] = error_msg
                self._update_status(error_msg, state="error")
                if api_retry_attempt == max_retries_on_fail:
                    raise # Re-raise if max retries are exceeded
        
        # If the loop finishes without returning or raising, it means max retries were exceeded.
        raise LLMUnexpectedError(f"Max retries exceeded for persona '{persona_name}'.")

    def run_debate(self, max_turns: int = 5) -> Tuple[Any, Dict[str, Any]]:
        """Executes the full Socratic debate loop."""
        if not self.personas:
            return {"error": "No personas loaded for the selected domain. Cannot run debate."}, {}
        self._update_status("Starting Socratic Arbitration Loop...",
                            current_total_tokens=self.cumulative_token_usage,
                            current_total_cost=self.cumulative_usd_cost)
        
        # Prepare context using the new strategic method
        context_string = self.prepare_context()
        
        # Analyze context, now using the improved _analyze_codebase_context
        codebase_analysis_summary = self._analyze_codebase_context()
        
        def visionary_prompt_gen():
            # Ensure analysis_str is properly formatted if summary is available
            # Check if analysis was already run and stored in intermediate_steps
            analysis_data = self.intermediate_steps.get("Context_Analysis_Output")
            if analysis_data:
                analysis_str = json.dumps(analysis_data, indent=2)
            else:
                # If analysis failed or was skipped, use a default summary
                analysis_str = json.dumps(self._get_default_analysis_summary(), indent=2)
            
            return (f"USER PROMPT: {self.initial_prompt}\n\nCODEBASE CONTEXT ANALYSIS:\n{analysis_str}\n\n"
                    f"CODEBASE CONTEXT:\n{context_string}\n\nINSTRUCTIONS:\n"
                    f"1. **Analyze the provided codebase context and its analysis thoroughly.** Understand its structure, style, patterns, dependencies, and overall logic, incorporating insights from the analysis.\n"
                    f"2. **Propose an initial implementation strategy or code snippet.** Your proposal should be consistent with the existing codebase and address any identified concerns from the analysis.\n"
                    f"3. **Ensure your proposed code fits naturally into the existing architecture and follows its conventions.** Use the provided `GeminiProvider` and `SocraticDebate` classes as examples of how to integrate.")
        try:
            visionary_output = self._execute_persona_step("Visionary_Generator", visionary_prompt_gen, "Visionary_Generator_Output", update_current_thought=True)
            def skeptical_prompt_gen():
                return f"Critique the following proposal from a highly skeptical, risk-averse perspective. Identify potential failure points, architectural flaws, or critical vulnerabilities:\n\n{visionary_output}"
            skeptical_critique = self._execute_persona_step("Skeptical_Generator", skeptical_prompt_gen, "Skeptical_Critique")
            
            domain_critiques_text = ""
            
            # Define personas that are explicitly handled before or after this loop
            explicitly_handled_personas = {
                "Visionary_Generator", "Skeptical_Generator", "Constructive_Critic",
                "Impartial_Arbitrator", "Devils_Advocate", "Generalist_Assistant",
                "Context_Aware_Assistant"
            }
            
            # Iterate over the personas *active in the current domain* (self.personas)
            # and exclude the ones that are explicitly called before or after this loop.
            sorted_domain_personas = sorted(self.personas.keys())

            for persona_name_in_domain in sorted_domain_personas:
                if persona_name_in_domain not in explicitly_handled_personas:
                    def expert_prompt_gen(name=persona_name_in_domain, proposal=visionary_output, skeptical_critique=skeptical_critique):
                        # Provide all previous critiques to the domain-specific persona
                        return (
                            f"As a {name.replace('_', ' ')}, analyze the following proposal and existing critiques from your expert perspective. "
                            f"Original Proposal:\n{proposal}\n\n"
                            f"Skeptical Critique:\n{skeptical_critique}\n\n"
                            f"Identify specific points of concern, risks, or areas for improvement relevant to your domain. "
                            f"Your insights will be crucial for subsequent synthesis and refinement steps, so be thorough and specific. "
                            f"Present your analysis in a structured format, using clear headings or bullet points for 'Concerns' and 'Recommendations'.\n\n"
                            f"Focus on your persona's specific expertise (e.g., for Code Architect: modularity, scalability; for Security Auditor: vulnerabilities, threat model; for DevOps: CI/CD, monitoring; for Test Engineer: test coverage, testability)."
                        )
                    critique = self._execute_persona_step(persona_name_in_domain, expert_prompt_gen, f"{persona_name_in_domain}_Critique")
                    domain_critiques_text += f"\n\n--- {persona_name_in_domain.replace('_', ' ')} Critique ---\n{critique}"
            
            def constructive_prompt_gen():
                return (
                    f"Original Proposal:\n{visionary_output}\n\n"
                    f"Skeptical Critique:\n{skeptical_critique}\n\n"
                    f"{domain_critiques_text}\n\n"
                    # Removed reference to constructive_feedback here
                    f"Synthesize all the above information into a single, balanced, and definitive final answer. Adhere strictly to the JSON output format and escaping rules provided in your system prompt. Prioritize enhancements to reasoning quality, robustness, efficiency, and developer maintainability based on the Pareto principle.")
            constructive_feedback = self._execute_persona_step("Constructive_Critic", constructive_prompt_gen, "Constructive_Critic_Output")
            
            # The Impartial_Arbitrator's system prompt already contains the JSON schema and instructions.
            # This prompt should provide the context for synthesis.
            def arbitrator_prompt_gen():
                return (
                    f"Original Proposal:\n{visionary_output}\n\n"
                    f"Skeptical Critique:\n{skeptical_critique}\n\n"
                    f"{domain_critiques_text}\n\n"
                    f"Constructive Critic Feedback:\n{constructive_feedback}\n\n" # This is the output from the previous step
                    f"Synthesize all the above information into a single, balanced, and definitive final answer. Adhere strictly to the JSON output format and escaping rules provided in your system prompt. Prioritize enhancements to reasoning quality, robustness, efficiency, and developer maintainability based on the Pareto principle.")
            
            # The Impartial_Arbitrator will receive all previous outputs and synthesize the final answer.
            arbitrator_output = self._execute_persona_step(
                "Impartial_Arbitrator",
                arbitrator_prompt_gen,
                "Impartial_Arbitrator_Output",
                schema_model_for_validation=LLMOutput, # Assuming LLMOutput is the Pydantic model for the final JSON
                update_current_thought=True,
                is_final_answer_step=True
            )
            
            # The final_answer is already updated within _execute_persona_step if is_final_answer_step is True.
            # We just need to return it along with the intermediate steps.
            return self.final_answer, self.intermediate_steps

        except (TokenBudgetExceededError, LLMProviderError, ValueError, RuntimeError, Exception) as e:
            error_message = f"[ERROR] Socratic debate failed: {e}"
            self.intermediate_steps["Socratic_Debate_Error"] = error_message
            self._update_status(error_message, state="error")
            return {"error": str(e)}, self.intermediate_steps

    def _get_default_analysis_summary(self) -> Dict[str, Any]:
        """Returns a default summary if context analysis fails or is skipped."""
        return {
            "key_modules": [],
            "security_concerns": [],
            "architectural_patterns": [],
            "performance_bottlenecks": [],
            "malformed_blocks": [] # Ensure this key exists even if empty
        }