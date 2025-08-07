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
# These are now defined in src/exceptions.py as per suggestion 1
# For this file, we'll import them.
from src.exceptions import (
    ChimeraError,
    ContextAnalysisError,
    LLMResponseValidationError,
    TokenBudgetExceededError
)

# --- Context Relevance Analyzer ---
# This is now defined in src/context/context_analyzer.py as per suggestion 2
from src.context.context_analyzer import ContextRelevanceAnalyzer

# --- Persona Router ---
# This is now defined in src/persona/routing.py as per suggestion 3
from src.persona.routing import PersonaRouter

# --- Response Validator ---
# This is now defined in src/utils/response_validator.py as per suggestion 5
from src.utils.response_validator import LLMResponseValidator

logger = logging.getLogger(__name__)

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
                 gemini_provider: Optional[GeminiProvider] = None,
                 domain: str = "General",
                 status_callback: Callable = None,
                 rich_console: Optional[Console] = None,
                 codebase_context: Optional[Dict[str, str]] = None,
                 context_token_budget_ratio: float = 0.25):
        
        self.initial_prompt = initial_prompt
        self.max_total_tokens_budget = max_total_tokens_budget
        self.model_name = model_name
        self.personas = personas # Personas for the current domain
        self.domain = domain
        self.all_personas = all_personas # All loaded personas
        self.persona_sets = persona_sets
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
        self.current_thought = initial_prompt # This might be replaced by persona-specific thoughts
        self.final_answer = "Process did not complete."
        self.codebase_context = codebase_context
        
        # Initialize context analyzer
        self.context_analyzer = None
        if self.codebase_context:
            self.context_analyzer = ContextRelevanceAnalyzer()
            self.context_analyzer.compute_file_embeddings(self.codebase_context)
        
        # Initialize persona router
        self.persona_router = PersonaRouter(self.all_personas)
        
        # Initialize retry count for persona steps
        self.persona_step_retry_count = 0
        self.MAX_PERSONA_STEP_RETRIES = 2 # Max retries for a single persona step

    def _update_status(self, message: str, **kwargs):
        """Helper to print to console and call Streamlit callback."""
        self.rich_console.print(message)
        if self.status_callback:
            # Ensure kwargs are passed correctly, especially for token/cost updates
            status_kwargs = {k: v for k, v in kwargs.items() if k in ['state', 'expanded', 'current_total_tokens', 'current_total_cost', 'estimated_next_step_tokens', 'estimated_next_step_cost']}
            self.status_callback(message=message, **status_kwargs)

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
    def _handle_token_budget_exceeded(self, error: TokenBudgetExceededError) -> Dict[str, Any]:
        """Handles token budget exceeded errors by returning a fallback response."""
        self._update_status(
            f"[red]Token budget exceeded: {error.message}. Cannot proceed with detailed analysis. Returning default summary.[/red]",
            state="error"
        )
        # Return a structured error response that mimics a failed persona step
        return {
            "error": "TOKEN_BUDGET_EXCEEDED",
            "message": error.message,
            "details": error.details,
            "original_response": "Context analysis could not be performed due to token limits."
        }

    # --- MODIFIED METHOD: prepare_context ---
    def prepare_context(self) -> str:
        """Intelligently prioritizes and truncates codebase context based on semantic relevance."""
        if not self.codebase_context:
            return "No codebase context provided."
        
        context_budget = int(self.max_total_tokens_budget * self.context_token_budget_ratio)
        
        # Identify most relevant files using the analyzer
        relevant_files_with_scores = []
        if self.context_analyzer:
            relevant_files_with_scores = self.context_analyzer.get_relevant_files(
                self.initial_prompt, 
                top_k=min(len(self.codebase_context), 50)
            )
        else:
            # Fallback to simple keyword matching if analyzer is not initialized
            self._update_status("[yellow]Warning: Context analyzer not initialized. Falling back to basic keyword relevance.[/yellow]")
            relevant_files_with_scores = self._identify_relevant_files_basic(context_budget) # Use a basic fallback
            
        # Build context with prioritized files, respecting budget
        context_str_parts = []
        current_tokens = 0
        selected_file_paths = []
        
        for path, score in relevant_files_with_scores:
            content = self.codebase_context[path]
            header = f"\n=== {path} ===\n"
            header_tokens = self.gemini_provider.count_tokens(header, "", _status_callback=self.status_callback)
            
            # Check if we have enough budget for this file's header and content
            remaining_budget = context_budget - current_tokens
            if remaining_budget <= 0:
                self._update_status(f"Skipping file '{path}' due to context token budget exhaustion.")
                break
                
            # Strategically truncate file content to preserve important sections
            # Estimate tokens for content to ensure it fits within remaining budget
            estimated_content_tokens = self.gemini_provider.count_tokens(content, "", _status_callback=self.status_callback)
            
            # If header + content exceeds budget, truncate content
            if header_tokens + estimated_content_tokens > remaining_budget:
                file_content_to_add = self._truncate_file_content(content, remaining_budget - header_tokens)
            else:
                file_content_to_add = content
            
            # Append if there's actual content to add
            if file_content_to_add.strip():
                context_str_parts.append(header + file_content_to_add)
                selected_file_paths.append(path)
                current_tokens += header_tokens + self.gemini_provider.count_tokens(file_content_to_add, "", _status_callback=self.status_callback)
            
            # Break if we've exhausted the budget
            if current_tokens >= context_budget:
                break
        
        # Add summary of excluded files for transparency
        excluded_files = set(self.codebase_context.keys()) - set(selected_file_paths)
        if excluded_files and len(context_str_parts) > 0:
            excluded_summary = f"\n\n=== EXCLUDED FILES ({len(excluded_files)} total) ===\n"
            excluded_summary += "The following relevant files were not included due to token constraints:\n"
            excluded_summary += ", ".join(list(excluded_files)[:10])  # Show first 10
            if len(excluded_files) > 10:
                excluded_summary += f", and {len(excluded_files) - 10} more..."
            
            # Only add if we have tokens for it
            excluded_summary_tokens = self.gemini_provider.count_tokens(excluded_summary, "", _status_callback=self.status_callback)
            if current_tokens + excluded_summary_tokens <= context_budget:
                context_str_parts.append(excluded_summary)
                current_tokens += excluded_summary_tokens
        
        final_context_string = "".join(context_str_parts)
        self._update_status(
            f"Prepared codebase context using {current_tokens} "
            f"tokens from {len(selected_file_paths)} relevant files (budget: {context_budget} tokens)."
        )
        return final_context_string

    # --- NEW HELPER METHOD FOR prepare_context (basic fallback) ---
    def _identify_relevant_files_basic(self, context_budget: int) -> List[Tuple[str, float]]:
        """Basic keyword-based relevance for fallback."""
        prompt_keywords = self._extract_keywords_from_prompt()
        file_scores = []
        for path, content in self.codebase_context.items():
            score = 0
            path_lower = path.lower()
            for keyword in prompt_keywords:
                if keyword in path_lower:
                    score += 3
            content_sample = content[:1000].lower()
            for keyword in prompt_keywords:
                if keyword in content_sample:
                    score += 2
            file_scores.append((path, score))
        
        file_scores.sort(key=lambda x: x[1], reverse=True)
        return [(path, score) for path, score in file_scores if score > 0] or list(self.codebase_context.keys())[:5]

    # --- NEW HELPER METHOD FOR prepare_context ---
    def _extract_keywords_from_prompt(self) -> List[str]:
        """Extracts meaningful keywords from the user prompt using simple NLP techniques."""
        prompt = self.initial_prompt.lower()
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'is', 'are', 'was', 'were', 'in', 'on', 
                     'at', 'to', 'for', 'with', 'by', 'of', 'as', 'it', 'this', 'that', 'code', 'file'}
        words = [word.strip('.,!?()[]{}') for word in prompt.split() 
                 if word.lower() not in stop_words and len(word) > 2]
        return list(set(words))

    # --- NEW HELPER METHOD FOR prepare_context ---
    def _truncate_file_content(self, content: str, max_tokens: int) -> str:
        """Strategically truncates file content to preserve the most important sections."""
        if content.endswith('.py'):
            first_part = content[:max_tokens//2]
            if len(content) > max_tokens//2:
                method_pattern = r'def\s+\w+\s*\('
                method_matches = list(re.finditer(method_pattern, content))
                
                if method_matches:
                    selected_methods = []
                    for i, match in enumerate(method_matches[:3]):
                        start_idx = match.start()
                        end_idx = len(content)
                        if i < len(method_matches) - 1:
                            end_idx = method_matches[i+1].start()
                        
                        method_content = content[start_idx:end_idx].strip()
                        if len(method_content) > 500:
                            method_content = method_content[:500] + "\n    # [Truncated for token budget]"
                        selected_methods.append(method_content)
                    
                    return first_part + "\n\n# Key methods:\n" + "\n\n".join(selected_methods)
            return first_part
        
        return content[:max_tokens]

    # --- MODIFIED METHOD: _analyze_codebase_context ---
    def _analyze_codebase_context(self) -> Dict[str, Any]:
        """
        Analyzes the codebase context using a dedicated persona.
        """
        if not self.codebase_context:
            self._update_status("[yellow]No codebase context provided for analysis.[/yellow]")
            return self._get_default_analysis_summary()
        
        # Check if Context_Aware_Assistant persona is available
        if "Context_Aware_Assistant" not in self.all_personas:
            self._update_status("[yellow]Context_Aware_Assistant persona not found. Skipping context analysis.[/yellow]")
            return self._get_default_analysis_summary()
        
        self._update_status("Analyzing codebase context with Context_Aware_Assistant...")
        
        # Prepare context string using the new strategic method
        context_string_for_analysis = self.prepare_context()
        
        def analysis_prompt_gen():
            analysis_data = self.intermediate_steps.get("Context_Analysis_Output")
            analysis_str = json.dumps(analysis_data, indent=2) if analysis_data else json.dumps(self._get_default_analysis_summary(), indent=2)
            
            return (f"CODEBASE CONTEXT:\n{context_string_for_analysis}\n\n"
                    f"INSTRUCTIONS:\n"
                    f"Analyze the provided codebase context thoroughly. Understand its structure, style, patterns, dependencies, and overall logic.\n"
                    f"Provide a concise summary of the codebase in the specified JSON format.")

        analysis_output_key = "Context_Analysis_Output"
        try:
            analysis_response = self._execute_persona_step(
                "Context_Aware_Assistant",
                analysis_prompt_gen,
                analysis_output_key,
                schema_model_for_validation=ContextAnalysisOutput,
                update_current_thought=False,
                is_final_answer_step=False
            )
            
            # Check for specific error conditions or malformed blocks
            if isinstance(analysis_response, dict) and analysis_response.get("error"):
                raise ContextAnalysisError(f"Persona returned an error: {analysis_response.get('message', 'Unknown error')}")
            if isinstance(analysis_response, dict) and analysis_response.get("malformed_blocks"):
                raise MalformedContextError("Context analysis response contained malformed blocks", details=analysis_response["malformed_blocks"])
                
            self._update_status("Codebase context analysis successful.")
            self.intermediate_steps[analysis_output_key] = analysis_response
            return analysis_response
                
        except (MalformedContextError, ContextAnalysisError, TokenBudgetExceededError) as e:
            self._update_status(f"[yellow]Warning: {str(e)}. Using default summary.[/yellow]")
            logger.warning(f"Context analysis issue: {str(e)}")
            if isinstance(e, TokenBudgetExceededError):
                return self._handle_token_budget_exceeded(e)
            return self._get_default_analysis_summary()
            
        except Exception as e: # Catch-all for unexpected errors with detailed logging
            self._update_status(f"[red]Unexpected error during context analysis: {str(e)}. Using default summary.[/red]")
            logger.exception("Unexpected error during context analysis")
            return self._get_default_analysis_summary()

    # --- MODIFIED METHOD: _execute_persona_step ---
    def _execute_persona_step(self, persona_name: str, prompt_gen_func: Callable[[], str], step_name: str, 
                              schema_model_for_validation: Optional[Type[BaseModel]] = None, 
                              update_current_thought: bool = False,
                              is_final_answer_step: bool = False) -> Dict[str, Any]:
        """
        Execute a single persona step with comprehensive error handling, validation,
        and retry mechanisms using LLMResponseValidator.
        """
        persona = self._get_persona(persona_name)
        
        # Reset retry count for each new persona step
        self.persona_step_retry_count = 0
        
        while self.persona_step_retry_count <= self.MAX_PERSONA_STEP_RETRIES:
            try:
                # Generate prompt for the current step
                prompt = prompt_gen_func()
                
                # Calculate token budget for this request
                estimated_input_tokens = self.gemini_provider.count_tokens(prompt=prompt, system_prompt=persona.system_prompt, _status_callback=self.status_callback)
                remaining_budget = self.max_total_tokens_budget - self.cumulative_token_usage
                
                # Ensure we don't exceed total budget, and respect persona's max_tokens
                max_output_for_request = max(0, min(persona.max_tokens, remaining_budget - estimated_input_tokens))
                
                if estimated_input_tokens >= remaining_budget:
                    raise TokenBudgetExceededError(
                        current_tokens=self.cumulative_token_usage + estimated_input_tokens,
                        budget=self.max_total_tokens_budget,
                        details={"persona": persona_name, "step": step_name}
                    )
                
                # Update status with estimated tokens for the next step
                estimated_total_tokens_this_step = estimated_input_tokens + max_output_for_request
                estimated_next_step_cost = self.gemini_provider.calculate_usd_cost(estimated_input_tokens, max_output_for_request)
                
                self._update_status(
                    f"Executing persona: {persona_name}...",
                    current_total_tokens=self.cumulative_token_usage,
                    current_total_cost=self.cumulative_usd_cost,
                    estimated_next_step_tokens=estimated_total_tokens_this_step,
                    estimated_next_step_cost=estimated_next_step_cost
                )
                
                # Call the LLM provider
                raw_response_text, input_tokens, output_tokens = self.gemini_provider.generate(
                    prompt=prompt,
                    system_prompt=persona.system_prompt,
                    temperature=persona.temperature,
                    max_tokens=max_output_for_request,
                    _status_callback=self.status_callback
                )
                
                tokens_used_in_step = input_tokens + output_tokens
                cost_this_step = self.gemini_provider.calculate_usd_cost(input_tokens, output_tokens)
                
                # --- Validation and Correction ---
                validated_response_data = None
                if schema_model_for_validation:
                    try:
                        # Use LLMResponseValidator for parsing and validation
                        validated_response_data = LLMResponseValidator.validate_response(
                            raw_response_text, 
                            schema_model_for_validation
                        )
                        # If validation succeeds, we are done with this step
                        self.intermediate_steps[step_name] = validated_response_data
                        self.cumulative_token_usage += tokens_used_in_step
                        self.cumulative_usd_cost += cost_this_step
                        if update_current_thought: self.current_thought = raw_response_text # Or validated_response_data?
                        self._update_status(f"{persona_name} completed. Used {tokens_used_in_step} tokens.",
                                            current_total_tokens=self.cumulative_token_usage,
                                            current_total_cost=self.cumulative_usd_cost)
                        return validated_response_data
                        
                    except LLMResponseValidationError as validation_error:
                        # If validation fails, prepare for retry
                        self.persona_step_retry_count += 1
                        if self.persona_step_retry_count > self.MAX_PERSONA_STEP_RETRIES:
                            self.logger.error(f"Max retries ({self.MAX_PERSONA_STEP_RETRIES}) exceeded for {persona_name} due to validation errors.")
                            # Return a structured error response
                            return {
                                "error": "VALIDATION_FAILED",
                                "message": f"Failed to validate response for {persona_name} after multiple retries.",
                                "details": validation_error.details,
                                "original_response": raw_response_text[:500] + "..." if raw_response_text else "N/A"
                            }
                        
                        # Generate a correction prompt
                        correction_prompt = LLMResponseValidator.generate_correction_prompt(
                            validation_error.details.get("invalid_response"),
                            schema_model_for_validation,
                            str(validation_error)
                        )
                        
                        self._update_status(
                            f"[yellow]Warning: {str(validation_error)}. Attempting correction (retry {self.persona_step_retry_count}/{self.MAX_PERSONA_STEP_RETRIES})[/yellow]",
                            state="running"
                        )
                        self.logger.info(f"Attempting to correct invalid response for {persona_name}")
                        
                        # Store the correction prompt for debugging/transparency
                        self.intermediate_steps[f"{step_name}_Correction_Prompt_Attempt_{self.persona_step_retry_count}"] = correction_prompt
                        
                        # Call LLM again with the correction prompt
                        # Note: This recursive call might be problematic if not managed carefully.
                        # A better approach might be to re-assign prompt_gen_func or pass the correction prompt directly.
                        # For now, let's simulate by calling generate again with the correction prompt.
                        # We need to ensure token counts are handled correctly for this retry.
                        
                        # Re-calculate budget for the correction call
                        correction_input_tokens = self.gemini_provider.count_tokens(prompt=correction_prompt, system_prompt=persona.system_prompt, _status_callback=self.status_callback)
                        remaining_budget_for_correction = self.max_total_tokens_budget - (self.cumulative_token_usage + tokens_used_in_step) # Budget remaining after initial failed attempt
                        max_output_for_correction = max(0, min(persona.max_tokens, remaining_budget_for_correction - correction_input_tokens))
                        
                        if correction_input_tokens >= remaining_budget_for_correction:
                            raise TokenBudgetExceededError(
                                current_tokens=self.cumulative_token_usage + tokens_used_in_step + correction_input_tokens,
                                budget=self.max_total_tokens_budget,
                                details={"persona": persona_name, "step": step_name, "retry_attempt": self.persona_step_retry_count}
                            )
                        
                        # Call generate for the correction
                        corrected_raw_text, correction_input_tokens, correction_output_tokens = self.gemini_provider.generate(
                            prompt=correction_prompt,
                            system_prompt=persona.system_prompt,
                            temperature=persona.temperature,
                            max_tokens=max_output_for_request, # Use original max_tokens for correction
                            _status_callback=self.status_callback
                        )
                        
                        # Add tokens used for correction to the step's total
                        tokens_used_in_step += (correction_input_tokens + correction_output_tokens)
                        cost_this_step += self.gemini_provider.calculate_usd_cost(correction_input_tokens, correction_output_tokens)
                        
                        self.intermediate_steps[f"{step_name}_Corrected_Output_Attempt_{self.persona_step_retry_count}"] = corrected_raw_text
                        
                        # Validate the corrected response
                        validated_corrected_response_data = LLMResponseValidator.validate_response(
                            corrected_raw_text, 
                            schema_model_for_validation
                        )
                        
                        # If correction succeeded, update and return
                        self.intermediate_steps[step_name] = validated_corrected_response_data
                        self.cumulative_token_usage += tokens_used_in_step
                        self.cumulative_usd_cost += cost_this_step
                        if update_current_thought: self.current_thought = corrected_raw_text
                        self._update_status(f"{persona_name} corrected. Used {tokens_used_in_step} tokens in total for this step.",
                                            current_total_tokens=self.cumulative_token_usage,
                                            current_total_cost=self.cumulative_usd_cost)
                        return validated_corrected_response_data
                        
                else: # No schema validation required, return raw text
                    self.intermediate_steps[step_name] = raw_response_text
                    self.cumulative_token_usage += tokens_used_in_step
                    self.cumulative_usd_cost += cost_this_step
                    if update_current_thought: self.current_thought = raw_response_text
                    self._update_status(f"{persona_name} completed. Used {tokens_used_in_step} tokens.",
                                        current_total_tokens=self.cumulative_token_usage,
                                        current_total_cost=self.cumulative_usd_cost)
                    return raw_response_text
            
            except TokenBudgetExceededError as e:
                # Handle token budget exceeded specifically
                return self._handle_token_budget_exceeded(e)
                
            except LLMProviderError as e: # Catch API errors and other LLM provider issues
                self.persona_step_retry_count += 1
                error_msg = f"[ERROR] LLM Provider Error during '{persona_name}': {str(e)}"
                self.intermediate_steps[step_name] = {"error": "LLM_PROVIDER_ERROR", "message": error_msg}
                self._update_status(error_msg, state="error")
                
                if self.persona_step_retry_count > self.MAX_PERSONA_STEP_RETRIES:
                    self.logger.error(f"Max retries ({self.MAX_PERSONA_STEP_RETRIES}) exceeded for {persona_name} due to LLM Provider Error.")
                    raise ChimeraError(f"Failed to execute persona '{persona_name}' after multiple LLM provider errors.", details={"original_error": str(e)}) from e
                
                # Implement backoff for retries
                backoff_time = min(self.DEFAULT_MAX_RETRIES * (2 ** (self.persona_step_retry_count - 1)), self.MAX_BACKOFF_SECONDS)
                jitter = random.uniform(0, 0.5 * backoff_time)
                sleep_time = backoff_time + jitter
                self._update_status(f"[yellow]Retrying {persona_name} in {sleep_time:.2f} seconds...[/yellow]", state="running")
                time.sleep(sleep_time)
                
            except Exception as e: # Catch any other unexpected errors
                error_msg = f"[ERROR] Unexpected error during '{persona_name}' execution: {str(e)}"
                self.intermediate_steps[step_name] = {"error": "UNEXPECTED_ERROR", "message": error_msg}
                self._update_status(error_msg, state="error")
                self.logger.exception(f"Unexpected error during persona step '{persona_name}'")
                
                # Re-raise as ChimeraError for consistent handling
                raise ChimeraError(f"Unexpected error during persona execution: {str(e)}", details={"persona": persona_name, "step": step_name, "traceback": traceback.format_exc()}) from e
        
        # If loop finishes without returning, it means max retries were exceeded for validation
        # This case is handled within the validation block, but as a safeguard:
        raise ChimeraError(f"Failed to get a valid response for persona '{persona_name}' after all retries.")

    def _get_default_analysis_summary(self) -> Dict[str, Any]:
        """Returns a default summary if context analysis fails or is skipped."""
        return {
            "key_modules": [],
            "security_concerns": [],
            "architectural_patterns": [],
            "performance_bottlenecks": [],
            "malformed_blocks": [] # Ensure this key exists even if empty
        }

    # --- MODIFIED METHOD: run_debate ---
    def run_debate(self, initial_prompt: str, codebase_context: Optional[Dict[str, str]] = None) -> Tuple[Any, Dict[str, Any]]:
        """Executes the full Socratic debate loop using dynamic persona routing."""
        self._initialize_debate(initial_prompt, codebase_context)
        
        # Determine the optimal persona sequence using the router
        persona_sequence = self.persona_router.determine_persona_sequence(initial_prompt, self.intermediate_steps)
        
        self._update_status(f"[blue]Optimal persona sequence determined: {', '.join(persona_sequence)}[/blue]")
        
        # Execute personas in the determined sequence
        for persona_name in persona_sequence:
            self._update_status(f"[green]Executing {persona_name} persona...[/green]")
            
            # Define prompt generation functions for each persona type
            prompt_gen_func = None
            step_name = f"{persona_name}_Output"
            schema_model = None
            update_thought = True
            is_final_step = False
            
            if persona_name == "Visionary_Generator":
                def visionary_prompt_gen():
                    context_analysis_data = self.intermediate_steps.get("Context_Analysis_Output", self._get_default_analysis_summary())
                    context_str = json.dumps(context_analysis_data, indent=2)
                    context_code_str = self.prepare_context()
                    return (f"USER PROMPT: {self.initial_prompt}\n\n"
                            f"CODEBASE CONTEXT ANALYSIS:\n{context_str}\n\n"
                            f"CODEBASE CONTEXT:\n{context_code_str}\n\n"
                            f"INSTRUCTIONS:\n"
                            f"1. Analyze the provided codebase context and its analysis thoroughly.\n"
                            f"2. Propose an initial implementation strategy or code snippet consistent with the existing codebase.\n"
                            f"3. Ensure your proposed code fits naturally into the existing architecture and follows its conventions.")
                prompt_gen_func = visionary_prompt_gen
                # The output of Visionary_Generator might be used by others, so it's not the final answer.
                
            elif persona_name == "Skeptical_Generator":
                def skeptical_prompt_gen():
                    visionary_output = self.intermediate_steps.get("Visionary_Generator_Output", "No visionary output available.")
                    return f"Critique the following proposal from a highly skeptical, risk-averse perspective. Identify potential failure points, architectural flaws, or critical vulnerabilities:\n\n{visionary_output}"
                prompt_gen_func = skeptical_prompt_gen
                step_name = "Skeptical_Critique" # More descriptive step name
                
            elif persona_name in ["Code_Architect", "Security_Auditor", "DevOps_Engineer", "Test_Engineer", "Constructive_Critic"]:
                # Generic handler for domain-specific critics
                def domain_critique_prompt_gen(p_name=persona_name):
                    visionary_output = self.intermediate_steps.get("Visionary_Generator_Output", "No visionary output available.")
                    skeptical_critique = self.intermediate_steps.get("Skeptical_Critique", "No skeptical critique available.")
                    
                    # Collect all previous critiques for context
                    previous_critiques = ""
                    for key, value in self.intermediate_steps.items():
                        if "_Critique" in key or "_Output" in key and "Generator" not in key and "Arbitrator" not in key and "Devils_Advocate" not in key:
                            previous_critiques += f"\n\n--- {key.replace('_Output', '').replace('_Critique', '')} ---\n{value}"
                    
                    return (f"As a {p_name.replace('_', ' ')}, analyze the following proposal and existing critiques from your expert perspective. "
                            f"Original Proposal:\n{visionary_output}\n\n"
                            f"Skeptical Critique:\n{skeptical_critique}\n\n"
                            f"Previous Critiques:\n{previous_critiques}\n\n"
                            f"Identify specific points of concern, risks, or areas for improvement relevant to your domain. "
                            f"Present your analysis in a structured format, using clear headings or bullet points for 'Concerns' and 'Recommendations'.")
                prompt_gen_func = domain_critique_prompt_gen
                step_name = f"{persona_name}_Critique"
                
            elif persona_name == "Impartial_Arbitrator":
                def arbitrator_prompt_gen():
                    visionary_output = self.intermediate_steps.get("Visionary_Generator_Output", "No visionary output available.")
                    skeptical_critique = self.intermediate_steps.get("Skeptical_Critique", "No skeptical critique available.")
                    
                    domain_critiques_text = ""
                    for key, value in self.intermediate_steps.items():
                        if "_Critique" in key or "_Output" in key:
                            # Exclude the final arbitrator and visionary/skeptical if they are not domain specific
                            if not (key.startswith("Visionary_") or key.startswith("Skeptical_") or key.startswith("Impartial_Arbitrator") or key.startswith("Devils_Advocate")):
                                domain_critiques_text += f"\n\n--- {key.replace('_Output', '').replace('_Critique', '')} ---\n{value}"
                    
                    constructive_feedback = self.intermediate_steps.get("Constructive_Critic_Output", "No constructive feedback available.") # Assuming Constructive_Critic is always run
                    
                    return (f"Original Proposal:\n{visionary_output}\n\n"
                            f"Skeptical Critique:\n{skeptical_critique}\n\n"
                            f"Domain Critiques:\n{domain_critiques_text}\n\n"
                            f"Constructive Critic Feedback:\n{constructive_feedback}\n\n"
                            f"Synthesize all the above information into a single, balanced, and definitive final answer. "
                            f"Adhere strictly to the JSON output format and escaping rules provided in your system prompt. "
                            f"Prioritize enhancements to reasoning quality, robustness, efficiency, and developer maintainability based on the Pareto principle.")
                prompt_gen_func = arbitrator_prompt_gen
                schema_model = LLMOutput # The final output should conform to LLMOutput
                is_final_step = True
                update_thought = True # Update current thought with final answer
                
            elif persona_name == "Devils_Advocate":
                def devils_advocate_prompt_gen():
                    arbitrator_output = self.intermediate_steps.get("Impartial_Arbitrator_Output", "No final answer available.")
                    return (f"You are the Devil's Advocate. Critically examine the final proposed answer and the process that led to it. "
                            f"Identify any fundamental flaws, unintended consequences, or overlooked risks in the final proposed answer or the overall approach.\n\n"
                            f"Final Proposed Answer:\n{arbitrator_output}\n\n"
                            f"Provide your critique in a structured markdown format.")
                prompt_gen_func = devils_advocate_prompt_gen
                step_name = "Devils_Advocate_Critique"
                
            else: # Generic persona execution
                def generic_prompt_gen(p_name=persona_name):
                    # This generic prompt might need refinement based on how custom personas are designed
                    # For now, it assumes the persona can handle the initial prompt and context.
                    context_analysis_data = self.intermediate_steps.get("Context_Analysis_Output", self._get_default_analysis_summary())
                    context_str = json.dumps(context_analysis_data, indent=2)
                    context_code_str = self.prepare_context()
                    return (f"USER PROMPT: {self.initial_prompt}\n\n"
                            f"CODEBASE CONTEXT ANALYSIS:\n{context_str}\n\n"
                            f"CODEBASE CONTEXT:\n{context_code_str}\n\n"
                            f"INSTRUCTIONS:\n"
                            f"As a {p_name.replace('_', ' ')}, analyze the provided information and contribute to the overall reasoning process.")
                prompt_gen_func = generic_prompt_gen
                step_name = f"{persona_name}_Output"
            
            # Execute the persona step
            if prompt_gen_func:
                response = self._execute_persona_step(
                    persona_name,
                    prompt_gen_func,
                    step_name,
                    schema_model_for_validation=schema_model,
                    update_current_thought=update_thought,
                    is_final_answer_step=is_final_step
                )
                
                # Store the response in intermediate steps
                self.intermediate_steps[step_name] = response
                
                # If it's the final step, store it as the final answer
                if is_final_step:
                    self.final_answer = response
            else:
                self._update_status(f"[yellow]Warning: No prompt generator defined for persona '{persona_name}'. Skipping.[/yellow]")
                self.intermediate_steps[step_name] = {"error": "NO_PROMPT_GENERATOR", "message": f"Persona '{persona_name}' skipped due to missing prompt generator."}

            # --- Dynamic Sequence Re-evaluation (Advanced Feature) ---
            # This part is complex and might require careful state management.
            # For now, we'll stick to the initial sequence determined by the router.
            # If implemented, it would look something like:
            # if len(self.intermediate_steps) > 3: # Example condition
            #     new_sequence = self.persona_router.determine_persona_sequence(initial_prompt, self.intermediate_steps)
            #     # Filter out already executed personas from the sequence
            #     executed_personas = {p.split('_')[0] for p in self.intermediate_steps.keys() if '_Output' in p or '_Critique' in p}
            #     persona_sequence = [p for p in new_sequence if p not in executed_personas]
            #     self._update_status(f"[blue]Re-evaluated persona sequence: {', '.join(persona_sequence)}[/blue]")

        # After all personas have run, format the final result
        return self._format_final_result()

    def _initialize_debate(self, initial_prompt: str, codebase_context: Optional[Dict[str, str]] = None):
        """Initializes debate state, context, and persona sequence."""
        self.cumulative_token_usage = 0
        self.cumulative_usd_cost = 0.0
        self.intermediate_steps = {}
        self.final_answer = "Process did not complete."
        self.current_thought = initial_prompt
        self.codebase_context = codebase_context
        
        # Re-initialize context analyzer if context changed or is new
        if self.codebase_context:
            if not self.context_analyzer:
                self.context_analyzer = ContextRelevanceAnalyzer()
            self.context_analyzer.compute_file_embeddings(self.codebase_context)
        else:
            self.context_analyzer = None # Clear analyzer if no context

        # Re-initialize persona router with potentially updated persona sets (though unlikely in a single run)
        self.persona_router = PersonaRouter(self.all_personas)
        
        # Analyze codebase context if provided
        if self.codebase_context:
            self._analyze_codebase_context()
        else:
            self._update_status("[yellow]No codebase context provided. Skipping context analysis.[/yellow]")
            self.intermediate_steps["Context_Analysis_Output"] = self._get_default_analysis_summary()

    def _format_final_result(self) -> Tuple[Any, Dict[str, Any]]:
        """Formats the final answer and returns it along with intermediate steps."""
        # Ensure the final answer is structured correctly, especially if it's the LLMOutput model
        if isinstance(self.final_answer, LLMOutput):
            final_output_data = self.final_answer.model_dump(by_alias=True)
        elif isinstance(self.final_answer, dict) and "error" in self.final_answer:
            # If the final answer is an error dictionary, ensure it's structured
            final_output_data = self.final_answer
        elif isinstance(self.final_answer, str):
            # If the final answer is just a string (e.g., from a non-schema persona), wrap it
            final_output_data = {
                "COMMIT_MESSAGE": "Final Answer",
                "RATIONALE": self.final_answer,
                "CODE_CHANGES": [],
                "CONFLICT_RESOLUTION": None,
                "UNRESOLVED_CONFLICT": None
            }
        else:
            # Fallback for unexpected final answer types
            final_output_data = {
                "COMMIT_MESSAGE": "Final Answer (Unstructured)",
                "RATIONALE": f"The final answer was not in a recognized format. Raw: {str(self.final_answer)[:500]}...",
                "CODE_CHANGES": [],
                "CONFLICT_RESOLUTION": None,
                "UNRESOLVED_CONFLICT": None
            }
            
        # Add total tokens and cost to intermediate steps for reporting
        self.intermediate_steps["Total_Tokens_Used"] = self.cumulative_token_usage
        self.intermediate_steps["Total_Estimated_Cost_USD"] = self.cumulative_usd_cost
        
        return final_output_data, self.intermediate_steps