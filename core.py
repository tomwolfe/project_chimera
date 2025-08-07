# core.py
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
import random # Needed for backoff jitter
from pathlib import Path
from collections import defaultdict
from typing import List, Dict, Tuple, Any, Callable, Optional, Type
from pydantic import BaseModel, Field, validator, model_validator, ValidationError

# --- LLM Provider and Tokenizer Imports ---
from llm_provider import GeminiProvider, LLMProviderError, GeminiAPIError, LLMUnexpectedError
# Assuming Tokenizer and GeminiTokenizer are available in src.tokenizers
from src.tokenizers import Tokenizer, GeminiTokenizer # Import the abstract base and concrete implementation

# --- Custom Exceptions ---
from src.exceptions import (
    ChimeraError,
    ContextAnalysisError,
    LLMResponseValidationError,
    TokenBudgetExceededError
)

# --- Context Relevance Analyzer ---
from src.context.context_analyzer import ContextRelevanceAnalyzer

# --- Persona Router ---
from src.persona.routing import PersonaRouter

# --- Response Validator ---
from src.utils.response_validator import LLMResponseValidator

# --- Configuration Settings ---
from src.config.settings import ChimeraSettings # Import the new settings model

# --- Models ---
from src.models import PersonaConfig, ReasoningFrameworkConfig, LLMOutput, ContextAnalysisOutput

logger = logging.getLogger(__name__)

class SocraticDebate:
    
    def __init__(self,
                 initial_prompt: str,
                 api_key: str,
                 max_total_tokens_budget: int, # This will be used by the tokenizer to determine allocation
                 model_name: str,
                 personas: Dict[str, PersonaConfig], # Personas active in the current domain
                 all_personas: Dict[str, PersonaConfig], # All loaded personas
                 persona_sets: Dict[str, List[str]],
                 gemini_provider: Optional[GeminiProvider] = None,
                 tokenizer: Optional[Tokenizer] = None, # Inject tokenizer
                 domain: str = "General",
                 status_callback: Callable = None,
                 rich_console: Optional[Console] = None,
                 codebase_context: Optional[Dict[str, str]] = None,
                 settings: Optional[ChimeraSettings] = None): # Inject settings
        
        self.initial_prompt = initial_prompt
        self.max_total_tokens_budget = max_total_tokens_budget # Store the overall budget
        self.model_name = model_name
        self.personas = personas # Personas for the current domain
        self.domain = domain
        self.all_personas = all_personas # All loaded personas
        self.persona_sets = persona_sets
        self.status_callback = status_callback
        
        # Initialize GeminiProvider if not provided
        if gemini_provider:
            self.gemini_provider = gemini_provider
        else:
            self.gemini_provider = GeminiProvider(api_key=api_key, model_name=model_name)
            
        # Initialize Tokenizer if not provided
        if tokenizer:
            self.tokenizer = tokenizer
        else:
            # Default to GeminiTokenizer, passing model_name for its configuration
            self.tokenizer = GeminiTokenizer(model_name=model_name)
            
        self.parser = LLMOutputParser() # Initialize without a specific schema model
        self.cumulative_token_usage = 0
        self.cumulative_usd_cost = 0.0
        self.intermediate_steps: Dict[str, Any] = {}
        self.rich_console = rich_console if rich_console else Console()
        self.current_thought = initial_prompt # This might be replaced by persona-specific thoughts
        self.final_answer = "Process did not complete."
        self.codebase_context = codebase_context
        
        # Load settings, using defaults if not provided
        self.settings = settings or ChimeraSettings()
        
        # Initialize token budgets based on settings and prompt analysis
        self.context_token_budget = 0
        self.debate_token_budget = 0
        
        # Initialize context analyzer
        self.context_analyzer = None
        if self.codebase_context:
            self.context_analyzer = ContextRelevanceAnalyzer()
            self.context_analyzer.compute_file_embeddings(self.codebase_context)
        
        # Initialize persona router
        self.persona_router = PersonaRouter(self.all_personas)
        
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

    # --- NEW METHOD FOR TOKEN ALLOCATION HEURISTICS ---
    def _determine_token_allocation(self, initial_prompt: str) -> Dict[str, int]:
        """
        Determine token allocation based on simple prompt analysis heuristics.
        Returns dictionary with context and debate token budgets.
        """
        # Check for code-related keywords
        code_keywords = ['code', 'function', 'class', 'implement', 'refactor', 'debug', 'python', 'javascript', 'api', 'script', 'module', 'file', 'repository', 'software', 'program', 'algorithm']
        is_code_task = any(keyword in initial_prompt.lower() for keyword in code_keywords)
        
        # Check for analysis keywords
        analysis_keywords = ['analyze', 'review', 'examine', 'critique', 'evaluate', 'report', 'audit', 'test', 'security']
        is_analysis_task = any(keyword in initial_prompt.lower() for keyword in analysis_keywords)
        
        # Adjust ratios based on task type
        # Use the ratio from settings as a base, but adjust dynamically
        base_context_ratio = self.settings.context_token_budget_ratio
        
        if is_code_task or is_analysis_task:
            # Increase context budget for code/analysis tasks
            context_ratio = min(0.4, base_context_ratio + 0.15) 
        else:
            # Slightly decrease context budget for general creative/discussion prompts
            context_ratio = max(0.1, base_context_ratio - 0.05)
        
        # Ensure debate gets minimum allocation and ratios sum to 1
        debate_ratio = max(0.6, 1.0 - context_ratio)
        
        # Ensure ratios are within reasonable bounds and sum to 1
        context_ratio = max(0.05, min(0.5, context_ratio)) # Clamp context ratio
        debate_ratio = 1.0 - context_ratio # Ensure they sum to 1.0

        return {
            'context': int(self.max_total_tokens_budget * context_ratio),
            'debate': int(self.max_total_tokens_budget * debate_ratio)
        }

    # --- MODIFIED METHOD: prepare_context ---
    def prepare_context(self) -> str:
        """Intelligently prioritizes and truncates codebase context based on semantic relevance."""
        if not self.codebase_context:
            return "No codebase context provided."
        
        # Use the dynamically determined context budget
        context_budget = self.context_token_budget
        
        # Identify most relevant files using the analyzer
        relevant_files_with_scores = []
        if self.context_analyzer:
            # Ensure top_k is not more than the number of available files
            top_k_files = min(len(self.codebase_context), 50) 
            relevant_files_with_scores = self.context_analyzer.get_relevant_files(
                self.initial_prompt, 
                top_k=top_k_files
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
            header_tokens = self.tokenizer.count_tokens(header) # Use injected tokenizer
            
            # Check if we have enough budget for this file's header and content
            remaining_budget = context_budget - current_tokens
            if remaining_budget <= 0:
                self._update_status(f"Skipping file '{path}' due to context token budget exhaustion.")
                break
                
            # Strategically truncate file content to preserve important sections
            # Estimate tokens for content to ensure it fits within remaining budget
            estimated_content_tokens = self.tokenizer.count_tokens(content) # Use injected tokenizer
            
            # If header + content exceeds budget, truncate content
            if header_tokens + estimated_content_tokens > remaining_budget:
                file_content_to_add = self._truncate_file_content(content, remaining_budget - header_tokens)
            else:
                file_content_to_add = content
            
            # Append if there's actual content to add
            if file_content_to_add.strip():
                context_str_parts.append(header + file_content_to_add)
                selected_file_paths.append(path)
                current_tokens += header_tokens + self.tokenizer.count_tokens(file_content_to_add) # Use injected tokenizer
            
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
            excluded_summary_tokens = self.tokenizer.count_tokens(excluded_summary) # Use injected tokenizer
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
        
        # Sort by score, then by path for deterministic order
        file_scores.sort(key=lambda x: (-x[1], x[0]))
        
        # Return files with score > 0, or up to 5 if no scores are positive
        relevant_files = [(path, score) for path, score in file_scores if score > 0]
        if not relevant_files and self.codebase_context:
            # If no scores are positive, return the first few files by path for consistency
            return [(path, 0.0) for path in sorted(self.codebase_context.keys())][:5]
        return relevant_files

    # --- NEW HELPER METHOD FOR prepare_context ---
    def _extract_keywords_from_prompt(self) -> List[str]:
        """Extracts meaningful keywords from the user prompt using simple NLP techniques."""
        prompt = self.initial_prompt.lower()
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'is', 'are', 'was', 'were', 'in', 'on', 
                     'at', 'to', 'for', 'with', 'by', 'of', 'as', 'it', 'this', 'that', 'code', 'file',
                     'project', 'system', 'application', 'design', 'develop', 'create', 'build'}
        words = [word.strip('.,!?()[]{}":;') for word in prompt.split() 
                 if word.lower() not in stop_words and len(word) > 2]
        return list(set(words))

    # --- NEW HELPER METHOD FOR prepare_context ---
    def _truncate_file_content(self, content: str, max_tokens: int) -> str:
        """Strategically truncates file content to preserve the most important sections."""
        # This is a simplified heuristic. A more advanced version might use AST analysis.
        if content.endswith('.py'):
            # Try to keep the beginning and some key methods/classes
            lines = content.splitlines()
            
            # Prioritize imports and definitions
            priority_content = []
            other_content = []
            
            # Simple heuristic: Keep first N lines and any lines containing 'def' or 'class'
            # This is a very basic approach.
            for i, line in enumerate(lines):
                if i < 5 or 'def ' in line or 'class ' in line or 'import ' in line:
                    priority_content.append(line)
                else:
                    other_content.append(line)
            
            combined_content = "\n".join(priority_content)
            
            # Now, truncate the combined content to fit max_tokens
            return self._truncate_text_by_tokens(self.gemini_provider, combined_content + "\n" + "\n".join(other_content), max_tokens, _status_callback=self.status_callback)
        
        # For non-Python files, simple character truncation might suffice,
        # but ideally, we'd use the tokenizer for character-to-token mapping.
        return self._truncate_text_by_tokens(self.gemini_provider, content, max_tokens, _status_callback=self.status_callback)

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
            analysis_data = self.intermediate_steps.get("Context_Analysis_Output", self._get_default_analysis_summary())
            # Ensure analysis_data is a dict for json.dumps
            if not isinstance(analysis_data, dict):
                analysis_data = {"error": "Invalid context analysis data", "details": str(analysis_data)}
            
            analysis_str = json.dumps(analysis_data, indent=2)
            
            return (f"CODEBASE CONTEXT:\n{context_string_for_analysis}\n\n"
                    f"INSTRUCTIONS:\n"
                    f"Analyze the provided codebase context and its analysis thoroughly. Understand its structure, style, patterns, dependencies, and overall logic.\n"
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
            
            # Check for specific error conditions or malformed blocks returned by the parser
            if isinstance(analysis_response, dict) and analysis_response.get("error"):
                raise ContextAnalysisError(f"Persona returned an error: {analysis_response.get('message', 'Unknown error')}")
            if isinstance(analysis_response, dict) and analysis_response.get("malformed_blocks"):
                # This indicates a parsing/validation issue with the LLM's output itself
                raise LLMResponseValidationError(
                    "Context analysis response contained malformed blocks",
                    invalid_response=analysis_response, # Pass the dict itself
                    expected_schema=ContextAnalysisOutput.__name__,
                    details={"malformed_blocks": analysis_response.get("malformed_blocks")}
                )
                
            self._update_status("Codebase context analysis successful.")
            self.intermediate_steps[analysis_output_key] = analysis_response
            return analysis_response
                
        except (LLMResponseValidationError, ContextAnalysisError, TokenBudgetExceededError) as e:
            self._update_status(f"[yellow]Warning: {str(e)}. Using default summary.[/yellow]")
            logger.warning(f"Context analysis issue: {str(e)}")
            if isinstance(e, TokenBudgetExceededError):
                return self._handle_token_budget_exceeded(e)
            return self._get_default_analysis_summary()
            
        except Exception as e: # Catch-all for unexpected errors with detailed logging
            self._update_status(f"[red]Unexpected error during context analysis: {str(e)}. Using default summary.[/red]")
            logger.exception("Unexpected error during context analysis")
            return self._get_default_analysis_summary()

    def _get_default_analysis_summary(self) -> Dict[str, Any]:
        """Returns a default summary if context analysis fails or is skipped."""
        return {
            "key_modules": [],
            "security_concerns": [],
            "architectural_patterns": [],
            "performance_bottlenecks": [],
            "malformed_blocks": [] # Ensure this key exists even if empty
        }

    # --- NEW: Refactored Persona Execution Logic ---
    
    def _perform_execution(self, persona_name: str, prompt_gen_func: Callable[[], str], execution_state: Dict[str, Any]) -> Any:
        """Performs the actual LLM call for a persona step."""
        persona = self._get_persona(persona_name)
        prompt = prompt_gen_func()
        
        # Calculate token budget for this request
        estimated_input_tokens = self.tokenizer.count_tokens(f"{persona.system_prompt}\n\n{prompt}")
        
        # Use the dynamically allocated debate budget for the LLM call
        # The total budget is max_total_tokens_budget, split into context and debate.
        # The debate budget is what's available for persona calls.
        # We need to estimate tokens used by previous persona outputs for a more accurate remaining budget.
        # This is a simplification; a more precise calculation would track output tokens per step.
        previous_output_tokens = sum(
            self.tokenizer.count_tokens(str(step_output.get("raw_response", ""))) # Estimate tokens from raw response
            for step_name, step_output in self.intermediate_steps.items() 
            if isinstance(step_output, dict) and "raw_response" in step_output
        )
        
        remaining_debate_budget = self.debate_token_budget - previous_output_tokens
        
        # Ensure we don't exceed total budget, and respect persona's max_tokens
        # The max_output_for_request should be limited by the persona's max_tokens AND the remaining debate budget.
        max_output_for_request = max(0, min(persona.max_tokens, remaining_debate_budget - estimated_input_tokens))
        
        if estimated_input_tokens >= remaining_debate_budget:
            raise TokenBudgetExceededError(
                current_tokens=estimated_input_tokens, # This is just input tokens for this step
                budget=remaining_debate_budget, # Budget available for debate steps
                details={"persona": persona_name, "step": "execution", "type": "debate_budget"}
            )
        
        # Update status with estimated tokens for the next step
        estimated_total_tokens_this_step = estimated_input_tokens + max_output_for_request
        estimated_next_step_cost = self.gemini_provider.calculate_usd_cost(estimated_input_tokens, max_output_for_request)
        
        self._update_status(
            f"Executing persona: {persona_name}...",
            current_total_tokens=self.cumulative_token_usage, # This is cumulative across all steps so far
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
        
        return {
            "raw_response": raw_response_text,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "tokens_used_in_step": tokens_used_in_step,
            "cost_this_step": cost_this_step,
            "persona_name": persona_name,
            "step_name": f"{persona_name}_Output" # Default step name
        }

    def _validate_execution_result(self, persona_name: str, execution_result: Dict[str, Any], 
                                   schema_model_for_validation: Optional[Type[BaseModel]]) -> Any:
        """Validates the raw LLM response against the schema if provided."""
        raw_response = execution_result.get("raw_response")
        if schema_model_for_validation:
            try:
                # Use LLMOutputParser for parsing and validation
                validated_data = self.parser.parse_and_validate(raw_response, schema_model_for_validation)
                
                # Check for malformed blocks returned by the parser
                if validated_data.get("malformed_blocks"):
                    # If malformed blocks exist, treat it as a validation error for this step
                    raise LLMResponseValidationError(
                        "LLM output contained malformed blocks.",
                        invalid_response=validated_data,
                        expected_schema=schema_model_for_validation.__name__,
                        details={"malformed_blocks": validated_data.get("malformed_blocks")}
                    )
                
                return validated_data
                
            except (LLMResponseValidationError, ValidationError) as validation_error:
                # If validation fails, prepare for retry or return error structure
                self._update_status(f"[yellow]Validation failed for {persona_name}: {str(validation_error)}[/yellow]", state="running")
                raise validation_error # Re-raise to be caught by the retry loop
        else:
            # No schema validation required, return raw response
            return raw_response

    def _update_execution_state(self, persona_name: str, processed_response: Any, step_name: str, 
                                update_current_thought: bool, is_final_answer_step: bool):
        """Updates cumulative stats, intermediate steps, and current thought."""
        
        # Update cumulative token usage and cost
        self.cumulative_token_usage += processed_response.get("tokens_used_in_step", 0)
        self.cumulative_usd_cost += processed_response.get("cost_this_step", 0.0)
        
        # Store the processed response in intermediate steps
        self.intermediate_steps[step_name] = processed_response
        
        # Update current thought if requested
        if update_current_thought:
            self.current_thought = processed_response.get("raw_response", str(processed_response)) # Use raw response for thought
        
        # If this is the final step, store the response as the final answer
        if is_final_answer_step:
            self.final_answer = processed_response
        
        # Update status bar with latest cumulative totals
        self._update_status(
            f"{persona_name} completed. Used {processed_response.get('tokens_used_in_step', 0)} tokens for this step.",
            current_total_tokens=self.cumulative_token_usage,
            current_total_cost=self.cumulative_usd_cost,
            state="running" # Keep running state unless it's an error/complete
        )

    def _handle_execution_attempt_failure(self, persona_name: str, error: Exception, execution_state: Dict[str, Any]):
        """Applies error-specific recovery strategies and handles retries."""
        error_category = self._categorize_error(error)
        
        # Apply recovery strategy based on error category
        if error_category == "TOKEN_LIMIT":
            # Reduce context size for next attempt
            self._reduce_context_size(0.2) # Example: reduce context by 20%
            self._update_status(f"[yellow]Reducing context size due to token limits...[/yellow]")
        
        elif error_category == "VALIDATION":
            # Request simplified output format or fewer constraints
            self._simplify_validation_schema() # Example: ask for less strict output
            self._update_status(f"[yellow]Simplifying response format requirements...[/yellow]")
        
        # For network/provider errors, standard retry with backoff
        elif error_category in ["NETWORK", "PROVIDER"] and execution_state['attempts'] < self.settings.max_retries:
            backoff_time = min(
                self.settings.max_backoff_seconds, 
                2 ** (execution_state['attempts'] - 1) # Exponential backoff
            )
            jitter = random.uniform(0, 0.5 * backoff_time)
            sleep_time = backoff_time + jitter
            self._update_status(f"[yellow]Retrying {persona_name} due to {error_category} error in {sleep_time:.2f}s...[/yellow]", state="running")
            time.sleep(sleep_time)
        
        # Log the error attempt
        self._log_error_attempt(persona_name, error, execution_state)

    def _log_error_attempt(self, persona_name: str, error: Exception, execution_state: Dict[str, Any]):
        """Logs the details of a failed attempt."""
        error_message = str(error)
        error_type = type(error).__name__
        
        # Store error details for potential debugging or final reporting
        error_log_entry = {
            "attempt": execution_state['attempts'],
            "persona": persona_name,
            "error_type": error_type,
            "error_message": error_message,
            "timestamp": datetime.datetime.now().isoformat()
        }
        # Add specific details if available from the exception
        if hasattr(error, 'details'):
            error_log_entry["details"] = error.details
        
        # Store this log entry, perhaps keyed by persona and attempt number
        if "error_logs" not in self.intermediate_steps:
            self.intermediate_steps["error_logs"] = {}
        if persona_name not in self.intermediate_steps["error_logs"]:
            self.intermediate_steps["error_logs"][persona_name] = []
        self.intermediate_steps["error_logs"][persona_name].append(error_log_entry)

    def _categorize_error(self, error: Exception) -> str:
        """Determines the error category for targeted recovery."""
        if isinstance(error, TokenBudgetExceededError):
            return "TOKEN_LIMIT"
        elif isinstance(error, LLMResponseValidationError):
            return "VALIDATION"
        elif isinstance(error, GeminiAPIError): # Check for specific API errors
            if error.code in [429, 500, 502, 503, 504]: # Retryable HTTP codes
                return "PROVIDER" # Treat as provider error for retry logic
            else:
                return "PROVIDER" # Other API errors
        elif isinstance(error, socket.gaierror) or "network" in str(error).lower() or "connection" in str(error).lower():
            return "NETWORK"
        else:
            return "PROVIDER" # Default to provider for other LLM-related errors

    def _reduce_context_size(self, reduction_factor: float):
        """Reduces the context token budget by a given factor."""
        # This is a placeholder. A real implementation would need to re-evaluate
        # context preparation or adjust the budget for subsequent steps.
        # For now, we'll just log the intent.
        self.context_token_budget = max(1000, int(self.context_token_budget * (1 - reduction_factor)))
        self._update_status(f"[yellow]Context budget adjusted to {self.context_token_budget} tokens.[/yellow]")

    def _simplify_validation_schema(self):
        """Placeholder to simplify validation requirements for the next LLM call."""
        # This would involve modifying the schema_model_for_validation passed to _execute_persona_step
        # or instructing the LLM to be less strict in its prompt.
        # For now, we just log the intent.
        self._update_status(f"[yellow]Instructing LLM to use a simpler response format for next attempt.[/yellow]")

    def _handle_execution_failure(self, persona_name: str, execution_state: Dict[str, Any]) -> Dict[str, Any]:
        """Handles the final failure after all retries are exhausted."""
        last_error = execution_state.get('last_error')
        error_message = f"Failed to execute persona '{persona_name}' after {execution_state['attempts']} attempts."
        
        if last_error:
            error_message += f" Last error: {str(last_error)}"
            
        self._update_status(f"[red]{error_message}[/red]", state="error")
        
        # Return a structured error response
        return {
            "error": "EXECUTION_FAILED",
            "message": error_message,
            "details": {"last_error": str(last_error), "attempts": execution_state['attempts']},
            "original_response": "Execution failed for this persona."
        }

    # --- MODIFIED METHOD: _execute_persona_step ---
    def _execute_persona_step(self, persona_name: str, prompt_gen_func: Callable[[], str], step_name: str, 
                              schema_model_for_validation: Optional[Type[BaseModel]] = None, 
                              update_current_thought: bool = False,
                              is_final_answer_step: bool = False) -> Dict[str, Any]:
        """
        Execute a single persona step with comprehensive error handling, validation,
        and retry mechanisms using state management and error categorization.
        """
        execution_state = {
            'attempts': 0,
            'last_error': None,
            'response': None,
            'error_logs': {} # To store logs for each attempt
        }
        
        # Loop for retries
        while execution_state['attempts'] < self.settings.max_retries:
            try:
                # 1. Perform the LLM call
                execution_result = self._perform_execution(
                    persona_name, 
                    prompt_gen_func,
                    execution_state
                )
                
                # 2. Validate the execution result
                processed_response = self._validate_execution_result(
                    persona_name,
                    execution_result,
                    schema_model_for_validation
                )
                
                # 3. Update cumulative state and intermediate steps
                self._update_execution_state(
                    persona_name,
                    processed_response,
                    step_name,
                    update_current_thought,
                    is_final_answer_step
                )
                
                # If successful, return the processed response
                return processed_response
                
            except TokenBudgetExceededError as e:
                # Handle token budget exceeded specifically, as it might require immediate action
                # and not necessarily a retry in the same way as other errors.
                # The _handle_token_budget_exceeded method returns a structured error dict.
                return self._handle_token_budget_exceeded(e)
                
            except Exception as e: # Catch all other exceptions (validation, provider, network, etc.)
                execution_state['last_error'] = e
                execution_state['attempts'] += 1
                
                # Apply error-specific recovery and retry logic
                self._handle_execution_attempt_failure(persona_name, e, execution_state)
                
                # If max retries are reached, break the loop to handle final failure
                if execution_state['attempts'] >= self.settings.max_retries:
                    break # Exit loop to trigger final failure handling
        
        # If the loop finishes without returning, it means max retries were exhausted.
        return self._handle_execution_failure(persona_name, execution_state)

    # --- MODIFIED METHOD: run_debate ---
    def run_debate(self, initial_prompt: str, codebase_context: Optional[Dict[str, str]] = None) -> Tuple[Any, Dict[str, Any]]:
        """Executes the full Socratic debate loop using dynamic persona routing."""
        self._initialize_debate(initial_prompt, codebase_context)
        
        # Determine the optimal persona sequence using the router
        # The router will now use intermediate_results to dynamically adjust the sequence
        persona_sequence = self.persona_router.determine_persona_sequence(initial_prompt, self.intermediate_steps)
        
        self._update_status(f"[blue]Optimal persona sequence determined: {', '.join(persona_sequence)}[/blue]")
        
        # Execute personas in the determined sequence
        for persona_name in persona_sequence:
            self._update_status(f"[green]Executing {persona_name} persona...[/green]")
            
            # Define prompt generation functions for each persona type
            prompt_gen_func = None
            step_name = f"{persona_name}_Output" # Default step name
            schema_model = None
            update_thought = True
            is_final_step = False
            
            # --- Persona-specific prompt generation ---
            if persona_name == "Visionary_Generator":
                def visionary_prompt_gen():
                    context_analysis_data = self.intermediate_steps.get("Context_Analysis_Output", self._get_default_analysis_summary())
                    context_str = json.dumps(context_analysis_data, indent=2) if isinstance(context_analysis_data, dict) else str(context_analysis_data)
                    context_code_str = self.prepare_context()
                    return (f"USER PROMPT: {self.initial_prompt}\n\n"
                            f"CODEBASE CONTEXT ANALYSIS:\n{context_str}\n\n"
                            f"CODEBASE CONTEXT:\n{context_code_str}\n\n"
                            f"INSTRUCTIONS:\n"
                            f"1. Analyze the provided codebase context and its analysis thoroughly.\n"
                            f"2. Propose an initial implementation strategy or code snippet consistent with the existing codebase.\n"
                            f"3. Ensure your proposed code fits naturally into the existing architecture and follows its conventions.")
                prompt_gen_func = visionary_prompt_gen
                
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
                    # Iterate through intermediate steps to gather relevant critique/output data
                    for key, value in self.intermediate_steps.items():
                        # Look for outputs or critiques from personas other than the core ones or the current one
                        if ("_Critique" in key or "_Output" in key) and \
                           not key.startswith(("Visionary_", "Skeptical_", "Impartial_Arbitrator", "Devils_Advocate", p_name)):
                            
                            # Ensure value is string for concatenation
                            value_str = json.dumps(value, indent=2) if isinstance(value, dict) else str(value)
                            previous_critiques += f"\n\n--- {key.replace('_Output', '').replace('_Critique', '')} ---\n{value_str}"
                    
                    # Assuming Constructive_Critic is always run and its output is relevant
                    constructive_feedback = self.intermediate_steps.get("Constructive_Critic_Output", "No constructive feedback available.")
                    
                    return (f"Original Proposal:\n{visionary_output}\n\n"
                            f"Skeptical Critique:\n{skeptical_critique}\n\n"
                            f"Domain Critiques:\n{previous_critiques}\n\n"
                            f"Constructive Critic Feedback:\n{constructive_feedback}\n\n"
                            f"As a {p_name.replace('_', ' ')}, analyze the provided proposal and existing critiques from your expert perspective. "
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
                        # Collect critiques from personas that are not core debate roles or the arbitrator itself
                        if ("_Critique" in key or "_Output" in key) and \
                           not key.startswith(("Visionary_", "Skeptical_", "Impartial_Arbitrator", "Devils_Advocate")):
                            value_str = json.dumps(value, indent=2) if isinstance(value, dict) else str(value)
                            domain_critiques_text += f"\n\n--- {key.replace('_Output', '').replace('_Critique', '')} ---\n{value_str}"
                    
                    constructive_feedback = self.intermediate_steps.get("Constructive_Critic_Output", "No constructive feedback available.")
                    
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
                    # The arbitrator's output is the final proposed answer before this critique
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
                    context_str = json.dumps(context_analysis_data, indent=2) if isinstance(context_analysis_data, dict) else str(context_analysis_data)
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
                
                # The response from _execute_persona_step is already processed and validated
                # Store it in intermediate steps.
                self.intermediate_steps[step_name] = response
                
                # If it's the final step, the response is already stored in self.final_answer
                # by _update_execution_state.
            else:
                self._update_status(f"[yellow]Warning: No prompt generator defined for persona '{persona_name}'. Skipping.[/yellow]")
                self.intermediate_steps[step_name] = {"error": "NO_PROMPT_GENERATOR", "message": f"Persona '{persona_name}' skipped due to missing prompt generator."}

            # --- Dynamic Sequence Re-evaluation (Integrated into PersonaRouter) ---
            # The PersonaRouter now handles dynamic sequence adjustment based on intermediate results.
            # No explicit re-sequencing logic needed here.

        # After all personas have run, format the final result
        return self._format_final_result()

    def _initialize_debate(self, initial_prompt: str, codebase_context: Optional[Dict[str, str]] = None):
        """Initializes debate state, context, and persona sequence."""
        self.cumulative_token_usage = 0
        self.cumulative_usd_cost = 0.0
        self.intermediate_steps = {} # Clear intermediate steps for a new debate
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
        # The router will be used by run_debate to determine the sequence.
        self.persona_router = PersonaRouter(self.all_personas)
        
        # Analyze codebase context if provided
        if self.codebase_context:
            self._analyze_codebase_context()
        else:
            self._update_status("[yellow]No codebase context provided. Skipping context analysis.[/yellow]")
            self.intermediate_steps["Context_Analysis_Output"] = self._get_default_analysis_summary()
            
        # Determine token allocation based on prompt and settings
        token_allocation = self._determine_token_allocation(initial_prompt)
        self.context_token_budget = token_allocation['context']
        self.debate_token_budget = token_allocation['debate']
        
        self._update_status(f"Token budget allocated: Context={self.context_token_budget} tokens, Debate={self.debate_token_budget} tokens.")


    def _format_final_result(self) -> Tuple[Any, Dict[str, Any]]:
        """Formats the final answer and returns it along with intermediate steps."""
        # Ensure the final answer is structured correctly, especially if it's the LLMOutput model
        final_output_data = {}
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
        
        # Add context and debate budgets to intermediate steps for transparency
        self.intermediate_steps["Context_Token_Budget"] = self.context_token_budget
        self.intermediate_steps["Debate_Token_Budget"] = self.debate_token_budget
        
        return final_output_data, self.intermediate_steps