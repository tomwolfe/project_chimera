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
import random
from pathlib import Path
from collections import defaultdict
from typing import List, Dict, Tuple, Any, Callable, Optional, Type, Union
import numpy as np
from google import genai
from google.genai import types
from google.genai.errors import APIError
import traceback
from rich.console import Console
from pydantic import ValidationError
from functools import lru_cache
import uuid # <--- ADDED THIS LINE

# --- IMPORT MODIFICATIONS ---
from src.llm_provider import GeminiProvider
from src.context.context_analyzer import ContextRelevanceAnalyzer
from src.persona.routing import PersonaRouter
from src.utils.output_parser import LLMOutputParser
# Ensure DeploymentAnalysisOutput is imported
from src.models import PersonaConfig, ReasoningFrameworkConfig, LLMOutput, CodeChange, ContextAnalysisOutput, CritiqueOutput, GeneralOutput, ConflictReport, SelfImprovementAnalysisOutput, ConfigurationAnalysisOutput, SelfImprovementAnalysisOutputV1, DeploymentAnalysisOutput
from src.config.settings import ChimeraSettings
from src.exceptions import ChimeraError, LLMResponseValidationError, SchemaValidationError, TokenBudgetExceededError, LLMProviderError, CircuitBreakerError
from src.constants import SELF_ANALYSIS_KEYWORDS, is_self_analysis_prompt
from src.logging_config import setup_structured_logging
from src.utils.error_handler import handle_errors
from src.persona_manager import PersonaManager
# NEW IMPORTS FOR SELF-IMPROVEMENT
from src.self_improvement.metrics_collector import ImprovementMetricsCollector
from src.self_improvement.content_validator import ContentAlignmentValidator

# Configure logging for the core module itself
logger = logging.getLogger(__name__)

class SocraticDebate:
    PERSONA_OUTPUT_SCHEMAS = {
        "Impartial_Arbitrator": LLMOutput,
        "Context_Aware_Assistant": ContextAnalysisOutput,
        "Constructive_Critic": CritiqueOutput,
        "General_Synthesizer": GeneralOutput,
        "Devils_Advocate": ConflictReport,
        "Self_Improvement_Analyst": SelfImprovementAnalysisOutputV1, # Changed to V1 as per app.py output
        # ADDED/MODIFIED: Map engineering personas to CritiqueOutput
        "Code_Architect": CritiqueOutput,
        "Security_Auditor": CritiqueOutput,
        "DevOps_Engineer": CritiqueOutput,
        "Test_Engineer": CritiqueOutput,
    }

    def __init__(self, initial_prompt: str, api_key: str,
                 codebase_context: Optional[Dict[str, str]] = None,
                 settings: Optional[ChimeraSettings] = None,
                 all_personas: Optional[Dict[str, PersonaConfig]] = None,
                 persona_sets: Optional[Dict[str, List[str]]] = None,
                 domain: Optional[str] = None,
                 model_name: str = "gemini-2.5-flash-lite",
                 status_callback: Optional[Callable] = None,
                 rich_console: Optional[Console] = None,
                 context_analyzer: Optional[ContextRelevanceAnalyzer] = None,
                 is_self_analysis: bool = False,
                 persona_manager: Optional[PersonaManager] = None
                 ):
        """
        Initialize a Socratic debate session.
        """
        setup_structured_logging(log_level=logging.INFO)
        self.logger = logging.getLogger(__name__)

        self.settings = settings or ChimeraSettings()
        self.max_total_tokens_budget = self.settings.total_budget
        self.tokens_used = 0
        self.model_name = model_name
        self.status_callback = status_callback
        self.rich_console = rich_console or Console(stderr=True)
        self.is_self_analysis = is_self_analysis

        self.request_id = str(uuid.uuid4())[:8]
        self._log_extra = {"request_id": self.request_id or "N/A"}

        self.initial_prompt = initial_prompt
        self.codebase_context = codebase_context or {} # Ensure it's a dict
        self.domain = domain

        # Initialize the LLM provider.
        try:
            self.llm_provider = GeminiProvider(api_key=api_key, model_name=self.model_name, rich_console=self.rich_console, request_id=self.request_id)
        except LLMProviderError as e:
            self._log_with_context("error", f"Failed to initialize LLM provider: {e.message}", exc_info=True)
            raise ChimeraError(f"LLM provider initialization failed: {e.message}", original_exception=e) from e
        except Exception as e:
            self._log_with_context("error", f"An unexpected error occurred during LLM provider initialization: {e}", exc_info=True)
            raise ChimeraError(f"LLM provider initialization failed unexpectedly: {e}", original_exception=e) from e

        try:
            self.tokenizer = self.llm_provider.tokenizer
        except AttributeError:
            raise ChimeraError("LLM provider tokenizer is not available.")

        # NEW: Store PersonaManager instance
        self.persona_manager = persona_manager
        if not self.persona_manager:
            self.logger.warning("PersonaManager instance not provided to SocraticDebate. Initializing a new one. This might affect state persistence in UI.")
            self.persona_manager = PersonaManager({}) # Pass empty dict as fallback for domain_keywords
            self.all_personas = self.persona_manager.all_personas
            self.persona_sets = self.persona_manager.persona_sets
        else:
            self.all_personas = self.persona_manager.all_personas
            self.persona_sets = self.persona_manager.persona_sets

        # Initialize PersonaRouter with all loaded personas AND persona_sets
        self.persona_router = self.persona_manager.persona_router
        if not self.persona_router: # Fallback if PersonaManager didn't set it up
             self.logger.warning("PersonaRouter not found in PersonaManager. Initializing a new one. This might indicate an issue in PersonaManager setup.")
             self.persona_router = PersonaRouter(self.all_personas, self.persona_sets, self.persona_manager.prompt_analyzer)

        # NEW: Initialize ContentAlignmentValidator
        self.content_validator = ContentAlignmentValidator(
            original_prompt=self.initial_prompt,
            debate_domain=self.domain
        )

        # If codebase_context was provided, compute embeddings now if context_analyzer is available.
        if self.codebase_context and self.context_analyzer:
            if isinstance(self.codebase_context, dict):
                try:
                    if not self.context_analyzer.file_embeddings:
                        self.context_analyzer.compute_file_embeddings(self.codebase_context)
                except Exception as e:
                    self._log_with_context("error", f"Failed to compute embeddings for codebase context: {e}", exc_info=True)
                    if self.status_callback:
                        self.status_callback(message=f"[red]Error computing context embeddings: {e}[/red]")
            else:
                self.logger.warning("codebase_context was not a dictionary, skipping embedding computation.")

        # Calculate token budgets for different phases AFTER context analyzer is ready
        self._calculate_token_budgets()

    def _log_with_context(self, level: str, message: str, **kwargs):
        """Helper to add request context to all logs from this instance using the class-specific logger."""
        exc_info = kwargs.pop('exc_info', None)
        original_exception = kwargs.pop('original_exception', None)
        log_data = {**self._log_extra, **kwargs}

        # Convert non-serializable objects to strings for logging to prevent errors
        for k, v in log_data.items():
            try:
                json.dumps({k: v})
            except TypeError:
                log_data[k] = str(v)

        logger_method = getattr(self.logger, level)
        if exc_info is not None:
            logger_method(message, exc_info=exc_info, extra=log_data)
        elif original_exception is not None:
            log_data['original_exception_type'] = type(original_exception).__name__
            log_data['original_exception_message'] = str(original_exception)
            logger_method(message, extra=log_data)
        else:
            logger_method(message, extra=log_data)

    def _calculate_token_budgets(self):
        """Calculates token budgets for different phases based on context, model limits, and prompt type."""
        try:
            # Analyze prompt complexity using persona_manager's new method
            prompt_analysis = self.persona_manager._analyze_prompt_complexity(self.initial_prompt)
            complexity_score = prompt_analysis['complexity_score']

            # Adjust ratios based on complexity and self-analysis flag
            if self.is_self_analysis:
                context_ratio = self.settings.self_analysis_context_ratio
                debate_ratio = self.settings.self_analysis_debate_ratio
                synthesis_ratio = self.settings.self_analysis_synthesis_ratio
            else:
                # Base ratios
                base_context_ratio = self.settings.context_token_budget_ratio
                base_output_ratio = self.settings.synthesis_token_budget_ratio

                # Adjust ratios based on prompt characteristics
                if 'code' in self.initial_prompt.lower() or 'implement' in self.initial_prompt.lower():
                    # Code generation needs more output tokens
                    context_ratio = max(0.4, base_context_ratio - 0.15)
                    synthesis_ratio = min(0.5, base_output_ratio + 0.15)
                elif complexity_score > 0.7:
                    # Complex analytical prompts need more context processing
                    context_ratio = min(0.7, base_context_ratio + 0.15)
                    synthesis_ratio = max(0.2, base_output_ratio - 0.05)
                else:
                    context_ratio = base_context_ratio
                    synthesis_ratio = base_output_ratio

                # Ensure debate_ratio is adjusted to fill the remaining
                debate_ratio = 1.0 - context_ratio - synthesis_ratio
                if debate_ratio < 0.05:
                    debate_ratio = 0.05
                    # Re-normalize if debate_ratio was too low
                    remaining_for_context_synthesis = 1.0 - debate_ratio
                    context_ratio = context_ratio / (context_ratio + synthesis_ratio) * remaining_for_context_synthesis
                    synthesis_ratio = synthesis_ratio / (context_ratio + synthesis_ratio) * remaining_for_context_synthesis


            # NEW: Explicitly normalize ratios to ensure they sum to 1.0 at this point
            total_current_ratios = context_ratio + debate_ratio + synthesis_ratio
            if total_current_ratios > 0 and abs(total_current_ratios - 1.0) > 1e-6:
                self._log_with_context("warning", f"Token budget ratios sum to {total_current_ratios}, normalizing them.",
                                       original_context_ratio=context_ratio, original_debate_ratio=debate_ratio,
                                       original_synthesis_ratio=synthesis_ratio)
                normalization_factor = 1.0 / total_current_ratios
                context_ratio *= normalization_factor
                debate_ratio *= normalization_factor
                synthesis_ratio *= normalization_factor
                self._log_with_context("debug", f"Normalized ratios: context={context_ratio}, debate={debate_ratio}, synthesis={synthesis_ratio}")

            # Estimate tokens for context and initial input
            context_str = self.context_analyzer.get_context_summary() if self.context_analyzer else ""
            self.initial_input_tokens = self.tokenizer.count_tokens(context_str + self.initial_prompt)

            remaining_tokens = max(0, self.max_total_tokens_budget - self.initial_input_tokens)

            # Calculate phase tokens based on adjusted ratios
            context_tokens_budget = int(remaining_tokens * context_ratio)
            debate_tokens_budget = int(remaining_tokens * debate_ratio)
            synthesis_tokens_budget = int(remaining_tokens * synthesis_ratio)

            # Define a minimum token allocation to ensure phases can function
            MIN_PHASE_TOKENS = 250

            # Apply minimums and re-distribute if necessary
            context_tokens_budget = max(MIN_PHASE_TOKENS, context_tokens_budget)
            debate_tokens_budget = max(MIN_PHASE_TOKENS, debate_tokens_budget)
            synthesis_tokens_budget = max(MIN_PHASE_TOKENS, synthesis_tokens_budget)

            total_allocated = context_tokens_budget + debate_tokens_budget + synthesis_tokens_budget
            if total_allocated > remaining_tokens:
                # Simple proportional reduction if over budget
                reduction_factor = remaining_tokens / total_allocated
                context_tokens_budget = int(context_tokens_budget * reduction_factor)
                debate_tokens_budget = int(debate_tokens_budget * reduction_factor)
                synthesis_tokens_budget = int(synthesis_tokens_budget * reduction_factor)

                # Ensure minimums are still met after reduction, if possible
                context_tokens_budget = max(MIN_PHASE_TOKENS, context_tokens_budget)
                debate_tokens_budget = max(MIN_PHASE_TOKENS, debate_tokens_budget)
                synthesis_tokens_budget = max(MIN_PHASE_TOKENS, synthesis_tokens_budget)

            self.phase_budgets = {
                "context": context_tokens_budget,
                "debate": debate_tokens_budget,
                "synthesis": synthesis_tokens_budget,
                "persona_turn_budgets": {} # Initialize empty, will be populated later
            }

            self._log_with_context("info", "SocraticDebate token budgets initialized",
                                   initial_input_tokens=self.initial_input_tokens,
                                   context_budget=self.phase_budgets["context"],
                                   debate_budget=self.phase_budgets["debate"],
                                   synthesis_budget=self.phase_budgets["synthesis"],
                                   max_total_tokens_budget=self.max_total_tokens_budget,
                                   prompt_complexity=prompt_analysis,
                                   is_self_analysis=self.is_self_analysis)

        except Exception as e:
            self._log_with_context("error", "Token budget calculation failed",
                                   error=str(e), context="token_budget", exc_info=True, original_exception=e)
            # Fallback to hardcoded values if calculation fails
            self.phase_budgets = {"context": 500, "debate": 15000, "synthesis": 1000, "persona_turn_budgets": {}}
            self.initial_input_tokens = 0
            raise ChimeraError("Failed to calculate token budgets due to an unexpected error.", original_exception=e) from e

    def track_token_usage(self, phase: str, tokens: int):
        """Tracks token usage for a given phase."""
        self.tokens_used += tokens
        cost = self.llm_provider.calculate_usd_cost(tokens, 0)
        self.intermediate_steps.setdefault(f"{phase}_Tokens_Used", 0)
        self.intermediate_steps[f"{phase}_Tokens_Used"] += tokens
        self.intermediate_steps.setdefault(f"{phase}_Estimated_Cost_USD", 0.0)
        self.intermediate_steps[f"{phase}_Estimated_Cost_USD"] += cost
        self._log_with_context("debug", f"Tokens used in {phase}: {tokens}. Total: {self.tokens_used}",
                               phase=phase, tokens_added=tokens, total_tokens=self.tokens_used)

    def check_budget(self, phase: str, tokens_needed: int, step_name: str):
        """Checks if adding tokens for the next step would exceed the budget."""
        if self.tokens_used + tokens_needed > self.max_total_tokens_budget:
            self._log_with_context("warning", f"Token budget exceeded for {step_name} in {phase} phase.",
                                   current_tokens=self.tokens_used, tokens_needed=tokens_needed,
                                   budget=self.max_total_tokens_budget, step=step_name, phase=phase)
            raise TokenBudgetExceededError(self.tokens_used, self.max_total_tokens_budget,
                                           details={"phase": phase, "step_name": step_name, "tokens_needed": tokens_needed})

    def get_total_used_tokens(self) -> int:
        """Returns the total tokens used so far."""
        return self.tokens_used

    def get_total_estimated_cost(self) -> float:
        """Returns the total estimated cost so far."""
        total_cost = 0.0
        for key, value in self.intermediate_steps.items():
            if key.endswith("_Estimated_Cost_USD"):
                total_cost += value
        return total_cost

    def get_progress_pct(self, phase: str, completed: bool = False, error: bool = False) -> float:
        """Calculates the progress percentage for the debate."""
        phase_weights = {
            "context": 0.1,
            "debate": 0.7,
            "synthesis": 0.2
        }

        current_progress = 0.0
        if phase == "context":
            current_progress = 0.05
        elif phase == "debate":
            total_debate_personas = len(self.intermediate_steps.get("Persona_Sequence", [])) - 1
            completed_debate_personas = sum(1 for k in self.intermediate_steps if k.endswith("_Output") and not k.startswith(("Final_", "Context_Analysis_Output")) and k != "Self_Improvement_Metrics" and k != "Debate_History" and k != "Conflict_Resolution_Attempt" and k != "Unresolved_Conflict")

            if total_debate_personas > 0:
                current_progress = phase_weights["context"] + (completed_debate_personas / total_debate_personas) * phase_weights["debate"]
            else:
                current_progress = phase_weights["context"]
        elif phase == "synthesis":
            current_progress = phase_weights["context"] + phase_weights["debate"] + 0.1

        if completed:
            current_progress = 1.0
        elif error:
            current_progress = max(current_progress, 0.99)

        return min(max(0.0, current_progress), 1.0)

    # --- MODIFIED METHOD: _execute_llm_turn with retry logic ---
    def _execute_llm_turn(self, persona_name: str, persona_config: PersonaConfig,
                          prompt_for_llm: str, phase: str,
                          max_output_tokens_for_turn: int, # NEW ARGUMENT
                          requested_model_name: Optional[str] = None,
                          max_retries: int = 2) -> Dict[str, Any]: # Added max_retries parameter
        """
        Executes a single LLM turn for a given persona, handles API calls,
        parsing, validation, and token tracking, with retry logic for validation failures.
        """
        self._log_with_context("debug", f"Executing LLM turn for {persona_name} in {phase} phase.",
                               persona=persona_name, phase=phase)

        # Get the appropriate schema model for the persona's output
        output_schema = self.PERSONA_OUTPUT_SCHEMAS.get(persona_name, GeneralOutput)
        self._log_with_context("debug", f"Using schema {output_schema.__name__} for {persona_name}.")

        raw_llm_output = ""
        input_tokens = 0
        output_tokens = 0
        has_schema_error = False
        is_truncated = False # Assume not truncated initially
        has_content_misalignment = False # NEW: Initialize content misalignment flag

        current_prompt = prompt_for_llm # Use a mutable variable for prompt modification

        for attempt in range(max_retries + 1):
            try:
                # Update status callback for attempt
                if self.status_callback:
                    self.status_callback(message=f"LLM Call: [bold]{persona_name.replace('_', ' ')}[/bold] generating response (Attempt {attempt + 1}/{max_retries + 1})...",
                                         state="running",
                                         current_total_tokens=self.tokens_used,
                                         current_total_cost=self.get_total_estimated_cost(),
                                         progress_pct=self.get_progress_pct(phase),
                                         current_persona_name=persona_name)

                # Token capping logic
                actual_model_max_output_tokens = self.llm_provider.tokenizer.max_output_tokens
                effective_max_output_tokens = min(max_output_tokens_for_turn, actual_model_max_output_tokens)
                self._log_with_context("debug", f"Adjusting max_output_tokens for {persona_name}. Requested: {max_output_tokens_for_turn}, Model Max: {actual_model_max_output_tokens}, Effective: {effective_max_output_tokens}")

                # Call the LLM provider's generate method
                raw_llm_output, input_tokens, output_tokens = self.llm_provider.generate(
                    prompt=current_prompt, # Use the potentially modified prompt
                    system_prompt=persona_config.system_prompt,
                    temperature=persona_config.temperature,
                    max_tokens=effective_max_output_tokens,
                    persona_config=persona_config,
                    requested_model_name=requested_model_name or self.model_name
                )

                # Track token usage
                self.track_token_usage(phase, input_tokens + output_tokens)

                # Store actual parameters used for this turn
                self.intermediate_steps[f"{persona_name}_Actual_Temperature"] = persona_config.temperature
                self.intermediate_steps[f"{persona_name}_Actual_Max_Tokens"] = effective_max_output_tokens

                # Check for truncation
                if output_tokens >= effective_max_output_tokens * 0.95:
                    is_truncated = True
                    self._log_with_context("warning", f"Output for {persona_name} might be truncated. Output tokens ({output_tokens}) close to max_tokens ({effective_max_output_tokens}).")

                # Parse and validate the LLM's raw output
                parser = LLMOutputParser()
                parsed_output = parser.parse_and_validate(raw_llm_output, output_schema) # Use output_schema here

                # Check for malformed blocks reported by the parser itself (e.g., invalid JSON structure before validation)
                if parsed_output.get("malformed_blocks"):
                    has_schema_error = True # Treat parser malformed blocks as schema errors for reporting
                    self.intermediate_steps.setdefault("malformed_blocks", []).extend(parsed_output["malformed_blocks"])
                    self._log_with_context("warning", f"Parser reported malformed blocks for {persona_name}.",
                                           persona=persona_name, malformed_blocks=parsed_output["malformed_blocks"])

                # If we reach here, parsing and validation succeeded. Break the retry loop.
                break

            except (LLMProviderError, TokenBudgetExceededError, CircuitBreakerError, ChimeraError) as e:
                # These are non-retryable errors or errors that should halt the process.
                self._log_with_context("error", f"Non-retryable error during LLM turn for {persona_name}: {e}",
                                       persona=persona_name, phase=phase, exc_info=True, original_exception=e)
                if self.persona_manager:
                    self.persona_manager.record_persona_performance(persona_name, False, is_truncated, True, has_content_misalignment)
                raise e # Re-raise the specific error

            except SchemaValidationError as e: # Catching SchemaValidationError as it's more likely from parse_and_validate
                # This is a retryable validation error.
                if attempt < max_retries:
                    self._log_with_context("warning", f"Validation error for {persona_name} (Attempt {attempt + 1}/{max_retries + 1}). Retrying. Error: {e}",
                                           persona=persona_name, phase=phase, exc_info=True, original_exception=e)
                    
                    # Extract specific details from SchemaValidationError for more actionable feedback
                    error_details = e.details if hasattr(e, 'details') and isinstance(e.details, dict) else {}
                    error_type = error_details.get('error_type', 'Unknown validation error')
                    field_path = error_details.get('field_path', 'N/A')
                    invalid_value_snippet = str(error_details.get('invalid_value', 'N/A'))[:200]

                    retry_feedback = f"PREVIOUS OUTPUT INVALID: {error_type} at '{field_path}'. Problematic value snippet: '{invalid_value_snippet}'.\n"
                    retry_feedback += "CRITICAL: Your output failed schema validation. You MUST correct this.\n"
                    retry_feedback += "REMEMBER: OUTPUT MUST BE RAW JSON ONLY WITH NO MARKDOWN. STRICTLY ADHERE TO THE SCHEMA.\n\n"
                    
                    current_prompt = f"{retry_feedback}Original prompt: {prompt_for_llm}"
          
                    # Add the error to intermediate steps for debugging
                    self.intermediate_steps.setdefault("malformed_blocks", []).append({
                        "type": "RETRYABLE_VALIDATION_ERROR",
                        "message": str(e),
                        "attempt": attempt + 1,
                        "persona": persona_name
                    })
                    
                    # Continue to the next iteration of the loop
                    continue
                else:
                    # If it's the last attempt and it failed, return the fallback JSON
                    self._log_with_context("error", f"Max retries ({max_retries}) reached for {persona_name}. Returning fallback JSON.", persona=persona_name)
                    if self.persona_manager:
                        self.persona_manager.record_persona_performance(persona_name, False, is_truncated, True, has_content_misalignment)
                    return {
                        "ANALYSIS_SUMMARY": "JSON validation failed after multiple attempts",
                        "IMPACTFUL_SUGGESTIONS": [{
                            "AREA": "Robustness",
                            "PROBLEM": f"Failed to produce valid JSON after {max_retries} attempts",
                            "PROPOSED_SOLUTION": "Review system prompts and validation logic",
                            "IMPACT": "Critical failure in self-improvement process"
                        }]
                    }
            
            except Exception as e: # Catch any other unexpected errors during the turn
                # If it's not a retryable error, log and re-raise.
                self._log_with_context("error", f"An unexpected error occurred during LLM turn for {persona_name}: {e}",
                                       persona=persona_name, phase=phase, exc_info=True, original_exception=e)
                if self.persona_manager:
                    self.persona_manager.record_persona_performance(persona_name, False, is_truncated, True, has_content_misalignment)
                raise ChimeraError(f"Unexpected error in LLM turn for {persona_name}: {e}", original_exception=e) from e

        # --- End of retry loop ---
        # This part is reached if the loop broke successfully (i.e., no exception or retryable exception occurred)
        
        # Store the final parsed output
        self.intermediate_steps[f"{persona_name}_Output"] = parsed_output
        self.intermediate_steps[f"{persona_name}_Tokens_Used"] = input_tokens + output_tokens
        self.intermediate_steps[f"{persona_name}_Estimated_Cost_USD"] = self.llm_provider.calculate_usd_cost(input_tokens, output_tokens)

        self._log_with_context("info", f"LLM turn for {persona_name} completed successfully.",
                               persona=persona_name, phase=phase,
                               input_tokens=input_tokens, output_tokens=output_tokens,
                               total_tokens_for_turn=input_tokens + output_tokens)
        
        # Record persona performance for adaptive adjustments
        if self.persona_manager:
            self.persona_manager.record_persona_performance(persona_name, True, is_truncated, has_schema_error, has_content_misalignment)

        return parsed_output
    # --- END MODIFIED METHOD ---

    def _initialize_debate_state(self):
        """Initializes or resets the debate's internal state variables."""
        self.intermediate_steps = {}
        self.tokens_used = 0
        self.rich_console.print(f"[bold green]Starting Socratic Debate for prompt:[/bold green] [italic]{self.initial_prompt}[/italic]")
        self._log_with_context("info", "Debate state initialized.")

    def _perform_context_analysis(self, persona_sequence: Tuple[str, ...]) -> Optional[Dict[str, Any]]:
        """
        Performs context analysis based on the initial prompt and codebase context.
        """
        if not self.codebase_context or not self.context_analyzer:
            self._log_with_context("info", "No codebase context or analyzer available. Skipping context analysis.")
            return None

        self._log_with_context("info", "Performing context analysis.")
        try:
            # Pass the context budget for dynamic file selection
            relevant_files = self.context_analyzer.find_relevant_files(
                self.initial_prompt,
                max_context_tokens=self.phase_budgets["context"],
                active_personas=persona_sequence
            )

            # Generate context summary using the intelligent summarizer
            context_summary_str = self.context_analyzer.generate_context_summary(
                [f[0] for f in relevant_files],
                self.phase_budgets["context"],
                self.initial_prompt
            )

            estimated_context_tokens = self.tokenizer.count_tokens(context_summary_str)

            self.check_budget("context", estimated_context_tokens, "Context Analysis Summary")
            self.track_token_usage("context", estimated_context_tokens)

            context_analysis_output = {
                "relevant_files": relevant_files,
                "context_summary": context_summary_str,
                "estimated_tokens": estimated_context_tokens
            }
            self._log_with_context("info", "Context analysis completed.",
                                   relevant_files=[f[0] for f in relevant_files],
                                   estimated_tokens=estimated_context_tokens)
            return context_analysis_output
        except Exception as e:
            self._log_with_context("error", f"Error during context analysis: {e}", exc_info=True, original_exception=e)
            self.rich_console.print(f"[red]Error during context analysis: {e}[/red]")
            return {"error": f"Context analysis failed: {e}"}

    def _get_final_persona_sequence(self, prompt: str, context_analysis_results: Optional[Dict[str, Any]]) -> List[str]:
        """
        Delegates to the PersonaRouter to determine the optimal persona sequence,
        incorporating prompt analysis, domain, and context analysis results.
        """
        if not self.persona_router:
            self._log_with_context("error", "PersonaRouter not initialized. Falling back to default sequence.")
            return ["Visionary_Generator", "Skeptical_Generator", "Impartial_Arbitrator"]

        # Call the PersonaRouter's main method for sequence determination
        sequence = self.persona_router.determine_persona_sequence(
            prompt=prompt,
            domain=self.domain,
            intermediate_results=self.intermediate_steps,
            context_analysis_results=context_analysis_results
        )
        return sequence

    def _distribute_debate_persona_budgets(self, persona_sequence: List[str]):
        """
        Distributes the total debate token budget among the actual personas in the sequence.
        This is called *after* the final persona sequence is determined.
        """
        MIN_PERSONA_TOKENS = 256 # Minimum for each persona turn

        active_debate_personas = [
            p for p in persona_sequence
            if p not in ["Context_Aware_Assistant", "Impartial_Arbitrator", "General_Synthesizer", "Self_Improvement_Analyst"]
        ]

        num_debate_personas = len(active_debate_personas)
        if num_debate_personas == 0:
            self.phase_budgets["persona_turn_budgets"] = {}
            self.phase_budgets["debate"] = 0 # No debate personas, so no debate budget
            return

        base_allocation_per_persona = self.phase_budgets["debate"] // num_debate_personas
        remaining_budget = self.phase_budgets["debate"]

        # FIX: Initialize persona_turn_budgets before its first use
        persona_turn_budgets: Dict[str, int] = {}

        # First pass: allocate minimums and a base share, respecting persona's max_tokens
        for p_name in active_debate_personas:
            persona_config = self.persona_manager.get_adjusted_persona_config(p_name)
            allocated = max(MIN_PERSONA_TOKENS, min(base_allocation_per_persona, persona_config.max_tokens))
            persona_turn_budgets[p_name] = allocated
            remaining_budget -= allocated

        # Second pass: if there's remaining budget (e.g., due to some personas hitting their max_tokens early),
        # redistribute it proportionally to those not yet at their max.
        if remaining_budget > 0:
            redistribution_pool_personas = [
                p_name for p_name in active_debate_personas
                if persona_turn_budgets[p_name] < self.persona_manager.get_adjusted_persona_config(p_name).max_tokens
            ]
            if redistribution_pool_personas:
                share_per_redistribution_persona = remaining_budget // len(redistribution_pool_personas)
                for p_name in redistribution_pool_personas:
                    persona_config = self.persona_manager.get_adjusted_persona_config(p_name)
                    persona_turn_budgets[p_name] = min(persona_config.max_tokens, persona_turn_budgets[p_name] + share_per_redistribution_persona)

        # Final check: if total allocated still exceeds debate_tokens_budget (e.g., due to minimums),
        # proportionally reduce, but ensure minimums are still met.
        current_total_persona_budget = sum(persona_turn_budgets.values())
        if current_total_persona_budget > self.phase_budgets["debate"]:
            reduction_factor = self.phase_budgets["debate"] / current_total_persona_budget
            for p_name in active_debate_personas:
                persona_turn_budgets[p_name] = max(MIN_PERSONA_TOKENS, int(persona_turn_budgets[p_name] * reduction_factor))

        self.phase_budgets["persona_turn_budgets"] = persona_turn_budgets
        self._log_with_context("info", "Debate persona turn budgets distributed.",
                               persona_turn_budgets=self.phase_budgets["persona_turn_budgets"])


    def _process_context_persona_turn(self, persona_sequence: List[str], context_analysis_results: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """
        Executes the Context_Aware_Assistant persona turn if it's in the sequence.
        """
        if "Context_Aware_Assistant" not in persona_sequence:
            self._log_with_context("info", "Context_Aware_Assistant not in sequence. Skipping turn.")
            return None

        self._log_with_context("info", "Executing Context_Aware_Assistant turn.")
        persona_config = self.all_personas.get("Context_Aware_Assistant")
        if not persona_config:
            self._log_with_context("error", "Context_Aware_Assistant persona configuration not found.")
            return {"error": "Context_Aware_Assistant config missing."}

        context_prompt_content = ""
        if context_analysis_results and context_analysis_results.get("relevant_files"):
            for file_path, _ in context_analysis_results["relevant_files"]:
                content = self.codebase_context.get(file_path, "")
                if content:
                    context_prompt_content += f"### File: {file_path}\n```\n{content}\n```\n\n"

        prompt = f"Analyze the following codebase context and provide a structured analysis:\n\n{context_prompt_content}"

        # Use the specific budget for Context_Aware_Assistant if available, otherwise fallback to persona_config.max_tokens
        max_output_tokens_for_turn = self.phase_budgets.get("persona_turn_budgets", {}).get("Context_Aware_Assistant", persona_config.max_tokens)
        estimated_tokens = self.tokenizer.count_tokens(prompt) + max_output_tokens_for_turn
        self.check_budget("debate", estimated_tokens, "Context_Aware_Assistant")

        try:
            output = self._execute_llm_turn("Context_Aware_Assistant", persona_config, prompt, "debate", max_output_tokens_for_turn)
            self._log_with_context("info", "Context_Aware_Assistant turn completed.")
            return output
        except Exception as e:
            self._log_with_context("error", f"Error during Context_Aware_Assistant turn: {e}", exc_info=True, original_exception=e)
            self.rich_console.print(f"[red]Error during Context_Aware_Assistant turn: {e}[/red]")
            return {"error": f"Context_Aware_Assistant turn failed: {e}"}

    def _execute_debate_persona_turns(self, persona_sequence: List[str], context_persona_turn_results: Optional[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Executes the main debate turns for each persona in the sequence.
        """
        debate_history = []
        
        # MODIFIED: Initial previous_output handling
        previous_output_for_llm: Union[str, Dict[str, Any]]
        if context_persona_turn_results:
            previous_output_for_llm = f"Initial Prompt: {self.initial_prompt}\n\nStructured Context Analysis:\n{json.dumps(context_persona_turn_results, indent=2)}"
        else:
            previous_output_for_llm = f"Initial Prompt: {self.initial_prompt}"


        personas_for_debate = [
            p for p in persona_sequence
            if p not in ["Context_Aware_Assistant", "Impartial_Arbitrator", "General_Synthesizer", "Self_Improvement_Analyst"]
        ]

        # Ensure Devils_Advocate is placed strategically if present
        if "Devils_Advocate" in persona_sequence and "Devils_Advocate" not in personas_for_debate:
            insert_idx = len(personas_for_debate)
            for i, p_name in reversed(list(enumerate(personas_for_debate))):
                if "Critic" in p_name or "Analyst" in p_name or "Engineer" in p_name or "Architect" in p_name:
                    insert_idx = i + 1
                    break
            personas_for_debate.insert(insert_idx, "Devils_Advocate")

        total_debate_steps = len(personas_for_debate)
        for i, persona_name in enumerate(personas_for_debate):
            self._log_with_context("info", f"Executing debate turn for persona: {persona_name}", persona=persona_name)
            self.status_callback(
                message=f"Executing: [bold]{persona_name.replace('_', ' ')}[/bold]...",
                state="running",
                current_total_tokens=self.tokens_used,
                current_total_cost=self.get_total_estimated_cost(),
                progress_pct=self.get_progress_pct("debate"),
                current_persona_name=persona_name
            )

            # Get potentially adjusted persona config from PersonaManager
            persona_config = self.persona_manager.get_adjusted_persona_config(persona_name)
            if not persona_config:
                persona_config = self.all_personas.get(persona_name)
                if not persona_config:
                    self._log_with_context("error", f"Persona configuration not found for {persona_name}. Skipping turn.", persona=persona_name)
                    debate_history.append({"persona": persona_name, "error": "Config not found"})
                    continue

            # NEW: Prepare persona-specific context
            persona_specific_context_str = ""
            if context_persona_turn_results: # This was the undefined variable
                if persona_name == "Security_Auditor" and context_persona_turn_results.get("security_summary"):
                    persona_specific_context_str = f"Security Context Summary:\n{json.dumps(context_persona_turn_results['security_summary'], indent=2)}"
                elif persona_name == "Code_Architect" and context_persona_turn_results.get("architecture_summary"):
                    persona_specific_context_str = f"Architecture Context Summary:\n{json.dumps(context_persona_turn_results['architecture_summary'], indent=2)}"
                elif persona_name == "DevOps_Engineer" and context_persona_turn_results.get("devops_summary"):
                    persona_specific_context_str = f"DevOps Context Summary:\n{json.dumps(context_persona_turn_results['devops_summary'], indent=2)}"
                elif persona_name == "Test_Engineer" and context_persona_turn_results.get("testing_summary"):
                    persona_specific_context_str = f"Testing Context Summary:\n{json.dumps(context_persona_turn_results['testing_summary'], indent=2)}"
                elif context_persona_turn_results.get("general_overview"):
                    persona_specific_context_str = f"General Codebase Overview:\n{context_persona_turn_results['general_overview']}"
                # NEW: Add structured configuration summary if available
                if context_persona_turn_results.get("configuration_summary"):
                    persona_specific_context_str += f"\n\nStructured Configuration Analysis:\n{json.dumps(context_persona_turn_results['configuration_summary'], indent=2)}"
                # NEW: Add structured deployment summary if available
                if context_persona_turn_results.get("deployment_summary"):
                    persona_specific_context_str += f"\n\nStructured Deployment Robustness Analysis:\n{json.dumps(context_persona_turn_results['deployment_summary'], indent=2)}"


            # MODIFIED: Construct the current prompt with a summary of previous_output if it was problematic
            current_prompt_parts = [f"Initial Problem: {self.initial_prompt}\n\n"]
            if persona_specific_context_str:
                current_prompt_parts.append(f"Relevant Code Context:\n{persona_specific_context_str}\n\n")

            # Check if previous_output_for_llm was problematic (malformed or misaligned)
            is_previous_output_problematic = False
            if isinstance(previous_output_for_llm, dict):
                if previous_output_for_llm.get("malformed_blocks") or previous_output_for_llm.get("content_misalignment_warning"):
                    is_previous_output_problematic = True
            elif isinstance(previous_output_for_llm, str) and ("malformed" in previous_output_for_llm.lower() or "error" in previous_output_for_llm.lower()):
                is_previous_output_problematic = True

            if is_previous_output_problematic:
                summary_of_previous = "Previous persona's output had issues (malformed JSON or content misalignment). "
                if isinstance(previous_output_for_llm, dict):
                    if previous_output_for_llm.get("CRITIQUE_SUMMARY"):
                        summary_of_previous += f"Summary of previous critique: {previous_output_for_llm['CRITIQUE_SUMMARY']}"
                    elif previous_output_for_llm.get("ANALYSIS_SUMMARY"):
                        summary_of_previous += f"Summary of previous analysis: {previous_output_for_llm['ANALYSIS_SUMMARY']}"
                    elif previous_output_for_llm.get("general_output"):
                        summary_of_previous += f"Summary of previous general output: {previous_output_for_llm['general_output'][:200]}..."
                    else:
                        summary_of_previous += "Details in malformed_blocks."
                else: # It's a string error message
                    summary_of_previous += f"Raw error snippet: {str(previous_output_for_llm)[:200]}..."
                current_prompt_parts.append(f"Previous Debate Output Summary (with issues):\n{summary_of_previous}\n\n")
            else:
                # If previous output was good, pass its main summary/content
                if isinstance(previous_output_for_llm, dict):
                    if previous_output_for_llm.get("CRITIQUE_SUMMARY"):
                        current_prompt_parts.append(f"Previous Debate Output:\n{json.dumps({'CRITIQUE_SUMMARY': previous_output_for_llm['CRITIQUE_SUMMARY'], 'SUGGESTIONS': previous_output_for_llm.get('SUGGESTIONS', [])}, indent=2)}\n\n")
                    elif previous_output_for_llm.get("ANALYSIS_SUMMARY"):
                        current_prompt_parts.append(f"Previous Debate Output:\n{json.dumps({'ANALYSIS_SUMMARY': previous_output_for_llm['ANALYSIS_SUMMARY'], 'IMPACTFUL_SUGGESTIONS': previous_output_for_llm.get('IMPACTFUL_SUGGESTIONS', [])}, indent=2)}\n\n")
                    elif previous_output_for_llm.get("general_output"):
                        current_prompt_parts.append(f"Previous Debate Output:\n{json.dumps({'general_output': previous_output_for_llm['general_output']}, indent=2)}\n\n")
                    else:
                        current_prompt_parts.append(f"Previous Debate Output:\n{json.dumps(previous_output_for_llm, indent=2)}\n\n")
                else:
                    current_prompt_parts.append(f"Previous Debate Output:\n{previous_output_for_llm}\n\n")

            current_prompt = "".join(current_prompt_parts)

            # Use the dynamically calculated budget for this persona's turn
            max_output_tokens_for_turn = self.phase_budgets.get("persona_turn_budgets", {}).get(persona_name, persona_config.max_tokens)
            estimated_tokens = self.tokenizer.count_tokens(current_prompt) + max_output_tokens_for_turn
            self.check_budget("debate", estimated_tokens, persona_name)

            try:
                output = self._execute_llm_turn(persona_name, persona_config, current_prompt, "debate", max_output_tokens_for_turn)

                # NEW: Content Alignment Validation
                is_aligned, validation_message, nuanced_feedback = self.content_validator.validate(persona_name, output) # Get nuanced feedback
                if not is_aligned:
                    has_content_misalignment = True # Set flag
                    self._log_with_context("warning", f"Content misalignment detected for {persona_name}: {validation_message}",
                                           persona=persona_name, validation_message=validation_message)
                    # Add a malformed block to indicate content drift
                    self.intermediate_steps.setdefault("malformed_blocks", []).extend([{
                        "type": "CONTENT_MISALIGNMENT",
                        "message": f"Output from {persona_name} drifted from the core topic: {validation_message}",
                        "persona": persona_name,
                        "raw_string_snippet": str(output)[:500]
                    }] + nuanced_feedback.get("malformed_blocks", [])) # Include any malformed blocks from content validator
                    
                    # If the output is a dict, we can add a specific field.
                    if isinstance(output, dict):
                        output["content_misalignment_warning"] = validation_message
                    else:
                        output = f"WARNING: Content misalignment detected: {validation_message}\n\n{output}"
                    
                # NEW: Check for ConflictReport from Devils_Advocate
                if persona_name == "Devils_Advocate" and isinstance(output, dict):
                    try:
                        # Validate output against ConflictReport schema
                        conflict_report = ConflictReport.model_validate(output)
                        if conflict_report.conflict_found:
                            resolution_result = self._trigger_conflict_sub_debate(conflict_report, debate_history)
                            if resolution_result and resolution_result.get("conflict_resolved"):
                                # Update previous_output to reflect resolution, clear unresolved conflict
                                previous_output_for_llm = {"status": "conflict_resolved", "resolution": resolution_result["resolution_summary"]}
                                self.intermediate_steps["Conflict_Resolution_Attempt"] = resolution_result
                                self.intermediate_steps["Unresolved_Conflict"] = None
                            else:
                                # If not resolved, keep the conflict in intermediate steps
                                self.intermediate_steps["Unresolved_Conflict"] = conflict_report.model_dump()
                                previous_output_for_llm = {"status": "conflict_unresolved", "conflict_report": conflict_report.model_dump()}
                        else:
                            # No conflict found, proceed normally
                            self._log_with_context("info", f"Devils_Advocate reported no conflict: {conflict_report.summary}")
                            self.intermediate_steps["Unresolved_Conflict"] = None
                            self.intermediate_steps["Conflict_Resolution_Attempt"] = None
                            previous_output_for_llm = {"status": "no_conflict_reported", "summary": conflict_report.summary}
                    except ValidationError:
                        # If the output is not a valid ConflictReport, treat it as a general output
                        self._log_with_context("warning", f"Devils_Advocate output was not a valid ConflictReport. Treating as general output.")
                        previous_output_for_llm = output # Use the raw output as previous output
                    except Exception as e:
                        self._log_with_context("error", f"Error processing Devils_Advocate output: {e}", exc_info=True)
                        previous_output_for_llm = {"error": f"Error processing Devils_Advocate output: {e}"}
                debate_history.append({"persona": persona_name, "output": output})
                previous_output_for_llm = output # Update for the next turn
            except Exception as e:
                self._log_with_context("error", f"Error during {persona_name} turn: {e}", persona=persona_name, exc_info=True, original_exception=e)
                self.rich_console.print(f"[red]Error during {persona_name} turn: {e}[/red]")
                previous_output_for_llm = {"error": f"Turn failed for {persona_name}: {str(e)}", "malformed_blocks": [{"type": "DEBATE_TURN_ERROR", "message": str(e)}]}
                continue

        self._log_with_context("info", "All debate turns completed.")
        return debate_history

    def _trigger_conflict_sub_debate(self, conflict_report: ConflictReport, debate_history: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """
        Triggers a focused sub-debate to resolve a specific conflict.
        Returns a resolution summary or None if unresolved.
        """
        self._log_with_context("info", f"Initiating sub-debate for conflict: {conflict_report.summary}")

        # Determine personas for sub-debate: involved personas + a dynamically selected mediator
        sub_debate_personas_set = set(conflict_report.involved_personas)

        # Dynamically select a mediator based on conflict type
        if conflict_report.conflict_type in ["LOGICAL_INCONSISTENCY", "DATA_DISCREPANCY", "METHODOLOGY_DISAGREEMENT"]:
            # For analytical disagreements, a critic might help find common ground
            if "Constructive_Critic" in self.all_personas:
                sub_debate_personas_set.add("Constructive_Critic")

        # For final decision-making, resource constraints, or security vs. performance trade-offs,
        # an arbitrator is key. Also, ensure the main synthesis persona is involved if different.
        main_synthesis_persona = "Self_Improvement_Analyst" if self.is_self_analysis else \
                                 ("Impartial_Arbitrator" if self.domain == "Software Engineering" else "General_Synthesizer")
        if main_synthesis_persona in self.all_personas and main_synthesis_persona not in sub_debate_personas_set:
            sub_debate_personas_set.add(main_synthesis_persona)

        # Filter to only include personas that actually exist in all_personas
        sub_debate_personas = [p for p in list(sub_debate_personas_set) if p in self.all_personas]

        if not sub_debate_personas:
            self._log_with_context("warning", "No relevant personas found for conflict sub-debate. Skipping resolution.")
            return None

        # Allocate a small portion of remaining budget for sub-debate
        remaining_tokens = self.max_total_tokens_budget - self.tokens_used
        sub_debate_budget = max(1000, min(5000, int(remaining_tokens * 0.1)))

        if sub_debate_budget < 1000:
            self._log_with_context("warning", "Insufficient tokens for sub-debate, skipping conflict resolution.")
            return None

        # Construct a focused prompt for the sub-debate
        sub_debate_prompt = f"""
        CRITICAL CONFLICT RESOLUTION REQUIRED:

        The following conflict has been identified in the main debate:
        Conflict Type: {conflict_report.conflict_type}
        Summary: {conflict_report.summary}
        Involved Personas: {', '.join(conflict_report.involved_personas)}
        Conflicting Outputs Snippet: {conflict_report.conflicting_outputs_snippet}
        Proposed Resolution Paths: {'; '.join(conflict_report.proposed_resolution_paths) if conflict_report.proposed_resolution_paths else 'None provided.'}

        Your task is to:
        1.  Analyze the conflicting perspectives from the debate history.
        2.  Identify the root cause of the disagreement.
        3.  Propose a definitive resolution or a clear compromise.
        4.  Provide a concise rationale for your resolution.

        Focus ONLY on resolving this specific conflict. Output a clear resolution summary.
        """

        resolution_attempts = []
        for persona_name in sub_debate_personas:
            self.status_callback(
                message=f"Sub-Debate: [bold]{persona_name.replace('_', ' ')}[/bold] resolving conflict...",
                state="running",
                current_total_tokens=self.tokens_used,
                current_total_cost=self.get_total_estimated_cost(),
                progress_pct=self.get_progress_pct("debate"),
                current_persona_name=persona_name
            )

            # Get potentially adjusted persona config from PersonaManager
            persona_config = self.persona_manager.get_adjusted_persona_config(persona_name)

            # Calculate max_output_tokens for this specific sub-debate turn
            # Distribute sub_debate_budget among sub_debate_personas
            max_output_tokens_for_sub_debate_turn = sub_debate_budget // len(sub_debate_personas)
            max_output_tokens_for_sub_debate_turn = max(250, min(max_output_tokens_for_sub_debate_turn, persona_config.max_tokens)) # Respect persona's own max_tokens, ensure minimum
            
            try:
                # Pass the full debate history as context for the sub-debate
                turn_output = self._execute_llm_turn(
                    persona_name, persona_config, sub_debate_prompt, "sub_debate_phase", max_output_tokens_for_sub_debate_turn
                )
                resolution_attempts.append({"persona": persona_name, "output": turn_output})
            except Exception as e:
                self._log_with_context("error", f"Sub-debate turn for {persona_name} failed: {e}", exc_info=True)
                self.rich_console.print(f"[red]Error during sub-debate turn for {persona_name}: {e}[/red]")
            finally:
                pass # No need to reset persona_config.max_tokens as it's not mutated.

        # Synthesize sub-debate results into a final resolution
        if resolution_attempts:
            final_resolution_prompt_base = f"""
            Synthesize the following sub-debate attempts to resolve a conflict into a single, clear resolution.
            Conflict: {conflict_report.summary}
            Sub-debate results: {json.dumps(resolution_attempts, indent=2)}

            Provide a final resolution and its rationale.
            """
            # Allocate a portion of the sub_debate_budget for the synthesis of the resolution
            sub_debate_synthesis_budget = max(500, int(sub_debate_budget * 0.3)) # e.g., 30% for synthesis
            final_resolution_prompt = self.tokenizer.trim_text_to_tokens(final_resolution_prompt_base, sub_debate_synthesis_budget)

            synthesizer_persona = self.all_personas.get("Impartial_Arbitrator") or self.all_personas.get("General_Synthesizer")
            if synthesizer_persona:
                try:
                    final_resolution = self._execute_llm_turn(
                        synthesizer_persona.name, synthesizer_persona, final_resolution_prompt, "sub_debate_synthesis", synthesizer_persona.max_tokens # Use persona's max_tokens for synthesis
                    )
                    self._log_with_context("info", "Conflict successfully resolved.", resolution_summary=final_resolution)
                    return {"conflict_resolved": True, "resolution_summary": final_resolution}
                except Exception as e:
                    self._log_with_context("error", f"Failed to synthesize conflict resolution: {e}", exc_info=True)
                    self.rich_console.print(f"[red]Failed to synthesize conflict resolution: {e}[/red]")

        self._log_with_context("warning", "Conflict could not be resolved in sub-debate.")
        return None

    # NEW: Add a helper to summarize debate history for sub-debate synthesis
    def _summarize_sub_debate_history_for_llm(self, resolution_attempts: List[Dict[str, Any]], max_tokens: int) -> List[Dict[str, Any]]:
        return self._summarize_debate_history_for_llm(resolution_attempts, max_tokens)

    # --- NEW HELPER FUNCTIONS FOR SELF_IMPROVEMENT_ANALYST PROMPT TRUNCATION ---
    def _summarize_metrics_for_llm(self, metrics: Dict[str, Any], max_tokens: int) -> Dict[str, Any]:
        # Ensure a minimum number of tokens are reserved for the *summary string* if all else fails
        MIN_SUMMARY_STRING_TOKENS = 150
        effective_max_tokens = max(MIN_SUMMARY_STRING_TOKENS, max_tokens)

        """
        Intelligently summarizes the metrics dictionary to fit within a token budget.
        Prioritizes high-level summaries and truncates verbose lists like 'detailed_issues'.
        Ensures critical configuration and deployment analysis are preserved.
        """
        summarized_metrics = json.loads(json.dumps(metrics)) # Deep copy

        # Estimate current token count
        current_tokens = self.tokenizer.count_tokens(json.dumps(summarized_metrics))
        
        # If already within budget, return as is
        if current_tokens <= effective_max_tokens:
            return summarized_metrics
        
        self._log_with_context("debug", f"Summarizing metrics for LLM. Current tokens: {current_tokens}, Max: {max_tokens}")

        # --- Prioritize critical sections: configuration_analysis and deployment_robustness ---
        # Allocate a fixed, generous portion of the budget to these, or ensure they are minimally summarized.
        CRITICAL_SECTION_TOKEN_BUDGET = int(max_tokens * 0.4) # 40% of the budget for critical sections
        CRITICAL_SECTION_TOKEN_BUDGET = max(100, min(CRITICAL_SECTION_TOKEN_BUDGET, effective_max_tokens * 0.5)) # Cap at 50% of effective_max_tokens
        
        critical_sections_content = {}
        if 'configuration_analysis' in summarized_metrics:
            critical_sections_content['configuration_analysis'] = summarized_metrics['configuration_analysis']
            del summarized_metrics['configuration_analysis'] # Remove temporarily
        if 'deployment_robustness' in summarized_metrics:
            critical_sections_content['deployment_robustness'] = summarized_metrics['deployment_robustness']
            del summarized_metrics['deployment_robustness'] # Remove temporarily

        # Summarize critical sections if they exceed their dedicated budget
        summarized_critical_sections = {}
        critical_sections_tokens = self.tokenizer.count_tokens(json.dumps(critical_sections_content))
        
        if critical_sections_tokens > CRITICAL_SECTION_TOKEN_BUDGET:
            self._log_with_context("debug", f"Critical sections (config/deployment) exceed dedicated budget. Summarizing.")
            # This is a simplified summarization. A more advanced one would be persona-aware.
            for key, content in critical_sections_content.items():
                if key == 'configuration_analysis' and content:
                    # Keep high-level info, truncate lists of hooks/steps
                    summarized_config = {
                        "ci_workflow": {"name": content.get("ci_workflow", {}).get("name"), "jobs_count": len(content.get("ci_workflow", {}).get("jobs", {}))},
                        "pre_commit_hooks_count": len(content.get("pre_commit_hooks", [])),
                        "pyproject_toml_tools": {k: True for k, v in content.get("pyproject_toml", {}).items() if v},
                    }
                    summarized_critical_sections[key] = summarized_config
                elif key == 'deployment_robustness' and content:
                    summarized_deployment = {
                        "dockerfile_present": content.get("dockerfile_present"),
                        "dockerfile_healthcheck_present": content.get("dockerfile_healthcheck_present"),
                        "dockerfile_non_root_user": content.get("dockerfile_non_root_user"),
                        "dockerfile_exposed_ports": content.get("dockerfile_exposed_ports"),
                        "prod_dependency_count": content.get("prod_dependency_count"),
                        "unpinned_prod_dependencies_count": len(content.get("unpinned_prod_dependencies", [])),
                    }
                    summarized_critical_sections[key] = summarized_deployment
                else:
                    summarized_critical_sections[key] = f"Summary of {key} (truncated due to token limits)."
        else:
            summarized_critical_sections = critical_sections_content # Use full content if it fits

        # Re-insert summarized critical sections
        summarized_metrics.update(summarized_critical_sections)
        current_tokens = self.tokenizer.count_tokens(json.dumps(summarized_metrics))
        remaining_budget_for_issues = effective_max_tokens - current_tokens
        
        # --- Dynamically truncate 'detailed_issues' and 'ruff_violations' ---
        # Prioritize 'detailed_issues' as they are more comprehensive
        for issue_list_key in ['detailed_issues', 'ruff_violations']:
            if 'code_quality' in summarized_metrics and issue_list_key in summarized_metrics['code_quality'] and summarized_metrics['code_quality'][issue_list_key]:
                original_issues = summarized_metrics['code_quality'][issue_list_key]
                
                # Estimate tokens per issue (average)
                # Use a small sample to estimate average token count per issue
                sample_size = min(5, len(original_issues))
                if sample_size > 0:
                    sample_tokens = self.tokenizer.count_tokens(json.dumps(original_issues[:sample_size]))
                    avg_tokens_per_issue = max(1, sample_tokens // sample_size) # Ensure at least 1 token per issue
                else:
                    avg_tokens_per_issue = 10 # Default estimate if no issues
                
                # Calculate how many issues can fit
                num_issues_to_keep = int(remaining_budget_for_issues / avg_tokens_per_issue)
                num_issues_to_keep = max(0, min(len(original_issues), num_issues_to_keep)) # Ensure non-negative and not exceeding original count

                if num_issues_to_keep < len(original_issues):
                    summarized_metrics['code_quality'][issue_list_key] = original_issues[:num_issues_to_keep]
                    self._log_with_context("debug", f"Truncated {issue_list_key} from {len(original_issues)} to {num_issues_to_keep}.")
                    current_tokens = self.tokenizer.count_tokens(json.dumps(summarized_metrics))
                    remaining_budget_for_issues = effective_max_tokens - current_tokens
                elif num_issues_to_keep == 0 and len(original_issues) > 0:
                    del summarized_metrics['code_quality'][issue_list_key]
                    self._log_with_context("debug", f"Removed {issue_list_key} entirely due to lack of budget.")
                    current_tokens = self.tokenizer.count_tokens(json.dumps(summarized_metrics))
                    remaining_budget_for_issues = effective_max_tokens - current_tokens
            elif 'code_quality' in summarized_metrics and issue_list_key in summarized_metrics['code_quality'] and not summarized_metrics['code_quality'][issue_list_key]:
                del summarized_metrics['code_quality'][issue_list_key] # Remove empty lists
        
        # If still over budget, consider removing entire verbose sections
        if current_tokens > effective_max_tokens:
            if 'code_quality' in summarized_metrics:
                if 'detailed_issues' in summarized_metrics['code_quality']:
                    del summarized_metrics['code_quality']['detailed_issues']
                    self._log_with_context("debug", "Removed detailed_issues entirely.")
                    if current_tokens <= effective_max_tokens:
                        return summarized_metrics
                if 'ruff_violations' in summarized_metrics['code_quality']:
                    del summarized_metrics['code_quality']['ruff_violations']
                    self._log_with_context("debug", "Removed ruff_violations entirely.")
                    if current_tokens <= effective_max_tokens:
                        return summarized_metrics

            if 'maintainability' in summarized_metrics and 'test_coverage_summary' in summarized_metrics['maintainability']:
                if 'coverage_details' in summarized_metrics['maintainability']['test_coverage_summary']:
                    del summarized_metrics['maintainability']['test_coverage_summary']['coverage_details']
                    self._log_with_context("debug", "Removed coverage_details.")
                    if current_tokens <= effective_max_tokens:
                        return summarized_metrics

        if current_tokens > effective_max_tokens:
            self._log_with_context("warning", "Metrics still too large after truncation. Converting to high-level summary string.")
            summary_str = f"Overall Code Quality: Ruff issues: {metrics['code_quality']['ruff_issues_count']}, Code Smells: {metrics['code_quality']['code_smells_count']}. " \
                          f"Security Issues: Bandit: {metrics['security']['bandit_issues_count']}, AST: {metrics['security']['ast_security_issues_count']}. " \
                          f"Token Usage: {metrics['performance_efficiency']['token_usage_stats']['total_tokens']} tokens, Cost: ${metrics['performance_efficiency']['token_usage_stats']['total_cost_usd']:.4f}. " \
                          f"Robustness: Schema failures: {metrics['robustness']['schema_validation_failures_count']}, Unresolved conflicts: {metrics['robustness']['unresolved_conflict_present']}."
            # Ensure the summary string itself fits within the minimum tokens
            trimmed_summary_str = self.tokenizer.trim_text_to_tokens(summary_str, MIN_SUMMARY_STRING_TOKENS)
            return {"summary_string": trimmed_summary_str}
          
        return summarized_metrics

    def _summarize_debate_history_for_llm(self, debate_history: List[Dict[str, Any]], max_tokens: int) -> List[Dict[str, Any]]:
        """
        Summarizes the debate history to fit within a token budget.
        Prioritizes recent turns and concise summaries of each turn's output.
        """
        summarized_history = []
        current_tokens = 0

        # Start with the most recent turns
        for turn in reversed(debate_history):
            turn_copy = json.loads(json.dumps(turn))

            # Try to summarize the 'output' field if it's a dict
            if 'output' in turn_copy and isinstance(turn_copy['output'], dict):
                if 'CRITIQUE_SUMMARY' in turn_copy['output']:
                    turn_copy['output'] = {'CRITIQUE_SUMMARY': turn_copy['output']['CRITIQUE_SUMMARY']}
                elif 'summary' in turn_copy['output']:
                    turn_copy['output'] = {'summary': turn_copy['output']['summary']}
                elif 'general_output' in turn_copy['output']:
                    turn_copy['output'] = {'general_output': turn_copy['output']['general_output'][:200] + "..." if len(turn_copy['output']['general_output']) > 200 else turn_copy['output']['general_output']}
                elif 'architectural_analysis' in turn_copy['output']: # Corrected key from architecturalAnalysis
                    turn_copy['output'] = {'overall_assessment': turn_copy['output']['architectural_analysis']['overall_assessment']}
                elif 'operational_analysis' in turn_copy['output']:
                    turn_copy['output'] = {'operational_analysis_summary': "Operational analysis performed."}
                elif 'CRITIQUE_POINTS' in turn_copy['output']:
                    turn_copy['output'] = {'CRITIQUE_SUMMARY': turn_copy['output'].get('CRITIQUE_SUMMARY', 'Critique provided.')}
                else:
                    # Fallback for other dicts: keep only a few top-level keys or convert to string
                    turn_copy['output'] = {k: str(v)[:100] for k, v in list(turn_copy['output'].items())[:3]}
            elif 'output' in turn_copy and isinstance(turn_copy['output'], str):
                turn_copy['output'] = turn_copy['output'][:200] + "..." if len(turn_copy['output']) > 200 else turn_copy['output']

            turn_tokens = self.tokenizer.count_tokens(json.dumps(turn_copy))

            if current_tokens + turn_tokens <= max_tokens:
                summarized_history.insert(0, turn_copy)
                current_tokens += turn_tokens
            else:
                self._log_with_context("debug", f"Stopped summarizing debate history due to token limit. Included {len(summarized_history)} turns.")
                break

        if not summarized_history and debate_history:
            self._log_with_context("warning", "Debate history too large, providing minimal summary.")
            return [{"summary": f"Debate history contains {len(debate_history)} turns. Too verbose to include in full."}]

        return summarized_history
    # --- END NEW HELPER FUNCTIONS ---

    # --- NEW METHOD: _perform_synthesis_persona_turn ---
    def _perform_synthesis_persona_turn(self, persona_sequence: List[str], debate_persona_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Executes the final synthesis persona turn based on the debate history.
        """
        self._log_with_context("info", "Executing final synthesis persona turn.")

        # Determine the appropriate synthesis persona
        synthesis_persona_name = ""
        if self.is_self_analysis:
            synthesis_persona_name = "Self_Improvement_Analyst"
        elif self.domain == "Software Engineering":
            synthesis_persona_name = "Impartial_Arbitrator"
        else:
            synthesis_persona_name = "General_Synthesizer"

        synthesis_persona_config = self.all_personas.get(synthesis_persona_name)
        if not synthesis_persona_config:
            self._log_with_context("error", f"Synthesis persona '{synthesis_persona_name}' configuration not found. Falling back to General_Synthesizer.")
            synthesis_persona_name = "General_Synthesizer"
            synthesis_persona_config = self.all_personas.get(synthesis_persona_name)
            if not synthesis_persona_config:
                raise ChimeraError("No synthesis persona configuration found.")

        # Prepare the prompt for the synthesis persona
        synthesis_prompt_parts = [f"Initial Problem: {self.initial_prompt}\n\n"]

        # Summarize debate history to fit within token limits
        # Allocate a portion of the synthesis budget for the debate history summary
        debate_history_summary_budget = int(self.phase_budgets["synthesis"] * 0.4) # e.g., 40% for history
        summarized_debate_history = self._summarize_debate_history_for_llm(debate_persona_results, debate_history_summary_budget)
        synthesis_prompt_parts.append(f"Debate History:\n{json.dumps(summarized_debate_history, indent=2)}\n\n")

        # Include conflict resolution if it happened
        if self.intermediate_steps.get("Conflict_Resolution_Attempt"):
            synthesis_prompt_parts.append(f"Conflict Resolution Summary: {json.dumps(self.intermediate_steps['Conflict_Resolution_Attempt']['resolution_summary'], indent=2)}\n\n")
        elif self.intermediate_steps.get("Unresolved_Conflict"):
            synthesis_prompt_parts.append(f"Unresolved Conflict: {json.dumps(self.intermediate_steps['Unresolved_Conflict'], indent=2)}\n\n")

        # Special handling for Self_Improvement_Analyst
        if synthesis_persona_name == "Self_Improvement_Analyst":
            # Collect metrics
            metrics_collector = ImprovementMetricsCollector(
                initial_prompt=self.initial_prompt,
                debate_history=debate_persona_results,
                intermediate_steps=self.intermediate_steps,
                codebase_context=self.codebase_context,
                tokenizer=self.tokenizer,
                llm_provider=self.llm_provider,
                persona_manager=self.persona_manager,
                content_validator=self.content_validator # Pass the content validator
            )
            collected_metrics = metrics_collector.collect_all_metrics()
            self.intermediate_steps["Self_Improvement_Metrics"] = collected_metrics

            # Summarize metrics to fit into the prompt
            metrics_summary_budget = int(self.phase_budgets["synthesis"] * 0.3) # e.g., 30% for metrics
            summarized_metrics = self._summarize_metrics_for_llm(collected_metrics, metrics_summary_budget)
            synthesis_prompt_parts.append(f"Objective Metrics and Analysis:\n{json.dumps(summarized_metrics, indent=2)}\n\n")

            # Add specific instructions for Self_Improvement_Analyst
            synthesis_prompt_parts.append(
                "Based on the debate history, conflict resolution (if any), and objective metrics, "
                "critically analyze Project Chimera's codebase for self-improvement. "
                "Identify the most impactful code changes for self-improvement, focusing on the 80/20 Pareto principle. "
                "Prioritize enhancements to reasoning quality, robustness, efficiency, and developer maintainability. "
                "For each suggestion, provide a clear rationale and a specific, actionable code modification. "
                "Your output MUST strictly adhere to the SelfImprovementAnalysisOutputV1 JSON schema."
            )
            # Ensure the schema for Self_Improvement_Analyst is correctly set to V1
            self.PERSONA_OUTPUT_SCHEMAS["Self_Improvement_Analyst"] = SelfImprovementAnalysisOutputV1

        else:
            # General synthesis instructions
            synthesis_prompt_parts.append(
                "Based on the initial problem and the debate history, synthesize a final, comprehensive answer. "
                "Address all aspects of the initial problem and integrate insights from all personas. "
                "Your output MUST strictly adhere to the LLMOutput JSON schema."
            )
            self.PERSONA_OUTPUT_SCHEMAS["Impartial_Arbitrator"] = LLMOutput
            self.PERSONA_OUTPUT_SCHEMAS["General_Synthesizer"] = GeneralOutput


        final_synthesis_prompt_raw = "\n".join(synthesis_prompt_parts)
        
        # NEW: Trim the final synthesis prompt to ensure it fits within the input token budget
        # The max_output_tokens_for_turn is the budget for the *output*.
        # We need to consider the *input* budget for the prompt itself.
        # The total budget for synthesis phase is self.phase_budgets["synthesis"].
        # We need to ensure that the input prompt + expected output tokens does not exceed this.
        # Let's reserve a portion of the synthesis budget for the input prompt.
        input_budget_for_synthesis_prompt = int(self.phase_budgets["synthesis"] * 0.7) # e.g., 70% for input, 30% for output
        
        final_synthesis_prompt = self.tokenizer.trim_text_to_tokens(
            final_synthesis_prompt_raw,
            input_budget_for_synthesis_prompt,
            truncation_indicator="\n... (truncated for token limits) ..."
        )

        # Estimate tokens for the synthesis turn (this is for the *output* tokens)
        max_output_tokens_for_turn = self.phase_budgets["synthesis"] - self.tokenizer.count_tokens(final_synthesis_prompt)
        max_output_tokens_for_turn = max(500, max_output_tokens_for_turn) # Ensure a minimum output budget
        
        estimated_tokens_for_turn = self.tokenizer.count_tokens(final_synthesis_prompt) + max_output_tokens_for_turn
        self.check_budget("synthesis", estimated_tokens_for_turn, synthesis_persona_name)

        # Execute the LLM turn for synthesis
        synthesis_output = self._execute_llm_turn(
            synthesis_persona_name,
            synthesis_persona_config,
            final_synthesis_prompt,
            "synthesis",
            max_output_tokens_for_turn # Pass the correct max_output_tokens
        )
        self._log_with_context("info", "Final synthesis persona turn completed.")
        return synthesis_output
    # --- END NEW METHOD ---

    def _finalize_debate_results(self, context_persona_turn_results: Optional[Dict[str, Any]], debate_persona_results: List[Dict[str, Any]], synthesis_persona_results: Optional[Dict[str, Any]]) -> Tuple[Any, Dict[str, Any]]:
        """Synthesizes the final answer and prepares the intermediate steps for display."""
        final_answer = synthesis_persona_results

        if not isinstance(final_answer, dict):
            final_answer = {"general_output": str(final_answer), "malformed_blocks": []}
        if "malformed_blocks" not in final_answer:
            final_answer["malformed_blocks"] = []

        # NEW: Consolidate code changes for SelfImprovementAnalysisOutput
        # Check if it's a SelfImprovementAnalysisOutput (versioned or V1 direct)
        if isinstance(final_answer, dict):
            # Check if it's already the versioned wrapper
            if final_answer.get("version") == "1.0" and "data" in final_answer:
                final_answer["data"] = self._consolidate_self_improvement_code_changes(final_answer["data"])
                self._log_with_context("info", "Self-improvement code changes consolidated (versioned output).")
            # Check if it's a direct V1 output that needs wrapping
            elif "ANALYSIS_SUMMARY" in final_answer and "IMPACTFUL_SUGGESTIONS" in final_answer:
                # It's a V1 direct output, consolidate and then wrap it
                v1_data_consolidated = self._consolidate_self_improvement_code_changes(final_answer)
                final_answer = SelfImprovementAnalysisOutput(
                    version="1.0",
                    data=v1_data_consolidated,
                    malformed_blocks=final_answer.get("malformed_blocks", []) # Preserve existing malformed blocks
                ).model_dump(by_alias=True)
                self._log_with_context("info", "Self-improvement code changes consolidated and wrapped (direct V1 output).")


        self._update_intermediate_steps_with_totals()
        if "malformed_blocks" not in self.intermediate_steps:
            self.intermediate_steps["malformed_blocks"] = []

        # Ensure final_answer reflects resolution or unresolved conflict
        if self.intermediate_steps.get("Conflict_Resolution_Attempt"):
            # The resolution_summary from _trigger_conflict_sub_debate is already a dict,
            # so we should merge it or assign it appropriately.
            # Assuming final_answer is the main output, we can add a dedicated field.
            final_answer["CONFLICT_RESOLUTION"] = self.intermediate_steps["Conflict_Resolution_Attempt"]["resolution_summary"]
            final_answer["UNRESOLVED_CONFLICT"] = None
        elif self.intermediate_steps.get("Unresolved_Conflict"):
            final_answer["UNRESOLVED_CONFLICT"] = self.intermediate_steps["Unresolved_Conflict"]["summary"]
            final_answer["CONFLICT_RESOLUTION"] = None

        return final_answer, self.intermediate_steps

    def _update_intermediate_steps_with_totals(self):
        """Updates the intermediate steps dictionary with total token usage and estimated cost."""
        self.intermediate_steps["Total_Tokens_Used"] = self.tokens_used
        self.intermediate_steps["Total_Estimated_Cost_USD"] = self.get_total_estimated_cost()

    @handle_errors(default_return=None, log_level="ERROR") # Apply the decorator here
    def run_debate(self) -> Tuple[Any, Dict[str, Any]]:
        """
        Orchestrates the full Socratic debate process.
        Returns the final synthesized answer and a dictionary of intermediate steps.
        """
        self._initialize_debate_state()

        # Determine initial persona sequence (before context analysis)
        initial_persona_sequence = self._get_final_persona_sequence(self.initial_prompt, None)
        self.intermediate_steps["Persona_Sequence_Initial"] = initial_persona_sequence

        # Phase 1: Context Analysis (if applicable)
        self.status_callback(message="Phase 1: Analyzing Context...", state="running",
                             current_total_tokens=self.tokens_used, current_total_cost=self.get_total_estimated_cost(),
                             progress_pct=self.get_progress_pct("context"))
        context_analysis_results = self._perform_context_analysis(tuple(initial_persona_sequence))
        self.intermediate_steps["Context_Analysis_Output"] = context_analysis_results

        # Re-determine persona sequence after context analysis to allow for dynamic adjustments
        # based on the insights gained from the context.
        persona_sequence = self._get_final_persona_sequence(self.initial_prompt, context_analysis_results)
        self.intermediate_steps["Persona_Sequence"] = persona_sequence

        # NEW: Distribute debate budget among the *final* persona sequence
        self._distribute_debate_persona_budgets(persona_sequence)

        # Phase 2: Context Persona Turn (if in sequence)
        context_persona_turn_results = None
        if "Context_Aware_Assistant" in persona_sequence:
            self.status_callback(message="Phase 2: Context-Aware Assistant Turn...", state="running",
                                 current_total_tokens=self.tokens_used, current_total_cost=self.get_total_estimated_cost(),
                                 progress_pct=self.get_progress_pct("debate"), current_persona_name="Context_Aware_Assistant")
            context_persona_turn_results = self._process_context_persona_turn(persona_sequence, context_analysis_results)
            self.intermediate_steps["Context_Aware_Assistant_Output"] = context_persona_turn_results

        # Phase 3: Main Debate Persona Turns
        self.status_callback(message="Phase 3: Executing Debate Turns...", state="running",
                             current_total_tokens=self.tokens_used, current_total_cost=self.get_total_estimated_cost(),
                             progress_pct=self.get_progress_pct("debate"))
        debate_persona_results = self._execute_debate_persona_turns(persona_sequence, context_persona_turn_results)
        self.intermediate_steps["Debate_History"] = debate_persona_results

        # Phase 4: Final Synthesis Persona Turn
        self.status_callback(message="Phase 4: Synthesizing Final Answer...", state="running",
                             current_total_tokens=self.tokens_used, current_total_cost=self.get_total_estimated_cost(),
                             progress_pct=self.get_progress_pct("synthesis"))

        # THIS IS THE MISSING CALL:
        synthesis_persona_results = self._perform_synthesis_persona_turn(persona_sequence, debate_persona_results)
        self.intermediate_steps["Final_Synthesis_Output"] = synthesis_persona_results

        # Finalize results and update totals
        self.status_callback(message="Finalizing Results...", state="running",
                             current_total_tokens=self.tokens_used, current_total_cost=self.get_total_estimated_cost(),
                             progress_pct=0.95)
        final_answer, intermediate_steps = self._finalize_debate_results(context_persona_turn_results, debate_persona_results, synthesis_persona_results)

        self.status_callback(message="Socratic Debate Complete!", state="complete",
                             current_total_tokens=self.tokens_used, current_total_cost=self.get_total_estimated_cost(),
                             progress_pct=1.0)
        self._log_with_context("info", "Socratic Debate process completed successfully.",
                               total_tokens=self.tokens_used, total_cost=self.get_total_estimated_cost())

        return final_answer, intermediate_steps

    def _consolidate_self_improvement_code_changes(self, analysis_output: Dict[str, Any]) -> Dict[str, Any]:
        """
        Consolidates multiple CODE_CHANGES_SUGGESTED for the same file within
        SelfImprovementAnalysisOutput. Also filters out no-op changes.
        """
        if "IMPACTFUL_SUGGESTIONS" not in analysis_output:
            return analysis_output

        consolidated_suggestions = []
        for suggestion in analysis_output["IMPACTFUL_SUGGESTIONS"]:
            if "CODE_CHANGES_SUGGESTED" not in suggestion or not suggestion["CODE_CHANGES_SUGGESTED"]:
                consolidated_suggestions.append(suggestion)
                continue

            file_changes_map = defaultdict(list)
            for change_data in suggestion["CODE_CHANGES_SUGGESTED"]:
                try:
                    # Validate and convert to CodeChange model for easier manipulation
                    code_change = CodeChange.model_validate(change_data)
                    file_changes_map[code_change.file_path].append(code_change)
                except ValidationError as e:
                    self._log_with_context("warning", f"Malformed CodeChange item during consolidation: {e}. Skipping.",
                                           raw_change_data=change_data)
                    # Add to malformed_blocks if not already there
                    analysis_output.setdefault("malformed_blocks", []).append({
                        "type": "CODE_CHANGE_CONSOLIDATION_ERROR",
                        "message": f"Malformed CodeChange item skipped during consolidation: {e}",
                        "raw_string_snippet": str(change_data)[:500]
                    })
                    continue

            new_code_changes_for_suggestion = []
            for file_path, changes_for_file in file_changes_map.items():
                # Handle multiple changes for the same file
                # Prioritize REMOVE, then ADD, then consolidate MODIFY
                remove_actions = [c for c in changes_for_file if c.action == "REMOVE"]
                add_actions = [c for c in changes_for_file if c.action == "ADD"]
                modify_actions = [c for c in changes_for_file if c.action == "MODIFY"]

                if remove_actions:
                    # If any REMOVE action, assume the file is to be removed.
                    # Combine all lines to remove if multiple REMOVE actions exist.
                    all_lines_to_remove = []
                    for ra in remove_actions:
                        all_lines_to_remove.extend(ra.lines)
                    new_code_changes_for_suggestion.append(CodeChange(
                        FILE_PATH=file_path,
                        ACTION="REMOVE",
                        LINES=list(set(all_lines_to_remove)) # Ensure unique lines
                    ).model_dump(by_alias=True))
                    self._log_with_context("info", f"Consolidated multiple changes for {file_path} into a single REMOVE action.")
                elif add_actions:
                    # If any ADD action, assume the file is to be added.
                    # For simplicity, take the first ADD action's content.
                    new_code_changes_for_suggestion.append(add_actions[0].model_dump(by_alias=True))
                    self._log_with_context("info", f"Consolidated multiple changes for {file_path} into a single ADD action.")
                elif modify_actions:
                    # Consolidate multiple MODIFY actions into a single diff
                    original_content = self.codebase_context.get(file_path, "")
                    
                    # Determine the final content after all modifications
                    final_content_for_diff = original_content
                    last_full_content_provided = None
                    
                    # Iterate through modify actions to find the most recent full_content
                    for mod_change in modify_actions:
                        if mod_change.full_content is not None:
                            last_full_content_provided = mod_change.full_content
                    
                    consolidated_diff_content = None

                    if last_full_content_provided is not None:
                        final_content_for_diff = last_full_content_provided
                        # Generate unified diff
                        diff_lines = difflib.unified_diff(
                            original_content.splitlines(keepends=True),
                            final_content_for_diff.splitlines(keepends=True),
                            fromfile=f"a/{file_path}",
                            tofile=f"b/{file_path}",
                            lineterm=''
                        )
                        consolidated_diff_content = "".join(diff_lines)
                        self._log_with_context("debug", f"Generated diff from FULL_CONTENT for {file_path}.")
                    else:
                        # If no FULL_CONTENT was provided, check if any DIFF_CONTENT was provided.
                        # If multiple DIFF_CONTENTs, take the last one.
                        last_diff_from_llm = None
                        for mod_change in modify_actions:
                            if mod_change.diff_content is not None:
                                last_diff_from_llm = mod_change.diff_content
                        
                        if last_diff_from_llm is not None:
                            # If LLM provided a diff, use it directly.
                            consolidated_diff_content = last_diff_from_llm
                            self._log_with_context("debug", f"Using LLM-provided DIFF_CONTENT for {file_path}.")
                        else:
                            # If no FULL_CONTENT and no DIFF_CONTENT, then it's a no-op or malformed.
                            self._log_with_context("info", f"Consolidated MODIFY for {file_path} resulted in no effective change (no FULL_CONTENT or DIFF_CONTENT provided). Removing from suggestions.")
                            analysis_output.setdefault("malformed_blocks", []).append({
                                "type": "NO_OP_CODE_CHANGE_CONSOLIDATED",
                                "message": f"Consolidated MODIFY for {file_path} resulted in no effective change. Removed from final suggestions.",
                                "file_path": file_path,
                                "suggestion_area": suggestion.get("AREA")
                            })
                            continue # Skip adding this no-op change
                    
                    if consolidated_diff_content and consolidated_diff_content.strip():
                        new_code_changes_for_suggestion.append(CodeChange(
                            FILE_PATH=file_path,
                            ACTION="MODIFY",
                            DIFF_CONTENT=consolidated_diff_content
                        ).model_dump(by_alias=True))
                        self._log_with_context("info", f"Consolidated multiple MODIFY actions for {file_path} into a single DIFF_CONTENT.")
                    else:
                        # NEW: Check if consolidated diff is a no-op
                        self._log_with_context("info", f"Consolidated MODIFY for {file_path} resulted in no effective change. Removing from suggestions.")
                        analysis_output.setdefault("malformed_blocks", []).append({
                            "type": "NO_OP_CODE_CHANGE_CONSOLIDATED",
                            "message": f"Consolidated MODIFY for {file_path} resulted in no effective change. Removed from final suggestions.",
                            "file_path": file_path,
                            "suggestion_area": suggestion.get("AREA")
                        })

            suggestion["CODE_CHANGES_SUGGESTED"] = new_code_changes_for_suggestion
            consolidated_suggestions.append(suggestion)

        analysis_output["IMPACTFUL_SUGGESTIONS"] = consolidated_suggestions
        return analysis_output