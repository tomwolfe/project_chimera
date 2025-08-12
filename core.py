# -*- coding: utf-8 -*-
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
import json # Added for structured logging helper
import logging
import random  # Needed for backoff jitter
from pathlib import Path
from collections import defaultdict
from typing import List, Dict, Tuple, Any, Callable, Optional, Type
from google import genai
from google.genai import types
from google.genai.errors import APIError # Import APIError
import traceback # Needed for error handling in core.py
from rich.console import Console
from pydantic import ValidationError
from functools import lru_cache # Import lru_cache for caching
import uuid # Added for request tracing

# --- IMPORT MODIFICATIONS ---
# Import the corrected GeminiProvider from llm_provider.py
from llm_provider import GeminiProvider
# Import ContextRelevanceAnalyzer for dependency injection
from src.context.context_analyzer import ContextRelevanceAnalyzer
# Import PersonaRouter for persona sequence determination
from src.persona.routing import PersonaRouter
# Import LLMOutputParser for parsing LLM responses
from src.utils.output_parser import LLMOutputParser
# Import models and exceptions
from src.models import PersonaConfig, ReasoningFrameworkConfig, LLMOutput, CodeChange, ContextAnalysisOutput, CritiqueOutput # Added CritiqueOutput
from src.config.settings import ChimeraSettings
from src.exceptions import ChimeraError, LLMResponseValidationError, SchemaValidationError, TokenBudgetExceededError, LLMProviderError # Corrected import, added LLMProviderError
from src.constants import SELF_ANALYSIS_KEYWORDS # Import for self-analysis persona sequence

# --- NEW IMPORT FOR LOGGING CONFIG ---
# This import is crucial for setting up structured logging.
from src.logging_config import setup_structured_logging
# --- END NEW IMPORT ---

# Configure logging for the core module itself
logger = logging.getLogger(__name__)

class SocraticDebate:
    PERSONA_OUTPUT_SCHEMAS = {
        "Impartial_Arbitrator": LLMOutput,
        "Context_Aware_Assistant": ContextAnalysisOutput,
        "Constructive_Critic": CritiqueOutput,
    }

    def __init__(self, initial_prompt: str, api_key: str,
                 codebase_context: Optional[Dict[str, str]] = None,
                 settings: Optional[ChimeraSettings] = None,
                 all_personas: Optional[Dict[str, PersonaConfig]] = None,
                 persona_sets: Optional[Dict[str, List[str]]] = None,
                 domain: Optional[str] = None,
                 max_total_tokens_budget: int = 10000,
                 model_name: str = "gemini-2.5-flash-lite",
                 status_callback: Optional[Callable] = None,
                 rich_console: Optional[Console] = None,
                 context_token_budget_ratio: float = 0.25,
                 context_analyzer: Optional[ContextRelevanceAnalyzer] = None, # ADDED THIS PARAMETER
                 is_self_analysis: bool = False
                 ):
        """
        Initialize a Socratic debate session.
        """
        # Initialize structured logging early in the constructor
        # This ensures all subsequent logs within this instance are structured.
        setup_structured_logging(log_level=logging.INFO)
        # Get a logger specific to this class instance for better log organization.
        self.logger = logging.getLogger(self.__class__.__name__)
        
        self.settings = settings or ChimeraSettings()
        self.context_token_budget_ratio = context_token_budget_ratio
        self.max_total_tokens_budget = max_total_tokens_budget
        self.tokens_used = 0 # Total tokens used across all phases
        self.model_name = model_name
        self.status_callback = status_callback
        # Use provided console or default to stderr for rich output
        self.rich_console = rich_console or Console(stderr=True) 
        self.is_self_analysis = is_self_analysis # Store the flag
        
        # Generate a unique request ID for tracing and logging context
        self.request_id = str(uuid.uuid4())[:8]  # Short, unique ID for request tracing
        # Store log extra data, including the request_id, for all logs from this instance
        self._log_extra = {"request_id": self.request_id}
        
        self.initial_prompt = initial_prompt
        self.codebase_context = codebase_context

        # Initialize the LLM provider. This might raise LLMProviderError if API key is invalid.
        try:
            self.llm_provider = GeminiProvider(api_key=api_key, model_name=self.model_name, rich_console=self.rich_console)
        except LLMProviderError as e:
            if self.rich_console:
                self.rich_console.print(f"[red]Failed to initialize LLM provider: {e}[/red]")
            else:
                self.logger.error(f"Failed to initialize LLM provider: {e}")
            raise ChimeraError(f"LLM provider initialization failed: {e}") from e
        except Exception as e: # Catch any other unexpected errors during initialization
            if self.rich_console:
                self.rich_console.print(f"[red]An unexpected error occurred during LLM provider initialization: {e}[/red]")
            else:
                self.logger.error(f"An unexpected error occurred during LLM provider initialization: {e}")
            raise ChimeraError(f"LLM provider initialization failed unexpectedly: {e}") from e

        # Ensure tokenizer is initialized only if client is successful
        try:
            self.tokenizer = self.llm_provider.tokenizer
        except AttributeError:
            raise ChimeraError("LLM provider tokenizer is not available.")

        # Store the context analyzer instance
        self.context_analyzer = context_analyzer # ADDED THIS LINE

        # Calculate token budgets for different phases
        self._calculate_token_budgets()

        self.all_personas = all_personas or {}
        self.persona_sets = persona_sets or {} # Store persona_sets
        self.domain = domain
        
        # Initialize PersonaRouter with all loaded personas AND persona_sets
        self.persona_router = PersonaRouter(self.all_personas, self.persona_sets)
        
        # If codebase_context was provided, compute embeddings now if context_analyzer is available.
        if self.codebase_context and self.context_analyzer:
            if isinstance(self.codebase_context, dict):
                try:
                    if not self.context_analyzer.file_embeddings:
                        self.context_analyzer.compute_file_embeddings(self.codebase_context)
                except Exception as e:
                    self.logger.error(f"Failed to compute embeddings for codebase context: {e}")
                    if self.status_callback:
                        self.status_callback(message=f"[red]Error computing context embeddings: {e}[/red]")
            else:
                self.logger.warning("codebase_context was not a dictionary, skipping embedding computation.")

    def _log_with_context(self, level: str, message: str, **kwargs):
        """Helper to add request context to all logs using the class-specific logger."""
        # --- FIX START: Extract exc_info from kwargs before passing to extra ---
        exc_info = kwargs.pop('exc_info', None)
        # --- FIX END ---

        log_data = {**self._log_extra, **kwargs}
        # Convert non-serializable objects to strings for logging to prevent errors
        for k, v in log_data.items():
            try:
                # Attempt to serialize to check if it's JSON-compatible
                json.dumps({k: v}) 
            except TypeError:
                # If not serializable, convert to string representation
                log_data[k] = str(v) 
        
        # Use the class-specific logger instance
        logger_method = getattr(self.logger, level)
        # --- FIX START: Pass exc_info as a separate parameter ---
        if exc_info is not None:
            logger_method(message, exc_info=exc_info, extra=log_data)
        else:
            logger_method(message, extra=log_data)
        # --- FIX END ---

    def _calculate_token_budgets(self):
        """Calculates token budgets for different phases based on context and model limits."""
        try:
            # Estimate tokens for context and initial input
            context_str = self.context_analyzer.get_context_summary() if self.context_analyzer else ""
            # Use initial_prompt for token estimation
            self.initial_input_tokens = self.tokenizer.estimate_tokens_for_context(context_str, self.initial_prompt)
            
            # Ensure remaining tokens never goes negative
            remaining_tokens = max(0, self.max_total_tokens_budget - self.initial_input_tokens)
            
            # Allocate remaining tokens to debate and synthesis phases based on ratios
            # These ratios can be adjusted based on expected workload distribution.
            debate_ratio = 0.9  # Allocate 90% of remaining tokens to the debate phase
            synthesis_ratio = 0.1  # Allocate 10% to the synthesis phase
            
            # Define a minimum token allocation to ensure phases can function even with tight budgets
            MIN_PHASE_TOKENS = 250
            
            # Calculate debate tokens, ensuring it meets the minimum if possible
            debate_tokens = int(remaining_tokens * debate_ratio)
            if debate_tokens < MIN_PHASE_TOKENS and remaining_tokens > 0:
                # If debate tokens are too low, allocate minimum and adjust synthesis
                debate_tokens = min(remaining_tokens - MIN_PHASE_TOKENS, MIN_PHASE_TOKENS)
                
            # Synthesis gets whatever's left, ensuring it also meets the minimum
            synthesis_tokens = max(MIN_PHASE_TOKENS, remaining_tokens - debate_tokens)
            
            self.phase_budgets = {
                "context": self.initial_input_tokens, # Tokens for initial context analysis
                "debate": debate_tokens,             # Tokens for persona debate turns
                "synthesis": synthesis_tokens        # Tokens for final synthesis
            }
            
            # Additional safety check for extremely constrained scenarios where budgets might be critically low
            if self.phase_budgets["debate"] < 100 or self.phase_budgets["synthesis"] < 100:
                self._log_with_context("error", "Token allocation critically constrained",
                                       phase_budgets=self.phase_budgets,
                                       context_tokens=self.initial_input_tokens,
                                       total_budget=self.max_total_tokens_budget)
                # Fall back to a minimum viable allocation if budgets are too low
                total_needed = 200  # Minimum for debate + synthesis
                if self.max_total_tokens_budget >= total_needed:
                    self.phase_budgets = {
                        "context": 0,
                        "debate": 100,
                        "synthesis": 100
                    }
                else:
                    # If total budget is less than minimum needed, allocate proportionally
                    self.phase_budgets = {
                        "context": 0,
                        "debate": int(self.max_total_tokens_budget * 0.5),
                        "synthesis": self.max_total_tokens_budget - int(self.max_total_tokens_budget * 0.5)
                    }
            
            self._log_with_context("info", "SocraticDebate token budgets initialized",
                                   initial_input_tokens=self.initial_input_tokens,
                                   context_budget=self.phase_budgets["context"],
                                   debate_budget=self.phase_budgets["debate"],
                                   synthesis_budget=self.phase_budgets["synthesis"],
                                   max_total_tokens_budget=self.max_total_tokens_budget)

        except Exception as e:
            # Log any errors during budget calculation and provide fallback budgets
            self._log_with_context("error", "Token budget calculation failed",
                                   error=str(e), context="token_budget")
            # Provide fallback budgets to prevent startup failure
            self.phase_budgets = {"context": 500, "debate": 15000, "synthesis": 1000}
            self.initial_input_tokens = 0
            raise ChimeraError("Failed to calculate token budgets due to an unexpected error.") from e
    
    def count_tokens(self, text: str) -> int:
        """Counts tokens in text using the provider's tokenizer, with an enhanced fallback and caching."""
        if not text:
            return 0
            
        # Use a hash of the text for cache key to avoid recomputing for identical inputs
        text_hash = hash(text)
        
        if text_hash in self._cache:
            self.logger.debug(f"Cache hit for token count (hash: {text_hash}).")
            return self._cache[text_hash]
        
        try:
            # Ensure text is properly encoded for the API call, handling potential errors
            try:
                text_encoded = text.encode('utf-8')
                text_for_api = text_encoded.decode('utf-8', errors='replace')
            except UnicodeEncodeError:
                # Fallback if encoding fails, replace problematic characters
                text_for_api = text.encode('utf-8', errors='replace').decode('utf-8', errors='replace')
                self.logger.warning("Fixed encoding issues in text for token counting by replacing problematic characters.")
            
            # Use the LLM provider's tokenizer to count tokens
            response = self.llm_provider.tokenizer.count_tokens(text_for_api)
            tokens = response # Assuming count_tokens returns the integer token count directly
            
            # Cache the result for future use
            self._cache[text_hash] = tokens
            self.logger.debug(f"Token count for text (hash: {text_hash}) is {tokens}. Stored in cache.")
            return tokens
            
        except Exception as e:
            # Log the error and provide a fallback approximation if the API call fails
            self.logger.error(f"Gemini token counting failed for model '{self.model_name}': {str(e)}")
            # IMPROVED FALLBACK: Use a more accurate approximation (e.g., 4 chars per token)
            approx_tokens = max(1, int(len(text) / 4))  # More accurate fallback than character count alone
            self.logger.warning(f"Falling back to improved token approximation ({approx_tokens}) due to error: {str(e)}")
            return approx_tokens

    def track_token_usage(self, phase: str, tokens: int):
        """Tracks token usage for a specific phase and updates total used tokens."""
        if phase in self.tokens_used_per_phase:
            self.tokens_used_per_phase[phase] += tokens
        else:
            # Log a warning if an unknown phase is encountered
            self.logger.warning(f"Attempted to track tokens for unknown phase: {phase}", method="track_token_usage")
        self.tokens_used += tokens # Always update the total tokens used

    def check_budget(self, phase: str, tokens_needed: int, step_name: str):
        """Checks if adding tokens_needed would exceed the budget for the given phase."""
        if phase not in self.phase_budgets:
            # Log a warning if the phase is not configured in the budget
            self.logger.warning(f"Phase '{phase}' not found in budget configuration.", method="check_budget")
            return # Cannot check budget if phase is not configured

        current_phase_usage = self.tokens_used_per_phase.get(phase, 0)
        phase_budget = self.phase_budgets.get(phase, 0)
        
        # Raise TokenBudgetExceededError if the budget would be exceeded
        if current_phase_usage + tokens_needed > phase_budget:
            raise TokenBudgetExceededError(
                current_tokens=current_phase_usage,
                budget=phase_budget,
                details={"phase": phase, "step": step_name, "tokens_requested": tokens_needed}
            )

    def get_total_used_tokens(self) -> int:
        """Returns the total tokens used across all phases and initial input."""
        return self.tokens_used

    def get_total_estimated_cost(self) -> float:
        """Calculates the total estimated cost based on token usage and LLM provider pricing."""
        try:
            # Calculate cost using the LLM provider's pricing method
            cost = self.llm_provider.calculate_usd_cost(
                input_tokens=self.initial_input_tokens,
                # Estimate output tokens as total used minus initial input tokens
                output_tokens=max(0, self.tokens_used - self.initial_input_tokens) 
            )
            return cost

        except Exception as e:
            # Log any errors during cost calculation
            self._log_with_context("error", "Could not estimate total cost", error=str(e), method="get_total_estimated_cost")
            return 0.0

    def run_debate(self) -> Tuple[Any, Dict[str, Any]]:
        """
        Executes the Socratic debate process by orchestrating persona interactions.
        Refactored into private methods for clarity and maintainability.
        """
        # Initialize final_answer to a default error state, in case of early failure
        final_answer = {
            "COMMIT_MESSAGE": "Debate Failed - Unhandled Error",
            "RATIONALE": "An unexpected error occurred before a final answer could be synthesized.",
            "CODE_CHANGES": [],
            "malformed_blocks": [{"type": "UNHANDLED_ERROR_INIT", "message": "Debate failed during initialization or early phase."}]
        }
        # Initialize intermediate_steps here to ensure it's always available for error reporting
        self.intermediate_steps = {
            "Total_Tokens_Used": 0,
            "Total_Estimated_Cost_USD": 0.0,
            "CODE_CHANGES": [],
            "malformed_blocks": [{"type": "UNHANDLED_ERROR_INIT", "message": "Debate failed during initialization or early phase."}]
        }
        
        try:
            # 1. Initialize state variables for the debate run
            self._initialize_debate_state()
            
            # Log the start of the debate process
            self._log_with_context("info", "Starting Socratic Debate...", state="running")
            if self.status_callback:
                self.status_callback(
                    message="Starting Socratic Debate...",
                    state="running",
                    current_total_tokens=self.get_total_used_tokens(),
                    current_total_cost=self.get_total_estimated_cost()
                )

            # 2. Perform Context Analysis and Determine Persona Sequence
            context_analysis_results = self._perform_context_analysis()
            persona_sequence = self._determine_persona_sequence(context_analysis_results)
            
            # Log and report the determined persona sequence
            self._log_with_context("info", f"Persona sequence determined: {persona_sequence}", state="running", progress_pct=0.2)
            if self.status_callback:
                self.status_callback(
                    message=f"Persona sequence determined: [bold]{', '.join(persona_sequence)}[/bold]",
                    state="running",
                    current_total_tokens=self.get_total_used_tokens(),
                    current_total_cost=self.get_total_estimated_cost(),
                    progress_pct=0.2 # Set initial progress after sequence determination
                )
            
            # 3. Process Context Persona Turn (if applicable)
            self._log_with_context("info", "Analyzing context...", state="running", progress_pct=0.25)
            context_persona_turn_results = self._process_context_persona_turn(persona_sequence, context_analysis_results)
            
            # 4. Execute Debate Persona Turns
            debate_persona_results = self._execute_debate_persona_turns(persona_sequence, context_persona_turn_results)
            
            # Log progress and report before proceeding to synthesis
            self._log_with_context("info", "All debate turns completed. Proceeding to synthesis.", state="running", progress_pct=0.7)
            if self.status_callback:
                self.status_callback(
                    message="All debate turns completed. Proceeding to synthesis.",
                    state="running",
                    current_total_tokens=self.get_total_used_tokens(),
                    current_total_cost=self.get_total_estimated_cost(),
                    progress_pct=0.7 # Set progress before synthesis phase
                )
            
            # 5. Perform Synthesis Persona Turn
            final_answer = self._perform_synthesis_persona_turn(persona_sequence, debate_persona_results)
            
            # 6. Finalize Results: Update intermediate_steps and ensure final_answer is a dictionary
            final_answer, intermediate_steps = self._finalize_debate_results(
                context_persona_turn_results, debate_persona_results, final_answer
            )
            
            # Ensure malformed_blocks field exists in the final answer
            if "malformed_blocks" not in final_answer:
                final_answer["malformed_blocks"] = []

            # Log and report the completion of the debate process
            self._log_with_context("info", "Socratic Debate process finalized.", state="complete", progress_pct=1.0)
            if self.status_callback:
                self.status_callback(
                    message="Socratic Debate process finalized.",
                    state="complete",
                    current_total_tokens=self.get_total_used_tokens(),
                    current_total_cost=self.get_total_estimated_cost(),
                    progress_pct=1.0 # Final progress update
                )

        # --- Exception Handling ---
        except TokenBudgetExceededError as e:
            # Log the specific error and update final answer/intermediate steps
            self._log_with_context("error", f"Socratic Debate failed: Token budget exceeded.", error=str(e), state="error")
            if not isinstance(final_answer, dict):
                final_answer = {
                    "COMMIT_MESSAGE": "Debate Failed - Token Budget Exceeded",
                    "RATIONALE": f"The Socratic debate exceeded the allocated token budget. Please consider increasing the budget or simplifying the prompt. Error details: {str(e)}",
                    "CODE_CHANGES": [],
                    "malformed_blocks": [{"type": "TOKEN_BUDGET_ERROR", "message": str(e), "details": e.details}]
                }
            elif "malformed_blocks" not in final_answer:
                final_answer["malformed_blocks"] = [{"type": "TOKEN_BUDGET_ERROR", "message": str(e), "details": e.details}]
            
            self._update_intermediate_steps_with_totals() # Ensure totals are updated even on error
            return final_answer, self.intermediate_steps
        
        except ChimeraError as e:
            # Handle Chimera-specific errors
            self._log_with_context("error", f"Socratic Debate failed due to ChimeraError: {e}", error=str(e), state="error")
            if not isinstance(final_answer, dict):
                final_answer = {
                    "COMMIT_MESSAGE": "Debate Failed (Chimera Error)",
                    "RATIONALE": f"A Chimera-specific error occurred during the debate: {str(e)}",
                    "CODE_CHANGES": [],
                    "malformed_blocks": [{"type": "CHIMERA_ERROR", "message": str(e), "details": e.details}]
                }
            elif "malformed_blocks" not in final_answer:
                final_answer["malformed_blocks"] = [{"type": "CHIMERA_ERROR", "message": str(e), "details": e.details}]
            
            self._update_intermediate_steps_with_totals()
            return final_answer, self.intermediate_steps

        except Exception as e:
            # Handle any other unexpected exceptions
            self._log_with_context("error", f"Socratic Debate failed due to an unexpected error: {e}", error=str(e), exc_info=True, state="error")
            if not isinstance(final_answer, dict):
                final_answer = {
                    "COMMIT_MESSAGE": "Debate Failed (Unexpected Error)",
                    "RATIONALE": f"An unexpected error occurred during the Socratic debate: {str(e)}",
                    "CODE_CHANGES": [],
                    "malformed_blocks": [{"type": "UNEXPECTED_ERROR", "message": str(e), "error_details": {"traceback": traceback.format_exc()}}]
                }
            elif "malformed_blocks" not in final_answer:
                final_answer["malformed_blocks"] = [{"type": "UNEXPECTED_ERROR", "message": str(e), "error_details": {"traceback": traceback.format_exc()}}]
            
            self._update_intermediate_steps_with_totals()
            return final_answer, self.intermediate_steps

        # Return the final answer and intermediate steps upon successful completion
        return final_answer, self.intermediate_steps

    def _initialize_debate_state(self):
        """Initializes state variables for a new debate run."""
        self.intermediate_steps = {} # Clear previous intermediate steps
        self.final_answer = None      # Reset final answer
        # Reset token usage per phase
        self.tokens_used_per_phase = {"context": 0, "debate": 0, "synthesis": 0}
        # Start total tokens used with the initial input tokens
        self.tokens_used = self.initial_input_tokens 
        self._log_with_context("debug", "Debate state initialized.")
        
        # Ensure intermediate_steps is a dictionary and update with initial totals
        if not isinstance(self.intermediate_steps, dict):
            self.intermediate_steps = {}
        self.intermediate_steps.update({
            "Total_Tokens_Used": self.tokens_used, 
            "Total_Estimated_Cost_USD": self.get_total_estimated_cost()
        })

    def _perform_context_analysis(self) -> Optional[Dict[str, Any]]:
        """Performs context analysis (finding relevant files) if context is available."""
        context_analysis_results = None
        # Proceed only if a context analyzer and codebase context are provided
        if self.context_analyzer and self.codebase_context:
            try:
                # Determine an initial persona sequence for relevance scoring
                initial_sequence_for_relevance = self.persona_router.determine_persona_sequence(
                    self.initial_prompt,
                    domain=self.domain,
                    intermediate_results=self.intermediate_steps
                )
                
                # Find relevant files based on the prompt and active personas
                relevant_files_info = self.context_analyzer.find_relevant_files(
                    self.initial_prompt,
                    active_personas=initial_sequence_for_relevance
                )
                context_analysis_results = {"relevant_files": relevant_files_info}
                # Store results in intermediate steps for later use
                self.intermediate_steps["Relevant_Files_Context"] = {"relevant_files": relevant_files_info}
                self._log_with_context("info", f"Context analysis completed. Found {len(relevant_files_info)} relevant files.")
                
                # Update status callback if provided
                if self.status_callback:
                    self.status_callback(
                        message=f"Context analysis complete. Found [bold]{len(relevant_files_info)}[/bold] relevant files.",
                        state="running",
                        current_total_tokens=self.get_total_used_tokens(),
                        current_total_cost=self.get_total_estimated_cost(),
                        progress_pct=0.2 # Set progress after context analysis
                    )
            except Exception as e:
                # Log errors during context analysis and update status
                self._log_with_context("error", "Error during context analysis file finding", error=str(e))
                self.intermediate_steps["Context_Analysis_Error"] = {"error": str(e)}
                if self.status_callback:
                    self.status_callback(message=f"[red]Error during context analysis: {e}[/red]", state="warning")
        else:
            # Log if context analysis is skipped due to missing components
            self._log_with_context("info", "No context analyzer or codebase context available. Skipping context analysis.")
        return context_analysis_results

    def _determine_persona_sequence(self, context_analysis_results: Optional[Dict[str, Any]]) -> List[str]:
        """Determines the persona sequence based on prompt, context, and self-analysis detection."""
        # Use the persona router to determine the sequence
        unique_sequence = self.persona_router.determine_persona_sequence(
            prompt=self.initial_prompt,
            domain=self.domain,
            intermediate_results=self.intermediate_steps,
            context_analysis_results=context_analysis_results
        )
        
        # Store the determined sequence in intermediate steps
        self.intermediate_steps["Persona_Sequence_Order"] = unique_sequence
        self._log_with_context("info", f"Final persona sequence determined: {unique_sequence}")
        return unique_sequence

    def _process_context_persona_turn(self, persona_sequence: List[str], context_analysis_results: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """Processes the turn for a dedicated context processing persona."""
        context_processing_persona_name = None
        context_processing_persona_config = None
        
        # Identify a context processing persona (e.g., Generalist_Assistant or Context_Aware_Assistant)
        for p_name in persona_sequence:
            if "Generalist_Assistant" in p_name or "Context_Aware_Assistant" in p_name:
                context_processing_persona_name = p_name
                break
        
        # Execute the turn if a suitable persona is found
        if context_processing_persona_name and context_processing_persona_name in self.all_personas:
            context_processing_persona_config = self.all_personas[context_processing_persona_name]
            
            # Construct the prompt for this persona, including relevant context
            context_prompt_for_persona = self.initial_prompt
            if context_analysis_results and context_analysis_results.get("relevant_files"):
                context_prompt_for_persona = f"Initial Prompt: {self.initial_prompt}\n\n"
                context_prompt_for_persona += "Relevant Code Files:\n"
                # Include top 5 relevant files for context
                for file_path, score in context_analysis_results["relevant_files"][:5]: 
                    context_prompt_for_persona += f"- {file_path} (Relevance: {score:.2f})\n"
                    if file_path in self.codebase_context:
                        # Include a snippet of the file content
                        context_prompt_for_persona += f"```\n{self.codebase_context[file_path][:500]}...\n```\n"
            
            try:
                self._log_with_context("info", f"Performing context processing with persona: {context_processing_persona_name}")
                if self.status_callback:
                    self.status_callback(
                        message=f"Running [bold]{context_processing_persona_name}[/bold] for context processing...",
                        state="running",
                        current_total_tokens=self.get_total_used_tokens(),
                        current_total_cost=self.get_total_estimated_cost(),
                        progress_pct=0.25,
                        current_persona_name=context_processing_persona_name
                    )
                # Execute the LLM turn for this persona
                turn_results = self._execute_llm_turn(
                    persona_name=context_processing_persona_name,
                    persona_config=context_processing_persona_config,
                    prompt=context_prompt_for_persona if context_prompt_for_persona is not None else self.initial_prompt,
                    phase="context"
                )
                # Update status callback upon completion
                if self.status_callback:
                    self.status_callback(
                        message=f"Completed persona: [bold]{context_processing_persona_name}[/bold] (context phase).",
                        state="running",
                        current_total_tokens=self.get_total_used_tokens(),
                        current_total_cost=self.get_total_estimated_cost(),
                        progress_pct=0.3
                    )
                return turn_results
                
            except TokenBudgetExceededError as e:
                self._log_with_context("error", f"Token budget exceeded during context processing", error=str(e))
                raise e # Re-raise to be caught by the main run_debate handler
            except Exception as e:
                # Log and handle errors during LLM turn execution
                self._log_with_context("error", f"Error during context processing generation for {context_processing_persona_name}", error=str(e))
                self.intermediate_steps[f"{context_processing_persona_name}_Error"] = str(e)
                # Estimate tokens for error logging and track usage
                error_tokens = self.tokenizer.count_tokens(f"Error processing {context_processing_persona_name}: {str(e)}") + 50
                self.track_token_usage("context", error_tokens)
                self.check_budget("context", 0, f"Error handling {context_processing_persona_name} context processing")
                if self.status_callback:
                    self.status_callback(
                        message=f"[red]Error with persona [bold]{context_processing_persona_name}[/bold]: {e}[/red]",
                        state="error",
                        current_total_tokens=self.get_total_used_tokens(),
                        current_total_cost=self.get_total_estimated_cost(),
                        progress_pct=0.3
                    )
                # FIX: Instead of returning None, re-raise the exception to be handled by the main run_debate loop
                raise ChimeraError(f"Context processing failed for {context_processing_persona_name}: {e}") from e

        else:
            # Log if no context processing persona was found or needed
            self._log_with_context("info", "No dedicated context processing persona found or no context available. Skipping dedicated context processing phase.")
        return None

    def _execute_debate_persona_turns(self, persona_sequence: List[str], context_persona_turn_results: Optional[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Executes the debate turns for the personas in the sequence."""
        all_debate_turns = []
        # Add results from context persona turn if it occurred
        if context_persona_turn_results:
            all_debate_turns.append(context_persona_turn_results)

        # Update status callback for the start of debate turns
        if self.status_callback:
            self.status_callback(
                message="Starting final [bold]debate turns[/bold] with core personas...",
                state="running",
                current_total_tokens=self.get_total_used_tokens(),
                current_total_cost=self.get_total_estimated_cost(),
                progress_pct=0.35
                )
        
        debate_personas_to_run = persona_sequence
        # Identify if the last persona is a synthesizer and exclude it from the debate loop
        synthesis_persona_name = None
        if persona_sequence and persona_sequence[-1] in ["Impartial_Arbitrator", "General_Synthesizer"]:
            synthesis_persona_name = persona_sequence[-1]
            debate_personas_to_run = persona_sequence[:-1] # Exclude the synthesizer from debate turns

        # Iterate through personas designated for debate turns
        for i, persona_name in enumerate(debate_personas_to_run):
            # Skip if persona is not found in the loaded configurations
            if persona_name not in self.all_personas:
                self._log_with_context("warning", f"Persona '{persona_name}' not found in loaded personas. Skipping.", persona_name=persona_name)
                if self.status_callback:
                    self.status_callback(
                        message=f"[yellow]Skipping persona '{persona_name}' (not found).[/yellow]",
                        state="running",
                        current_total_tokens=self.get_total_used_tokens(),
                        current_total_cost=self.get_total_estimated_cost(),
                        # Calculate progress percentage based on current step
                        progress_pct=0.35 + (0.3 / len(debate_personas_to_run) if debate_personas_to_run else 0),
                        current_persona_name=persona_name
                    )
                continue
            
            persona_config = self.all_personas[persona_name]
            
            # Determine the prompt for the current persona: use initial prompt or output from previous persona
            current_persona_prompt = self.initial_prompt
            if all_debate_turns:
                # Use the output of the last turn as input for the current turn
                current_persona_prompt = all_debate_turns[-1]["output"]
                if isinstance(current_persona_prompt, dict):
                    # If output is a dict, serialize it to JSON string for the LLM
                    current_persona_prompt = json.dumps(current_persona_prompt, indent=2)
                elif not isinstance(current_persona_prompt, str):
                    # Convert other types to string
                    current_persona_prompt = str(current_persona_prompt)

            try:
                self._log_with_context("info", f"Executing debate turn with persona: {persona_name}", persona_name=persona_name)
                if self.status_callback:
                    self.status_callback(
                        message=f"Running persona: [bold]{persona_name}[/bold]...",
                        state="running",
                        current_total_tokens=self.get_total_used_tokens(),
                        current_total_cost=self.get_total_estimated_cost(),
                        # Calculate progress percentage
                        progress_pct=0.35 + (0.3 * (i + 1) / len(debate_personas_to_run) if debate_personas_to_run else 0),
                        current_persona_name=persona_name
                    )
                # Execute the LLM turn for the current persona
                turn_results = self._execute_llm_turn(
                    persona_name=persona_name,
                    persona_config=persona_config,
                    prompt=current_persona_prompt,
                    phase="debate"
                )
                # Update status callback upon completion of the turn
                if self.status_callback:
                    self.status_callback(
                        message=f"Completed persona: [bold]{persona_name}[/bold].",
                        state="running",
                        current_total_tokens=self.get_total_used_tokens(),
                        current_total_cost=self.get_total_estimated_cost(),
                        progress_pct=0.35 + (0.3 * (i + 1) / len(debate_personas_to_run) if debate_personas_to_run else 0),
                        current_persona_name=None # Clear active persona
                    )
                all_debate_turns.append(turn_results)
            except TokenBudgetExceededError as e:
                self._log_with_context("error", f"Token budget exceeded during debate turn for persona {persona_name}", error=str(e))
                raise e # Re-raise to be caught by the main run_debate handler
            except Exception as e:
                # Handle errors during LLM turn execution
                self._log_with_context("error", f"An unexpected error occurred during debate turn for persona {persona_name}", error=str(e))
                self.intermediate_steps[f"{persona_name}_Error"] = str(e)
                # Estimate tokens for error logging and track usage
                error_tokens = self.tokenizer.count_tokens(f"Error processing {persona_name}: {str(e)}") + 50
                self.track_token_usage("debate", error_tokens)
                self.check_budget("debate", 0, f"Error handling {persona_name} debate turn")
                if self.status_callback:
                    self.status_callback(
                        message=f"[red]Error with persona [bold]{persona_name}[/bold]: {e}[/red]",
                        state="error",
                        current_total_tokens=self.get_total_used_tokens(),
                        current_total_cost=self.get_total_estimated_cost(),
                        progress_pct=0.35 + (0.3 * (i + 1) / len(debate_personas_to_run) if debate_personas_to_run else 0),
                        current_persona_name=persona_name
                    )
                # Do NOT append turn_results here, as it would be None or an error state.
                # The loop will continue to the next persona.
        
        return all_debate_turns

    def _perform_synthesis_persona_turn(self, persona_sequence: List[str], debate_persona_results: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """
        Performs the synthesis turn with retry logic for JSON validation.
        Returns the final structured answer (LLMOutput or general_output dict).
        """
        synthesis_persona_name = None
        synthesis_persona_config = None
        
        # Identify the synthesis persona (usually the last one in the sequence)
        if persona_sequence:
            potential_synthesis_persona = persona_sequence[-1]
            if potential_synthesis_persona in ["Impartial_Arbitrator", "General_Synthesizer"]:
                synthesis_persona_name = potential_synthesis_persona
                if synthesis_persona_name in self.all_personas:
                    synthesis_persona_config = self.all_personas[synthesis_persona_name]
        
        # Execute synthesis if a persona is found
        if synthesis_persona_name and synthesis_persona_config:
            
            self._log_with_context("info", f"Starting final [bold]synthesis[/bold] with persona: [bold]{synthesis_persona_name}[/bold]", state="running", progress_pct=0.7)
            if self.status_callback:
                self.status_callback(
                    message=f"Starting final [bold]synthesis[/bold] with persona: [bold]{synthesis_persona_name}[/bold]...",
                    state="running",
                    current_total_tokens=self.get_total_used_tokens(),
                    current_total_cost=self.get_total_estimated_cost(),
                    progress_pct=0.7
                )
            
            # Construct the synthesis prompt by combining initial prompt and debate turns
            synthesis_prompt = f"Initial Prompt: {self.initial_prompt}\n\n"
            synthesis_prompt += "Debate Turns:\n"
            for turn in debate_persona_results:
                synthesis_prompt += f"--- Persona: {turn['persona']} ---\n"
                output_str = turn['output']
                if isinstance(output_str, dict):
                    output_str = json.dumps(output_str, indent=2)
                elif not isinstance(output_str, str):
                    output_str = str(output_str)
                synthesis_prompt += f"{output_str}\n\n"
            
            max_retries = 2 # Number of retries for synthesis if validation fails
            requires_json_validation = (synthesis_persona_name == "Impartial_Arbitrator") # Only validate if it's the Impartial Arbitrator

            # Loop for retries if JSON validation is required and fails
            for attempt in range(max_retries + 1):
                try:
                    # Execute the LLM turn for the synthesis persona
                    turn_data_from_llm = self._execute_llm_turn(
                        persona_name=synthesis_persona_name,
                        persona_config=synthesis_persona_config,
                        prompt=synthesis_prompt,
                        phase="synthesis"
                    )
                    
                    current_synthesis_output = turn_data_from_llm.get('output')

                    # Perform JSON validation if required
                    if requires_json_validation:
                        is_failure = False
                        failure_reason = ""

                        # Check if output is a dictionary (expected for JSON)
                        if not isinstance(current_synthesis_output, dict):
                            is_failure = True
                            failure_reason = "Synthesis output is not a dictionary (expected JSON)."
                        else:
                            # Check for malformed blocks indicating JSON issues
                            malformed_blocks = current_synthesis_output.get("malformed_blocks", [])
                            if malformed_blocks:
                                # Check for specific types of JSON-related failures
                                if any(block.get("type") in ["JSON_EXTRACTION_FAILED", "JSON_DECODE_ERROR", "INVALID_JSON_STRUCTURE", "LLM_FAILED_JSON_ADHERENCE", "SYNTHESIS_EXECUTION_ERROR"] for block in malformed_blocks):
                                    is_failure = True
                                    failure_reason = "Output contains malformed blocks indicating JSON adherence failure."
                            # Check for fallback error messages from the LLM
                            elif current_synthesis_output.get("COMMIT_MESSAGE") == "LLM_GENERATION_ERROR" and \
                                 any(block.get("type") == "LLM_FAILED_JSON_ADHERENCE" for block in malformed_blocks):
                                is_failure = True
                                failure_reason = "LLM produced the fallback error JSON for JSON adherence."
                            # Check for missing essential keys
                            elif not ("COMMIT_MESSAGE" in current_synthesis_output and "RATIONALE" in current_synthesis_output):
                                is_failure = True
                                failure_reason = "Output dictionary is missing required keys (COMMIT_MESSAGE, RATIONALE)."

                        if not is_failure:
                            # If validation passes, log success and return the output
                            self._log_with_context("info", f"Synthesis output validated successfully on attempt {attempt + 1}")
                            return current_synthesis_output
                        else:
                            # If validation fails and retries are available, prepare for retry
                            if attempt < max_retries:
                                self._log_with_context("warning", f"Synthesis output validation failed on attempt {attempt + 1} ({failure_reason}). Retrying...", attempt=attempt+1)
                                # Construct a correction prompt to guide the LLM on the next attempt
                                correction_prompt_content = f"Previous output was invalid. Please re-generate the JSON output adhering strictly to the schema. The failure reason was: {failure_reason}. The previous output was:\n\n{json.dumps(current_synthesis_output, indent=2)}"
                                # Rebuild the synthesis prompt with the correction
                                synthesis_prompt = f"Initial Prompt: {self.initial_prompt}\n\n"
                                synthesis_prompt += "Debate Turns:\n"
                                for turn_data in debate_persona_results:
                                    synthesis_prompt += f"--- Persona: {turn_data['persona']} ---\n"
                                    output_str = turn_data['output']
                                    if isinstance(output_str, dict):
                                        output_str = json.dumps(output_str, indent=2)
                                    elif not isinstance(output_str, str):
                                        output_str = str(output_str)
                                    synthesis_prompt += f"{output_str}\n\n"
                                synthesis_prompt += f"\n\n{correction_prompt_content}"
                                # Store the failed attempt's output for debugging
                                self.intermediate_steps[f"{synthesis_persona_name}_Output_Attempt_{attempt+1}"] = current_synthesis_output
                                continue # Proceed to the next iteration (retry)
                            else:
                                # If max retries reached, log the final failure
                                self._log_with_context("error", f"Synthesis output validation failed after {max_retries} retries.")
                                return current_synthesis_output # Return the last (failed) output

                    else: # If JSON validation is not required
                        self._log_with_context("debug", f"Synthesis output for {synthesis_persona_name} does not require strict JSON validation. Returning raw output.")
                        if isinstance(current_synthesis_output, dict):
                            # Ensure standard fields exist for non-JSON outputs
                            if "general_output" not in current_synthesis_output:
                                current_synthesis_output["general_output"] = current_synthesis_output.get("raw_llm_output_snippet", "No specific general output found.")
                            if "malformed_blocks" not in current_synthesis_output:
                                current_synthesis_output["malformed_blocks"] = []
                            return current_synthesis_output
                        else:
                            # Wrap non-dict output in a dictionary for consistency
                            return {
                                "general_output": current_synthesis_output,
                                "malformed_blocks": []
                            }

                except Exception as e:
                    # Handle errors during LLM turn execution for synthesis
                    self._log_with_context("error", f"Error during synthesis turn execution", error=str(e), persona_name=synthesis_persona_name)
                    if attempt == max_retries:
                        # If max retries reached and an execution error occurred, return a final error state
                        self._log_with_context("error", f"Final synthesis attempt failed due to execution error", error=str(e))
                        return {
                            "COMMIT_MESSAGE": "Synthesis Execution Error",
                            "RATIONALE": f"An error occurred during the final synthesis turn: {str(e)}",
                            "CODE_CHANGES": [],
                            "malformed_blocks": [{"type": "SYNTHESIS_EXECUTION_ERROR", "message": str(e)}]
                        }
                    else:
                        # If retries are available, log the error and continue to the next attempt
                        self._log_with_context("warning", f"Execution error on synthesis attempt {attempt + 1}, retrying...", attempt=attempt+1)
                        self.intermediate_steps[f"{synthesis_persona_name}_Error_Attempt_{attempt+1}"] = str(e)
                        continue # Proceed to the next iteration (retry)
            
            # If loop finishes without returning, it means synthesis failed after all retries
            return {
                "COMMIT_MESSAGE": "Synthesis Failed",
                "RATIONALE": f"Failed to generate valid synthesis output after multiple attempts.",
                "CODE_CHANGES": [],
                "malformed_blocks": [{"type": "SYNTHESIS_FINAL_FAILURE", "message": "An unhandled failure occurred during synthesis."}]
            }

        else:
            # Log and return a fallback if no synthesis persona was identified
            self._log_with_context("warning", "No synthesis persona found or sequence is empty. Final answer may be incomplete.")
            return {
                "COMMIT_MESSAGE": "Synthesis Skipped",
                "RATIONALE": "No synthesis persona was identified in the sequence.",
                "CODE_CHANGES": [],
                "malformed_blocks": [{"type": "NO_SYNTHESIS_PERSONA", "message": "Synthesis persona not found in sequence."}]
            }

    def _execute_llm_turn(self, persona_name: str, persona_config: PersonaConfig, prompt: str, phase: str) -> Optional[Dict[str, Any]]:
        """Executes a single LLM turn for a given persona, handling generation, token tracking, and parsing."""
        
        # Estimate tokens needed for this turn (input + max output)
        # FIX: Combine system_prompt and prompt into a single string for token counting
        text_for_token_count = f"{persona_config.system_prompt}\n\n{prompt}" if persona_config.system_prompt else prompt
        estimated_next_step_input_tokens = self.tokenizer.count_tokens(text=text_for_token_count)
        estimated_next_step_output_tokens = persona_config.max_tokens
        estimated_next_step_total_tokens = estimated_next_step_input_tokens + estimated_next_step_output_tokens
        # Estimate cost for this turn
        estimated_next_step_cost = self.llm_provider.calculate_usd_cost(estimated_next_step_input_tokens, estimated_next_step_output_tokens)

        # Log the start of the LLM turn and update status callback
        self._log_with_context("info", f"Running persona: {persona_name} ({phase} phase)...",
                               persona_name=persona_name, phase=phase,
                               estimated_input_tokens=estimated_next_step_input_tokens,
                               estimated_output_tokens=estimated_next_step_output_tokens,
                               estimated_total_tokens=estimated_next_step_total_tokens,
                               estimated_cost=estimated_next_step_cost,
                               progress_pct=self.get_progress_pct(phase))
        if self.status_callback:
            self.status_callback(
                message=f"Running persona: [bold]{persona_name}[/bold] ({phase} phase)...",
                state="running",
                current_total_tokens=self.get_total_used_tokens(),
                current_total_cost=self.get_total_estimated_cost(),
                estimated_next_step_tokens=estimated_next_step_total_tokens,
                estimated_next_step_cost=estimated_next_step_cost,
                progress_pct=self.get_progress_pct(phase),
                current_persona_name=persona_name
            )

        # Check if the token budget for the current phase will be exceeded
        self.check_budget(phase, estimated_next_step_total_tokens, f"Start of {persona_name} turn")
        
        try:
            # Make the LLM API call
            response_text, input_tokens, output_tokens = self.llm_provider.generate(
                prompt=prompt,
                system_prompt=persona_config.system_prompt,
                temperature=persona_config.temperature,
                max_tokens=persona_config.max_tokens,
                requested_model_name=self.model_name,
                persona_config=persona_config,
                intermediate_results=self.intermediate_steps # FIX: Changed from self.intermediate_results
            )
            
            # Calculate total tokens used in this turn
            turn_tokens_used = input_tokens + output_tokens
            self.track_token_usage(phase, turn_tokens_used) # Track token usage
            self.check_budget(phase, 0, f"End of {persona_name} turn") # Re-check budget after usage
            
            # Log completion of the turn
            self._log_with_context("info", f"Completed persona: {persona_name} ({phase} phase).",
                                   persona_name=persona_name, phase=phase,
                                   tokens_used=turn_tokens_used, input_tokens=input_tokens, output_tokens=output_tokens,
                                   progress_pct=self.get_progress_pct(phase, completed=True))
            if self.status_callback:
                self.status_callback(
                    message=f"Completed persona: [bold]{persona_name}[/bold] ({phase} phase).",
                    state="running", 
                    current_total_tokens=self.get_total_used_tokens(), 
                    current_total_cost=self.get_total_estimated_cost(),
                    estimated_next_step_tokens=0, # Reset estimated next step tokens
                    estimated_next_step_cost=0.0,
                    progress_pct=self.get_progress_pct(phase, completed=True),
                    current_persona_name=None # Clear active persona
                )
        except TokenBudgetExceededError as e:
            self._log_with_context("error", f"Token budget exceeded during LLM generation for {persona_name}", error=str(e))
            raise e # Re-raise to be caught by the main run_debate handler
        except Exception as e:
            # Handle errors during LLM generation
            self._log_with_context("error", f"Error during LLM generation for {persona_name}", error=str(e), persona_name=persona_name)
            self.intermediate_steps[f"{persona_name}_Error"] = str(e)
            # Estimate tokens for error logging and track usage
            error_tokens = self.tokenizer.count_tokens(f"Error processing {persona_name}: {str(e)}") + 50
            self.track_token_usage(phase, error_tokens)
            self.check_budget(phase, 0, f"Error handling {persona_name} generation")
            
            if self.status_callback:
                self.status_callback(
                    message=f"[red]Error with persona [bold]{persona_name}[/bold]: {e}[/red]",
                    state="error",
                    current_total_tokens=self.get_total_used_tokens(),
                    current_total_cost=self.get_total_estimated_cost(),
                    progress_pct=self.get_progress_pct(phase, error=True),
                    current_persona_name=persona_name
                )
            # FIX: Instead of returning None, re-raise the exception to prevent None from being appended
            raise ChimeraError(f"LLM generation failed for {persona_name}: {e}") from e

        # Parse and validate the LLM's output
        parsed_output_data = {}
        expected_schema = self.PERSONA_OUTPUT_SCHEMAS.get(persona_name) # Get schema based on persona name

        if expected_schema:
            try:
                # Use the output parser to validate against the expected schema
                parsed_output_data = LLMOutputParser().parse_and_validate(response_text, expected_schema)
            except Exception as e:
                # Handle errors during parsing or validation
                self._log_with_context("error", f"Failed to parse/validate output for {persona_name} against {expected_schema.__name__} schema", error=str(e))
                malformed_blocks_for_fallback = [{"type": "PARSING_OR_VALIDATION_ERROR", "message": str(e), "raw_output": response_text[:500]}]
                
                # Create a fallback structure indicating parsing/validation failure
                parsed_output_data = {
                    "error_type": "Parsing/Validation Error",
                    "error_message": f"Failed to parse/validate output for {persona_name}. Error: {str(e)}",
                    "raw_llm_output_snippet": response_text[:500],
                    "malformed_blocks": malformed_blocks_for_fallback
                }
                self.intermediate_steps[f"{persona_name}_Error"] = str(e)
        else:
            # If no specific schema is defined for the persona, store the raw text output
            parsed_output_data = response_text
            self._log_with_context("debug", f"Persona {persona_name} does not have a specific JSON schema. Storing raw text output.", persona_name=persona_name)
        
        # Structure the results for this turn
        turn_results = {
            "persona": persona_name,
            "output": parsed_output_data,
            "tokens_used": turn_tokens_used,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens
        }
        # Store the parsed output and token usage in intermediate steps
        self.intermediate_steps[f"{persona_name}_Output"] = parsed_output_data
        self.intermediate_steps[f"{persona_name}_Tokens_Used"] = turn_tokens_used
        
        return turn_results

    def get_progress_pct(self, phase: str, completed: bool = False, error: bool = False) -> float:
        """Calculates a rough progress percentage based on phase and completion status."""
        # Define base progress contribution for each phase
        phase_progress = {
            "context": 0.2,  # 20% for context analysis
            "debate": 0.3,   # 30% for debate turns
            "synthesis": 0.3 # 30% for synthesis
        }
        
        base_progress = 0.0
        # Calculate base progress based on the current phase
        if phase == "context":
            base_progress = phase_progress["context"]
        elif phase == "debate":
            base_progress = phase_progress["context"] + phase_progress["debate"]
        elif phase == "synthesis":
            base_progress = phase_progress["context"] + phase_progress["debate"] + phase_progress["synthesis"]
        
        # Adjust progress based on completion or error status
        if error:
            return base_progress * 0.9 # Reduce progress slightly if there was an error
        elif completed:
            return base_progress * 1.0 # Full progress for the phase if completed
        else:
            return base_progress # Return base progress if still running

    def _finalize_debate_results(self, context_persona_turn_results: Optional[Dict[str, Any]], 
                                 debate_persona_results: List[Dict[str, Any]], 
                                 synthesis_persona_results: Optional[Dict[str, Any]]) -> Tuple[Any, Dict[str, Any]]:
        
        # Determine the final answer based on the results of each stage
        if synthesis_persona_results: 
            self.final_answer = synthesis_persona_results
        elif debate_persona_results: # If synthesis was skipped, use the last debate turn's output
            raw_output = debate_persona_results[-1]["output"]
            if isinstance(raw_output, dict):
                self.final_answer = raw_output
            else:
                # Wrap non-dict output in a standard format
                self.final_answer = {
                    "COMMIT_MESSAGE": "Debate Fallback - No Synthesis",
                    "RATIONALE": "Synthesis step was skipped or failed. Falling back to last debate persona's output.",
                    "CODE_CHANGES": [],
                    "general_output": str(raw_output),
                    "malformed_blocks": []
                }
        elif context_persona_turn_results: # If debate and synthesis were skipped, use context turn's output
            raw_output = context_persona_turn_results["output"]
            if isinstance(raw_output, dict):
                self.final_answer = raw_output
            else:
                # Wrap non-dict output in a standard format
                self.final_answer = {
                    "COMMIT_MESSAGE": "Debate Fallback - No Debate or Synthesis",
                    "RATIONALE": "Debate and synthesis steps were skipped or failed. Falling back to context persona's output.",
                    "CODE_CHANGES": [],
                    "general_output": str(raw_output),
                    "malformed_blocks": []
                }
        else: # If no turns were executed at all
            self.final_answer = {
                "COMMIT_MESSAGE": "Debate Failed - No Turns Executed",
                "RATIONALE": "The Socratic debate process could not execute any turns or perform synthesis.",
                "CODE_CHANGES": [],
                "malformed_blocks": [{"type": "NO_DEBATE_TURNS", "message": "No debate turns were executed."}]
            }
            self._log_with_context("error", "Socratic debate process completed without executing any turns.")

        # Ensure the final answer is always a dictionary
        if not isinstance(self.final_answer, dict):
            self._log_with_context("error", f"Final answer was not a dictionary: {type(self.final_answer).__name__}. Creating fallback error.", final_answer_type=type(self.final_answer).__name__)
            self.final_answer = {
                "COMMIT_MESSAGE": "Debate Failed - Final Answer Malformed",
                "RATIONALE": f"The final answer was not a dictionary. Type: {type(self.final_answer).__name__}",
                "CODE_CHANGES": [],
                "general_output": str(self.final_answer),
                "malformed_blocks": [{"type": "FINAL_ANSWER_MALFORMED", "message": f"Final answer was not a dictionary. Type: {type(self.final_answer).__name__}", "raw_output": str(self.final_answer)[:500]}]
            }

        # Ensure the 'malformed_blocks' field is present, even if empty
        if "malformed_blocks" not in self.final_answer:
            self.final_answer["malformed_blocks"] = []
        
        # Update intermediate steps with final totals
        self._update_intermediate_steps_with_totals()
        
        return self.final_answer, self.intermediate_steps

    def _update_intermediate_steps_with_totals(self):
        """Updates intermediate steps with total token counts and estimated cost."""
        # Ensure total tokens used is accurate
        self.tokens_used += sum(self.tokens_used_per_phase.values())
        
        # Update the totals in the intermediate steps dictionary
        self.intermediate_steps["Total_Tokens_Used"] = self.tokens_used
        self.intermediate_steps["Total_Estimated_Cost_USD"] = self.get_total_estimated_cost()
        
        # Add breakdown of tokens used per phase
        self.intermediate_steps["Initial_Prompt_Tokens"] = self.initial_input_tokens
        self.intermediate_steps["Context_Phase_Tokens"] = self.tokens_used_per_phase.get("context", 0)
        self.intermediate_steps["Debate_Phase_Tokens"] = self.tokens_used_per_phase.get("debate", 0)
        self.intermediate_steps["Synthesis_Phase_Tokens"] = self.tokens_used_per_phase.get("synthesis", 0)

    # The system_prompt for Impartial_Arbitrator was corrected in the original code
    # to escape backticks, resolving a SyntaxError. This is reflected in the provided code.
    # This is a class-level attribute, not used directly in __init__ but referenced by persona config.
    system_prompt = r"""
    You are the final arbiter in this Socratic debate. Your task is to synthesize all previous critiques and proposals into a coherent, actionable plan.
    
    **CRITICAL RULES:**
    1.  **YOUR ENTIRE RESPONSE MUST BE A SINGLE, VALID JSON OBJECT.** No other text, markdown, conversational filler, or explanations are allowed outside the JSON.
    2.  **STRICTLY ADHERE TO THE PROVIDED JSON SCHEMA.**
    3.  **IF YOU CANNOT PRODUCE VALID JSON**, output a JSON object with a specific error structure:
        ```json
        {
          "COMMIT_MESSAGE": "LLM_GENERATION_ERROR",
          "RATIONALE": "Error: Could not generate valid JSON output. Please check prompt adherence and LLM capabilities.",
          "CODE_CHANGES": [],
          "malformed_blocks": [{"type": "LLM_FAILED_JSON_ADHERENCE", "message": "LLM ignored JSON output instruction."}]
        }
        ```
    4.  Ensure all code snippets within `CODE_CHANGES` adhere to the specified structure (`FILE_PATH`, `ACTION`, `FULL_CONTENT`, `LINES`) and PEP8 (line length <= 88).
    5.  Include the `malformed_blocks` field in your JSON output, even if it's an empty list `[]`.

    **JSON Schema:**
    ```json
    {
      "COMMIT_MESSAGE": "<string>",
      "RATIONALE": "<string>",
      "CODE_CHANGES": [
        {
          "FILE_PATH": "<string>",
          "ACTION": "ADD | MODIFY | REMOVE",
          "FULL_CONTENT": "<string>" (Required for ADD/MODIFY actions)
        },
        {
          "FILE_PATH": "<string>",
          "ACTION": "REMOVE",
          "LINES": ["<string>", "<string>"] (Required for REMOVE action)
        }
      ],
      "CONFLICT_RESOLUTION": "<string>" (Optional),
      "UNRESOLVED_CONFLICT": "<string>" (Optional),
      "malformed_blocks": []
    }
    ```
    **Synthesize the following feedback into the specified JSON format:**
    [Insert debate results here]
    """
    temperature: 0.1
    max_tokens: 4096
    description: "Synthesizes debate outcomes into a final structured solution, strictly adhering to JSON format."