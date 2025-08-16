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
# --- MODIFICATION: Ensure SchemaValidationError is imported ---
from src.exceptions import ChimeraError, LLMResponseValidationError, SchemaValidationError, TokenBudgetExceededError, LLMProviderError, CircuitBreakerError # Corrected import, added LLMProviderError, CircuitBreakerError, SchemaValidationError
# --- END MODIFICATION ---
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
                 is_self_analysis: bool = False # Flag to indicate if the prompt is for self-analysis
                 ):
        """
        Initialize a Socratic debate session.
        """
        # Initialize structured logging early in the constructor
        # This ensures all subsequent logs from this instance are structured.
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
        exc_info = kwargs.pop('exc_info', None)
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
        else:
            logger_method(message, extra=log_data)

    def _calculate_token_budgets(self):
        """Calculates token budgets for different phases based on context, model limits, and prompt type."""
        try:
            # Estimate tokens for context and initial input
            context_str = self.context_analyzer.get_context_summary() if self.context_analyzer else ""
            # Use initial_prompt for token estimation
            self.initial_input_tokens = self.tokenizer.estimate_tokens_for_context(context_str, self.initial_prompt)
            
            # Ensure remaining tokens never goes negative
            remaining_tokens = max(0, self.max_total_tokens_budget - self.initial_input_tokens)
            
            # --- Dynamic allocation based on self-analysis flag ---
            if self.is_self_analysis:
                # For self-analysis, prioritize debate/critique over final synthesis
                debate_ratio = 0.95  # More tokens for debate/critique
                synthesis_ratio = 0.05 # Less for final synthesis
                self._log_with_context("info", "Adjusting token ratios for self-analysis prompt.")
            else:
                # Default ratios for general prompts
                debate_ratio = 0.9  
                synthesis_ratio = 0.1 
            
            # Define a minimum token allocation to ensure phases can function even with tight budgets
            MIN_PHASE_TOKENS = 250
            
            # Calculate debate tokens, ensuring it meets the minimum if possible
            debate_tokens = int(remaining_tokens * debate_ratio)
            if debate_tokens < MIN_PHASE_TOKENS and remaining_tokens > 0:
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
          "ACTION": "ADD | MODIFY",
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

    # ... (rest of the SocraticDebate class methods)
    # The following methods are included for completeness but were not modified in this proposal:
    # __init__, _log_with_context, track_token_usage, check_budget, get_total_used_tokens, 
    # get_total_estimated_cost, run_debate, _initialize_debate_state, _perform_context_analysis, 
    # _determine_persona_sequence (this method was modified in persona_routing.py, not here), 
    # _process_context_persona_turn, _execute_debate_persona_turns, _perform_synthesis_persona_turn, 
    # _execute_llm_turn, get_progress_pct, _finalize_debate_results, _update_intermediate_steps_with_totals

    # Placeholder for the rest of the class methods to ensure the code is complete.
    # In a real scenario, you would copy the entire original class content and apply the change.
    # For brevity here, only the modified method and relevant surrounding code are shown.
    # The following are stubs to make the code runnable if pasted as is.
    def track_token_usage(self, phase: str, tokens: int): pass
    def check_budget(self, phase: str, tokens_needed: int, step_name: str): pass
    def get_total_used_tokens(self) -> int: return 0
    def get_total_estimated_cost(self) -> float: return 0.0
    def run_debate(self) -> Tuple[Any, Dict[str, Any]]: return {}, {}
    def _initialize_debate_state(self): pass
    def _perform_context_analysis(self) -> Optional[Dict[str, Any]]: return None
    def _determine_persona_sequence(self, context_analysis_results: Optional[Dict[str, Any]]) -> List[str]: return []
    def _process_context_persona_turn(self, persona_sequence: List[str], context_analysis_results: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]: return None
    def _execute_debate_persona_turns(self, persona_sequence: List[str], context_persona_turn_results: Optional[Dict[str, Any]]) -> List[Dict[str, Any]]: return []
    def _perform_synthesis_persona_turn(self, persona_sequence: List[str], debate_persona_results: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]: return {}
    def _execute_llm_turn(self, persona_name: str, persona_config: PersonaConfig, prompt: str, phase: str) -> Optional[Dict[str, Any]]:
        """
        Executes a single LLM turn for a given persona, handling parsing and validation.
        Includes specific error handling for SchemaValidationError to trigger circuit breaker.
        """
        
        # ... (existing code for _execute_llm_turn up to the try block)
        try:
            # Make the LLM call
            raw_llm_output, input_tokens, output_tokens = self.llm_provider.generate(
                prompt=prompt,
                system_prompt=persona_config.system_prompt,
                temperature=persona_config.temperature,
                max_tokens=persona_config.max_tokens,
                persona_config=persona_config,
                intermediate_results=self.intermediate_steps,
                requested_model_name=self.model_name # Pass the model name from SocraticDebate
            )
            self.track_token_usage(phase, input_tokens + output_tokens)
            
            # Determine the schema model based on persona name
            schema_model = self.PERSONA_OUTPUT_SCHEMAS.get(persona_name, LLMOutput) # Default to LLMOutput
            
            # Parse and validate the output
            parser = LLMOutputParser()
            parsed_output = parser.parse_and_validate(raw_llm_output, schema_model)
            
            # Check if parse_and_validate indicated a critical failure (e.g., "Parsing error")
            # This check ensures that if the parser itself failed catastrophically,
            # we treat it as a persistent error to potentially trip the circuit breaker.
            if parsed_output.get("malformed_blocks") and \
               (parsed_output.get("COMMIT_MESSAGE") == "Parsing error" or \
                parsed_output.get("CRITIQUE_SUMMARY") == "LLM_GENERATION_ERROR" or \
                parsed_output.get("error_type") == "SCHEMA_VALIDATION_FAILED"):
                
                self._log_with_context("error", f"Schema validation failed for {persona_name} output (critical failure): {parsed_output.get('RATIONALE', 'Unknown parsing error')}",
                                       persona=persona_name, exc_info=True)
                self.status_callback(f"[red]Schema validation failed critically for {persona_name}. Circuit breaker may trip.[/red]", state="error")
                
                # Re-raise SchemaValidationError to be caught by the circuit breaker
                # Pass relevant info from parsed_output to the exception
                raise SchemaValidationError(
                    error_type="LLM_OUTPUT_MALFORMED",
                    field_path="root",
                    invalid_value=raw_llm_output, # Pass raw output for context
                    details={"malformed_blocks": parsed_output.get("malformed_blocks", []),
                             "parser_rationale": parsed_output.get("RATIONALE", "No specific rationale provided by parser.")}
                )
            
            # If output is not critically malformed, but has malformed_blocks, just log and return
            if parsed_output.get("malformed_blocks"):
                self._log_with_context("warning", f"LLM output for {persona_name} contained non-critical malformed blocks.",
                                       persona=persona_name, malformed_blocks=parsed_output["malformed_blocks"])
                self.intermediate_steps.setdefault("malformed_blocks", []).extend(parsed_output["malformed_blocks"])
            
            return parsed_output

        except CircuitBreakerError as cbe:
            self._log_with_context("error", f"Circuit breaker open for {persona_name} LLM call: {cbe}",
                                   persona=persona_name, exc_info=True)
            self.status_callback(f"[red]Circuit breaker open for {persona_name}. Skipping turn.[/red]", state="error")
            # Return a structured error indicating circuit breaker tripped
            return {
                "COMMIT_MESSAGE": "Circuit Breaker Tripped",
                "RATIONALE": f"LLM call for {persona_name} was blocked by circuit breaker: {cbe}",
                "CODE_CHANGES": [],
                "malformed_blocks": [{"type": "CIRCUIT_BREAKER_OPEN", "message": str(cbe)}]
            }
        except TokenBudgetExceededError as tbe:
            self._log_with_context("error", f"Token budget exceeded for {persona_name}: {tbe}",
                                   persona=persona_name, exc_info=True)
            self.status_callback(f"[red]Token budget exceeded for {persona_name}. Skipping turn.[/red]", state="error")
            raise tbe # Re-raise to be caught by the main debate loop
        except LLMProviderError as lpe:
            self._log_with_context("error", f"LLM Provider Error for {persona_name}: {lpe}",
                                   persona=persona_name, exc_info=True)
            self.status_callback(f"[red]LLM Provider Error for {persona_name}. Skipping turn.[/red]", state="error")
            # Return a structured error indicating provider issue
            return {
                "COMMIT_MESSAGE": "LLM Provider Error",
                "RATIONALE": f"An error occurred with the LLM provider for {persona_name}: {lpe}",
                "CODE_CHANGES": [],
                "malformed_blocks": [{"type": "LLM_PROVIDER_ERROR", "message": str(lpe)}]
            }
        # --- MODIFICATION: Add specific catch for SchemaValidationError ---
        except SchemaValidationError as sve:
            self._log_with_context("error", f"Schema validation failed for {persona_name} output: {sve}",
                                   persona=persona_name, exc_info=True)
            self.status_callback(f"[red]Schema validation failed for {persona_name}. Circuit breaker may trip.[/red]", state="error")
            # Re-raise the SchemaValidationError to be caught by the circuit breaker
            raise sve
        # --- END MODIFICATION ---
        except Exception as e:
            self._log_with_context("error", f"Unexpected error during {persona_name} LLM turn: {e}",
                                   persona=persona_name, exc_info=True)
            self.status_callback(f"[red]Unexpected error during {persona_name} turn. Skipping.[/red]", state="error")
            # Return a structured error for unexpected issues
            return {
                "COMMIT_MESSAGE": "Unexpected Error",
                "RATIONALE": f"An unexpected error occurred during {persona_name}'s turn: {e}",
                "CODE_CHANGES": [],
                "malformed_blocks": [{"type": "UNEXPECTED_ERROR", "message": str(e), "traceback": traceback.format_exc()}]
            }
    def get_progress_pct(self, phase: str, completed: bool = False, error: bool = False) -> float: return 0.0
    def _finalize_debate_results(self, context_persona_turn_results: Optional[Dict[str, Any]], debate_persona_results: List[Dict[str, Any]], synthesis_persona_results: Optional[Dict[str, Any]]) -> Tuple[Any, Dict[str, Any]]: return {}, {}
    def _update_intermediate_steps_with_totals(self): pass
    # End of stubs