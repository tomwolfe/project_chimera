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
# --- MODIFICATION: Ensure GeneralOutput is imported ---
from src.models import PersonaConfig, ReasoningFrameworkConfig, LLMOutput, CodeChange, ContextAnalysisOutput, CritiqueOutput, GeneralOutput # Added CritiqueOutput, GeneralOutput
# --- END MODIFICATION ---
from src.config.settings import ChimeraSettings # Corrected import path
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
    # --- MODIFICATION: Add GeneralOutput to PERSONA_OUTPUT_SCHEMAS ---
    PERSONA_OUTPUT_SCHEMAS = {
        "Impartial_Arbitrator": LLMOutput,
        "Context_Aware_Assistant": ContextAnalysisOutput,
        "Constructive_Critic": CritiqueOutput,
        "General_Synthesizer": GeneralOutput, # ADD THIS LINE
    }
    # --- END MODIFICATION ---

    def __init__(self, initial_prompt: str, api_key: str,
                 codebase_context: Optional[Dict[str, str]] = None,
                 settings: Optional[ChimeraSettings] = None, # ADDED THIS PARAMETER
                 all_personas: Optional[Dict[str, PersonaConfig]] = None,
                 persona_sets: Optional[Dict[str, List[str]]] = None,
                 domain: Optional[str] = None,
                 max_total_tokens_budget: int = 10000,
                 model_name: str = "gemini-2.5-flash-lite",
                 status_callback: Optional[Callable] = None,
                 rich_console: Optional[Console] = None,
                 # REMOVED: context_token_budget_ratio: float = 0.25, # This is now managed by ChimeraSettings
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
        
        self.settings = settings or ChimeraSettings() # Use provided settings or default
        # REMOVED: self.context_token_budget_ratio = context_token_budget_ratio # Now managed by self.settings
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
            
            # --- Dynamic allocation based on self-analysis flag and ChimeraSettings ---
            if self.is_self_analysis:
                debate_ratio = self.settings.self_analysis_debate_ratio
                # Synthesis ratio is derived to ensure sum is 1.0
                synthesis_ratio = 1.0 - debate_ratio 
                self._log_with_context("info", "Adjusting token ratios for self-analysis prompt using ChimeraSettings.",
                                       self_analysis_debate_ratio=debate_ratio, synthesis_ratio=synthesis_ratio)
            else:
                debate_ratio = self.settings.debate_token_budget_ratio
                # Synthesis ratio is derived to ensure sum is 1.0
                synthesis_ratio = 1.0 - debate_ratio 
                self._log_with_context("info", "Using default token ratios from ChimeraSettings.",
                                       debate_ratio=debate_ratio, synthesis_ratio=synthesis_ratio)
            
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

    def track_token_usage(self, phase: str, tokens: int):
        """Tracks token usage for a given phase."""
        self.tokens_used += tokens
        cost = self.llm_provider.calculate_usd_cost(tokens, 0) # Assuming input tokens for tracking cost
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
        # Define approximate weights for each phase
        phase_weights = {
            "context": 0.1,
            "debate": 0.7,
            "synthesis": 0.2
        }
        
        current_progress = 0.0
        if phase == "context":
            current_progress = 0.05 # Initial context analysis
        elif phase == "debate":
            # Estimate progress within debate phase
            total_debate_personas = len(self.intermediate_steps.get("Persona_Sequence", [])) - 1 # Exclude synthesis persona
            completed_debate_personas = sum(1 for k in self.intermediate_steps if k.endswith("_Output") and k != "Final_Synthesis_Output")
            
            if total_debate_personas > 0:
                current_progress = phase_weights["context"] + (completed_debate_personas / total_debate_personas) * phase_weights["debate"]
            else:
                current_progress = phase_weights["context"] # If no debate personas, just context phase
        elif phase == "synthesis":
            current_progress = phase_weights["context"] + phase_weights["debate"] + 0.1 # Start of synthesis
        
        if completed:
            current_progress = 1.0
        elif error:
            current_progress = max(current_progress, 0.99) # Indicate near completion but with error

        return min(max(0.0, current_progress), 1.0) # Ensure between 0 and 1

    def _initialize_debate_state(self):
        """Initializes or resets the debate's internal state variables."""
        self.intermediate_steps = {}
        self.tokens_used = 0
        self.rich_console.print(f"[bold green]Starting Socratic Debate for prompt:[/bold green] [italic]{self.initial_prompt}[/italic]")
        self._log_with_context("info", "Debate state initialized.")

    def _perform_context_analysis(self) -> Optional[Dict[str, Any]]:
        """
        Performs context analysis based on the initial prompt and codebase context.
        """
        if not self.codebase_context or not self.context_analyzer:
            self._log_with_context("info", "No codebase context or analyzer available. Skipping context analysis.")
            return None

        self._log_with_context("info", "Performing context analysis.")
        try:
            # Find relevant files based on the initial prompt
            # Pass the current persona sequence to help context analyzer prioritize
            current_persona_names = self.persona_router.determine_persona_sequence(
                self.initial_prompt, self.domain, intermediate_results=self.intermediate_steps
            )
            relevant_files = self.context_analyzer.find_relevant_files(
                self.initial_prompt, top_k=5, active_personas=current_persona_names
            )
            
            context_summary_str = self.context_analyzer.get_context_summary()
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
            self._log_with_context("error", f"Error during context analysis: {e}", exc_info=True)
            self.rich_console.print(f"[red]Error during context analysis: {e}[/red]")
            return {"error": f"Context analysis failed: {e}"}

    def _determine_persona_sequence(self, prompt: str, domain: str, intermediate_results: Dict[str, Any], context_analysis_results: Optional[Dict[str, Any]]) -> List[str]:
        """
        Determines the optimal sequence of personas for processing the prompt.
        """
        self._log_with_context("info", "Determining persona sequence.", prompt=prompt, domain=domain)
        try:
            sequence = self.persona_router.determine_persona_sequence(
                prompt, domain, intermediate_results, context_analysis_results
            )
            self._log_with_context("info", f"Persona sequence determined: {sequence}", sequence=sequence)
            return sequence
        except Exception as e:
            self._log_with_context("error", f"Error determining persona sequence: {e}", exc_info=True)
            self.rich_console.print(f"[red]Error determining persona sequence: {e}[/red]")
            # Fallback to a minimal sequence if routing fails
            return ["Visionary_Generator", "Skeptical_Generator", "Impartial_Arbitrator"]

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
        
        # Estimate tokens for this turn
        estimated_tokens = self.tokenizer.count_tokens(prompt) + persona_config.max_tokens
        self.check_budget("debate", estimated_tokens, "Context_Aware_Assistant")

        try:
            output = self._execute_llm_turn("Context_Aware_Assistant", persona_config, prompt, "debate")
            self._log_with_context("info", "Context_Aware_Assistant turn completed.")
            return output
        except Exception as e:
            self._log_with_context("error", f"Error during Context_Aware_Assistant turn: {e}", exc_info=True)
            self.rich_console.print(f"[red]Error during Context_Aware_Assistant turn: {e}[/red]")
            return {"error": f"Context_Aware_Assistant turn failed: {e}"}

    def _execute_debate_persona_turns(self, persona_sequence: List[str], context_persona_turn_results: Optional[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Executes the main debate turns for each persona in the sequence.
        """
        debate_history = []
        previous_output = self.initial_prompt # Start with the user's initial prompt
        
        # Include context analysis results in the initial debate context if available
        if context_persona_turn_results:
            previous_output = f"Initial Prompt: {self.initial_prompt}\n\nContext Analysis:\n{json.dumps(context_persona_turn_results, indent=2)}"

        # Filter out Context_Aware_Assistant and the final synthesis persona (Impartial_Arbitrator or General_Synthesizer)
        # and Devils_Advocate (as it typically critiques the final proposal)
        personas_for_debate = [
            p for p in persona_sequence
            if p not in ["Context_Aware_Assistant", "Impartial_Arbitrator", "General_Synthesizer", "Devils_Advocate"]
        ]
        
        # Add Devils_Advocate before the final synthesizer if it's in the sequence
        if "Devils_Advocate" in persona_sequence and "Devils_Advocate" not in personas_for_debate:
            # Insert it before the last "critic" type persona or before the end
            insert_idx = len(personas_for_debate)
            for i, p_name in reversed(list(enumerate(personas_for_debate))):
                if "Critic" in p_name or "Analyst" in p_name: # Heuristic for critic-like personas
                    insert_idx = i + 1
                    break
            personas_for_debate.insert(insert_idx, "Devils_Advocate")

        total_debate_steps = len(personas_for_debate)
        for i, persona_name in enumerate(personas_for_debate):
            self._log_with_context("info", f"Executing debate turn for persona: {persona_name}", persona=persona_name)
            self.status_callback(
                f"Executing: [bold]{persona_name.replace('_', ' ')}[/bold]...",
                "running",
                self.tokens_used,
                self.get_total_estimated_cost(),
                progress_pct=self.get_progress_pct("debate"), # REMOVED current_persona_name from this call
                current_persona_name=persona_name # ADDED current_persona_name directly to status_callback
            )

            persona_config = self.all_personas.get(persona_name)
            if not persona_config:
                self._log_with_context("error", f"Persona configuration not found for {persona_name}. Skipping turn.", persona=persona_name)
                debate_history.append({"persona": persona_name, "error": "Config not found"})
                continue

            # Construct the prompt for the current persona
            # The prompt includes the initial problem and the output from the previous persona
            current_prompt = f"Initial Problem: {self.initial_prompt}\n\nPrevious Debate Output:\n{json.dumps(previous_output, indent=2) if isinstance(previous_output, dict) else previous_output}"
            
            # Estimate tokens for this turn
            estimated_tokens = self.tokenizer.count_tokens(current_prompt) + persona_config.max_tokens
            self.check_budget("debate", estimated_tokens, persona_name)

            try:
                output = self._execute_llm_turn(persona_name, persona_config, current_prompt, "debate")
                debate_history.append({"persona": persona_name, "output": output})
                previous_output = output # Update previous output for the next turn
            except Exception as e:
                self._log_with_context("error", f"Error during {persona_name} turn: {e}", persona=persona_name, exc_info=True)
                self.rich_console.print(f"[red]Error during {persona_name} turn: {e}[/red]")
                debate_history.append({"persona": persona_name, "error": str(e)})
                # Decide whether to stop or continue. For now, we continue but log the error.
                # If the error is critical (e.g., TokenBudgetExceeded), it will be re-raised by _execute_llm_turn.
                # If it's a recoverable error (e.g., malformed JSON that was salvaged), output will be a dict with error info.
                # We should ensure 'previous_output' is a dict for subsequent JSON parsing.
                previous_output = {"error": f"Turn failed for {persona_name}: {str(e)}", "malformed_blocks": [{"type": "DEBATE_TURN_ERROR", "message": str(e)}]}
                continue # Continue to the next persona

        self._log_with_context("info", "All debate turns completed.")
        return debate_history

    def _perform_synthesis_persona_turn(self, persona_sequence: List[str], debate_persona_results: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """
        Executes the final synthesis persona turn (Impartial_Arbitrator or General_Synthesizer).
        """
        synthesis_persona_name = None
        if "Impartial_Arbitrator" in persona_sequence:
            synthesis_persona_name = "Impartial_Arbitrator"
        elif "General_Synthesizer" in persona_sequence:
            synthesis_persona_name = "General_Synthesizer"
        
        if not synthesis_persona_name:
            self._log_with_context("error", "No synthesis persona (Impartial_Arbitrator or General_Synthesizer) found in sequence.")
            return {"error": "No synthesis persona found."}

        self._log_with_context("info", f"Executing final synthesis turn for persona: {synthesis_persona_name}")
        persona_config = self.all_personas.get(synthesis_persona_name)
        if not persona_config:
            self._log_with_context("error", f"Synthesis persona configuration not found for {synthesis_persona_name}.")
            return {"error": f"{synthesis_persona_name} config missing."}

        # Prepare the full debate history for the synthesis persona
        full_debate_context = {
            "initial_prompt": self.initial_prompt,
            "debate_history": debate_persona_results
        }
        
        prompt = f"Synthesize the following debate results into a coherent final answer, adhering strictly to your JSON schema:\n\n{json.dumps(full_debate_context, indent=2)}"
        
        # Estimate tokens for this turn
        estimated_tokens = self.tokenizer.count_tokens(prompt) + persona_config.max_tokens
        self.check_budget("synthesis", estimated_tokens, synthesis_persona_name)

        try:
            output = self._execute_llm_turn(synthesis_persona_name, persona_config, prompt, "synthesis")
            self._log_with_context("info", f"Final synthesis turn completed by {synthesis_persona_name}.")
            return output
        except Exception as e:
            self._log_with_context("error", f"Error during final synthesis turn by {synthesis_persona_name}: {e}", exc_info=True)
            self.rich_console.print(f"[red]Error during final synthesis turn: {e}[/red]")
            # Re-raise critical exceptions, otherwise return a structured error
            if isinstance(e, (TokenBudgetExceededError, ChimeraError, CircuitBreakerError)):
                raise e
            return {"error": f"Synthesis turn failed: {e}", "malformed_blocks": [{"type": "SYNTHESIS_ERROR", "message": str(e)}]}

    def _execute_llm_turn(self, persona_name: str, persona_config: PersonaConfig, prompt: str, phase: str) -> Any: # Changed return type to Any
        """
        Executes a single LLM turn for a given persona, handling parsing and validation.
        Includes specific error handling for SchemaValidationError to trigger circuit breaker.
        """
        
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
            
            # --- FIX: Conditionally apply JSON parsing and validation ---
            if persona_name in self.PERSONA_OUTPUT_SCHEMAS:
                schema_model = self.PERSONA_OUTPUT_SCHEMAS[persona_name] # Get schema directly, no default
                
                parser = LLMOutputParser()
                parsed_output = parser.parse_and_validate(raw_llm_output, schema_model)
                
                # MODIFIED: Remove the aggressive re-raising of SchemaValidationError.
                # The LLMOutputParser already returns a structured dictionary (even if it's an error structure)
                # and populates 'malformed_blocks' if issues are found.
                # The 'malformed_blocks' field is part of the expected schema for reporting errors.
                # Therefore, if the parser successfully returns a dictionary with this field,
                # it should not be treated as a SchemaValidationError at this stage.
                
                # If the parsed_output contains malformed_blocks, log it and add to intermediate steps.
                # This allows the debate to continue and the UI to display the malformed blocks.
                if parsed_output.get("malformed_blocks"):
                    self._log_with_context("warning", f"LLM output for {persona_name} contained malformed blocks.",
                                           persona=persona_name, malformed_blocks=parsed_output["malformed_blocks"])
                    self.intermediate_steps.setdefault("malformed_blocks", []).extend(parsed_output["malformed_blocks"])
                
                # --- ADDED LOGIC TO RE-RAISE AS SchemaValidationError IF PARSER INDICATES CRITICAL FAILURE ---
                # Check if the parser itself flagged a critical issue that should halt the debate.
                # This happens if JSON extraction or basic structure is fundamentally broken.
                if parsed_output.get("error_type") == "SCHEMA_VALIDATION_FAILED" or \
                   any(block.get("type") in ["JSON_EXTRACTION_FAILED", "JSON_DECODE_ERROR", "INVALID_JSON_STRUCTURE"] for block in parsed_output.get("malformed_blocks", [])):
                    # Re-raise as SchemaValidationError for consistent handling in run_debate
                    raise SchemaValidationError(
                        error_type="LLM_OUTPUT_MALFORMED",
                        field_path="N/A", # Or more specific if available from parsed_output
                        invalid_value=raw_llm_output[:500],
                        details={"persona": persona_name, "raw_output_snippet": raw_llm_output[:500], "malformed_blocks": parsed_output.get("malformed_blocks", [])}
                    )
                # --- END ADDED LOGIC ---
                
                return parsed_output # Always return the parsed_output, even if it contains error info
            else:
                # Persona is not expected to produce structured JSON, return raw text
                self._log_with_context("info", f"Persona {persona_name} is not configured for structured JSON output. Returning raw text.", persona=persona_name)
                return raw_llm_output # Return the raw string directly
            # --- END FIX ---

        except CircuitBreakerError as cbe:
            self._log_with_context("error", f"Circuit breaker open for {persona_name} LLM call: {cbe}",
                                   persona=persona_name, exc_info=True)
            self.status_callback(f"[red]Circuit breaker open for {persona_name}. Skipping turn.[/red]",
                                 state="error",
                                 current_total_tokens=self.tokens_used,
                                 current_total_cost=self.get_total_estimated_cost())
            raise cbe # Consistently re-raise
        except TokenBudgetExceededError as tbe:
            self._log_with_context("error", f"Token budget exceeded for {persona_name}: {tbe}",
                                   persona=persona_name, exc_info=True)
            self.status_callback(f"[red]Token budget exceeded for {persona_name}. Skipping turn.[/red]",
                                 state="error",
                                 current_total_tokens=self.tokens_used,
                                 current_total_cost=self.get_total_estimated_cost())
            raise tbe # Re-raise to be caught by the main debate loop
        except LLMProviderError as lpe:
            self._log_with_context("error", f"LLM Provider Error for {persona_name}: {lpe}",
                                   persona=persona_name, exc_info=True)
            self.status_callback(f"[red]LLM Provider Error for {persona_name}. Skipping turn.[/red]",
                                 state="error",
                                 current_total_tokens=self.tokens_used,
                                 current_total_cost=self.get_total_estimated_cost())
            raise lpe # Consistently re-raise
        except SchemaValidationError as sve:
            self._log_with_context("error", f"Schema validation failed for {persona_name} output: {sve}",
                                   persona=persona_name, exc_info=True)
            self.status_callback(f"[red]Schema validation failed for {persona_name}. Circuit breaker may trip.[/red]",
                                 state="error",
                                 current_total_tokens=self.tokens_used,
                                 current_total_cost=self.get_total_estimated_cost())
            raise sve # Consistently re-raise
        except Exception as e:
            self._log_with_context("error", f"Unexpected error during {persona_name} LLM turn: {e}",
                                   persona=persona_name, exc_info=True)
            self.status_callback(f"[red]Unexpected error during {persona_name} turn. Skipping.[/red]",
                                 state="error",
                                 current_total_tokens=self.tokens_used,
                                 current_total_cost=self.get_total_estimated_cost())
            # Wrap any other unexpected errors in a generic ChimeraError for consistent handling
            raise ChimeraError(f"An unexpected error occurred during {persona_name}'s turn: {e}",
                               details={"persona": persona_name, "traceback": traceback.format_exc()}) from e

    def _finalize_debate_results(self, context_persona_turn_results: Optional[Dict[str, Any]], debate_persona_results: List[Dict[str, Any]], synthesis_persona_results: Optional[Dict[str, Any]]) -> Tuple[Any, Dict[str, Any]]:
        """
        Synthesizes the final answer and prepares the intermediate steps for display.
        """
        final_answer = synthesis_persona_results
        
        # Ensure malformed_blocks is always present in the final output
        # If the final answer is a string (from a non-JSON persona), wrap it
        if not isinstance(final_answer, dict):
            final_answer = {"general_output": str(final_answer), "malformed_blocks": []}
        if "malformed_blocks" not in final_answer:
            final_answer["malformed_blocks"] = []

        self._update_intermediate_steps_with_totals()
        return final_answer, self.intermediate_steps

    def _update_intermediate_steps_with_totals(self):
        """Updates the intermediate steps dictionary with total token usage and estimated cost."""
        self.intermediate_steps["Total_Tokens_Used"] = self.tokens_used
        self.intermediate_steps["Total_Estimated_Cost_USD"] = self.get_total_estimated_cost()

    def run_debate(self) -> Tuple[Any, Dict[str, Any]]:
        """
        Orchestrates the full Socratic debate process.
        Returns the final synthesized answer and a dictionary of intermediate steps.
        """
        self._initialize_debate_state()
        
        # Phase 1: Context Analysis (if applicable)
        self.status_callback("Phase 1: Analyzing Context...", "running", self.tokens_used, self.get_total_estimated_cost(), progress_pct=self.get_progress_pct("context"))
        context_analysis_results = self._perform_context_analysis()
        self.intermediate_steps["Context_Analysis_Output"] = context_analysis_results
        
        # Determine persona sequence based on initial prompt and context analysis
        persona_sequence = self._determine_persona_sequence(self.initial_prompt, self.domain, self.intermediate_steps, context_analysis_results)
        self.intermediate_steps["Persona_Sequence"] = persona_sequence
        
        # Phase 2: Context Persona Turn (if in sequence)
        context_persona_turn_results = None
        if "Context_Aware_Assistant" in persona_sequence:
            self.status_callback("Phase 2: Context-Aware Assistant Turn...", "running", self.tokens_used, self.get_total_estimated_cost(), progress_pct=self.get_progress_pct("debate"), current_persona_name="Context_Aware_Assistant")
            context_persona_turn_results = self._process_context_persona_turn(persona_sequence, context_analysis_results)
            self.intermediate_steps["Context_Aware_Assistant_Output"] = context_persona_turn_results
        
        # Phase 3: Main Debate Persona Turns
        self.status_callback("Phase 3: Executing Debate Turns...", "running", self.tokens_used, self.get_total_estimated_cost(), progress_pct=self.get_progress_pct("debate"))
        debate_persona_results = self._execute_debate_persona_turns(persona_sequence, context_persona_turn_results)
        self.intermediate_steps["Debate_History"] = debate_persona_results
        
        # Phase 4: Final Synthesis Persona Turn
        self.status_callback("Phase 4: Synthesizing Final Answer...", "running", self.tokens_used, self.get_total_estimated_cost(), progress_pct=self.get_progress_pct("synthesis"))
        synthesis_persona_results = self._perform_synthesis_persona_turn(persona_sequence, debate_persona_results)
        self.intermediate_steps["Final_Synthesis_Output"] = synthesis_persona_results
        
        # Finalize results and update totals
        self.status_callback("Finalizing Results...", "running", self.tokens_used, self.get_total_estimated_cost(), progress_pct=0.95)
        final_answer, intermediate_steps = self._finalize_debate_results(context_persona_turn_results, debate_persona_results, synthesis_persona_results)
        
        self.status_callback("Socratic Debate Complete!", "complete", self.tokens_used, self.get_total_estimated_cost(), progress_pct=1.0)
        self._log_with_context("info", "Socratic Debate process completed successfully.",
                               total_tokens=self.tokens_used, total_cost=self.get_total_estimated_cost())
        
        return final_answer, intermediate_steps