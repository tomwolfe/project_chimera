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
import json
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
from src.constants import SELF_ANALYSIS_PERSONA_SEQUENCE # Import for self-analysis persona sequence

# Configure logging
logger = logging.getLogger(__name__)

class SocraticDebate:
    # --- FIX START ---
    # Define the mapping of persona names to their expected Pydantic output schemas
    # This attribute was missing, causing the AttributeError.
    PERSONA_OUTPUT_SCHEMAS = {
        "Impartial_Arbitrator": LLMOutput,
        "Context_Aware_Assistant": ContextAnalysisOutput,
        "Constructive_Critic": CritiqueOutput, # Added based on system prompt and model definition
        # Add other personas here if they are expected to produce structured output
        # e.g., "Code_Architect": LLMOutput, "Security_Auditor": LLMOutput, etc.
        # For now, only explicitly known structured output personas are mapped.
    }
    # --- FIX END ---

    def __init__(self, initial_prompt: str, api_key: str,
                 codebase_context: Optional[Dict[str, str]] = None,
                 settings: Optional[ChimeraSettings] = None,
                 all_personas: Optional[Dict[str, PersonaConfig]] = None,
                 persona_sets: Optional[Dict[str, List[str]]] = None,
                 # REMOVED persona_sequence from arguments
                 domain: Optional[str] = None,
                 max_total_tokens_budget: int = 10000,
                 model_name: str = "gemini-2.5-flash-lite",
                 status_callback: Optional[Callable] = None,
                 rich_console: Optional[Console] = None,
                 context_token_budget_ratio: float = 0.25,
                 context_analyzer: Optional[ContextRelevanceAnalyzer] = None
                 ):
        """
        Initialize a Socratic debate session.
        
        Args:
            initial_prompt: The user's initial prompt/question
            api_key: API key for the LLM provider
            codebase_context: Optional context about the codebase for code-related prompts
            settings: Optional custom settings; uses defaults if not provided
            all_personas: Optional custom personas; uses defaults if not provided
            persona_sets: Optional custom persona groupings; uses defaults if not provided
            domain: The selected reasoning domain/framework.
            max_total_tokens_budget: Maximum token budget for the entire debate process
            model_name: Name of the LLM model to use (user's explicit choice)
            status_callback: Callback function for updating UI status.
            rich_console: Rich Console instance for logging.
            context_token_budget_ratio: Ratio of total budget allocated to context analysis.
            context_analyzer: An optional pre-initialized and cached ContextRelevanceAnalyzer instance.
        """
        self.settings = settings or ChimeraSettings()
        self.context_token_budget_ratio = context_token_budget_ratio
        self.max_total_tokens_budget = max_total_tokens_budget
        self.tokens_used = 0 # Total tokens used across all phases
        self.model_name = model_name
        self.status_callback = status_callback
        self.rich_console = rich_console
        
        self.initial_prompt = initial_prompt
        self.codebase_context = codebase_context

        # Initialize the LLM provider. This might raise LLMProviderError if API key is invalid.
        try:
            # Pass rich_console to GeminiProvider for its internal logging
            self.llm_provider = GeminiProvider(api_key=api_key, model_name=self.model_name, rich_console=self.rich_console)
        except LLMProviderError as e:
            # Log the error using the console if available, or logger
            if self.rich_console:
                self.rich_console.print(f"[red]Failed to initialize LLM provider: {e}[/red]")
            else:
                logger.error(f"Failed to initialize LLM provider: {e}")
            # Re-raise as ChimeraError for consistent error handling in the app
            raise ChimeraError(f"LLM provider initialization failed: {e}") from e
        except Exception as e: # Catch any other unexpected errors during initialization
            if self.rich_console:
                self.rich_console.print(f"[red]An unexpected error occurred during LLM provider initialization: {e}[/red]")
            else:
                logger.error(f"An unexpected error occurred during LLM provider initialization: {e}")
            raise ChimeraError(f"LLM provider initialization failed unexpectedly: {e}") from e

        # Ensure tokenizer is initialized only if client is successful
        try:
            self.tokenizer = self.llm_provider.tokenizer # Use the tokenizer from the provider
        except AttributeError:
            # This should ideally not happen if LLMProvider init is successful
            raise ChimeraError("LLM provider tokenizer is not available.")

        # Calculate token budgets for different phases
        self._calculate_token_budgets()

        self.context_analyzer = context_analyzer # Use the provided analyzer instance
        
        self.all_personas = all_personas or {}
        self.persona_sets = persona_sets or {} # Store persona_sets
        # REMOVED: self.persona_sequence = persona_sequence or []
        self.domain = domain
        
        # Initialize PersonaRouter with all loaded personas AND persona_sets
        self.persona_router = PersonaRouter(self.all_personas, self.persona_sets) # Pass persona_sets here
        
        # If codebase_context was provided, compute embeddings now if context_analyzer is available.
        if self.codebase_context and self.context_analyzer:
            if isinstance(self.codebase_context, dict):
                try:
                    # Ensure embeddings are computed if not already done.
                    if not self.context_analyzer.file_embeddings:
                        self.context_analyzer.compute_file_embeddings(self.codebase_context)
                except Exception as e:
                    logger.error(f"Failed to compute embeddings for codebase context: {e}")
                    if self.status_callback:
                        self.status_callback(message=f"[red]Error computing context embeddings: {e}[/red]", state="warning")
            else:
                logger.warning("codebase_context was not a dictionary, skipping embedding computation.")

    def _calculate_token_budgets(self):
        """
        Calculates token budgets for different phases of the debate using
        max_total_tokens_budget and context_token_budget_ratio.
        Handles potential errors during LLM provider interactions.
        """
        # Ensure context_token_budget_ratio is within reasonable bounds
        context_ratio = max(0.05, min(0.5, self.context_token_budget_ratio)) # Clamp between 5% and 50%
        
        # Calculate remaining ratio for debate and synthesis
        remaining_ratio = 1.0 - context_ratio
        # Split remaining ratio for debate and synthesis (e.g., 50/50)
        debate_ratio = remaining_ratio / 2.0
        synthesis_ratio = remaining_ratio / 2.0
        
        # Estimate tokens for initial input (context + prompt)
        context_str = ""
        if self.codebase_context:
            context_str = "\n".join(f"{fname}:\n{content}" for fname, content in self.codebase_context.items())
        
        prompt_for_estimation = self.initial_prompt if self.initial_prompt else ""

        try:
            combined_input_text = f"{context_str}\n\n{prompt_for_estimation}" if context_str else prompt_for_estimation
            
            self.initial_input_tokens = self.llm_provider.count_tokens(combined_input_text)
            
            available_tokens_for_phases = max(0, self.max_total_tokens_budget - self.initial_input_tokens)
            
            self.phase_budgets = {
                "context": max(200, int(available_tokens_for_phases * context_ratio)),
                "debate": max(500, int(available_tokens_for_phases * debate_ratio)),
                "synthesis": max(400, int(available_tokens_for_phases * synthesis_ratio))
            }

            logger.info(f"SocraticDebate token budgets initialized: "
                       f"Initial Input={self.initial_input_tokens}, "
                       f"Context={self.phase_budgets['context']}, "
                       f"Debate={self.phase_budgets['debate']}, "
                       f"Synthesis={self.phase_budgets['synthesis']}")

        except LLMProviderError as e:
            logger.error(f"LLM Provider Error during token calculation: {e}")
            self.phase_budgets = {"context": 500, "debate": 15000, "synthesis": 1000}
            self.initial_input_tokens = 0
            raise ChimeraError(f"LLM provider error: {e}") from e
        except Exception as e:
            logger.error(f"An unexpected error occurred calculating token budgets: {e}")
            self.phase_budgets = {"context": 500, "debate": 15000, "synthesis": 1000}
            self.initial_input_tokens = 0
            raise ChimeraError("Failed to calculate token budgets due to an unexpected error.") from e
    
    def track_token_usage(self, phase: str, tokens: int):
        """Tracks token usage for a specific phase and updates total used tokens."""
        if phase in self.tokens_used_per_phase:
            self.tokens_used_per_phase[phase] += tokens
        else:
            logger.warning(f"Attempted to track tokens for unknown phase: {phase}")
        self.tokens_used += tokens # Always update total tokens used

    def check_budget(self, phase: str, tokens_needed: int, step_name: str):
        """Checks if adding tokens_needed would exceed the budget for the given phase."""
        if phase not in self.phase_budgets:
            logger.warning(f"Phase '{phase}' not found in budget configuration.")
            return # Cannot check budget if phase is not configured

        current_phase_usage = self.tokens_used_per_phase.get(phase, 0)
        phase_budget = self.phase_budgets.get(phase, 0)
        
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
            cost = self.llm_provider.calculate_usd_cost(
                input_tokens=self.initial_input_tokens,
                output_tokens=max(0, self.tokens_used - self.initial_input_tokens) # Rough estimate of output tokens
            )
            return cost
        except Exception as e:
            logger.error(f"Could not estimate total cost: {e}")
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
            # 1. Initialize state variables
            self._initialize_debate_state()
            
            if self.status_callback:
                self.status_callback(
                    message="Starting Socratic Debate...",
                    state="running",
                    current_total_tokens=self.get_total_used_tokens(),
                    current_total_cost=self.get_total_estimated_cost()
                )

            # 2. Perform Context Analysis and Determine Persona Sequence
            context_analysis_results = self._perform_context_analysis()
            # Pass domain to determine_persona_sequence for context analysis
            persona_sequence = self._determine_persona_sequence(context_analysis_results)
            if self.status_callback:
                # Update status after sequence is determined
                self.status_callback(
                    message=f"Persona sequence determined: [bold]{', '.join(persona_sequence)}[/bold]",
                    state="running",
                    current_total_tokens=self.get_total_used_tokens(),
                    current_total_cost=self.get_total_estimated_cost()
                )

            # 3. Process Context Persona Turn (if applicable)
            if self.status_callback:
                self.status_callback(
                    message="Processing initial context...",
                    state="running",
                    current_total_tokens=self.get_total_used_tokens(),
                    current_total_cost=self.get_total_estimated_cost()
                )
            context_persona_turn_results = self._process_context_persona_turn(persona_sequence, context_analysis_results)
            
            # 4. Execute Debate Persona Turns
            debate_persona_results = self._execute_debate_persona_turns(persona_sequence, context_persona_turn_results)
            if self.status_callback:
                # Update status after all debate turns are done, before synthesis
                self.status_callback(
                    message="All debate turns completed. Proceeding to synthesis.",
                    state="running",
                    current_total_tokens=self.get_total_used_tokens(),
                    current_total_cost=self.get_total_estimated_cost()
                )
            
            # 5. Perform Synthesis Persona Turn
            # This method now returns the final answer structure directly
            final_answer = self._perform_synthesis_persona_turn(persona_sequence, debate_persona_results)
            
            # 6. Finalize Results (updates intermediate_steps and ensures final_answer is a dict)
            final_answer, intermediate_steps = self._finalize_debate_results(
                context_persona_turn_results, debate_persona_results, final_answer # Pass final_answer directly
            )
            
            # Ensure malformed_blocks is always present, even if empty
            if "malformed_blocks" not in final_answer:
                final_answer["malformed_blocks"] = []

            # Final status update upon successful completion
            if self.status_callback:
                self.status_callback(
                    message="Socratic Debate process finalized.",
                    state="complete",
                    current_total_tokens=self.get_total_used_tokens(),
                    current_total_cost=self.get_total_estimated_cost()
                )

        except TokenBudgetExceededError as e:
            self.logger.error(f"Socratic Debate failed: Token budget exceeded. {e}")
            if self.status_callback:
                self.status_callback(message=f"[red]Socratic Debate Failed: Token Budget Exceeded[/red]", state="error")
            
            # Ensure final_answer is a dict with malformed_blocks for consistent UI display
            if not isinstance(final_answer, dict):
                final_answer = {
                    "COMMIT_MESSAGE": "Debate Failed - Token Budget Exceeded",
                    "RATIONALE": f"The Socratic debate exceeded the allocated token budget. Please consider increasing the budget or simplifying the prompt. Error details: {str(e)}",
                    "CODE_CHANGES": [],
                    "malformed_blocks": [{"type": "TOKEN_BUDGET_ERROR", "message": str(e), "details": e.details}]
                }
            elif "malformed_blocks" not in final_answer:
                final_answer["malformed_blocks"] = [{"type": "TOKEN_BUDGET_ERROR", "message": str(e), "details": e.details}]
            
            self._update_intermediate_steps_with_totals()
            return final_answer, self.intermediate_steps
        
        except ChimeraError as e:
            self.logger.error(f"Socratic Debate failed due to ChimeraError: {e}")
            if self.status_callback:
                self.status_callback(message=f"[red]Socratic Debate Failed: {e}[/red]", state="error")
            
            # Ensure final_answer is a dict with malformed_blocks for consistent UI display
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
            self.logger.error(f"Socratic Debate failed due to an unexpected error: {e}", exc_info=True)
            if self.status_callback:
                self.status_callback(message=f"[red]Socratic Debate Failed: Unexpected Error[/red]", state="error")
            
            # Ensure final_answer is a dict with malformed_blocks for consistent UI display
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

    def _initialize_debate_state(self):
        """Initializes state variables for a new debate run."""
        self.intermediate_steps = {}
        self.final_answer = None
        self.tokens_used_per_phase = {"context": 0, "debate": 0, "synthesis": 0}
        self.tokens_used = self.initial_input_tokens # Start with initial input tokens
        logger.debug("Debate state initialized.")
        
        # Ensure intermediate_steps is always a dictionary
        if not isinstance(self.intermediate_steps, dict):
            self.intermediate_steps = {}
        self.intermediate_steps.update({
            "Total_Tokens_Used": self.tokens_used, "Total_Estimated_Cost_USD": self.get_total_estimated_cost()
        })

    def _perform_context_analysis(self) -> Optional[Dict[str, Any]]:
        """Performs context analysis (finding relevant files) if context is available."""
        context_analysis_results = None
        if self.context_analyzer and self.codebase_context:
            try:
                # Determine initial persona sequence for relevance scoring
                # Pass domain to determine_persona_sequence for context analysis
                if self.status_callback:
                    self.status_callback(
                        message="[bold]Analyzing context[/bold] to find relevant files...",
                        state="running",
                        current_total_tokens=self.get_total_used_tokens(),
                        current_total_cost=self.get_total_estimated_cost()
                    )
                initial_sequence_for_relevance = self.persona_router.determine_persona_sequence(
                    self.initial_prompt,
                    domain=self.domain, # Pass domain here
                    intermediate_results=self.intermediate_steps
                )
                
                relevant_files_info = self.context_analyzer.find_relevant_files(
                    self.initial_prompt,
                    active_personas=initial_sequence_for_relevance
                )
                context_analysis_results = {"relevant_files": relevant_files_info}
                self.intermediate_steps["Relevant_Files_Context"] = {"relevant_files": relevant_files_info}
                logger.info(f"Context analysis completed. Found {len(relevant_files_info)} relevant files.")
                if self.status_callback:
                    self.status_callback(
                        message=f"Context analysis complete. Found [bold]{len(relevant_files_info)}[/bold] relevant files.",
                        state="running",
                        current_total_tokens=self.get_total_used_tokens(),
                        current_total_cost=self.get_total_estimated_cost()
                    )
            except Exception as e:
                logger.error(f"Error during context analysis file finding: {e}")
                self.intermediate_steps["Context_Analysis_Error"] = {"error": str(e)}
                if self.status_callback:
                    self.status_callback(message=f"[red]Error during context analysis: {e}[/red]", state="warning")
        else:
            logger.info("No context analyzer or codebase context available. Skipping context analysis.")
        return context_analysis_results

    def _determine_persona_sequence(self, context_analysis_results: Optional[Dict[str, Any]]) -> List[str]:
        """Determines the persona sequence based on prompt, context, and self-analysis detection."""
        # Call the PersonaRouter's method, passing the domain
        unique_sequence = self.persona_router.determine_persona_sequence(
            prompt=self.initial_prompt,
            domain=self.domain, # Pass the domain here
            intermediate_results=self.intermediate_steps,
            context_analysis_results=context_analysis_results
        )
        
        self.intermediate_steps["Persona_Sequence_Order"] = unique_sequence
        logger.info(f"Final persona sequence determined: {unique_sequence}")
        return unique_sequence

    def _process_context_persona_turn(self, persona_sequence: List[str], context_analysis_results: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """Processes the turn for a dedicated context processing persona."""
        context_processing_persona_name = None
        context_processing_persona_config = None
        
        # Find a persona suitable for initial context processing
        # This logic might need adjustment if persona names change or are more dynamic
        for p_name in persona_sequence:
            if "Generalist_Assistant" in p_name or "Context_Aware_Assistant" in p_name:
                context_processing_persona_name = p_name
                break
        
        if context_processing_persona_name and context_processing_persona_name in self.all_personas:
            context_processing_persona_config = self.all_personas[context_processing_persona_name]
            
            # Construct prompt for context processing
            context_prompt_for_persona = self.initial_prompt
            if context_analysis_results and context_analysis_results.get("relevant_files"):
                context_prompt_for_persona = f"Initial Prompt: {self.initial_prompt}\n\n"
                context_prompt_for_persona += "Relevant Code Files:\n"
                for file_path, score in context_analysis_results["relevant_files"][:5]: # Limit to top 5
                    context_prompt_for_persona += f"- {file_path} (Relevance: {score:.2f})\n"
                    if file_path in self.codebase_context:
                        context_prompt_for_persona += f"```\n{self.codebase_context[file_path][:500]}...\n```\n"
            
            try:
                logger.info(f"Performing context processing with persona: {context_processing_persona_name}")
                if self.status_callback:
                    self.status_callback(
                        message=f"Running [bold]{context_processing_persona_name}[/bold] for context processing...",
                        state="running",
                        current_total_tokens=self.get_total_used_tokens(),
                        current_total_cost=self.get_total_estimated_cost()
                    )
                turn_results = self._execute_llm_turn(
                    persona_name=context_processing_persona_name,
                    persona_config=context_processing_persona_config,
                    # Ensure prompt is a string, handle potential None from context_analysis_results
                    prompt=context_prompt_for_persona if context_prompt_for_persona is not None else self.initial_prompt,
                    phase="context"
                )
                if self.status_callback:
                    self.status_callback(
                        message=f"Completed persona: [bold]{context_processing_persona_name}[/bold] (context phase).",
                        state="running",
                        current_total_tokens=self.get_total_used_tokens(),
                        current_total_cost=self.get_total_estimated_cost()
                    )
                return turn_results
                
            except TokenBudgetExceededError as e:
                logger.error(f"Token budget exceeded during context processing: {e}")
                raise e
            except Exception as e:
                logger.error(f"Error during context processing generation for {context_processing_persona_name}: {e}")
                self.intermediate_steps[f"{context_processing_persona_name}_Error"] = str(e)
                error_tokens = self.llm_provider.count_tokens(f"Error processing {context_processing_persona_name}: {str(e)}") + 50
                self.track_token_usage("context", error_tokens)
                self.check_budget("context", 0, f"Error handling {context_processing_persona_name} context processing")
        else:
            logger.info("No dedicated context processing persona found or no context available. Skipping dedicated context processing phase.")
        return None

    def _execute_debate_persona_turns(self, persona_sequence: List[str], context_persona_turn_results: Optional[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Executes the debate turns for the personas in the sequence."""
        all_debate_turns = []
        if context_persona_turn_results:
            all_debate_turns.append(context_persona_turn_results)

        # Determine the synthesis persona name
        synthesis_persona_name = None
        
        if self.status_callback:
            self.status_callback(
                message="Starting final [bold]debate turns[/bold] with core personas...",
                state="running",
                current_total_tokens=self.get_total_used_tokens(),
                current_total_cost=self.get_total_estimated_cost())
        # Check if the last persona in the sequence is a designated synthesizer
        if persona_sequence and persona_sequence[-1] in ["Impartial_Arbitrator", "General_Synthesizer"]:
            synthesis_persona_name = persona_sequence[-1]
        
        # Define which personas to run in the debate loop (exclude the synthesis persona if it's the last one)
        debate_personas_to_run = persona_sequence
        if synthesis_persona_name and persona_sequence[-1] == synthesis_persona_name:
            debate_personas_to_run = persona_sequence[:-1] # Exclude the last one for the debate loop

        for i, persona_name in enumerate(debate_personas_to_run):
            if persona_name not in self.all_personas:
                logger.warning(f"Persona '{persona_name}' not found in loaded personas. Skipping.")
                if self.status_callback:
                    self.status_callback(
                        message=f"[yellow]Skipping persona '{persona_name}' (not found).[/yellow]",
                        state="running",
                        current_total_tokens=self.get_total_used_tokens(),
                        current_total_cost=self.get_total_estimated_cost()
                    )
                continue
            
            persona_config = self.all_personas[persona_name]
            
            # Construct the prompt for the current persona
            current_persona_prompt = self.initial_prompt
            if all_debate_turns:
                # Use the output of the *last processed turn* as context for the current persona.
                current_persona_prompt = all_debate_turns[-1]["output"]
                if isinstance(current_persona_prompt, dict): # If output is dict, convert to string for prompt
                    current_persona_prompt = json.dumps(current_persona_prompt, indent=2)
                elif not isinstance(current_persona_prompt, str): # Ensure it's a string
                    current_persona_prompt = str(current_persona_prompt)

            # --- Execute LLM Turn ---
            try:
                logger.info(f"Executing debate turn with persona: {persona_name}")
                if self.status_callback:
                    self.status_callback(
                        message=f"Running persona: [bold]{persona_name}[/bold]...",
                        state="running",
                        current_total_tokens=self.get_total_used_tokens(),
                        current_total_cost=self.get_total_estimated_cost()
                    )
                turn_results = self._execute_llm_turn(
                    persona_name=persona_name,
                    persona_config=persona_config,
                    prompt=current_persona_prompt,
                    phase="debate"
                )
                if turn_results:
                    if self.status_callback:
                        self.status_callback(
                            message=f"Completed persona: [bold]{persona_name}[/bold].",
                            state="running",
                            current_total_tokens=self.get_total_used_tokens(),
                            current_total_cost=self.get_total_estimated_cost()
                        )
                    all_debate_turns.append(turn_results)
            except TokenBudgetExceededError as e:
                logger.error(f"Token budget exceeded during debate turn for persona {persona_name}: {e}")
                raise e
            except Exception as e:
                logger.error(f"An unexpected error occurred during debate turn for persona {persona_name}: {e}")
                self.intermediate_steps[f"{persona_name}_Error"] = str(e)
                error_tokens = self.llm_provider.count_tokens(f"Error processing {persona_name}: {str(e)}") + 50
                self.track_token_usage("debate", error_tokens)
                self.check_budget("debate", 0, f"Error handling {persona_name} debate turn")
                if self.status_callback:
                    self.status_callback(
                        message=f"[red]Error with persona [bold]{persona_name}[/bold]: {e}[/red]",
                        state="error",
                        current_total_tokens=self.get_total_used_tokens(),
                        current_total_cost=self.get_total_estimated_cost()
                    )
        
        return all_debate_turns

    def _perform_synthesis_persona_turn(self, persona_sequence: List[str], debate_persona_results: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """
        Performs the synthesis turn with retry logic for JSON validation.
        Returns the final structured answer (LLMOutput or general_output dict).
        """
        synthesis_persona_name = None
        synthesis_persona_config = None
        
        # Find the synthesis persona, typically the last one if it's a designated synthesizer
        if persona_sequence:
            potential_synthesis_persona = persona_sequence[-1]
            if potential_synthesis_persona in ["Impartial_Arbitrator", "General_Synthesizer"]:
                synthesis_persona_name = potential_synthesis_persona
                if synthesis_persona_name in self.all_personas:
                    synthesis_persona_config = self.all_personas[synthesis_persona_name]
        
        if synthesis_persona_name and synthesis_persona_config:
            
            if self.status_callback:
                self.status_callback(
                    message=f"Starting final [bold]synthesis[/bold] with persona: [bold]{synthesis_persona_name}[/bold]...",
                    state="running",
                    current_total_tokens=self.get_total_used_tokens(),
                    current_total_cost=self.get_total_estimated_cost()
                )
            # Construct the synthesis prompt
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
            
            max_retries = 2  # Number of retries for synthesis output validation
            
            # Determine if JSON validation is required for this synthesis persona
            requires_json_validation = (synthesis_persona_name == "Impartial_Arbitrator")

            for attempt in range(max_retries + 1):
                try:
                    # _execute_llm_turn returns a dict like {'persona': ..., 'output': LLM_GENERATED_CONTENT, ...}
                    turn_data_from_llm = self._execute_llm_turn(
                        persona_name=synthesis_persona_name,
                        persona_config=synthesis_persona_config,
                        prompt=synthesis_prompt,
                        phase="synthesis"
                    )
                    
                    # Extract the actual LLM-generated content from the turn_data_from_llm
                    llm_generated_content = turn_data_from_llm.get('output')

                    # Store this for potential logging/display if retries fail
                    # This is the content that will be validated/returned as the final answer
                    current_synthesis_output = llm_generated_content 

                    # --- Conditional JSON Validation ---
                    if requires_json_validation:
                        is_failure = False
                        failure_reason = ""

                        if not isinstance(current_synthesis_output, dict):
                            is_failure = True
                            failure_reason = "Synthesis output is not a dictionary (expected JSON)."
                        else:
                            malformed_blocks = current_synthesis_output.get("malformed_blocks", [])
                            if malformed_blocks:
                                # Check if any malformed block indicates a JSON adherence problem
                                if any(block.get("type") in ["JSON_EXTRACTION_FAILED", "JSON_DECODE_ERROR", "INVALID_JSON_STRUCTURE", "LLM_FAILED_JSON_ADHERENCE", "SYNTHESIS_EXECUTION_ERROR"] for block in malformed_blocks):
                                    is_failure = True
                                    failure_reason = "Output contains malformed blocks indicating JSON adherence failure."
                            # Also check for the specific fallback error JSON structure
                            elif current_synthesis_output.get("COMMIT_MESSAGE") == "LLM_GENERATION_ERROR" and \
                                 any(block.get("type") == "LLM_FAILED_JSON_ADHERENCE" for block in malformed_blocks):
                                is_failure = True
                                failure_reason = "LLM produced the fallback error JSON for JSON adherence."
                            # If it's a valid structure but missing key fields (e.g., COMMIT_MESSAGE), it's also a failure
                            elif not ("COMMIT_MESSAGE" in current_synthesis_output and "RATIONALE" in current_synthesis_output):
                                is_failure = True
                                failure_reason = "Output dictionary is missing required keys (COMMIT_MESSAGE, RATIONALE)."

                        if not is_failure:
                            logger.info(f"Synthesis output validated successfully on attempt {attempt + 1}.")
                            return current_synthesis_output # Success! Return the validated LLMOutput dict
                        else:
                            # Output is a failure, proceed to retry if possible
                            if attempt < max_retries:
                                logger.warning(f"Synthesis output validation failed on attempt {attempt + 1} ({failure_reason}). Retrying...")
                                # Generate a correction prompt based on the failure
                                correction_prompt_content = f"Previous output was invalid. Please re-generate the JSON output adhering strictly to the schema. The failure reason was: {failure_reason}. The previous output was:\n\n{json.dumps(current_synthesis_output, indent=2)}"
                                
                                # Re-construct the synthesis prompt to include the correction instruction
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
                                synthesis_prompt += f"\n\n{correction_prompt_content}" # Append correction instruction
                                
                                # Update the intermediate steps with the failed attempt's output
                                # This is for logging/display in the UI, not for the final answer
                                self.intermediate_steps[f"{synthesis_persona_name}_Output_Attempt_{attempt+1}"] = current_synthesis_output
                                continue # Proceed to the next attempt
                            else:
                                logger.error(f"Synthesis output validation failed after {max_retries} retries.")
                                # If all retries fail, return the last recorded result
                                return current_synthesis_output # Return the last result, which will contain error information

                    else: # No JSON validation required (e.g., General_Synthesizer)
                        logger.info(f"Synthesis output for {synthesis_persona_name} does not require strict JSON validation. Returning raw output.")
                        # current_synthesis_output is the raw text from LLM, or an error dict from LLMOutputParser
                        # if it tried to parse JSON and failed (even though it wasn't asked to).
                        
                        if isinstance(current_synthesis_output, dict):
                            # If LLMOutputParser returned an error dict (e.g., JSON_EXTRACTION_FAILED)
                            # for a persona that wasn't supposed to produce JSON, we should still
                            # return a structured error, but ensure it's not confused with LLMOutput.
                            # It should have 'general_output' if it's a general error, or just pass through.
                            if "general_output" not in current_synthesis_output:
                                current_synthesis_output["general_output"] = current_synthesis_output.get("raw_llm_output_snippet", "No specific general output found.")
                            if "malformed_blocks" not in current_synthesis_output:
                                current_synthesis_output["malformed_blocks"] = []
                            # Ensure it doesn't have COMMIT_MESSAGE/RATIONALE unless it's an error from LLMOutputParser
                            # that *explicitly* sets them.
                            # For General_Synthesizer, if it's an error dict from LLMOutputParser, it might have them.
                            # We should just return it as is.
                            return current_synthesis_output
                        else:
                            # This is the expected case: raw text from General_Synthesizer
                            return {
                                "general_output": current_synthesis_output,
                                "malformed_blocks": []
                            }

                except Exception as e: # Catch errors from _execute_llm_turn itself (e.g., API errors)
                    if self.status_callback:
                        self.status_callback(
                            message=f"[red]Error during synthesis turn: {e}[/red]",
                            state="error",
                            current_total_tokens=self.get_total_used_tokens(),
                            current_total_cost=self.get_total_estimated_cost()
                        )
                    logger.error(f"Error during synthesis turn execution: {e}")
                    if attempt == max_retries:
                        logger.error(f"Final synthesis attempt failed due to execution error: {e}")
                        # If it's the last attempt and execution fails, return a specific error
                        return {
                            "COMMIT_MESSAGE": "Synthesis Execution Error",
                            "RATIONALE": f"An error occurred during the final synthesis turn: {str(e)}",
                            "CODE_CHANGES": [],
                            "malformed_blocks": [{"type": "SYNTHESIS_EXECUTION_ERROR", "message": str(e)}]
                        }
                    else:
                        logger.warning(f"Execution error on synthesis attempt {attempt + 1}, retrying...")
                        # Store the error for intermediate steps, but don't return it as final answer yet
                        self.intermediate_steps[f"{synthesis_persona_name}_Error_Attempt_{attempt+1}"] = str(e)
                        continue # Retry if not the last attempt
            
            # If the loop finishes without returning, it means all retries failed or an error occurred.
            # Return a generic failure message if no specific error was returned above.
            return {
                "COMMIT_MESSAGE": "Synthesis Failed",
                "RATIONALE": f"Failed to generate valid synthesis output after multiple attempts.",
                "CODE_CHANGES": [],
                "malformed_blocks": [{"type": "SYNTHESIS_FINAL_FAILURE", "message": "An unhandled failure occurred during synthesis."}]
            }

        else:
            logger.warning("No synthesis persona found or sequence is empty. Final answer may be incomplete.")
            # Return a default error if no synthesis persona was identified
            return {
                "COMMIT_MESSAGE": "Synthesis Skipped",
                "RATIONALE": "No synthesis persona was identified in the sequence.",
                "CODE_CHANGES": [],
                "malformed_blocks": [{"type": "NO_SYNTHESIS_PERSONA", "message": "Synthesis persona not found in sequence."}]
            }

    def _execute_llm_turn(self, persona_name: str, persona_config: PersonaConfig, prompt: str, phase: str) -> Optional[Dict[str, Any]]:
        """Executes a single LLM turn for a given persona, handling generation, token tracking, and parsing."""
        
        # Estimate tokens for the *next* step (input + max_output for this persona)
        estimated_next_step_input_tokens = self.llm_provider.count_tokens(prompt=prompt, system_prompt=persona_config.system_prompt)
        estimated_next_step_output_tokens = persona_config.max_tokens
        estimated_next_step_total_tokens = estimated_next_step_input_tokens + estimated_next_step_output_tokens
        estimated_next_step_cost = self.llm_provider.calculate_usd_cost(estimated_next_step_input_tokens, estimated_next_step_output_tokens)

        if self.status_callback:
            self.status_callback(
                message=f"Running persona: [bold]{persona_name}[/bold] ({phase} phase)...",
                state="running",
                current_total_tokens=self.get_total_used_tokens(),
                current_total_cost=self.get_total_estimated_cost(),
                estimated_next_step_tokens=estimated_next_step_total_tokens,
                estimated_next_step_cost=estimated_next_step_cost
            )

        self.check_budget(phase, estimated_next_step_total_tokens, f"Start of {persona_name} turn") # Check budget before starting the turn
        
        try:
            response_text, input_tokens, output_tokens = self.llm_provider.generate(
                prompt=prompt,
                system_prompt=persona_config.system_prompt,
                temperature=persona_config.temperature,
                max_tokens=persona_config.max_tokens,
                requested_model_name=self.model_name,
                persona_config=persona_config,
                intermediate_results=self.intermediate_steps
            )
            
            turn_tokens_used = input_tokens + output_tokens
            self.track_token_usage(phase, turn_tokens_used)
            self.check_budget(phase, 0, f"End of {persona_name} turn") # Check budget after the turn
            if self.status_callback:
                self.status_callback(
                    message=f"Completed persona: [bold]{persona_name}[/bold] ({phase} phase).",
                    state="running", current_total_tokens=self.get_total_used_tokens(), current_total_cost=self.get_total_estimated_cost(),
                    estimated_next_step_tokens=0, estimated_next_step_cost=0.0) # Reset next step estimate
        except TokenBudgetExceededError as e:
            logger.error(f"Token budget exceeded during LLM generation for {persona_name}: {e}")
            raise e # Re-raise to be caught by the main run_debate handler
        except Exception as e:
            logger.error(f"Error during LLM generation for {persona_name}: {e}")
            self.intermediate_steps[f"{persona_name}_Error"] = str(e)
            error_tokens = self.llm_provider.count_tokens(f"Error processing {persona_name}: {str(e)}") + 50
            self.track_token_usage(phase, error_tokens)
            self.check_budget(phase, 0, f"Error handling {persona_name} generation")
            
            # Update UI status to error
            if self.status_callback:
                self.status_callback(
                    message=f"[red]Error with persona [bold]{persona_name}[/bold]: {e}[/red]",
                    state="error",
                    current_total_tokens=self.get_total_used_tokens(),
                    current_total_cost=self.get_total_estimated_cost()
                )
            return None # Return None to indicate failure for this turn

        # Parse and Validate Output
        parsed_output_data = {}
        # Use the PERSONA_OUTPUT_SCHEMAS attribute defined at the class level
        expected_schema = self.PERSONA_OUTPUT_SCHEMAS.get(persona_name)

        if expected_schema: # If a schema is defined (e.g., LLMOutput, CritiqueOutput)
            try:
                # Use the parser's validate method, which handles extraction and schema validation
                parsed_output_data = LLMOutputParser().parse_and_validate(response_text, expected_schema)
            except Exception as e:
                # This catch might be redundant if parse_and_validate always returns a dict with error info
                logger.error(f"Failed to parse/validate output for {persona_name} against {expected_schema.__name__} schema: {e}")
                malformed_blocks_for_fallback = [{"type": "PARSING_OR_VALIDATION_ERROR", "message": str(e), "raw_output": response_text[:500]}]
                
                parsed_output_data = {
                    "error_type": "Parsing/Validation Error",
                    "error_message": f"Failed to parse/validate output for {persona_name}. Error: {str(e)}",
                    "raw_llm_output_snippet": response_text[:500],
                    "malformed_blocks": malformed_blocks_for_fallback
                }
                self.intermediate_steps[f"{persona_name}_Error"] = str(e)
        else: # If expected_schema is None (e.g., General_Synthesizer)
            # For personas that produce free-form text/markdown, store raw text
            parsed_output_data = response_text
            logger.debug(f"Persona {persona_name} does not have a specific JSON schema. Storing raw text output.")
        
        turn_results = {
            "persona": persona_name,
            "output": parsed_output_data,
            "tokens_used": turn_tokens_used,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens
        }
        self.intermediate_steps[f"{persona_name}_Output"] = parsed_output_data
        self.intermediate_steps[f"{persona_name}_Tokens_Used"] = turn_tokens_used
        
        return turn_results

    def _finalize_debate_results(self, context_persona_turn_results: Optional[Dict[str, Any]], 
                                 debate_persona_results: List[Dict[str, Any]], 
                                 synthesis_persona_results: Optional[Dict[str, Any]]) -> Tuple[Any, Dict[str, Any]]:
        
        # Determine the final answer
        # synthesis_persona_results now directly contains the final structured output
        if synthesis_persona_results: 
            self.final_answer = synthesis_persona_results
        elif debate_persona_results: # Fallback to last debate turn if no synthesis
            # Ensure fallback output is a dictionary for consistency
            raw_output = debate_persona_results[-1]["output"]
            if isinstance(raw_output, dict):
                self.final_answer = raw_output
            else:
                self.final_answer = {
                    "COMMIT_MESSAGE": "Debate Fallback - No Synthesis",
                    "RATIONALE": "Synthesis step was skipped or failed. Falling back to last debate persona's output.",
                    "CODE_CHANGES": [],
                    "general_output": str(raw_output),
                    "malformed_blocks": []
                }
        elif context_persona_turn_results: # Fallback to context turn if no debate/synthesis
            # Ensure fallback output is a dictionary for consistency
            raw_output = context_persona_turn_results["output"]
            if isinstance(raw_output, dict):
                self.final_answer = raw_output
            else:
                self.final_answer = {
                    "COMMIT_MESSAGE": "Debate Fallback - No Debate or Synthesis",
                    "RATIONALE": "Debate and synthesis steps were skipped or failed. Falling back to context persona's output.",
                    "CODE_CHANGES": [],
                    "general_output": str(raw_output),
                    "malformed_blocks": []
                }
        else:
            # If no turns were executed, provide a default error response
            self.final_answer = {
                "COMMIT_MESSAGE": "Debate Failed - No Turns Executed",
                "RATIONALE": "The Socratic debate process could not execute any turns or perform synthesis.",
                "CODE_CHANGES": [],
                "malformed_blocks": [{"type": "NO_DEBATE_TURNS", "message": "No debate turns were executed."}]
            }
            logger.error("Socratic debate process completed without executing any turns.")

        # Ensure final_answer is a dictionary, especially if it was None or malformed
        if not isinstance(self.final_answer, dict):
            self.logger.error(f"Final answer was not a dictionary: {type(self.final_answer).__name__}. Creating fallback error.")
            # Convert it to a general output dict.
            self.final_answer = {
                "COMMIT_MESSAGE": "Debate Failed - Final Answer Malformed",
                "RATIONALE": f"The final answer was not a dictionary. Type: {type(self.final_answer).__name__}", # CORRECTED LINE
                "CODE_CHANGES": [], # Empty list for non-code output
                "general_output": str(self.final_answer), # Store the raw string here
                "malformed_blocks": [{"type": "FINAL_ANSWER_MALFORMED", "message": f"Final answer was not a dictionary. Type: {type(self.final_answer).__name__}", "raw_output": str(self.final_answer)[:500]}]
            }
            self.logger.error(f"Final answer was not a dictionary. Type: {type(self.final_answer).__name__}")

        # Ensure malformed_blocks is always present, even if empty
        if "malformed_blocks" not in self.final_answer:
            self.final_answer["malformed_blocks"] = []
        
        # Update intermediate steps with totals
        self._update_intermediate_steps_with_totals()
        
        if self.status_callback:
            self.status_callback(
                message="Socratic Debate process finalized.",
                state="complete",
                current_total_tokens=self.get_total_used_tokens(),
                current_total_cost=self.get_total_estimated_cost()
            )
        return self.final_answer, self.intermediate_steps

    def _update_intermediate_steps_with_totals(self):
        """Updates intermediate steps with total token counts and estimated cost."""
        self.tokens_used += sum(self.tokens_used_per_phase.values())
        
        self.intermediate_steps["Total_Tokens_Used"] = self.tokens_used
        self.intermediate_steps["Total_Estimated_Cost_USD"] = self.get_total_estimated_cost()
        
        self.intermediate_steps["Initial_Prompt_Tokens"] = self.initial_input_tokens
        self.intermediate_steps["Context_Phase_Tokens"] = self.tokens_used_per_phase.get("context", 0)
        self.intermediate_steps["Debate_Phase_Tokens"] = self.tokens_used_per_phase.get("debate", 0)
        self.intermediate_steps["Synthesis_Phase_Tokens"] = self.tokens_used_per_phase.get("synthesis", 0)

        # Add any explicit errors logged during the process to intermediate steps
        # Ensure malformed_blocks is always present, even if empty
        if "malformed_blocks" not in self.intermediate_steps:
            self.intermediate_steps["malformed_blocks"] = []
        # Ensure final_answer is a dictionary, especially if it was None or malformed
        if not isinstance(self.final_answer, dict):
            self.final_answer = {"malformed_blocks": [{"type": "FINAL_ANSWER_MALFORMED", "message": f"Final answer was not a dictionary. Type: {type(self.final_answer).__name__}"}]}