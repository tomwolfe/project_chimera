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
from src.models import PersonaConfig, ReasoningFrameworkConfig, LLMOutput, CodeChange
from src.config.settings import ChimeraSettings
from src.exceptions import ChimeraError, LLMResponseValidationError, SchemaValidationError, TokenBudgetExceededError, LLMProviderError # Corrected import, added LLMProviderError
from src.constants import SELF_ANALYSIS_PERSONA_SEQUENCE # Import for self-analysis persona sequence

# Configure logging
logger = logging.getLogger(__name__)

class SocraticDebate:
    def __init__(self, initial_prompt: str, api_key: str,
                 codebase_context: Optional[Dict[str, str]] = None,
                 settings: Optional[ChimeraSettings] = None,
                 all_personas: Optional[Dict[str, PersonaConfig]] = None,
                 persona_sets: Optional[Dict[str, List[str]]] = None,
                 persona_sequence: Optional[List[str]] = None,
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
            persona_sequence: Optional default persona execution order; uses defaults if not provided
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
            self.llm_provider = GeminiProvider(api_key=api_key, model_name=self.model_name)
        except LLMProviderError as e:
            logger.error(f"Failed to initialize LLM provider: {e}")
            # Re-raise as ChimeraError for consistent error handling in the app
            raise ChimeraError(f"LLM provider initialization failed: {e}") from e

        # Calculate token budgets for different phases
        self._calculate_token_budgets()

        self.context_analyzer = context_analyzer # Use the provided analyzer instance
        
        self.all_personas = all_personas or {}
        self.persona_sets = persona_sets or {}
        self.persona_sequence = persona_sequence or []
        self.domain = domain
        
        # Initialize PersonaRouter with all loaded personas
        self.persona_router = PersonaRouter(self.all_personas)
        
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
        try:
            # 1. Initialize state variables
            self._initialize_debate_state()

            # 2. Perform Context Analysis and Determine Persona Sequence
            context_analysis_results = self._perform_context_analysis()
            persona_sequence = self._determine_persona_sequence(context_analysis_results)

            # 3. Process Context Persona Turn (if applicable)
            context_persona_turn_results = self._process_context_persona_turn(persona_sequence, context_analysis_results)
            
            # 4. Execute Debate Persona Turns
            debate_persona_results = self._execute_debate_persona_turns(persona_sequence, context_persona_turn_results)
            
            # 5. Perform Synthesis Persona Turn
            synthesis_persona_results = self._perform_synthesis_persona_turn(persona_sequence, debate_persona_results)
            
            # 6. Finalize Results
            final_answer, intermediate_steps = self._finalize_debate_results(
                context_persona_turn_results, debate_persona_results, synthesis_persona_results
            )
            
            return final_answer, intermediate_steps

        except TokenBudgetExceededError as e:
            logger.error(f"Socratic Debate failed: Token budget exceeded. {e}")
            if self.status_callback:
                self.status_callback(message=f"[red]Socratic Debate Failed: Token Budget Exceeded[/red]", state="error")
            final_answer = {
                "COMMIT_MESSAGE": "Debate Failed - Token Budget Exceeded",
                "RATIONALE": f"The Socratic debate exceeded the allocated token budget. Please consider increasing the budget or simplifying the prompt. Error details: {str(e)}",
                "CODE_CHANGES": [],
                "malformed_blocks": [{"type": "TOKEN_BUDGET_ERROR", "message": str(e), "details": e.details}]
            }
            self._update_intermediate_steps_with_totals()
            return final_answer, self.intermediate_steps
        
        except ChimeraError as e:
            logger.error(f"Socratic Debate failed due to ChimeraError: {e}")
            if self.status_callback:
                self.status_callback(message=f"[red]Socratic Debate Failed: {e}[/red]", state="error")
            final_answer = {
                "COMMIT_MESSAGE": "Debate Failed (Chimera Error)",
                "RATIONALE": f"A Chimera-specific error occurred during the debate: {str(e)}",
                "CODE_CHANGES": [],
                "malformed_blocks": [{"type": "CHIMERA_ERROR", "message": str(e), "details": e.details}]
            }
            self._update_intermediate_steps_with_totals()
            return final_answer, self.intermediate_steps

        except Exception as e:
            logger.error(f"Socratic Debate failed due to an unexpected error: {e}", exc_info=True)
            if self.status_callback:
                self.status_callback(message=f"[red]Socratic Debate Failed: Unexpected Error[/red]", state="error")
            final_answer = {
                "COMMIT_MESSAGE": "Debate Failed (Unexpected Error)",
                "RATIONALE": f"An unexpected error occurred during the Socratic debate: {str(e)}",
                "CODE_CHANGES": [],
                "malformed_blocks": [{"type": "UNEXPECTED_ERROR", "message": str(e), "error_details": {"traceback": traceback.format_exc()}}]
            }
            self._update_intermediate_steps_with_totals()
            return final_answer, self.intermediate_steps

    def _initialize_debate_state(self):
        """Initializes state variables for a new debate run."""
        self.intermediate_steps = {}
        self.final_answer = None
        self.tokens_used_per_phase = {"context": 0, "debate": 0, "synthesis": 0}
        self.tokens_used = self.initial_input_tokens # Start with initial input tokens
        logger.debug("Debate state initialized.")

    def _perform_context_analysis(self) -> Optional[Dict[str, Any]]:
        """Performs context analysis (finding relevant files) if context is available."""
        context_analysis_results = None
        if self.context_analyzer and self.codebase_context:
            try:
                # Determine initial persona sequence for relevance scoring
                initial_sequence_for_relevance = self.persona_router.determine_persona_sequence(self.initial_prompt)
                
                relevant_files_info = self.context_analyzer.find_relevant_files(
                    self.initial_prompt,
                    active_personas=initial_sequence_for_relevance
                )
                context_analysis_results = {"relevant_files": relevant_files_info}
                self.intermediate_steps["Relevant_Files_Context"] = {"relevant_files": relevant_files_info}
                logger.info(f"Context analysis completed. Found {len(relevant_files_info)} relevant files.")
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
        if self.persona_router.is_self_analysis_prompt(self.initial_prompt):
            logger.info("Detected self-analysis prompt. Using standardized specialized persona sequence.")
            base_sequence = SELF_ANALYSIS_PERSONA_SEQUENCE
        else:
            # Accessing protected methods for domain analysis and persona selection
            domains = self.persona_router._analyze_prompt_domain(self.initial_prompt)
            base_sequence = self.persona_router._get_domain_specific_personas(domains)
            
            # Reorder core personas and domain experts
            core_order = ["Visionary_Generator", "Skeptical_Generator"]
            domain_experts = [p for p in base_sequence if p not in core_order and p != "Impartial_Arbitrator"]
            
            base_sequence = core_order + domain_experts
            if "Impartial_Arbitrator" in base_sequence:
                base_sequence.append("Impartial_Arbitrator")
        
        # Apply dynamic adjustments based on context analysis and intermediate results
        adjusted_sequence = self.persona_router._apply_dynamic_adjustment(
            base_sequence,
            self.intermediate_steps,
            self.initial_prompt.lower()
        )
        
        # Further adjustments based on context analysis results (e.g., presence of test files)
        if context_analysis_results and context_analysis_results.get("relevant_files"):
            relevant_files = context_analysis_results["relevant_files"]
            test_file_count = sum(1 for file_path, _ in relevant_files if file_path.startswith('tests/'))
            code_file_count = sum(1 for file_path, _ in relevant_files if file_path.endswith(('.py', '.js', '.ts', '.java', '.go')))
            
            if test_file_count > 3 and "Test_Engineer" not in adjusted_sequence:
                self.persona_router._insert_persona_before_arbitrator(adjusted_sequence, "Test_Engineer")
            
            if code_file_count > 5:
                if "Code_Architect" not in adjusted_sequence:
                    self.persona_router._insert_persona_before_arbitrator(adjusted_sequence, "Code_Architect")
                if "Security_Auditor" not in adjusted_sequence:
                    self.persona_router._insert_persona_before_arbitrator(adjusted_sequence, "Security_Auditor")

        # Ensure uniqueness and order
        seen = set()
        unique_sequence = []
        for persona in adjusted_sequence:
            if persona not in seen:
                unique_sequence.append(persona)
                seen.add(persona)
        
        self.intermediate_steps["Persona_Sequence_Order"] = unique_sequence
        logger.info(f"Final persona sequence determined: {unique_sequence}")
        return unique_sequence

    def _process_context_persona_turn(self, persona_sequence: List[str], context_analysis_results: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """Processes the turn for a dedicated context processing persona."""
        context_processing_persona_name = None
        context_processing_persona_config = None
        
        # Find a persona suitable for initial context processing
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
                turn_results = self._execute_llm_turn(
                    persona_name=context_processing_persona_name,
                    persona_config=context_processing_persona_config,
                    prompt=context_prompt_for_persona,
                    phase="context"
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

        # Determine the personas to run in the debate loop (exclude synthesis persona if it's the last one)
        synthesis_persona_name = None
        if persona_sequence and persona_sequence[-1] in ["Impartial_Arbitrator", "Constructive_Critic"]: # Add other synthesis personas if needed
            synthesis_persona_name = persona_sequence[-1]
        
        debate_personas_to_run = persona_sequence
        if synthesis_persona_name and persona_sequence[-1] == synthesis_persona_name:
            debate_personas_to_run = persona_sequence[:-1] # Exclude the last one for the debate loop

        for i, persona_name in enumerate(debate_personas_to_run):
            if persona_name not in self.all_personas:
                logger.warning(f"Persona '{persona_name}' not found in loaded personas. Skipping.")
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
                turn_results = self._execute_llm_turn(
                    persona_name=persona_name,
                    persona_config=persona_config,
                    prompt=current_persona_prompt,
                    phase="debate"
                )
                if turn_results:
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
        
        return all_debate_turns

    def _perform_synthesis_persona_turn(self, persona_sequence: List[str], debate_persona_results: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """Performs the synthesis turn using the designated synthesis persona."""
        synthesis_persona_name = None
        synthesis_persona_config = None
        
        # Determine the synthesis persona (usually the last one in sequence if designated)
        if persona_sequence and persona_sequence[-1] in ["Impartial_Arbitrator", "Constructive_Critic"]: # Add other synthesis personas if needed
            synthesis_persona_name = persona_sequence[-1]
        # Removed the problematic elif block that incorrectly assigned the last persona as synthesizer.

        if synthesis_persona_name and synthesis_persona_name in self.all_personas:
            synthesis_persona_config = self.all_personas[synthesis_persona_name]
            
            # Construct the synthesis prompt: combine initial prompt and all debate turns
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
            
            try:
                logger.info(f"Performing final synthesis with persona: {synthesis_persona_name}")
                turn_results = self._execute_llm_turn(
                    persona_name=synthesis_persona_name,
                    persona_config=synthesis_persona_config,
                    prompt=synthesis_prompt,
                    phase="synthesis"
                )
                return turn_results
            except TokenBudgetExceededError as e:
                logger.error(f"Token budget exceeded during final synthesis: {e}")
                raise e
            except Exception as e:
                logger.error(f"Error during final synthesis for {synthesis_persona_name}: {e}")
                self.intermediate_steps[f"{synthesis_persona_name}_Error"] = str(e)
                error_tokens = self.llm_provider.count_tokens(f"Error processing {synthesis_persona_name}: {str(e)}") + 50
                self.track_token_usage("synthesis", error_tokens)
                self.check_budget("synthesis", 0, f"Error handling {synthesis_persona_name} synthesis")
        else:
            logger.warning("No synthesis persona found or sequence is empty. Final answer may be incomplete.")
        return None # Return None if no explicit synthesis persona was found or executed.

    def _execute_llm_turn(self, persona_name: str, persona_config: PersonaConfig, prompt: str, phase: str) -> Optional[Dict[str, Any]]:
        """Executes a single LLM turn for a given persona, handling generation, token tracking, and parsing."""
        
        self.check_budget(phase, 0, f"Start of {persona_name} turn") # Check budget before starting the turn

        try:
            response_text, input_tokens, output_tokens = self.llm_provider.generate(
                prompt=prompt,
                system_prompt=persona_config.system_prompt,
                temperature=persona_config.temperature,
                max_tokens=persona_config.max_tokens,
                _status_callback=self.status_callback,
                requested_model_name=self.model_name,
                persona_config=persona_config,
                intermediate_results=self.intermediate_steps
            )
        except TokenBudgetExceededError as e:
            logger.error(f"Token budget exceeded during LLM generation for {persona_name}: {e}")
            raise e # Re-raise to be caught by the main run_debate handler
        except Exception as e:
            logger.error(f"Error during LLM generation for {persona_name}: {e}")
            self.intermediate_steps[f"{persona_name}_Error"] = str(e)
            error_tokens = self.llm_provider.count_tokens(f"Error processing {persona_name}: {str(e)}") + 50
            self.track_token_usage(phase, error_tokens)
            self.check_budget(phase, 0, f"Error handling {persona_name} generation")
            return None # Return None to indicate failure for this turn

        turn_tokens_used = input_tokens + output_tokens
        self.track_token_usage(phase, turn_tokens_used)
        self.check_budget(phase, 0, f"End of {persona_name} turn") # Check budget after the turn

        # Parse and Validate Output
        parsed_output_data = {}
        try:
            parsed_output_data = LLMOutputParser().parse_and_validate(response_text, LLMOutput)
        except Exception as e:
            logger.error(f"Failed to parse/validate output for {persona_name} against LLMOutput schema: {e}")
            parsed_output_data = {
                "COMMIT_MESSAGE": f"Parsing/Validation Error for {persona_name}",
                "RATIONALE": f"Error processing output from {persona_name}: {str(e)}",
                "CODE_CHANGES": [],
                "malformed_blocks": [{"type": "PARSING_OR_VALIDATION_ERROR", "message": str(e), "raw_output": response_text[:500]}]
            }
            self.intermediate_steps[f"{persona_name}_Error"] = str(e)
        
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
        """Aggregates results from all phases and finalizes the output."""
        
        # Determine the final answer
        if synthesis_persona_results and synthesis_persona_results.get("output"):
            self.final_answer = synthesis_persona_results["output"]
        elif debate_persona_results: # Fallback to last debate turn if no synthesis
            self.final_answer = debate_persona_results[-1]["output"]
        elif context_persona_turn_results: # Fallback to context turn if no debate/synthesis
            self.final_answer = context_persona_turn_results["output"]
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
            self.final_answer = {
                "COMMIT_MESSAGE": "Debate Failed - Final Answer Malformed",
                "RATIONALE": f"The final answer was not a valid dictionary. Type received: {type(self.final_answer).__name__}",
                "CODE_CHANGES": [],
                "malformed_blocks": [{"type": "FINAL_ANSWER_MALFORMED", "message": f"Expected dict, got {type(self.final_answer).__name__}", "raw_output": str(self.final_answer)[:500]}]
            }
            logger.error(f"Final answer was not a dictionary. Type: {type(self.final_answer).__name__}")

        # Update intermediate steps with totals
        self._update_intermediate_steps_with_totals()
        
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
        for key, value in list(self.intermediate_steps.items()): # Iterate over a copy
            if isinstance(value, dict) and "Error" in key:
                pass # Already added as a dict
            elif isinstance(value, str) and "Error" in key:
                pass # Already added as a string
            elif "Error" in key: # Catch other types of errors logged as keys
                self.intermediate_steps[key] = {"error": str(value)}