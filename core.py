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
import uuid

# --- IMPORT MODIFICATIONS ---
from llm_provider import GeminiProvider
from src.context.context_analyzer import ContextRelevanceAnalyzer
from src.persona.routing import PersonaRouter
from src.utils.output_parser import LLMOutputParser
from src.models import PersonaConfig, ReasoningFrameworkConfig, LLMOutput, CodeChange, ContextAnalysisOutput, CritiqueOutput, GeneralOutput, ConflictReport, SelfImprovementAnalysisOutput 
from src.config.settings import ChimeraSettings
from src.exceptions import ChimeraError, LLMResponseValidationError, SchemaValidationError, TokenBudgetExceededError, LLMProviderError, CircuitBreakerError
from src.constants import SELF_ANALYSIS_KEYWORDS, is_self_analysis_prompt 
from src.logging_config import setup_structured_logging
from src.utils.error_handler import handle_errors
from src.persona_manager import PersonaManager 
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
        "Self_Improvement_Analyst": SelfImprovementAnalysisOutput, 
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
        self.logger = logging.getLogger(self.__class__.__name__)
        
        self.settings = settings or ChimeraSettings()
        self.max_total_tokens_budget = self.settings.total_budget
        self.tokens_used = 0
        self.model_name = model_name
        self.status_callback = status_callback
        self.rich_console = rich_console or Console(stderr=True) 
        self.is_self_analysis = is_self_analysis
        
        self.request_id = str(uuid.uuid4())[:8]
        self._log_extra = {"request_id": self.request_id}
        
        self.initial_prompt = initial_prompt
        self.codebase_context = codebase_context
        self.domain = domain 

        # Initialize the LLM provider.
        try:
            self.llm_provider = GeminiProvider(api_key=api_key, model_name=self.model_name, rich_console=self.rich_console, request_id=self.request_id)
        except LLMProviderError as e:
            self._log_with_context("error", f"Failed to initialize LLM provider: {e.message}", exc_info=True, details=e.details, original_exception=e)
            raise ChimeraError(f"LLM provider initialization failed: {e.message}", original_exception=e) from e
        except Exception as e:
            self._log_with_context("error", f"An unexpected error occurred during LLM provider initialization: {e}", exc_info=True, original_exception=e)
            raise ChimeraError(f"LLM provider initialization failed unexpectedly: {e}", original_exception=e) from e

        try:
            self.tokenizer = self.llm_provider.tokenizer
        except AttributeError:
            raise ChimeraError("LLM provider tokenizer is not available.")

        # NEW: Store PersonaManager instance
        self.persona_manager = persona_manager
        if not self.persona_manager:
            self.logger.warning("PersonaManager instance not provided to SocraticDebate. Initializing a new one. This might affect state persistence in UI.")
            # This fallback should ideally pass DOMAIN_KEYWORDS, but for core.py, it's not directly available.
            # Assuming PersonaManager's default init handles this or it's always passed from app.py.
            self.persona_manager = PersonaManager({}) # Pass empty dict as fallback for domain_keywords
            self.all_personas = self.persona_manager.all_personas 
            self.persona_sets = self.persona_manager.persona_sets 
        else:
            self.all_personas = self.persona_manager.all_personas
            self.persona_sets = self.persona_manager.persona_sets

        # Initialize PersonaRouter with all loaded personas AND persona_sets
        # Ensure PersonaRouter gets the PromptAnalyzer from PersonaManager
        self.persona_router = self.persona_manager.persona_router
        if not self.persona_router: # Fallback if PersonaManager didn't set it up
             self.logger.warning("PersonaRouter not found in PersonaManager. Initializing a new one. This might indicate an issue in PersonaManager setup.")
             self.persona_router = PersonaRouter(self.all_personas, self.persona_sets, self.persona_manager.prompt_analyzer)
        
        # Store the context analyzer instance and ensure it has codebase_context
        self.context_analyzer = context_analyzer
        if self.context_analyzer and self.codebase_context and not self.context_analyzer.codebase_context:
            self.context_analyzer.codebase_context = self.codebase_context
        # Ensure context analyzer has the persona router if it wasn't set during its init
        if self.context_analyzer and not self.context_analyzer.persona_router:
            self.context_analyzer.set_persona_router(self.persona_router)

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
                    self.logger.error(f"Failed to compute embeddings for codebase context: {e}", exc_info=True)
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
            completed_debate_personas = sum(1 for k in self.intermediate_steps if k.endswith("_Output") and k != "Final_Synthesis_Output")
            
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
            # FIX: Convert persona_sequence to a tuple for lru_cache compatibility
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
        
        persona_turn_budgets = {}
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
            output = self._execute_llm_turn("Context_Aware_Assistant", persona_config, prompt, "debate")
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
        previous_output = self.initial_prompt
        
        # Store the full context analysis output for later persona-specific extraction
        full_context_analysis_output = self.intermediate_steps.get("Context_Aware_Assistant_Output")

        if context_persona_turn_results:
            previous_output = f"Initial Prompt: {self.initial_prompt}\n\nContext Analysis:\n{json.dumps(context_persona_turn_results, indent=2)}"

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
                f"Executing: [bold]{persona_name.replace('_', ' ')}[/bold]...",
                "running",
                self.tokens_used,
                self.get_total_estimated_cost(),
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
            if full_context_analysis_output:
                if persona_name == "Security_Auditor" and full_context_analysis_output.get("security_summary"):
                    persona_specific_context_str = f"Security Context Summary:\n{json.dumps(full_context_analysis_output['security_summary'], indent=2)}"
                elif persona_name == "Code_Architect" and full_context_analysis_output.get("architecture_summary"):
                    persona_specific_context_str = f"Architecture Context Summary:\n{json.dumps(full_context_analysis_output['architecture_summary'], indent=2)}"
                elif persona_name == "DevOps_Engineer" and full_context_analysis_output.get("devops_summary"):
                    persona_specific_context_str = f"DevOps Context Summary:\n{json.dumps(full_context_analysis_output['devops_summary'], indent=2)}"
                elif persona_name == "Test_Engineer" and full_context_analysis_output.get("testing_summary"):
                    persona_specific_context_str = f"Testing Context Summary:\n{json.dumps(full_context_analysis_output['testing_summary'], indent=2)}"
                elif full_context_analysis_output.get("general_overview"):
                    persona_specific_context_str = f"General Codebase Overview:\n{full_context_analysis_output['general_overview']}"
            
            # Construct the current prompt with the persona-specific context
            current_prompt = f"Initial Problem: {self.initial_prompt}\n\n"
            if persona_specific_context_str:
                current_prompt += f"Relevant Code Context:\n{persona_specific_context_str}\n\n"
            current_prompt += f"Previous Debate Output:\n{json.dumps(previous_output, indent=2) if isinstance(previous_output, dict) else previous_output}"
            
            # Use the dynamically calculated budget for this persona's turn
            max_output_tokens_for_turn = self.phase_budgets.get("persona_turn_budgets", {}).get(persona_name, persona_config.max_tokens)
            estimated_tokens = self.tokenizer.count_tokens(current_prompt) + max_output_tokens_for_turn
            self.check_budget("debate", estimated_tokens, persona_name)

            try:
                output = self._execute_llm_turn(persona_name, persona_config, current_prompt, "debate")
                
                # NEW: Content Alignment Validation
                is_aligned, validation_message = self.content_validator.validate(persona_name, output)
                if not is_aligned:
                    self._log_with_context("warning", f"Content misalignment detected for {persona_name}: {validation_message}",
                                           persona=persona_name, validation_message=validation_message)
                    # Add a malformed block to indicate content drift
                    self.intermediate_steps.setdefault("malformed_blocks", []).append({
                        "type": "CONTENT_MISALIGNMENT",
                        "message": f"Output from {persona_name} drifted from the core topic: {validation_message}",
                        "persona": persona_name,
                        "raw_string_snippet": str(output)[:500]
                    })
                    # Optionally, modify the output to reflect the issue, or even trigger a re-prompt
                    # For now, we'll just log and add a malformed block.
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
                                previous_output = {"status": "conflict_resolved", "resolution": resolution_result["resolution_summary"]}
                                self.intermediate_steps["Conflict_Resolution_Attempt"] = resolution_result
                                self.intermediate_steps["Unresolved_Conflict"] = None 
                            else:
                                # If not resolved, keep the conflict in intermediate steps
                                self.intermediate_steps["Unresolved_Conflict"] = conflict_report.model_dump()
                                previous_output = {"status": "conflict_unresolved", "conflict_report": conflict_report.model_dump()}
                        else:
                            # No conflict found, proceed normally
                            self._log_with_context("info", f"Devils_Advocate reported no conflict: {conflict_report.summary}")
                            self.intermediate_steps["Unresolved_Conflict"] = None 
                            self.intermediate_steps["Conflict_Resolution_Attempt"] = None 
                            previous_output = {"status": "no_conflict_reported", "summary": conflict_report.summary}
                    except ValidationError:
                        pass 
                
                debate_history.append({"persona": persona_name, "output": output})
                previous_output = output 
            except Exception as e:
                self._log_with_context("error", f"Error during {persona_name} turn: {e}", persona=persona_name, exc_info=True, original_exception=e)
                self.rich_console.print(f"[red]Error during {persona_name} turn: {e}[/red]")
                previous_output = {"error": f"Turn failed for {persona_name}: {str(e)}", "malformed_blocks": [{"type": "DEBATE_TURN_ERROR", "message": str(e)}]}
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
        if conflict_report.conflict_type in ["RESOURCE_CONSTRAINT", "SECURITY_VS_PERFORMANCE"] or \
           "Impartial_Arbitrator" in self.all_personas: # Always include Arbitrator if available for final synthesis
            sub_debate_personas_set.add("Impartial_Arbitrator")

        # Ensure the synthesis persona for the main debate is also considered if it's not the Arbitrator
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
            self.status_callback(f"Sub-Debate: [bold]{persona_name.replace('_', ' ')}[/bold] resolving conflict...",
                                 "running", self.tokens_used, self.get_total_estimated_cost(),
                                 progress_pct=self.get_progress_pct("debate"), current_persona_name=persona_name)
            
            # Get potentially adjusted persona config from PersonaManager
            persona_config = self.persona_manager.get_adjusted_persona_config(persona_name)

            # Temporarily adjust max_tokens for sub-debate to fit budget
            original_max_tokens = persona_config.max_tokens
            persona_config.max_tokens = min(original_max_tokens, sub_debate_budget // len(sub_debate_personas))
            
            try:
                # Pass the full debate history as context for the sub-debate
                turn_output = self._execute_llm_turn(
                    persona_name, persona_config, sub_debate_prompt, "sub_debate_phase"
                )
                resolution_attempts.append({"persona": persona_name, "output": turn_output})
            except Exception as e:
                self._log_with_context("error", f"Sub-debate turn for {persona_name} failed: {e}")
            finally:
                persona_config.max_tokens = original_max_tokens 

        # Synthesize sub-debate results into a final resolution
        if resolution_attempts:
            final_resolution_prompt = f"""
            Synthesize the following sub-debate attempts to resolve a conflict into a single, clear resolution.
            Conflict: {conflict_report.summary}
            Sub-debate results: {json.dumps(resolution_attempts, indent=2)}
            
            Provide a final resolution and its rationale.
            """
            synthesizer_persona = self.all_personas.get("Impartial_Arbitrator") or self.all_personas.get("General_Synthesizer")
            if synthesizer_persona:
                try:
                    final_resolution = self._execute_llm_turn(
                        synthesizer_persona.name, synthesizer_persona, final_resolution_prompt, "sub_debate_synthesis"
                    )
                    self._log_with_context("info", "Conflict successfully resolved.")
                    return {"conflict_resolved": True, "resolution_summary": final_resolution}
                except Exception as e:
                    self._log_with_context("error", f"Failed to synthesize conflict resolution: {e}")
        
        self._log_with_context("warning", "Conflict could not be resolved in sub-debate.")
        return None

    # --- NEW HELPER FUNCTIONS FOR SELF_IMPROVEMENT_ANALYST PROMPT TRUNCATION ---
    def _summarize_metrics_for_llm(self, metrics: Dict[str, Any], max_tokens: int) -> Dict[str, Any]:
        """
        Intelligently summarizes the metrics dictionary to fit within a token budget.
        Prioritizes high-level summaries and truncates verbose lists like 'detailed_issues'.
        """
        summarized_metrics = json.loads(json.dumps(metrics)) 
        
        # Estimate current token count
        current_tokens = self.tokenizer.count_tokens(json.dumps(summarized_metrics))
        
        # If already within budget, return as is
        if current_tokens <= max_tokens:
            return summarized_metrics

        self._log_with_context("debug", f"Summarizing metrics for LLM. Current tokens: {current_tokens}, Max: {max_tokens}")

        # Strategy: Aggressively truncate 'detailed_issues' first
        if 'code_quality' in summarized_metrics and 'detailed_issues' in summarized_metrics['code_quality']:
            original_issue_count = len(summarized_metrics['code_quality']['detailed_issues'])
            if original_issue_count > 10: 
                summarized_metrics['code_quality']['detailed_issues'] = summarized_metrics['code_quality']['detailed_issues'][:10]
                self._log_with_context("debug", f"Truncated detailed_issues from {original_issue_count} to 10.")
                current_tokens = self.tokenizer.count_tokens(json.dumps(summarized_metrics))
                if current_tokens <= max_tokens:
                    return summarized_metrics

        # Further truncation: Remove less critical detailed metrics if still over budget
        if 'code_quality' in summarized_metrics and 'complexity_metrics' in summarized_metrics['code_quality']:
            pass 

        # If still over budget, consider removing entire verbose sections
        if current_tokens > max_tokens:
            if 'code_quality' in summarized_metrics:
                # Remove detailed_issues entirely if still too large
                if 'detailed_issues' in summarized_metrics['code_quality']:
                    del summarized_metrics['code_quality']['detailed_issues']
                    self._log_with_context("debug", "Removed detailed_issues entirely.")
                    current_tokens = self.tokenizer.count_tokens(json.dumps(summarized_metrics))
                    if current_tokens <= max_tokens:
                        return summarized_metrics
            
            if 'maintainability' in summarized_metrics and 'test_coverage_summary' in summarized_metrics['maintainability']:
                # Keep only overall_coverage_percentage if present
                if 'coverage_details' in summarized_metrics['maintainability']['test_coverage_summary']:
                    del summarized_metrics['maintainability']['test_coverage_summary']['coverage_details']
                    self._log_with_context("debug", "Removed coverage_details.")
                    current_tokens = self.tokenizer.count_tokens(json.dumps(summarized_metrics))
                    if current_tokens <= max_tokens:
                        return summarized_metrics

        # As a last resort, convert to a very high-level summary string
        if current_tokens > max_tokens:
            self._log_with_context("warning", "Metrics still too large after truncation. Converting to high-level summary string.")
            summary_str = f"Overall Code Quality: PEP8 issues: {metrics['code_quality']['pep8_issues_count']}, Code Smells: {metrics['code_quality']['code_smells_count']}. " \
                          f"Security Issues: Bandit: {metrics['security']['bandit_issues_count']}, AST: {metrics['security']['ast_security_issues_count']}. " \
                          f"Token Usage: {metrics['performance_efficiency']['token_usage_stats']['total_tokens']} tokens, Cost: ${metrics['performance_efficiency']['token_usage_stats']['total_cost_usd']:.4f}. " \
                          f"Robustness: Schema failures: {metrics['robustness']['schema_validation_failures_count']}, Unresolved conflicts: {metrics['robustness']['unresolved_conflict_present']}."
            return {"summary_string": summary_str}

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
                elif 'architecturalAnalysis' in turn_copy['output']:
                    turn_copy['output'] = {'overallAssessment': turn_copy['output']['architecturalAnalysis']['overallAssessment']}
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

    def _perform_synthesis_persona_turn(self, persona_sequence: List[str], debate_persona_results: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """
        Executes the final synthesis persona turn (Impartial_Arbitrator or General_Synthesizer or Self_Improvement_Analyst).
        """
        synthesis_persona_name = None
        if self.is_self_analysis: 
            synthesis_persona_name = "Self_Improvement_Analyst"
        elif "Impartial_Arbitrator" in persona_sequence:
            synthesis_persona_name = "Impartial_Arbitrator"
        elif "General_Synthesizer" in persona_sequence:
            synthesis_persona_name = "General_Synthesizer"
        
        if not synthesis_persona_name:
            self._log_with_context("error", "No synthesis persona (Impartial_Arbitrator, General_Synthesizer, or Self_Improvement_Analyst) found in sequence.")
            return {"error": "No synthesis persona found."}

        self._log_with_context("info", f"Executing final synthesis turn for persona: {synthesis_persona_name}")
        persona_config = self.persona_manager.get_adjusted_persona_config(synthesis_persona_name)
        if not persona_config: 
            persona_config = self.all_personas.get(synthesis_persona_name)
            if not persona_config:
                self._log_with_context("error", f"Synthesis persona configuration not found for {synthesis_persona_name}.")
                return {"error": f"{synthesis_persona_name} config missing."}

        final_synthesis_prompt_content = ""
        if synthesis_persona_name == "Self_Improvement_Analyst":
            try:
                from src.utils.path_utils import PROJECT_ROOT
                project_root_path = str(PROJECT_ROOT)
            except ImportError:
                self.logger.warning("Could not import PROJECT_ROOT for self-improvement metrics. Using current working directory.")
                project_root_path = os.getcwd()
            
            try:
                self_improvement_metrics = ImprovementMetricsCollector.collect_all_metrics(
                    project_root_path, self.intermediate_steps
                )
                self.intermediate_steps["Self_Improvement_Metrics"] = self_improvement_metrics
                
                # --- START MODIFICATION FOR SELF_IMPROVEMENT_ANALYST PROMPT GENERATION ---
                metrics = self_improvement_metrics 
                debate_history_full = debate_persona_results 

                # Calculate token budget for the prompt content (excluding system prompt and max_output_tokens)
                # Allocate a portion of the synthesis budget for the prompt content
                # Use a slightly lower ratio for prompt content to leave more room for LLM output
                # MODIFIED: Changed 0.6 to 0.5 to give more room for LLM output
                prompt_content_budget = int(self.phase_budgets["synthesis"] * 0.5) 
                
                # Summarize metrics and debate history to fit the budget
                # Allocate 60% of prompt_content_budget to metrics, 40% to debate history
                summarized_metrics = self._summarize_metrics_for_llm(metrics, int(prompt_content_budget * 0.6))
                summarized_debate_history = self._summarize_debate_history_for_llm(debate_history_full, int(prompt_content_budget * 0.4))

                normalized_impact_scores = {}
                
                # Define reasonable maximums for normalization (illustrative, can be refined)
                MAX_TOKENS_FOR_NORM = 100000 
                MAX_SCHEMA_FAILURES_FOR_NORM = 10
                MAX_CONFLICT_ATTEMPTS_FOR_NORM = 3
                MAX_CODE_SMELLS_FOR_NORM = 50
                MAX_SECURITY_ISSUES_FOR_NORM = 20
                CRITICAL_AREA_THRESHOLD = 30 

                # Efficiency: Higher token usage -> higher impact score
                normalized_impact_scores['Efficiency'] = min(100, (metrics.get('performance_efficiency', {}).get('token_usage_stats', {}).get('total_tokens', 0) / MAX_TOKENS_FOR_NORM) * 100)
                
                # Robustness: Higher schema failures/unresolved conflicts -> higher impact score
                schema_failures = metrics.get('robustness', {}).get('schema_validation_failures_count', 0)
                unresolved_conflict_score = 50 if metrics.get('robustness', {}).get('unresolved_conflict_present', False) else 0
                normalized_impact_scores['Robustness'] = min(100, ((schema_failures / MAX_SCHEMA_FAILURES_FOR_NORM) * 100) + unresolved_conflict_score)
                
                # Reasoning Quality: Higher conflict resolution attempts (implies issues) -> higher impact score
                conflict_attempts = metrics.get('performance_efficiency', {}).get('debate_efficiency_summary', {}).get('conflict_resolution_attempts', 0)
                normalized_impact_scores['Reasoning Quality'] = min(100, (conflict_attempts / MAX_CONFLICT_ATTEMPTS_FOR_NORM) * 100)
                
                # Maintainability: Higher code smells -> higher impact score
                code_smells = metrics.get('code_quality', {}).get('code_smells_count', 0)
                normalized_impact_scores['Maintainability'] = min(100, (code_smells / MAX_CODE_SMELLS_FOR_NORM) * 100)

                # Security: Higher bandit/AST issues -> higher impact score
                bandit_issues = metrics.get('security', {}).get('bandit_issues_count', 0)
                ast_issues = metrics.get('security', {}).get('ast_security_issues_count', 0)
                normalized_impact_scores['Security'] = min(100, ((bandit_issues + ast_issues) / MAX_SECURITY_ISSUES_FOR_NORM) * 100)
                
                # Sort by impact (higher = more critical)
                sorted_impact = sorted(normalized_impact_scores.items(), key=lambda x: x[1], reverse=True)[:5] 
                
                # Select top N areas, but always include critical ones if their score is above a minimum threshold
                top_areas = []
                for area, score in sorted_impact:
                    if len(top_areas) < 2 or score >= CRITICAL_AREA_THRESHOLD:
                        top_areas.append(area)
                    # Limit to a reasonable number of areas to avoid overwhelming the LLM
                    if len(top_areas) >= 4: 
                        break
                
                # Ensure unique areas (in case critical_area_threshold added duplicates)
                top_areas = list(dict.fromkeys(top_areas))
                
                final_synthesis_prompt_content = f"""
                Analyze the Project Chimera codebase focusing PRIMARILY on these areas: {', '.join(top_areas)}. Provide concrete, specific suggestions for code changes or process adjustments, backed by the provided metrics.
                
                ## Objective Metrics:
                {json.dumps(summarized_metrics, indent=2)}
                
                ## Debate History:
                {json.dumps(summarized_debate_history, indent=2)}
                
                CRITICAL REMINDER FOR CODE_CHANGES_SUGGESTED:
                - For 'MODIFY' actions, always provide the `DIFF_CONTENT` field in standard unified diff format (lines starting with `+`, `-`, or ` `). This is crucial for clarity and token efficiency.
                - Only provide `FULL_CONTENT` for 'ADD' actions, or for very small 'MODIFY' actions (e.g., < 10 lines) where a full content replacement is simpler than a diff.
                - Ensure all code snippets within `CODE_CHANGES_SUGGESTED` adhere to PEP8 (line length <= 88).
                - Prioritize minimal, focused changes.
                """
                # --- END MODIFICATION ---

            except Exception as e:
                self._log_with_context("error", f"Failed to collect self-improvement metrics: {e}")
                final_synthesis_prompt_content = f"""
                Analyze the Project Chimera codebase and the recent debate process.
                (Note: Failed to collect automated metrics: {e})
                
                ## Debate History:
                {json.dumps(debate_persona_results, indent=2)}
                
                Your task is to identify the most impactful areas for self-improvement across reasoning quality, robustness, efficiency, and developer maintainability. Provide concrete, specific suggestions for code changes or process adjustments.
                """
        else:
            full_debate_context = {
                "initial_prompt": self.initial_prompt,
                "debate_history": debate_persona_results
            }
            final_synthesis_prompt_content = f"Synthesize the following debate results into a coherent final answer, adhering strictly to your JSON schema:\\n\\n{json.dumps(full_debate_context, indent=2)}"

        prompt = final_synthesis_prompt_content 
        
        # Use the dynamically calculated budget for this persona's turn
        max_output_tokens_for_turn = self.phase_budgets.get("persona_turn_budgets", {}).get(synthesis_persona_name, persona_config.max_tokens)
        estimated_tokens = self.tokenizer.count_tokens(prompt) + max_output_tokens_for_turn
        self.check_budget("synthesis", estimated_tokens, synthesis_persona_name)

        try:
            output = self._execute_llm_turn(synthesis_persona_name, persona_config, prompt, "synthesis")
            self._log_with_context("info", f"Final synthesis turn completed by {synthesis_persona_name}.")
            return output
        except Exception as e:
            self._log_with_context("error", f"Error during final synthesis turn by {synthesis_persona_name}: {e}", exc_info=True, original_exception=e)
            self.rich_console.print(f"[red]Error during final synthesis turn: {e}[/red]")
            if isinstance(e, (TokenBudgetExceededError, ChimeraError, CircuitBreakerError, LLMProviderError, SchemaValidationError)):
                raise e
            return {"error": f"Synthesis turn failed: {e}", "malformed_blocks": [{"type": "SYNTHESIS_ERROR", "message": str(e)}]}

    def _execute_llm_turn(self, persona_name: str, persona_config: PersonaConfig, prompt: str, phase: str) -> Any:
        """
        Executes a single LLM turn for a given persona, handling parsing and validation.
        Includes specific error handling for SchemaValidationError to trigger circuit breaker.
        """
        is_truncated = False
        has_schema_error = False
        
        # Use the persona_config passed to this method, which is already adjusted by PersonaManager
        adjusted_persona_config = persona_config 

        # --- START MODIFICATION: Add retry loop for SchemaValidationError ---
        max_schema_retries = 2 
        original_system_prompt = adjusted_persona_config.system_prompt 
        original_max_tokens = adjusted_persona_config.max_tokens 

        for retry_attempt in range(max_schema_retries + 1):
            try:
                # Use the dynamically calculated budget for this persona's turn
                max_output_tokens_for_turn = self.phase_budgets.get("persona_turn_budgets", {}).get(persona_name, persona_config.max_tokens)
                
                raw_llm_output, input_tokens, output_tokens = self.llm_provider.generate(
                    prompt=prompt,
                    system_prompt=adjusted_persona_config.system_prompt, 
                    temperature=adjusted_persona_config.temperature,     
                    max_tokens=max_output_tokens_for_turn, # Use the dynamically calculated budget
                    persona_config=adjusted_persona_config, 
                    intermediate_results=self.intermediate_steps,
                    requested_model_name=self.model_name
                )
                self.track_token_usage(phase, input_tokens + output_tokens)
                self.check_budget(phase, input_tokens + output_tokens, persona_name)
                
                # NEW: Store actual parameters used for this turn
                self.intermediate_steps[f"{persona_name}_Actual_Temperature"] = adjusted_persona_config.temperature
                self.intermediate_steps[f"{persona_name}_Actual_Max_Tokens"] = adjusted_persona_config.max_tokens
                # --- END MODIFICATION ---

                # Check for truncation (heuristic, can be improved with LLM-specific signals)
                if output_tokens >= max_output_tokens_for_turn * 0.95: # Use max_output_tokens_for_turn
                    is_truncated = True

                if persona_name in self.PERSONA_OUTPUT_SCHEMAS:
                    schema_model = self.PERSONA_OUTPUT_SCHEMAS[persona_name]
                    parser = LLMOutputParser()
                    parsed_output = parser.parse_and_validate(raw_llm_output, schema_model)
                    
                    if parsed_output.get("malformed_blocks"):
                        self._log_with_context("warning", f"LLM output for {persona_name} contained malformed blocks.",
                                               persona=persona_name, malformed_blocks=parsed_output["malformed_blocks"])
                        self.intermediate_steps.setdefault("malformed_blocks", []).extend(parsed_output["malformed_blocks"])
                    
                    # Check for specific error types from the parser indicating schema adherence issues
                    if parsed_output.get("error_type") == "LLM_OUTPUT_MALFORMED" or \
                       any(block.get("type") in ["JSON_EXTRACTION_FAILED", "JSON_DECODE_ERROR", "INVALID_JSON_STRUCTURE", "SCHEMA_VALIDATION_ERROR"] for block in parsed_output.get("malformed_blocks", [])):
                        has_schema_error = True
                        # If this is a retry, and it still fails, re-raise. Otherwise, try again.
                        if retry_attempt < max_schema_retries:
                            self._log_with_context("warning", f"Schema validation failed for {persona_name} (attempt {retry_attempt+1}/{max_schema_retries+1}). Retrying with stricter JSON prompt.",
                                                   persona=persona_name, malformed_blocks=parsed_output["malformed_blocks"])
                            # Modify system_prompt for retry to emphasize strict JSON adherence
                            adjusted_persona_config.system_prompt = original_system_prompt + "\n\nCRITICAL: Your output MUST be a single, valid JSON object. DO NOT include any conversational text or markdown fences outside the JSON. STRICTLY adhere to the schema."
                            # Also, reduce max_tokens to force conciseness, which might help with truncation
                            adjusted_persona_config.max_tokens = max(512, int(original_max_tokens * 0.75)) 
                            continue 
                        else:
                            # Max retries reached, re-raise the SchemaValidationError
                            raise SchemaValidationError( 
                                error_type="LLM_OUTPUT_MALFORMED",
                                field_path="N/A",
                                invalid_value=raw_llm_output[:500],
                                details={"persona": persona_name, "raw_output_snippet": raw_llm_output[:500], "malformed_blocks": parsed_output.get("malformed_blocks", [])}
                            )
                    
                    # If we reach here, parsing and validation were successful for this attempt
                    self.persona_manager.record_persona_performance(persona_name, True, is_truncated, has_schema_error)
                    return parsed_output
                else:
                    self._log_with_context("info", f"Persona {persona_name} is not configured for structured JSON output. Returning raw text.", persona=persona_name)
                    # Record success for persona performance (even if raw text)
                    self.persona_manager.record_persona_performance(persona_name, True, is_truncated, has_schema_error)
                    return raw_llm_output

            except (CircuitBreakerError, TokenBudgetExceededError, LLMProviderError) as e:
                # These are not schema validation errors, re-raise immediately
                if isinstance(e, TokenBudgetExceededError) and "tokens_needed" in e.details and e.details["tokens_needed"] > max_output_tokens_for_turn: # Use max_output_tokens_for_turn
                    is_truncated = True
                self.persona_manager.record_persona_performance(persona_name, False, is_truncated, has_schema_error)
                raise e
            except Exception as e:
                # Catch all other unexpected errors, record failure, and re-raise as ChimeraError
                self.persona_manager.record_persona_performance(persona_name, False, is_truncated, has_schema_error)
                raise ChimeraError(f"An unexpected error occurred during {persona_name}'s turn: {e}",
                                   details={"persona": persona_name, "traceback": traceback.format_exc()}, original_exception=e) from e
        raise ChimeraError(f"Exited schema validation retry loop for {persona_name} without a valid output or re-raising an error.")
        # --- END MODIFICATION ---

    def _finalize_debate_results(self, context_persona_turn_results: Optional[Dict[str, Any]], debate_persona_results: List[Dict[str, Any]], synthesis_persona_results: Optional[Dict[str, Any]]) -> Tuple[Any, Dict[str, Any]]:
        """
        Synthesizes the final answer and prepares the intermediate steps for display.
        """
        final_answer = synthesis_persona_results
        
        if not isinstance(final_answer, dict):
            final_answer = {"general_output": str(final_answer), "malformed_blocks": []}
        if "malformed_blocks" not in final_answer:
            final_answer["malformed_blocks"] = []

        self._update_intermediate_steps_with_totals()
        if "malformed_blocks" not in self.intermediate_steps:
            self.intermediate_steps["malformed_blocks"] = []
        
        # Ensure final_answer reflects resolution or unresolved conflict
        if self.intermediate_steps.get("Conflict_Resolution_Attempt"):
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

    @handle_errors(log_level="ERROR") 
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
        self.status_callback("Phase 1: Analyzing Context...", "running", self.tokens_used, self.get_total_estimated_cost(), progress_pct=self.get_progress_pct("context"))
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