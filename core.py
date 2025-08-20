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
from typing import List, Dict, Tuple, Any, Callable, Optional, Type
import numpy as np # NEW IMPORT for PersonaRouter semantic embeddings
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
from src.models import PersonaConfig, ReasoningFrameworkConfig, LLMOutput, CodeChange, ContextAnalysisOutput, CritiqueOutput, GeneralOutput
from src.config.settings import ChimeraSettings
from src.exceptions import ChimeraError, LLMResponseValidationError, SchemaValidationError, TokenBudgetExceededError, LLMProviderError, CircuitBreakerError
from src.constants import SELF_ANALYSIS_KEYWORDS
from src.logging_config import setup_structured_logging
from src.utils.error_handler import handle_errors # NEW IMPORT for comprehensive error handling

# Configure logging for the core module itself
logger = logging.getLogger(__name__)

class SocraticDebate:
    PERSONA_OUTPUT_SCHEMAS = {
        "Impartial_Arbitrator": LLMOutput,
        "Context_Aware_Assistant": ContextAnalysisOutput,
        "Constructive_Critic": CritiqueOutput,
        "General_Synthesizer": GeneralOutput,
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
                 is_self_analysis: bool = False
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

        self.all_personas = all_personas or {}
        self.persona_sets = persona_sets or {}
        self.domain = domain
        
        # Initialize PersonaRouter with all loaded personas AND persona_sets
        self.persona_router = PersonaRouter(self.all_personas, self.persona_sets)
        
        # Store the context analyzer instance and ensure it has codebase_context
        self.context_analyzer = context_analyzer
        if self.context_analyzer and self.codebase_context and not self.context_analyzer.codebase_context:
            self.context_analyzer.codebase_context = self.codebase_context

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
        """Helper to add request context to all logs using the class-specific logger."""
        exc_info = kwargs.pop('exc_info', None)
        original_exception = kwargs.pop('original_exception', None) # Extract original_exception
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
        elif original_exception is not None: # Log original exception details if provided
            log_data['original_exception_type'] = type(original_exception).__name__
            log_data['original_exception_message'] = str(original_exception)
            logger_method(message, extra=log_data)
        else:
            logger_method(message, extra=log_data)

    def _calculate_token_budgets(self):
        """Calculates token budgets for different phases based on context, model limits, and prompt type."""
        try:
            # Analyze prompt complexity
            prompt_complexity = self._analyze_prompt_complexity(self.initial_prompt)
            
            # Adjust ratios based on complexity
            debate_ratio = self.settings.debate_token_budget_ratio
            synthesis_ratio = self.settings.synthesis_token_budget_ratio

            if prompt_complexity == "high":
                debate_ratio = min(self.settings.debate_token_budget_ratio * 1.1, 0.9)
                synthesis_ratio = max(self.settings.synthesis_token_budget_ratio * 0.9, 0.05)
            elif prompt_complexity == "medium":
                pass # Use default ratios
            else:  # low complexity
                debate_ratio = max(self.settings.debate_token_budget_ratio * 0.9, 0.5)
                synthesis_ratio = min(self.settings.synthesis_token_budget_ratio * 1.1, 0.3)
            
            # Further adjust for self-analysis prompts
            if self.is_self_analysis:
                debate_ratio = max(debate_ratio, self.settings.self_analysis_debate_ratio)
                synthesis_ratio = min(synthesis_ratio, 1.0 - debate_ratio)
            
            # Normalize to ensure sum is 1.0 for debate and synthesis
            total_dynamic_ratio = debate_ratio + synthesis_ratio
            if total_dynamic_ratio > 0:
                debate_ratio /= total_dynamic_ratio
                synthesis_ratio /= total_dynamic_ratio
            else: # Fallback if ratios are zero
                debate_ratio = 0.8
                synthesis_ratio = 0.2

            # Estimate tokens for context and initial input
            context_str = self.context_analyzer.get_context_summary() if self.context_analyzer else ""
            self.initial_input_tokens = self.tokenizer.count_tokens(context_str + self.initial_prompt)
            
            remaining_tokens = max(0, self.max_total_tokens_budget - self.initial_input_tokens)
            
            # Calculate debate tokens, ensuring it meets the minimum if possible
            debate_tokens = int(remaining_tokens * debate_ratio)
            synthesis_tokens = int(remaining_tokens * synthesis_ratio)
            
            # Define a minimum token allocation to ensure phases can function
            MIN_PHASE_TOKENS = 250 # Keep this as a safeguard
            debate_tokens = max(MIN_PHASE_TOKENS, debate_tokens)
            synthesis_tokens = max(MIN_PHASE_TOKENS, synthesis_tokens)
            
            # Re-distribute if sum exceeds remaining_tokens due to min_phase_tokens
            total_allocated = debate_tokens + synthesis_tokens
            if total_allocated > remaining_tokens:
                diff = total_allocated - remaining_tokens
                if debate_tokens > synthesis_tokens:
                    debate_tokens -= diff
                else:
                    synthesis_tokens -= diff
                debate_tokens = max(MIN_PHASE_TOKENS, debate_tokens)
                synthesis_tokens = max(MIN_PHASE_TOKENS, synthesis_tokens)
            
            self.phase_budgets = {
                "context": self.initial_input_tokens,
                "debate": debate_tokens,
                "synthesis": synthesis_tokens
            }
            
            self._log_with_context("info", "SocraticDebate token budgets initialized",
                                   initial_input_tokens=self.initial_input_tokens,
                                   context_budget=self.phase_budgets["context"],
                                   debate_budget=self.phase_budgets["debate"],
                                   synthesis_budget=self.phase_budgets["synthesis"],
                                   max_total_tokens_budget=self.max_total_tokens_budget,
                                   prompt_complexity=prompt_complexity)

        except Exception as e:
            self._log_with_context("error", "Token budget calculation failed",
                                   error=str(e), context="token_budget", exc_info=True, original_exception=e)
            self.phase_budgets = {"context": 500, "debate": 15000, "synthesis": 1000}
            self.initial_input_tokens = 0
            raise ChimeraError("Failed to calculate token budgets due to an unexpected error.", original_exception=e) from e
    
    def _analyze_prompt_complexity(self, prompt: str) -> str:
        """Analyze prompt complexity based on various factors."""
        word_count = len(prompt.split())
        sentence_count = len(re.findall(r'[.!?]+', prompt))
        
        technical_terms = ["implement", "algorithm", "architecture", "refactor", 
                          "optimize", "debug", "vulnerability", "security", "design pattern",
                          "scalability", "performance", "maintainability", "robustness", "efficiency"]
        technical_count = sum(1 for term in technical_terms if term in prompt.lower())
        
        step_indicators = ["first", "then", "next", "after", "finally", "step", "phase", "plan", "strategy"]
        step_count = sum(1 for indicator in step_indicators if indicator in prompt.lower())
        
        complexity_score = (word_count / 50) + (technical_count * 2) + (step_count * 1.5)
        
        if complexity_score > 3:
            return "high"
        elif complexity_score > 1.5:
            return "medium"
        else:
            return "low"

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

    def _perform_context_analysis(self) -> Optional[Dict[str, Any]]:
        """
        Performs context analysis based on the initial prompt and codebase context.
        """
        if not self.codebase_context or not self.context_analyzer:
            self._log_with_context("info", "No codebase context or analyzer available. Skipping context analysis.")
            return None

        self._log_with_context("info", "Performing context analysis.")
        try:
            current_persona_names = self.persona_router.determine_persona_sequence(
                self.initial_prompt, self.domain, intermediate_results=self.intermediate_steps
            )
            
            # Pass the context budget for dynamic file selection
            relevant_files = self.context_analyzer.find_relevant_files(
                self.initial_prompt, 
                max_context_tokens=self.phase_budgets["context"], # Pass the allocated context budget
                active_personas=current_persona_names
            )
            
            # Generate context summary using the intelligent summarizer
            context_summary_str = self.context_analyzer.generate_context_summary(
                [f[0] for f in relevant_files], # Pass only file paths
                self.phase_budgets["context"], # Max tokens for the summary
                self.initial_prompt # Pass initial prompt for context
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
            self._log_with_context("error", f"Error determining persona sequence: {e}", exc_info=True, original_exception=e)
            self.rich_console.print(f"[red]Error determining persona sequence: {e}[/red]")
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
        
        estimated_tokens = self.tokenizer.count_tokens(prompt) + persona_config.max_tokens
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
        
        if context_persona_turn_results:
            previous_output = f"Initial Prompt: {self.initial_prompt}\n\nContext Analysis:\n{json.dumps(context_persona_turn_results, indent=2)}"

        personas_for_debate = [
            p for p in persona_sequence
            if p not in ["Context_Aware_Assistant", "Impartial_Arbitrator", "General_Synthesizer", "Devils_Advocate"]
        ]
        
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

            persona_config = self.all_personas.get(persona_name)
            if not persona_config:
                self._log_with_context("error", f"Persona configuration not found for {persona_name}. Skipping turn.", persona=persona_name)
                debate_history.append({"persona": persona_name, "error": "Config not found"})
                continue

            current_prompt = f"Initial Problem: {self.initial_prompt}\n\nPrevious Debate Output:\n{json.dumps(previous_output, indent=2) if isinstance(previous_output, dict) else previous_output}"
            
            estimated_tokens = self.tokenizer.count_tokens(current_prompt) + persona_config.max_tokens
            self.check_budget("debate", estimated_tokens, persona_name)

            try:
                output = self._execute_llm_turn(persona_name, persona_config, current_prompt, "debate")
                debate_history.append({"persona": persona_name, "output": output})
                previous_output = output
            except Exception as e:
                self._log_with_context("error", f"Error during {persona_name} turn: {e}", persona=persona_name, exc_info=True, original_exception=e)
                self.rich_console.print(f"[red]Error during {persona_name} turn: {e}[/red]")
                previous_output = {"error": f"Turn failed for {persona_name}: {str(e)}", "malformed_blocks": [{"type": "DEBATE_TURN_ERROR", "message": str(e)}]}
                continue

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

        full_debate_context = {
            "initial_prompt": self.initial_prompt,
            "debate_history": debate_persona_results
        }
        
        prompt = f"Synthesize the following debate results into a coherent final answer, adhering strictly to your JSON schema:\n\n{json.dumps(full_debate_context, indent=2)}"
        
        estimated_tokens = self.tokenizer.count_tokens(prompt) + persona_config.max_tokens
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
        try:
            raw_llm_output, input_tokens, output_tokens = self.llm_provider.generate(
                prompt=prompt,
                system_prompt=persona_config.system_prompt,
                temperature=persona_config.temperature,
                max_tokens=persona_config.max_tokens,
                persona_config=persona_config,
                intermediate_results=self.intermediate_steps,
                requested_model_name=self.model_name
            )
            self.track_token_usage(phase, input_tokens + output_tokens)
            self.check_budget(phase, input_tokens + output_tokens, persona_name)
            
            if persona_name in self.PERSONA_OUTPUT_SCHEMAS:
                schema_model = self.PERSONA_OUTPUT_SCHEMAS[persona_name]
                
                parser = LLMOutputParser()
                parsed_output = parser.parse_and_validate(raw_llm_output, schema_model)
                
                if parsed_output.get("malformed_blocks"):
                    self._log_with_context("warning", f"LLM output for {persona_name} contained malformed blocks.",
                                           persona=persona_name, malformed_blocks=parsed_output["malformed_blocks"])
                    self.intermediate_steps.setdefault("malformed_blocks", []).extend(parsed_output["malformed_blocks"])
                
                if parsed_output.get("error_type") == "SCHEMA_VALIDATION_FAILED" or \
                   any(block.get("type") in ["JSON_EXTRACTION_FAILED", "JSON_DECODE_ERROR", "INVALID_JSON_STRUCTURE"] for block in parsed_output.get("malformed_blocks", [])):
                    raise SchemaValidationError(
                        error_type="LLM_OUTPUT_MALFORMED",
                        field_path="N/A",
                        invalid_value=raw_llm_output[:500],
                        details={"persona": persona_name, "raw_output_snippet": raw_llm_output[:500], "malformed_blocks": parsed_output.get("malformed_blocks", [])}
                    )
                
                return parsed_output
            else:
                self._log_with_context("info", f"Persona {persona_name} is not configured for structured JSON output. Returning raw text.", persona=persona_name)
                return raw_llm_output

        except CircuitBreakerError as cbe:
            self._log_with_context("error", f"Circuit breaker open for {persona_name} LLM call: {cbe.message}",
                                   persona=persona_name, exc_info=True, original_exception=cbe)
            self.status_callback(f"[red]Circuit breaker open for {persona_name}. Skipping turn.[/red]",
                                 state="error",
                                 current_total_tokens=self.tokens_used,
                                 current_total_cost=self.get_total_estimated_cost())
            raise cbe
        except TokenBudgetExceededError as tbe:
            self._log_with_context("error", f"Token budget exceeded for {persona_name}: {tbe.message}",
                                   persona=persona_name, exc_info=True, original_exception=tbe)
            self.status_callback(f"[red]Token budget exceeded for {persona_name}. Skipping turn.[/red]",
                                 state="error",
                                 current_total_tokens=self.tokens_used,
                                 current_total_cost=self.get_total_estimated_cost())
            raise tbe
        except LLMProviderError as lpe:
            self._log_with_context("error", f"LLM Provider Error for {persona_name}: {lpe.message}",
                                   persona=persona_name, exc_info=True, original_exception=lpe)
            self.status_callback(f"[red]LLM Provider Error for {persona_name}. Skipping turn.[/red]",
                                 state="error",
                                 current_total_tokens=self.tokens_used,
                                 current_total_cost=self.get_total_estimated_cost())
            raise lpe
        except SchemaValidationError as sve:
            self._log_with_context("error", f"Schema validation failed for {persona_name} output: {sve.message}",
                                   persona=persona_name, exc_info=True, original_exception=sve)
            self.status_callback(f"[red]Schema validation failed for {persona_name}. Circuit breaker may trip.[/red]",
                                 state="error",
                                 current_total_tokens=self.tokens_used,
                                 current_total_cost=self.get_total_estimated_cost())
            raise sve
        except Exception as e:
            self._log_with_context("error", f"Unexpected error during {persona_name} LLM turn: {e}",
                                   persona=persona_name, exc_info=True, original_exception=e)
            self.status_callback(f"[red]Unexpected error during {persona_name} turn. Skipping.[/red]",
                                 state="error",
                                 current_total_tokens=self.tokens_used,
                                 current_total_cost=self.get_total_estimated_cost())
            raise ChimeraError(f"An unexpected error occurred during {persona_name}'s turn: {e}",
                               details={"persona": persona_name, "traceback": traceback.format_exc()}, original_exception=e) from e

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
        return final_answer, self.intermediate_steps

    def _update_intermediate_steps_with_totals(self):
        """Updates the intermediate steps dictionary with total token usage and estimated cost."""
        self.intermediate_steps["Total_Tokens_Used"] = self.tokens_used
        self.intermediate_steps["Total_Estimated_Cost_USD"] = self.get_total_estimated_cost()

    @handle_errors(log_level="ERROR") # Apply the decorator here
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