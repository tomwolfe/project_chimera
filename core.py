# core.py
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
from google.genai.errors import APIError
from rich.console import Console
from pydantic import ValidationError
from functools import lru_cache # Import lru_cache for caching

# --- ADDED IMPORT ---
# Import the corrected GeminiProvider from llm_provider.py
from llm_provider import GeminiProvider
# --- END ADDED IMPORT ---

# Import models and settings
from src.models import PersonaConfig, ReasoningFrameworkConfig # Assuming LLMOutput is defined here or accessible
from src.config.settings import ChimeraSettings
from src.persona.routing import PersonaRouter
from src.context.context_analyzer import ContextRelevanceAnalyzer
from src.utils import LLMOutputParser
# NEW: Import LLMResponseValidationError and other exceptions
from src.exceptions import ChimeraError, LLMResponseValidationError, SchemaValidationError, TokenBudgetExceededError # Corrected import

# Configure logging
logger = logging.getLogger(__name__)

class SocraticDebate:
    def __init__(self, initial_prompt: str, api_key: str,
                 codebase_context: Optional[Dict[str, str]] = None, # Changed type hint to Dict[str, str]
                 settings: Optional[ChimeraSettings] = None,
                 all_personas: Optional[Dict[str, PersonaConfig]] = None,
                 persona_sets: Optional[Dict[str, List[str]]] = None, # Added persona_sets
                 persona_sequence: Optional[List[str]] = None, # Added persona_sequence
                 domain: Optional[str] = None, # Added domain
                 max_total_tokens_budget: int = 10000,
                 model_name: str = "gemini-2.5-flash-lite", # Default model name
                 status_callback: Optional[Callable] = None, # Added status_callback
                 rich_console: Optional[Console] = None, # Added rich_console
                 context_token_budget_ratio: float = 0.25 # ADDED THIS LINE
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
        """
        self.settings = settings or ChimeraSettings()
        self.settings.context_token_budget_ratio = context_token_budget_ratio
        self.max_total_tokens_budget = max_total_tokens_budget
        self.tokens_used = 0
        self.model_name = model_name # Store the model name selected by the user
        
        self.context_analyzer = None
        self.codebase_context = None
        if codebase_context:
            self.codebase_context = codebase_context
            self.context_analyzer = ContextRelevanceAnalyzer()
            if isinstance(codebase_context, dict):
                self.context_analyzer.compute_file_embeddings(self.codebase_context)
            else:
                logger.warning("codebase_context was not a dictionary, skipping embedding computation.")
        
        self.all_personas = all_personas or {}
        self.persona_sets = persona_sets or {}
        self.persona_sequence = persona_sequence or []
        self.domain = domain
        self.persona_router = PersonaRouter(self.all_personas)
        
        self.llm_provider = GeminiProvider(api_key=api_key, model_name=self.model_name)
        
        self.initial_prompt = initial_prompt
        
        try:
            self.initial_input_tokens = self.llm_provider.count_tokens(self.initial_prompt, system_prompt=None)
        except Exception as e:
            logger.error(f"Failed to count tokens for initial prompt: {e}. Setting initial_input_tokens to 0.")
            self.initial_input_tokens = 0
        
        self.phase_budgets = {}
        self.tokens_used_per_phase = {"context": 0, "debate": 0, "synthesis": 0}
        self.tokens_used_per_step = {}

        self._calculate_token_budgets()
        
        self.intermediate_steps = {}
        self.final_answer = None
        self.process_log = []
        
        self.status_callback = status_callback
        self.rich_console = rich_console or Console()
        
        self._prev_context_ratio = None
    
    def _calculate_token_budgets(self):
        """Calculate dynamic token budgets based on settings and prompt analysis."""
        
        from src.constants import is_self_analysis_prompt
        is_self_analysis = is_self_analysis_prompt(self.initial_prompt)
        
        base_context_ratio = self.settings.context_token_budget_ratio
        base_debate_ratio = self.settings.debate_token_budget_ratio
        
        if self.initial_input_tokens >= self.max_total_tokens_budget:
            logger.warning(f"Initial prompt tokens ({self.initial_input_tokens}) are equal to or exceed the total budget ({self.max_total_tokens_budget}). Adjusting max_total_tokens_budget.")
            self.max_total_tokens_budget = self.initial_input_tokens + 500 
            logger.info(f"Adjusted max_total_tokens_budget to: {self.max_total_tokens_budget}")
        
        available_tokens = self.max_total_tokens_budget - self.initial_input_tokens
        
        complexity_score = self._calculate_complexity_score(self.initial_prompt)
        
        adjusted_ratio = base_context_ratio + (complexity_score ** 1.5) * 0.15
        
        if self._prev_context_ratio is not None:
            max_change = 0.05
            adjusted_ratio = max(self._prev_context_ratio - max_change, 
                                  min(self._prev_context_ratio + max_change, adjusted_ratio))
        
        context_ratio = max(0.15, min(0.25, adjusted_ratio))
        self._prev_context_ratio = context_ratio
        
        remaining_budget_share = 1.0 - context_ratio
        debate_ratio = remaining_budget_share * 0.85
        synthesis_ratio = remaining_budget_share * 0.15
        
        context_tokens = max(300, int(available_tokens * context_ratio))
        debate_tokens = max(800, int(available_tokens * debate_ratio))
        synthesis_tokens = max(300, int(available_tokens * synthesis_ratio))
        
        total_allocated = context_tokens + debate_tokens + synthesis_tokens
        if total_allocated > available_tokens:
            scale_factor = available_tokens / total_allocated
            context_tokens = int(context_tokens * scale_factor)
            debate_tokens = int(debate_tokens * scale_factor)
            synthesis_tokens = int(synthesis_tokens * scale_factor)
        
        if synthesis_tokens < 300:
            needed = 300 - synthesis_tokens
            debate_reduction = min(needed, debate_tokens - 800)
            context_reduction = needed - debate_reduction
            
            debate_tokens = max(800, debate_tokens - debate_reduction)
            context_tokens = max(300, context_tokens - context_reduction)
            synthesis_tokens = 300
        elif debate_tokens < 800:
            needed = 800 - debate_tokens
            context_reduction = min(needed, context_tokens - 300)
            debate_tokens = max(800, debate_tokens - context_reduction)
            context_tokens = max(300, context_tokens - context_reduction)
        elif context_tokens < 300:
            needed = 300 - context_tokens
            debate_tokens = max(800, debate_tokens - needed)
            context_tokens = 300
        
        final_total = context_tokens + debate_tokens + synthesis_tokens
        if final_total > available_tokens:
            trim_amount = final_total - available_tokens
            debate_tokens = max(800, debate_tokens - trim_amount)
            if debate_tokens < 800:
                debate_tokens = 800
                trim_amount = final_total - available_tokens - (debate_tokens - 800)
                context_tokens = max(300, context_tokens - trim_amount)
        
        self.phase_budgets["context"] = context_tokens
        self.phase_budgets["debate"] = debate_tokens
        self.phase_budgets["synthesis"] = synthesis_tokens
        
        logger.info(f"Token budgets calculated: Context={self.phase_budgets['context']}, "
                   f"Debate={self.phase_budgets['debate']}, Synthesis={self.phase_budgets['synthesis']}")
    
    def _check_token_budget(self, prompt_text: str, step_name: str, system_prompt: str = "", phase: str = None) -> int:
        """
        Check if using the specified tokens would exceed the budget using accurate counting.
        Returns the number of tokens used for this step.
        Raises TokenBudgetExceededError if budget is exceeded.
        """
        try:
            try:
                prompt_text = prompt_text.encode('utf-8').decode('utf-8')
            except UnicodeEncodeError:
                prompt_text = prompt_text.encode('utf-8', 'replace').decode('utf-8', 'replace')
                logger.warning(f"Fixed encoding issues in prompt for step '{step_name}' by replacing problematic characters.")

            actual_tokens = self.llm_provider.count_tokens(prompt_text, system_prompt=system_prompt)

            if self.tokens_used + actual_tokens > self.max_total_tokens_budget:
                raise TokenBudgetExceededError(
                    current_tokens=self.tokens_used,
                    budget=self.max_total_tokens_budget,
                    details={"step": step_name, "tokens_requested": actual_tokens}
                )
            
            if phase and phase in self.phase_budgets:
                if self.tokens_used_per_phase.get(phase, 0) + actual_tokens > self.phase_budgets[phase]:
                    raise TokenBudgetExceededError(
                        current_tokens=self.tokens_used_per_phase.get(phase, 0),
                        budget=self.phase_budgets[phase],
                        details={"phase": phase, "step": step_name, "tokens_requested": actual_tokens}
                    )
            
            self.tokens_used += actual_tokens 
            if phase and phase in self.tokens_used_per_phase:
                self.tokens_used_per_phase[phase] += actual_tokens
            
            self.tokens_used_per_step[step_name] = actual_tokens
            
            return actual_tokens
        except TokenBudgetExceededError:
            raise
        except Exception as e:
            logger.error(f"Error during token budget check for step '{step_name}': {e}")
            raise TokenBudgetExceededError(
                current_tokens=self.tokens_used,
                budget=self.max_total_tokens_budget,
                details={"step": step_name, "error": f"Token counting failed: {e}"}
            ) from e

    def _analyze_context(self) -> Dict[str, Any]:
        """Analyze the context of the prompt to determine the best approach."""
        if not self.codebase_context or not self.context_analyzer:
            logger.info("No codebase context provided, skipping context analysis")
            return {"domain": "General", "relevant_files": []}
        
        prompt_words = self.initial_prompt.lower().split()
        keywords = list(dict.fromkeys(prompt_words))[:5]
        
        relevant_files = self.context_analyzer.find_relevant_files(self.initial_prompt)
        
        domain = self.persona_router.determine_domain(self.initial_prompt)
        
        logger.info(f"Context analysis complete. Domain: {domain}, Relevant files: {len(relevant_files)}")
        
        return {
            "domain": domain,
            "relevant_files": relevant_files,
            "keywords": keywords
        }
    
    def _prepare_context(self, context_analysis: Dict[str, Any]) -> str:
        """Prepare the context for the debate based on the context analysis,
        respecting the context token budget."""
        
        from src.constants import is_self_analysis_prompt
        if is_self_analysis_prompt(self.initial_prompt):
            return self._prepare_self_analysis_context(context_analysis)
        
        if not self.codebase_context or not context_analysis.get("relevant_files"):
            return ""
        
        context_parts = []
        current_context_tokens = 0
        
        for file_path, _ in context_analysis.get("relevant_files", []):  
            if file_path not in self.codebase_context:
                continue

            content = self.codebase_context[file_path]
            
            remaining_budget_chars = (self.phase_budgets.get("context", 200) - current_context_tokens) * 4
            
            if remaining_budget_chars <= 0:
                break

            key_elements = self._extract_key_elements(content)
            relevant_segment = self.context_analyzer.extract_relevant_code_segments(
                content, max_chars=int(remaining_budget_chars)
            )
            
            file_context_part = (
                f"File: {file_path}\n"
                f"Key elements: {key_elements}\n"
                f"Content snippet:\n```\n{relevant_segment}\n```\n"
            )
            
            estimated_file_tokens = self.llm_provider.count_tokens(file_context_part)
            
            if current_context_tokens + estimated_file_tokens > self.phase_budgets.get("context", 200):
                logger.info(f"Skipping {file_path} due to context budget. Current: {current_context_tokens}, Estimated for file: {estimated_file_tokens}, Budget: {self.phase_budgets.get('context', 200)}")
                break
            
            context_parts.append(file_context_part)
            current_context_tokens += estimated_file_tokens
        
        logger.info(f"Prepared context with {len(context_parts)} files, total estimated tokens: {current_context_tokens}")
        return "".join(context_parts)
    
    def _prepare_self_analysis_context(self, context_analysis: Dict[str, Any]) -> str:
        """Prepare specialized context for self-analysis with core files prioritized."""
        if not self.codebase_context or not context_analysis.get("relevant_files"):
            return ""
        
        core_files = [
            "src/core.py", "src/persona/routing.py", "src/token_manager.py", "src/constants.py",
            "src/exceptions.py", "src/models.py", "src/llm_provider.py", "src/utils/output_parser.py",
            "src/utils/code_validator.py", "src/utils/path_utils.py", "src/config/settings.py",
            "src/config/persistence.py", "src/persona_manager.py", "src/context/context_analyzer.py",
            "src/tokenizers/base.py", "src/tokenizers/gemini_tokenizer.py", "app.py"
        ]
        
        context_parts = []
        current_context_tokens = 0
        context_budget = self.phase_budgets.get("context", 200)
        
        for file_path in core_files:
            if file_path in self.codebase_context:
                content = self.codebase_context[file_path]
                file_context_part = f"### {file_path}\n{content}\n"
                estimated_file_tokens = self.llm_provider.count_tokens(file_context_part)
                
                if current_context_tokens + estimated_file_tokens > context_budget:
                    logger.warning(f"Context budget exceeded while adding core file '{file_path}'. Stopping context preparation.")
                    break
                    
                context_parts.append(file_context_part)
                current_context_tokens += estimated_file_tokens
        
        for file_path, _ in context_analysis.get("relevant_files", []):
            if file_path in core_files or file_path not in self.codebase_context:
                continue
                
            content = self.codebase_context[file_path]
            file_context_part = f"### {file_path}\n{content}\n"
            estimated_file_tokens = self.llm_provider.count_tokens(file_context_part)
            
            if current_context_tokens + estimated_file_tokens > context_budget:
                logger.warning(f"Context budget exceeded while adding relevant file '{file_path}'. Stopping context preparation.")
                break
                
            context_parts.append(file_context_part)
            current_context_tokens += estimated_file_tokens
        
        logger.info(f"Prepared self-analysis context with {len(context_parts)} files, total estimated tokens: {current_context_tokens}")
        return "".join(context_parts)
    
    def _generate_persona_sequence(self, context_analysis: Dict[str, Any]) -> List[str]:
        """Generate the sequence of personas to participate in the debate."""
        domain_for_sequence = context_analysis.get("domain", self.domain) or "General"
        
        if domain_for_sequence and domain_for_sequence in self.persona_sets:
            base_sequence = self.persona_sets[domain_for_sequence]
        else:
            base_sequence = self.persona_sequence
            if not base_sequence:
                base_sequence = ["Visionary_Generator", "Skeptical_Generator", "Impartial_Arbitrator"]

        # Pass context_analysis_results to determine_persona_sequence
        final_sequence = self.persona_router.determine_persona_sequence(
            self.initial_prompt,
            intermediate_results=None,
            context_analysis_results=context_analysis # Pass context analysis results
        )
        
        seen = set()
        unique_sequence = []
        for persona in final_sequence:
            if persona not in seen:
                unique_sequence.append(persona)
                seen.add(persona)
        
        return unique_sequence
    
    def _run_debate_round(self, current_response: str, persona_name: str, requested_model_name: str = None) -> str:
        """Run a single round of the debate with the specified persona."""
        if persona_name not in self.all_personas:
            logger.warning(f"Persona '{persona_name}' not found. Skipping this round.")
            return current_response

        persona = self.all_personas[persona_name]
        
        prompt_for_llm = f"""
You are {persona_name}: {persona.description}
{persona.system_prompt}

Current debate state:
{current_response}

User's original prompt:
{self.initial_prompt}
"""
        
        input_tokens_for_call = self._check_token_budget(
            prompt_for_llm, 
            f"debate_round_{persona_name}", 
            system_prompt=persona.system_prompt,
            phase="debate"
        )
        
        logger.info(f"Running debate round with {persona_name} using model: {requested_model_name or self.model_name}")
        generated_text, input_tokens, output_tokens = self.llm_provider.generate(
            prompt=prompt_for_llm,
            system_prompt=persona.system_prompt,
            temperature=persona.temperature,
            max_tokens=persona.max_tokens,
            persona_config=persona,
            intermediate_results=self.intermediate_steps,
            requested_model_name=requested_model_name # Pass user's explicit choice
        )
        
        self.tokens_used += output_tokens
        self.tokens_used_per_phase["debate"] += output_tokens
        
        self.intermediate_steps[f"{persona_name}_Output"] = generated_text
        self.intermediate_steps[f"{persona_name}_Input_Tokens"] = input_tokens
        self.intermediate_steps[f"{persona_name}_Output_Tokens"] = output_tokens
        self.tokens_used_per_step[f"{persona_name}_Input"] = input_tokens
        self.tokens_used_per_step[f"{persona_name}_Output"] = output_tokens
        
        return generated_text
    
    def _synthesize_final_answer(self, final_debate_state: str, requested_model_name: str = None) -> Dict[str, Any]:
        """
        Synthesize the final answer from the debate state, with retry logic
        for schema validation failures.
        """
        arbitrator = None
        for persona_name, persona in self.all_personas.items():
            if "arbitrator" in persona_name.lower():
                arbitrator = persona
                break
        
        if not arbitrator:
            logger.error("Impartial_Arbitrator persona not found. Cannot synthesize final answer.")
            return {
                "COMMIT_MESSAGE": "Error: Arbitrator Persona Missing",
                "RATIONALE": "The 'Impartial_Arbitrator' persona is required for synthesis but was not found in the loaded personas.",
                "CODE_CHANGES": [],
                "malformed_blocks": [{"type": "CRITICAL_CONFIG_ERROR", "message": "Arbitrator persona not found."}]
            }

        max_retries = 2
        for attempt in range(max_retries + 1):
            prompt_for_synthesis = f"""
{arbitrator.system_prompt}

Based on the following debate, provide a final synthesized answer:

Debate Summary:
{final_debate_state}

User's Original Prompt:
{self.initial_prompt}
"""
            if attempt > 0:
                prompt_for_synthesis += f"\n\n**ATTENTION: PREVIOUS RESPONSE FAILED VALIDATION.**\n" \
                                        f"Please ensure your response is a PERFECTLY VALID JSON object " \
                                        f"adhering to the `LLMOutput` schema. Double-check all commas, " \
                                        f"quotes, and nested structures. Do NOT include any text outside " \
                                        f"the JSON block. This is attempt {attempt+1}/{max_retries+1}."
                logger.warning(f"Retrying final answer synthesis (attempt {attempt+1}).")

            tokens_used_in_synthesis = self._check_token_budget(
                prompt_for_synthesis, 
                "final_synthesis", 
                system_prompt=arbitrator.system_prompt,
                phase="synthesis"
            )
            
            generated_text, input_tokens, output_tokens = self.llm_provider.generate(
                prompt=prompt_for_synthesis,
                system_prompt=arbitrator.system_prompt,
                temperature=arbitrator.temperature,
                max_tokens=arbitrator.max_tokens,
                persona_config=arbitrator,
                intermediate_results=self.intermediate_steps,
                requested_model_name=requested_model_name # Pass user's explicit choice
            )
            raw_final_answer = generated_text
            self.tokens_used += output_tokens
            self.tokens_used_per_phase["synthesis"] += output_tokens
            
            self.tokens_used_per_step["final_synthesis_Input"] = input_tokens
            self.tokens_used_per_step["final_synthesis_Output"] = output_tokens
            
            try:
                llm_output_parser = LLMOutputParser()
                validated_output_dict = llm_output_parser.parse_and_validate(raw_final_answer, LLMOutput)
                
                self.final_answer = validated_output_dict
                self.intermediate_steps["Final_Answer_Output"] = validated_output_dict
                self.intermediate_steps["Total_Tokens_Used"] = self.tokens_used 
                self.intermediate_steps["Total_Estimated_Cost_USD"] = self._calculate_cost()
                self.intermediate_steps["Tokens_Used_Per_Phase"] = self.tokens_used_per_phase
                self.intermediate_steps["Tokens_Used_Per_Step"] = self.tokens_used_per_step
                
                return validated_output_dict
            except SchemaValidationError as sve:
                logger.error(f"Schema validation failed for final answer (attempt {attempt+1}): {sve}", exc_info=True)
                self.intermediate_steps[f"Final_Answer_Validation_Error_Attempt_{attempt+1}"] = {
                    "message": str(sve),
                    "details": sve.details
                }
                if attempt == max_retries:
                    raise
            except Exception as e:
                logger.error(f"Unexpected error during final answer processing (attempt {attempt+1}): {e}", exc_info=True)
                self.intermediate_steps[f"Final_Answer_Processing_Error_Attempt_{attempt+1}"] = str(e)
                if attempt == max_retries:
                    raise LLMResponseValidationError(
                        f"Final answer processing failed after {max_retries} retries: {str(e)}",
                        invalid_response=raw_final_answer,
                        expected_schema="LLMOutput",
                        details={"processing_error": str(e)}
                    ) from e
        
        raise Exception("Unexpected state in _synthesize_final_answer.")
    
    def _calculate_cost(self) -> float:
        """Calculate the estimated cost based on token usage."""
        return self.tokens_used * 0.000003
    
    def _calculate_complexity_score(self, prompt: str) -> float:
        """
        Calculate a semantic complexity score for the prompt (0.0 to 1.0).
        """
        prompt_lower = prompt.lower()
        complexity = 0.0
        
        length_factor = min(1.0, len(prompt) / 2000.0)
        complexity += length_factor * 0.5
        
        technical_keywords = [
            "code", "analyze", "refactor", "algorithm", "architecture", "system",
            "science", "research", "business", "market", "creative", "art",
            "security", "test", "deploy", "optimize", "debug"
        ]
        keyword_count = sum(1 for kw in technical_keywords if kw in prompt_lower)
        complexity += (keyword_count / len(technical_keywords) if technical_keywords else 0) * 0.5
        
        return max(0.0, min(1.0, complexity))

    def _extract_quality_metrics_from_results(self, intermediate_results: Dict[str, Any]) -> Dict[str, float]:
        """
        Placeholder for extracting quality metrics from intermediate results.
        """
        metrics = {
            "code_quality": 0.5, "security_risk_score": 0.5, "maintainability_index": 0.5,
            "test_coverage_estimate": 0.5, "reasoning_depth": 0.5, "architectural_coherence": 0.5
        }
        
        if "Context_Aware_Assistant_Output" in intermediate_results:
            caa_output = intermediate_results["Context_Aware_Assistant_Output"]
            if isinstance(caa_output, dict) and "quality_metrics" in caa_output and isinstance(caa_output["quality_metrics"], dict):
                quality_metrics_from_caa = caa_output["quality_metrics"]
                for metric_name, value in quality_metrics_from_caa.items():
                    if metric_name in metrics:
                        metrics[metric_name] = max(metrics[metric_name], value)
        
        for key in metrics:
            metrics[key] = max(0.0, min(1.0, metrics[key]))
            
        logger.debug(f"Extracted quality metrics (placeholder): {metrics}")
        return metrics

    def run_debate(self) -> Tuple[Any, Dict[str, Any]]:
        """
        Run the complete Socratic debate process and return the results.
        """
        try:
            initial_prompt_tokens = self._check_token_budget(self.initial_prompt, "initial_prompt_count")
            
            context_analysis = self._analyze_context()
            
            self.intermediate_steps["Context_Analysis"] = context_analysis
            context_str = self._prepare_context(context_analysis)
            if context_str:
                self._check_token_budget(context_str, "context_preparation", phase="context")
            
            # Pass context analysis results to _generate_persona_sequence
            self.persona_sequence = self._generate_persona_sequence(context_analysis)
            self.intermediate_steps["Persona_Sequence"] = self.persona_sequence
            
            current_response = ""
            if self.persona_sequence:
                persona_name = self.persona_sequence[0]
                persona = self.all_personas.get(persona_name)
                
                # Pass the user's selected model name to _run_debate_round
                current_response = self._run_debate_round(
                    "No previous responses. Starting the debate.", 
                    persona_name,
                    requested_model_name=self.model_name # Pass user's selected model
                )
                
                for persona_name in self.persona_sequence[1:]:
                    current_response = self._run_debate_round(current_response, persona_name, requested_model_name=self.model_name)
            else:
                raise ChimeraError("No persona sequence generated. Debate cannot proceed.")
            
            # Pass the user's selected model name to _synthesize_final_answer
            final_answer = self._synthesize_final_answer(current_response, requested_model_name=self.model_name)
            
            return final_answer, self.intermediate_steps
            
        except TokenBudgetExceededError as e:
            return self.handle_token_budget_exceeded(e)
        except SchemaValidationError as sve:
            logger.error(f"Schema validation failed during debate: {sve}", exc_info=True)
            raise
        except ChimeraError as ce:
            logger.error(f"Chimera-specific error during debate: {ce}", exc_info=True)
            raise
        except Exception as e:
            logger.exception("Unexpected error during debate process")
            raise

    def handle_token_budget_exceeded(self, e: TokenBudgetExceededError) -> Tuple[Any, Dict[str, Any]]:
        """
        Handles TokenBudgetExceededError with multi-stage graceful degradation.
        """
        logger.warning(f"Token budget exceeded: {str(e)}. Attempting graceful degradation.")
        
        if self.context_token_budget_ratio > 0.15:
            self.context_token_budget_ratio = max(0.15, self.context_token_budget_ratio * 0.8)
            logger.info(f"Reducing context ratio to {self.context_token_budget_ratio} and retrying.")
            return self.run_debate_process()
        
        elif len(self.persona_sequence) > 3 and 'Impartial_Arbitrator' in self.persona_sequence:
            core_personas = ["Visionary_Generator", "Skeptical_Generator", "Constructive_Critic", "Impartial_Arbitrator"]
            self.persona_sequence = [p for p in self.persona_sequence if p in core_personas]
            logger.info(f"Simplifying persona sequence to core personas and retrying.")
            return self.run_debate_process()
        
        elif self.intermediate_steps:
            logger.warning("Returning partial results due to token constraints.")
            partial_result = self._synthesize_partial_results()
            partial_result += "\n\n[WARNING: Output truncated due to token constraints - full analysis not possible]"
            self.intermediate_steps["Partial_Result_Warning"] = partial_result
            return partial_result, self.intermediate_steps
        
        else:
            logger.error("Graceful degradation failed. Re-raising with diagnostic info.", exc_info=True)
            e.details = {
                **(e.details or {}),
                "degradation_failed": True,
                "context_ratio": self.context_token_budget_ratio,
                "persona_sequence_length": len(self.persona_sequence),
                "token_usage_breakdown": self._get_token_usage_breakdown()
            }
            raise

    def _get_token_usage_breakdown(self) -> Dict[str, int]:
        """Helper to provide token usage details for error reporting."""
        return {
            "total_used": self.tokens_used,
            "total_budget": self.max_total_tokens_budget,
            "initial_input": self.initial_input_tokens,
            "phase_budgets": self.phase_budgets,
            "phase_usage": self.tokens_used_per_phase,
            "step_usage": self.tokens_used_per_step
        }

    def run_debate_process(self):
        """Helper method to retry debate with adjusted parameters after budget exceeded."""
        self._calculate_token_budgets()
        
        context_analysis = self._analyze_context()
        context_str = self._prepare_context(context_analysis)
        if context_str:
            self._check_token_budget(context_str, "context_preparation_retry", phase="context")
        
        self.persona_sequence = self._generate_persona_sequence(context_analysis)
        self.intermediate_steps["Persona_Sequence"] = self.persona_sequence
        
        current_response = ""
        if self.persona_sequence:
            persona_name = self.persona_sequence[0]
            persona = self.all_personas.get(persona_name)
            current_response = self._run_debate_round(
                "Retrying debate after budget adjustment.", 
                persona_name,
                requested_model_name=self.model_name # Use the model that was initially selected
            )
            for persona_name in self.persona_sequence[1:]:
                current_response = self._run_debate_round(current_response, persona_name, requested_model_name=self.model_name)
        else:
            logger.warning("No persona sequence generated during retry. Debate cannot proceed.")
            return "Error: No persona sequence generated during retry.", self.intermediate_steps
        
        final_answer = self._synthesize_final_answer(current_response, requested_model_name=self.model_name)
        return final_answer, self.intermediate_steps

    def _synthesize_partial_results(self) -> str:
        """
        Synthesizes available intermediate steps into a partial result string.
        """
        partial_output = "--- Partial Debate Results ---\n\n"
        partial_output += f"Original Prompt: {self.initial_prompt}\n\n"
        
        persona_outputs = sorted([
            (name, result) for name, result in self.intermediate_steps.items()
            if name.endswith("_Output") and isinstance(result, str)
        ])
        
        for persona_name, output_text in persona_outputs:
            partial_output += f"### {persona_name.replace('_Output', '')}:\n"
            partial_output += f"{output_text[:300]}...\n\n"
        
        partial_output += f"Total Tokens Used (approx): {self.tokens_used}\n"
        partial_output += f"Estimated Cost (approx): ${self._calculate_cost():.4f}\n"
        
        return partial_output

    def _extract_key_elements(self, content: str) -> str:
        """Extract key structural elements from code for better semantic representation."""
        class_defs = re.findall(r'class\s+(\w+)', content)
        func_defs = re.findall(r'def\s+(\w+)', content)
        imports = re.findall(r'import\s+([\w.]+)', content)
        elements = []
        if class_defs: elements.append(f"Classes: {', '.join(class_defs[:5])}")
        if func_defs: elements.append(f"Functions: {', '.join(func_defs[:10])}")
        if imports: elements.append(f"Imports: {', '.join(imports[:5])}")
        return " ".join(elements)

# Additional helper functions
def load_personas_from_yaml(yaml_path: str) -> Dict[str, PersonaConfig]:
    """Load personas configuration from a YAML file."""
    try:
        with open(yaml_path, 'r') as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        logger.error(f"Personas file not found at {yaml_path}. Cannot load personas.")
        return {}
    except yaml.YAMLError as e:
        logger.error(f"Error parsing personas YAML file {yaml_path}: {e}")
        return {}
    
    personas = {}
    for persona_data in config.get('personas', []):
        try:
            personas[persona_data['name']] = PersonaConfig(**persona_data)
        except (ValidationError, KeyError) as e:
            logger.error(f"Invalid persona data in {yaml_path} for persona '{persona_data.get('name', 'Unnamed')}': {e}")
    
    return personas

def load_frameworks_from_yaml(yaml_path: str) -> Dict[str, ReasoningFrameworkConfig]:
    """Load reasoning frameworks from a YAML file."""
    try:
        with open(yaml_path, 'r') as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        logger.error(f"Frameworks file not found at {yaml_path}. Cannot load frameworks.")
        return {}
    except yaml.YAMLError as e:
        logger.error(f"Error parsing frameworks YAML file {yaml_path}: {e}")
        return {}
    
    frameworks = {}
    for framework_name, framework_data in config.get('reasoning_frameworks', {}).items():
        try:
            frameworks[framework_name] = ReasoningFrameworkConfig(
                framework_name=framework_name,
                personas={}, # Personas are loaded separately into all_personas
                persona_sets=framework_data.get('persona_sets', {}),
                version=framework_data.get('version', 1)
            )
        except (ValidationError, KeyError) as e:
            logger.error(f"Invalid framework data in {yaml_path} for framework '{framework_name}': {e}")
    
    return frameworks