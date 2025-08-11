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

# --- IMPORT MODIFICATIONS ---
# Import the corrected GeminiProvider from llm_provider.py
from llm_provider import GeminiProvider
# Import ContextRelevanceAnalyzer for dependency injection
from src.context.context_analyzer import ContextRelevanceAnalyzer
# --- END IMPORT MODIFICATIONS ---

# Import models and settings
from src.models import PersonaConfig, ReasoningFrameworkConfig # Assuming LLMOutput is defined here or accessible
from src.config.settings import ChimeraSettings
from src.persona.routing import PersonaRouter
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
                 context_token_budget_ratio: float = 0.25, # ADDED THIS LINE
                 context_analyzer: Optional[ContextRelevanceAnalyzer] = None # Added for caching dependency injection
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
            context_analyzer: An optional pre-initialized and cached ContextRelevanceAnalyzer instance.
        """
        self.settings = settings or ChimeraSettings()
        # Ensure the ratio from settings is used, or the provided default if settings is None
        self.settings.context_token_budget_ratio = context_token_budget_ratio 
        self.max_total_tokens_budget = max_total_tokens_budget
        self.tokens_used = 0
        self.model_name = model_name # Store the model name selected by the user
        
        self.context_analyzer = context_analyzer # Use the provided analyzer instance
        self.codebase_context = None
        if codebase_context and self.context_analyzer:
            self.codebase_context = codebase_context
            if isinstance(self.codebase_context, dict):
                # Compute embeddings if context is provided and analyzer is available.
                # This assumes the analyzer instance passed is already cached and potentially has embeddings computed.
                # If context changes, the analyzer's embeddings might need recomputation, handled by app.py caching.
                if not self.context_analyzer.file_embeddings: # Only compute if not already done
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
        
        self._prev_context_ratio = None # For adaptive context ratio adjustment
    
    def _calculate_token_budgets(self):
        """Calculate dynamic token budgets based on settings and prompt analysis."""
        
        from src.constants import is_self_analysis_prompt
        is_self_analysis = is_self_analysis_prompt(self.initial_prompt)
        
        # Use the ratio from settings, which might have been adjusted by app.py
        base_context_ratio = self.settings.context_token_budget_ratio
        
        # If self-analysis, use specific ratios from settings
        if is_self_analysis:
            base_context_ratio = self.settings.self_analysis_context_ratio
            # debate_ratio = self.settings.self_analysis_debate_ratio # Not directly used here, but for context

        if self.initial_input_tokens >= self.max_total_tokens_budget:
            logger.warning(f"Initial prompt tokens ({self.initial_input_tokens}) are equal to or exceed the total budget ({self.max_total_tokens_budget}). Adjusting max_total_tokens_budget.")
            self.max_total_tokens_budget = self.initial_input_tokens + 500 
            logger.info(f"Adjusted max_total_tokens_budget to: {self.max_total_tokens_budget}")
        
        available_tokens = self.max_total_tokens_budget - self.initial_input_tokens
        
        # Use complexity score for finer-grained adjustment of context ratio
        complexity_score = self._calculate_complexity_score(self.initial_prompt)
        
        # Adjust context ratio based on complexity, ensuring it stays within reasonable bounds
        # Formula: context_ratio = max(0.1, min(0.3, base_ratio + complexity_score * 0.05))
        # This ensures context ratio is between 10% and 30% of available tokens.
        adjusted_context_ratio = max(0.1, min(0.3, base_context_ratio + complexity_score * 0.05))
        
        # Apply smoothing/jitter to context ratio to prevent drastic changes between similar prompts
        if self._prev_context_ratio is not None:
            max_change = 0.05 # Limit change per step
            adjusted_context_ratio = max(self._prev_context_ratio - max_change, 
                                          min(self._prev_context_ratio + max_change, adjusted_context_ratio))
        
        context_ratio = adjusted_context_ratio
        self._prev_context_ratio = context_ratio # Store for next calculation
        
        # Allocate remaining budget to debate and synthesis
        remaining_budget_share = 1.0 - context_ratio
        debate_ratio = remaining_budget_share * 0.85 # Default split: 85% debate, 15% synthesis
        synthesis_ratio = remaining_budget_share * 0.15
        
        # Calculate token counts, ensuring minimums for critical phases
        context_tokens = max(300, int(available_tokens * context_ratio))
        debate_tokens = max(800, int(available_tokens * debate_ratio))
        synthesis_tokens = max(300, int(available_tokens * synthesis_ratio))
        
        # --- Normalize budgets to fit within available_tokens ---
        total_allocated = context_tokens + debate_tokens + synthesis_tokens
        if total_allocated > available_tokens:
            scale_factor = available_tokens / total_allocated
            context_tokens = int(context_tokens * scale_factor)
            debate_tokens = int(debate_tokens * scale_factor)
            synthesis_tokens = int(synthesis_tokens * scale_factor)
        
        # --- Re-enforce minimums and adjust if necessary ---
        # This section ensures minimums are met, potentially by borrowing from other phases.
        # It's a bit complex, but aims to preserve essential phase capabilities.
        if synthesis_tokens < 300:
            needed = 300 - synthesis_tokens
            debate_reduction = min(needed, debate_tokens - 800) # Try to take from debate first
            context_reduction = needed - debate_reduction
            
            debate_tokens = max(800, debate_tokens - debate_reduction)
            context_tokens = max(300, context_tokens - context_reduction)
            synthesis_tokens = 300
        elif debate_tokens < 800:
            needed = 800 - debate_tokens
            context_reduction = min(needed, context_tokens - 300) # Try to take from context
            debate_tokens = max(800, debate_tokens - context_reduction)
            context_tokens = max(300, context_tokens - context_reduction)
        elif context_tokens < 300:
            needed = 300 - context_tokens
            debate_tokens = max(800, debate_tokens - needed) # Take from debate
            context_tokens = 300
        
        # Final check to ensure total doesn't exceed available tokens after re-enforcing minimums
        final_total = context_tokens + debate_tokens + synthesis_tokens
        if final_total > available_tokens:
            trim_amount = final_total - available_tokens
            # Trim from debate first, then context if necessary
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
            # Ensure text is properly encoded for token counting to prevent errors
            try:
                text_encoded = prompt_text.encode('utf-8')
                text_for_tokenizer = text_encoded.decode('utf-8', errors='replace') 
            except UnicodeEncodeError:
                text_for_tokenizer = prompt_text.encode('utf-8', errors='replace').decode('utf-8', errors='replace')
                logger.warning(f"Fixed encoding issues in prompt for step '{step_name}' by replacing problematic characters.")

            actual_tokens = self.llm_provider.count_tokens(text_for_tokenizer, system_prompt=system_prompt)

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
            logger.info("No codebase context provided or analyzer not initialized, skipping context analysis")
            return {"domain": "General", "relevant_files": []}
        
        # Use the router's domain determination for consistency
        domain = self.persona_router.determine_domain(self.initial_prompt)
        
        # Find relevant files using the analyzer
        relevant_files = self.context_analyzer.find_relevant_files(self.initial_prompt)
        
        logger.info(f"Context analysis complete. Domain: {domain}, Relevant files: {len(relevant_files)}")
        
        return {
            "domain": domain,
            "relevant_files": relevant_files,
            # Keywords can be extracted here if needed by persona router
            "keywords": self.persona_router._extract_prompt_keywords(self.initial_prompt) 
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
        context_budget = self.phase_budgets.get("context", 200) # Default to 200 if not calculated
        
        for file_path, _ in context_analysis.get("relevant_files", []):  
            if file_path not in self.codebase_context:
                continue

            content = self.codebase_context[file_path]
            
            # Calculate remaining budget in characters, assuming ~4 chars/token
            remaining_budget_chars = (context_budget - current_context_tokens) * 4
            
            if remaining_budget_chars <= 0:
                break

            key_elements = self._extract_key_elements(content)
            # Extract relevant segment, respecting remaining character budget
            relevant_segment = self.context_analyzer.extract_relevant_code_segments(
                content, max_chars=int(remaining_budget_chars)
            )
            
            file_context_part = (
                f"File: {file_path}\n"
                f"Key elements: {key_elements}\n"
                f"Content snippet:\n```\n{relevant_segment}\n```\n"
            )
            
            # Estimate tokens for this file's context part
            estimated_file_tokens = self.llm_provider.count_tokens(file_context_part)
            
            if current_context_tokens + estimated_file_tokens > context_budget:
                logger.info(f"Skipping {file_path} due to context budget. Current: {current_context_tokens}, Estimated for file: {estimated_file_tokens}, Budget: {context_budget}")
                break
            
            context_parts.append(file_context_part)
            current_context_tokens += estimated_file_tokens
        
        logger.info(f"Prepared context with {len(context_parts)} files, total estimated tokens: {current_context_tokens}")
        return "".join(context_parts)
    
    def _prepare_self_analysis_context(self, context_analysis: Dict[str, Any]) -> str:
        """Prepare specialized context for self-analysis with core files prioritized."""
        if not self.codebase_context or not context_analysis.get("relevant_files"):
            return ""
        
        # Prioritize core Chimera files for self-analysis
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
        
        # Add core files first
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
        
        # Add other relevant files if budget allows
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
        # Use domain from context analysis, falling back to the initial domain setting
        domain_for_sequence = context_analysis.get("domain", self.domain) or "General"
        
        # Get base sequence from persona sets or default
        if domain_for_sequence and domain_for_sequence in self.persona_sets:
            base_sequence = self.persona_sets[domain_for_sequence]
        else:
            base_sequence = self.persona_sequence # Use the default sequence if domain not found
            if not base_sequence: # Fallback if default sequence is also empty
                base_sequence = ["Visionary_Generator", "Skeptical_Generator", "Impartial_Arbitrator"]

        # Pass context_analysis_results to determine_persona_sequence for dynamic adjustments
        final_sequence = self.persona_router.determine_persona_sequence(
            self.initial_prompt,
            intermediate_results=None, # No intermediate results at this stage
            context_analysis_results=context_analysis # Pass context analysis results
        )
        
        # Ensure uniqueness and maintain order
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

        max_retries = 2 # Number of retries for synthesis
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
                # Pass the raw output string to parse_and_validate
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
                    raise # Re-raise if it's the last attempt
            except Exception as e: # Catch other unexpected errors during processing
                logger.error(f"Unexpected error during final answer processing (attempt {attempt+1}): {e}", exc_info=True)
                self.intermediate_steps[f"Final_Answer_Processing_Error_Attempt_{attempt+1}"] = str(e)
                if attempt == max_retries:
                    raise LLMResponseValidationError(
                        f"Final answer processing failed after {max_retries} retries: {str(e)}",
                        invalid_response=raw_final_answer,
                        expected_schema="LLMOutput",
                        details={"processing_error": str(e)}
                    ) from e
        
        # This point should ideally not be reached if retries are handled correctly.
        # If it is, it means an error occurred that wasn't caught or re-raised.
        raise ChimeraError("Unexpected state: Final synthesis failed after all attempts.")
    
    def _calculate_cost(self) -> float:
        """Calculate the estimated cost based on token usage."""
        # This calculation is a simplification. Actual costs might vary by model and pricing tiers.
        # Using a placeholder cost per 1k tokens.
        return self.tokens_used * 0.000003 # Example cost: $0.003 per 1k tokens
    
    def _calculate_complexity_score(self, prompt: str) -> float:
        """
        Calculate a semantic complexity score for the prompt (0.0 to 1.0).
        This score influences the dynamic allocation of token budgets.
        """
        prompt_lower = prompt.lower()
        complexity = 0.0
        
        # Factor 1: Prompt length (normalized)
        # Longer prompts are generally more complex.
        length_factor = min(1.0, len(prompt) / 2000.0) # Normalize length to a 0-1 scale
        complexity += length_factor * 0.5 # Contribute up to 0.5 to complexity
        
        # Factor 2: Presence of technical keywords
        # Keywords related to code, analysis, or specific domains increase complexity.
        technical_keywords = [
            "code", "analyze", "refactor", "algorithm", "architecture", "system",
            "science", "research", "business", "market", "creative", "art",
            "security", "test", "deploy", "optimize", "debug"
        ]
        keyword_count = sum(1 for kw in technical_keywords if kw in prompt_lower)
        keyword_density = keyword_count / len(technical_keywords) if technical_keywords else 0
        complexity += keyword_density * 0.5 # Contribute up to 0.5 for keyword density
        
        # Ensure complexity score is within the [0.0, 1.0] range
        return max(0.0, min(1.0, complexity))

    # --- REFACTORED: run_debate ---
    def run_debate(self) -> Tuple[Any, Dict[str, Any]]:
        """
        Run the complete Socratic debate process and return the results.
        Orchestrates context analysis, persona sequencing, debate rounds, and synthesis.
        """
        try:
            # 1. Initial prompt token count
            initial_prompt_tokens = self._check_token_budget(self.initial_prompt, "initial_prompt_count")
            
            # 2. Context Analysis
            context_analysis = self._analyze_context()
            self.intermediate_steps["Context_Analysis"] = context_analysis
            
            # 3. Prepare Context String
            context_str = self._prepare_context(context_analysis)
            if context_str:
                # Check budget for the prepared context string
                self._check_token_budget(context_str, "context_preparation", phase="context")
            
            # 4. Generate Persona Sequence
            # Pass context_analysis results to the router for dynamic sequence generation.
            self.persona_sequence = self.persona_router.determine_persona_sequence(
                self.initial_prompt,
                intermediate_results=None, # No intermediate results at this stage
                context_analysis_results=context_analysis # Pass context analysis results
            )
            self.intermediate_steps["Persona_Sequence"] = self.persona_sequence
            
            # 5. Execute Debate Rounds
            # Pass context_str to debate rounds if it's needed for prompt construction.
            debate_output = self._execute_debate_rounds(self.persona_sequence, context_str) 
            
            # 6. Synthesize Final Answer
            final_answer = self._synthesize_final_answer(debate_output, requested_model_name=self.model_name)
            
            return final_answer, self.intermediate_steps
            
        except TokenBudgetExceededError as e:
            # Handle token budget issues with graceful degradation.
            return self.handle_token_budget_exceeded(e)
        except SchemaValidationError as sve:
            logger.error(f"Schema validation failed during debate: {sve}", exc_info=True)
            raise # Re-raise for app.py to handle
        except ChimeraError as ce:
            logger.error(f"Chimera-specific error during debate: {ce}", exc_info=True)
            raise # Re-raise for app.py to handle
        except Exception as e:
            logger.exception("Unexpected error during debate process")
            raise # Re-raise for app.py to handle

    def _execute_debate_rounds(self, persona_sequence: List[str], context_str: str) -> str:
        """Executes the debate rounds sequentially based on the persona sequence."""
        current_response = ""
        if not persona_sequence:
            raise ChimeraError("No persona sequence generated. Debate cannot proceed.")
        
        # Construct the initial prompt for the first persona.
        initial_debate_prompt = f"User's original prompt: {self.initial_prompt}\n"
        if context_str:
            initial_debate_prompt += f"Context:\n{context_str}\n"
        initial_debate_prompt += "Starting the debate..."

        # Run the first persona's round.
        first_persona_name = persona_sequence[0]
        current_response = self._run_debate_round(
            initial_debate_prompt, 
            first_persona_name,
            requested_model_name=self.model_name # Pass user's selected model
        )
        
        # Run subsequent persona rounds, feeding the previous response into the next.
        for persona_name in persona_sequence[1:]:
            current_response = self._run_debate_round(current_response, persona_name, requested_model_name=self.model_name)
        
        return current_response

    def handle_token_budget_exceeded(self, e: TokenBudgetExceededError) -> Tuple[Any, Dict[str, Any]]:
        """
        Handles TokenBudgetExceededError with multi-stage adaptive graceful degradation.
        This method attempts to recover from token budget issues by adjusting parameters.
        """
        logger.warning(f"Token budget exceeded: {str(e)}. Attempting adaptive graceful degradation.")
        
        # Stage 1: Reduce context ratio slightly and re-calculate budgets.
        # This is the least disruptive step.
        if self.settings.context_token_budget_ratio > 0.15: # If context ratio is not already at minimum
            new_context_ratio = max(0.15, self.settings.context_token_budget_ratio * 0.8) # Reduce by 20%
            logger.info(f"Reducing context ratio from {self.settings.context_token_budget_ratio:.2f} to {new_context_ratio:.2f} and recalculating budgets.")
            self.settings.context_token_budget_ratio = new_context_ratio # Update setting for recalculation
            self._calculate_token_budgets() # Recalculate budgets with new ratio
            # Retry the entire debate process with adjusted budgets.
            return self.run_debate_process()
        
        # Stage 2: Simplify persona sequence if context reduction wasn't enough.
        # This is done if the sequence is longer than a minimal set and contains non-essential personas.
        elif len(self.persona_sequence) > 3 and 'Impartial_Arbitrator' in self.persona_sequence:
            # Define a core set of essential personas for simplification.
            core_personas_for_simplification = ["Visionary_Generator", "Skeptical_Generator", "Constructive_Critic", "Impartial_Arbitrator"]
            # Filter the current sequence to keep only core personas.
            simplified_sequence = [p for p in self.persona_sequence if p in core_personas_for_simplification]
            
            if len(simplified_sequence) < len(self.persona_sequence): # Only proceed if simplification occurred.
                logger.info(f"Simplifying persona sequence to core personas: {simplified_sequence}")
                self.persona_sequence = simplified_sequence # Update the sequence.
                # Recalculate budgets based on potentially shorter debate.
                self._calculate_token_budgets() 
                # Retry the debate process with the simplified sequence.
                return self.run_debate_process()
        
        # Stage 3: Synthesize partial results if degradation fails to keep within budget.
        # This is the last resort, providing incomplete but potentially useful output.
        elif self.intermediate_steps: # Check if any steps have been completed.
            logger.warning("Graceful degradation failed to keep within token budget. Returning partial results.")
            partial_result = self._synthesize_partial_results()
            partial_result += "\n\n[WARNING: Output truncated due to token constraints - full analysis not possible]"
            self.intermediate_steps["Partial_Result_Warning"] = partial_result
            # Return partial results and intermediate steps.
            return partial_result, self.intermediate_steps
        
        else: # If no intermediate steps were even completed, re-raise the original error.
            logger.error("Adaptive degradation failed completely. Re-raising original error with diagnostic info.", exc_info=True)
            e.details = {
                **(e.details or {}),
                "degradation_failed": True,
                "context_ratio_attempted": self.settings.context_token_budget_ratio,
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
        # Recalculate budgets based on potentially adjusted settings (e.g., context ratio).
        self._calculate_token_budgets()
        
        # Re-analyze context and prepare context string with new budget.
        context_analysis = self._analyze_context()
        context_str = self._prepare_context(context_analysis)
        if context_str:
            self._check_token_budget(context_str, "context_preparation_retry", phase="context")
        
        # Regenerate persona sequence based on context analysis and potentially simplified list.
        self.persona_sequence = self._generate_persona_sequence(context_analysis)
        self.intermediate_steps["Persona_Sequence"] = self.persona_sequence # Update intermediate steps.
        
        # Re-execute debate rounds with the new sequence and potentially adjusted context.
        debate_output = self._execute_debate_rounds(self.persona_sequence, context_str)
        
        # Re-synthesize final answer.
        final_answer = self._synthesize_final_answer(debate_output, requested_model_name=self.model_name)
        return final_answer, self.intermediate_steps

    def _synthesize_partial_results(self) -> str:
        """
        Synthesizes available intermediate steps into a partial result string.
        This is a fallback mechanism when token budgets are severely exceeded.
        """
        partial_output = "--- Partial Debate Results ---\n\n"
        partial_output += f"Original Prompt: {self.initial_prompt}\n\n"
        
        # Collect and sort persona outputs for a structured partial result.
        persona_outputs = sorted([
            (name, result) for name, result in self.intermediate_steps.items()
            if name.endswith("_Output") and isinstance(result, str)
        ])
        
        for persona_name, output_text in persona_outputs:
            # Truncate output text to keep partial result concise.
            partial_output += f"### {persona_name.replace('_Output', '')}:\n"
            partial_output += f"{output_text[:300]}...\n\n" # Show first 300 chars
        
        partial_output += f"Total Tokens Used (approx): {self.tokens_used}\n"
        partial_output += f"Estimated Cost (approx): ${self._calculate_cost():.4f}\n"
        
        return partial_output

    # --- Additional helper methods for context analysis ---
    def _calculate_complexity_score(self, prompt: str) -> float:
        """
        Calculate a semantic complexity score for the prompt (0.0 to 1.0).
        This score influences the dynamic allocation of token budgets.
        """
        prompt_lower = prompt.lower()
        complexity = 0.0
        
        # Factor 1: Prompt length (normalized)
        # Longer prompts are generally more complex.
        length_factor = min(1.0, len(prompt) / 2000.0) # Normalize length to a 0-1 scale
        complexity += length_factor * 0.5 # Contribute up to 0.5 to complexity
        
        # Factor 2: Presence of technical keywords
        # Keywords related to code, analysis, or specific domains increase complexity.
        technical_keywords = [
            "code", "analyze", "refactor", "algorithm", "architecture", "system",
            "science", "research", "business", "market", "creative", "art",
            "security", "test", "deploy", "optimize", "debug"
        ]
        keyword_count = sum(1 for kw in technical_keywords if kw in prompt_lower)
        keyword_density = keyword_count / len(technical_keywords) if technical_keywords else 0
        complexity += keyword_density * 0.5 # Contribute up to 0.5 for keyword density
        
        # Ensure complexity score is within the [0.0, 1.0] range
        return max(0.0, min(1.0, complexity))

    # --- Additional helper methods for context analysis ---
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

# Additional helper functions (moved from app.py for better separation if needed, but kept here for now)
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