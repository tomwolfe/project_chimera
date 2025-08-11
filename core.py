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

# --- ADDED IMPORT ---\n# Import the corrected GeminiProvider from llm_provider.py
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

# The definition of TokenBudgetExceededError is now correctly imported from src.exceptions.py
# and does not need to be redefined here.

class SocraticDebate:
    def __init__(self, initial_prompt: str, api_key: str,
                 codebase_context: Optional[Dict[str, str]] = None, # Changed type hint to Dict[str, str]
                 settings: Optional[ChimeraSettings] = None,
                 all_personas: Optional[Dict[str, PersonaConfig]] = None,
                 persona_sets: Optional[Dict[str, List[str]]] = None, # Added persona_sets
                 persona_sequence: Optional[List[str]] = None, # Added persona_sequence
                 domain: Optional[str] = None, # Added domain
                 max_total_tokens_budget: int = 10000,
                 model_name: str = "gemini-2.5-flash-lite",
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
            model_name: Name of the LLM model to use
            status_callback: Callback function for updating UI status.
            rich_console: Rich Console instance for logging.
        """
        # Load settings, using defaults if not provided
        self.settings = settings or ChimeraSettings()
        # Apply the passed context_token_budget_ratio to settings
        # NOTE: This line in the original code might be redundant if settings are loaded and validated later.
        # However, it ensures the passed ratio is considered if settings are initialized here.
        self.settings.context_token_budget_ratio = context_token_budget_ratio # ADDED THIS LINE
        self.max_total_tokens_budget = max_total_tokens_budget # FIXED: Changed from 'max_total_tokens'
        self.tokens_used = 0 # Total tokens consumed across all LLM calls
        self.model_name = model_name
        
        # Initialize context analyzer
        self.context_analyzer = None
        self.codebase_context = None
        if codebase_context:
            self.codebase_context = codebase_context
            self.context_analyzer = ContextRelevanceAnalyzer()
            # Ensure context is a dict of strings, not just a single string
            if isinstance(codebase_context, dict):
                self.context_analyzer.compute_file_embeddings(self.codebase_context)
            else:
                logger.warning("codebase_context was not a dictionary, skipping embedding computation.")
        
        # Initialize persona router
        self.all_personas = all_personas or {}
        self.persona_sets = persona_sets or {}
        self.persona_sequence = persona_sequence or []
        self.domain = domain
        self.persona_router = PersonaRouter(self.all_personas)
        
        # Set up the LLM provider
        # This line now correctly instantiates the imported GeminiProvider
        self.llm_provider = GeminiProvider(api_key=api_key, model_name=model_name)
        
        # Store the initial prompt
        self.initial_prompt = initial_prompt
        
        # --- FIX START ---
        # Calculate initial prompt tokens and set self.initial_input_tokens.
        # This must happen after llm_provider and initial_prompt are available.
        try:
            # Use the LLM provider's count_tokens method for accuracy.
            # Pass None for system_prompt as it's not relevant for initial prompt token count.
            self.initial_input_tokens = self.llm_provider.count_tokens(self.initial_prompt, system_prompt=None)
        except Exception as e:
            logger.error(f"Failed to count tokens for initial prompt: {e}. Setting initial_input_tokens to 0.")
            self.initial_input_tokens = 0 # Fallback if token counting fails
        # --- FIX END ---

        # Initialize phase budgets dictionary - THIS IS THE FIX
        self.phase_budgets = {}

        # Calculate token budgets based on settings and prompt analysis
        # This call now has self.initial_input_tokens available.
        self._calculate_token_budgets()
        
        # Track the debate progress
        self.intermediate_steps = {}
        self.final_answer = None
        self.process_log = []
        
        # Status callback and console for UI updates
        self.status_callback = status_callback
        self.rich_console = rich_console or Console()
    
    # --- MODIFIED METHOD FOR SUGGESTION 1 ---\n
    # This method was part of the original code, not a new suggestion.
    # The LLM's suggestion was to modify _calculate_token_budgets.
    # def _calculate_context_ratio(self, base_ratio: float, complexity_score: float) -> float:
    #     """
    #     Placeholder for context ratio calculation, applying complexity and bounds.
    #     This function encapsulates the logic previously scattered and inconsistently applied.
    #     """
    #     calculated = base_ratio + (complexity_score * 0.05)
    #     # Apply consistent bounds: min 15%, max 35% for context ratio
    #     return max(0.15, min(0.35, calculated))

    # --- MODIFIED METHOD FOR #2 PRIORITY (ROBUSTNESS) ---\n
    def _calculate_token_budgets(self):
        """Calculate dynamic token budgets based on settings and prompt analysis."""
        
        # Import is_self_analysis_prompt here to ensure it's available
        from src.constants import is_self_analysis_prompt
        is_self_analysis = is_self_analysis_prompt(self.initial_prompt)
        
        # --- Enhancement: Use ratios from ChimeraSettings as base, then adjust by complexity ---
        # The original code used hardcoded base ratios (0.15, 0.75). This change leverages
        # the configurable ratios from ChimeraSettings for more flexibility.
        base_context_ratio = self.settings.context_token_budget_ratio
        base_debate_ratio = self.settings.debate_token_budget_ratio
        # --- END ENHANCEMENT ---
        
        # Calculate available tokens for the debate/synthesis phases
        # --- FIX START: Handle case where initial tokens exceed total budget ---
        # This is the critical fix identified by the LLM.
        if self.initial_input_tokens >= self.max_total_tokens_budget:
            logger.warning(f"Initial tokens ({self.initial_input_tokens}) exceed total budget ({self.max_total_tokens_budget}). "
                          "Adjusting max_total_tokens_budget to accommodate context phase and prevent negative allocations.")
            # Adjust the budget to be just enough to cover initial tokens plus a small buffer for context phase.
            # This prevents negative available_tokens and ensures at least some budget for context.
            self.max_total_tokens_budget = self.initial_input_tokens + 500 # Add a small buffer
        # --- FIX END ---
        
        available_tokens = self.max_total_tokens_budget - self.initial_input_tokens
        
        # --- Apply semantic complexity for more dynamic ratio calculation ---
        # Use the existing method _calculate_complexity_score
        complexity_score = self._calculate_complexity_score(self.initial_prompt)
        
        # Dynamic adjustment with bounds as per suggestion
        # The bounds (0.1-0.3 for context) are kept as per the LLM's rationale.
        # The prompt specified: context_ratio = max(0.1, min(0.3, base_ratio + complexity_score * 0.05))
        context_ratio = max(0.1, min(0.3, base_context_ratio + complexity_score * 0.05))
        
        # Distribute remaining budget between debate and synthesis
        # Using a common split: 85% debate, 15% synthesis
        remaining_budget_share = 1.0 - context_ratio
        debate_ratio = remaining_budget_share * 0.85
        synthesis_ratio = remaining_budget_share * 0.15
        
        # Calculate token allocations, ensuring minimums for critical phases
        # LLM suggested specific minimums: context=300, debate=800, synthesis=300
        context_tokens = max(300, int(available_tokens * context_ratio))
        debate_tokens = max(800, int(available_tokens * debate_ratio))
        synthesis_tokens = max(300, int(available_tokens * synthesis_ratio))
        
        # --- Final validation - ensure no negative values and minimums are met ---
        # This block ensures that even if ratios are extreme, minimums are respected
        # and the total doesn't exceed the (potentially adjusted) max_total_tokens_budget.
        # It also handles the case where synthesis_tokens might be less than its minimum.
        
        # Recalculate total allocated to ensure it doesn't exceed available_tokens
        total_allocated = context_tokens + debate_tokens + synthesis_tokens
        if total_allocated > available_tokens:
            # Scale down proportionally if over budget
            scale_factor = available_tokens / total_allocated
            context_tokens = int(context_tokens * scale_factor)
            debate_tokens = int(debate_tokens * scale_factor)
            synthesis_tokens = int(synthesis_tokens * scale_factor)
        
        # Ensure minimums are still met after scaling, adjusting from the largest phase (debate) first
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
        
        # Final check to ensure total doesn't exceed budget due to rounding or minimums
        final_total = context_tokens + debate_tokens + synthesis_tokens
        if final_total > available_tokens:
            # If still over, trim from the largest phase (debate)
            trim_amount = final_total - available_tokens
            debate_tokens = max(800, debate_tokens - trim_amount) # Ensure debate doesn't go below its minimum
            # Re-check if trimming debate made it too small, then trim context
            if debate_tokens < 800:
                debate_tokens = 800
                trim_amount = final_total - available_tokens - (debate_tokens - 800)
                context_tokens = max(300, context_tokens - trim_amount)
        
        # Assign budgets
        self.phase_budgets["context"] = context_tokens
        self.phase_budgets["debate"] = debate_tokens
        self.phase_budgets["synthesis"] = synthesis_tokens
        
        logger.info(f"Token budgets calculated: Context={self.phase_budgets['context']} ({context_ratio:.2%}), "
                   f"Debate={self.phase_budgets['debate']} ({debate_ratio:.2%}), "
                   f"Synthesis={self.phase_budgets['synthesis']} ({synthesis_ratio:.2%})")
    # --- END MODIFIED METHOD ---
    
    def _check_token_budget(self, prompt_text: str, step_name: str, system_prompt: str = "") -> int:
        """
        Check if using the specified tokens would exceed the budget using accurate counting.
        Returns the number of tokens used for this step.
        Raises TokenBudgetExceededError if budget is exceeded.
        """
        try:
            # --- FIX APPLIED HERE ---
            # Proper encoding for token counting to prevent errors with special characters.
            try:
                prompt_text = prompt_text.encode('utf-8').decode('utf-8')
            except UnicodeEncodeError:
                prompt_text = prompt_text.encode('utf-8', 'replace').decode('utf-8', 'replace')
                logger.warning(f"Fixed encoding issues in prompt for step '{step_name}' by replacing problematic characters.")
            # --- END FIX ---

            # Use GeminiProvider's accurate count_tokens for both prompt and system prompt.
            actual_tokens = self.llm_provider.count_tokens(prompt_text, system_prompt=system_prompt)

            if self.tokens_used + actual_tokens > self.max_total_tokens_budget:
                raise TokenBudgetExceededError(
                    current_tokens=self.tokens_used,
                    budget=self.max_total_tokens_budget,
                    details={"step": step_name, "tokens_requested": actual_tokens}
                )
            
            # Update total tokens used *after* successful check
            self.tokens_used += actual_tokens 
            return actual_tokens # Return tokens used for this specific call
        except TokenBudgetExceededError:
            raise # Re-raise if budget is exceeded
        except Exception as e:
            # Handle errors from the tokenizer itself or other unexpected issues
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
        
        # Extract keywords from the prompt
        # Assuming context_analyzer has a method to extract keywords from prompt
        # --- FIX APPLIED HERE ---
        # Proper keyword extraction - take first 5 unique words from prompt as placeholder
        prompt_words = self.initial_prompt.lower().split()
        keywords = list(dict.fromkeys(prompt_words))[:5]  # Remove duplicates, take first 5
        # --- END FIX ---
        
        # Find relevant files based on keywords
        relevant_files = self.context_analyzer.find_relevant_files(self.initial_prompt) # Corrected method name
        
        # Determine the domain based on the prompt
        # Use the router to determine domain, potentially using context analysis results
        domain = self.persona_router.determine_domain(self.initial_prompt) # Assuming determine_domain exists
        
        logger.info(f"Context analysis complete. Domain: {domain}, Relevant files: {len(relevant_files)}")
        
        return {
            "domain": domain,
            "relevant_files": relevant_files,
            "keywords": keywords # This might be a string representation of keywords
        }
    
    def _prepare_context(self, context_analysis: Dict[str, Any]) -> str:
        """Prepare the context for the debate based on the context analysis,
        respecting the context token budget."""
        
        # Check if this is a self-analysis prompt
        from src.constants import is_self_analysis_prompt
        if is_self_analysis_prompt(self.initial_prompt):
            return self._prepare_self_analysis_context(context_analysis)
        
        if not self.codebase_context or not context_analysis.get("relevant_files"):
            return ""
        
        context_parts = []
        current_context_tokens = 0
        
        # Iterate through all relevant files, ordered by relevance
        for file_path, _ in context_analysis.get("relevant_files", []):  
            if file_path not in self.codebase_context:
                continue

            content = self.codebase_context[file_path]
            
            # Use extract_relevant_code_segments for intelligent content selection
            # Pass a max_chars that is a fraction of remaining context budget
            # This is an estimate, actual tokens will be counted later.
            # A simple heuristic: 4 chars ~ 1 token.
            remaining_budget_chars = (self.phase_budgets.get("context", 200) - current_context_tokens) * 4 # Use phase_budgets for context
            
            # Ensure at least some content is extracted if budget allows
            if remaining_budget_chars <= 0:
                break # No more budget for context

            # Extract key elements and relevant code segments
            key_elements = self._extract_key_elements(content)
            relevant_segment = self.context_analyzer.extract_relevant_code_segments(
                content, max_chars=int(remaining_budget_chars)
            )
            
            # Construct the part for this file
            file_context_part = (
                f"File: {file_path}\n"
                f"Key elements: {key_elements}\n"
                f"Content snippet:\n```\n{relevant_segment}\n```\n"
            )
            
            # Check if adding this file's context would exceed the budget
            # Use the actual tokenizer for precise counting
            # --- MODIFICATION FOR SUGGESTION 6: Use cached token counting ---
            # estimated_file_tokens = self.llm_provider.count_tokens(file_context_part)
            estimated_file_tokens = self.llm_provider.count_tokens(file_context_part) # Use cached count_tokens
            # --- END MODIFICATION ---
            
            if current_context_tokens + estimated_file_tokens > self.phase_budgets.get("context", 200): # Use phase_budgets for context
                logger.info(f"Skipping {file_path} due to context budget. "
                            f"Current: {current_context_tokens}, Estimated for file: {estimated_file_tokens}, "
                            f"Budget: {self.phase_budgets.get('context', 200)}")
                break # Stop adding files if budget is exceeded
            
            context_parts.append(file_context_part)
            current_context_tokens += estimated_file_tokens
        
        logger.info(f"Prepared context with {len(context_parts)} files, total estimated tokens: {current_context_tokens}")
        return "".join(context_parts)
    
    def _prepare_self_analysis_context(self, context_analysis: Dict[str, Any]) -> str:
        """Prepare specialized context for self-analysis with core files prioritized."""
        if not self.codebase_context or not context_analysis.get("relevant_files"):
            return ""
        
        # For self-analysis, prioritize core system files
        core_files = [
            "src/core.py",
            "src/persona/routing.py",
            "src/token_manager.py",
            "src/constants.py",
            "src/exceptions.py",
            "src/models.py",
            "src/llm_provider.py",
            "src/utils/output_parser.py",
            "src/utils/code_validator.py",
            "src/utils/path_utils.py",
            "src/config/settings.py",
            "src/config/persistence.py",
            "src/persona_manager.py",
            "src/context/context_analyzer.py",
            "src/tokenizers/base.py",
            "src/tokenizers/gemini_tokenizer.py",
            "app.py" # Include app.py itself for analysis of the UI/orchestration layer
        ]
        
        context_parts = []
        current_context_tokens = 0
        context_budget = self.phase_budgets.get("context", 200) # Use phase budget for context
        
        # First add core files if they exist in the codebase context
        for file_path in core_files:
            if file_path in self.codebase_context:
                content = self.codebase_context[file_path]
                # Add the entire file for self-analysis of core components
                file_context_part = f"### {file_path}\n{content}\n"
                # --- MODIFICATION FOR SUGGESTION 6: Use cached token counting ---
                # estimated_file_tokens = self.llm_provider.count_tokens(file_context_part)
                estimated_file_tokens = self.llm_provider.count_tokens(file_context_part) # Use cached count_tokens
                # --- END MODIFICATION ---
                
                # Check if adding this file would exceed budget
                if current_context_tokens + estimated_file_tokens > context_budget:
                    logger.warning(f"Context budget exceeded while adding core file '{file_path}'. Stopping context preparation.")
                    break
                    
                context_parts.append(file_context_part)
                current_context_tokens += estimated_file_tokens
        
        # Then add other relevant files up to token budget
        for file_path, _ in context_analysis.get("relevant_files", []):
            # Skip if file is already included as a core file or not in context
            if file_path in core_files or file_path not in self.codebase_context:
                continue
                
            content = self.codebase_context[file_path]
            file_context_part = f"### {file_path}\n{content}\n"
            # --- MODIFICATION FOR SUGGESTION 6: Use cached token counting ---
            # estimated_file_tokens = self.llm_provider.count_tokens(file_context_part)
            estimated_file_tokens = self.llm_provider.count_tokens(file_context_part) # Use cached count_tokens
            # --- END MODIFICATION ---
            
            # Check if adding this file would exceed budget
            if current_context_tokens + estimated_file_tokens > context_budget:
                logger.warning(f"Context budget exceeded while adding relevant file '{file_path}'. Stopping context preparation.")
                break
                
            context_parts.append(file_context_part)
            current_context_tokens += estimated_file_tokens
        
        logger.info(f"Prepared self-analysis context with {len(context_parts)} files, total estimated tokens: {current_context_tokens}")
        return "".join(context_parts)
    # --- END MODIFICATION ---
    
    def _generate_persona_sequence(self, context_analysis: Dict[str, Any]) -> List[str]:
        """Generate the sequence of personas to participate in the debate."""
        # Use the domain determined from context analysis or provided domain
        domain_for_sequence = context_analysis.get("domain", self.domain) or "General"
        
        # If a domain is specified, use it to get the persona sequence
        if domain_for_sequence and domain_for_sequence in self.persona_sets:
            base_sequence = self.persona_sets[domain_for_sequence]
        else:
            # Fallback to a default sequence if domain is not found or not specified
            base_sequence = self.persona_sequence # Use the default sequence loaded from file
            if not base_sequence: # If default sequence is also empty
                base_sequence = ["Visionary_Generator", "Skeptical_Generator", "Impartial_Arbitrator"]

        # Use persona_router to dynamically adjust sequence based on prompt and intermediate results
        # For initial sequence generation, only prompt is available.
        final_sequence = self.persona_router.determine_persona_sequence(
            self.initial_prompt,
            intermediate_results=None # No intermediate results on the first pass
        )
        
        # Ensure the final sequence is unique and maintains a logical order.
        seen = set()
        unique_sequence = []
        for persona in final_sequence:
            if persona not in seen:
                unique_sequence.append(persona)
                seen.add(persona)
        
        return unique_sequence
    
    def _run_debate_round(self, current_response: str, persona_name: str) -> str:
        """Run a single round of the debate with the specified persona."""
        if persona_name not in self.all_personas:
            logger.warning(f"Persona '{persona_name}' not found. Skipping this round.")
            return current_response # Return previous response if persona is missing

        persona = self.all_personas[persona_name]
        
        # Create the prompt for this persona
        # Construct the prompt text that will be passed to the LLM
        prompt_for_llm = f"""
You are {persona_name}: {persona.description}
{persona.system_prompt}

Current debate state:
{current_response}

User's original prompt:
{self.initial_prompt}
"""
        
        # Check token budget for the input prompt and system instruction.
        # This call updates self.tokens_used with the input tokens.
        # It returns the count of input tokens for this specific call.
        input_tokens_for_call = self._check_token_budget(
            prompt_for_llm, 
            f"debate_round_{persona_name}", 
            system_prompt=persona.system_prompt
        )
        
        # Generate response using the new method
        logger.info(f"Running debate round with {persona_name}")
        # The generate method returns (response_text, input_tokens, output_tokens)
        generated_text, input_tokens, output_tokens = self.llm_provider.generate(
            prompt=prompt_for_llm,
            system_prompt=persona.system_prompt,
            temperature=persona.temperature,
            max_tokens=persona.max_tokens
        )
        
        # Update total tokens used with output tokens.
        # input_tokens_returned should match input_tokens_for_call, which was already added to self.tokens_used by _check_token_budget.
        self.tokens_used += output_tokens
        
        # Log the step details
        self.intermediate_steps[f"{persona_name}_Output"] = generated_text # Use generated_text
        self.intermediate_steps[f"{persona_name}_Input_Tokens"] = input_tokens # Log input tokens
        self.intermediate_steps[f"{persona_name}_Output_Tokens"] = output_tokens # Log output tokens
        self.process_log.append({
            "step": f"{persona_name}_Output",
            "input_tokens": input_tokens, # Log input tokens
            "output_tokens": output_tokens, # Log output tokens
            "response_length": len(generated_text)
        })
        
        # Return the generated text response
        return generated_text
    
    def _synthesize_final_answer(self, final_debate_state: str) -> Dict[str, Any]:
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
            # Return a structured error that can be handled by app.py
            return {
                "COMMIT_MESSAGE": "Error: Arbitrator Persona Missing",
                "RATIONALE": "The 'Impartial_Arbitrator' persona is required for synthesis but was not found in the loaded personas.",
                "CODE_CHANGES": [],
                "malformed_blocks": [{"type": "CRITICAL_CONFIG_ERROR", "message": "Arbitrator persona not found."}]
            }

        max_retries = 2 # Allow up to 2 retries for JSON formatting/schema issues
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

            # Check token budget for the synthesis step
            # Use phase budget for synthesis
            tokens_used_in_synthesis = self._check_token_budget(
                prompt_for_synthesis, 
                "final_synthesis", 
                system_prompt=arbitrator.system_prompt # Pass system prompt for accurate counting
            )
            
            # --- FIX APPLIED HERE ---
            # Original: raw_final_answer = self.llm_provider.generate_content(...)
            # Fixed: Use generate() and unpack the tuple. Add output_tokens to self.tokens_used.
            generated_text, input_tokens, output_tokens = self.llm_provider.generate(
                prompt=prompt_for_synthesis,
                system_prompt=arbitrator.system_prompt, # Pass system_prompt
                temperature=arbitrator.temperature,
                max_tokens=arbitrator.max_tokens
            )
            raw_final_answer = generated_text
            # Add output tokens to the total count for consistency with _run_debate_round
            self.tokens_used += output_tokens
            # --- END FIX ---
            
            # Attempt to parse and validate the raw output
            try:
                # Use LLMOutputParser to handle extraction and validation
                llm_output_parser = LLMOutputParser()
                # Assuming LLMOutput is correctly imported or defined elsewhere
                validated_output_dict = llm_output_parser.parse_and_validate(raw_final_answer, LLMOutput)
                
                # If successful, store and return
                self.final_answer = validated_output_dict
                self.intermediate_steps["Final_Answer_Output"] = validated_output_dict
                self.intermediate_steps["Final_Answer_Tokens_Used"] = tokens_used_in_synthesis # This is input tokens for synthesis
                # Total tokens used is updated by _check_token_budget calls (input) and the line above (output)
                self.intermediate_steps["Total_Tokens_Used"] = self.tokens_used 
                self.intermediate_steps["Total_Estimated_Cost_USD"] = self._calculate_cost()
                return validated_output_dict
            except SchemaValidationError as sve: # Catch specific schema validation errors
                logger.error(f"Schema validation failed for final answer (attempt {attempt+1}): {sve}")
                # Store the error details for reporting
                self.intermediate_steps[f"Final_Answer_Validation_Error_Attempt_{attempt+1}"] = {
                    "message": str(sve),
                    "details": sve.details
                }
                if attempt == max_retries:
                    # If max retries reached, raise the specific error to app.py
                    raise # Re-raise the caught SchemaValidationError
                # Continue to next attempt
            except Exception as e: # Catch any other unexpected errors from parser/validator
                logger.error(f"Unexpected error during final answer processing (attempt {attempt+1}): {e}")
                self.intermediate_steps[f"Final_Answer_Processing_Error_Attempt_{attempt+1}"] = str(e)
                if attempt == max_retries:
                    # If max retries reached, raise a generic error to app.py
                    raise LLMResponseValidationError(
                        f"Final answer processing failed after {max_retries} retries: {str(e)}",
                        invalid_response=raw_final_answer,
                        expected_schema="LLMOutput",
                        details={"processing_error": str(e)}
                    ) from e
                # Continue to next attempt
        
        # Should not be reached if max_retries logic is sound
        raise Exception("Unexpected state in _synthesize_final_answer.")
    
    def _calculate_cost(self) -> float:
        """Calculate the estimated cost based on token usage."""
        # This is a placeholder - actual cost calculation would depend on the model
        # For Gemini, as of 2023, pricing is approximately:
        # $0.00000025 per character for input, $0.0000005 per character for output
        
        # Simplified estimate: $0.000003 per token (as used in app.py)
        # This should ideally be derived from a configuration or model pricing lookup.
        return self.tokens_used * 0.000003
    
    # --- NEW HELPER FUNCTION FOR #2 PRIORITY (ROBUSTNESS) ---\n
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

    # --- NEW HELPER FUNCTION FOR #3 PRIORITY (EFFICIENCY) ---\n
    def _extract_quality_metrics(self, intermediate_results: Dict[str, Any]) -> Dict[str, float]:
        """
        Placeholder for extracting quality metrics from intermediate results.
        Analyzes text for keywords related to quality, security, etc.
        """
        metrics = {
            "context_quality": 0.5, # Default
            "security_risk_score": 0.5, # Default
            "code_quality": 0.5, # Default
            "maintainability_index": 0.5, # Default
            "test_coverage_estimate": 0.5, # Default
            "reasoning_depth": 0.5, # Added for dynamic routing
            "architectural_coherence": 0.5 # Added for dynamic routing
        }
        
        # Keywords for different metrics
        quality_keywords = {
            "code_quality": ["good code", "clean code", "readable", "well-structured", "efficient"],
            "security_risk_score": ["vulnerab", "risk", "exploit", "hack", "insecure", "threat", "malware", "attack"],
            "maintainability_index": ["maintainable", "easy to modify", "scalable", "modular", "well-documented"],
            "test_coverage_estimate": ["tested", "coverage", "unit test", "integration test", "robust testing"],
            "reasoning_depth": ["deep analysis", "thorough", "comprehensive", "nuanced", "insightful", "detailed"],
            "architectural_coherence": ["coherent architecture", "well-integrated", "consistent design", "logical structure"]
        }
        
        # Combine all intermediate results into a single text for analysis
        all_results_text = ""
        for step_name, result in intermediate_results.items():
            if isinstance(result, str):
                all_results_text += result.lower() + " "
            elif isinstance(result, dict):
                # Try to extract relevant text from dicts (e.g., rationale, concerns)
                for key, value in result.items():
                    if isinstance(value, str):
                        all_results_text += value.lower() + " "
                    elif isinstance(value, list): # For lists of concerns/modules
                        all_results_text += " ".join(str(v).lower() for v in value) + " "
        
        # Calculate scores based on keyword presence
        for metric_name, keywords in quality_keywords.items():
            score_sum = 0
            for keyword in keywords:
                if keyword in all_results_text:
                    score_sum += 1
            
            # Normalize score based on number of keywords for that metric
            if keywords:
                metrics[metric_name] = min(1.0, score_sum / len(keywords))
        
        # Special handling for context quality based on Context_Aware_Assistant output if available
        if "Context_Aware_Assistant_Output" in intermediate_results:
            caa_output = intermediate_results["Context_Aware_Assistant_Output"]
            if isinstance(caa_output, dict): # Check if it's a dict
                if "quality_metrics" in caa_output and isinstance(caa_output["quality_metrics"], dict):
                    quality_metrics_from_caa = caa_output["quality_metrics"]
                    if "overall_code_quality" in quality_metrics_from_caa:
                        metrics["code_quality"] = max(metrics["code_quality"], quality_metrics_from_caa["overall_code_quality"])
                    if "security_risk_score" in quality_metrics_from_caa:
                        metrics["security_risk_score"] = max(metrics["security_risk_score"], quality_metrics_from_caa["security_risk_score"])
                    if "maintainability_index" in quality_metrics_from_caa:
                        metrics["maintainability_index"] = max(metrics["maintainability_index"], quality_metrics_from_caa["maintainability_index"])
                    if "test_coverage_estimate" in quality_metrics_from_caa:
                        metrics["test_coverage_estimate"] = max(metrics["test_coverage_estimate"], quality_metrics_from_caa["test_coverage_estimate"])
        
        # Ensure scores are within bounds
        for key in metrics:
            metrics[key] = max(0.0, min(1.0, metrics[key]))
            
        return metrics

    def run_debate(self) -> Tuple[Any, Dict[str, Any]]: # Changed return type hint
        """
        Run the complete Socratic debate process and return the results.
        
        Returns:
            A tuple containing the final answer and a dictionary of intermediate steps.
        """
        try:
            # 1. Analyze context
            # Check token budget for initial prompt and context analysis phase.
            # The prompt_text for this check is the initial prompt.
            initial_prompt_tokens = self._check_token_budget(self.initial_prompt, "initial_prompt_count")
            
            context_analysis = self._analyze_context()
            
            # The token budgets are now calculated correctly in __init__ because
            # self.initial_input_tokens is set there.
            # No need to call _calculate_token_budgets() here again.
            
            self.intermediate_steps["Context_Analysis"] = context_analysis
            # 2. Prepare context
            context_str = self._prepare_context(context_analysis)
            # Count tokens for context preparation if it's significant
            if context_str:
                # Use _check_token_budget to account for context preparation tokens
                # Note: _check_token_budget already updates self.tokens_used
                # The prompt_text for context preparation is the context_str itself.
                self._check_token_budget(context_str, "context_preparation") # This call will update self.tokens_used
            
            # 3. Generate persona sequence
            # The persona_router.determine_persona_sequence is called here.
            # It might use context_analysis or other internal state.
            self.persona_sequence = self.persona_router.determine_persona_sequence(
                self.initial_prompt,
                intermediate_results=None # No intermediate results on the first pass
            )
            self.intermediate_steps["Persona_Sequence"] = self.persona_sequence
            
            # 4. Run initial generation (Visionary Generator)
            current_response = ""
            if self.persona_sequence:
                # The _run_debate_round method handles constructing the prompt and checking budget
                current_response = self._run_debate_round(
                    "No previous responses. Starting the debate.", 
                    self.persona_sequence[0]
                )
                
                # 5. Run subsequent debate rounds
                for persona_name in self.persona_sequence[1:]:
                    current_response = self._run_debate_round(current_response, persona_name)
            else:
                logger.warning("No persona sequence generated. Debate cannot proceed.")
                # Handle case where no personas are selected
                return (
                    "Error: No personas selected for debate.", # final_answer
                    { # intermediate_steps
                        "error": "No persona sequence generated.",
                        "process_log": self.process_log,
                        "Total_Tokens_Used": self.tokens_used,
                        "Total_Estimated_Cost_USD": self._calculate_cost()
                    }
                )
            
            # 6. Synthesize final answer
            final_answer = self._synthesize_final_answer(current_response)
            
            # 7. Return results as a tuple (final_answer, intermediate_steps)
            # app.py expects this format and retrieves other details from intermediate_steps.
            return final_answer, self.intermediate_steps
            
        except TokenBudgetExceededError as e:
            logger.warning(f"Token budget exceeded: {str(e)}")
            # Re-raise the exception to be caught by app.py's error handling.
            # app.py is designed to populate session state with error details from the caught exception.
            raise 
        except Exception as e:
            logger.exception("Unexpected error during debate process")
            # Re-raise the exception to be caught by the app.py handler
            raise

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
            # Convert YAML data to PersonaConfig
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
            # Convert YAML data to ReasoningFrameworkConfig
            frameworks[framework_name] = ReasoningFrameworkConfig(
                framework_name=framework_name,
                personas={}, # Personas are loaded separately into all_personas
                persona_sets=framework_data.get('persona_sets', {}),
                version=framework_data.get('version', 1) # Load version if present
            )
        except (ValidationError, KeyError) as e:
            logger.error(f"Invalid framework data in {yaml_path} for framework '{framework_name}': {e}")
    
    return frameworks