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
from typing import List, Dict, Tuple, Callable, Optional, Type, Any # This line is already present and correct.

# --- FIX START ---
# Import the necessary Pydantic models from src.models that are used in PERSONA_OUTPUT_SCHEMAS
# Added ChimeraSettings to this import statement.
from src.models import PersonaConfig, LLMOutput, ContextAnalysisOutput, CritiqueOutput, ChimeraSettings, CodeChange, ReasoningFrameworkConfig
# --- FIX END ---

from google import genai
from google.genai import types
from google.genai.errors import APIError # Import APIError
import traceback # Needed for error handling in core.py
from rich.console import Console
from pydantic import ValidationError, BaseModel
from functools import lru_cache # Import lru_cache for caching
import signal # Import signal for graceful shutdown

# Configure logging
logger = logging.getLogger(__name__)

# --- Placeholder for GeminiProvider and PersonaRouter ---
# These classes are assumed to exist and be imported from other modules (e.g., src.llm_providers, src.persona_manager)
# For this example, we'll define minimal placeholder classes if they aren't provided.
# In a real project, these would be fully implemented.

class LLMProviderError(Exception):
    """Custom exception for LLM provider errors."""
    pass

class TokenBudgetExceededError(Exception):
    """Custom exception for token budget exceeded errors."""
    def __init__(self, message, current_tokens, budget, details=None):
        super().__init__(message)
        self.current_tokens = current_tokens
        self.budget = budget
        self.details = details or {}

class ChimeraError(Exception):
    """Custom exception for Chimera-specific errors."""
    def __init__(self, message, details=None):
        super().__init__(message)
        self.details = details or {}

class ContextRelevanceAnalyzer:
    """Placeholder for ContextRelevanceAnalyzer."""
    def __init__(self):
        self.file_embeddings = {}
        self.persona_router = None
        logger.info("ContextRelevanceAnalyzer initialized (placeholder).")

    def set_persona_router(self, router):
        self.persona_router = router
        logger.info("PersonaRouter set for ContextRelevanceAnalyzer.")

    def compute_file_embeddings(self, codebase_context: Dict[str, str]):
        """Placeholder for computing embeddings."""
        logger.info(f"Computing embeddings for {len(codebase_context)} files (placeholder).")
        # Simulate embedding computation
        for fname, content in codebase_context.items():
            self.file_embeddings[fname] = hashlib.md5(content.encode()).hexdigest()[:10] # Dummy embedding
        logger.info("Embeddings computed (placeholder).")

    def find_relevant_files(self, prompt: str, active_personas: List[str]) -> List[Tuple[str, float]]:
        """Placeholder for finding relevant files."""
        logger.info(f"Finding relevant files for prompt: '{prompt[:50]}...' (placeholder).")
        if not self.file_embeddings:
            return []
        # Simulate relevance scoring
        relevant = []
        for fname, embedding in self.file_embeddings.items():
            score = random.random() # Dummy score
            if score > 0.5: # Arbitrary threshold
                relevant.append((fname, score))
        relevant.sort(key=lambda item: item[1], reverse=True)
        logger.info(f"Found {len(relevant)} relevant files (placeholder).")
        return relevant[:5] # Return top 5

class PersonaRouter:
    """Placeholder for PersonaRouter."""
    def __init__(self, all_personas: Dict[str, PersonaConfig], persona_sets: Dict[str, List[str]]):
        self.all_personas = all_personas
        self.persona_sets = persona_sets
        logger.info(f"PersonaRouter initialized with {len(all_personas)} personas and {len(persona_sets)} persona sets (placeholder).")

    def determine_persona_sequence(self, prompt: str, domain: Optional[str], intermediate_results: Dict[str, Any], context_analysis_results: Optional[Dict[str, Any]] = None) -> List[str]:
        """Placeholder for determining persona sequence."""
        logger.info(f"Determining persona sequence for prompt: '{prompt[:50]}...', domain: {domain} (placeholder).")
        if domain and domain in self.persona_sets:
            sequence = self.persona_sets[domain]
            # Ensure all personas in the sequence are actually loaded
            valid_sequence = [p for p in sequence if p in self.all_personas]
            if len(valid_sequence) < len(sequence):
                logger.warning(f"Some personas in domain '{domain}' sequence were not found in loaded personas.")
            if valid_sequence:
                return valid_sequence
        
        # Fallback sequence if domain is not found or invalid
        fallback_sequence = ["Generalist_Assistant", "Constructive_Critic", "Impartial_Arbitrator"]
        valid_fallback = [p for p in fallback_sequence if p in self.all_personas]
        if valid_fallback:
            return valid_fallback
        else:
            # If even fallback personas are missing, return a minimal sequence
            return ["Generalist_Assistant"] if "Generalist_Assistant" in self.all_personas else list(self.all_personas.keys())[:1]

class GeminiProvider:
    """Placeholder for GeminiProvider."""
    def __init__(self, api_key: str, model_name: str, rich_console: Optional[Console] = None):
        self.api_key = api_key
        self.model_name = model_name
        self.rich_console = rich_console
        self.tokenizer = self # Dummy tokenizer
        self.model_pricing = {"gemini-2.5-flash-lite": {"input": 0.000125, "output": 0.000375},
                              "gemini-2.5-flash": {"input": 0.000125, "output": 0.000375},
                              "gemini-2.5-pro": {"input": 0.0005, "output": 0.0015}} # Example pricing
        
        if not api_key:
            raise LLMProviderError("API key is missing.")
        
        # Simulate model availability check
        if model_name not in self.model_pricing:
            raise LLMProviderError(f"Model '{model_name}' is not supported or pricing is unknown.")
        
        logger.info(f"GeminiProvider initialized for model: {model_name} (placeholder).")

    def count_tokens(self, prompt: str, system_prompt: Optional[str] = None) -> int:
        """Dummy token count."""
        # Simple heuristic: count words + 10% for structure, plus system prompt tokens
        prompt_tokens = len(prompt.split()) * 1.1
        if system_prompt:
            prompt_tokens += len(system_prompt.split()) * 1.1
        return int(prompt_tokens) + 50 # Add a base overhead

    def calculate_usd_cost(self, input_tokens: int, output_tokens: int) -> float:
        """Calculates estimated USD cost."""
        pricing = self.model_pricing.get(self.model_name, {"input": 0.0001, "output": 0.0001}) # Default low pricing
        cost = (input_tokens * pricing["input"]) + (output_tokens * pricing["output"])
        return cost

    def generate(self, prompt: str, system_prompt: Optional[str] = None, temperature: float = 0.7, max_tokens: int = 1024, requested_model_name: str = "", persona_config: Optional[PersonaConfig] = None, intermediate_results: Dict[str, Any] = {}) -> Tuple[str, int, int]:
        """Simulates LLM generation."""
        logger.info(f"Simulating LLM generation for model: {requested_model_name}, temp: {temperature}, max_tokens: {max_tokens}")
        
        # Simulate token usage
        input_tokens = self.count_tokens(prompt=prompt, system_prompt=system_prompt)
        
        # Adjust output tokens based on max_tokens and a random factor
        output_tokens = min(max_tokens, int(input_tokens * random.uniform(0.5, 1.5))) + random.randint(50, 200)
        output_tokens = max(50, output_tokens) # Ensure at least some output tokens

        # Simulate response based on persona and prompt
        response_content = f"Simulated response for persona '{persona_config.name if persona_config else 'Unknown'}' based on prompt: '{prompt[:100]}...'"
        
        # Add some variation based on persona and temperature
        if persona_config:
            if "Critic" in persona_config.name:
                response_content = f"Critique: This is a simulated critique. The prompt '{prompt[:50]}...' is interesting. Suggestion: Improve clarity. (Temp: {temperature})"
            elif "Arbitrator" in persona_config.name:
                response_content = f"Synthesis: Based on the debate, the final plan is to implement feature X. (Temp: {temperature})"
            elif "Assistant" in persona_config.name:
                response_content = f"Analysis: The context provided suggests focusing on file Y. (Temp: {temperature})"
        
        # Simulate potential JSON output for specific personas
        if persona_config and persona_config.name == "Impartial_Arbitrator":
            # Simulate valid JSON output
            simulated_json_output = {
                "COMMIT_MESSAGE": "FEAT: Implement core debate logic",
                "RATIONALE": "Synthesized debate results into a functional core logic.",
                "CODE_CHANGES": [
                    {
                        "FILE_PATH": "core.py",
                        "ACTION": "MODIFY",
                        "FULL_CONTENT": "def run_debate(...):\n    # ... implementation ...\n    pass"
                    }
                ],
                "malformed_blocks": []
            }
            response_content = json.dumps(simulated_json_output)
        elif persona_config and persona_config.name == "General_Synthesizer":
            response_content = "This is a general synthesis of the discussion. Key points are A, B, and C."

        # Simulate token budget check failure if needed (for testing)
        # if self.count_tokens(response_content) > max_tokens * 0.9: # If output is too large
        #     raise TokenBudgetExceededError("Simulated token budget exceeded", current_tokens=self.count_tokens(response_content), budget=max_tokens)

        return response_content, input_tokens, output_tokens

class LLMOutputParser:
    """Placeholder for LLMOutputParser."""
    def __init__(self):
        logger.info("LLMOutputParser initialized (placeholder).")

    def parse_and_validate(self, response_text: str, schema: Type[BaseModel]) -> Dict[str, Any]:
        """Parses and validates LLM output against a Pydantic schema."""
        logger.info(f"Parsing and validating response against schema: {schema.__name__} (placeholder).")
        
        # Try to parse as JSON first
        try:
            data = json.loads(response_text)
        except json.JSONDecodeError:
            # If it's not JSON, check if the schema expects raw text or has a specific error handling for non-JSON
            if schema == LLMOutput: # Special handling for LLMOutput which might expect JSON
                logger.warning("Response is not valid JSON, attempting to parse as LLMOutput error structure.")
                # Try to find malformed_blocks or a general error structure
                if "malformed_blocks" in response_text or "LLM_GENERATION_ERROR" in response_text:
                    # Attempt to extract error structure if it's not pure JSON
                    # This is a very basic heuristic
                    error_data = {"malformed_blocks": [{"type": "JSON_DECODE_ERROR", "message": "Response was not valid JSON.", "raw_output": response_text[:500]}]}
                    if "LLM_GENERATION_ERROR" in response_text:
                        error_data["COMMIT_MESSAGE"] = "LLM_GENERATION_ERROR"
                        error_data["RATIONALE"] = "LLM failed to produce valid JSON output."
                    return error_data
                else:
                    return {"malformed_blocks": [{"type": "JSON_DECODE_ERROR", "message": "Response was not valid JSON.", "raw_output": response_text[:500]}]}
            else: # For other schemas, assume it's raw text if not JSON
                return {"general_output": response_text, "malformed_blocks": []}

        # If it's JSON, validate against the schema
        try:
            # Use model_validate for Pydantic v2 compatibility
            validated_data = schema.model_validate(data)
            
            # Ensure malformed_blocks is present, even if empty
            if not isinstance(validated_data.model_dump().get("malformed_blocks"), list):
                validated_data.model_dump()["malformed_blocks"] = []
            
            return validated_data.model_dump() # Return as dict
        except ValidationError as e:
            logger.error(f"Schema validation failed: {e}")
            # Return a structured error indicating validation failure
            return {
                "malformed_blocks": [{
                    "type": "SCHEMA_VALIDATION_ERROR",
                    "message": f"Validation failed: {e}",
                    "raw_output": response_text[:500]
                }]
            }
        except Exception as e:
            logger.error(f"Unexpected error during parsing/validation: {e}")
            return {
                "malformed_blocks": [{
                    "type": "UNEXPECTED_PARSING_ERROR",
                    "message": f"An unexpected error occurred: {e}",
                    "raw_output": response_text[:500]
                }]
            }

# --- End Placeholder Definitions ---


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
                 settings: Optional[ChimeraSettings] = None, # ChimeraSettings is now importable
                 all_personas: Optional[Dict[str, PersonaConfig]] = None,
                 persona_sets: Optional[Dict[str, List[str]]] = None,
                 domain: Optional[str] = None,
                 max_total_tokens_budget: int = 10000,
                 model_name: str = "gemini-2.5-flash-lite",
                 status_callback: Optional[Callable] = None,
                 rich_console: Optional[Console] = None,
                 context_token_budget_ratio: float = 0.25,
                 context_analyzer: Optional[ContextRelevanceAnalyzer] = None,
                 # --- MODIFICATION: Add is_self_analysis parameter ---
                 is_self_analysis: bool = False
                 # --- END MODIFICATION ---
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
            is_self_analysis: Flag indicating if the current prompt is for self-analysis.
        """
        # If settings is None, ChimeraSettings() will be called. This requires ChimeraSettings to be defined.
        self.settings = settings or ChimeraSettings() 
        self.context_token_budget_ratio = context_token_budget_ratio
        self.max_total_tokens_budget = max_total_tokens_budget
        self.tokens_used = 0 # Total tokens used across all phases
        self.model_name = model_name
        self.status_callback = status_callback
        self.rich_console = rich_console
        self.is_self_analysis = is_self_analysis # Store the flag
        
        # --- FIX START ---
        # Initialize logger for the class
        self.logger = logging.getLogger(self.__class__.__name__)
        # --- FIX END ---
        
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
                self.logger.error(f"Failed to initialize LLM provider: {e}")
            # Re-raise as ChimeraError for consistent error handling in the app
            raise ChimeraError(f"LLM provider initialization failed: {e}") from e
        except Exception as e: # Catch any other unexpected errors during initialization
            if self.rich_console:
                self.rich_console.print(f"[red]An unexpected error occurred during LLM provider initialization: {e}[/red]")
            else:
                self.logger.error(f"An unexpected error occurred during LLM provider initialization: {e}")
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
        # REMOVED: self.persona_sequence = persona_sequence or [] # This is now determined dynamically
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
                    self.logger.error(f"Failed to compute embeddings for codebase context: {e}")
                    if self.status_callback:
                        self.status_callback(message=f"[red]Error computing context embeddings: {e}[/red]")
            else:
                self.logger.warning("codebase_context was not a dictionary, skipping embedding computation.")

    def _calculate_token_budgets(self):
        """
        Calculates token budgets for different phases of the debate using
        max_total_tokens_budget and context_token_budget_ratio.
        Handles potential errors during LLM provider interactions.
        Dynamically adjusts ratios for self-analysis tasks.
        """
        # Determine which ratios to use based on the is_self_analysis flag
        if self.is_self_analysis:
            context_ratio = max(0.05, min(0.5, self.settings.self_analysis_context_ratio))
            # Let's allocate a fixed portion for synthesis, and debate gets the rest of the non-context budget.
            synthesis_ratio = 0.2 # Fixed portion for synthesis
            debate_ratio = max(0.1, 1.0 - context_ratio - synthesis_ratio) # Debate gets remaining budget
            self.logger.info(f"Self-analysis mode detected. Using context ratio: {context_ratio:.2f}, debate ratio: {debate_ratio:.2f}, synthesis ratio: {synthesis_ratio:.2f}")
        else:
            # Use standard ratios for non-self-analysis tasks
            context_ratio = max(0.05, min(0.5, self.context_token_budget_ratio))
            # Calculate remaining ratio for debate and synthesis (e.g., 50/50)
            remaining_ratio = 1.0 - context_ratio
            debate_ratio = remaining_ratio / 2.0
            synthesis_ratio = remaining_ratio / 2.0
            self.logger.info(f"Standard mode. Using context ratio: {context_ratio:.2f}, debate ratio: {debate_ratio:.2f}, synthesis ratio: {synthesis_ratio:.2f}")

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

            self.logger.info(f"SocraticDebate token budgets initialized: "
                       f"Initial Input={self.initial_input_tokens}, "
                       f"Context={self.phase_budgets['context']}, "
                       f"Debate={self.phase_budgets['debate']}, "
                       f"Synthesis={self.phase_budgets['synthesis']}")

        except LLMProviderError as e:
            self.logger.error(f"LLM Provider Error during token calculation: {e}")
            # Provide fallback budgets if token calculation fails
            self.phase_budgets = {"context": 500, "debate": 15000, "synthesis": 1000}
            self.initial_input_tokens = 0
            raise ChimeraError(f"LLM provider error: {e}") from e
        except Exception as e:
            self.logger.error(f"An unexpected error occurred calculating token budgets: {e}")
            # Provide fallback budgets if token calculation fails
            self.phase_budgets = {"context": 500, "debate": 15000, "synthesis": 1000}
            self.initial_input_tokens = 0
            raise ChimeraError("Failed to calculate token budgets due to an unexpected error.") from e
    
    def track_token_usage(self, phase: str, tokens: int):
        """Tracks token usage for a specific phase and updates total used tokens."""
        if phase in self.tokens_used_per_phase:
            self.tokens_used_per_phase[phase] += tokens
        else:
            self.logger.warning(f"Attempted to track tokens for unknown phase: {phase}")
        self.tokens_used += tokens # Always update total tokens used

    def check_budget(self, phase: str, tokens_needed: int, step_name: str):
        """Checks if adding tokens_needed would exceed the budget for the given phase."""
        if phase not in self.phase_budgets:
            self.logger.warning(f"Phase '{phase}' not found in budget configuration.")
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
            # Rough estimate of output tokens: total used minus initial input tokens
            # This is an approximation, as actual output tokens per turn are tracked.
            # A more precise calculation would sum output tokens from each turn.
            estimated_output_tokens = max(0, self.tokens_used - self.initial_input_tokens)
            cost = self.llm_provider.calculate_usd_cost(
                input_tokens=self.initial_input_tokens,
                output_tokens=estimated_output_tokens
            )
            return cost
        except Exception as e:
            self.logger.error(f"Could not estimate total cost: {e}")
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
                    current_total_cost=self.get_total_estimated_cost(),
                    progress_pct=0.2 # Set initial progress
                )
            
            # 3. Process Context Persona Turn (if applicable)
            if self.status_callback:
                self.status_callback(
                    message="Analyzing context...",
                    state="running",
                    current_total_tokens=self.get_total_used_tokens(),
                    current_total_cost=self.get_total_estimated_cost(),
                    progress_pct=0.25 # Slightly advance progress
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
                    current_total_cost=self.get_total_estimated_cost(),
                    progress_pct=0.7 # Set progress before synthesis
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
                    current_total_cost=self.get_total_estimated_cost(),
                    progress_pct=1.0 # Final progress
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

        return final_answer, self.intermediate_steps # Return final answer and steps

    def _initialize_debate_state(self):
        """Initializes state variables for a new debate run."""
        self.intermediate_steps = {}
        self.final_answer = None
        self.tokens_used_per_phase = {"context": 0, "debate": 0, "synthesis": 0}
        self.tokens_used = self.initial_input_tokens # Start with initial input tokens
        self.logger.debug("Debate state initialized.")
        
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
                        current_total_cost=self.get_total_estimated_cost(),
                        progress_pct=0.1 # Set progress for context analysis start
                    )
                initial_sequence_for_relevance = self.persona_router.determine_persona_sequence(
                    self.initial_prompt,
                    domain=self.domain, # Pass domain to determine_persona_sequence
                    intermediate_results=self.intermediate_steps
                )
                
                relevant_files_info = self.context_analyzer.find_relevant_files(
                    self.initial_prompt,
                    active_personas=initial_sequence_for_relevance
                )
                context_analysis_results = {"relevant_files": relevant_files_info}
                self.intermediate_steps["Relevant_Files_Context"] = {"relevant_files": relevant_files_info}
                self.logger.info(f"Context analysis completed. Found {len(relevant_files_info)} relevant files.")
                if self.status_callback:
                    self.status_callback(
                        message=f"Context analysis complete. Found [bold]{len(relevant_files_info)}[/bold] relevant files.",
                        state="running",
                        current_total_tokens=self.get_total_used_tokens(),
                        current_total_cost=self.get_total_estimated_cost(),
                        progress_pct=0.2 # Set progress after context analysis
                    )
            except Exception as e:
                self.logger.error(f"Error during context analysis file finding: {e}")
                self.intermediate_steps["Context_Analysis_Error"] = {"error": str(e)}
                if self.status_callback:
                    self.status_callback(message=f"[red]Error during context analysis: {e}[/red]", state="warning")
        else:
            self.logger.info("No context analyzer or codebase context available. Skipping context analysis.")
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
        self.logger.info(f"Final persona sequence determined: {unique_sequence}")
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
                self.logger.info(f"Performing context processing with persona: {context_processing_persona_name}")
                if self.status_callback:
                    self.status_callback(
                        message=f"Running [bold]{context_processing_persona_name}[/bold] for context processing...",
                        state="running",
                        current_total_tokens=self.get_total_used_tokens(),
                        current_total_cost=self.get_total_estimated_cost(),
                        progress_pct=0.25, # Set progress for this phase
                        current_persona_name=context_processing_persona_name # Pass current persona
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
                        current_total_cost=self.get_total_estimated_cost(),
                        progress_pct=0.3 # Advance progress
                    )
                return turn_results
                
            except TokenBudgetExceededError as e:
                self.logger.error(f"Token budget exceeded during context processing: {e}")
                raise e
            except Exception as e:
                self.logger.error(f"Error during context processing generation for {context_processing_persona_name}: {e}")
                self.intermediate_steps[f"{context_processing_persona_name}_Error"] = str(e)
                error_tokens = self.llm_provider.count_tokens(f"Error processing {context_processing_persona_name}: {str(e)}") + 50
                self.track_token_usage("context", error_tokens)
                self.check_budget("context", 0, f"Error handling {context_processing_persona_name} context processing")
                if self.status_callback:
                    self.status_callback(
                        message=f"[red]Error with persona [bold]{context_processing_persona_name}[/bold]: {e}[/red]",
                        state="error",
                        current_total_tokens=self.get_total_used_tokens(),
                        current_total_cost=self.get_total_estimated_cost(),
                        progress_pct=0.3 # Set progress even on error
                    )
        else:
            self.logger.info("No dedicated context processing persona found or no context available. Skipping dedicated context processing phase.")
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
                current_total_cost=self.get_total_estimated_cost(),
                progress_pct=0.35 # Set progress for debate phase start
                )
        # Check if the last persona in the sequence is a designated synthesizer
        if persona_sequence and persona_sequence[-1] in ["Impartial_Arbitrator", "General_Synthesizer"]:
            synthesis_persona_name = persona_sequence[-1]
        
        # Define which personas to run in the debate loop (exclude the synthesis persona if it's the last one)
        debate_personas_to_run = persona_sequence
        if synthesis_persona_name and persona_sequence[-1] == synthesis_persona_name:
            debate_personas_to_run = persona_sequence[:-1] # Exclude the last one for the debate loop

        for i, persona_name in enumerate(debate_personas_to_run):
            if persona_name not in self.all_personas:
                self.logger.warning(f"Persona '{persona_name}' not found in loaded personas. Skipping.")
                if self.status_callback:
                    self.status_callback(
                        message=f"[yellow]Skipping persona '{persona_name}' (not found).[/yellow]",
                        state="running",
                        current_total_tokens=self.get_total_used_tokens(),
                        current_total_cost=self.get_total_estimated_cost(),
                        progress_pct=0.35 + (0.3 / len(debate_personas_to_run) if debate_personas_to_run else 0), # Distribute progress
                        current_persona_name=persona_name # Indicate which persona is being skipped
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
                self.logger.info(f"Executing debate turn with persona: {persona_name}")
                if self.status_callback:
                    self.status_callback(
                        message=f"Running persona: [bold]{persona_name}[/bold]...",
                        state="running",
                        current_total_tokens=self.get_total_used_tokens(),
                        current_total_cost=self.get_total_estimated_cost(),
                        progress_pct=0.35 + (0.3 * (i + 1) / len(debate_personas_to_run) if debate_personas_to_run else 0), # Distribute progress
                        current_persona_name=persona_name # Pass current persona
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
                            current_total_cost=self.get_total_estimated_cost(),
                            progress_pct=0.35 + (0.3 * (i + 1) / len(debate_personas_to_run) if debate_personas_to_run else 0), # Distribute progress
                            current_persona_name=None # Clear persona name after completion
                        )
                    all_debate_turns.append(turn_results)
            except TokenBudgetExceededError as e:
                self.logger.error(f"Token budget exceeded during debate turn for persona {persona_name}: {e}")
                raise e
            except Exception as e:
                self.logger.error(f"An unexpected error occurred during debate turn for persona {persona_name}: {e}")
                self.intermediate_steps[f"{persona_name}_Error"] = str(e)
                error_tokens = self.llm_provider.count_tokens(f"Error processing {persona_name}: {str(e)}") + 50
                self.track_token_usage("debate", error_tokens)
                self.check_budget("debate", 0, f"Error handling {persona_name} debate turn")
                if self.status_callback:
                    self.status_callback(
                        message=f"[red]Error with persona [bold]{persona_name}[/bold]: {e}[/red]",
                        state="error",
                        current_total_tokens=self.get_total_used_tokens(),
                        current_total_cost=self.get_total_estimated_cost(),
                        progress_pct=0.35 + (0.3 * (i + 1) / len(debate_personas_to_run) if debate_personas_to_run else 0), # Distribute progress
                        current_persona_name=persona_name # Indicate persona with error
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
                    current_total_cost=self.get_total_estimated_cost(),
                    progress_pct=0.7 # Set progress for synthesis start
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
                            self.logger.info(f"Synthesis output validated successfully on attempt {attempt + 1}.")
                            return current_synthesis_output # Success! Return the validated LLMOutput dict
                        else:
                            # Output is a failure, proceed to retry if possible
                            if attempt < max_retries:
                                self.logger.warning(f"Synthesis output validation failed on attempt {attempt + 1} ({failure_reason}). Retrying...")
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
                                self.logger.error(f"Synthesis output validation failed after {max_retries} retries.")
                                # If all retries fail, return the last recorded result
                                return current_synthesis_output # Return the last result, which will contain error information

                    else: # No JSON validation required (e.g., General_Synthesizer)
                        self.logger.debug(f"Synthesis output for {synthesis_persona_name} does not require strict JSON validation. Returning raw output.")
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
                            current_total_cost=self.get_total_estimated_cost(),
                            progress_pct=0.9 # Set progress to near completion even on error
                        )
                    self.logger.error(f"Error during synthesis turn execution: {e}")
                    if attempt == max_retries:
                        self.logger.error(f"Final synthesis attempt failed due to execution error: {e}")
                        # If it's the last attempt and execution fails, return a specific error
                        return {
                            "COMMIT_MESSAGE": "Synthesis Execution Error",
                            "RATIONALE": f"An error occurred during the final synthesis turn: {str(e)}",
                            "CODE_CHANGES": [],
                            "malformed_blocks": [{"type": "SYNTHESIS_EXECUTION_ERROR", "message": str(e)}]
                        }
                    else:
                        self.logger.warning(f"Execution error on synthesis attempt {attempt + 1}, retrying...")
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
            self.logger.warning("No synthesis persona found or sequence is empty. Final answer may be incomplete.")
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
                estimated_next_step_cost=estimated_next_step_cost,
                progress_pct=self.get_progress_pct(phase), # Pass progress percentage
                current_persona_name=persona_name # Pass current persona name
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
                    state="running", 
                    current_total_tokens=self.get_total_used_tokens(), 
                    current_total_cost=self.get_total_estimated_cost(),
                    estimated_next_step_tokens=0, # Clear next step estimates
                    estimated_next_step_cost=0.0,
                    progress_pct=self.get_progress_pct(phase, completed=True), # Update progress
                    current_persona_name=None # Clear current persona name
                )
        except TokenBudgetExceededError as e:
            self.logger.error(f"Token budget exceeded during LLM generation for {persona_name}: {e}")
            raise e # Re-raise to be caught by the main run_debate handler
        except Exception as e:
            self.logger.error(f"Error during LLM generation for {persona_name}: {e}")
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
                    current_total_cost=self.get_total_estimated_cost(),
                    progress_pct=self.get_progress_pct(phase, error=True), # Set progress to indicate error
                    current_persona_name=persona_name # Indicate persona with error
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
                self.logger.error(f"Failed to parse/validate output for {persona_name} against {expected_schema.__name__} schema: {e}")
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
            self.logger.debug(f"Persona {persona_name} does not have a specific JSON schema. Storing raw text output.")
        
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

    # --- Helper method to calculate progress percentage ---
    def get_progress_pct(self, phase: str, completed: bool = False, error: bool = False) -> float:
        """Calculates a rough progress percentage based on phase and completion status."""
        phase_progress = {
            "context": 0.2,
            "debate": 0.3,
            "synthesis": 0.3
        }
        
        base_progress = 0.0
        if phase == "context":
            base_progress = phase_progress["context"]
        elif phase == "debate":
            base_progress = phase_progress["context"] + phase_progress["debate"]
        elif phase == "synthesis":
            base_progress = phase_progress["context"] + phase_progress["debate"] + phase_progress["synthesis"]
        
        if error:
            return base_progress * 0.9 # Indicate progress but with error state
        elif completed:
            return base_progress * 1.0 # Full progress for the phase
        else:
            return base_progress # Partial progress for the phase

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
            self.logger.error("Socratic debate process completed without executing any turns.")

        # Ensure final_answer is a dictionary, especially if it was None or malformed
        if not isinstance(self.final_answer, dict):
            self.logger.error(f"Final answer was not a dictionary: {type(self.final_answer).__name__}. Creating fallback error.")
            # Convert it to a general output dict.
            self.final_answer = {
                "COMMIT_MESSAGE": "Debate Failed - Final Answer Malformed",
                "RATIONALE": f"The final answer was not a dictionary. Type: {type(self.final_answer).__name__}",
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
                current_total_cost=self.get_total_estimated_cost(),
                progress_pct=1.0 # Final progress
            )
        return self.final_answer, self.intermediate_steps

    def _update_intermediate_steps_with_totals(self):
        """Updates intermediate steps with total token counts and estimated cost."""
        # Ensure total tokens used is accurate by summing phase tokens
        self.tokens_used = sum(self.tokens_used_per_phase.values()) + self.initial_input_tokens
        
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

    # --- Graceful Shutdown Handler ---
    # This method should be registered as a signal handler.
    def _handle_shutdown_signal(self, signum, frame):
        """Handles shutdown signals for graceful termination."""
        self.logger.warning(f"Received signal {signum}. Initiating graceful shutdown...")
        # Add any specific cleanup logic here if needed (e.g., closing file handles, releasing resources)
        # For Streamlit applications, the server often handles SIGINT/SIGTERM gracefully,
        # but explicit cleanup can be added if necessary.
        sys.exit(0)

    # --- MODIFICATION: Escape backticks in system prompt ---
    # The system_prompt for Impartial_Arbitrator contained markdown code block delimiters (```json```).
    # Python's parser can misinterpret these literal backticks within a string literal.
    # Escaping them with a backslash (`\`) tells Python to treat them as literal characters.
    # This change addresses the SyntaxError reported by the user.
    # Note: The original prompt used `|` for multiline string, which is generally good for YAML,
    # but the Python parser was still having issues. Escaping is a more direct fix for the Python syntax error.
    # We will use triple-quoted raw strings (r"""...""") for better handling of backslashes and quotes.
    # ---
    # The system_prompt for Impartial_Arbitrator needs to be updated.
    # This is a change within the PersonaManager or personas.yaml, not core.py itself.
    # However, if the prompt is hardcoded here, it would look like this:
    #
    # system_prompt: r"""You are the final arbiter in this Socratic debate. Your task is to synthesize all previous critiques and proposals into a coherent, actionable plan.
    #   
    #   **CRITICAL RULES:**
    #   1.  **YOUR ENTIRE RESPONSE MUST BE A SINGLE, VALID JSON OBJECT.** No other text, markdown, conversational filler, or explanations are allowed outside the JSON.
    #   2.  **STRICTLY ADHERE TO THE PROVIDED JSON SCHEMA.**
    #   3.  **IF YOU CANNOT PRODUCE VALID JSON**, output a JSON object with a specific error structure:
    #       ```json  <-- This is the line causing the SyntaxError. It needs escaping.
    #       {
    #         "COMMIT_MESSAGE": "LLM_GENERATION_ERROR",
    #         "RATIONALE": "Error: Could not generate valid JSON output. Please check prompt adherence and LLM capabilities.",
    #         "CODE_CHANGES": [],
    #         "malformed_blocks": [{"type": "LLM_FAILED_JSON_ADHERENCE", "message": "LLM ignored JSON output instruction."}]
    #       }
    #       ```
    #   4.  Ensure all code snippets within `CODE_CHANGES` adhere to the specified structure (`FILE_PATH`, `ACTION`, `FULL_CONTENT`, `LINES`) and PEP8 (line length <= 88).
    #   5.  Include the `malformed_blocks` field in your JSON output, even if it's an empty list `[]`.
    #
    #   **JSON Schema:**
    #   ```json  <-- This line also needs escaping.
    #   {
    #     "COMMIT_MESSAGE": "<string>",
    #     "RATIONALE": "<string>",
    #     "CODE_CHANGES": [
    #       {
    #         "FILE_PATH": "<string>",
    #         "ACTION": "ADD | MODIFY | REMOVE",
    #         "FULL_CONTENT": "<string>" (Required for ADD/MODIFY actions)
    #       },
    #       {
    #         "FILE_PATH": "<string>",
    #         "ACTION": "REMOVE",
    #         "LINES": ["<string>", "<string>"] (Required for REMOVE action)
    #       }
    #     ],
    #     "CONFLICT_RESOLUTION": "<string>" (Optional),
    #     "UNRESOLVED_CONFLICT": "<string>" (Optional),
    #     "malformed_blocks": []
    #   }
    #   ```
    #   **Synthesize the following feedback into the specified JSON format:**
    #   [Insert debate results here]
    #
    # The fix is to escape the backticks:
    # ```python
    # system_prompt: |
    #   ...
    #       \`\`\`json
    #       { ... }
    #       \`\`\`
    #   ...
    #       \`\`\`json
    #   ...
    #       \`\`\`
    #   ...
    # ```
    # This change is applied below.
    # ---
    
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