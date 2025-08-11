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
# Import TokenManager for token budget calculations (though not directly instantiated here, its logic is mimicked)
# from src.token_manager import TokenManager # Not directly used in SocraticDebate, but its logic is implemented here.
# --- END IMPORT MODIFICATIONS ---

# Import models and settings
from src.models import PersonaConfig, ReasoningFrameworkConfig # Assuming LLMOutput is defined here or accessible
from src.config.settings import ChimeraSettings
from src.persona.routing import PersonaRouter
from src.utils import LLMOutputParser
# NEW: Import LLMResponseValidationError and other exceptions
from src.exceptions import ChimeraError, LLMResponseValidationError, SchemaValidationError, TokenBudgetExceededError, LLMProviderError # Corrected import, added LLMProviderError

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
            context_analyzer: An optional pre-initialized and cached ContextRelevanceAnalyzer instance.
        """
        self.settings = settings or ChimeraSettings()
        self.context_token_budget_ratio = context_token_budget_ratio
        self.max_total_tokens_budget = max_total_tokens_budget
        self.tokens_used = 0
        self.model_name = model_name
        
        # Assign initial_prompt and other necessary attributes BEFORE they are used.
        self.initial_prompt = initial_prompt
        self._prev_context_ratio = None
        self.codebase_context = codebase_context

        # --- FIX START ---
        # Initialize phase_budgets directly before calling _calculate_token_budgets
        self.phase_budgets = {"context": 0, "debate": 0, "synthesis": 0}
        
        # Initialize the LLM provider. This might raise LLMProviderError if API key is invalid.
        try:
            self.llm_provider = GeminiProvider(api_key=api_key, model_name=self.model_name)
        except LLMProviderError as e:
            logger.error(f"Failed to initialize LLM provider: {e}")
            # Re-raise as ChimeraError for consistent error handling in the app
            raise ChimeraError(f"LLM provider initialization failed: {e}") from e
        # --- FIX END ---

        # Now that llm_provider is initialized, we can safely calculate token budgets.
        # This method itself handles potential errors during token calculation.
        self._calculate_token_budgets()

        self.context_analyzer = context_analyzer # Use the provided analyzer instance
        
        self.all_personas = all_personas or {}
        self.persona_sets = persona_sets or {}
        self.persona_sequence = persona_sequence or []
        self.domain = domain
        self.persona_router = PersonaRouter(self.all_personas)
        
        # If codebase_context was provided, compute embeddings now if context_analyzer is available.
        # This ensures embeddings are ready if context is used early.
        if self.codebase_context and self.context_analyzer:
            if isinstance(self.codebase_context, dict):
                # Compute embeddings if not already done.
                # The analyzer instance passed is assumed to be cached and potentially has embeddings computed.
                # If context changes, the analyzer's embeddings might need recomputation, handled by app.py caching.
                if not self.context_analyzer.file_embeddings: # Only compute if not already done
                    self.context_analyzer.compute_file_embeddings(self.codebase_context)
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
            # Simple concatenation of all context files for estimation.
            # A more sophisticated approach might involve summarizing or embedding.
            context_str = "\n".join(f"{fname}:\n{content}" for fname, content in self.codebase_context.items())
        
        prompt_for_estimation = self.initial_prompt if self.initial_prompt else ""

        try:
            # Use the LLM provider's count_tokens method to estimate tokens for the combined input.
            combined_input_text = f"{context_str}\n\n{prompt_for_estimation}" if context_str else prompt_for_estimation
            
            # --- FIX START ---
            # Ensure llm_provider and its tokenizer are properly initialized before calling count_tokens.
            # The GeminiProvider.__init__ should raise LLMProviderError if it fails.
            if not hasattr(self.llm_provider, 'tokenizer') or not self.llm_provider.tokenizer:
                 raise ChimeraError("LLM provider or its tokenizer is not properly initialized.")

            self.initial_input_tokens = self.llm_provider.count_tokens(combined_input_text)
            # --- FIX END ---

            # Calculate available tokens for debate and synthesis phases
            available_tokens_for_phases = max(0, self.max_total_tokens_budget - self.initial_input_tokens)
            
            # Distribute budgets, ensuring minimums for critical phases
            self.phase_budgets["context"] = max(200, int(available_tokens_for_phases * context_ratio))
            self.phase_budgets["debate"] = max(500, int(available_tokens_for_phases * debate_ratio))
            self.phase_budgets["synthesis"] = max(400, int(available_tokens_for_phases * synthesis_ratio))

            logger.info(f"SocraticDebate token budgets initialized: "
                       f"Initial Input={self.initial_input_tokens}, "
                       f"Context={self.phase_budgets['context']}, "
                       f"Debate={self.phase_budgets['debate']}, "
                       f"Synthesis={self.phase_budgets['synthesis']}")

        # Catch specific errors from Gemini API or token counting
        except LLMProviderError as e:
            logger.error(f"LLM Provider Error during token calculation: {e}")
            # If the error is related to API key, provide specific feedback.
            if "api key not valid" in str(e).lower() or "API_KEY_INVALID" in str(e) or "INVALID_ARGUMENT" in str(e):
                 raise ChimeraError("LLM provider failed: Invalid API Key. Please check your Gemini API Key.") from e
            else:
                # For other LLMProviderErrors, use a generic message.
                # Fallback to default budgets if LLM provider methods are missing or other API errors occur
                self.phase_budgets = {"context": 500, "debate": 15000, "synthesis": 1000}
                self.initial_input_tokens = 0
                raise ChimeraError(f"LLM provider error: {e}") from e
        except AttributeError as e: # Keep AttributeError catch for other potential issues (e.g., llm_provider not initialized)
            logger.error(f"AttributeError during token calculation: {e}. Cannot calculate token budgets.")
            # Fallback to default budgets if LLM provider methods are missing
            self.phase_budgets = {"context": 500, "debate": 15000, "synthesis": 1000}
            self.initial_input_tokens = 0
            raise ChimeraError("LLM provider is missing required methods for token calculation.") from e
        except Exception as e: # Catch any other unexpected errors
            logger.error(f"An unexpected error occurred calculating token budgets: {e}")
            # Fallback to default budgets on any other error
            self.phase_budgets = {"context": 500, "debate": 15000, "synthesis": 1000}
            self.initial_input_tokens = 0
            raise ChimeraError("Failed to calculate token budgets due to an unexpected error.") from e
    
    def run_debate(self) -> Tuple[Any, Dict[str, Any]]:
        """
        Executes the Socratic debate process.
        This is a placeholder for the actual debate logic which would involve
        iteratively calling personas, analyzing responses, and synthesizing results.
        """
        # Placeholder for the actual debate execution logic.
        # This would involve:
        # 1. Initializing the persona router.
        # 2. Determining the persona sequence based on the prompt and domain.
        # 3. Iteratively calling personas for context analysis, debate, and synthesis.
        # 4. Tracking token usage per phase and step.
        # 5. Handling errors and retries.
        
        # For now, returning dummy data to allow the app to run without the full debate logic.
        # This part needs to be replaced with the actual debate orchestration.
        
        # Simulate a successful run for demonstration purposes if no actual debate logic is present.
        # This part needs to be replaced with the actual debate orchestration.
        
        # Example of how token tracking might be used:
        # try:
        #     # Simulate token usage for context analysis
        #     context_tokens_used = self.llm_provider.count_tokens("Analyzing context...")
        #     self.track_token_usage("context", context_tokens_used)
        #     self.check_budget("context", context_tokens_used, "Context Analysis")
        #
        #     # Simulate token usage for debate phase
        #     debate_tokens_used = self.llm_provider.count_tokens("Debating the prompt...")
        #     self.track_token_usage("debate", debate_tokens_used)
        #     self.check_budget("debate", debate_tokens_used, "Debate Phase")
        #
        #     # Simulate token usage for synthesis phase
        #     synthesis_tokens_used = self.llm_provider.count_tokens("Synthesizing results...")
        #     self.track_token_usage("synthesis", synthesis_tokens_used)
        #     self.check_budget("synthesis", synthesis_tokens_used, "Synthesis Phase")
        #
        # except TokenBudgetExceededError as e:
        #     logger.error(f"Token budget exceeded during run_debate: {e}")
        #     # Handle budget exceeded error appropriately, e.g., return an error state
        #     raise e
        # except Exception as e:
        #     logger.error(f"An error occurred during run_debate: {e}")
        #     # Handle other exceptions
        #     raise ChimeraError(f"An error occurred during debate execution: {e}")

        # Dummy return values for now
        dummy_final_answer = {
            "COMMIT_MESSAGE": "Debate Simulation Complete",
            "RATIONALE": "This is a simulated response. The actual debate logic needs to be implemented.",
            "CODE_CHANGES": [],
            "malformed_blocks": []
        }
        dummy_intermediate_steps = {
            "Total_Tokens_Used": self.initial_input_tokens, # Placeholder
            "Total_Estimated_Cost_USD": 0.0, # Placeholder
            "Initial_Prompt_Tokens": self.initial_input_tokens,
            "Context_Analysis_Tokens": self.phase_budgets.get("context", 0),
            "Debate_Phase_Tokens": self.phase_budgets.get("debate", 0),
            "Synthesis_Phase_Tokens": self.phase_budgets.get("synthesis", 0),
        }
        
        # Update total tokens used based on initial input
        self.tokens_used = self.initial_input_tokens
        
        return dummy_final_answer, dummy_intermediate_steps

    def track_token_usage(self, phase: str, tokens: int):
        """Tracks token usage for a specific phase and updates total used tokens."""
        if phase in self.tokens_used_per_phase:
            self.tokens_used_per_phase[phase] += tokens
            self.tokens_used += tokens
        else:
            logger.warning(f"Attempted to track tokens for unknown phase: {phase}")
            self.tokens_used += tokens # Still track if phase is unknown

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
        # This requires access to the LLM provider's cost calculation method.
        # For now, it's a placeholder.
        try:
            total_tokens = self.get_total_used_tokens()
            # This calculation is simplified; ideally, it would sum costs per phase based on input/output tokens.
            # For demonstration, we'll use a rough estimate.
            # A more accurate calculation would track input/output tokens separately for each LLM call.
            cost = self.llm_provider.calculate_usd_cost(total_tokens, 0) # Assuming output tokens are not yet tracked accurately here
            return cost
        except Exception as e:
            logger.error(f"Could not estimate total cost: {e}")
            return 0.0