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
        
        # --- FIX START ---
        # Initialize _prev_context_ratio BEFORE calling _calculate_token_budgets
        # This prevents an AttributeError in _calculate_token_budgets when it checks `if self._prev_context_ratio is not None:`
        self._prev_context_ratio = None 
        # --- FIX END ---

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
        
        # The line below was the original location of the error, now redundant as _prev_context_ratio is initialized above.
        # self._prev_context_ratio = None 