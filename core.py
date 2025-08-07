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

# Import models and settings
from src.models import PersonaConfig, ReasoningFrameworkConfig, LLMOutput, ContextAnalysisOutput
from src.config.settings import ChimeraSettings
from src.persona.routing import PersonaRouter
from src.context.context_analyzer import ContextRelevanceAnalyzer
from src.utils import LLMOutputParser

# Configure logging
logger = logging.getLogger(__name__)

class TokenBudgetExceededError(Exception):
    """Custom exception for when token usage exceeds the budget."""
    def __init__(self, current_tokens: int, budget: int, details: Optional[Dict[str, Any]] = None):
        error_details = {
            "current_tokens": current_tokens,
            "budget": budget,
            **(details or {})
        }
        super().__init__(f"Token budget exceeded: {current_tokens}/{budget} tokens used")
        self.details = error_details
        self.severity = "WARNING"
        self.recoverable = True

class GeminiProvider:
    def __init__(self, api_key: str, model_name: str = "gemini-2.5-flash-lite"):
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(model_name)
        self.token_usage = defaultdict(int)
        self.console = Console()
        self.model_name = model_name

    def generate_content(self, prompt: str, temperature: float = 0.3, max_tokens: int = 2048) -> str:
        """Generate content using Gemini API with retry logic and token tracking."""
        start_time = time.time()
        retries = 0
        max_retries = 3
        
        while retries < max_retries:
            try:
                # Log the request
                logger.debug(f"Sending request to Gemini (model: {self.model_name})")
                logger.debug(f"Prompt length: {len(prompt)} characters")
                
                # Generate content
                response = self.model.generate_content(
                    prompt,
                    generation_config=types.GenerationConfig(
                        temperature=temperature,
                        max_output_tokens=max_tokens
                    )
                )
                
                # Extract and return the response text
                if response.text:
                    # Track token usage
                    prompt_tokens = response.usage_metadata.prompt_token_count
                    completion_tokens = response.usage_metadata.candidates_token_count
                    total_tokens = response.usage_metadata.total_token_count
                    
                    self.token_usage['prompt'] += prompt_tokens
                    self.token_usage['completion'] += completion_tokens
                    self.token_usage['total'] += total_tokens
                    
                    # Log token usage
                    elapsed = time.time() - start_time
                    logger.info(f"Generated response in {elapsed:.2f}s | "
                               f"Tokens: {prompt_tokens}+{completion_tokens}={total_tokens}")
                    
                    return response.text
                else:
                    logger.warning("Gemini API returned empty response")
                    return ""
                    
            except APIError as e:
                logger.error(f"Gemini API error: {str(e)}")
                retries += 1
                if retries >= max_retries:
                    raise
                # Exponential backoff with jitter
                wait_time = (2 ** retries) + random.uniform(0, 1)
                logger.info(f"Retrying in {wait_time:.2f} seconds...")
                time.sleep(wait_time)
            except Exception as e:
                logger.exception(f"Unexpected error in GeminiProvider: {str(e)}")
                raise

class SocraticDebate:
    def __init__(self, initial_prompt: str, api_key: str,
                 codebase_context: Optional[str] = None,
                 settings: Optional[ChimeraSettings] = None,
                 all_personas: Optional[Dict[str, PersonaConfig]] = None,
                 frameworks: Optional[Dict[str, ReasoningFrameworkConfig]] = None,
                 max_total_tokens_budget: int = 10000,
                 model_name: str = "gemini-2.5-flash-lite"):
        """
        Initialize a Socratic debate session.
        
        Args:
            initial_prompt: The user's initial prompt/question
            api_key: API key for the LLM provider
            codebase_context: Optional context about the codebase for code-related prompts
            settings: Optional custom settings; uses defaults if not provided
            all_personas: Optional custom personas; uses defaults if not provided
            frameworks: Optional custom reasoning frameworks; uses defaults if not provided
            max_total_tokens_budget: Maximum token budget for the entire debate process
            model_name: Name of the LLM model to use
        """
        # Load settings, using defaults if not provided
        self.settings = settings or ChimeraSettings()
        self.max_total_tokens_budget = max_total_tokens_budget
        self.tokens_used = 0
        self.model_name = model_name
        
        # Initialize token budgets based on settings and prompt analysis
        self.context_token_budget = 0
        self.debate_token_budget = 0
        
        # Initialize context analyzer
        self.context_analyzer = None
        self.codebase_context = None
        if codebase_context:
            self.codebase_context = codebase_context
            self.context_analyzer = ContextRelevanceAnalyzer()
            self.context_analyzer.compute_file_embeddings(self.codebase_context)
        
        # Initialize persona router
        self.all_personas = all_personas or {}
        self.frameworks = frameworks or {}
        self.persona_router = PersonaRouter(self.all_personas)
        
        # Set up the LLM provider
        self.llm_provider = GeminiProvider(api_key=api_key, model_name=model_name)
        
        # Store the initial prompt
        self.initial_prompt = initial_prompt
        
        # Track the debate progress
        self.intermediate_steps = {}
        self.final_answer = None
        self.process_log = []
        
        # Initialize persona sequence
        self.persona_sequence = []
        
        # Calculate token budgets based on prompt complexity
        self._calculate_token_budgets()
    
    def _calculate_token_budgets(self):
        """Calculate token budgets for context analysis and debate phases."""
        # Simple heuristic: context gets 20% of tokens, debate gets 80%
        self.context_token_budget = int(self.max_total_tokens_budget * self.settings.context_token_budget_ratio)
        self.debate_token_budget = int(self.max_total_tokens_budget * self.settings.debate_token_budget_ratio)
        
        logger.info(f"Token budgets: Context={self.context_token_budget}, Debate={self.debate_token_budget}")
    
    def _check_token_budget(self, tokens_to_use: int, phase: str = "general"):
        """Check if using the specified tokens would exceed the budget."""
        if self.tokens_used + tokens_to_use > self.max_total_tokens_budget:
            raise TokenBudgetExceededError(
                current_tokens=self.tokens_used,
                budget=self.max_total_tokens_budget,
                details={"phase": phase, "tokens_requested": tokens_to_use}
            )
    
    def _analyze_context(self) -> Dict[str, Any]:
        """Analyze the context of the prompt to determine the best approach."""
        if not self.codebase_context or not self.context_analyzer:
            logger.info("No codebase context provided, skipping context analysis")
            return {"domain": "General", "relevant_files": []}
        
        # Extract keywords from the prompt
        keywords = self.context_analyzer._extract_keywords_from_prompt()
        
        # Find relevant files based on keywords
        relevant_files = self.context_analyzer.get_relevant_files(keywords)
        
        # Determine the domain based on the prompt
        domain = self.persona_router.determine_domain(self.initial_prompt)
        
        logger.info(f"Context analysis complete. Domain: {domain}, Relevant files: {len(relevant_files)}")
        
        return {
            "domain": domain,
            "relevant_files": relevant_files,
            "keywords": keywords
        }
    
    def _prepare_context(self, context_analysis: Dict[str, Any]) -> str:
        """Prepare the context for the debate based on the context analysis."""
        if not self.codebase_context or not context_analysis.get("relevant_files"):
            return ""
        
        # Get the content of relevant files
        context_parts = []
        for file_path, _ in context_analysis["relevant_files"][:5]:  # Limit to top 5 files
            if file_path in self.codebase_context:
                content = self.codebase_context[file_path]
                # Create a meaningful representation of the file
                clean_content = self.context_analyzer._clean_code_content(content)
                key_elements = self.context_analyzer._extract_key_elements(content)
                
                context_parts.append(f"File: {file_path}")
                context_parts.append(f"Key elements: {key_elements}")
                context_parts.append(f"Content snippet:\n{clean_content[:500]}...")
                context_parts.append("")
        
        return "\n".join(context_parts)
    
    def _generate_persona_sequence(self, context_analysis: Dict[str, Any]) -> List[str]:
        """Generate the sequence of personas to participate in the debate."""
        domain = context_analysis["domain"]
        return self.persona_router.determine_persona_sequence(
            self.initial_prompt,
            context_analysis
        )
    
    def _run_debate_round(self, current_response: str, persona_name: str) -> str:
        """Run a single round of the debate with the specified persona."""
        persona = self.all_personas[persona_name]
        
        # Create the prompt for this persona
        prompt = f"""
You are {persona_name}: {persona.description}
{persona.system_prompt}

Current debate state:
{current_response}

User's original prompt:
{self.initial_prompt}

Please provide your critique, feedback, or new perspective on the current debate state.
Focus on logical reasoning, identifying flaws, or offering improvements.
        """
        
        # Check token budget
        estimated_tokens = len(prompt) // 4  # Rough estimate
        self._check_token_budget(estimated_tokens, f"debate_round_{persona_name}")
        
        # Generate response
        logger.info(f"Running debate round with {persona_name}")
        response = self.llm_provider.generate_content(
            prompt,
            temperature=persona.temperature,
            max_tokens=persona.max_tokens
        )
        
        # Update token usage
        self.tokens_used += self.llm_provider.token_usage['total']
        
        # Log the step
        self.intermediate_steps[f"{persona_name}_Output"] = response
        self.intermediate_steps[f"{persona_name}_Tokens_Used"] = self.llm_provider.token_usage['total']
        self.process_log.append({
            "step": f"{persona_name}_Output",
            "tokens_used": self.llm_provider.token_usage['total'],
            "response_length": len(response)
        })
        
        return response
    
    def _synthesize_final_answer(self, final_debate_state: str) -> str:
        """Synthesize the final answer from the debate state."""
        # Find the impartial arbitrator
        arbitrator = None
        for persona_name, persona in self.all_personas.items():
            if "arbitrator" in persona_name.lower():
                arbitrator = persona
                break
        
        if not arbitrator:
            # Default arbitrator prompt if no specific one is found
            prompt = f"""
Based on the following debate about the user's prompt, synthesize a clear, 
comprehensive, and balanced final answer that incorporates the best insights 
from all perspectives:

Debate Summary:
{final_debate_state}

User's Original Prompt:
{self.initial_prompt}

Provide a final answer that addresses the user's needs directly and thoroughly.
            """
        else:
            # Use the arbitrator's system prompt
            prompt = f"""
{arbitrator.system_prompt}

Based on the following debate, provide a final synthesized answer:

Debate Summary:
{final_debate_state}

User's Original Prompt:
{self.initial_prompt}
            """
        
        # Check token budget
        estimated_tokens = len(prompt) // 4
        self._check_token_budget(estimated_tokens, "final_synthesis")
        
        # Generate final answer
        logger.info("Synthesizing final answer")
        final_answer = self.llm_provider.generate_content(
            prompt,
            temperature=0.3,
            max_tokens=1024
        )
        
        # Update token usage
        self.tokens_used += self.llm_provider.token_usage['total']
        
        # Store the result
        self.final_answer = final_answer
        self.intermediate_steps["Final_Answer_Output"] = final_answer
        self.intermediate_steps["Final_Answer_Tokens_Used"] = self.llm_provider.token_usage['total']
        self.intermediate_steps["Total_Tokens_Used"] = self.tokens_used
        self.intermediate_steps["Total_Estimated_Cost_USD"] = self._calculate_cost()
        
        return final_answer
    
    def _calculate_cost(self) -> float:
        """Calculate the estimated cost based on token usage."""
        # This is a placeholder - actual cost calculation would depend on the model
        # For Gemini, as of 2023, pricing is approximately:
        # $0.00000025 per character for input, $0.0000005 per character for output
        # But this varies by model and over time
        
        # Simplified estimate: $0.000003 per token
        return self.tokens_used * 0.000003
    
    def run_debate(self) -> Dict[str, Any]:
        """
        Run the complete Socratic debate process and return the results.
        
        Returns:
            Dictionary containing the final answer and intermediate steps
        """
        try:
            # 1. Analyze context
            self._check_token_budget(self.context_token_budget, "context_analysis")
            context_analysis = self._analyze_context()
            self.intermediate_steps["Context_Analysis"] = context_analysis
            
            # 2. Prepare context
            context_str = self._prepare_context(context_analysis)
            self.intermediate_steps["Context_Preparation"] = context_str
            
            # 3. Generate persona sequence
            self.persona_sequence = self._generate_persona_sequence(context_analysis)
            self.intermediate_steps["Persona_Sequence"] = self.persona_sequence
            
            # 4. Run initial generation (Visionary Generator)
            current_response = ""
            if self.persona_sequence:
                current_response = self._run_debate_round(
                    "No previous responses.", 
                    self.persona_sequence[0]
                )
                
                # 5. Run subsequent debate rounds
                for persona_name in self.persona_sequence[1:]:
                    current_response = self._run_debate_round(current_response, persona_name)
            
            # 6. Synthesize final answer
            final_answer = self._synthesize_final_answer(current_response)
            
            # 7. Return results
            return {
                "final_answer": final_answer,
                "intermediate_steps": self.intermediate_steps,
                "process_log": self.process_log,
                "token_usage": dict(self.llm_provider.token_usage),
                "total_tokens_used": self.tokens_used
            }
            
        except TokenBudgetExceededError as e:
            logger.warning(f"Token budget exceeded: {str(e)}")
            # Return partial results with error information
            return {
                "final_answer": "Process terminated early due to token budget constraints.",
                "intermediate_steps": self.intermediate_steps,
                "process_log": self.process_log,
                "token_usage": dict(self.llm_provider.token_usage),
                "total_tokens_used": self.tokens_used,
                "error": str(e),
                "error_details": e.details
            }
        except Exception as e:
            logger.exception("Unexpected error during debate process")
            raise

# Additional helper functions
def load_personas_from_yaml(yaml_path: str) -> Dict[str, PersonaConfig]:
    """Load personas configuration from a YAML file."""
    with open(yaml_path, 'r') as f:
        config = yaml.safe_load(f)
    
    personas = {}
    for persona_data in config.get('personas', []):
        # Convert YAML data to PersonaConfig
        personas[persona_data['name']] = PersonaConfig(**persona_data)
    
    return personas

def load_frameworks_from_yaml(yaml_path: str) -> Dict[str, ReasoningFrameworkConfig]:
    """Load reasoning frameworks from a YAML file."""
    with open(yaml_path, 'r') as f:
        config = yaml.safe_load(f)
    
    frameworks = {}
    for framework_name, framework_data in config.get('reasoning_frameworks', {}).items():
        # Convert YAML data to ReasoningFrameworkConfig
        frameworks[framework_name] = ReasoningFrameworkConfig(
            framework_name=framework_name,
            personas={},
            persona_sets=framework_data.get('persona_sets', {})
        )
    
    return frameworks