# src/core.py
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
# Import GeminiTokenizer for GeminiProvider
from src.tokenizers import GeminiTokenizer 
# NEW: Import LLMResponseValidator and LLMResponseValidationError
from src.utils.response_validator import LLMResponseValidator
from src.exceptions import LLMResponseValidationError # Ensure this is imported for catching

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
        # Initialize tokenizer for accurate counting
        self.tokenizer = GeminiTokenizer(model_name=self.model_name)
        
    def count_tokens(self, text: str) -> int:
        """Accurate token counting using Gemini API via tokenizer, with an improved fallback."""
        if not text:
            return 0
            
        try:
            # Ensure tokenizer is initialized and available
            if not hasattr(self, 'tokenizer') or not self.tokenizer:
                logger.warning("Tokenizer not initialized in GeminiProvider. Using improved fallback estimation.")
                return self._improved_token_estimate(text) # Use improved fallback
            
            # Use the tokenizer for accurate counting
            return self.tokenizer.count_tokens(text)
        except Exception as e:
            # Catch potential errors from the tokenizer itself or API issues
            logger.warning(f"Token counting API failed: {str(e)}. Using improved fallback estimation.")
            return self._improved_token_estimate(text) # Use improved fallback

    def _improved_token_estimate(self, text: str) -> int:
        """
        Provides a more robust token estimation heuristic when the primary tokenizer fails.
        This heuristic is designed to better approximate Gemini's tokenization behavior
        than a simple character-to-token ratio.
        """
        if not text:
            return 0
            
        # Heuristic: Estimate based on word count, considering common patterns.
        # Gemini's tokenization is complex, but word count is a reasonable proxy.
        # Average tokens per word can vary, but ~1.3 is a common estimate.
        words = text.split()
        estimated_tokens = len(words) * 1.3
        
        # Adjust for common code elements which might be tokenized differently
        # (e.g., symbols, keywords, indentation). This is a simplified adjustment.
        code_indicators = ['{', '}', '[', ']', '(', ')', '=', '+', '-', '*', '/', '#', '//', '/*', ':', ';']
        code_density = sum(text.count(ind) for ind in code_indicators) / max(1, len(text))
        
        if code_density > 0.05: # If text appears to be code-heavy
            estimated_tokens *= 1.2 # Increase estimate slightly for code
        
        # Ensure a minimum of 1 token for any non-empty text
        return max(1, int(round(estimated_tokens)))

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
                    # Note: GeminiProvider.count_tokens should be used for accurate counts
                    # The response.usage_metadata might be available and more precise.
                    # For now, we rely on the count_tokens method for consistency.
                    prompt_tokens = self.count_tokens(prompt) # Use our accurate counter
                    completion_tokens = self.count_tokens(response.text) # Use our accurate counter
                    total_tokens = prompt_tokens + completion_tokens # Sum for this call
                    
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
                 codebase_context: Optional[Dict[str, str]] = None, # Changed type hint to Dict[str, str]
                 settings: Optional[ChimeraSettings] = None,
                 all_personas: Optional[Dict[str, PersonaConfig]] = None,
                 persona_sets: Optional[Dict[str, List[str]]] = None, # Added persona_sets
                 persona_sequence: Optional[List[str]] = None, # Added persona_sequence
                 domain: Optional[str] = None, # Added domain
                 max_total_tokens_budget: int = 10000,
                 model_name: str = "gemini-2.5-flash-lite",
                 status_callback: Optional[Callable] = None, # Added status_callback
                 rich_console: Optional[Console] = None # Added rich_console
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
        self.llm_provider = GeminiProvider(api_key=api_key, model_name=model_name)
        
        # Store the initial prompt
        self.initial_prompt = initial_prompt
        
        # Track the debate progress
        self.intermediate_steps = {}
        self.final_answer = None
        self.process_log = []
        
        # Status callback and console for UI updates
        self.status_callback = status_callback
        self.rich_console = rich_console or Console()
        
        # Calculate token budgets based on prompt complexity
        self._calculate_token_budgets()
    
    def _calculate_token_budgets(self):
        """Calculate dynamic token budgets based on analysis type"""
        # Base ratios from settings
        base_context_ratio = self.settings.context_token_budget_ratio
        base_debate_ratio = self.settings.debate_token_budget_ratio
        
        # ADAPT FOR CODE ANALYSIS (NEW LOGIC)
        prompt_lower = self.initial_prompt.lower()
        # Check for keywords indicating code analysis or self-analysis
        if "code" in prompt_lower or "analyze" in prompt_lower or "refactor" in prompt_lower or "chimera" in prompt_lower or "self-analysis" in prompt_lower:
            # For code analysis, prioritize context understanding
            # Scale up context ratio, ensuring it doesn't exceed a reasonable max (e.g., 70%)
            context_ratio = min(0.7, base_context_ratio * 3.5)  # Example scaling factor
            debate_ratio = 1.0 - context_ratio
        else:
            # Use default ratios if not a code analysis prompt
            context_ratio = base_context_ratio
            debate_ratio = base_debate_ratio
        
        # Normalize to ensure ratios sum to 1.0, respecting boundaries
        total = context_ratio + debate_ratio
        if abs(total - 1.0) > 0.01: # Handle potential floating point inaccuracies
            context_ratio = context_ratio / total
            debate_ratio = debate_ratio / total
        
        self.context_token_budget = int(self.max_total_tokens_budget * context_ratio)
        self.debate_token_budget = int(self.max_total_tokens_budget * debate_ratio)
        
        logger.info(f"Token budgets: Context={self.context_token_budget}, Debate={self.debate_token_budget}")
    
    def _check_token_budget(self, prompt_text: str, step_name: str) -> int:
        """
        Check if using the specified tokens would exceed the budget using accurate counting.
        Returns the number of tokens used for this step.
        Raises TokenBudgetExceededError if budget is exceeded.
        """
        try:
            actual_tokens = self.llm_provider.count_tokens(prompt_text)
            if self.tokens_used + actual_tokens > self.max_total_tokens_budget:
                raise TokenBudgetExceededError(
                    current_tokens=self.tokens_used,
                    budget=self.max_total_tokens_budget,
                    details={"step": step_name, "tokens_requested": actual_tokens}
                )
            return actual_tokens # Return tokens used for this step
        except TokenBudgetExceededError:
            raise # Re-raise if budget is exceeded
        except Exception as e:
            logger.error(f"Error during token budget check for step '{step_name}': {e}")
            # If counting fails, we can't reliably check the budget.
            # For safety, assume it might exceed or log a critical warning.
            raise TokenBudgetExceededError(
                current_tokens=self.tokens_used,
                budget=self.max_total_tokens_budget,
                details={"step": step_name, "error": f"Token counting failed: {e}"}
            )
    
    def _analyze_context(self) -> Dict[str, Any]:
        """Analyze the context of the prompt to determine the best approach."""
        if not self.codebase_context or not self.context_analyzer:
            logger.info("No codebase context provided, skipping context analysis")
            return {"domain": "General", "relevant_files": []}
        
        # Extract keywords from the prompt
        # Assuming context_analyzer has a method to extract keywords from prompt
        keywords = self.context_analyzer.model_dump_json() # Placeholder, needs actual keyword extraction
        
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
        """
        Prepare the context for the debate based on the context analysis,
        respecting the context token budget.
        """
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
            remaining_budget_chars = (self.context_token_budget - current_context_tokens) * 4
            
            # Ensure at least some content is extracted if budget allows
            if remaining_budget_chars <= 0:
                break # No more budget for context

            # Extract key elements and relevant code segments
            key_elements = self.context_analyzer._extract_key_elements(content)
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
            estimated_file_tokens = self.llm_provider.count_tokens(file_context_part)
            
            if current_context_tokens + estimated_file_tokens > self.context_token_budget:
                logger.info(f"Skipping {file_path} due to context budget. "
                            f"Current: {current_context_tokens}, Estimated for file: {estimated_file_tokens}, "
                            f"Budget: {self.context_token_budget}")
                break # Stop adding files if budget is exceeded
            
            context_parts.append(file_context_part)
            current_context_tokens += estimated_file_tokens
            
        logger.info(f"Prepared context with {len(context_parts)} files, total estimated tokens: {current_context_tokens}")
        return "\n".join(context_parts)
    
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

Please provide your critique, feedback, or new perspective on the current debate state.
Focus on logical reasoning, identifying flaws, or offering improvements.
        """
        
        # Check token budget using the constructed prompt text
        tokens_used_in_round = self._check_token_budget(prompt_for_llm, f"debate_round_{persona_name}")
        
        # Generate response
        logger.info(f"Running debate round with {persona_name}")
        response = self.llm_provider.generate_content(
            prompt_for_llm, # Use the constructed prompt
            temperature=persona.temperature,
            max_tokens=persona.max_tokens
        )
        
        # Update total token usage
        self.tokens_used += tokens_used_in_round
        
        # Log the step
        self.intermediate_steps[f"{persona_name}_Output"] = response
        self.intermediate_steps[f"{persona_name}_Tokens_Used"] = tokens_used_in_round
        self.process_log.append({
            "step": f"{persona_name}_Output",
            "tokens_used": tokens_used_in_round,
            "response_length": len(response)
        })
        
        return response
    
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
            return {"error": "Impartial_Arbitrator persona not found."}

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

            tokens_used_in_synthesis = self._check_token_budget(prompt_for_synthesis, "final_synthesis")
            
            raw_final_answer = self.llm_provider.generate_content(
                prompt_for_synthesis,
                temperature=arbitrator.temperature, # Use arbitrator's temperature
                max_tokens=arbitrator.max_tokens # Use arbitrator's max_tokens
            )
            
            self.tokens_used += tokens_used_in_synthesis
            
            # Attempt to parse and validate the raw output
            try:
                # Use LLMOutputParser to handle extraction and validation
                llm_output_parser = LLMOutputParser()
                validated_output_dict = llm_output_parser.parse_and_validate(raw_final_answer, LLMOutput)
                
                # If successful, store and return
                self.final_answer = validated_output_dict
                self.intermediate_steps["Final_Answer_Output"] = validated_output_dict
                self.intermediate_steps["Final_Answer_Tokens_Used"] = tokens_used_in_synthesis
                self.intermediate_steps["Total_Tokens_Used"] = self.tokens_used
                self.intermediate_steps["Total_Estimated_Cost_USD"] = self._calculate_cost()
                return validated_output_dict
            except Exception as e: # Catch any exception from parse_and_validate
                logger.error(f"Validation failed for final answer (attempt {attempt+1}): {e}")
                # Store the raw, invalid output for debugging
                self.intermediate_steps[f"Final_Answer_Output_Attempt_{attempt+1}_Raw"] = raw_final_answer
                self.intermediate_steps[f"Final_Answer_Output_Attempt_{attempt+1}_Error"] = str(e)
                if attempt == max_retries:
                    # If max retries reached, raise the error to app.py
                    raise LLMResponseValidationError(
                        f"Final answer failed validation after {max_retries} retries: {e}",
                        invalid_response=raw_final_answer,
                        expected_schema="LLMOutput",
                        details={"validation_error": str(e)}
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
    
    def run_debate(self) -> Dict[str, Any]:
        """
        Run the complete Socratic debate process and return the results.
        
        Returns:
            Dictionary containing the final answer and intermediate steps
        """
        try:
            # 1. Analyze context
            # Check token budget for context analysis phase
            # Note: _check_token_budget expects prompt_text, not a phase.
            # For context analysis, we might not have a single prompt text,
            # but rather the initial prompt and codebase context.
            # A simple approach is to count the initial prompt.
            initial_prompt_tokens = self._check_token_budget(self.initial_prompt, "initial_prompt_count")
            
            context_analysis = self._analyze_context()
            self.intermediate_steps["Context_Analysis"] = context_analysis
            
            # 2. Prepare context
            context_str = self._prepare_context(context_analysis)
            self.intermediate_steps["Context_Preparation"] = context_str
            # Count tokens for context preparation if it's significant
            if context_str:
                context_tokens_used = self._check_token_budget(context_str, "context_preparation")
                self.tokens_used += context_tokens_used
            
            # 3. Generate persona sequence
            self.persona_sequence = self._generate_persona_sequence(context_analysis)
            self.intermediate_steps["Persona_Sequence"] = self.persona_sequence
            
            # 4. Run initial generation (Visionary Generator)
            current_response = ""
            if self.persona_sequence:
                # The first persona's prompt will include context and initial prompt
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
                return {
                    "final_answer": "Error: No personas selected for debate.",
                    "intermediate_steps": self.intermediate_steps,
                    "process_log": self.process_log,
                    "token_usage": dict(self.llm_provider.token_usage),
                    "total_tokens_used": self.tokens_used,
                    "error": "No persona sequence generated."
                }
            
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