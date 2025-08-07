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
import random
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
# Import custom exceptions
from src.exceptions import ChimeraError, LLMResponseValidationError, TokenBudgetExceededError
# Import TokenManager
from src.token_manager import TokenManager # Assuming this file exists

# Configure logging
logger = logging.getLogger(__name__)

# --- Placeholder for GeminiProvider (as it's defined in llm_provider.py) ---
# In a real scenario, this would be imported. For this snippet, we assume it's available.
# We'll include a minimal mock for demonstration if it were not imported.
# For this context, we assume it's correctly imported and functional.
# If GeminiProvider were defined here, it would need its own imports.
# The key methods needed by SocraticDebate are:
# - count_tokens(text: str) -> int
# - estimate_tokens_for_context(context_str: str, prompt: str) -> int
# - generate_content(prompt: str, temperature: float, max_tokens: int) -> str
# - calculate_usd_cost(input_tokens: int, output_tokens: int) -> float
# - tokenizer attribute (instance of Tokenizer)

# Mock GeminiProvider if not imported (for standalone context, but ideally imported)
# class GeminiProvider:
#     def __init__(self, api_key: str, model_name: str = "gemini-2.5-flash-lite"):
#         # Minimal mock init
#         self.model_name = model_name
#         self.tokenizer = GeminiTokenizer(model_name=self.model_name) # Assume GeminiTokenizer is available
#         self.token_usage = defaultdict(int)
#         logger.warning("Using mock GeminiProvider. Real implementation should be imported.")
#
#     def count_tokens(self, text: str) -> int:
#         return self.tokenizer.count_tokens(text) if text else 0
#
#     def estimate_tokens_for_context(self, context_str: str, prompt: str) -> int:
#         return self.count_tokens(context_str) + self.count_tokens(prompt)
#
#     def generate_content(self, prompt: str, temperature: float = 0.3, max_tokens: int = 2048) -> str:
#         # Mock response
#         return "Mock response from GeminiProvider."
#
#     def calculate_usd_cost(self, input_tokens: int, output_tokens: int) -> float:
#         return (input_tokens + output_tokens) * 0.000003 # Mock cost

# --- End Mock ---


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
                 rich_console: Optional[Console] = None
                 ):
        """
        Initialize a Socratic debate session.
        """
        self.settings = settings or ChimeraSettings()
        self.max_total_tokens_budget = max_total_tokens_budget
        self.model_name = model_name
        
        # Initialize context analyzer
        self.context_analyzer = None
        self.codebase_context = None
        if codebase_context:
            self.codebase_context = codebase_context
            self.context_analyzer = ContextRelevanceAnalyzer()
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

        # --- Token Manager Integration ---
        self.token_manager = TokenManager(self.llm_provider, self.max_total_tokens_budget)
        # Prepare context first to pass to budget calculation
        context_analysis = self._analyze_context() # Ensure this is called before budget calculation
        context_str = self._prepare_context(context_analysis)
        self.token_manager.calculate_phase_budgets(context_str, self.initial_prompt)
        # --- End Token Manager Integration ---

    # --- NEW: Token Manager Integration Methods ---
    def _truncate_prompt_for_tokens(self, prompt: str, max_tokens: int) -> str:
        """Truncates a prompt to fit within a token limit."""
        if max_tokens <= 0:
            return ""
        
        # Use the tokenizer for more accurate truncation
        try:
            # This is a simplified approach; a real implementation might need to
            # tokenize, truncate the token list, and then detokenize.
            # For now, we'll use a character limit as a proxy.
            # Estimate: 1 token ~ 4 characters.
            max_chars = max_tokens * 4 
            if len(prompt) > max_chars:
                return prompt[:max_chars] + "..."
            return prompt
        except Exception as e:
            logger.error(f"Error during prompt truncation: {e}")
            return prompt[:max_tokens * 4] # Fallback to character truncation

    def _check_token_budget(self, phase: str, prompt_text: str, step_name: str) -> int:
        """Wrapper to check budget using TokenManager."""
        try:
            # Use the LLM provider's tokenizer for accurate counting
            tokens_needed = self.llm_provider.count_tokens(prompt_text)
            return self.token_manager.check_budget(phase, tokens_needed, step_name)
        except TokenBudgetExceededError:
            raise # Re-raise to be caught by the main run_debate handler
        except Exception as e:
            logger.error(f"Error during token budget check for step '{step_name}': {e}")
            # Raise a specific error if token counting itself fails
            raise TokenBudgetExceededError(
                current_tokens=self.token_manager.used_tokens.get(phase, 0),
                budget=self.token_manager.phase_budgets.get(phase, 0),
                details={"step": step_name, "error": f"Token counting failed: {e}"}
            )

    def _analyze_validation_error(self, validation_error: Exception, response_text: str) -> str:
        """Analyze validation errors to provide specific correction instructions."""
        try:
            import json
            from jsonschema import ValidationError # Ensure jsonschema is available (via pydantic)
            
            # Attempt to extract the JSON part of the response
            json_start = response_text.find('{')
            json_end = response_text.rfind('}') + 1
            
            if json_start >= 0 and json_end > json_start:
                partial_json = response_text[json_start:json_end]
                try:
                    json.loads(partial_json) # Validate JSON structure
                    
                    # If it's a Pydantic ValidationError, extract specific field issues
                    if isinstance(validation_error, ValidationError):
                        # Extracting path and message from Pydantic ValidationError
                        error_details = validation_error.errors()
                        if error_details:
                            # Take the first error for simplicity
                            first_error = error_details[0]
                            error_path = " > ".join(str(p) for p in first_error.get('loc', []))
                            error_msg = first_error.get('msg', 'Unknown error')
                            
                            instructions = "**SPECIFIC VALIDATION ERROR:**\n"
                            instructions += f"Field: `{error_path or 'root'}`\n"
                            instructions += f"Problem: {error_msg}\n\n"
                            
                            # Provide targeted remediation based on common error types
                            if "is a required property" in error_msg:
                                field_name = error_msg.split("'")[1] # Extract field name
                                instructions += f"**ACTION REQUIRED:** Add the missing required field '{field_name}'.\n"
                            elif "is not of type" in error_msg:
                                if "string" in error_msg:
                                    instructions += "**ACTION REQUIRED:** This field must be a string. Ensure values are enclosed in double quotes.\n"
                                elif "integer" in error_msg:
                                    instructions += "**ACTION REQUIRED:** This field must be a number. Remove any quotes around the value.\n"
                            elif "does not match pattern" in error_msg:
                                instructions += "**ACTION REQUIRED:** The value does not match the expected format.\n"
                            
                            return instructions
                        else:
                            return "**VALIDATION ERROR DETECTED**\nCould not extract specific field error details. Please ensure all fields conform to the schema and JSON format."
                            
                except json.JSONDecodeError:
                    # Handle cases where the extracted JSON is malformed
                    line_num = response_text.count('\n', 0, json_start) + 1
                    col_num = json_start - response_text.rfind('\n', 0, json_start)
                    return (
                        f"**JSON STRUCTURE ERROR near line {line_num}, column {col_num}**\n"
                        "Your response contains invalid JSON formatting. Please ensure:\n"
                        "- All keys and string values are enclosed in double quotes.\n"
                        "- Commas correctly separate key-value pairs and array elements.\n"
                        "- No trailing commas before closing braces/brackets.\n"
                        "**ACTION REQUIRED:** Provide ONLY a valid JSON object."
                    )
            
            # Fallback for when no JSON structure is found
            return (
                "**CRITICAL ERROR: NO JSON FOUND**\n"
                "Your response must contain a valid JSON object enclosed in curly braces `{}`.\n"
                "**ACTION REQUIRED:** Start your response with '{' and end with '}'. Do NOT include any text outside the JSON block."
            )
        except Exception as e:
            self.logger.error(f"Error analyzing validation failure: {str(e)}")
            return (
                "**VALIDATION ERROR DETECTED**\n"
                "Please ensure your response is a valid JSON object that strictly follows the required schema.\n"
                "Check for missing fields, incorrect data types, and proper JSON formatting."
            )

    def _create_simplified_final_answer(self, debate_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Create a simplified final answer when validation repeatedly fails."""
        logger.warning("Creating simplified final answer due to repeated validation failures")
        
        # Extract key points from debate results
        key_points = []
        for result in debate_results:
            if 'response' in result and isinstance(result['response'], str):
                # Take the first sentence of the response if it's long enough
                sentences = [s.strip() for s in result['response'].split('.') if len(s.strip()) > 10]
                if sentences:
                    key_points.append(sentences[0] + '.')
        
        # Create minimal valid response adhering to LLMOutput structure
        return {
            "COMMIT_MESSAGE": "Simplified Answer: Validation Failed",
            "RATIONALE": "This simplified response was generated due to repeated validation failures. "
                         "The system extracted the most important points from the debate. "
                         "Please review the process log for details.",
            "CODE_CHANGES": [], # Ensure CODE_CHANGES is an empty list
            "CONFLICT_RESOLUTION": None,
            "UNRESOLVED_CONFLICT": None,
            "malformed_blocks": [{"type": "SIMPLIFIED_FALLBACK", "message": "Final output could not be validated. Simplified answer provided."}]
        }

    def _construct_synthesis_prompt(self, debate_results: List[Dict[str, Any]]) -> str:
        """Constructs the prompt for the final synthesis step."""
        debate_summary = "\n\n".join([f"Persona: {r.get('persona', 'Unknown')}\nResponse:\n{r.get('response', '')}" for r in debate_results if r.get('response')])
        
        arbitrator = None
        for persona_name, persona in self.all_personas.items():
            if "arbitrator" in persona_name.lower():
                arbitrator = persona
                break
        
        if not arbitrator:
            logger.error("Impartial_Arbitrator persona not found. Cannot construct synthesis prompt.")
            return "Error: Impartial_Arbitrator persona not found."

        return f"""
{arbitrator.system_prompt}

Based on the following debate, provide a final synthesized answer:

Debate Summary:
{debate_summary}

User's Original Prompt:
{self.initial_prompt}
"""

    def _synthesize_final_answer(self, debate_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Synthesize final answer with intelligent error recovery and specific guidance."""
        
        # Construct the initial prompt for synthesis
        prompt_for_synthesis = self._construct_synthesis_prompt(debate_results)
        
        for attempt in range(self.max_retries + 1):
            try:
                # Check token budget for the synthesis phase
                tokens_needed = self._check_token_budget("synthesis", prompt_for_synthesis, "final_synthesis")
                
                # Generate content, ensuring max_tokens respects the phase budget
                # Use a small buffer for generation, but don't exceed the budget
                # Assuming GeminiProvider.tokenizer has a max_output_tokens attribute or similar
                # If not, this might need adjustment based on GeminiProvider's capabilities.
                generation_max_tokens = min(self.llm_provider.tokenizer.max_output_tokens if hasattr(self.llm_provider.tokenizer, 'max_output_tokens') else 4096, tokens_needed + 50)
                
                raw_final_answer = self.llm_provider.generate_content(
                    prompt_for_synthesis,
                    temperature=0.3, # Use arbitrator's temperature
                    max_tokens=generation_max_tokens
                )
                
                # Validate response using the parser
                final_answer = self._validate_synthesis_response(raw_final_answer)
                
                # Track actual tokens used for synthesis
                actual_tokens_used = self.llm_provider.count_tokens(prompt_for_synthesis) # Count tokens for the prompt sent
                self.token_manager.track_usage("synthesis", actual_tokens_used)
                
                return final_answer
                
            except LLMResponseValidationError as e:
                if attempt < self.max_retries:
                    # Generate SPECIFIC correction instructions using the new helper
                    correction_instructions = self._analyze_validation_error(e, raw_final_answer)
                    # Append instructions to the prompt for the next retry
                    prompt_for_synthesis += f"\n\n{correction_instructions}"
                    self.logger.warning(f"Synthesis validation failed (attempt {attempt+1}). Applying specific corrections.")
                else:
                    # Final fallback: If all retries fail, create a simplified, valid response
                    self.logger.error("Final synthesis validation failed after all retries. Using simplified format.")
                    return self._create_simplified_final_answer(debate_results)
            except Exception as e: # Catch other potential errors during generation/validation
                self.logger.exception(f"Unexpected error during synthesis attempt {attempt+1}: {e}")
                if attempt == self.max_retries:
                    return self._create_simplified_final_answer(debate_results) # Fallback on unexpected errors too
        
        # Should not be reached if max_retries logic is sound
        raise Exception("Unexpected state in _synthesize_final_answer.")

    def _execute_debate_round(self, persona_name: str, current_state: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a debate round with proper token accounting."""
        if persona_name not in self.all_personas:
            logger.warning(f"Persona '{persona_name}' not found. Skipping this round.")
            return {"response": current_state.get("response", ""), "tokens_used": 0} # Return previous state if persona missing

        persona = self.all_personas[persona_name]
        
        # Construct the prompt for this persona
        prompt_for_llm = f"""
You are {persona_name}: {persona.description}
{persona.system_prompt}

Current debate state:
{current_state.get("response", "No previous response.")}

User's original prompt:
{self.initial_prompt}
        """
        
        # Check token budget for the 'debate' phase
        tokens_needed = self._check_token_budget("debate", prompt_for_llm, f"debate_round_{persona_name}")
        
        # Handle potential truncation if budget is tight
        remaining_debate_tokens = self.token_manager.get_remaining_tokens("debate")
        if tokens_needed > remaining_debate_tokens:
            # Truncate prompt to fit remaining budget
            truncated_prompt = self._truncate_prompt_for_tokens(prompt_for_llm, remaining_debate_tokens)
            # Re-check tokens needed for the truncated prompt
            tokens_needed = self._check_token_budget("debate", truncated_prompt, f"debate_round_{persona_name}_truncated")
            logger.warning(f"Truncated prompt for {persona_name} to fit debate budget.")
        else:
            truncated_prompt = prompt_for_llm

        # Generate response, respecting the calculated tokens_needed (which is already capped by budget)
        # Use the persona's max_tokens, but ensure it doesn't exceed the budget for this round.
        generation_max_tokens = min(persona.max_tokens, tokens_needed + 50) # Add a small buffer if needed, but stay within budget
        
        response = self.llm_provider.generate_content(
            truncated_prompt,
            temperature=persona.temperature,
            max_tokens=generation_max_tokens
        )
        
        # Track actual usage for the 'debate' phase
        actual_tokens_used = self.llm_provider.count_tokens(truncated_prompt) # Count tokens for the prompt sent
        self.token_manager.track_usage("debate", actual_tokens_used)
        
        # Log the step
        self.intermediate_steps[f"{persona_name}_Output"] = response
        self.intermediate_steps[f"{persona_name}_Tokens_Used"] = actual_tokens_used
        self.process_log.append({
            "step": f"{persona_name}_Output",
            "tokens_used": actual_tokens_used,
            "response_length": len(response)
        })
        
        return {"response": response, "tokens_used": actual_tokens_used}

    def run_debate(self) -> Dict[str, Any]:
        """
        Run the complete Socratic debate process and return the results.
        """
        try:
            # 1. Analyze context
            context_analysis = self._analyze_context()
            self.intermediate_steps["Context_Analysis"] = context_analysis
            
            # 2. Prepare context
            context_str = self._prepare_context(context_analysis)
            self.intermediate_steps["Context_Preparation"] = context_str
            # Token budget calculation is now done in __init__ using context_str and initial_prompt

            # 3. Generate persona sequence
            self.persona_sequence = self.persona_router.determine_persona_sequence(
                self.initial_prompt,
                intermediate_results=None # No intermediate results on the first pass
            )
            self.intermediate_steps["Persona_Sequence"] = self.persona_sequence
            
            # 4. Run initial generation
            current_response_data = {"response": "No previous response. Starting the debate.", "tokens_used": 0}
            if self.persona_sequence:
                current_response_data = self._execute_debate_round(
                    self.persona_sequence[0], 
                    current_response_data # Pass previous state
                )
                
                # 5. Run subsequent debate rounds
                for persona_name in self.persona_sequence[1:]:
                    current_response_data = self._execute_debate_round(persona_name, current_response_data)
            else:
                logger.warning("No persona sequence generated. Debate cannot proceed.")
                # Handle case where no personas are selected
                return {
                    "final_answer": "Error: No personas selected for debate.",
                    "intermediate_steps": self.intermediate_steps,
                    "process_log": self.process_log,
                    "token_usage": dict(self.token_manager.used_tokens), # Use token_manager for total usage
                    "total_tokens_used": self.token_manager.get_total_used_tokens(),
                    "error": "No persona sequence generated."
                }
            
            # 6. Synthesize final answer
            final_answer = self._synthesize_final_answer(
                [{"persona": p_name, "response": self.intermediate_steps.get(f"{p_name}_Output", "")} for p_name in self.persona_sequence]
            )
            
            # 7. Update final results and token usage
            self.final_answer = final_answer
            self.intermediate_steps["Final_Answer"] = final_answer
            # Token usage for synthesis is handled within _synthesize_final_answer
            
            # Log total tokens used
            total_tokens_used = self.token_manager.get_total_used_tokens()
            self.intermediate_steps["Total_Tokens_Used"] = total_tokens_used
            self.intermediate_steps["Total_Estimated_Cost_USD"] = self.llm_provider.calculate_usd_cost(
                self.token_manager.initial_input_tokens,
                total_tokens_used - self.token_manager.initial_input_tokens # Completion tokens = Total - Input
            )
            
            # Return results
            return {
                "final_answer": self.final_answer,
                "intermediate_steps": self.intermediate_steps,
                "process_log": self.process_log,
                "token_usage": dict(self.token_manager.used_tokens),
                "total_tokens_used": total_tokens_used
            }
            
        except TokenBudgetExceededError as e:
            logger.warning(f"Token budget exceeded: {str(e)}")
            # Return partial results with error information
            return {
                "final_answer": "Process terminated early due to token budget constraints.",
                "intermediate_steps": self.intermediate_steps,
                "process_log": self.process_log,
                "token_usage": dict(self.token_manager.used_tokens),
                "total_tokens_used": self.token_manager.get_total_used_tokens(),
                "error": str(e),
                "error_details": e.details
            }
        except Exception as e:
            logger.exception("Unexpected error during debate process")
            # Re-raise the exception to be caught by the app.py handler
            raise

    # --- Helper methods that were implied or needed for the proposed changes ---
    # These might need to be added if they don't exist in the original core.py
    
    # Placeholder for _validate_synthesis_response if not present
    def _validate_synthesis_response(self, raw_final_answer: str) -> Dict[str, Any]:
        """Validates the raw synthesis response against the LLMOutput schema."""
        from src.utils.output_parser import LLMOutputParser
        parser = LLMOutputParser()
        try:
            # The parse_and_validate method returns a dict, which is what we need.
            return parser.parse_and_validate(raw_final_answer, LLMOutput)
        except Exception as e:
            # If parsing/validation fails, return an error structure
            logger.error(f"Failed to parse/validate synthesis response: {e}")
            return {
                "COMMIT_MESSAGE": "Validation Failed",
                "RATIONALE": f"Failed to parse or validate the final synthesis response. Error: {e}",
                "CODE_CHANGES": [],
                "malformed_blocks": [{"type": "SYNTHESIS_VALIDATION_ERROR", "message": str(e), "raw_string_snippet": raw_final_answer[:500]}]
            }

    # --- End NEW Methods ---

    # Removed: _calculate_token_budgets (replaced by TokenManager)
    # Removed: self.tokens_used (replaced by TokenManager)
    # Removed: self.context_token_budget, self.debate_token_budget (replaced by TokenManager)

    # --- Original Methods (kept for context, some might be slightly adjusted by TokenManager integration) ---
    def _analyze_context(self) -> Dict[str, Any]:
        """Analyze the context of the prompt to determine the best approach."""
        if not self.codebase_context or not self.context_analyzer:
            logger.info("No codebase context provided, skipping context analysis")
            return {"domain": "General", "relevant_files": []}
        
        # Use the router's domain analysis for the primary domain
        # Note: _analyze_prompt_domain is called by PersonaRouter.determine_persona_sequence
        # Here we just need a general domain for context analysis if not specified.
        # A simple approach is to use the router's domain analysis on the initial prompt.
        domain = self.persona_router.determine_domain(self.initial_prompt) # Assuming determine_domain exists in router
        
        # Find relevant files based on prompt and context analysis
        relevant_files = self.context_analyzer.find_relevant_files(self.initial_prompt)
        
        logger.info(f"Context analysis complete. Domain: {domain}, Relevant files: {len(relevant_files)}")
        
        return {
            "domain": domain,
            "relevant_files": relevant_files,
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
            remaining_budget_chars = (self.token_manager.phase_budgets.get("context", 1000) - current_context_tokens) * 4 # Estimate chars from tokens
            
            if remaining_budget_chars <= 0: break

            # Extract key elements and relevant code segments
            key_elements = self.context_analyzer._extract_key_elements(content) # Assuming this method exists
            relevant_segment = self.context_analyzer.extract_relevant_code_segments(
                content, max_chars=int(remaining_budget_chars)
            )
            
            file_context_part = (
                f"File: {file_path}\n"
                f"Key elements: {key_elements}\n"
                f"Content snippet:\n```\n{relevant_segment}\n```\n"
            )
            
            # Check if adding this file's context would exceed the budget
            estimated_file_tokens = self.llm_provider.count_tokens(file_context_part)
            
            if current_context_tokens + estimated_file_tokens > self.token_manager.phase_budgets.get("context", 1000):
                logger.info(f"Skipping {file_path} due to context budget. "
                            f"Current: {current_context_tokens}, Estimated for file: {estimated_file_tokens}, "
                            f"Budget: {self.token_manager.phase_budgets.get('context', 1000)}")
                break # Stop adding files if budget is exceeded
            
            context_parts.append(file_context_part)
            current_context_tokens += estimated_file_tokens
            
        logger.info(f"Prepared context with {len(context_parts)} files, total estimated tokens: {current_context_tokens}")
        return "\n".join(context_parts)
    
    def _generate_persona_sequence(self, context_analysis: Dict[str, Any]) -> List[str]:
        """Generate the sequence of personas to participate in the debate."""
        # This method is called by run_debate, and it calls persona_router.determine_persona_sequence
        # The changes for self-analysis and dynamic routing are in persona_router.py
        
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
    
    # ... (rest of the original core.py methods like _run_debate_round, etc.) ...
    # The _run_debate_round method is modified above to integrate token manager.
    # The original _calculate_token_budgets is removed.
    # The original _check_token_budget is modified.
    # The original _synthesize_final_answer is modified.

# Additional helper functions (load_personas_from_yaml, load_frameworks_from_yaml)
# These are not directly modified by the requested changes, but are part of the original core.py.
# They are kept for completeness.
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