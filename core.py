# core.py
import yaml
import time
from collections import defaultdict
from pydantic import BaseModel, Field, ValidationError, model_validator
from typing import List, Dict, Tuple, Any, Callable, Optional
from llm_provider import GeminiProvider
from llm_provider import LLMProviderError, GeminiAPIError, LLMUnexpectedError, TOKEN_COSTS_PER_1K_TOKENS # Import custom exceptions and token costs

# --- Custom Exception for Token Budget ---
class TokenBudgetExceededError(LLMProviderError):
    """Raised when an LLM call would exceed the total token budget."""
    pass

# --- Pydantic Models for Persona Configuration ---
class Persona(BaseModel):
    name: str
    system_prompt: str
    temperature: float = Field(..., ge=0.0, le=1.0) # Ensure temperature is between 0 and 1
    max_tokens: int = Field(..., gt=0) # Ensure max_tokens is positive


# New Pydantic model for the overall configuration structure
class FullPersonaConfig(BaseModel):
    personas: List[Persona] = Field(default_factory=list)
    persona_sets: Dict[str, List[str]] = Field(default_factory=lambda: {"General": []})

    @model_validator(mode='after')
    def validate_persona_sets_references(self):
        all_persona_names = {p.name for p in self.personas}
        for set_name, persona_names_in_set in self.persona_sets.items():
            if not isinstance(persona_names_in_set, list):
                raise ValueError(f"Persona set '{set_name}' must be a list of persona names.")
            for p_name in persona_names_in_set:
                if p_name not in all_persona_names:
                    raise ValueError(f"Persona '{p_name}' referenced in set '{set_name}' not found in 'personas' list.")
        return self

    @property
    def default_persona_set(self) -> str:
        """Return the first persona set as default if 'General' doesn't exist"""
        return "General" if "General" in self.persona_sets else next(iter(self.persona_sets.keys()))

# --- Core Logic ---
def load_personas(file_path: str = "personas.yaml") -> Tuple[Dict[str, Persona], Dict[str, List[str]], str]:
    """
    Loads and validates persona configurations and sets from a YAML file.
    Returns a tuple: (all_personas_dict, persona_sets_dict, default_set_name).
    """
    try:
        with open(file_path, 'r') as f:
            data = yaml.safe_load(f)
            
        # Use FullPersonaConfig to validate the entire structure
        full_config = FullPersonaConfig(**data)
        
        all_personas_dict = {p.name: p for p in full_config.personas}
        persona_sets_dict = full_config.persona_sets
        default_set = full_config.default_persona_set

        return all_personas_dict, persona_sets_dict, default_set
    except FileNotFoundError:
        print(f"Error: Persona configuration file not found at '{file_path}'")
        raise
    except ValidationError as e:
        print(f"Error: Invalid persona configuration in {file_path}: {e}")
        raise
    except yaml.YAMLError as e:
        print(f"Error: Could not parse YAML file {file_path}: {e}")
        raise
    except Exception as e:
        print(f"Unexpected error loading personas: {str(e)}")
        raise

class SocraticDebate:
    # Constants for retry and degradation
    DEFAULT_MAX_RETRIES = 3
    MAX_BACKOFF_SECONDS = 30
    BUDGET_TIGHT_THRESHOLD = 2000 # Tokens remaining below which budget is considered tight
    MIN_DEGRADED_TOKENS = 256 # Minimum max_tokens to request during degradation
    DEGRADE_FACTOR = 2 # Factor to divide max_tokens by during degradation (e.g., 1000 -> 500 -> 250)

    # Fallback personas for critical roles if primary fails
    FALLBACK_PERSONAS = {
        "Visionary_Generator": ["Generalist_Assistant"],
        "Skeptical_Generator": ["Generalist_Assistant"],
        "Constructive_Critic": ["Generalist_Assistant"],
        "Impartial_Arbitrator": ["Generalist_Assistant"],
        "Devils_Advocate": ["Generalist_Assistant"]
    }

    def __init__(self,
                 initial_prompt: str,
                 api_key: str,
                 max_total_tokens_budget: int,
                 model_name: str, # Non-default
                 personas: Dict[str, Persona], # The *selected* personas for the current domain
                 all_personas: Dict[str, Persona], # All personas loaded from file
                 persona_sets: Dict[str, List[str]], # All persona sets loaded from file
                 domain: str = "General", # Default argument
                 status_callback: Callable = None): # Default argument
        self.initial_prompt = initial_prompt
        self.max_total_tokens_budget = max_total_tokens_budget
        self.model_name = model_name
        self.personas = personas # Dictionary of personas (the active set for the current domain)
        self.domain = domain
        self.all_personas = all_personas # Store all personas for lookup by get_persona_set
        self.persona_sets = persona_sets # Store all persona sets for lookup by get_persona_set
        self.status_callback = status_callback
        self.gemini_provider = GeminiProvider(api_key=api_key, model_name=model_name, status_callback=self._update_status)
        self.cumulative_token_usage = 0
        self.cumulative_usd_cost = 0.0
        self.intermediate_steps: Dict[str, Any] = {}
        # The evolving thought/answer
        # Initialize with the initial prompt, but this will be updated by the Visionary_Generator
        self.current_thought = initial_prompt
        self.final_answer = "Process did not complete." # Default until arbitrator runs

    def _get_sanitized_step_output(self, key: str, default_message: str = "No relevant input provided.") -> str:
        """
        Retrieves a step's output, sanitizing error/N/A messages for subsequent LLM prompts.
        Returns a clean default_message if the content indicates an error or was skipped.
        """
        content = self.intermediate_steps.get(key) # Get raw content
        if isinstance(content, str) and ("[ERROR]" in content or "N/A - Persona skipped/failed" in content):
            return default_message
        return content if content is not None else default_message # Return original content if valid, else default
    
    def _is_budget_tight(self) -> bool:
        """Determine if the remaining token budget is tight."""
        return (self.max_total_tokens_budget - self.cumulative_token_usage) < self.BUDGET_TIGHT_THRESHOLD

    def _update_status(self, message: str, state: str = "running", expanded: bool = True,
                       current_total_tokens: int = 0, current_total_cost: float = 0.0,
                       estimated_next_step_tokens: int = 0, estimated_next_step_cost: float = 0.0):
        """Internal helper to call the external status callback and print to console."""
        if self.status_callback:
            self.status_callback(
                message=message,
                state=state,
                expanded=expanded,
                current_total_tokens=current_total_tokens,
                current_total_cost=current_total_cost,
                estimated_next_step_tokens=estimated_next_step_tokens,
                estimated_next_step_cost=estimated_next_step_cost
            )
        print(message) # Keep print for rich console capture / CLI output

    def _get_persona(self, name: str) -> Persona:
        """Retrieves a persona by name."""
        # Now use self.all_personas to get any persona by name
        persona = self.all_personas.get(name)
        if not persona:
            self._update_status(f"Error: Persona '{name}' not found in configuration.", state="error")
            raise ValueError(f"Persona '{name}' not found in configuration.")
        return persona

    def _execute_persona_step(self,
                              persona_name: str,
                              step_prompt_generator: Callable[[], str], # Function to generate the prompt for this step
                              output_key: str, # Key to store the output in intermediate_steps
                              update_current_thought: bool = False,
                              is_final_answer_step: bool = False
                             ) -> str:
        """
        Executes a single persona's generation step, handles token budgeting,
        updates cumulative metrics, and stores results.
        Returns the generated response or an error message.
        Implements retry mechanism with fallback personas, graceful degradation,
        and error-specific recovery strategies.
        """
        persona = self._get_persona(persona_name)
        step_output_content = "" # Initialize to empty string
        original_persona_name = persona_name
        recovered_from_error = False
        recovery_message = ""
        
        # Generate the prompt for this specific step
        step_prompt = step_prompt_generator()
        
        # Determine if we're close to budget limit
        budget_tight = self._is_budget_tight()
        
        # Try primary persona with retries
        for retry_count in range(self.DEFAULT_MAX_RETRIES):
            try:
                # Estimate tokens for this step (prompt + max_output_tokens)
                estimated_input_tokens = self.gemini_provider.count_tokens(
                    prompt=step_prompt,
                    system_prompt=persona.system_prompt
                )
                estimated_step_output_tokens = persona.max_tokens
                # Calculate the actual max_tokens to request from the LLM,
                # ensuring it doesn't exceed the overall budget.
                # It should be no more than persona's configured max_tokens,
                # AND no more than the remaining budget after input tokens.
                remaining_budget_for_output = self.max_total_tokens_budget - self.cumulative_token_usage - estimated_input_tokens
                actual_max_output_tokens_to_request = min(persona.max_tokens, max(0, remaining_budget_for_output))
                estimated_step_total_tokens = estimated_input_tokens + actual_max_output_tokens_to_request
                estimated_step_cost = self.gemini_provider.calculate_usd_cost(estimated_input_tokens, actual_max_output_tokens_to_request)
                
                # If budget is tight and this is a retry, apply graceful degradation
                if budget_tight and retry_count > 0:
                    # Reduce max_tokens for retry
                    degraded_max_tokens = max(self.MIN_DEGRADED_TOKENS, persona.max_tokens // (self.DEGRADE_FACTOR ** retry_count))
                    actual_max_output_tokens_to_request = min(degraded_max_tokens, max(0, remaining_budget_for_output))
                    # Update estimated tokens for this degraded retry
                    estimated_step_total_tokens = estimated_input_tokens + actual_max_output_tokens_to_request
                    estimated_step_cost = self.gemini_provider.calculate_usd_cost(estimated_input_tokens, actual_max_output_tokens_to_request)
                    
                    self._update_status(
                        f"Retrying {persona.name} with degraded parameters (max_tokens={actual_max_output_tokens_to_request}) due to tight budget...",
                        state="warning",
                        current_total_tokens=self.cumulative_token_usage,
                        current_total_cost=self.cumulative_usd_cost,
                        estimated_next_step_tokens=estimated_step_total_tokens,
                        estimated_next_step_cost=estimated_step_cost
                    )
                else:
                    self._update_status(
                        f"Running persona: {persona.name}...",
                        current_total_tokens=self.cumulative_token_usage,
                        current_total_cost=self.cumulative_usd_cost,
                        estimated_next_step_tokens=estimated_step_total_tokens,
                        estimated_next_step_cost=estimated_step_cost
                    )
                
                # Check if we're going to exceed budget
                if self.cumulative_token_usage + estimated_step_total_tokens > self.max_total_tokens_budget:
                    error_msg = (f"Step '{persona.name}' would exceed total token budget. "
                                 f"Estimated {estimated_step_total_tokens} tokens for this step, "
                                 f"but only {self.max_total_tokens_budget - self.cumulative_token_usage} remaining.")
                    if retry_count < self.DEFAULT_MAX_RETRIES - 1:
                        # Try with reduced max_tokens
                        degraded_max_tokens = max(self.MIN_DEGRADED_TOKENS, persona.max_tokens // (self.DEGRADE_FACTOR ** (retry_count + 1)))
                        actual_max_output_tokens_to_request = min(degraded_max_tokens, max(0, remaining_budget_for_output))
                        estimated_step_total_tokens = estimated_input_tokens + actual_max_output_tokens_to_request
                        estimated_step_cost = self.gemini_provider.calculate_usd_cost(estimated_input_tokens, actual_max_output_tokens_to_request)
                        
                        self._update_status(
                            f"Attempting with reduced max_tokens={actual_max_output_tokens_to_request} to fit budget...",
                            state="warning",
                            current_total_tokens=self.cumulative_token_usage,
                            current_total_cost=self.cumulative_usd_cost,
                            estimated_next_step_tokens=estimated_step_total_tokens,
                            estimated_next_step_cost=estimated_step_cost
                        )
                        # Continue to try generating with reduced tokens
                    else:
                        # Last attempt failed
                        self.intermediate_steps[output_key] = f"[ERROR] {error_msg}"
                        self.intermediate_steps[f"{output_key.replace('_Output', '')}_Tokens_Used"] = "N/A (Budget Exceeded)"
                        self._update_status(error_msg, state="error")
                        raise TokenBudgetExceededError(error_msg)
                
                response_text, input_tokens_used, output_tokens_used = self.gemini_provider.generate(
                    prompt=step_prompt,
                    system_prompt=persona.system_prompt,
                    temperature=persona.temperature,
                    max_tokens=actual_max_output_tokens_to_request
                )
                
                tokens_used_this_step = input_tokens_used + output_tokens_used
                cost_this_step = self.gemini_provider.calculate_usd_cost(input_tokens_used, output_tokens_used)
                
                # Post-generation strict budget check
                if self.cumulative_token_usage + tokens_used_this_step > self.max_total_tokens_budget:
                    # Calculate the maximum allowed output tokens for this step to stay within budget
                    allowed_tokens_this_step = self.max_total_tokens_budget - self.cumulative_token_usage
                    max_allowed_output_tokens = max(0, allowed_tokens_this_step - input_tokens_used)
                    if output_tokens_used > max_allowed_output_tokens:
                        # Truncate the response_text to approximately fit the allowed tokens.
                        approx_chars_per_token = 4
                        truncated_chars = max_allowed_output_tokens * approx_chars_per_token
                        response_text = response_text[:truncated_chars] + "..." if len(response_text) > truncated_chars else response_text
                        # Adjust reported tokens and cost
                        output_tokens_used = max_allowed_output_tokens
                        tokens_used_this_step = input_tokens_used + output_tokens_used
                        cost_this_step = self.gemini_provider.calculate_usd_cost(input_tokens_used, output_tokens_used)
                        error_msg = (f"Step '{persona.name}' generated more tokens than budget allowed. "
                                     f"Response truncated. Actual tokens used for step: {input_tokens_used} input + {output_tokens_used} output. "
                                     f"Total cumulative tokens: {self.cumulative_token_usage + tokens_used_this_step}. "
                                     f"Budget: {self.max_total_tokens_budget}.")
                        self.intermediate_steps[output_key] = f"[TRUNCATED] {response_text}\n{error_msg}"
                        self.intermediate_steps[f"{output_key.replace('_Output', '')}_Tokens_Used"] = tokens_used_this_step
                        self.cumulative_token_usage += tokens_used_this_step
                        self.cumulative_usd_cost += cost_this_step
                        self._update_status(error_msg, state="warning")  # Not a full error, just a warning
                        # Continue with truncated response
                
                # If we got here without exceptions, the step succeeded
                if retry_count > 0:
                    recovery_message = f"[RECOVERED] After {retry_count} retry(ies)"
                    response_text = f"{recovery_message}\n{response_text}"
                    recovered_from_error = True
                
                # Store the (potentially truncated) response text
                self.intermediate_steps[output_key] = response_text
                self.intermediate_steps[f"{output_key.replace('_Output', '')}_Tokens_Used"] = tokens_used_this_step
                step_output_content = response_text
                self.cumulative_token_usage += tokens_used_this_step
                self.cumulative_usd_cost += cost_this_step
                if update_current_thought:
                    self.current_thought = step_output_content
                if is_final_answer_step:
                    self.final_answer = step_output_content
                
                self._update_status(
                    f"{persona.name} completed. Used {tokens_used_this_step} tokens." + 
                    (" [RECOVERED]" if recovered_from_error else ""),
                    current_total_tokens=self.cumulative_token_usage,
                    current_total_cost=self.cumulative_usd_cost,
                    estimated_next_step_tokens=0, estimated_next_step_cost=0.0
                )
                return step_output_content
                
            except LLMProviderError as e:
                if retry_count < self.DEFAULT_MAX_RETRIES - 1:
                    # Handle error-specific recovery strategies
                    backoff = min(self.MAX_BACKOFF_SECONDS, 2 ** (retry_count + 1))
                    
                    # Error-specific recovery strategies
                    if isinstance(e, GeminiAPIError):
                        if e.code == 429:  # Rate limit error
                            backoff = min(self.MAX_BACKOFF_SECONDS, 5 * (retry_count + 1))  # Longer backoff for rate limits
                            self._update_status(
                                f"Rate limit reached for {persona.name}. Waiting {backoff} seconds before retry...",
                                state="warning"
                            )
                        elif e.code == 404:  # Model not found
                            self._update_status(
                                f"Model not found for {persona.name}. Trying to continue with default model...",
                                state="warning"
                            )
                        elif e.code == 401:  # Unauthorized (invalid API key)
                            self._update_status(
                                f"Invalid API key for {persona.name}. Check your credentials.",
                                state="error"
                            )
                            # Don't retry on invalid API key
                            break
                    
                    self._update_status(
                        f"Error running {persona.name}: {e}. Retrying in {backoff} seconds... (Attempt {retry_count + 1}/{self.DEFAULT_MAX_RETRIES})",
                        state="warning"
                    )
                    time.sleep(backoff)
                    continue
                
                # If retries failed, try fallback personas
                fallback_personas = self.FALLBACK_PERSONAS.get(persona_name, [])
                for fallback_name in fallback_personas:
                    try:
                        fallback_persona = self._get_persona(fallback_name)
                        self._update_status(
                            f"Attempting fallback persona: {fallback_name}...",
                            state="warning"
                        )
                        
                        # Use the fallback persona for this step
                        response_text, input_tokens_used, output_tokens_used = self.gemini_provider.generate(
                            prompt=step_prompt,
                            system_prompt=fallback_persona.system_prompt,
                            temperature=fallback_persona.temperature,
                            max_tokens=min(fallback_persona.max_tokens, actual_max_output_tokens_to_request)
                        )
                        
                        tokens_used_this_step = input_tokens_used + output_tokens_used
                        cost_this_step = self.gemini_provider.calculate_usd_cost(input_tokens_used, output_tokens_used)
                        
                        # Store the fallback response
                        recovery_message = f"[RECOVERED] Using fallback persona: {fallback_name}"
                        response_text = f"{recovery_message}\n{response_text}"
                        self.intermediate_steps[output_key] = response_text
                        self.intermediate_steps[f"{output_key.replace('_Output', '')}_Tokens_Used"] = tokens_used_this_step
                        step_output_content = response_text
                        self.cumulative_token_usage += tokens_used_this_step
                        self.cumulative_usd_cost += cost_this_step
                        if update_current_thought:
                            self.current_thought = step_output_content
                        if is_final_answer_step:
                            self.final_answer = step_output_content
                        
                        self._update_status(
                            f"Successfully used fallback persona {fallback_name} for {original_persona_name}.",
                            current_total_tokens=self.cumulative_token_usage,
                            current_total_cost=self.cumulative_usd_cost
                        )
                        return step_output_content
                        
                    except Exception as fallback_e:
                        self._update_status(
                            f"Fallback persona {fallback_name} also failed: {fallback_e}",
                            state="warning"
                        )
                        continue  # Try next fallback
                
                # If all retries and fallbacks failed
                error_msg = f"[ERROR] {type(e).__name__}: {e}"
                self.intermediate_steps[output_key] = error_msg
                self.intermediate_steps[f"{output_key.replace('_Output', '')}_Tokens_Used"] = "N/A (Error)"
                self._update_status(f"Error running {persona.name}: {e}", state="error")
                raise e  # Re-raise the specific LLM error for app.py to catch
            
            except Exception as e:  # Catch any other unexpected errors during step execution
                if retry_count < self.DEFAULT_MAX_RETRIES - 1:
                    backoff = min(self.MAX_BACKOFF_SECONDS, 2 ** (retry_count + 1))
                    self._update_status(
                        f"Unexpected error: {e}. Retrying in {backoff} seconds... (Attempt {retry_count + 1}/{self.DEFAULT_MAX_RETRIES})",
                        state="warning"
                    )
                    time.sleep(backoff)
                    continue
                
                # If retries failed, try fallback personas
                fallback_personas = self.FALLBACK_PERSONAS.get(persona_name, [])
                for fallback_name in fallback_personas:
                    try:
                        fallback_persona = self._get_persona(fallback_name)
                        self._update_status(
                            f"Attempting fallback persona: {fallback_name}...",
                            state="warning"
                        )
                        
                        # Use the fallback persona for this step
                        response_text, input_tokens_used, output_tokens_used = self.gemini_provider.generate(
                            prompt=step_prompt,
                            system_prompt=fallback_persona.system_prompt,
                            temperature=fallback_persona.temperature,
                            max_tokens=min(fallback_persona.max_tokens, actual_max_output_tokens_to_request)
                        )
                        
                        tokens_used_this_step = input_tokens_used + output_tokens_used
                        cost_this_step = self.gemini_provider.calculate_usd_cost(input_tokens_used, output_tokens_used)
                        
                        # Store the fallback response
                        recovery_message = f"[RECOVERED] Using fallback persona: {fallback_name}"
                        response_text = f"{recovery_message}\n{response_text}"
                        self.intermediate_steps[output_key] = response_text
                        self.intermediate_steps[f"{output_key.replace('_Output', '')}_Tokens_Used"] = tokens_used_this_step
                        step_output_content = response_text
                        self.cumulative_token_usage += tokens_used_this_step
                        self.cumulative_usd_cost += cost_this_step
                        if update_current_thought:
                            self.current_thought = step_output_content
                        if is_final_answer_step:
                            self.final_answer = step_output_content
                        
                        self._update_status(
                            f"Successfully used fallback persona {fallback_name} for {original_persona_name}.",
                            current_total_tokens=self.cumulative_token_usage,
                            current_total_cost=self.cumulative_usd_cost
                        )
                        return step_output_content
                        
                    except Exception as fallback_e:
                        self._update_status(
                            f"Fallback persona {fallback_name} also failed: {fallback_e}",
                            state="warning"
                        )
                        continue  # Try next fallback
                
                error_msg = f"[ERROR] Unexpected error during {persona.name} step: {e}"
                self.intermediate_steps[output_key] = error_msg
                self.intermediate_steps[f"{output_key.replace('_Output', '')}_Tokens_Used"] = "N/A (Error)"
                self._update_status(f"Unexpected error running {persona.name}: {e}", state="error")
                raise LLMUnexpectedError(error_msg)  # Wrap in a known exception type
        
        # This point should not be reached if exceptions are properly handled
        raise LLMUnexpectedError(f"Failed to execute persona step: {persona_name}")

    def run_debate(self) -> Tuple[str, Dict[str, Any]]:
        """
        Orchestrates the Socratic debate process.
        Returns the final synthesized answer and a dictionary of intermediate steps.
        """
        self._update_status("Starting Socratic Arbitration Loop...",
                            current_total_tokens=self.cumulative_token_usage,
                            current_total_cost=self.cumulative_usd_cost)
        # --- Step 1: Initial Generation (Visionary_Generator) ---
        try:
            self._execute_persona_step(
                persona_name="Visionary_Generator",
                step_prompt_generator=lambda: self.initial_prompt,
                output_key="Visionary_Generator_Output",
                update_current_thought=True
            )
        except LLMProviderError:
            self.intermediate_steps["Visionary_Generator_Output"] = "N/A - Persona skipped/failed."
            self.intermediate_steps["Visionary_Generator_Tokens_Used"] = "N/A (Skipped/Failed)"
            self._update_status("Visionary_Generator skipped/failed, subsequent steps may be affected.", state="running")
            # If the first step fails, we might want to stop or continue with a fallback.
            # For now, we'll let the subsequent steps handle "N/A" inputs.
            # Re-raise if it's a critical error like budget exceeded or API error
            if isinstance(self.intermediate_steps.get("Visionary_Generator_Output"), str) and "[ERROR]" in self.intermediate_steps["Visionary_Generator_Output"]:
                raise # Re-raise the original exception
        # --- Step 2: Skeptical Critique ---
        # Check original output to decide if step should run.
        visionary_output_raw = self.intermediate_steps.get("Visionary_Generator_Output", "")
        if not (isinstance(visionary_output_raw, str) and ("[ERROR]" in visionary_output_raw or "N/A" in visionary_output_raw)):
            try:
                visionary_output_sanitized = self._get_sanitized_step_output("Visionary_Generator_Output", "No initial proposal available.")
                self._execute_persona_step(
                    persona_name="Skeptical_Generator",
                    step_prompt_generator=lambda: f"Critique the following proposal/idea from a highly skeptical, risk-averse perspective. Identify at least three potential failure points or critical vulnerabilities. Focus on fundamental flaws, not minor details:\n{visionary_output_sanitized}",
                    output_key="Skeptical_Critique"
                )
            except LLMProviderError:
                self.intermediate_steps["Skeptical_Critique"] = "N/A - Persona skipped/failed."
                self.intermediate_steps["Skeptical_Critique_Tokens_Used"] = "N/A (Skipped/Failed)"
                self._update_status("Skeptical_Generator skipped/failed.", state="running")
                if isinstance(self.intermediate_steps.get("Skeptical_Critique"), str) and "[ERROR]" in self.intermediate_steps["Skeptical_Critique"]:
                    raise # Re-raise the original exception
        else:
            self.intermediate_steps["Skeptical_Critique"] = "N/A - Previous step skipped/failed."
            self.intermediate_steps["Skeptical_Critique_Tokens_Used"] = "N/A (Skipped)"
            self._update_status("Skeptical_Generator skipped due to previous step status.", state="running")
        # --- Step 3: Constructive Criticism & Improvement ---
        # Check original output to decide if step should run.
        visionary_output_raw = self.intermediate_steps.get("Visionary_Generator_Output", "")
        if not (isinstance(visionary_output_raw, str) and ("[ERROR]" in visionary_output_raw or "N/A" in visionary_output_raw)):
            try:
                visionary_output_sanitized = self._get_sanitized_step_output("Visionary_Generator_Output", "No original proposal available.")
                skeptical_critique_sanitized = self._get_sanitized_step_output("Skeptical_Critique", "No skeptical critique provided.")
                self._execute_persona_step(
                    persona_name="Constructive_Critic",
                    step_prompt_generator=lambda: f"Original Proposal:\n{visionary_output_sanitized}\nSkeptical Critique:\n{skeptical_critique_sanitized}\nBased on the above, provide specific, actionable improvements to the original proposal.",
                    output_key="Constructive_Feedback"
                )
            except LLMProviderError:
                self.intermediate_steps["Constructive_Feedback"] = "N/A - Persona skipped/failed."
                self.intermediate_steps["Constructive_Feedback_Tokens_Used"] = "N/A (Skipped/Failed)"
                self._update_status("Constructive_Critic skipped/failed.", state="running")
                if isinstance(self.intermediate_steps.get("Constructive_Feedback"), str) and "[ERROR]" in self.intermediate_steps["Constructive_Feedback"]:
                    raise # Re-raise the original exception
        else:
            self.intermediate_steps["Constructive_Feedback"] = "N/A - Previous step skipped/failed."
            self.intermediate_steps["Constructive_Feedback_Tokens_Used"] = "N/A (Skipped)"
            self._update_status("Constructive_Critic skipped due to previous step status.", state="running")
        # --- Step 4: Impartial Arbitration/Synthesis ---
        # This step should always attempt to run, even if previous steps failed,
        # but its output will reflect the quality of its inputs.
        try:
            visionary_output_arb = self._get_sanitized_step_output("Visionary_Generator_Output", "N/A")
            skeptical_critique_arb = self._get_sanitized_step_output("Skeptical_Critique", "N/A")
            constructive_feedback_arb = self._get_sanitized_step_output("Constructive_Feedback", "N/A")
            self._execute_persona_step(
                persona_name="Impartial_Arbitrator",
                step_prompt_generator=lambda: f"""\nOriginal Prompt: {self.initial_prompt}\nVisionary Proposal:\n{visionary_output_arb}\nSkeptical Critique:\n{skeptical_critique_arb}\nConstructive Feedback:\n{constructive_feedback_arb}\nSynthesize the above information into a single, balanced, and definitive final answer. Incorporate the best elements from all inputs, address critiques, and propose a refined solution. If any previous step resulted in an error, acknowledge it and try to synthesize based on available information, or state the limitation.\n""",
                output_key="Arbitrator_Output",
                is_final_answer_step=True
            )
        except LLMProviderError:
            self.intermediate_steps["Arbitrator_Output"] = "N/A - Persona skipped/failed."
            self.intermediate_steps["Arbitrator_Output_Tokens_Used"] = "N/A (Skipped/Failed)"
            self.final_answer = "Error: Arbitration failed or skipped."
            self._update_status("Impartial_Arbitrator skipped/failed.", state="error") # Keep as error, as this is critical
            if isinstance(self.intermediate_steps.get("Arbitrator_Output"), str) and "[ERROR]" in self.intermediate_steps["Arbitrator_Output"]:
                raise # Re-raise the original exception
        # --- Step 5: Devil's Advocate (Optional, but good for robustness) ---
        # Check original final answer to decide if step should run
        final_answer_raw = self.final_answer
        if not (isinstance(final_answer_raw, str) and ("[ERROR]" in final_answer_raw or "Error: Arbitration failed" in final_answer_raw)):
            try:
                final_answer_sanitized = self._get_sanitized_step_output("Arbitrator_Output", "No final answer available for critique.")
                self._execute_persona_step(
                    persona_name="Devils_Advocate",
                    step_prompt_generator=lambda: f"Critique the following final synthesized answer. Find the single most critical, fundamental flaw. Do not offer solutions, only expose the weakness:\n{final_answer_sanitized}",
                    output_key="Devils_Advocate_Critique"
                )
            except LLMProviderError:
                self.intermediate_steps["Devils_Advocate_Critique"] = "N/A - Persona skipped/failed."
                self.intermediate_steps["Devils_Advocate_Critique_Tokens_Used"] = "N/A (Skipped/Failed)"
                self._update_status("Devils_Advocate skipped/failed.", state="running")
                if isinstance(self.intermediate_steps.get("Devils_Advocate_Critique"), str) and "[ERROR]" in self.intermediate_steps["Devils_Advocate_Critique"]:
                    raise # Re-raise the original exception
        else:
            self.intermediate_steps["Devils_Advocate_Critique"] = "N/A - Final answer has errors/was skipped."
            self.intermediate_steps["Devils_Advocate_Critique_Tokens_Used"] = "N/A (Skipped)"
            self._update_status("Devils_Advocate skipped due to final answer status.", state="running")
        self.intermediate_steps["Total_Tokens_Used"] = self.cumulative_token_usage
        self.intermediate_steps["Total_Estimated_Cost_USD"] = self.cumulative_usd_cost
        self._update_status(f"Socratic Arbitration Loop finished. Total tokens used: {self.cumulative_token_usage:,}. Total estimated cost: ${self.cumulative_usd_cost:.4f}",
                            state="complete", expanded=False,
                            current_total_tokens=self.cumulative_token_usage,
                            current_total_cost=self.cumulative_usd_cost)
        return self.final_answer, self.intermediate_steps

def run_isal_process(
    prompt: str,
    api_key: str,
    max_total_tokens_budget: int = 10000,
    model_name: str = "gemini-2.5-flash-lite",
    domain: str = "auto",
    streamlit_status_callback=None,
    all_personas: Optional[Dict[str, Persona]] = None,
    persona_sets: Optional[Dict[str, List[str]]] = None,
    personas_override: Optional[Dict[str, Persona]] = None
) -> 'SocraticDebate': # Changed return type hint to SocraticDebate instance
    """
    Initializes and returns the SocraticDebate instance.
    The caller is responsible for running the debate and handling exceptions.
    """
    if personas_override:
        personas = personas_override
        domain = "Custom"
    else:
        # Load personas and sets if not provided (for CLI or initial app load)
        if all_personas is None or persona_sets is None:
            all_personas, persona_sets, default_set = load_personas()
        else: # If provided (from app.py session state), use them
            _, _, default_set = load_personas() # Still need default_set from file
        
        # Determine domain to use
        if domain == "auto" and prompt.strip() and api_key.strip(): # Only auto-recommend if prompt and key are present
            from llm_provider import recommend_domain
            domain = recommend_domain(prompt, api_key)
            if domain not in persona_sets:
                domain = default_set
        elif domain not in persona_sets:
            domain = default_set
        
        # Get the personas for the selected domain
        personas = {name: all_personas[name] for name in persona_sets[domain]}

    debate = SocraticDebate(
        initial_prompt=prompt,
        api_key=api_key,
        max_total_tokens_budget=max_total_tokens_budget,
        model_name=model_name,
        personas=personas, # Pass the edited/selected personas
        all_personas=all_personas, # Pass the full dictionary of personas
        persona_sets=persona_sets, # Pass the full dictionary of persona sets
        domain=domain,
        status_callback=streamlit_status_callback
    )
    # Removed the call to debate.run_debate() here.
    # The caller (app.py or main.py) will now call debate.run_debate()
    # and handle its return values and exceptions.

    return debate