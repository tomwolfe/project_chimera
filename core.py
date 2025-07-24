# core.py
import yaml
from pydantic import BaseModel, Field, ValidationError
from typing import List, Dict, Tuple, Any, Callable # Added Callable
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

class PersonaConfig(BaseModel):
    personas: List[Persona]

# --- Core Logic ---
def load_personas(file_path: str = "personas.yaml") -> Dict[str, Persona]:
    """Loads and validates persona configurations from a YAML file, returning a dictionary."""
    try:
        with open(file_path, 'r') as f:
            data = yaml.safe_load(f)
        config = PersonaConfig(personas=data.get('personas', []))
        return {p.name: p for p in config.personas} # Return a dict for easy lookup
    except FileNotFoundError:
        print(f"Error: Persona configuration file not found at {file_path}")
        raise
    except ValidationError as e:
        print(f"Error: Invalid persona configuration in {file_path}: {e}")
        raise
    except yaml.YAMLError as e:
        print(f"Error: Could not parse YAML file {file_path}: {e}")
        raise

class SocraticDebate:
    def __init__(self,
                 initial_prompt: str,
                 api_key: str,
                 max_total_tokens_budget: int,
                 model_name: str,
                 personas: Dict[str, Persona],
                 status_callback: Callable = None):
        
        self.initial_prompt = initial_prompt
        self.max_total_tokens_budget = max_total_tokens_budget
        self.model_name = model_name
        self.personas = personas # Dictionary of personas
        self.status_callback = status_callback
        
        self.gemini_provider = GeminiProvider(api_key=api_key, model_name=model_name, status_callback=self._update_status)
        
        self.cumulative_token_usage = 0
        self.cumulative_usd_cost = 0.0
        self.intermediate_steps: Dict[str, Any] = {}
        self.current_thought = initial_prompt # The evolving thought/answer
        self.final_answer = "Process did not complete." # Default until arbitrator runs

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
        persona = self.personas.get(name)
        if not persona:
            self._update_status(f"Error: Persona '{name}' not found in configuration.", state="error")
            raise ValueError(f"Persona '{name}' not found.")
        return persona

    def _execute_persona_step(self,
                              persona_name: str,
                              step_prompt_generator: Callable[[], str], # Function to generate the prompt for this step
                              output_key: str, # Key to store the output in intermediate_steps
                              update_current_thought: bool = False, # Whether this step's output updates current_thought
                              is_final_answer_step: bool = False # Whether this step produces the final answer
                             ) -> str:
        """
        Executes a single persona's generation step, handles token budgeting,
        updates cumulative metrics, and stores results.
        Returns the generated response or an error message.
        Raises TokenBudgetExceededError or LLMProviderError on failure.
        """
        persona = self._get_persona(persona_name)
        step_output_content = "" # Initialize to empty string
        
        # Generate the prompt for this specific step
        step_prompt = step_prompt_generator()

        try:
            # Estimate tokens for this step (prompt + max_output_tokens)
            estimated_input_tokens = self.gemini_provider.count_tokens(
                prompt=step_prompt,
                system_prompt=persona.system_prompt
            )
            estimated_step_output_tokens = persona.max_tokens
            estimated_step_total_tokens = estimated_input_tokens + estimated_step_output_tokens
            estimated_step_cost = self.gemini_provider.calculate_usd_cost(estimated_input_tokens, estimated_step_output_tokens)

            self._update_status(f"Running persona: {persona.name}...",
                                current_total_tokens=self.cumulative_token_usage,
                                current_total_cost=self.cumulative_usd_cost,
                                estimated_next_step_tokens=estimated_step_total_tokens,
                                estimated_next_step_cost=estimated_step_cost)

            if self.cumulative_token_usage + estimated_step_total_tokens > self.max_total_tokens_budget:
                error_msg = (f"Step '{persona.name}' would exceed total token budget. "
                             f"Estimated {estimated_step_total_tokens} tokens for this step, "
                             f"but only {self.max_total_tokens_budget - self.cumulative_token_usage} remaining. Process stopped.")
                self.intermediate_steps[output_key] = f"[ERROR] {error_msg}"
                self.intermediate_steps[f"{output_key.replace('_Output', '')}_Tokens_Used"] = "N/A (Budget Exceeded)"
                self._update_status(error_msg, state="error")
                raise TokenBudgetExceededError(error_msg)
            
            response_text, input_tokens_used, output_tokens_used = self.gemini_provider.generate(
                prompt=step_prompt,
                system_prompt=persona.system_prompt,
                temperature=persona.temperature,
                max_tokens=persona.max_tokens
            )
            tokens_used_this_step = input_tokens_used + output_tokens_used
            cost_this_step = self.gemini_provider.calculate_usd_cost(input_tokens_used, output_tokens_used)

            step_output_content = response_text
            self.intermediate_steps[output_key] = step_output_content
            self.intermediate_steps[f"{output_key.replace('_Output', '')}_Tokens_Used"] = tokens_used_this_step
            self.cumulative_token_usage += tokens_used_this_step
            self.cumulative_usd_cost += cost_this_step
            
            if update_current_thought:
                self.current_thought = step_output_content
            if is_final_answer_step:
                self.final_answer = step_output_content

            self._update_status(f"{persona.name} completed. Used {tokens_used_this_step} tokens.",
                                current_total_tokens=self.cumulative_token_usage,
                                current_total_cost=self.cumulative_usd_cost,
                                estimated_next_step_tokens=0, estimated_next_step_cost=0.0)
            return step_output_content

        except LLMProviderError as e:
            error_msg = f"[ERROR] {type(e).__name__}: {e}"
            self.intermediate_steps[output_key] = error_msg
            self.intermediate_steps[f"{output_key.replace('_Output', '')}_Tokens_Used"] = "N/A (Error)"
            self._update_status(f"Error running {persona.name}: {e}", state="error")
            raise e # Re-raise the specific LLM error for app.py to catch
        except Exception as e: # Catch any other unexpected errors during step execution
            error_msg = f"[ERROR] Unexpected error during {persona.name} step: {e}"
            self.intermediate_steps[output_key] = error_msg
            self.intermediate_steps[f"{output_key.replace('_Output', '')}_Tokens_Used"] = "N/A (Error)"
            self._update_status(f"Unexpected error running {persona.name}: {e}", state="error")
            raise LLMUnexpectedError(error_msg) # Wrap in a known exception type

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
            self._update_status("Visionary_Generator skipped/failed, subsequent steps may be affected.", state="warning")
            # If the first step fails, we might want to stop or continue with a fallback.
            # For now, we'll let the subsequent steps handle "N/A" inputs.
            # Re-raise if it's a critical error like budget exceeded or API error
            if isinstance(self.intermediate_steps.get("Visionary_Generator_Output"), str) and "[ERROR]" in self.intermediate_steps["Visionary_Generator_Output"]:
                raise # Re-raise the original exception

        # --- Step 2: Skeptical Critique ---
        # Only run if Visionary_Generator_Output is not an error/skipped
        prev_step_output = self.intermediate_steps.get("Visionary_Generator_Output", "")
        if not (isinstance(prev_step_output, str) and ("[ERROR]" in prev_step_output or "N/A" in prev_step_output)):
            try:
                self._execute_persona_step(
                    persona_name="Skeptical_Generator",
                    step_prompt_generator=lambda: f"Critique the following proposal/idea from a highly skeptical, risk-averse perspective. Identify at least three potential failure points or critical vulnerabilities:\n\n{self.current_thought}",
                    output_key="Skeptical_Critique"
                )
            except LLMProviderError:
                self.intermediate_steps["Skeptical_Critique"] = "N/A - Persona skipped/failed."
                self.intermediate_steps["Skeptical_Critique_Tokens_Used"] = "N/A (Skipped/Failed)"
                self._update_status("Skeptical_Generator skipped/failed.", state="warning")
                if isinstance(self.intermediate_steps.get("Skeptical_Critique"), str) and "[ERROR]" in self.intermediate_steps["Skeptical_Critique"]:
                    raise # Re-raise the original exception
        else:
            self.intermediate_steps["Skeptical_Critique"] = "N/A - Previous step skipped/failed."
            self.intermediate_steps["Skeptical_Critique_Tokens_Used"] = "N/A (Skipped)"
            self._update_status("Skeptical_Generator skipped due to previous step status.", state="warning")


        # --- Step 3: Constructive Criticism & Improvement ---
        # Only run if Visionary_Generator_Output is not an error/skipped
        prev_step_output = self.intermediate_steps.get("Visionary_Generator_Output", "")
        if not (isinstance(prev_step_output, str) and ("[ERROR]" in prev_step_output or "N/A" in prev_step_output)):
            try:
                self._execute_persona_step(
                    persona_name="Constructive_Critic",
                    step_prompt_generator=lambda: f"Original Proposal:\n{self.current_thought}\n\nSkeptical Critique:\n{self.intermediate_steps.get('Skeptical_Critique', 'No skeptical critique provided.')}\n\nBased on the above, provide specific, actionable improvements to the original proposal.",
                    output_key="Constructive_Feedback"
                )
            except LLMProviderError:
                self.intermediate_steps["Constructive_Feedback"] = "N/A - Persona skipped/failed."
                self.intermediate_steps["Constructive_Feedback_Tokens_Used"] = "N/A (Skipped/Failed)"
                self._update_status("Constructive_Critic skipped/failed.", state="warning")
                if isinstance(self.intermediate_steps.get("Constructive_Feedback"), str) and "[ERROR]" in self.intermediate_steps["Constructive_Feedback"]:
                    raise # Re-raise the original exception
        else:
            self.intermediate_steps["Constructive_Feedback"] = "N/A - Previous step skipped/failed."
            self.intermediate_steps["Constructive_Feedback_Tokens_Used"] = "N/A (Skipped)"
            self._update_status("Constructive_Critic skipped due to previous step status.", state="warning")

        # --- Step 4: Impartial Arbitration/Synthesis ---
        # This step should always attempt to run, even if previous steps failed,
        # but its output will reflect the quality of its inputs.
        try:
            self._execute_persona_step(
                persona_name="Impartial_Arbitrator",
                step_prompt_generator=lambda: f"""
Original Prompt: {self.initial_prompt}

Visionary Proposal:
{self.intermediate_steps.get('Visionary_Generator_Output', 'N/A')}

Skeptical Critique:
{self.intermediate_steps.get('Skeptical_Critique', 'N/A')}

Constructive Feedback:
{self.intermediate_steps.get('Constructive_Feedback', 'N/A')}

Synthesize the above information into a single, balanced, and definitive final answer. Incorporate the best elements from all inputs, address critiques, and propose a refined solution. If any previous step resulted in an error, acknowledge it and try to synthesize based on available information, or state the limitation.
""",
                output_key="Arbitrator_Output",
                is_final_answer_step=True
            )
        except LLMProviderError:
            self.intermediate_steps["Arbitrator_Output"] = "N/A - Persona skipped/failed."
            self.intermediate_steps["Arbitrator_Output_Tokens_Used"] = "N/A (Skipped/Failed)"
            self.final_answer = "Error: Arbitration failed or skipped."
            self._update_status("Impartial_Arbitrator skipped/failed.", state="error")
            if isinstance(self.intermediate_steps.get("Arbitrator_Output"), str) and "[ERROR]" in self.intermediate_steps["Arbitrator_Output"]:
                raise # Re-raise the original exception

        # --- Step 5: Devil's Advocate (Optional, but good for robustness) ---
        # Only run if the final answer itself isn't an error or was skipped
        if not (isinstance(self.final_answer, str) and ("[ERROR]" in self.final_answer or "Error: Arbitration failed" in self.final_answer)):
            try:
                self._execute_persona_step(
                    persona_name="Devils_Advocate",
                    step_prompt_generator=lambda: f"Critique the following final synthesized answer. Find the single most critical, fundamental flaw. Do not offer solutions, only expose the weakness:\n\n{self.final_answer}",
                    output_key="Devils_Advocate_Critique"
                )
            except LLMProviderError:
                self.intermediate_steps["Devils_Advocate_Critique"] = "N/A - Persona skipped/failed."
                self.intermediate_steps["Devils_Advocate_Critique_Tokens_Used"] = "N/A (Skipped/Failed)"
                self._update_status("Devils_Advocate skipped/failed.", state="warning")
                if isinstance(self.intermediate_steps.get("Devils_Advocate_Critique"), str) and "[ERROR]" in self.intermediate_steps["Devils_Advocate_Critique"]:
                    raise # Re-raise the original exception
        else:
            self.intermediate_steps["Devils_Advocate_Critique"] = "N/A - Final answer has errors/was skipped."
            self.intermediate_steps["Devils_Advocate_Critique_Tokens_Used"] = "N/A (Skipped)"
            self._update_status("Devils_Advocate skipped due to final answer status.", state="warning")

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
    streamlit_status_callback=None
) -> Tuple[str, Dict[str, Any]]:
    """
    Runs the Iterative Socratic Arbitration Loop (ISAL) process.
    Returns the final synthesized answer and a dictionary of intermediate steps.
    """
    personas = load_personas() # Load personas as a dictionary
    
    debate = SocraticDebate(
        initial_prompt=prompt,
        api_key=api_key,
        max_total_tokens_budget=max_total_tokens_budget,
        model_name=model_name,
        personas=personas,
        status_callback=streamlit_status_callback
    )
    
    return debate.run_debate()