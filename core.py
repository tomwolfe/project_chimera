# core.py
import yaml
from pydantic import BaseModel, Field, ValidationError
from typing import List, Dict, Tuple, Any # Added Any for Dict[str, Any]
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
def load_personas(file_path: str = "personas.yaml") -> List[Persona]:
    """Loads and validates persona configurations from a YAML file."""
    try:
        with open(file_path, 'r') as f:
            data = yaml.safe_load(f)
        config = PersonaConfig(personas=data.get('personas', []))
        return config.personas
    except FileNotFoundError:
        print(f"Error: Persona configuration file not found at {file_path}")
        raise
    except ValidationError as e:
        print(f"Error: Invalid persona configuration in {file_path}: {e}")
        raise
    except yaml.YAMLError as e:
        print(f"Error: Could not parse YAML file {file_path}: {e}")
        raise

def run_isal_process(
    prompt: str,
    api_key: str,
    max_total_tokens_budget: int = 10000,  # Default budget for the entire process
    model_name: str = "gemini-2.5-flash-lite", # New: Model selection parameter
    streamlit_status_callback=None # Changed parameter name for clarity
) -> Tuple[str, Dict[str, Any]]: # Changed Dict[str, str] to Dict[str, Any] to accommodate int token counts
    """
    Runs the Iterative Socratic Arbitration Loop (ISAL) process.
    Returns the final synthesized answer and a dictionary of intermediate steps.
    """
    personas = load_personas()
    
    cumulative_token_usage = 0 # Tracks total tokens used
    cumulative_usd_cost = 0.0 # Tracks total estimated USD cost
    intermediate_steps: Dict[str, Any] = {} # Changed to Any
    current_thought = prompt # The evolving thought/answer

    # Helper function to update Streamlit status and print to console (for logs)
    # This function now accepts more detailed metrics for the UI
    def update_status(message: str, state: str = "running", expanded: bool = True,
                      current_total_tokens: int = 0, current_total_cost: float = 0.0,
                      estimated_next_step_tokens: int = 0, estimated_next_step_cost: float = 0.0):
        if streamlit_status_callback:
            streamlit_status_callback(
                message=message,
                state=state,
                expanded=expanded,
                current_total_tokens=current_total_tokens,
                current_total_cost=current_total_cost,
                estimated_next_step_tokens=estimated_next_step_tokens,
                estimated_next_step_cost=estimated_next_step_cost
            )
        print(message) # Keep print for rich console capture / CLI output

    # Initialize GeminiProvider, passing the status update callback
    gemini_provider = GeminiProvider(api_key=api_key, model_name=model_name, status_callback=update_status)
    
    update_status("Starting Socratic Arbitration Loop...",
                  current_total_tokens=cumulative_token_usage,
                  current_total_cost=cumulative_usd_cost)
    # Helper to get persona by name
    def get_persona(name):
        return next((p for p in personas if p.name == name), None)

    # --- Step 1: Initial Generation (Visionary_Generator) ---
    visionary_persona = get_persona("Visionary_Generator")
    if visionary_persona:
        step_name_prefix = "Visionary_Generator"
        try:
            # Estimate tokens for this step (prompt + max_output_tokens)
            estimated_input_tokens = gemini_provider.count_tokens(
                prompt=prompt,
                system_prompt=visionary_persona.system_prompt
            )
            estimated_step_output_tokens = visionary_persona.max_tokens
            estimated_step_total_tokens = estimated_input_tokens + estimated_step_output_tokens
            estimated_step_cost = gemini_provider.calculate_usd_cost(estimated_input_tokens, estimated_step_output_tokens)

            update_status(f"Running persona: {visionary_persona.name}...",
                          current_total_tokens=cumulative_token_usage,
                          current_total_cost=cumulative_usd_cost,
                          estimated_next_step_tokens=estimated_step_total_tokens,
                          estimated_next_step_cost=estimated_step_cost)

            if cumulative_token_usage + estimated_step_total_tokens > max_total_tokens_budget:
                error_msg = (f"Step '{visionary_persona.name}' would exceed total token budget. "
                             f"Estimated {estimated_step_total_tokens} tokens for this step, "
                             f"but only {max_total_tokens_budget - cumulative_token_usage} remaining. Process stopped.")
                intermediate_steps[f"{step_name_prefix}_Output"] = f"[ERROR] {error_msg}"
                intermediate_steps[f"{step_name_prefix}_Tokens_Used"] = "N/A (Budget Exceeded)"
                update_status(error_msg, state="error")
                raise TokenBudgetExceededError(error_msg)
            
            visionary_response, input_tokens_used, output_tokens_used = gemini_provider.generate(
                prompt=prompt, # Initial prompt for the first generator
                system_prompt=visionary_persona.system_prompt,
                temperature=visionary_persona.temperature,
                max_tokens=visionary_persona.max_tokens
            )
            tokens_used_this_step = input_tokens_used + output_tokens_used
            cost_this_step = gemini_provider.calculate_usd_cost(input_tokens_used, output_tokens_used)

            intermediate_steps[f"{step_name_prefix}_Output"] = visionary_response
            intermediate_steps[f"{step_name_prefix}_Tokens_Used"] = tokens_used_this_step # Store per-step usage
            cumulative_token_usage += tokens_used_this_step
            cumulative_usd_cost += cost_this_step
            current_thought = visionary_response # This becomes the base for critique
            update_status(f"{visionary_persona.name} completed. Used {tokens_used_this_step} tokens.",
                          current_total_tokens=cumulative_token_usage,
                          current_total_cost=cumulative_usd_cost,
                          estimated_next_step_tokens=0, estimated_next_step_cost=0.0) # Reset next step estimate
        except LLMProviderError as e: # Catch specific LLM errors
            intermediate_steps[f"{step_name_prefix}_Output"] = f"[ERROR] {type(e).__name__}: {e}"
            intermediate_steps[f"{step_name_prefix}_Tokens_Used"] = "N/A (Error)"
            update_status(f"Error running {visionary_persona.name}: {e}", state="error")
            raise e # Re-raise the specific LLM error for app.py to catch
    else:
        update_status("Warning: Visionary_Generator persona not found. Skipping initial generation.", state="warning")
        intermediate_steps["Visionary_Generator_Output"] = "N/A - Persona not found."
        intermediate_steps["Visionary_Generator_Tokens_Used"] = "N/A (Persona Missing)"
        current_thought = prompt # Fallback to original prompt if no generator

    # --- Step 2: Skeptical Critique ---
    skeptical_persona = get_persona("Skeptical_Generator")
    prev_step_output = intermediate_steps.get("Visionary_Generator_Output", "")
    if skeptical_persona and not (prev_step_output.startswith("[ERROR]") or prev_step_output.startswith("Skipped:") or prev_step_output.startswith("N/A - Persona not found.")):
        step_name_prefix = "Skeptical_Critique"
        step_prompt = f"Critique the following proposal/idea from a highly skeptical, risk-averse perspective. Identify at least three potential failure points or critical vulnerabilities:\n\n{current_thought}"
        try:
            estimated_input_tokens = gemini_provider.count_tokens(
                prompt=step_prompt,
                system_prompt=skeptical_persona.system_prompt
            )
            estimated_step_output_tokens = skeptical_persona.max_tokens
            estimated_step_total_tokens = estimated_input_tokens + estimated_step_output_tokens
            estimated_step_cost = gemini_provider.calculate_usd_cost(estimated_input_tokens, estimated_step_output_tokens)

            update_status(f"Running persona: {skeptical_persona.name}...",
                          current_total_tokens=cumulative_token_usage,
                          current_total_cost=cumulative_usd_cost,
                          estimated_next_step_tokens=estimated_step_total_tokens,
                          estimated_next_step_cost=estimated_step_cost)

            if cumulative_token_usage + estimated_step_total_tokens > max_total_tokens_budget:
                error_msg = (f"Step '{skeptical_persona.name}' would exceed total token budget. "
                             f"Estimated {estimated_step_total_tokens} tokens for this step, "
                             f"but only {max_total_tokens_budget - cumulative_token_usage} remaining. Process stopped.")
                intermediate_steps[f"{step_name_prefix}"] = f"[ERROR] {error_msg}"
                intermediate_steps[f"{step_name_prefix}_Tokens_Used"] = "N/A (Budget Exceeded)"
                update_status(error_msg, state="error")
                raise TokenBudgetExceededError(error_msg)

            skeptical_critique, input_tokens_used, output_tokens_used = gemini_provider.generate(
                prompt=step_prompt,
                system_prompt=skeptical_persona.system_prompt,
                temperature=skeptical_persona.temperature,
                max_tokens=skeptical_persona.max_tokens
            )
            tokens_used_this_step = input_tokens_used + output_tokens_used
            cost_this_step = gemini_provider.calculate_usd_cost(input_tokens_used, output_tokens_used)
            intermediate_steps[f"{step_name_prefix}"] = skeptical_critique
            intermediate_steps[f"{step_name_prefix}_Tokens_Used"] = tokens_used_this_step
            cumulative_token_usage += tokens_used_this_step
            cumulative_usd_cost += cost_this_step
            update_status(f"{skeptical_persona.name} completed. Used {tokens_used_this_step} tokens.",
                          current_total_tokens=cumulative_token_usage,
                          current_total_cost=cumulative_usd_cost,
                          estimated_next_step_tokens=0, estimated_next_step_cost=0.0)
        except LLMProviderError as e:
            intermediate_steps[f"{step_name_prefix}"] = f"[ERROR] {type(e).__name__}: {e}"
            intermediate_steps[f"{step_name_prefix}_Tokens_Used"] = "N/A (Error)"
            update_status(f"Error running {skeptical_persona.name}: {e}", state="error")
            raise e
    else:
        intermediate_steps["Skeptical_Critique"] = "N/A - Persona not found or previous step was skipped/failed."
        intermediate_steps["Skeptical_Critique_Tokens_Used"] = "N/A (Skipped/Persona Missing)"
        if skeptical_persona:
            update_status("Skeptical_Generator skipped due to previous step status or persona not found.", state="warning")

    # --- Step 3: Constructive Criticism & Improvement ---
    constructive_persona = get_persona("Constructive_Critic")
    prev_step_output = intermediate_steps.get("Visionary_Generator_Output", "") # Check Visionary_Generator again as it's the base
    if constructive_persona and not (prev_step_output.startswith("[ERROR]") or prev_step_output.startswith("Skipped:") or prev_step_output.startswith("N/A - Persona not found.")):
        step_name_prefix = "Constructive_Feedback"
        # Combine original thought and skeptical critique for constructive feedback
        combined_input = f"Original Proposal:\n{current_thought}\n\nSkeptical Critique:\n{intermediate_steps.get('Skeptical_Critique', 'No skeptical critique provided.')}\n\nBased on the above, provide specific, actionable improvements to the original proposal."
        
        try:
            estimated_input_tokens = gemini_provider.count_tokens(
                prompt=combined_input,
                system_prompt=constructive_persona.system_prompt
            )
            estimated_step_output_tokens = constructive_persona.max_tokens
            estimated_step_total_tokens = estimated_input_tokens + estimated_step_output_tokens
            estimated_step_cost = gemini_provider.calculate_usd_cost(estimated_input_tokens, estimated_step_output_tokens)

            update_status(f"Running persona: {constructive_persona.name}...",
                          current_total_tokens=cumulative_token_usage,
                          current_total_cost=cumulative_usd_cost,
                          estimated_next_step_tokens=estimated_step_total_tokens,
                          estimated_next_step_cost=estimated_step_cost)

            if cumulative_token_usage + estimated_step_total_tokens > max_total_tokens_budget:
                error_msg = (f"Step '{constructive_persona.name}' would exceed total token budget. "
                             f"Estimated {estimated_step_total_tokens} tokens for this step, "
                             f"but only {max_total_tokens_budget - cumulative_token_usage} remaining. Process stopped.")
                intermediate_steps[f"{step_name_prefix}"] = f"[ERROR] {error_msg}"
                intermediate_steps[f"{step_name_prefix}_Tokens_Used"] = "N/A (Budget Exceeded)"
                update_status(error_msg, state="error")
                raise TokenBudgetExceededError(error_msg)

            constructive_feedback, input_tokens_used, output_tokens_used = gemini_provider.generate(
                prompt=combined_input,
                system_prompt=constructive_persona.system_prompt,
                temperature=constructive_persona.temperature,
                max_tokens=constructive_persona.max_tokens
            )
            tokens_used_this_step = input_tokens_used + output_tokens_used
            cost_this_step = gemini_provider.calculate_usd_cost(input_tokens_used, output_tokens_used)
            intermediate_steps[f"{step_name_prefix}"] = constructive_feedback
            intermediate_steps[f"{step_name_prefix}_Tokens_Used"] = tokens_used_this_step
            cumulative_token_usage += tokens_used_this_step
            cumulative_usd_cost += cost_this_step
            update_status(f"{constructive_persona.name} completed. Used {tokens_used_this_step} tokens.",
                          current_total_tokens=cumulative_token_usage,
                          current_total_cost=cumulative_usd_cost,
                          estimated_next_step_tokens=0, estimated_next_step_cost=0.0)
        except LLMProviderError as e:
            intermediate_steps[f"{step_name_prefix}"] = f"[ERROR] {type(e).__name__}: {e}"
            intermediate_steps[f"{step_name_prefix}_Tokens_Used"] = "N/A (Error)"
            update_status(f"Error running {constructive_persona.name}: {e}", state="error")
            raise e
    else:
        intermediate_steps["Constructive_Feedback"] = "N/A - Persona not found or previous step was skipped/failed."
        intermediate_steps["Constructive_Feedback_Tokens_Used"] = "N/A (Skipped/Persona Missing)"
        if constructive_persona:
            update_status("Constructive_Critic skipped due to previous step status or persona not found.", state="warning")

    # --- Step 4: Impartial Arbitration/Synthesis ---
    arbitrator_persona = get_persona("Impartial_Arbitrator")
    final_answer = "Error: Arbitration failed." # Default error
    if arbitrator_persona:
        step_name_prefix = "Arbitrator_Output"
        # Prepare input for the arbitrator
        arbitration_input = f"""
        Original Prompt: {prompt}

        Visionary Proposal:
        {intermediate_steps.get('Visionary_Generator_Output', 'N/A')}

        Skeptical Critique:
        {intermediate_steps.get('Skeptical_Critique', 'N/A')}

        Constructive Feedback:
        {intermediate_steps.get('Constructive_Feedback', 'N/A')}

        Synthesize the above information into a single, balanced, and definitive final answer. Incorporate the best elements from all inputs, address critiques, and propose a refined solution. If any previous step resulted in an error, acknowledge it and try to synthesize based on available information, or state the limitation.
        """
        try:
            estimated_input_tokens = gemini_provider.count_tokens(
                prompt=arbitration_input,
                system_prompt=arbitrator_persona.system_prompt
            )
            estimated_step_output_tokens = arbitrator_persona.max_tokens
            estimated_step_total_tokens = estimated_input_tokens + estimated_step_output_tokens
            estimated_step_cost = gemini_provider.calculate_usd_cost(estimated_input_tokens, estimated_step_output_tokens)

            update_status(f"Running persona: {arbitrator_persona.name}...",
                          current_total_tokens=cumulative_token_usage,
                          current_total_cost=cumulative_usd_cost,
                          estimated_next_step_tokens=estimated_step_total_tokens,
                          estimated_next_step_cost=estimated_step_cost)

            if cumulative_token_usage + estimated_step_total_tokens > max_total_tokens_budget:
                error_msg = (f"Step '{arbitrator_persona.name}' would exceed total token budget. "
                             f"Estimated {estimated_step_total_tokens} tokens for this step, "
                             f"but only {max_total_tokens_budget - cumulative_token_usage} remaining. "
                             f"Cannot provide final answer. Process stopped.")
                final_answer = f"[ERROR] {error_msg}"
                intermediate_steps[f"{step_name_prefix}"] = final_answer
                intermediate_steps[f"{step_name_prefix}_Tokens_Used"] = "N/A (Budget Exceeded)"
                update_status(error_msg, state="error")
                raise TokenBudgetExceededError(error_msg)

            final_answer, input_tokens_used, output_tokens_used = gemini_provider.generate(
                prompt=arbitration_input,
                system_prompt=arbitrator_persona.system_prompt,
                temperature=arbitrator_persona.temperature,
                max_tokens=arbitrator_persona.max_tokens
            )
            tokens_used_this_step = input_tokens_used + output_tokens_used
            cost_this_step = gemini_provider.calculate_usd_cost(input_tokens_used, output_tokens_used)
            intermediate_steps[f"{step_name_prefix}"] = final_answer
            intermediate_steps[f"{step_name_prefix}_Tokens_Used"] = tokens_used_this_step
            cumulative_token_usage += tokens_used_this_step
            cumulative_usd_cost += cost_this_step
            update_status(f"{arbitrator_persona.name} completed. Used {tokens_used_this_step} tokens.",
                          current_total_tokens=cumulative_token_usage,
                          current_total_cost=cumulative_usd_cost,
                          estimated_next_step_tokens=0, estimated_next_step_cost=0.0)
        except LLMProviderError as e:
            final_answer = f"[ERROR] {type(e).__name__}: Arbitration failed: {e}"
            intermediate_steps[f"{step_name_prefix}"] = final_answer
            intermediate_steps[f"{step_name_prefix}_Tokens_Used"] = "N/A (Error)"
            update_status(f"Error running {arbitrator_persona.name}: {e}", state="error")
            raise e
    else:
        intermediate_steps["Arbitrator_Output"] = "N/A - Persona not found."
        intermediate_steps["Arbitrator_Output_Tokens_Used"] = "N/A (Persona Missing)"
        final_answer = "Error: Impartial_Arbitrator persona not found."
        update_status("Impartial_Arbitrator persona not found.", state="error")

    # --- Step 5: Devil's Advocate (Optional, but good for robustness) ---
    devils_advocate_persona = get_persona("Devils_Advocate")
    # Only run if the final answer itself isn't an error or was skipped
    if devils_advocate_persona and not (final_answer.startswith("[ERROR]") or final_answer.startswith("Error: Impartial_Arbitrator skipped")):
        step_name_prefix = "Devils_Advocate_Critique"
        step_prompt = f"Critique the following final synthesized answer. Find the single most critical, fundamental flaw. Do not offer solutions, only expose the weakness:\n\n{final_answer}"
        try:
            estimated_input_tokens = gemini_provider.count_tokens(
                prompt=step_prompt,
                system_prompt=devils_advocate_persona.system_prompt
            )
            estimated_step_output_tokens = devils_advocate_persona.max_tokens
            estimated_step_total_tokens = estimated_input_tokens + estimated_step_output_tokens
            estimated_step_cost = gemini_provider.calculate_usd_cost(estimated_input_tokens, estimated_step_output_tokens)

            update_status(f"Running persona: {devils_advocate_persona.name}...",
                          current_total_tokens=cumulative_token_usage,
                          current_total_cost=cumulative_usd_cost,
                          estimated_next_step_tokens=estimated_step_total_tokens,
                          estimated_next_step_cost=estimated_step_cost)

            if cumulative_token_usage + estimated_step_total_tokens > max_total_tokens_budget:
                error_msg = (f"Step '{devils_advocate_persona.name}' would exceed total token budget. "
                             f"Estimated {estimated_step_total_tokens} tokens for this step, "
                             f"but only {max_total_tokens_budget - cumulative_token_usage} remaining. Process stopped.")
                intermediate_steps[f"{step_name_prefix}"] = f"[ERROR] {error_msg}"
                intermediate_steps[f"{step_name_prefix}_Tokens_Used"] = "N/A (Budget Exceeded)"
                update_status(error_msg, state="error")
                raise TokenBudgetExceededError(error_msg)

            devils_advocate_critique, input_tokens_used, output_tokens_used = gemini_provider.generate(
                prompt=step_prompt,
                system_prompt=devils_advocate_persona.system_prompt,
                temperature=devils_advocate_persona.temperature,
                max_tokens=devils_advocate_persona.max_tokens
            )
            tokens_used_this_step = input_tokens_used + output_tokens_used
            cost_this_step = gemini_provider.calculate_usd_cost(input_tokens_used, output_tokens_used)
            intermediate_steps[f"{step_name_prefix}"] = devils_advocate_critique
            intermediate_steps[f"{step_name_prefix}_Tokens_Used"] = tokens_used_this_step
            cumulative_token_usage += tokens_used_this_step
            cumulative_usd_cost += cost_this_step
            update_status(f"{devils_advocate_persona.name} completed. Used {tokens_used_this_step} tokens.",
                          current_total_tokens=cumulative_token_usage,
                          current_total_cost=cumulative_usd_cost,
                          estimated_next_step_tokens=0, estimated_next_step_cost=0.0)
        except LLMProviderError as e:
            intermediate_steps[f"{step_name_prefix}"] = f"[ERROR] {type(e).__name__}: {e}"
            intermediate_steps[f"{step_name_prefix}_Tokens_Used"] = "N/A (Error)"
            update_status(f"Error running {devils_advocate_persona.name}: {e}", state="error")
            raise e
    else:
        intermediate_steps["Devils_Advocate_Critique"] = "N/A - Persona not found or final answer has errors/was skipped."
        intermediate_steps["Devils_Advocate_Critique_Tokens_Used"] = "N/A (Skipped/Persona Missing)"
        if devils_advocate_persona:
            update_status("Devils_Advocate skipped due to previous step status or persona not found.", state="warning")

    intermediate_steps["Total_Tokens_Used"] = cumulative_token_usage # Add token count to steps
    intermediate_steps["Total_Estimated_Cost_USD"] = cumulative_usd_cost # Store as float
    update_status(f"Socratic Arbitration Loop finished. Total tokens used: {cumulative_token_usage}. Total estimated cost: ${cumulative_usd_cost:.4f}",
                  state="complete", expanded=False,
                  current_total_tokens=cumulative_token_usage,
                  current_total_cost=cumulative_usd_cost)
    return final_answer, intermediate_steps