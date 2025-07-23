# core.py
import yaml
from pydantic import BaseModel, Field, ValidationError
from typing import List, Dict, Tuple
from llm_provider import GeminiProvider
from llm_provider import LLMProviderError, GeminiAPIError, LLMUnexpectedError # Import custom exceptions

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
    streamlit_status=None # Added parameter for Streamlit status updates
) -> Tuple[str, Dict[str, str]]:
    """
    Runs the Iterative Socratic Arbitration Loop (ISAL) process.
    Returns the final synthesized answer and a dictionary of intermediate steps.
    """
    personas = load_personas()
    gemini_provider = GeminiProvider(api_key=api_key, model_name=model_name)
    
    cumulative_token_usage = 0
    intermediate_steps: Dict[str, str] = {}
    current_thought = prompt # The evolving thought/answer

    # Helper function to update Streamlit status and print to console (for logs)
    def update_status(message: str, state: str = "running", expanded: bool = True):
        if streamlit_status:
            streamlit_status.update(label=message, state=state, expanded=expanded)
        print(message) # Keep print for rich console capture / CLI output

    update_status("Starting Socratic Arbitration Loop...")

    # --- Step 1: Initial Generation (Visionary_Generator) ---
    visionary_persona = next((p for p in personas if p.name == "Visionary_Generator"), None)
    if visionary_persona:
        update_status(f"Running persona: {visionary_persona.name} (Tokens used so far: {cumulative_token_usage}/{max_total_tokens_budget})...")
        if cumulative_token_usage + visionary_persona.max_tokens > max_total_tokens_budget:
            intermediate_steps["Visionary_Generator_Output"] = "Skipped: Exceeded total token budget."
            update_status("Warning: Visionary_Generator skipped due to token budget.", state="warning")
        else:
            try:
                visionary_response, tokens_used = gemini_provider.generate(
                    prompt=prompt, # Initial prompt for the first generator
                    system_prompt=visionary_persona.system_prompt,
                    temperature=visionary_persona.temperature,
                    max_tokens=visionary_persona.max_tokens
                )
                intermediate_steps["Visionary_Generator_Output"] = visionary_response
                cumulative_token_usage += tokens_used
                current_thought = visionary_response # This becomes the base for critique
                update_status(f"{visionary_persona.name} completed.", state="running") # Keep running for next step
            except LLMProviderError as e: # Catch specific LLM errors
                intermediate_steps["Visionary_Generator_Output"] = f"[ERROR] {type(e).__name__}: {e}"
                update_status(f"Error running {visionary_persona.name}: {e}", state="error")
                raise e # Re-raise the specific LLM error for app.py to catch
    else:
        update_status("Warning: Visionary_Generator persona not found. Skipping initial generation.", state="warning")
        intermediate_steps["Visionary_Generator_Output"] = "N/A - Persona not found."
        current_thought = prompt # Fallback to original prompt if no generator

    # --- Step 2: Skeptical Critique ---
    # Only proceed if the previous step was successful (no error message in its output)
    prev_step_output = intermediate_steps.get("Visionary_Generator_Output", "")
    if skeptical_persona and not (prev_step_output.startswith("[ERROR]") or prev_step_output.startswith("Skipped:") or prev_step_output.startswith("N/A - Persona not found.")):
        update_status(f"Running persona: {skeptical_persona.name} (Tokens used so far: {cumulative_token_usage}/{max_total_tokens_budget})...")
        if cumulative_token_usage + skeptical_persona.max_tokens > max_total_tokens_budget:
            intermediate_steps["Skeptical_Critique"] = "Skipped: Exceeded total token budget."
            update_status("Warning: Skeptical_Generator skipped due to token budget.", state="warning")
        else:
            try:
                skeptical_critique, tokens_used = gemini_provider.generate(
                    prompt=f"Critique the following proposal/idea from a highly skeptical, risk-averse perspective. Identify at least three potential failure points or critical vulnerabilities:\n\n{current_thought}",
                    system_prompt=skeptical_persona.system_prompt,
                    temperature=skeptical_persona.temperature,
                    max_tokens=skeptical_persona.max_tokens
                )
                intermediate_steps["Skeptical_Critique"] = skeptical_critique
                cumulative_token_usage += tokens_used
                update_status(f"{skeptical_persona.name} completed.", state="running")
            except LLMProviderError as e: # Catch specific LLM errors
                intermediate_steps["Skeptical_Critique"] = f"[ERROR] {type(e).__name__}: {e}"
                update_status(f"Error running {skeptical_persona.name}: {e}", state="error")
                raise e # Re-raise the specific LLM error
    else:
        intermediate_steps["Skeptical_Critique"] = "N/A - Persona not found or previous step was skipped/failed."
        if skeptical_persona:
            update_status("Skeptical_Generator skipped due to previous step status or persona not found.", state="warning")


    # --- Step 3: Constructive Criticism & Improvement ---
    prev_step_output = intermediate_steps.get("Visionary_Generator_Output", "") # Check Visionary_Generator again as it's the base
    if constructive_persona and not (prev_step_output.startswith("[ERROR]") or prev_step_output.startswith("Skipped:") or prev_step_output.startswith("N/A - Persona not found.")):
        update_status(f"Running persona: {constructive_persona.name} (Tokens used so far: {cumulative_token_usage}/{max_total_tokens_budget})...")
        # Combine original thought and skeptical critique for constructive feedback
        combined_input = f"Original Proposal:\n{current_thought}\n\nSkeptical Critique:\n{intermediate_steps.get('Skeptical_Critique', 'No skeptical critique provided.')}\n\nBased on the above, provide specific, actionable improvements to the original proposal."
        
        if cumulative_token_usage + constructive_persona.max_tokens > max_total_tokens_budget:
            intermediate_steps["Constructive_Feedback"] = "Skipped: Exceeded total token budget."
            update_status("Warning: Constructive_Critic skipped due to token budget.", state="warning")
        else:
            try:
                constructive_feedback, tokens_used = gemini_provider.generate(
                    prompt=combined_input,
                    system_prompt=constructive_persona.system_prompt,
                    temperature=constructive_persona.temperature,
                    max_tokens=constructive_persona.max_tokens
                )
                intermediate_steps["Constructive_Feedback"] = constructive_feedback
                cumulative_token_usage += tokens_used
                update_status(f"{constructive_persona.name} completed.", state="running") # Keep running for next step
            except LLMProviderError as e: # Catch specific LLM errors
                intermediate_steps["Constructive_Feedback"] = f"[ERROR] {type(e).__name__}: {e}"
                update_status(f"Error running {constructive_persona.name}: {e}", state="error")
                raise e # Re-raise the specific LLM error
    else:
        intermediate_steps["Constructive_Feedback"] = "N/A - Persona not found or previous step was skipped/failed."
        if constructive_persona:
            update_status("Constructive_Critic skipped due to previous step status or persona not found.", state="warning")


    # --- Step 4: Impartial Arbitration/Synthesis ---
    arbitrator_persona = next((p for p in personas if p.name == "Impartial_Arbitrator"), None)
    final_answer = "Error: Arbitration failed." # Default error
    if arbitrator_persona:
        update_status(f"Running persona: {arbitrator_persona.name} (Tokens used so far: {cumulative_token_usage}/{max_total_tokens_budget})...")
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
        if cumulative_token_usage + arbitrator_persona.max_tokens > max_total_tokens_budget:
            final_answer = "Error: Impartial_Arbitrator skipped due to total token budget. Cannot provide final answer."
            intermediate_steps["Arbitrator_Output"] = final_answer
            update_status("Error: Impartial_Arbitrator skipped due to token budget.", state="error")
        else:
            try:
                final_answer, tokens_used = gemini_provider.generate(
                    prompt=arbitration_input,
                    system_prompt=arbitrator_persona.system_prompt,
                    temperature=arbitrator_persona.temperature,
                    max_tokens=arbitrator_persona.max_tokens
                )
                intermediate_steps["Arbitrator_Output"] = final_answer
                cumulative_token_usage += tokens_used
                update_status(f"{arbitrator_persona.name} completed.", state="running") # Keep running for next step
            except LLMProviderError as e: # Catch specific LLM errors
                final_answer = f"[ERROR] {type(e).__name__}: Arbitration failed: {e}"
                intermediate_steps["Arbitrator_Output"] = final_answer
                update_status(f"Error running {arbitrator_persona.name}: {e}", state="error")
                raise e # Re-raise the specific LLM error
    else:
        intermediate_steps["Arbitrator_Output"] = "N/A - Persona not found."
        final_answer = "Error: Impartial_Arbitrator persona not found."
        update_status("Impartial_Arbitrator persona not found.", state="error")


    # --- Step 5: Devil's Advocate (Optional, but good for robustness) ---
    devils_advocate_persona = next((p for p in personas if p.name == "Devils_Advocate"), None)
    # Only run if the final answer itself isn't an error or was skipped
    if devils_advocate_persona and not (final_answer.startswith("[ERROR]") or final_answer.startswith("Error: Impartial_Arbitrator skipped")):
        update_status(f"Running persona: {devils_advocate_persona.name} (Tokens used so far: {cumulative_token_usage}/{max_total_tokens_budget})...")
        if cumulative_token_usage + devils_advocate_persona.max_tokens > max_total_tokens_budget:
            intermediate_steps["Devils_Advocate_Critique"] = "Skipped: Exceeded total token budget."
            update_status("Warning: Devils_Advocate skipped due to token budget.", state="warning")
        else:
            try:
                devils_advocate_critique, tokens_used = gemini_provider.generate(
                    prompt=f"Critique the following final synthesized answer. Find the single most critical, fundamental flaw. Do not offer solutions, only expose the weakness:\n\n{final_answer}",
                    system_prompt=devils_advocate_persona.system_prompt,
                    temperature=devils_advocate_persona.temperature,
                    max_tokens=devils_advocate_persona.max_tokens
                )
                intermediate_steps["Devils_Advocate_Critique"] = devils_advocate_critique
                cumulative_token_usage += tokens_used
                update_status(f"{devils_advocate_persona.name} completed.", state="running") # Keep running for next step
            except LLMProviderError as e: # Catch specific LLM errors
                intermediate_steps["Devils_Advocate_Critique"] = f"[ERROR] {type(e).__name__}: {e}"
                update_status(f"Error running {devils_advocate_persona.name}: {e}", state="error")
                raise e # Re-raise the specific LLM error
    else:
        intermediate_steps["Devils_Advocate_Critique"] = "N/A - Persona not found or final answer has errors/was skipped."
        if devils_advocate_persona:
            update_status("Devils_Advocate skipped due to previous step status or persona not found.", state="warning")


    intermediate_steps["Total_Tokens_Used"] = str(cumulative_token_usage) # Add token count to steps
    update_status(f"Socratic Arbitration Loop finished. Total tokens used: {cumulative_token_usage}", state="complete", expanded=False)
    return final_answer, intermediate_steps