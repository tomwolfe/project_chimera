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
    max_total_tokens_budget: int = 10000 # Default budget for the entire process
) -> Tuple[str, Dict[str, str]]:
    """
    Runs the Iterative Socratic Arbitration Loop (ISAL) process.
    Returns the final synthesized answer and a dictionary of intermediate steps.
    """
    personas = load_personas()
    gemini_provider = GeminiProvider(api_key=api_key)
    
    cumulative_token_usage = 0
    intermediate_steps: Dict[str, str] = {}
    current_thought = prompt # The evolving thought/answer

    # --- Step 1: Initial Generation (Visionary_Generator) ---
    visionary_persona = next((p for p in personas if p.name == "Visionary_Generator"), None)
    if visionary_persona:
        print(f"Running persona: {visionary_persona.name} (Tokens used so far: {cumulative_token_usage}/{max_total_tokens_budget})...")
        if cumulative_token_usage + visionary_persona.max_tokens > max_total_tokens_budget:
            intermediate_steps["Visionary_Generator_Output"] = "Skipped: Exceeded total token budget."
            print("Warning: Visionary_Generator skipped due to token budget.")
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
            except LLMProviderError as e:
                intermediate_steps["Visionary_Generator_Output"] = f"[ERROR] {type(e).__name__}: {e}"
                print(f"Error running Visionary_Generator: {e}")
                current_thought = prompt # Fallback to original prompt if error
    else:
        print("Warning: Visionary_Generator persona not found. Skipping initial generation.")
        intermediate_steps["Visionary_Generator_Output"] = "N/A - Persona not found."
        current_thought = prompt # Fallback to original prompt if no generator

    # --- Step 2: Skeptical Critique ---
    # Only proceed if the previous step was successful (no error message in its output)
    skeptical_persona = next((p for p in personas if p.name == "Skeptical_Generator"), None)
    if skeptical_persona and not intermediate_steps["Visionary_Generator_Output"].startswith("[ERROR]"):
        print(f"Running persona: {skeptical_persona.name} (Tokens used so far: {cumulative_token_usage}/{max_total_tokens_budget})...")
        if cumulative_token_usage + skeptical_persona.max_tokens > max_total_tokens_budget:
            intermediate_steps["Skeptical_Critique"] = "Skipped: Exceeded total token budget."
            print("Warning: Skeptical_Generator skipped due to token budget.")
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
            except LLMProviderError as e:
                intermediate_steps["Skeptical_Critique"] = f"[ERROR] {type(e).__name__}: {e}"
                print(f"Error running Skeptical_Generator: {e}")
    else:
        intermediate_steps["Skeptical_Critique"] = "N/A - Persona not found or previous step failed."

    # --- Step 3: Constructive Criticism & Improvement ---
    constructive_persona = next((p for p in personas if p.name == "Constructive_Critic"), None)
    if constructive_persona and not intermediate_steps["Visionary_Generator_Output"].startswith("[ERROR]"):
        print(f"Running persona: {constructive_persona.name} (Tokens used so far: {cumulative_token_usage}/{max_total_tokens_budget})...")
        # Combine original thought and skeptical critique for constructive feedback
        combined_input = f"Original Proposal:\n{current_thought}\n\nSkeptical Critique:\n{intermediate_steps.get('Skeptical_Critique', 'No skeptical critique provided.')}\n\nBased on the above, provide specific, actionable improvements to the original proposal."
        
        if cumulative_token_usage + constructive_persona.max_tokens > max_total_tokens_budget:
            intermediate_steps["Constructive_Feedback"] = "Skipped: Exceeded total token budget."
            print("Warning: Constructive_Critic skipped due to token budget.")
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
            except LLMProviderError as e:
                intermediate_steps["Constructive_Feedback"] = f"[ERROR] {type(e).__name__}: {e}"
                print(f"Error running Constructive_Critic: {e}")
    else:
        intermediate_steps["Constructive_Feedback"] = "N/A - Persona not found or previous step failed."

    # --- Step 4: Impartial Arbitration/Synthesis ---
    arbitrator_persona = next((p for p in personas if p.name == "Impartial_Arbitrator"), None)
    final_answer = "Error: Arbitration failed." # Default error
    if arbitrator_persona:
        print(f"Running persona: {arbitrator_persona.name} (Tokens used so far: {cumulative_token_usage}/{max_total_tokens_budget})...")
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
            print("Error: Impartial_Arbitrator skipped due to token budget.")
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
            except LLMProviderError as e:
                final_answer = f"[ERROR] {type(e).__name__}: Arbitration failed: {e}"
                intermediate_steps["Arbitrator_Output"] = final_answer
                print(f"Error running Impartial_Arbitrator: {e}")
    else:
        intermediate_steps["Arbitrator_Output"] = "N/A - Persona not found."
        final_answer = "Error: Impartial_Arbitrator persona not found."

    # --- Step 5: Devil's Advocate (Optional, but good for robustness) ---
    devils_advocate_persona = next((p for p in personas if p.name == "Devils_Advocate"), None)
    # Only run if the final answer itself isn't an error
    if devils_advocate_persona and not final_answer.startswith("[ERROR]"):
        print(f"Running persona: {devils_advocate_persona.name} (Tokens used so far: {cumulative_token_usage}/{max_total_tokens_budget})...")
        if cumulative_token_usage + devils_advocate_persona.max_tokens > max_total_tokens_budget:
            intermediate_steps["Devils_Advocate_Critique"] = "Skipped: Exceeded total token budget."
            print("Warning: Devils_Advocate skipped due to token budget.")
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
            except LLMProviderError as e:
                intermediate_steps["Devils_Advocate_Critique"] = f"[ERROR] {type(e).__name__}: {e}"
                print(f"Error running Devils_Advocate: {e}")
    else:
        intermediate_steps["Devils_Advocate_Critique"] = "N/A - Persona not found or final answer has errors."

    intermediate_steps["Total_Tokens_Used"] = str(cumulative_token_usage) # Add token count to steps
    return final_answer, intermediate_steps