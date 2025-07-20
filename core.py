# core.py
import yaml
from pydantic import BaseModel, Field, ValidationError
from typing import List, Dict, Tuple
from llm_provider import GeminiProvider
import os

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

def run_isal_process(prompt: str, api_key: str) -> Tuple[str, Dict[str, str]]:
    """
    Runs the Iterative Socratic Arbitration Loop (ISAL) process.
    Returns the final synthesized answer and a dictionary of intermediate steps.
    """
    personas = load_personas()
    gemini_provider = GeminiProvider(api_key=api_key)
    
    intermediate_steps: Dict[str, str] = {}
    current_thought = prompt # The evolving thought/answer

    # --- Step 1: Initial Generation (Visionary_Generator) ---
    visionary_persona = next((p for p in personas if p.name == "Visionary_Generator"), None)
    if visionary_persona:
        print(f"Running persona: {visionary_persona.name}...")
        visionary_response = gemini_provider.generate(
            prompt=prompt, # Initial prompt for the first generator
            system_prompt=visionary_persona.system_prompt,
            temperature=visionary_persona.temperature,
            max_tokens=visionary_persona.max_tokens
        )
        intermediate_steps["Visionary_Generator_Output"] = visionary_response
        current_thought = visionary_response # This becomes the base for critique
    else:
        print("Warning: Visionary_Generator persona not found. Skipping initial generation.")
        intermediate_steps["Visionary_Generator_Output"] = "N/A - Persona not found."
        current_thought = prompt # Fallback to original prompt if no generator

    # --- Step 2: Skeptical Critique ---
    skeptical_persona = next((p for p in personas if p.name == "Skeptical_Generator"), None)
    if skeptical_persona and "[ERROR]" not in current_thought: # Only proceed if no prior error
        print(f"Running persona: {skeptical_persona.name}...")
        skeptical_critique = gemini_provider.generate(
            prompt=f"Critique the following proposal/idea from a highly skeptical, risk-averse perspective. Identify at least three potential failure points or critical vulnerabilities:\n\n{current_thought}",
            system_prompt=skeptical_persona.system_prompt,
            temperature=skeptical_persona.temperature,
            max_tokens=skeptical_persona.max_tokens
        )
        intermediate_steps["Skeptical_Critique"] = skeptical_critique
    else:
        intermediate_steps["Skeptical_Critique"] = "N/A - Persona not found or previous step failed."

    # --- Step 3: Constructive Criticism & Improvement ---
    constructive_persona = next((p for p in personas if p.name == "Constructive_Critic"), None)
    if constructive_persona and "[ERROR]" not in current_thought:
        print(f"Running persona: {constructive_persona.name}...")
        # Combine original thought and skeptical critique for constructive feedback
        combined_input = f"Original Proposal:\n{current_thought}\n\nSkeptical Critique:\n{intermediate_steps.get('Skeptical_Critique', 'No skeptical critique provided.')}\n\nBased on the above, provide specific, actionable improvements to the original proposal."
        
        constructive_feedback = gemini_provider.generate(
            prompt=combined_input,
            system_prompt=constructive_persona.system_prompt,
            temperature=constructive_persona.temperature,
            max_tokens=constructive_persona.max_tokens
        )
        intermediate_steps["Constructive_Feedback"] = constructive_feedback
        
        # Optionally, you could have another step here to "revise" the current_thought
        # based on the constructive_feedback, but for MVP, we'll synthesize later.
    else:
        intermediate_steps["Constructive_Feedback"] = "N/A - Persona not found or previous step failed."

    # --- Step 4: Impartial Arbitration/Synthesis ---
    arbitrator_persona = next((p for p in personas if p.name == "Impartial_Arbitrator"), None)
    final_answer = "Error: Arbitration failed." # Default error
    if arbitrator_persona:
        print(f"Running persona: {arbitrator_persona.name}...")
        # Prepare input for the arbitrator
        arbitration_input = f"""
        Original Prompt: {prompt}

        Visionary Proposal:
        {intermediate_steps.get('Visionary_Generator_Output', 'N/A')}

        Skeptical Critique:
        {intermediate_steps.get('Skeptical_Critique', 'N/A')}

        Constructive Feedback:
        {intermediate_steps.get('Constructive_Feedback', 'N/A')}

        Synthesize the above information into a single, balanced, and definitive final answer. Incorporate the best elements from all inputs, address critiques, and propose a refined solution.
        """
        final_answer = gemini_provider.generate(
            prompt=arbitration_input,
            system_prompt=arbitrator_persona.system_prompt,
            temperature=arbitrator_persona.temperature,
            max_tokens=arbitrator_persona.max_tokens
        )
        intermediate_steps["Arbitrator_Output"] = final_answer
    else:
        intermediate_steps["Arbitrator_Output"] = "N/A - Persona not found."
        final_answer = "Error: Impartial_Arbitrator persona not found."

    # --- Step 5: Devil's Advocate (Optional, but good for robustness) ---
    devils_advocate_persona = next((p for p in personas if p.name == "Devils_Advocate"), None)
    if devils_advocate_persona and "[ERROR]" not in final_answer:
        print(f"Running persona: {devils_advocate_persona.name}...")
        devils_advocate_critique = gemini_provider.generate(
            prompt=f"Critique the following final synthesized answer. Find the single most critical, fundamental flaw. Do not offer solutions, only expose the weakness:\n\n{final_answer}",
            system_prompt=devils_advocate_persona.system_prompt,
            temperature=devils_advocate_persona.temperature,
            max_tokens=devils_advocate_persona.max_tokens
        )
        intermediate_steps["Devils_Advocate_Critique"] = devils_advocate_critique
        # For MVP, we just display this. In a more advanced loop, this would trigger another revision.
    else:
        intermediate_steps["Devils_Advocate_Critique"] = "N/A - Persona not found or final answer has errors."

    return final_answer, intermediate_steps
