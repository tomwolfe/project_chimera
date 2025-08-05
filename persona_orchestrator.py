# persona_orchestrator.py
import yaml
import logging
from typing import Dict, Any, List
import json # Added import for json.dumps

# Assume llm_provider and llm_output_validator are available
# from llm_provider import call_llm_persona
# from llm_output_validator import validate_llm_output

# Mock implementations for demonstration purposes
# In a real app, these would be imported from their respective modules
class MockLLMProvider:
    def call_llm_persona(self, persona_config: Dict[str, Any], context: Dict[str, Any], retries: int = 3, delay: int = 5) -> str:
        persona_name = persona_config.get('name', 'Unknown')
        logging.info(f"Mock LLM Call: {persona_name}")
        if persona_name == "Impartial_Arbitrator":
            # Simulate a valid, structured response for the mock
            return json.dumps({
                "COMMIT_MESSAGE": "Mock Synthesized Change",
                "RATIONALE": "Mock rationale addressing previous critiques.",
                "CODE_CHANGES": [
                    {
                        "file_path": "mock_file.py",
                        "action": "ADD",
                        "full_content": "def mock_function():\n    print('Mock function')\n"
                    }
                ]
            })
        else:
            return json.dumps({
                "COMMIT_MESSAGE": f"Mock response for {persona_name}",
                "RATIONALE": "Mock rationale.",
                "CODE_CHANGES": []
            })

class MockLLMOutputValidator:
    def validate_llm_output(self, raw_output: str) -> dict:
        try:
            data = json.loads(raw_output)
            # Basic validation: check for required keys and CODE_CHANGES structure
            if not isinstance(data, dict) or 'CODE_CHANGES' not in data or not isinstance(data['CODE_CHANGES'], list):
                raise ValueError("Invalid structure or missing CODE_CHANGES.")
            # Simulate successful validation for mock
            logging.info("Mock validation successful.")
            return data
        except (json.JSONDecodeError, ValueError) as e:
            logging.error(f"Mock validation failed: {e}")
            # In a real scenario, this would raise a specific validation error
            raise Exception(f"Mock validation failed: {e}") # Using generic Exception for mock

# Replace with actual imports when available
# from llm_provider import GeminiProvider # Assuming GeminiProvider is the actual class
# from llm_output_validator import validate_llm_output, LLMOutputParsingError, InvalidSchemaError, PathTraversalError # Assuming these are the actual exceptions

# Dummy classes to satisfy imports if actual modules aren't present for the mock
class GeminiProvider:
    def __init__(self, *args, **kwargs): pass
    def call_llm_persona(self, *args, **kwargs): return MockLLMProvider().call_llm_persona(*args, **kwargs)
    def calculate_usd_cost(self, *args, **kwargs): return 0.0
    def count_tokens(self, *args, **kwargs): return 100

class LLMOutputParsingError(Exception): pass
class InvalidSchemaError(LLMOutputParsingError): pass
class PathTraversalError(LLMOutputParsingError): pass
class LLMUnexpectedError(Exception): pass

llm_provider = MockLLMProvider() # Use actual provider in production
llm_output_validator = MockLLMOutputValidator() # Use actual validator in production

class PersonaOrchestrator:
    def __init__(self, personas_config_path: str = 'personas.yaml'):
        self.personas = self._load_personas(personas_config_path)
        self.llm_provider = llm_provider # Use actual provider
        self.validator = llm_output_validator # Use actual validator
        self.context: Dict[str, Any] = {}
        self.persona_sequence = [
            "Visionary_Generator",
            "Skeptical_Generator",
            "Constructive_Critic",
            "Impartial_Arbitrator",
            "Devils_Advocate"
        ] # Default sequence, can be configurable

    def _load_personas(self, config_path: str) -> List[Dict[str, Any]]:
        """Loads persona configurations from a YAML file."""
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
                if config and 'personas' in config:
                    logging.info(f"Loaded {len(config['personas'])} personas from {config_path}")
                    return config['personas']
                else:
                    logging.error(f"Personas not found or invalid format in {config_path}")
                    return []
        except FileNotFoundError:
            logging.error(f"Personas configuration file not found at {config_path}")
            return []
        except Exception as e:
            logging.error(f"Error loading personas from {config_path}: {e}")
            return []

    def run_socratic_debate(self, initial_context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Executes the Socratic debate sequence using the loaded personas.
        Returns the final synthesized output from the Impartial_Arbitrator.
        """
        self.context = initial_context.copy() # Initialize context
        final_synthesized_output = None

        logging.info("Starting Socratic Debate...")

        for persona_name in self.persona_sequence:
            persona_config = next((p for p in self.personas if p['name'] == persona_name), None)
            if not persona_config:
                logging.warning(f"Persona '{persona_name}' not found in configuration. Skipping.")
                continue

            try:
                # Prepare context for the current persona if needed
                # For simplicity, passing the entire context. Could be refined.
                persona_specific_context = self.context.copy()
                persona_specific_context['current_persona'] = persona_name

                # Call the LLM persona
                raw_response = self.llm_provider.call_llm_persona(
                    persona_config, persona_specific_context
                )

                # Validate the output if it's the Impartial_Arbitrator
                if persona_name == "Impartial_Arbitrator":
                    validated_output = self.validator.validate_llm_output(raw_response)
                    final_synthesized_output = validated_output
                    logging.info(f"Successfully processed and validated output from {persona_name}.")
                    # Optionally break here if Impartial_Arbitrator is the final synthesis step
                    # break 
                else:
                    # Store intermediate results or update context for next persona
                    self.context[f'{persona_name}_output'] = raw_response
                    logging.info(f"Processed output from {persona_name}.")

            except (LLMOutputParsingError, InvalidSchemaError, PathTraversalError, Exception) as ve: # Catch specific validation errors and general exceptions
                logging.error(f"Validation or processing failed for {persona_name}: {ve}. Stopping debate.")
                # Handle validation failure: could retry, use fallback, or stop
                final_synthesized_output = {
                    "COMMIT_MESSAGE": "Validation Error",
                    "RATIONALE": f"LLM output from {persona_name} failed validation: {ve}",
                    "CODE_CHANGES": []
                }
                break # Stop the debate on critical validation failure
            except Exception as e: # Catch other errors (e.g., network issues, LLM errors)
                logging.error(f"Error during interaction with {persona_name}: {e}. Stopping debate.")
                # Handle other errors (e.g., network issues, LLM errors)
                final_synthesized_output = {
                    "COMMIT_MESSAGE": "Processing Error",
                    "RATIONALE": f"An error occurred during {persona_name}: {e}",
                    "CODE_CHANGES": []
                }
                break # Stop the debate on critical processing error

        if final_synthesized_output:
             logging.info("Socratic Debate finished.")
             return final_synthesized_output
        else:
             # Handle cases where the debate didn't reach a final output (e.g., sequence incomplete)
             logging.warning("Socratic Debate did not produce a final synthesized output.")
             return {
                 "COMMIT_MESSAGE": "Debate Incomplete",
                 "RATIONALE": "The Socratic debate sequence did not complete successfully.",
                 "CODE_CHANGES": []
             }

# --- Example Usage (e.g., in app.py) ---
# if __name__ == "__main__":
#     # Load initial context (e.g., from user input, files)
#     initial_context = {
#         "user_prompt": "Refactor this code.",
#         "codebase": {"file1.py": "def func(): pass"}
#     }
# 
#     orchestrator = PersonaOrchestrator()
#     result = orchestrator.run_socratic_debate(initial_context)
# 
#     print("Final Result:", json.dumps(result, indent=2))
