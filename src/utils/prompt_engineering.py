# src/utils/prompt_engineering.py
import json
from typing import Dict, Any

# NEW IMPORTS for Pydantic validation
from pydantic import BaseModel, ValidationError
import logging

# Assuming LLMResponseModel is defined in src/models.py
from src.models import LLMResponseModel

logger = logging.getLogger(__name__)

def generate_structured_prompt(user_input: str, context: dict) -> str:
    """Generates a structured prompt for the LLM based on user input and context."""
    # Applying the LLM's suggested line-length fix
    prompt = (f"User Input: {user_input}\nContext: {context}\n\nProvide a concise and accurate response "
              f"based on the provided information.")
    return prompt

def parse_llm_response(response: str) -> Dict[str, Any]:
    """
    Parses and validates the LLM's raw response using Pydantic.
    This replaces the simple json.loads with robust schema validation.
    """
    try:
        # Attempt to parse the raw response using the defined LLMResponseModel
        # .dict() converts the Pydantic model instance back to a dictionary
        parsed_response = LLMResponseModel.model_validate_json(response).dict()
        return parsed_response
    except ValidationError as e:
        logger.error(f"LLM response validation failed: {e}. Raw response: {response[:500]}...")
        # Re-raise the validation error, or return a structured error response
        # For robustness, we'll re-raise, allowing upstream error handling to catch it.
        # The SocraticDebate's _execute_llm_turn already handles SchemaValidationError.
        raise ValueError(f"LLM response validation failed: {e}") from e
    except json.JSONDecodeError as e:
        logger.error(f"Failed to decode JSON from LLM response: {e}. Raw response: {response[:500]}...")
        raise ValueError(f"LLM response is not valid JSON: {e}") from e
    except Exception as e:
        logger.error(f"An unexpected error occurred during LLM response parsing: {e}. Raw response: {response[:500]}...")
        raise ValueError(f"Unexpected error parsing LLM response: {e}") from e
