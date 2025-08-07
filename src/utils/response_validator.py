"""
Structured validation and correction system for LLM responses.
"""

from typing import Any, Dict, Type, Optional
from pydantic import BaseModel, ValidationError
import json
import re
from src.exceptions import LLMResponseValidationError

class LLMResponseValidator:
    """Validates and corrects LLM responses against expected schemas."""
    
    @staticmethod
    def validate_response(response: Any, schema_model: Type[BaseModel]) -> Dict[str, Any]:
        """
        Validate an LLM response against a Pydantic model schema.
        
        Returns the validated response as a dictionary.
        """
        try:
            # Handle string responses that should be JSON
            if isinstance(response, str):
                # Try to extract JSON from markdown code blocks
                json_match = re.search(r'```(?:json)?\s*({.*?})\s*```', response, re.DOTALL)
                if json_match:
                    response = json_match.group(1)
                
                # Parse JSON string
                try:
                    response = json.loads(response)
                except json.JSONDecodeError:
                    # Try to fix common JSON issues
                    response = LLMResponseValidator._attempt_json_fix(response)
            
            # Validate against schema
            validated = schema_model(**response)
            return validated.model_dump()
            
        except (TypeError, ValueError, json.JSONDecodeError) as e:
            raise LLMResponseValidationError(
                f"Response is not valid JSON: {str(e)}",
                invalid_response=response,
                expected_schema=schema_model.__name__
            )
        except ValidationError as e:
            raise LLMResponseValidationError(
                f"Response failed schema validation: {str(e)}",
                invalid_response=response,
                expected_schema=schema_model.__name__,
                details={"validation_errors": e.errors()}
            )
    
    @staticmethod
    def _attempt_json_fix(json_str: str) -> Dict[str, Any]:
        """Attempt to fix common JSON formatting issues in LLM responses."""
        # Fix missing quotes around keys
        json_str = re.sub(r'(?<!")(\w+)(?=\s*:)', r'"\1"', json_str)
        # Fix single quotes to double quotes
        json_str = json_str.replace("'", '"')
        # Fix trailing commas
        json_str = re.sub(r',\s*}', '}', json_str)
        json_str = re.sub(r',\s*]', ']', json_str)
        
        try:
            return json.loads(json_str)
        except json.JSONDecodeError as e:
            # If still invalid, raise with context
            raise ValueError(f"Could not fix JSON: {str(e)}\nOriginal: {json_str[:200]}...") from None
    
    @staticmethod
    def generate_correction_prompt(invalid_response: Any, schema_model: Type[BaseModel], 
                                 error_message: str) -> str:
        """
        Generate a prompt to correct an invalid LLM response.
        
        This prompt can be used to request a corrected response from the LLM.
        """
        schema_description = schema_model.model_json_schema()
        
        return f"""Previous response was invalid. Please correct it according to these requirements:

**Error**: {error_message}

**Required JSON Schema**:
{json.dumps(schema_description, indent=2)}

**Previous Invalid Response**:
{json.dumps(invalid_response, indent=2) if isinstance(invalid_response, dict) else invalid_response}

Provide ONLY the corrected JSON response with no additional text or explanation. Ensure:
- All required fields are present
- Values match the expected types
- JSON is properly formatted with double quotes
- No trailing commas
- No comments in the JSON
"""
