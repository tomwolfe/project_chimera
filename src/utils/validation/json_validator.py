import json
import logging
from typing import Dict, Any, Optional, Type
from pydantic import BaseModel, ValidationError

logger = logging.getLogger(__name__)


def validate_llm_output(
    raw_output: str, schema_model: Type[BaseModel]
) -> Optional[Dict[str, Any]]:
    """
    Attempts to parse and validate raw LLM output against a Pydantic schema.
    Returns the validated data as a dictionary if successful, None otherwise.
    """
    try:
        # Clean common LLM output artifacts like markdown fences and conversational filler
        # This is a simplified version; a full implementation would use LLMOutputParser's _clean_llm_output
        cleaned_output = raw_output.strip()
        if cleaned_output.startswith("```json") and cleaned_output.endswith("```"):
            cleaned_output = cleaned_output[7:-3].strip()
        elif cleaned_output.startswith("```") and cleaned_output.endswith("```"):
            cleaned_output = cleaned_output[3:-3].strip()

        data = json.loads(cleaned_output)
        validated_data = schema_model.model_validate(data)
        return validated_data.model_dump(by_alias=True)
    except (json.JSONDecodeError, ValidationError) as e:
        logger.warning(
            f"JSON validation failed for schema {schema_model.__name__}: {e}. Raw output snippet: {raw_output[:200]}..."
        )
        return None
    except Exception as e:
        logger.error(f"Unexpected error during JSON validation: {e}", exc_info=True)
        return None
