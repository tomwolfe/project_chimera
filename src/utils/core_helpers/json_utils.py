# src/utils/json_utils.py
import json
import logging
from typing import Any, Callable, Optional

import numpy as np

logger = logging.getLogger(__name__)


def convert_to_json_friendly(obj: Any) -> Any:
    """Recursively converts Pydantic models and NumPy types to dictionaries/standard Python types
    that are compatible with json.dumps.
    """
    if hasattr(obj, "model_dump"):
        # Pydantic v2 models
        return obj.model_dump()
    elif hasattr(obj, "dict"):  # Fallback for Pydantic v1 if model_dump is not present
        return obj.dict()
    elif isinstance(obj, dict):
        return {k: convert_to_json_friendly(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_json_friendly(item) for item in obj]
    # Handle NumPy scalars (np.generic covers all scalar types like np.float32, np.int64, etc.)
    elif isinstance(obj, np.generic):
        return (
            obj.item()
        )  # Convert to a standard Python scalar (float, int, bool, etc.)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()  # Convert NumPy arrays to Python lists
    return obj


def safe_json_loads(
    json_string: str, default_value: Optional[Any] = None
) -> Optional[Any]:
    """Safely loads a JSON string, returning a default value if parsing fails.
    Logs errors for debugging.
    """
    if not isinstance(json_string, str):
        logger.error(
            f"Error decoding JSON: Input must be a string, not {type(json_string).__name__}."
        )
        return default_value
    try:
        return json.loads(json_string)
    except json.JSONDecodeError as e:
        logger.error(
            f"Error decoding JSON: {e}. Raw string snippet: {json_string[:200]}..."
        )
        return default_value
    except TypeError as e:
        logger.error(
            f"Error decoding JSON (TypeError): {e}. Raw string snippet: {json_string[:200]}..."
        )
        return default_value


def safe_json_dumps(
    data: Any,
    indent: Optional[int] = None,
    default: Optional[
        Callable[[Any], Any]
    ] = None,  # This is the 'default' callable for json.dumps
    on_error_return_str: str = "{}",  # This is the string to return if serialization fails completely
) -> str:
    """Safely dumps data to a JSON string.
    Uses a provided 'default' callable for non-serializable types, or falls back to 'str'.
    If serialization fails completely, returns 'on_error_return_str'.
    Logs errors for debugging.
    """
    try:
        # Attempt to dump with the provided default handler, or fall back to str
        return json.dumps(data, indent=indent, default=default or str)
    except TypeError as e:
        logger.error(
            f"Error encoding JSON: {e}. Attempting to serialize non-serializable types."
        )
        logger.warning("Fallback to returning on_error_return_str due to TypeError.")
        return on_error_return_str
    except Exception as e:
        logger.error(
            f"An unexpected error occurred during JSON encoding: {e}. Data: {str(data)[:200]}..."
        )
        return on_error_return_str
