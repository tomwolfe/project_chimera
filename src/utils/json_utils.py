import json
import logging
from typing import Any, Dict, Optional, Tuple

logger = logging.getLogger(__name__)


def safe_json_loads(
    json_string: str, default_value: Optional[Any] = None
) -> Optional[Any]:
    """
    Safely loads a JSON string, returning a default value if parsing fails.
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
    data: Any, default_value: str = "{}", indent: Optional[int] = None
) -> str:
    """
    Safely dumps data to a JSON string, returning a default value if dumping fails.
    Handles non-serializable types by converting them to strings.
    Logs errors for debugging.
    """
    try:
        # Attempt to dump with default handler for non-serializable types
        return json.dumps(data, default=str, indent=indent)
    except TypeError as e:
        logger.error(
            f"Error encoding JSON: {e}. Attempting to serialize non-serializable types."
        )
        try:
            # Fallback to a more aggressive default handler if needed, or just return default_value
            return json.dumps(data, default=lambda o: str(o), indent=indent)
        except Exception as final_e:
            logger.error(
                f"Failed to serialize object to JSON even with aggressive default handler: {final_e}. Data: {str(data)[:200]}..."
            )
            return default_value
    except Exception as e:
        logger.error(
            f"An unexpected error occurred during JSON encoding: {e}. Data: {str(data)[:200]}..."
        )
        return default_value
