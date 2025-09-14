# src/utils/json_utils.py
import json
import numpy as np
from typing import Any, Dict, List


def convert_to_json_friendly(obj: Any) -> Any:
    """
    Recursively converts Pydantic models and NumPy types to dictionaries/standard Python types
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
