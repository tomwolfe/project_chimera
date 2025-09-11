# src/services.py
import logging # NEW: Import logging
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__) # NEW: Initialize logger

# Assuming get_external_data is defined elsewhere, e.g., in src/api.py
# from src.api import get_external_data

def process_user_request(request_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Processes the incoming user request data.
    """
    user_id = request_data.get('user_id')
    action = request_data.get('action')

    # Validate input data structure and types
    if not isinstance(user_id, int) or user_id <= 0: # MODIFIED: Validation logic
        raise ValueError("Invalid or missing 'user_id'. Must be a positive integer.")
    if not action or not isinstance(action, str):
        raise ValueError("Invalid or missing 'action'. Must be a string.")

    # Simulate processing logic
    logger.info(f"Processing request for user {user_id} with action {action}") # MODIFIED: Use logger

    # Add more robust error handling for critical operations
    try:
        # Simulate a critical operation that requires careful error handling
        # e.g., database transaction, complex calculation
        result = perform_critical_operation(user_id, action)
        return {"status": "success", "data": result}
    except Exception as e:
        # Log the error with more context
        logger.error(f"CRITICAL ERROR processing user {user_id}, action {action}: {e}", exc_info=True) # MODIFIED: Use logger
        # Re-raise a specific application error or a generic one
        raise RuntimeError(f"Failed to process request for user {user_id}.") from e

# Placeholder for a more complex operation that might fail
def perform_critical_operation(user_id: int, action: str) -> str: # MODIFIED: Added type hints
    """
    Simulates a critical operation that might fail.
    """
    # In a real application, this would involve database calls, complex logic, etc.
    # For demonstration, simulate a potential failure.
    if action == "fail_example":
        raise ValueError("Simulated failure in critical operation.")
    return f"Processed {action} for user {user_id}"

# Assume get_external_data is defined elsewhere and might raise ConnectionError
# from src.api import get_external_data
