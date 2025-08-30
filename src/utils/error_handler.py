# src/utils/error_handler.py
import logging
from typing import Callable, Any, Dict, Optional
from functools import wraps
from src.exceptions import ChimeraError # Import the enhanced ChimeraError

logger = logging.getLogger(__name__)

def handle_errors(default_return: Any = None, log_level: str = "ERROR"):
    """
    Decorator for standardized error handling across the codebase.
    Catches exceptions, wraps them in ChimeraError, and logs them.
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except ChimeraError as e:
                # If it's already a ChimeraError, just log it with its structured info
                log_method = getattr(logger, log_level.lower())
                # Use str(e) to access the exception message robustly
                error_msg = str(e)
                
                # --- START MODIFICATION ---
                # The ChimeraError.to_dict() method includes a "message" key.
                # When passing this dictionary as 'extra', it conflicts with the LogRecord's
                # built-in 'message' attribute. We must remove it from the 'extra' dict.
                error_dict = e.to_dict()
                error_dict.pop('message', None)  # Remove 'message' to avoid conflict
                log_method(f"Structured error in {func.__name__}: {error_msg}", extra=error_dict)
                # --- END MODIFICATION ---
                
                raise e # Re-raise the structured error for upstream handling
            except Exception as e:
                # Use str(e) for general exceptions too, and ensure ChimeraError
                # is instantiated with the correct parameters (message, details, original_exception)
                error_msg = str(e)
                chimera_error = ChimeraError(
                    message=f"An unexpected error occurred in {func.__name__}: {error_msg}",
                    details={
                        "function": func.__name__,
                        "args_snippet": str(args)[:200], # Limit for logging
                        "kwargs_snippet": str(kwargs)[:200],
                        "error_type": type(e).__name__, # Add error_type
                        "error_details": error_msg, # Add error_details
                        "error_code": "UNEXPECTED_ERROR" # Add error_code to details
                    },
                    original_exception=e
                )
                log_method = getattr(logger, log_level.lower())
                error_dict = chimera_error.to_dict()
                error_dict.pop('message', None)  # Remove 'message' to avoid conflict
                log_method(f"Unstructured error wrapped: {chimera_error.message}", extra=error_dict)
                raise chimera_error # Re-raise the wrapped error
        return wrapper
    return decorator