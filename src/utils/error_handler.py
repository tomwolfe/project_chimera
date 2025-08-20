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
                log_method(f"Structured error in {func.__name__}: {e.message}", extra=e.to_dict())
                raise e # Re-raise the structured error for upstream handling
            except Exception as e:
                # Wrap any unexpected errors in a generic ChimeraError
                chimera_error = ChimeraError(
                    message=f"An unexpected error occurred in {func.__name__}: {str(e)}",
                    details={
                        "function": func.__name__,
                        "args_snippet": str(args)[:200], # Limit for logging
                        "kwargs_snippet": str(kwargs)[:200]
                    },
                    original_exception=e
                )
                log_method = getattr(logger, log_level.lower())
                log_method(f"Unstructured error wrapped: {chimera_error.message}", extra=chimera_error.to_dict())
                raise chimera_error # Re-raise the wrapped error
        return wrapper
    return decorator
