# src/utils/error_handler.py
import logging
import sys  # NEW: Import sys for sys.excepthook
import traceback  # NEW: Import traceback for handle_exception
from functools import wraps
from typing import (  # NEW: Added Type for handle_exception signature
    Any,
    Callable,
    Optional,
)

from src.exceptions import ChimeraError  # Import the enhanced ChimeraError

logger = logging.getLogger(__name__)


# NEW: Add log_event function
def log_event(message: str, level: str = "info", data: Optional[dict[str, Any]] = None):
    """
    Logs a structured event using the module's logger.
    Args:
        message: The main log message.
        level: The logging level (e.g., 'info', 'warning', 'error', 'critical').
        data: Optional dictionary of additional structured data to log.
    """
    log_method = getattr(logger, level.lower(), logger.info)
    log_method(message, extra=data)


# NEW: Add handle_exception function for sys.excepthook
def handle_exception(
    exc_type: type[BaseException], exc_value: BaseException, exc_traceback: Any
):
    """
    Global exception handler for sys.excepthook.
    Logs unhandled exceptions with full traceback.
    """
    if issubclass(exc_type, KeyboardInterrupt):
        # Don't log KeyboardInterrupt, just let it terminate
        sys.__excepthook__(exc_type, exc_value, exc_traceback)
        return

    log_event(
        "Unhandled exception caught by sys.excepthook",
        level="critical",
        data={
            "exception_type": exc_type.__name__,
            "exception_message": str(exc_value),
            "stack_trace": "".join(
                traceback.format_exception(exc_type, exc_value, exc_traceback)
            ),
        },
    )


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
                error_dict.pop("message", None)  # Remove 'message' to avoid conflict
                log_method(
                    f"Structured error in {func.__name__}: {error_msg}",
                    extra=error_dict,
                )
                # --- END MODIFICATION ---

                raise e  # Re-raise the structured error for upstream handling
            except Exception as e:
                # Use str(e) for general exceptions too, and ensure ChimeraError
                # is instantiated with the correct parameters (message, details, original_exception)
                error_msg = str(e)
                chimera_error = ChimeraError(
                    message=f"An unexpected error occurred in {func.__name__}: {error_msg}",
                    details={
                        "function": func.__name__,
                        "args_snippet": str(args)[:200],  # Limit for logging
                        "kwargs_snippet": str(kwargs)[:200],
                        "error_type": type(e).__name__,  # Add error_type
                        "error_details": error_msg,  # Add error_details
                        "error_code": "UNEXPECTED_ERROR",  # Add error_code to details
                    },
                    original_exception=e,
                )
                log_method = getattr(logger, log_level.lower())
                error_dict = chimera_error.to_dict()
                error_dict.pop("message", None)  # Remove 'message' to avoid conflict
                log_method(
                    f"Unstructured error wrapped: {chimera_error.message}",
                    extra=error_dict,
                )
                raise chimera_error from e  # FIX B904

        return wrapper

    return decorator
