# src/logging_config.py
import logging
import json
import uuid
from pythonjsonlogger import jsonlogger
from pathlib import Path
import sys

# Get the project root to ensure consistent logging configuration
try:
    # Attempt to import PROJECT_ROOT from path_utils
    from src.utils.path_utils import PROJECT_ROOT
except ImportError:
    # Fallback if path_utils is not available during initial setup or if running standalone
    # This fallback might not be perfect if the script is run from an unexpected directory.
    PROJECT_ROOT = Path.cwd() 
    logging.warning(f"Could not import PROJECT_ROOT from path_utils. Using CWD: {PROJECT_ROOT}")

class RequestIDFilter(logging.Filter):
    """Adds a request_id to log records if present."""
    def filter(self, record):
        # Ensure request_id is always present, default to 'N/A'
        record.request_id = getattr(record, 'request_id', 'N/A')
        return True

def setup_structured_logging(log_level=logging.INFO):
    """Configures the root logger for structured JSON output."""
    logger = logging.getLogger() # Get the root logger
    
    # Prevent adding handlers multiple times if this function is called more than once
    if logger.handlers:
        # If handlers already exist, assume logging is already configured.
        # You might want to clear existing handlers if re-configuration is intended,
        # but for simplicity, we'll just return.
        return

    logger.setLevel(log_level)
    
    # Create a handler that writes to standard output (common for containerized applications)
    log_handler = logging.StreamHandler(sys.stdout)
    
    # Configure the JSON formatter
    formatter = jsonlogger.JsonFormatter(
        '%(asctime)s %(levelname)s %(name)s %(request_id)s %(message)s',
        rename_fields={
            'asctime': 'timestamp',
            'levelname': 'level',
            'name': 'logger_name'
        },
        datefmt='%Y-%m-%dT%H:%M:%S%z' # ISO 8601 format for timestamps
    )
    log_handler.setFormatter(formatter)
    
    # Add the filter to inject the request_id into log records
    log_handler.addFilter(RequestIDFilter())
    
    # Add the handler to the root logger
    logger.addHandler(log_handler)
    
    # Configure specific loggers to manage verbosity from third-party libraries
    # Set these to WARNING or ERROR to reduce noise from libraries like google-genai, httpx, etc.
    logging.getLogger('google').setLevel(logging.WARNING)
    logging.getLogger('urllib3').setLevel(logging.WARNING)
    logging.getLogger('httpx').setLevel(logging.WARNING)
    logging.getLogger('fastapi').setLevel(logging.WARNING)
    logging.getLogger('uvicorn').setLevel(logging.WARNING)
    logging.getLogger('streamlit').setLevel(logging.WARNING) # Streamlit's own logs can be quite verbose

    logger.info("Structured JSON logging configured successfully.")