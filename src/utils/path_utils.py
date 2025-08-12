# src/utils/path_utils.py
import logging
import re
from pathlib import Path

logger = logging.getLogger(__name__)

PROJECT_ROOT_MARKERS = ['.git', 'config.yaml', 'pyproject.toml', 'Dockerfile'] # Added Dockerfile as a marker

def find_project_root(start_path: Path = None) -> Path:
    """Finds the project root directory by searching for known markers.
    Starts from the directory of the current file and traverses upwards.
    """
    if start_path is None:
        start_path = Path(__file__).resolve().parent

    current_dir = start_path
    # Limit search depth to prevent infinite loops in unusual file structures
    for _ in range(15): # Increased depth slightly for robustness
        if any(current_dir.joinpath(marker).exists() for marker in PROJECT_ROOT_MARKERS):
            logger.info(f"Project root identified at: {current_dir}")
            return current_dir
        
        parent_path = current_dir.parent
        if parent_path == current_dir: # Reached the filesystem root
            break
        current_dir = parent_path
    
    # If root is not found after extensive search, raise an error.
    raise FileNotFoundError(f"Project root markers ({PROJECT_ROOT_MARKERS}) not found after searching up to 15 levels from {start_path}. Cannot determine project root.")

# --- Define PROJECT_ROOT dynamically ---
# This should be called once at module load time.
try:
    PROJECT_ROOT = find_project_root()
except FileNotFoundError as e:
    logger.error(f"Failed to find project root: {e}")
    # Set a fallback or raise a critical error if project root is essential for startup.
    # For this context, we'll assume it's critical and let it fail if not found.
    raise e

def is_within_base_dir(file_path: Path) -> bool:
    """Checks if a file path is safely within the project base directory."""
    try:
        # Resolve the path to handle relative paths and symbolic links
        resolved_path = file_path.resolve()
        # Check if the resolved path is a subpath of the project root
        resolved_path.relative_to(PROJECT_ROOT)
        return True
    except ValueError:
        # Path is not within the project root
        logger.debug(f"Path '{file_path}' (resolved: '{resolved_path}') is outside the project base directory '{PROJECT_ROOT}'.")
        return False
    except FileNotFoundError:
        # The path itself might not exist, which is also a concern for validation
        logger.debug(f"Path '{file_path}' (resolved: '{resolved_path}') does not exist.")
        return False
    except Exception as e:
        # Catch any other unexpected errors during path resolution or comparison
        logger.error(f"Error validating path '{file_path}' against base directory '{PROJECT_ROOT}': {e}")
        return False

def sanitize_and_validate_file_path(raw_path: str) -> str:
    """Sanitizes and validates a file path for safety against traversal and invalid characters.
    Ensures the path is within the project's base directory.
    """
    if not raw_path:
        raise ValueError("File path cannot be empty.")

    # Remove potentially dangerous characters and sequences
    # This regex removes characters that are invalid in filenames on most OS,
    # and also common traversal sequences like '..'.
    # It also removes control characters.
    sanitized_path_str = re.sub(r'[<>:"|?*\x00-\x1f\x7f]', '', raw_path)
    # Explicitly remove '..' sequences to prevent directory traversal
    sanitized_path_str = re.sub(r'\.\./', '', sanitized_path_str)
    # Normalize multiple slashes
    sanitized_path_str = re.sub(r'//+', '/', sanitized_path_str)

    # Convert to Path object for easier manipulation and validation
    path_obj = Path(sanitized_path_str)

    # Check if the path is within the project root
    if not is_within_base_dir(path_obj):
        raise ValueError(f"File path '{raw_path}' (sanitized: '{sanitized_path_str}') resolves to a location outside the allowed project directory.")

    try:
        # Resolve the path to get its absolute, canonical form
        resolved_path = path_obj.resolve()
        return str(resolved_path)
    except Exception as e:
        # Catch errors during path resolution (e.g., if intermediate path components are invalid)
        raise ValueError(f"Failed to resolve validated path '{sanitized_path_str}': {e}") from e