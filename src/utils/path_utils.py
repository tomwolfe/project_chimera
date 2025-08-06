# src/utils/path_utils.py
import logging
import re
from pathlib import Path

logger = logging.getLogger(__name__)

# Define markers to identify the project root
PROJECT_ROOT_MARKERS = ['.git', 'config.yaml', 'pyproject.toml']

def find_project_root(start_path: Path = None) -> Path:
    """Finds the project root directory by searching for known markers.
    Starts from the directory of the current file and traverses upwards.
    """
    # Start search from the directory of this file (src/utils)
    if start_path is None:
        start_path = Path(__file__).resolve().parent

    current_dir = start_path
    # Traverse upwards to find the project root
    for _ in range(10): # Limit search depth to prevent infinite loops
        if any(current_dir.joinpath(marker).exists() for marker in PROJECT_ROOT_MARKERS):
            logger.info(f"Project root identified at: {current_dir}")
            return current_dir
        
        parent_path = current_dir.parent
        if parent_path == current_dir: # Reached filesystem root
            break
        current_dir = parent_path
    
    # If no markers are found after searching, raise an error.
    # This is more robust than falling back to the current working directory,
    # as it forces the user to ensure the script is run in a project context.
    raise FileNotFoundError(f"Project root markers ({PROJECT_ROOT_MARKERS}) not found starting from {start_path}. Cannot determine project root.")

# --- Define PROJECT_ROOT dynamically ---
# This ensures that paths used by tools like Bandit or pycodestyle are relative to the project root.
PROJECT_ROOT = find_project_root()

def is_within_base_dir(file_path: Path) -> bool:
    """Checks if a file path is safely within the project base directory.
    Handles potential exceptions during path resolution or comparison.
    """
    try:
        # Resolve the path to handle symlinks and relative paths correctly
        resolved_path = file_path.resolve()
        # Check if the resolved path is a subdirectory of the project base directory
        # BUG FIX: Use PROJECT_ROOT instead of PROJECT_BASE_DIR
        resolved_path.relative_to(PROJECT_ROOT)
        return True
    except ValueError:
        # Path is not relative to PROJECT_ROOT (outside the scope)
        # BUG FIX: Use PROJECT_ROOT in the log message
        logger.debug(f"Path '{file_path}' is outside the project base directory '{PROJECT_ROOT}'.")
        return False
    except Exception as e:
        # Catch other potential errors during path operations (e.g., permissions)
        # BUG FIX: Use PROJECT_ROOT in the log message
        logger.error(f"Error resolving or comparing path '{file_path}' against base directory '{PROJECT_ROOT}': {e}")
        return False

def sanitize_and_validate_file_path(raw_path: str) -> str:
    """Sanitizes and validates a file path for safety against traversal and invalid characters.
    Ensures the path is within the project's base directory.
    """
    if not raw_path:
        raise ValueError("File path cannot be empty.")

    # Basic character sanitization: remove characters invalid in most file systems
    # and control characters. This is a defense-in-depth measure.
    # Removed space from forbidden characters as it's a valid path character.
    sanitized_path_str = re.sub(r'[<>:"|?*\\\x00-\x1f]', '', raw_path)

    path_obj = Path(sanitized_path_str)

    # Crucial check: Ensure the path resides within the determined project base directory
    if not is_within_base_dir(path_obj):
        raise ValueError(f"File path '{raw_path}' resolves to a location outside the allowed project directory.")

    # Return the resolved and validated path string
    # Using resolve() here ensures we return a canonical path after validation.
    try:
        return str(path_obj.resolve())
    except Exception as e:
        raise ValueError(f"Failed to resolve validated path '{sanitized_path_str}': {e}") from e