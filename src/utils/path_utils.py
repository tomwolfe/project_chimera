# src/utils/path_utils.py
import logging
import re
from pathlib import Path
from typing import Optional # Added for find_project_root signature

logger = logging.getLogger(__name__)

PROJECT_ROOT_MARKERS = ['.git', 'config.yaml', 'pyproject.toml', 'Dockerfile']

# Renamed to avoid confusion with the module-level variable
def _find_project_root_internal(start_path: Path) -> Optional[Path]:
    """Internal helper to find the project root without raising an error."""
    current_dir = start_path
    # Limit search depth to prevent infinite loops in unusual file structures
    for _ in range(15):
        if any(current_dir.joinpath(marker).exists() for marker in PROJECT_ROOT_MARKERS):
            return current_dir
        
        parent_path = current_dir.parent
        if parent_path == current_dir: # Reached the filesystem root
            break
        current_dir = parent_path
    return None # Return None if not found

# --- Define PROJECT_ROOT dynamically ---
# This should be called once at module load time.
_initial_start_path = Path(__file__).resolve().parent
_found_root = _find_project_root_internal(_initial_start_path)

if _found_root:
    PROJECT_ROOT = _found_root
    logger.info(f"Project root identified at: {PROJECT_ROOT}")
else:
    # Fallback if project root markers are not found
    PROJECT_ROOT = Path.cwd() # Fallback to current working directory
    logger.warning(f"Project root markers ({PROJECT_ROOT_MARKERS}) not found after searching up to 15 levels from {_initial_start_path}. Falling back to CWD: {PROJECT_ROOT}. Path validation might be less effective.")

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