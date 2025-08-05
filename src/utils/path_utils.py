# src/utils/path_utils.py
import logging
import re
from pathlib import Path

logger = logging.getLogger(__name__)

# Define markers to identify the project root
PROJECT_ROOT_MARKERS = ['.git', 'config.yaml', 'pyproject.toml']

def find_project_root() -> Path:
    """Finds the project root directory by searching for known markers.
    Starts from the directory of the current file and traverses upwards.
    """
    # Start search from the directory of this file (src/utils)
    current_dir = Path(__file__).resolve().parent
    # Traverse upwards to find the project root
    for _ in range(10): # Limit search depth to prevent infinite loops
        if any(current_dir.joinpath(marker).exists() for marker in PROJECT_ROOT_MARKERS):
            logger.info(f"Project root identified at: {current_dir}")
            return current_dir
        
        parent_path = current_dir.parent
        if parent_path == current_dir: # Reached filesystem root
            break
        current_dir = parent_path
    
    # Fallback if no markers are found after reaching the filesystem root
    logger.warning("Project root markers not found. Falling back to current working directory.")
    return Path('.').resolve()

# Determine the project base directory once at module load time
# This should be the root of the project, not just 'src'
PROJECT_BASE_DIR = find_project_root()

def is_within_base_dir(file_path: Path) -> bool:
    """Checks if a file path is safely within the project base directory.
    Handles potential exceptions during path resolution or comparison.
    """
    try:
        # Resolve the path to handle symlinks and relative paths correctly
        resolved_path = file_path.resolve()
        # Check if the resolved path is a subdirectory of the project base directory
        resolved_path.relative_to(PROJECT_BASE_DIR)
        return True
    except ValueError:
        # Path is not relative to PROJECT_BASE_DIR (outside the scope)
        logger.debug(f"Path '{file_path}' is outside the project base directory '{PROJECT_BASE_DIR}'.")
        return False
    except Exception as e:
        # Catch other potential errors during path operations (e.g., permissions)
        logger.error(f"Error resolving or comparing path '{file_path}' against base directory '{PROJECT_BASE_DIR}': {e}")
        return False

def sanitize_and_validate_file_path(raw_path: str) -> str:
    """Sanitizes and validates a file path for safety against traversal and invalid characters.
    Ensures the path is within the project's base directory.
    """
    if not raw_path:
        raise ValueError("File path cannot be empty.")

    # Basic character sanitization: remove characters invalid in most file systems
    # and control characters. This is a defense-in-depth measure.
    # Added space to forbidden characters as it can be problematic in some contexts.
    sanitized_path_str = re.sub(r'[<>:"|?*\\ \x00-\x1f]', '', raw_path)

    # Prevent absolute paths that might try to escape the intended scope
    # The is_within_base_dir check using relative_to() is the primary defense,
    # but this adds an explicit layer.
    if Path(sanitized_path_str).is_absolute():
         raise ValueError("Absolute paths are not permitted.")

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
