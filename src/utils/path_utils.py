# src/utils/path_utils.py
import logging
import re
from pathlib import Path

logger = logging.getLogger(__name__)

PROJECT_ROOT_MARKERS = ['.git', 'config.yaml', 'pyproject.toml']

def find_project_root(start_path: Path = None) -> Path:
    """Finds the project root directory by searching for known markers.
    Starts from the directory of the current file and traverses upwards.
    """
    if start_path is None:
        start_path = Path(__file__).resolve().parent

    current_dir = start_path
    for _ in range(10):
        if any(current_dir.joinpath(marker).exists() for marker in PROJECT_ROOT_MARKERS):
            logger.info(f"Project root identified at: {current_dir}")
            return current_dir
        
        parent_path = current_dir.parent
        if parent_path == current_dir:
            break
        current_dir = parent_path
    
    raise FileNotFoundError(f"Project root markers ({PROJECT_ROOT_MARKERS}) not found starting from {start_path}. Cannot determine project root.")

# --- Define PROJECT_ROOT dynamically ---
PROJECT_ROOT = find_project_root()

def is_within_base_dir(file_path: Path) -> bool:
    """Checks if a file path is safely within the project base directory."""
    try:
        resolved_path = file_path.resolve()
        resolved_path.relative_to(PROJECT_ROOT)
        return True
    except ValueError:
        logger.debug(f"Path '{file_path}' is outside the project base directory '{PROJECT_ROOT}'.")
        return False
    except Exception as e:
        logger.error(f"Error resolving or comparing path '{file_path}' against base directory '{PROJECT_ROOT}': {e}")
        return False

def sanitize_and_validate_file_path(raw_path: str) -> str:
    """Sanitizes and validates a file path for safety against traversal and invalid characters.
    Ensures the path is within the project's base directory.
    """
    if not raw_path:
        raise ValueError("File path cannot be empty.")

    sanitized_path_str = re.sub(r'[<>:"|?*\\\x00-\x1f]', '', raw_path)

    path_obj = Path(sanitized_path_str)

    if not is_within_base_dir(path_obj):
        raise ValueError(f"File path '{raw_path}' resolves to a location outside the allowed project directory.")

    try:
        return str(path_obj.resolve())
    except Exception as e:
        raise ValueError(f"Failed to resolve validated path '{sanitized_path_str}': {e}") from e