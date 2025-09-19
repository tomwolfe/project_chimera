# src/utils/path_utils.py
import logging  # Used for logger
import re  # Used for regex in sanitize_and_validate_file_path
from pathlib import Path  # Used for Path objects
from typing import Optional, Dict, Any
import os  # NEW: Import os

logger = logging.getLogger(__name__)

PROJECT_ROOT_MARKERS = [
    ".git",
    "config.yaml",
    "pyproject.toml",
    "Dockerfile",
    "app.py",
    "core.py",
    "src/",
    "tests/",
    "docs/",
    "personas.yaml",
]  # MODIFIED: Added more markers for robustness


def _find_project_root_internal(start_path: Path) -> Optional[Path]:
    """Internal helper to find the project root without raising an error."""
    current_dir = start_path
    # Limit search depth to prevent infinite loops in unusual file structures
    for _ in range(15):
        if any(
            current_dir.joinpath(marker).exists() for marker in PROJECT_ROOT_MARKERS
        ):
            return current_dir

        parent_path = current_dir.parent
        if parent_path == current_dir:  # Reached the filesystem root
            break
        current_dir = parent_path
    return None  # Return None if not found


# --- Define PROJECT_ROOT dynamically ---
_initial_start_path = Path(__file__).resolve().parent
_found_root = _find_project_root_internal(_initial_start_path)

if _found_root:
    PROJECT_ROOT = _found_root
    logger.info(f"Project root identified at: {PROJECT_ROOT}")
else:
    PROJECT_ROOT = Path.cwd()  # Fallback to current working directory
    logger.warning(
        f"Project root markers ({PROJECT_ROOT_MARKERS}) not found after searching up to 15 levels from {_initial_start_path}. Falling back to CWD: {PROJECT_ROOT}. Path validation might be less effective."
    )


def is_within_base_dir(file_path: Path) -> bool:
    """Checks if a file path is safely within the project base directory."""
    try:
        resolved_path = file_path.resolve()
        resolved_path.relative_to(
            PROJECT_ROOT
        )  # This will raise ValueError if not a subpath
        return True
    except ValueError:
        logger.debug(
            f"Path '{file_path}' (resolved: '{resolved_path}') is outside the project base directory '{PROJECT_ROOT}'."
        )
        return False
    except FileNotFoundError:
        logger.debug(
            f"Path '{file_path}' (resolved: '{resolved_path}') does not exist."
        )
        return False
    except Exception as e:
        logger.error(
            f"Error validating path '{file_path}' against base directory '{PROJECT_ROOT}': {e}"
        )
        return False


def sanitize_and_validate_file_path(raw_path: str) -> str:
    """Sanitizes and validates a file path for safety against traversal and invalid characters.
    Ensures the path is within the project's base directory and returns it relative to PROJECT_ROOT.
    """
    if not raw_path:
        raise ValueError("File path cannot be empty.")

    # MODIFIED: Allow forward slashes in sanitized_path_str as they are valid path separators
    sanitized_path_str = re.sub(
        r'[<>:"\\|?*\x00-\x1f\x7f]', "", raw_path
    )  # Removed '/' from invalid chars
    sanitized_path_str = re.sub(r"\.\./", "", sanitized_path_str)
    sanitized_path_str = re.sub(r"//+", "/", sanitized_path_str)

    path_obj = Path(sanitized_path_str)

    # Resolve the path to get its absolute, canonical form
    try:
        resolved_path = path_obj.resolve()
    except Exception as e:
        raise ValueError(f"Failed to resolve path '{sanitized_path_str}': {e}") from e

    # Check if the resolved path is within the project root
    if not is_within_base_dir(resolved_path):
        raise ValueError(
            f"File path '{raw_path}' (sanitized: '{sanitized_path_str}') resolves to a location outside the allowed project directory."
        )

    # Return path relative to PROJECT_ROOT
    try:
        return str(resolved_path.relative_to(PROJECT_ROOT))
    except ValueError:
        # This should ideally not happen if is_within_base_dir returned True,
        # but as a safeguard, return the absolute path if relative_to fails.
        logger.warning(
            f"Could not get relative path for '{resolved_path}' from '{PROJECT_ROOT}'. Returning absolute path."
        )
        return str(resolved_path)


# NEW FUNCTION: _map_incorrect_file_path
def _map_incorrect_file_path(suggested_path: str) -> str:
    """Map common incorrect file paths to actual project structure paths."""
    path_mapping = {
        "core/agent.py": "core.py",
        "services/llm_service.py": "src/llm/client.py",
        "utils/validation.py": "src/utils/validation.py",
        "core/llm_cache.py": "src/llm_cache.py",
        "core/config.py": "config.py",
        "reasoning_engine.py": "core.py",  # Common LLM hallucination
        "token_manager.py": "src/token_tracker.py",  # Common LLM hallucination
        "routes.py": "app.py",  # Common LLM hallucination
        # --- NEW MAPPINGS ---
        "src/prompt_manager.py": "src/utils/prompt_optimizer.py",
        "src/main.py": "app.py",
        "src/llm_interface.py": "src/llm_provider.py",
        # --- END NEW MAPPINGS ---
    }

    # If the suggested path is in the mapping, return the correct path
    if suggested_path in path_mapping:
        return path_mapping[suggested_path]

    # Handle "core/" prefix cases (core.py is at root, not in core/ directory)
    if suggested_path.startswith("core/") and suggested_path != "core.py":
        # Only replace if it's not literally "core/core.py" or similar, but a file that should be at root
        if suggested_path.replace("core/", "", 1) in [
            "core.py",
            "config.py",
        ]:  # Add other root files if needed
            return suggested_path.replace("core/", "", 1)
        # Otherwise, assume it's a file that should be in src/
        return "src/" + suggested_path.replace("core/", "", 1)

    # Handle "services/" prefix (no services directory exists, usually maps to src/)
    if suggested_path.startswith("services/"):
        return "src/" + suggested_path.replace("services/", "", 1)

    # Handle "utils/" prefix (often maps to src/utils/)
    if suggested_path.startswith("utils/") and not suggested_path.startswith(
        "src/utils/"
    ):
        return "src/" + suggested_path

    return suggested_path


# NEW FUNCTION: can_create_file
def can_create_file(file_path: str) -> bool:
    """Check if a file can be created at the specified path."""
    # Check if the directory structure exists or can be created
    directory = os.path.dirname(file_path)
    if not directory:  # Root level file
        return True

    # Check if the directory exists or can be created
    if os.path.exists(directory):
        return True

    # Check if parent directories exist
    parent_dirs = []
    current = directory
    while current and current != "." and not os.path.exists(current):
        parent_dirs.append(current)
        current = os.path.dirname(current)

    # If all parent directories exist or can be created, it's okay
    # This means the path is valid if all its parent directories exist
    # or if it's a direct child of an an existing directory.
    # The `os.path.exists(d)` check is for the *parent* directories.
    # If `parent_dirs` is empty, it means `directory` itself exists.
    return len(parent_dirs) == 0 or all(os.path.exists(d) for d in parent_dirs)
