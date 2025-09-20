# src/utils/path_utils.py
import logging
import os
import re
from pathlib import Path

# Determine PROJECT_ROOT deterministically based on the location of this file.
# This is more robust than searching for marker files.
# This file is in src/utils/core_helpers, so the root is 3 levels up.
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent

logger = logging.getLogger(__name__)
logger.info(f"Project root deterministically set to: {PROJECT_ROOT}")


def is_within_base_dir(file_path: Path) -> bool:
    """Checks if a file path is safely within the project base directory."""
    try:
        resolved_path = file_path.resolve()
        resolved_path.relative_to(PROJECT_ROOT)
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

    sanitized_path_str = re.sub(r'[<>:"\\|?*\x00-\x1f\x7f]', "", raw_path)
    sanitized_path_str = re.sub(r"\.\./", "", sanitized_path_str)
    sanitized_path_str = re.sub(r"//+", "/", sanitized_path_str)

    path_obj = Path(sanitized_path_str)

    try:
        resolved_path = path_obj.resolve()
    except Exception as e:
        raise ValueError(f"Failed to resolve path '{sanitized_path_str}': {e}") from e

    if not is_within_base_dir(resolved_path):
        raise ValueError(
            f"File path '{raw_path}' (sanitized: '{sanitized_path_str}') resolves to a location outside the allowed project directory."
        )

    try:
        return str(resolved_path.relative_to(PROJECT_ROOT))
    except ValueError:
        logger.warning(
            f"Could not get relative path for '{resolved_path}' from '{PROJECT_ROOT}'. Returning absolute path."
        )
        return str(resolved_path)


def _map_incorrect_file_path(suggested_path: str) -> str:
    """Map common incorrect file paths to actual project structure paths."""
    path_mapping = {
        "core/agent.py": "core.py",
        "services/llm_service.py": "src/llm/client.py",
        "utils/validation.py": "src/utils/validation.py",
        "core/llm_cache.py": "src/llm_cache.py",
        "core/config.py": "config.py",
        "reasoning_engine.py": "core.py",
        "token_manager.py": "src/token_tracker.py",
        "routes.py": "app.py",
        # --- NEW MAPPINGS ---
        "src/main.py": "app.py",
        "src/llm_service.py": "src/llm_provider.py",
        "src/api_client.py": "src/llm_provider.py",
        "src/prompt_manager.py": "src/utils/prompting/prompt_optimizer.py",
        "src/prompt_engineering/prompt_manager.py": "src/utils/prompting/prompt_optimizer.py",
        "src/llm_interface.py": "src/llm_provider.py",
        "src/analyzers/self_improvement_analyst.py": "src/self_improvement/metrics_collector.py",  # Or link to persona file
        "src/core/reasoning_engine.py": "core.py",  # Common hallucination
        "project_chimera/utils/json_utils.py": "src/utils/core_helpers/json_utils.py",  # Common hallucination
        "project_chimera/utils/prompt_generator.py": "src/utils/prompting/prompt_engineering.py",  # Common hallucination
        # --- END NEW MAPPINGS ---
    }
    path_mapping["src/app.py"] = "app.py"  # Add specific mapping for this common error

    if suggested_path in path_mapping:
        return path_mapping[suggested_path]

    if suggested_path.startswith("core/") and suggested_path != "core.py":
        if suggested_path.replace("core/", "", 1) in ["core.py", "config.py"]:
            return suggested_path.replace("core/", "", 1)
        return "src/" + suggested_path.replace("core/", "", 1)

    if suggested_path.startswith("services/"):
        return "src/" + suggested_path.replace("services/", "", 1)

    if suggested_path.startswith("utils/") and not suggested_path.startswith(
        "src/utils/"
    ):
        return "src/" + suggested_path

    return suggested_path


def can_create_file(file_path: str) -> bool:
    """Check if a file can be created at the specified path, including write permissions."""
    directory = os.path.dirname(file_path)
    if not directory:
        return os.access(PROJECT_ROOT, os.W_OK)

    if os.path.exists(directory):
        return os.access(directory, os.W_OK)

    parent_dirs = []
    current = directory
    while current and current != "." and not os.path.exists(current):
        parent_dirs.append(current)
        current = os.path.dirname(current)

    if not os.path.exists(current) or not os.access(current, os.W_OK):
        logger.debug(
            f"Cannot create file: Parent directory '{current}' is not writable or does not exist."
        )
        return False

    return True
