from .code_utils import ComplexityVisitor, _get_code_snippet
from .command_executor import execute_command_safely
from .domain_recommender import recommend_domain_from_keywords
from .error_handler import handle_errors
from .json_utils import (
    convert_to_json_friendly,
    safe_json_dumps,
    safe_json_loads,
)  # MODIFIED: Added safe_json_loads, safe_json_dumps
from .path_utils import (
    PROJECT_ROOT,
    _map_incorrect_file_path,
    is_within_base_dir,
    sanitize_and_validate_file_path,
)

__all__ = [
    "convert_to_json_friendly",
    "safe_json_loads",  # NEW
    "safe_json_dumps",  # NEW
    "sanitize_and_validate_file_path",
    "PROJECT_ROOT",
    "_map_incorrect_file_path",
    "is_within_base_dir",
    "execute_command_safely",
    "_get_code_snippet",
    "ComplexityVisitor",
    "recommend_domain_from_keywords",
    "handle_errors",
]
