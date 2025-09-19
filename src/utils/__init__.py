# src/utils/__init__.py
# This file imports key functionalities from the refactored sub-packages
# to maintain a consistent import interface for the top-level 'utils' package.

from .prompting import PromptAnalyzer, PromptOptimizer, format_prompt
from .validation import (
    validate_gemini_api_key_format,
    test_gemini_api_key_functional,
    validate_code_output_batch,
    validate_and_resolve_file_path_for_action,
    can_create_file,
)
from .file_io import _create_file_backup, _apply_code_change, _apply_unified_diff
from .reporting import (
    OutputFormatter,
    LLMOutputParser,
    generate_markdown_report,
    strip_ansi_codes,
)
from .session import (
    _initialize_session_state,
    update_activity_timestamp,
    reset_app_state,
    check_session_expiration,
    SESSION_TIMEOUT_SECONDS,
    on_api_key_change,
    display_key_status,
    test_api_key,
    shutdown_streamlit,
)
from .core_helpers import (
    convert_to_json_friendly,
    sanitize_and_validate_file_path,
    PROJECT_ROOT,
    _map_incorrect_file_path,
    is_within_base_dir,
    execute_command_safely,
    _get_code_snippet,
    ComplexityVisitor,
    recommend_domain_from_keywords,
    handle_errors,
)

__all__ = [
    "PromptAnalyzer",
    "PromptOptimizer",
    "format_prompt",
    "validate_gemini_api_key_format",
    "test_gemini_api_key_functional",
    "validate_code_output_batch",
    "validate_and_resolve_file_path_for_action",
    "can_create_file",
    "_create_file_backup",
    "_apply_code_change",
    "_apply_unified_diff",
    "OutputFormatter",
    "LLMOutputParser",
    "generate_markdown_report",
    "strip_ansi_codes",
    "_initialize_session_state",
    "update_activity_timestamp",
    "reset_app_state",
    "check_session_expiration",
    "SESSION_TIMEOUT_SECONDS",
    "on_api_key_change",
    "display_key_status",
    "test_api_key",
    "shutdown_streamlit",
    "convert_to_json_friendly",
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
