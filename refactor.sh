#!/bin/bash

echo "Starting src/utils directory refactor..."

# Define base directory
BASE_DIR="src/utils"

# 1. Create new sub-directories
echo "Creating new sub-directories..."
mkdir -p "$BASE_DIR/prompting"
mkdir -p "$BASE_DIR/validation"
mkdir -p "$BASE_DIR/file_io"
mkdir -p "$BASE_DIR/reporting"
mkdir -p "$BASE_DIR/session"
mkdir -p "$BASE_DIR/core_helpers"

# 2. Move existing files into new sub-directories
echo "Moving existing utility files..."

# Prompting utilities
mv "$BASE_DIR/prompt_analyzer.py" "$BASE_DIR/prompting/"
mv "$BASE_DIR/prompt_engineering.py" "$BASE_DIR/prompting/"
mv "$BASE_DIR/prompt_optimizer.py" "$BASE_DIR/prompting/"

# Validation utilities
mv "$BASE_DIR/api_key_validator.py" "$BASE_DIR/validation/"
mv "$BASE_DIR/code_validator.py" "$BASE_DIR/validation/"

# File I/O utilities
mv "$BASE_DIR/file_operations.py" "$BASE_DIR/file_io/"

# Reporting utilities
mv "$BASE_DIR/output_formatter.py" "$BASE_DIR/reporting/"
mv "$BASE_DIR/output_parser.py" "$BASE_DIR/reporting/"
mv "$BASE_DIR/report_generator.py" "$BASE_DIR/reporting/"

# Session and UI helpers
mv "$BASE_DIR/session_manager.py" "$BASE_DIR/session/"
mv "$BASE_DIR/ui_helpers.py" "$BASE_DIR/session/"

# Core helpers (general, foundational utilities)
mv "$BASE_DIR/json_utils.py" "$BASE_DIR/core_helpers/"
mv "$BASE_DIR/path_utils.py" "$BASE_DIR/core_helpers/"
mv "$BASE_DIR/command_executor.py" "$BASE_DIR/core_helpers/"
mv "$BASE_DIR/code_utils.py" "$BASE_DIR/core_helpers/"
mv "$BASE_DIR/domain_recommender.py" "$BASE_DIR/core_helpers/"
mv "$BASE_DIR/error_handler.py" "$BASE_DIR/core_helpers/"

# 3. Remove redundant src/utils/constants.py (its content is in src/constants.py)
echo "Removing redundant src/utils/constants.py..."
rm -f "$BASE_DIR/constants.py"

# 4. Create __init__.py files for new sub-directories and the main utils package
echo "Creating __init__.py files..."

# src/utils/prompting/__init__.py
cat <<EOF > "$BASE_DIR/prompting/__init__.py"
from .prompt_analyzer import PromptAnalyzer
from .prompt_engineering import format_prompt
from .prompt_optimizer import PromptOptimizer

__all__ = ["PromptAnalyzer", "PromptOptimizer", "format_prompt"]
EOF

# src/utils/validation/__init__.py
cat <<EOF > "$BASE_DIR/validation/__init__.py"
from .api_key_validator import validate_gemini_api_key_format, test_gemini_api_key_functional
from .code_validator import validate_code_output_batch, validate_and_resolve_file_path_for_action, can_create_file

__all__ = ["validate_gemini_api_key_format", "test_gemini_api_key_functional", "validate_code_output_batch", "validate_and_resolve_file_path_for_action", "can_create_file"]
EOF

# src/utils/file_io/__init__.py
cat <<EOF > "$BASE_DIR/file_io/__init__.py"
from .file_operations import _create_file_backup, _apply_code_change, _apply_unified_diff

__all__ = ["_create_file_backup", "_apply_code_change", "_apply_unified_diff"]
EOF

# src/utils/reporting/__init__.py
cat <<EOF > "$BASE_DIR/reporting/__init__.py"
from .output_formatter import OutputFormatter
from .output_parser import LLMOutputParser
from .report_generator import generate_markdown_report, strip_ansi_codes

__all__ = ["OutputFormatter", "LLMOutputParser", "generate_markdown_report", "strip_ansi_codes"]
EOF

# src/utils/session/__init__.py
cat <<EOF > "$BASE_DIR/session/__init__.py"
from .session_manager import _initialize_session_state, update_activity_timestamp, reset_app_state, check_session_expiration, SESSION_TIMEOUT_SECONDS
from .ui_helpers import on_api_key_change, display_key_status, test_api_key, shutdown_streamlit

__all__ = ["_initialize_session_state", "update_activity_timestamp", "reset_app_state", "check_session_expiration", "SESSION_TIMEOUT_SECONDS", "on_api_key_change", "display_key_status", "test_api_key", "shutdown_streamlit"]
EOF

# src/utils/core_helpers/__init__.py
cat <<EOF > "$BASE_DIR/core_helpers/__init__.py"
from .json_utils import convert_to_json_friendly
from .path_utils import sanitize_and_validate_file_path, PROJECT_ROOT, _map_incorrect_file_path, is_within_base_dir
from .command_executor import execute_command_safely
from .code_utils import _get_code_snippet, ComplexityVisitor
from .domain_recommender import recommend_domain_from_keywords
from .error_handler import handle_errors

__all__ = ["convert_to_json_friendly", "sanitize_and_validate_file_path", "PROJECT_ROOT", "_map_incorrect_file_path", "is_within_base_dir", "execute_command_safely", "_get_code_snippet", "ComplexityVisitor", "recommend_domain_from_keywords", "handle_errors"]
EOF

# src/utils/__init__.py (main utils package __init__)
cat <<EOF > "$BASE_DIR/__init__.py"
# src/utils/__init__.py
# This file imports key functionalities from the refactored sub-packages
# to maintain a consistent import interface for the top-level 'utils' package.

from .prompting import PromptAnalyzer, PromptOptimizer, format_prompt
from .validation import validate_gemini_api_key_format, test_gemini_api_key_functional, validate_code_output_batch, validate_and_resolve_file_path_for_action, can_create_file
from .file_io import _create_file_backup, _apply_code_change, _apply_unified_diff
from .reporting import OutputFormatter, LLMOutputParser, generate_markdown_report, strip_ansi_codes
from .session import _initialize_session_state, update_activity_timestamp, reset_app_state, check_session_expiration, SESSION_TIMEOUT_SECONDS, on_api_key_change, display_key_status, test_api_key, shutdown_streamlit
from .core_helpers import convert_to_json_friendly, sanitize_and_validate_file_path, PROJECT_ROOT, _map_incorrect_file_path, is_within_base_dir, execute_command_safely, _get_code_snippet, ComplexityVisitor, recommend_domain_from_keywords, handle_errors

__all__ = [
    "PromptAnalyzer", "PromptOptimizer", "format_prompt",
    "validate_gemini_api_key_format", "test_gemini_api_key_functional",
    "validate_code_output_batch", "validate_and_resolve_file_path_for_action", "can_create_file",
    "_create_file_backup", "_apply_code_change", "_apply_unified_diff",
    "OutputFormatter", "LLMOutputParser", "generate_markdown_report", "strip_ansi_codes",
    "_initialize_session_state", "update_activity_timestamp", "reset_app_state", "check_session_expiration", "SESSION_TIMEOUT_SECONDS",
    "on_api_key_change", "display_key_status", "test_api_key", "shutdown_streamlit",
    "convert_to_json_friendly", "sanitize_and_validate_file_path", "PROJECT_ROOT", "_map_incorrect_file_path", "is_within_base_dir",
    "execute_command_safely", "_get_code_snippet", "ComplexityVisitor",
    "recommend_domain_from_keywords", "handle_errors"
]
EOF

echo "Refactor complete. Remember to manually update all import statements in other files!"
