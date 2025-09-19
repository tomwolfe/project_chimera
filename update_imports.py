# update_imports.py
import os
import re
from pathlib import Path
from collections import defaultdict

# Define the project root (assuming the script is run from the project root)
PROJECT_ROOT = Path(os.getcwd())

# Define mappings from old module names (without src.utils.) to their new full import paths
# This is for `from src.utils.OLD_MODULE import ...` and `import src.utils.OLD_MODULE`
MODULE_NAME_TO_NEW_PATH = {
    "prompt_analyzer": "src.utils.prompting.prompt_analyzer",
    "prompt_engineering": "src.utils.prompting.prompt_engineering",
    "prompt_optimizer": "src.utils.prompting.prompt_optimizer",
    "api_key_validator": "src.utils.validation.api_key_validator",
    "code_validator": "src.utils.validation.code_validator",
    "file_operations": "src.utils.file_io.file_operations",
    "output_formatter": "src.utils.reporting.output_formatter",
    "output_parser": "src.utils.reporting.output_parser",
    "report_generator": "src.utils.reporting.report_generator",
    "session_manager": "src.utils.session.session_manager",
    "ui_helpers": "src.utils.session.ui_helpers",
    "json_utils": "src.utils.core_helpers.json_utils",
    "path_utils": "src.utils.core_helpers.path_utils",
    "command_executor": "src.utils.core_helpers.command_executor",
    "code_utils": "src.utils.core_helpers.code_utils",
    "domain_recommender": "src.utils.core_helpers.domain_recommender",
    "error_handler": "src.utils.core_helpers.error_handler",
    # Special case for constants.py, which moved from src/utils/ to src/
    "constants": "src.constants",
}

# Mapping from individual symbols to their new full module import paths
# This is for `from src.utils import SYMBOL` cases.
SYMBOL_TO_NEW_MODULE_PATH = {
    "PromptAnalyzer": "src.utils.prompting.prompt_analyzer",
    "PromptOptimizer": "src.utils.prompting.prompt_optimizer",
    "format_prompt": "src.utils.prompting.prompt_engineering",
    "validate_gemini_api_key_format": "src.utils.validation.api_key_validator",
    "test_gemini_api_key_functional": "src.utils.validation.api_key_validator",
    "validate_code_output_batch": "src.utils.validation.code_validator",
    "validate_and_resolve_file_path_for_action": "src.utils.validation.code_validator",
    "can_create_file": "src.utils.validation.code_validator",
    "_create_file_backup": "src.utils.file_io.file_operations",
    "_apply_code_change": "src.utils.file_io.file_operations",
    "_apply_unified_diff": "src.utils.file_io.file_operations",
    "OutputFormatter": "src.utils.reporting.output_formatter",
    "LLMOutputParser": "src.utils.reporting.output_parser",
    "generate_markdown_report": "src.utils.reporting.report_generator",
    "strip_ansi_codes": "src.utils.reporting.report_generator",
    "_initialize_session_state": "src.utils.session.session_manager",
    "update_activity_timestamp": "src.utils.session.session_manager",
    "reset_app_state": "src.utils.session.session_manager",
    "check_session_expiration": "src.utils.session.session_manager",
    "SESSION_TIMEOUT_SECONDS": "src.utils.session.session_manager",
    "on_api_key_change": "src.utils.session.ui_helpers",
    "display_key_status": "src.utils.session.ui_helpers",
    "test_api_key": "src.utils.session.ui_helpers",
    "shutdown_streamlit": "src.utils.session.ui_helpers",
    "convert_to_json_friendly": "src.utils.core_helpers.json_utils",
    "sanitize_and_validate_file_path": "src.utils.core_helpers.path_utils",
    "PROJECT_ROOT": "src.utils.core_helpers.path_utils",
    "_map_incorrect_file_path": "src.utils.core_helpers.path_utils",
    "is_within_base_dir": "src.utils.core_helpers.path_utils",
    "execute_command_safely": "src.utils.core_helpers.command_executor",
    "_get_code_snippet": "src.utils.core_helpers.code_utils",
    "ComplexityVisitor": "src.utils.core_helpers.code_utils",
    "recommend_domain_from_keywords": "src.utils.core_helpers.domain_recommender",
    "handle_errors": "src.utils.core_helpers.error_handler",
    # For src/constants.py
    "SELF_ANALYSIS_KEYWORDS": "src.constants",
    "NEGATION_PATTERNS": "src.constants",
    "THRESHOLD": "src.constants",
    "SHARED_JSON_INSTRUCTIONS": "src.constants",
}


def update_imports_in_file(filepath: Path):
    """Reads a Python file, updates its import statements, and writes back."""
    original_content = filepath.read_text()
    new_content_lines = []
    changed = False

    for line in original_content.splitlines():
        # Regex to capture different import styles
        # Group 1: leading whitespace
        # Group 2: 'from' part (e.g., 'from src.utils.foo')
        # Group 3: 'import' part (e.g., 'import bar, baz as qux')
        # Group 4: 'import' part for `import src.utils.foo`

        # Pattern for `from src.utils.submodule import ...`
        match_from_sub = re.match(
            r"^(from\s+src\.utils\.)(?P<module_name>\w+)(\s+import\s+.*)$", line
        )
        if match_from_sub:
            old_module_name = match_from_sub.group("module_name")
            if old_module_name in MODULE_NAME_TO_NEW_PATH:
                new_full_path = MODULE_NAME_TO_NEW_PATH[old_module_name]
                new_line = f"from {new_full_path}{match_from_sub.group(3)}"
                new_content_lines.append(new_line)
                changed = True
                continue

        # Pattern for `import src.utils.submodule` or `import src.utils.submodule as alias`
        match_import_sub = re.match(
            r"^(import\s+src\.utils\.)(?P<module_name>\w+)(\s+as\s+\w+)?$", line
        )
        if match_import_sub:
            old_module_name = match_import_sub.group("module_name")
            if old_module_name in MODULE_NAME_TO_NEW_PATH:
                new_full_path = MODULE_NAME_TO_NEW_PATH[old_module_name]
                new_line = f"import {new_full_path}{match_import_sub.group(3) or ''}"
                new_content_lines.append(new_line)
                changed = True
                continue

        # Pattern for `from src.utils import SYMBOL1, SYMBOL2 as ALIAS, ...`
        match_from_utils_direct = re.match(
            r"^(from\s+src\.utils\s+import\s+)(?P<symbols>.*)$", line
        )
        if match_from_utils_direct:
            symbols_str = match_from_utils_direct.group("symbols")
            individual_symbols = [s.strip() for s in symbols_str.split(",")]

            new_imports_for_line = []
            for symbol_entry in individual_symbols:
                symbol_name = symbol_entry.split(" as ")[
                    0
                ].strip()  # Get the actual symbol name, ignoring alias
                if symbol_name in SYMBOL_TO_NEW_MODULE_PATH:
                    new_module_path = SYMBOL_TO_NEW_MODULE_PATH[symbol_name]
                    # Ensure we don't duplicate the 'import' keyword if it's already part of the symbol_entry
                    if " import " not in symbol_entry:
                        new_imports_for_line.append(
                            f"from {new_module_path} import {symbol_entry}"
                        )
                    else:  # Handle cases like `from src.utils import foo as bar`
                        new_imports_for_line.append(
                            f"from {new_module_path} {symbol_entry}"
                        )
                    changed = True
                else:
                    # If a symbol is not mapped, keep the original import for it, but this is a warning sign
                    print(
                        f"WARNING: Symbol '{symbol_name}' from 'src.utils' not found in mapping for {filepath}. Keeping original import for this symbol."
                    )
                    new_imports_for_line.append(f"from src.utils import {symbol_entry}")

            if new_imports_for_line:
                new_content_lines.extend(new_imports_for_line)
                continue

        # Pattern for `import src.utils` (less common, but possible)
        if line.strip() == "import src.utils":
            # This is a broad import. Given the refactor, it's better to explicitly import what's needed.
            # We'll replace it with a warning and keep the line, but it should be manually reviewed.
            print(
                f"WARNING: Found 'import src.utils' in {filepath}. This is a broad import. Consider replacing with specific imports from sub-packages."
            )
            new_content_lines.append(
                f"# TODO: Review 'import src.utils' in {filepath}. Consider specific imports from sub-packages."
            )
            new_content_lines.append(line)  # Keep original for now, but flag
            changed = True
            continue

        # Special handling for `src/constants.py` move
        # This handles `from src.utils.constants import ...` and `import src.utils.constants`
        match_constants_old_path = re.match(
            r"^(from|import)\s+src\.utils\.constants(\s+.*)?$", line
        )
        if match_constants_old_path:
            new_line = re.sub(r"src\.utils\.constants", "src.constants", line)
            new_content_lines.append(new_line)
            changed = True
            continue

        new_content_lines.append(line)

    if changed:
        filepath.write_text("\n".join(new_content_lines))
        print(f"Updated imports in {filepath}")
    else:
        print(f"No relevant import changes needed for {filepath}")


def main():
    print("Starting automated import updates...")
    python_files = list(PROJECT_ROOT.rglob("*.py"))

    # Exclude the new __init__.py files themselves from being modified by this script
    # as their content is specifically crafted for re-exports.
    exclude_patterns = [
        "src/utils/__init__.py",
        "src/utils/prompting/__init__.py",
        "src/utils/validation/__init__.py",
        "src/utils/file_io/__init__.py",
        "src/utils/reporting/__init__.py",
        "src/utils/session/__init__.py",
        "src/utils/core_helpers/__init__.py",
    ]

    files_to_process = []
    for f in python_files:
        relative_path = str(f.relative_to(PROJECT_ROOT))
        if not any(relative_path == p for p in exclude_patterns):
            files_to_process.append(f)

    for filepath in files_to_process:
        update_imports_in_file(filepath)

    print("Automated import updates complete. Please review all changes and run tests.")
    print(
        "Manual review is CRITICAL, especially for complex 'from src.utils import ...' statements."
    )


if __name__ == "__main__":
    main()
