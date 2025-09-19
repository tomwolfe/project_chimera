# src/utils/code_validator.py
import io
from typing import List, Tuple, Dict, Any, Optional, Union
import subprocess
import sys
import os
import tempfile
import hashlib
import re
import contextlib
import logging
from pathlib import Path
import ast
import json
from collections import defaultdict

from src.utils.core_helpers.command_executor import execute_command_safely
from src.utils.core_helpers.path_utils import (
    is_within_base_dir,
    sanitize_and_validate_file_path,
    PROJECT_ROOT,
    _map_incorrect_file_path,
    can_create_file,
)
from src.utils.core_helpers.code_utils import _get_code_snippet

logger = logging.getLogger(__name__)


class CodeValidationError(Exception):
    """Custom exception for code validation errors."""

    pass


def validate_and_resolve_file_path_for_action(
    suggested_path: str, action: str, codebase_raw_file_contents: Dict[str, str]
) -> Tuple[bool, str, str, Optional[str]]:
    """
    Validates a suggested file path and action against the actual codebase context.
    Attempts to map incorrect paths to correct ones.
    Returns: (is_valid, resolved_path, suggested_action_if_changed, error_message)
    """
    # 1. Map common incorrect paths to canonical ones
    mapped_path = _map_incorrect_file_path(suggested_path)

    # 2. Sanitize and validate the path against project root
    try:
        # Use sanitize_and_validate_file_path to ensure it's within project boundaries
        resolved_path = sanitize_and_validate_file_path(mapped_path)
    except ValueError as e:
        return False, mapped_path, action, f"Path security validation failed: {e}"

    # 3. Check if the resolved path exists in the current codebase context
    file_exists_in_codebase = resolved_path in codebase_raw_file_contents

    # 4. Validate action against file existence and creation feasibility
    error_message = None
    suggested_action_if_changed = action  # Default to original action

    if action in ["ADD", "CREATE"]:
        if file_exists_in_codebase:
            error_message = f"Action '{action}' suggested for existing file '{resolved_path}'. Consider 'MODIFY'."
            suggested_action_if_changed = "MODIFY"  # Suggest changing action
            return True, resolved_path, suggested_action_if_changed, error_message
        if not can_create_file(resolved_path):
            error_message = f"Cannot create file at '{resolved_path}': Parent directory does not exist or is inaccessible."
            return False, resolved_path, action, error_message
        return True, resolved_path, action, None

    elif action == "CREATE_DIRECTORY":
        # For directory creation, we just need to ensure the parent path is valid and can be created
        if Path(resolved_path).exists():
            error_message = f"Directory '{resolved_path}' already exists. Skipping CREATE_DIRECTORY."
            return True, resolved_path, action, error_message  # Valid, but no-op
        if not can_create_file(
            resolved_path
        ):  # can_create_file works for directories too
            error_message = f"Cannot create directory at '{resolved_path}': Parent directory does not exist or is inaccessible."
            return False, resolved_path, action, error_message
        return True, resolved_path, action, None

    elif action == "MODIFY":
        if not file_exists_in_codebase:
            error_message = f"Action 'MODIFY' suggested for non-existent file '{resolved_path}'. Converting to 'CREATE'."
            suggested_action_if_changed = "CREATE"  # Suggest changing action
            if not can_create_file(resolved_path):
                error_message = f"Cannot create file at '{resolved_path}': Parent directory does not exist or is inaccessible."
                return False, resolved_path, action, error_message
            return True, resolved_path, suggested_action_if_changed, error_message
        return True, resolved_path, action, None

    elif action == "REMOVE":
        if not file_exists_in_codebase:
            error_message = f"Action 'REMOVE' suggested for non-existent file '{resolved_path}'. Suggestion ignored."
            return False, resolved_path, action, error_message
        return True, resolved_path, action, None

    else:
        error_message = f"Unknown action type '{action}'."
        return False, resolved_path, action, error_message


# REMOVED: _run_pycodestyle function as it's redundant with Ruff.


def _run_ruff(content: str, filename: str) -> List[Dict[str, Any]]:
    """Runs Ruff (linter and formatter check) on the given content via subprocess."""
    issues = []
    tmp_file_path = None
    content_lines = content.splitlines()
    try:
        with tempfile.NamedTemporaryFile(
            mode="w+", suffix=".py", encoding="utf-8", delete=False
        ) as temp_file:
            temp_file.write(content)
            temp_file.flush()
            tmp_file_path = Path(temp_file.name)

            # 1. Run Ruff Linter
            lint_command = [
                "ruff",
                "check",
                "--output-format=json",
                "--isolated",
                "--force-exclude",
                str(tmp_file_path),
            ]

            return_code_lint, stdout_lint, stderr_lint = execute_command_safely(
                lint_command, timeout=30, check=False
            )

            if stdout_lint:
                if return_code_lint == 0 or return_code_lint == 1:
                    try:
                        lint_results = json.loads(stdout_lint)
                        for issue in lint_results:
                            line_num = issue.get("location", {}).get("row")
                            issues.append(
                                {
                                    "type": "Ruff Linting Issue",
                                    "file": filename,
                                    "line": line_num,
                                    "column": issue.get("location", {}).get("column"),
                                    "code": issue.get("code"),
                                    "message": issue.get("message"),
                                    "source": "ruff_lint",
                                    "code_snippet": _get_code_snippet(
                                        content_lines, line_num, context_lines=3
                                    ),
                                }
                            )
                    except json.JSONDecodeError as jde:
                        logger.error(
                            f"Failed to parse Ruff lint JSON output for {filename}: {jde}. Output: {stdout_lint}",
                            exc_info=True,
                        )
                        issues.append(
                            {
                                "type": "Validation Tool Error",
                                "file": filename,
                                "message": f"Failed to parse Ruff lint output: {jde}",
                            }
                        )
                else:
                    logger.error(
                        f"Ruff lint execution failed for {filename} with return code {return_code_lint}. Stderr: {stderr_lint}"
                    )
                    issues.append(
                        {
                            "type": "Validation Tool Error",
                            "file": filename,
                            "message": f"Ruff lint command failed with unexpected exit code {return_code_lint}: {stderr_lint}",
                        }
                    )

            if stderr_lint:
                logger.warning(f"Ruff lint stderr for {filename}: {stderr_lint}")

            # 2. Run Ruff Formatter Check
            format_command = [
                "ruff",
                "format",
                "--check",
                "--isolated",
                "--force-exclude",
                str(tmp_file_path),
            ]

            return_code_format, stdout_format, stderr_format = execute_command_safely(
                format_command, timeout=30, check=False
            )

            if return_code_format != 0:
                issues.append(
                    {
                        "type": "Ruff Formatting Issue",
                        "file": filename,
                        "line": None,
                        "column": None,
                        "code": "FMT",
                        "message": "Code is not formatted according to Ruff standards. Run `ruff format` to fix.",
                        "source": "ruff_format",
                        "code_snippet": None,
                    }
                )

            if stderr_format:
                logger.warning(f"Ruff format stderr for {filename}: {stderr_format}")

    except (
        FileNotFoundError,
        subprocess.TimeoutExpired,
        subprocess.CalledProcessError,
    ) as e:
        logger.error(f"Ruff execution failed for {filename}: {e}", exc_info=True)
        issues.append(
            {
                "type": "Validation Tool Error",
                "file": filename,
                "message": f"Ruff execution failed: {e}",
            }
        )
    except Exception as e:
        logger.error(f"Unexpected error running Ruff on {filename}: {e}", exc_info=True)
        issues.append(
            {
                "type": "Validation Tool Error",
                "file": filename,
                "message": f"Failed to run Ruff: {e}",
            }
        )
    finally:
        if tmp_file_path and tmp_file_path.exists():
            try:
                os.unlink(tmp_file_path)
            except OSError as e:
                logger.warning(
                    f"Failed to delete temporary Ruff file {tmp_file_path}: {e}"
                )
    return issues


def _run_bandit(
    content: str,
    filename: str,
    severity_level: str = "medium",
    confidence_level: str = "medium",
) -> List[Dict[str, Any]]:
    """Runs Bandit security analysis on the given content via subprocess."""
    issues = []
    tmp_file_path = None
    content_lines = content.splitlines()
    try:
        with tempfile.NamedTemporaryFile(
            mode="w+", suffix=".py", encoding="utf-8", delete=False
        ) as temp_file:
            temp_file.write(content)
            temp_file.flush()
            tmp_file_path = Path(temp_file.name)

            bandit_config_path = PROJECT_ROOT / "pyproject.toml"
            if not bandit_config_path.exists():
                logger.warning(
                    f"Bandit config file not found at {bandit_config_path}. Running Bandit without explicit config."
                )
                # Use the provided lowercase severity_level and confidence_level
                config_args = [
                    "--severity-level",
                    severity_level,
                    "--confidence-level",
                    confidence_level,
                ]
            else:
                config_args = ["-c", str(bandit_config_path)]
                # Use the provided lowercase severity_level and confidence_level
                config_args.extend(
                    [
                        "--severity-level",
                        severity_level,
                        "--confidence-level",
                        confidence_level,
                    ]
                )

            command = ["bandit", "-q", "-f", "json", str(tmp_file_path)] + config_args

            return_code, stdout, stderr = execute_command_safely(
                command, timeout=30, check=False
            )

            if return_code not in (0, 1):
                logger.error(
                    f"Bandit execution failed for {filename} with return code {return_code}. Stderr: {stderr}"
                )
                issues.append(
                    {
                        "type": "Validation Tool Error",
                        "file": filename,
                        "message": f"Bandit failed with exit code {return_code}: {stderr}",
                    }
                )
            else:
                try:
                    bandit_results = stdout.strip()
                    if bandit_results:
                        data = json.loads(bandit_results)
                        for issue in data.get("results", []):
                            line_num = issue.get("line_number")
                            issues.append(
                                {
                                    "type": "Bandit Security Issue",
                                    "file": filename,
                                    "line": line_num,
                                    "code": issue.get("test_id"),
                                    "message": f"[{issue.get('severity')}] {issue.get('description')}",
                                    "source": "bandit",
                                    "code_snippet": _get_code_snippet(
                                        content_lines, line_num, context_lines=3
                                    ),
                                }
                            )
                    if not bandit_results and stderr:
                        logger.error(
                            f"Bandit produced no JSON output but had stderr: {stderr}"
                        )
                        issues.append(
                            {
                                "type": "Validation Tool Error",
                                "file": filename,
                                "message": f"Bandit produced no output but had stderr: {stderr}",
                            }
                        )
                except json.JSONDecodeError as jde:
                    logger.error(
                        f"Failed to parse Bandit JSON output for {filename}: {jde}. Output: {stdout}",
                        exc_info=True,
                    )
                    issues.append(
                        {
                            "type": "Validation Tool Error",
                            "file": filename,
                            "message": f"Failed to parse Bandit output: {jde}",
                        }
                    )
                except Exception as e:
                    logger.error(
                        f"Unexpected error processing Bandit output for {filename}: {e}",
                        exc_info=True,
                    )
                    issues.append(
                        {
                            "type": "Validation Tool Error",
                            "file": filename,
                            "message": f"Error processing Bandit output: {e}",
                        }
                    )

    except FileNotFoundError as e:
        logger.error(
            f"Bandit command not found. Ensure Bandit is installed and in the PATH: {e}"
        )
        issues.append(
            {
                "type": "Validation Tool Error",
                "file": filename,
                "message": f"Bandit executable not found: {e}. Please install Bandit.",
            }
        )
    except subprocess.TimeoutExpired as e:
        logger.error(f"Bandit execution timed out for {filename}.")
        issues.append(
            {
                "type": "Validation Tool Error",
                "file": filename,
                "message": f"Bandit execution timed out: {e}.",
            }
        )
    except subprocess.CalledProcessError as e:
        logger.error(
            f"Bandit execution failed with non-zero exit code: {e.returncode}. Stderr: {e.stderr.strip()}"
        )
        issues.append(
            {
                "type": "Validation Tool Error",
                "file": filename,
                "message": f"Bandit execution failed: {e.stderr.strip()}",
            }
        )
    except Exception as e:
        logger.error(
            f"Unexpected error running Bandit on {filename}: {e}", exc_info=True
        )
        issues.append(
            {
                "type": "Validation Tool Error",
                "file": filename,
                "message": f"Failed to run Bandit: {e}",
            }
        )
    finally:
        if tmp_file_path and tmp_file_path.exists():
            try:
                os.unlink(tmp_file_path)
            except OSError as e:
                logger.warning(
                    f"Failed to delete temporary Bandit file {tmp_file_path}: {e}"
                )
    return issues


def _run_ast_security_checks(content: str, filename: str) -> List[Dict[str, Any]]:
    """Runs AST-based security checks on Python code."""
    issues = []
    content_lines = content.splitlines()
    try:
        tree = ast.parse(content)

        class EnhancedSecurityPatternVisitor(ast.NodeVisitor):
            def __init__(self, filename, content_lines):
                self.filename = filename
                self.content_lines = content_lines
                self.issues = []
                self.imports = set()

            def visit_Import(self, node):
                for alias in node.names:
                    self.imports.add(alias.name)
                self.generic_visit(node)

            def visit_ImportFrom(self, node):
                if node.module:
                    self.imports.add(node.module)
                self.generic_visit(node)

            def visit_Call(self, node):
                snippet = _get_code_snippet(
                    self.content_lines, node.lineno, context_lines=3
                )

                # Check for eval() and exec()
                if isinstance(node.func, ast.Name):
                    if node.func.id == "eval":
                        self.issues.append(
                            {
                                "type": "Security Vulnerability (AST)",
                                "file": self.filename,
                                "line": node.lineno,
                                "message": "Use of eval() is discouraged due to security risks.",
                                "code_snippet": snippet,
                            }
                        )
                    elif node.func.id == "exec":
                        self.issues.append(
                            {
                                "type": "Security Vulnerability (AST)",
                                "file": self.filename,
                                "line": node.lineno,
                                "message": "Use of exec() is discouraged due to security risks.",
                                "code_snippet": snippet,
                            }
                        )

                # Check for subprocess.run with shell=True
                if (
                    isinstance(node.func, ast.Attribute)
                    and isinstance(node.func.value, ast.Name)
                    and node.func.value.id == "subprocess"
                    and node.func.attr == "run"
                ):
                    for keyword in node.keywords:
                        if (
                            keyword.arg == "shell"
                            and isinstance(keyword.value, ast.Constant)
                            and keyword.value.value is True
                        ):
                            self.issues.append(
                                {
                                    "type": "Security Vulnerability (AST)",
                                    "file": self.filename,
                                    "line": node.lineno,
                                    "message": "subprocess.run() with shell=True is dangerous; consider shell=False and passing arguments as a list.",
                                    "code_snippet": snippet,
                                }
                            )

                # Check for pickle.load
                if (
                    isinstance(node.func, ast.Attribute)
                    and isinstance(node.func.value, ast.Name)
                    and node.func.value.id == "pickle"
                    and node.func.attr == "load"
                ):
                    self.issues.append(
                        {
                            "type": "Security Vulnerability (AST)",
                            "file": self.filename,
                            "line": node.lineno,
                            "message": "Use of pickle.load() with untrusted data is dangerous; it can execute arbitrary code.",
                            "code_snippet": snippet,
                        }
                    )

                # Check for os.system
                if (
                    isinstance(node.func, ast.Attribute)
                    and isinstance(node.func.value, ast.Name)
                    and node.func.value.id == "os"
                    and node.func.attr == "system"
                ):
                    self.issues.append(
                        {
                            "type": "Security Vulnerability (AST)",
                            "file": self.filename,
                            "line": node.lineno,
                            "message": "Use of os.system() is discouraged; it can execute arbitrary commands and is prone to shell injection. Consider subprocess.run() with shell=False.",
                            "code_snippet": snippet,
                        }
                    )

                # Check for XML External Entity (XXE) vulnerability in ElementTree
                if (
                    isinstance(node.func, ast.Attribute)
                    and isinstance(node.func.value, ast.Name)
                    and node.func.value.id == "ET"
                    and node.func.attr == "fromstring"
                ):
                    has_parser_none = False
                    for keyword in node.keywords:
                        if (
                            keyword.arg == "parser"
                            and isinstance(keyword.value, ast.Constant)
                            and keyword.value.value is None
                        ):
                            has_parser_none = True
                            break
                    if has_parser_none:
                        self.issues.append(
                            {
                                "type": "Security Vulnerability (AST)",
                                "file": self.filename,
                                "line": node.lineno,
                                "message": "xml.etree.ElementTree.fromstring() with parser=None is vulnerable to XML External Entity (XXE) attacks. Use a safe parser or disable DTDs.",
                                "code_snippet": snippet,
                            }
                        )

                # Enhanced deserialization vulnerability detection
                if isinstance(node.func, ast.Attribute) and isinstance(
                    node.func.value, ast.Name
                ):
                    # Check for pickle.loads with untrusted data
                    if (
                        node.func.value.id == "pickle"
                        and node.func.attr == "loads"
                        and self._is_potentially_untrusted_input(node)
                    ):
                        self.issues.append(
                            {
                                "type": "Security Vulnerability (AST)",
                                "file": self.filename,
                                "line": node.lineno,
                                "message": "pickle.loads() with untrusted data is dangerous; it can execute arbitrary code.",
                                "code_snippet": snippet,
                            }
                        )

                    # Check for yaml.load with Loader parameter missing
                    if (
                        node.func.value.id == "yaml"
                        and node.func.attr == "load"
                        and not self._has_safe_loader_parameter(node)
                    ):
                        self.issues.append(
                            {
                                "type": "Security Vulnerability (AST)",
                                "file": self.filename,
                                "line": node.lineno,
                                "message": "yaml.load() without Loader parameter is unsafe. Use yaml.safe_load() or specify Loader=yaml.SafeLoader.",
                                "code_snippet": snippet,
                            }
                        )

                # Check for shell injection patterns in subprocess calls
                if (
                    isinstance(node.func, ast.Attribute)
                    and isinstance(node.func.value, ast.Name)
                    and node.func.value.id == "subprocess"
                    and node.func.attr in ["call", "check_call", "check_output", "run"]
                ):
                    shell_true_arg = False
                    for keyword in node.keywords:
                        if (
                            keyword.arg == "shell"
                            and isinstance(keyword.value, ast.Constant)
                            and keyword.value.value is True
                        ):
                            shell_true_arg = True
                            break

                    if shell_true_arg:
                        self.issues.append(
                            {
                                "type": "Security Vulnerability (AST)",
                                "file": self.filename,
                                "line": node.lineno,
                                "message": "subprocess.run() with shell=True is dangerous; consider shell=False and passing arguments as a list.",
                                "code_snippet": snippet,
                            }
                        )
                    else:
                        for arg in node.args:
                            if isinstance(arg, ast.Constant) and isinstance(
                                arg.value, str
                            ):
                                if any(
                                    char in arg.value
                                    for char in [
                                        ";",
                                        "|",
                                        "&",
                                        "$",
                                        "`",
                                        ">",
                                        "<",
                                        "(",
                                        ")",
                                        "#",
                                        "*",
                                    ]
                                ):
                                    self.issues.append(
                                        {
                                            "type": "Security Vulnerability (AST)",
                                            "file": self.filename,
                                            "line": node.lineno,
                                            "message": f"Potential shell injection in subprocess.{node.func.attr} with string argument containing shell metacharacters. Consider passing arguments as a list.",
                                            "code_snippet": snippet,
                                        }
                                    )

                # NEW: Check for weak cryptographic hashes (e.g., MD5)
                if (
                    isinstance(node.func, ast.Attribute)
                    and isinstance(node.func.value, ast.Name)
                    and node.func.value.id == "hashlib"
                    and node.func.attr == "md5"
                ):
                    self.issues.append(
                        {
                            "type": "Security Vulnerability (AST)",
                            "file": self.filename,
                            "line": node.lineno,
                            "message": "Use of weak MD5 hash for security is discouraged. Consider stronger algorithms like SHA256 or SHA3.",
                            "code_snippet": snippet,
                        }
                    )

                self.generic_visit(node)

            def _is_potentially_untrusted_input(self, node) -> bool:
                """Heuristic to check if function argument might be untrusted input."""
                if node.args and isinstance(node.args[0], ast.Name):
                    arg_name = node.args[0].id
                    untrusted_keywords = [
                        "input",
                        "user",
                        "request",
                        "param",
                        "data",
                        "body",
                        "query",
                        "json",
                        "raw",
                    ]
                    return any(
                        keyword in arg_name.lower() for keyword in untrusted_keywords
                    )
                return True

            def _has_safe_loader_parameter(self, node) -> bool:
                """Check if yaml.load call has a safe Loader parameter."""
                for keyword in node.keywords:
                    if keyword.arg == "Loader":
                        if isinstance(
                            keyword.value, ast.Attribute
                        ) and keyword.value.attr in ["SafeLoader", "CSafeLoader"]:
                            return True
                        if isinstance(keyword.value, ast.Name) and keyword.value.id in [
                            "SafeLoader",
                            "CSafeLoader",
                        ]:
                            return True
                return False

        visitor = EnhancedSecurityPatternVisitor(filename, content_lines)
        visitor.visit(tree)
        issues.extend(visitor.issues)
    except SyntaxError as se:
        issues.append(
            {
                "type": "Syntax Error",
                "file": filename,
                "line": se.lineno,
                "column": se.offset,
                "message": f"Invalid Python syntax: {se.msg}",
                "code_snippet": _get_code_snippet(
                    content_lines, se.lineno, context_lines=3
                ),
            }
        )
    except Exception as e:
        logger.error(f"Error during AST analysis for {filename}: {e}", exc_info=True)
        issues.append(
            {
                "type": "Validation Tool Error",
                "file": filename,
                "message": f"Failed during AST analysis: {e}",
            }
        )
    return issues


def validate_code_output(
    parsed_change: Dict[str, Any],
    original_content: str = None,
    file_analysis_cache: Optional[Dict[str, Dict[str, Any]]] = None,
    file_exists_in_codebase: bool = False,  # NEW: Pass explicit flag
) -> Dict[str, Any]:
    """Validates a single code change (ADD, MODIFY, REMOVE) for syntax, style, and security."""
    file_path_str = parsed_change.get("FILE_PATH")
    action = parsed_change.get("ACTION")
    content_to_check = ""
    issues = []

    if not file_path_str or not action:
        return {
            "issues": [
                {
                    "type": "Validation Error",
                    "file": file_path_str or "N/A",
                    "message": "Missing FILE_PATH or ACTION in parsed change.",
                }
            ]
        }

    file_path_obj = Path(file_path_str)
    is_python = file_path_obj.suffix.lower() == ".py"

    # Removed redundant file existence checks here.
    # These checks are now handled by `validate_and_resolve_file_path_for_action`
    # before this function is called. The `file_exists_in_codebase` flag is passed in.

    if action == "ADD" or action == "CREATE":
        content_to_check = parsed_change.get("FULL_CONTENT", "")
        checksum = hashlib.sha256(content_to_check.encode("utf-8")).hexdigest()
        issues.append(
            {
                "type": "Content Integrity",
                "file": file_path_str,
                "message": f"New file SHA256: {checksum}",
            }
        )
        if is_python:
            issues.extend(_run_ruff(content_to_check, file_path_str))
            issues.extend(_run_bandit(content_to_check, file_path_str))
            issues.extend(_run_ast_security_checks(content_to_check, file_path_str))
    elif action == "MODIFY":
        content_to_check = parsed_change.get("FULL_CONTENT", "")
        # The file_exists_in_codebase check above should have caught if it's a non-existent file.
        # Now, proceed with content validation if content is provided.
        checksum_new = hashlib.sha256(
            content_to_check.encode("utf-8")
        ).hexdigest()  # Recalculate checksum for new content
        issues.append(
            {
                "type": "Content Integrity",
                "file": file_path_str,
                "message": f"Modified file (new content) SHA256: {checksum_new}",
            }
        )

        if original_content is not None:
            original_checksum = hashlib.sha256(
                original_content.encode("utf-8")
            ).hexdigest()
            if checksum_new == original_checksum:
                issues.append(
                    {
                        "type": "No Change Detected",
                        "file": file_path_str,
                        "message": "New content is identical to original.",
                    }
                )
            # If original_content is provided, it means the file existed.
            # We can also add pre-computed issues from cache if available.
            if file_analysis_cache and file_path_str in file_analysis_cache:
                cached_analysis = file_analysis_cache[file_path_str]
                if "ruff_issues" in cached_analysis:
                    issues.extend(cached_analysis["ruff_issues"])
                if "bandit_issues" in cached_analysis:
                    issues.extend(cached_analysis["bandit_issues"])
                if "ast_security_issues" in cached_analysis:
                    issues.extend(cached_analysis["ast_security_issues"])
                logger.debug(
                    f"Added pre-computed issues for original content of {file_path_str} from cache."
                )

            if is_python:
                issues.extend(_run_ruff(content_to_check, file_path_str))
                issues.extend(_run_bandit(content_to_check, file_path_str))
                issues.extend(_run_ast_security_checks(content_to_check, file_path_str))
        else:  # If original_content is None, it means the file didn't exist or wasn't provided.
            # The file_exists_in_codebase check above should have caught this.
            if is_python:
                issues.extend(_run_ruff(content_to_check, file_path_str))
                issues.extend(_run_bandit(content_to_check, file_path_str))
                issues.extend(_run_ast_security_checks(content_to_check, file_path_str))
    elif action == "REMOVE":
        if original_content is not None:
            original_lines = original_content.splitlines()
            lines_to_remove = parsed_change.get("LINES", [])
            original_lines_set = set(original_lines)

            for line_content_to_remove in lines_to_remove:
                if line_content_to_remove not in original_lines_set:
                    issues.append(
                        {
                            "type": "POTENTIAL_REMOVAL_MISMATCH",
                            "file": file_path_str,
                            "message": f"Line intended for removal not found exactly in original content: '{line_content_to_remove[:80]}'",
                        }
                    )
        else:
            issues.append(
                {
                    "type": "VALIDATION_WARNING",
                    "file": file_path_str,
                    "message": "Original content not provided for REMOVE action validation.",
                }
            )
        return {"issues": issues}
    elif action == "CREATE_DIRECTORY":  # No content to check for directory creation
        pass
    else:
        issues.append(
            {
                "type": "UNKNOWN_ACTION",
                "file": file_path_str,
                "message": f"Unknown action type '{action}'.",
            }
        )

    return {"issues": issues}


def validate_code_output_batch(
    parsed_data: Dict,
    original_contents: Optional[Dict[str, str]] = None,
    file_analysis_cache: Optional[Dict[str, Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """Validates a batch of code changes and aggregates issues per file."""
    if original_contents is None:
        original_contents = {}
    all_validation_results = {}

    if not isinstance(parsed_data, dict):
        logger.error(
            f"validate_code_output_batch received non-dictionary parsed_data: {type(parsed_data).__name__}"
        )
        malformed_blocks_content = []
        if isinstance(parsed_data, str):
            malformed_blocks_content.append(
                f"Raw output that failed type check: {parsed_data[:500]}..."
            )
        elif parsed_data is not None:
            malformed_blocks_content.append(
                f"Unexpected type for parsed_data: {type(parsed_data).__name__}"
            )

        return {
            "issues": [
                {
                    "type": "Internal Error",
                    "file": "N/A",
                    "message": f"Invalid input type for parsed_data: Expected dict, got {type(parsed_data).__name__}",
                }
            ],
            "malformed_blocks": parsed_data.get("malformed_blocks", []),
        }

    code_changes_list = parsed_data.get("CODE_CHANGES", [])
    if not isinstance(code_changes_list, list):
        logger.error(
            f"validate_code_output_batch received non-list 'CODE_CHANGES' field: {type(code_changes_list).__name__}"
        )
        return {
            "issues": [
                {
                    "type": "Internal Error",
                    "file": "N/A",
                    "message": f"Invalid type for 'CODE_CHANGES': Expected list, got {type(code_changes_list).__name__}",
                }
            ],
            "malformed_blocks": parsed_data.get("malformed_blocks", []),
        }

    # NEW: Process code changes to validate/resolve paths and actions upfront
    processed_code_changes_for_validation = []
    for i, change_entry in enumerate(code_changes_list):
        if not isinstance(change_entry, dict):
            issue_message = f"Code change entry at index {i} is not a dictionary. Type: {type(change_entry).__name__}, Value: {str(change_entry)[:100]}"
            logger.error(issue_message)
            all_validation_results.setdefault("N/A", []).append(
                {
                    "type": "Malformed Change Entry",
                    "file": "N/A",
                    "message": issue_message,
                }
            )
            continue

        suggested_file_path = change_entry.get("FILE_PATH")
        action = change_entry.get("ACTION")

        if not suggested_file_path or not action:
            logger.warning(
                f"Encountered a code change without a 'FILE_PATH' or 'ACTION' in output {i}. Skipping validation for this item."
            )
            all_validation_results.setdefault("N/A", []).append(
                {
                    "type": "VALIDATION_ERROR",
                    "file": "N/A",
                    "message": f"Change item at index {i} missing FILE_PATH or ACTION.",
                }
            )
            continue

        is_valid, resolved_path, suggested_action, error_msg = (
            validate_and_resolve_file_path_for_action(
                suggested_file_path, action, original_contents
            )
        )

        if not is_valid:
            all_validation_results.setdefault(suggested_file_path, []).append(
                {
                    "type": "INVALID_FILE_PATH",
                    "file": suggested_file_path,
                    "message": error_msg,
                }
            )
            continue  # Skip further validation for this invalid entry

        # Update the change_entry with resolved path and potentially changed action
        change_entry["FILE_PATH"] = resolved_path
        change_entry["ACTION"] = suggested_action

        # Now, perform content validation for the (potentially modified) change_entry
        try:
            original_content_for_file = original_contents.get(resolved_path)
            file_exists_in_codebase = (
                resolved_path in original_contents
            )  # Check existence based on resolved path

            validation_result = validate_code_output(
                change_entry,
                original_content_for_file,
                file_analysis_cache,
                file_exists_in_codebase,
            )
            all_validation_results.setdefault(resolved_path, []).extend(
                validation_result.get("issues", [])
            )
            logger.debug(
                f"Validation for {resolved_path} completed with {len(validation_result.get('issues', []))} issues."
            )
        except Exception as e:
            logger.error(
                f"Error during content validation of change entry {i} for file {resolved_path}: {e}"
            )
            all_validation_results.setdefault(resolved_path, []).append(
                {
                    "type": "VALIDATION_TOOL_ERROR",
                    "file": resolved_path,
                    "message": f"Failed to validate content: {e}",
                }
            )

    # --- Unit Test Presence Check ---
    python_files_modified_or_added = {
        change["FILE_PATH"]
        for change in code_changes_list
        if change.get("FILE_PATH", "").endswith(".py")
        and change.get("ACTION")
        in ["ADD", "MODIFY", "CREATE"]  # Include CREATE as it's an ADD
    }
    test_files_added = {
        change["FILE_PATH"]
        for change in code_changes_list
        if change.get("FILE_PATH", "").startswith("tests/")
        and change.get("ACTION") == "ADD"
    }

    for py_file in python_files_modified_or_added:
        expected_test_file_prefix = f"tests/test_{Path(py_file).stem}"
        if not any(
            test_file.startswith(expected_test_file_prefix)
            for test_file in test_files_added
        ):
            all_validation_results.setdefault(py_file, []).append(
                {
                    "type": "Missing Unit Test",
                    "file": py_file,
                    "message": f"No corresponding unit test file found for this Python change. Expected a file like '{expected_test_file_prefix}.py' in 'tests/'.",
                }
            )

    logger.info(
        f"Batch validation completed. Aggregated issues for {len(all_validation_results)} files."
    )
    return all_validation_results
