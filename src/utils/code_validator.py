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
import pycodestyle # Kept for potential fallback or specific checks if needed, but _run_ruff will be primary
import ast
import json # Added for Bandit output parsing
import yaml # Added for YAML security checks
from collections import defaultdict # Added for metrics aggregation
from src.utils.command_executor import execute_system_command # NEW IMPORT
from src.utils.path_utils import is_within_base_dir, sanitize_and_validate_file_path, PROJECT_ROOT # Import PROJECT_ROOT

logger = logging.getLogger(__name__)

class CodeValidationError(Exception):
    """Custom exception for code validation errors."""
    pass

# Helper to get a snippet around a line number
def _get_code_snippet(content_lines: List[str], line_number: Optional[int], context_lines: int = 2) -> Optional[str]:
    if line_number is None or not content_lines:
        return None
    
    # Adjust line_number to be 0-indexed for list access
    actual_line_idx = line_number - 1
    
    start_idx = max(0, actual_line_idx - context_lines)
    end_idx = min(len(content_lines), actual_line_idx + context_lines + 1) # +1 to include the end line

    snippet_lines = []
    for i in range(start_idx, end_idx):
        # Add 1 to i to display 1-indexed line numbers
        snippet_lines.append(f"{i + 1}: {content_lines[i].rstrip()}") # rstrip to remove trailing newlines
    return "\n".join(snippet_lines)

def _run_ruff(content: str, filename: str) -> List[Dict[str, Any]]:
    """Runs Ruff (linter and formatter check) on the given content via subprocess."""
    issues = []
    tmp_file_path = None
    content_lines = content.splitlines() # Split content into lines for snippet extraction
    try:
        with tempfile.NamedTemporaryFile(
            mode='w+', suffix='.py', encoding='utf-8', delete=False
        ) as temp_file:
            temp_file.write(content)
            temp_file.flush()
            tmp_file_path = Path(temp_file.name)

            # 1. Run Ruff Linter
            # --isolated: Ignores project-level configuration files.
            # --force-exclude: Excludes files even if they are explicitly passed as arguments.
            # We use these to ensure the temporary file is treated in isolation.
            # --output-format=json: Get structured output.
            lint_command = [
                sys.executable,
                "-m", "ruff", "check",
                "--output-format=json",
                "--isolated",
                "--force-exclude",
                str(tmp_file_path)
            ]
            
            stdout_lint, stderr_lint = execute_command_safely(
                lint_command,
                timeout=30,
                check=False # Ruff returns non-zero for linting issues, so check=False
            )

            if stdout_lint:
                try:
                    lint_results = json.loads(stdout_lint)
                    for issue in lint_results:
                        line_num = issue.get('location', {}).get('row')
                        issues.append({
                            'type': 'Ruff Linting Issue',
                            'file': filename,
                            'line': line_num,
                            'column': issue.get('location', {}).get('column'),
                            'code': issue.get('code'),
                            'message': issue.get('message'),
                            'source': 'ruff_lint',
                            'code_snippet': _get_code_snippet(content_lines, line_num) # ADDED
                        })
                except json.JSONDecodeError as jde:
                    logger.error(f"Failed to parse Ruff lint JSON output for {filename}: {jde}. Output: {stdout_lint}", exc_info=True)
                    issues.append({'type': 'Validation Tool Error', 'file': filename, 'message': f'Failed to parse Ruff lint output: {jde}'})
            
            if stderr_lint:
                logger.warning(f"Ruff lint stderr for {filename}: {stderr_lint}")
                # Optionally add stderr as an issue if it indicates a problem
                # issues.append({'type': 'Validation Tool Warning', 'file': filename, 'message': f'Ruff lint stderr: {stderr_lint}'})

            # 2. Run Ruff Formatter Check
            format_command = [
                sys.executable,
                "-m", "ruff", "format",
                "--check", # Only check, don't fix
                "--isolated",
                "--force-exclude",
                str(tmp_file_path)
            ]

            # Ruff format --check returns non-zero if formatting issues are found
            stdout_format, stderr_format = execute_command_safely(
                format_command,
                timeout=30,
                check=False # We handle the return code manually
            )
            
            # If stdout_format is not empty, it means there are formatting differences
            if stdout_format.strip():
                issues.append({
                    'type': 'Ruff Formatting Issue',
                    'file': filename,
                    'line': None,
                    'column': None,
                    'code': 'FMT',
                    'message': 'Code is not formatted according to Ruff standards. Run `ruff format` to fix.',
                    'source': 'ruff_format',
                    'code_snippet': None # No specific line for formatting issues
                })
            
            if stderr_format:
                logger.warning(f"Ruff format stderr for {filename}: {stderr_format}")
                # issues.append({'type': 'Validation Tool Warning', 'file': filename, 'message': f'Ruff format stderr: {stderr_format}'})

    except (FileNotFoundError, subprocess.TimeoutExpired, subprocess.CalledProcessError) as e:
        logger.error(f"Ruff execution failed for {filename}: {e}", exc_info=True)
        issues.append({'type': 'Validation Tool Error', 'file': filename, 'message': f'Ruff execution failed: {e}'})
    except Exception as e:
        logger.error(f"Unexpected error running Ruff on {filename}: {e}", exc_info=True)
        issues.append({'type': 'Validation Tool Error', 'file': filename, 'message': f'Failed to run Ruff: {e}'})
    finally:
        if tmp_file_path and tmp_file_path.exists():
            try:
                os.unlink(tmp_file_path)
            except OSError as e:
                logger.warning(f"Failed to delete temporary Ruff file {tmp_file_path}: {e}")
    return issues


def _run_bandit(content: str, filename: str) -> List[Dict[str, Any]]:
    """Runs Bandit security analysis on the given content via subprocess."""
    issues = []
    tmp_file_path = None # Initialize to None
    content_lines = content.splitlines() # Split content into lines for snippet extraction
    try:
        # FIX: Set delete=False for NamedTemporaryFile and add explicit cleanup
        with tempfile.NamedTemporaryFile(
            mode='w+', suffix='.py', encoding='utf-8', delete=False
        ) as temp_file:
            temp_file.write(content)
            temp_file.flush()
            tmp_file_path = Path(temp_file.name) # Store path for explicit unlink

            # MODIFICATION: Pass the project's pyproject.toml as the config file
            # Ensure PROJECT_ROOT is imported from src.utils.path_utils
            bandit_config_path = PROJECT_ROOT / "pyproject.toml"
            if not bandit_config_path.exists():
                logger.warning(f"Bandit config file not found at {bandit_config_path}. Running Bandit without explicit config.")
                config_args = []
            else:
                config_args = ["-c", str(bandit_config_path)]
            
            command = [
                sys.executable,
                "-m", "bandit",
                "-q",
                "-f", "json",
                str(tmp_file_path) # Use str(Path)
            ] + config_args # Add config arguments

            # Use the new execute_command_safely function
            stdout, stderr = execute_command_safely(
                command,
                timeout=30,
                check=False # Bandit returns 1 for issues, 0 for no issues, so check=True would fail.
                            # We handle return code manually below.
            )

            # The original subprocess.run call's 'process' object is not directly available here.
            # We need to infer success/failure from stdout/stderr and potential exceptions.
            # Bandit's JSON output is in stdout.
            try:
                bandit_results = stdout.strip()
                if bandit_results:
                    data = json.loads(bandit_results)
                    for issue in data.get('results', []):
                        line_num = issue.get('line_number')
                        # FIX: Removed the problematic 'if issue['level'] != 'info':'.
                        # Bandit's JSON output uses 'severity' and 'description', not 'level'.
                        issues.append({
                            'type': 'Bandit Security Issue',
                            'file': filename,
                            'line': line_num,
                            'code': issue.get('test_id'),
                            'message': f"[{issue.get('severity')}] {issue.get('description')}",
                            'code_snippet': _get_code_snippet(content_lines, line_num) # ADDED
                        })
                # If Bandit itself had an error (e.g., parsing its own config), it might print to stderr
                # and not produce valid JSON. We should check stderr for errors if stdout was empty or malformed.
                if not bandit_results and stderr:
                    logger.error(f"Bandit produced no JSON output but had stderr: {stderr}")
                    issues.append({'type': 'Validation Tool Error', 'file': filename, 'message': f'Bandit produced no output but had stderr: {stderr}'})
            except json.JSONDecodeError as jde:
                logger.error(f"Failed to parse Bandit JSON output for {filename}: {jde}. Output: {stdout}", exc_info=True)
                issues.append({'type': 'Validation Tool Error', 'file': filename, 'message': f'Failed to parse Bandit output: {jde}'})
            except Exception as e:
                logger.error(f"Unexpected error processing Bandit output for {filename}: {e}", exc_info=True)
                issues.append({'type': 'Validation Tool Error', 'file': filename, 'message': f'Error processing Bandit output: {e}'})

    except FileNotFoundError as e: # execute_command_safely can raise this
        logger.error("Bandit command not found. Ensure Bandit is installed and in the PATH.")
        issues.append({'type': 'Validation Tool Error', 'file': filename, 'message': f'Bandit executable not found: {e}. Please install Bandit.'})
    except subprocess.TimeoutExpired as e: # execute_command_safely can raise this
        logger.error(f"Bandit execution timed out for {filename}.")
        issues.append({'type': 'Validation Tool Error', 'file': filename, 'message': f'Bandit execution timed out: {e}.'})
    except subprocess.CalledProcessError as e: # execute_command_safely can raise this if check=True
        logger.error(f"Bandit execution failed for {filename} with exit code {e.returncode}. Stderr: {e.stderr.strip()}")
        issues.append({'type': 'Validation Tool Error', 'file': filename, 'message': f'Bandit failed: {e.stderr.strip()}'})
    except Exception as e: # Catch any other unexpected errors during the try block
        logger.error(f"Unexpected error running Bandit on {filename}: {e}", exc_info=True)
        issues.append({'type': 'Validation Tool Error', 'file': filename, 'message': f'Failed to run Bandit: {e}'})
    finally: # Ensure cleanup of the temporary file
        if tmp_file_path and tmp_file_path.exists():
            try:
                os.unlink(tmp_file_path)
            except OSError as e:
                logger.warning(f"Failed to delete temporary Bandit file {tmp_file_path}: {e}")

    return issues

def _run_ast_security_checks(content: str, filename: str) -> List[Dict[str, Any]]:
    """Runs AST-based security checks on Python code."""
    issues = []
    content_lines = content.splitlines() # Split content into lines for snippet extraction
    try:
        tree = ast.parse(content)

        class EnhancedSecurityPatternVisitor(ast.NodeVisitor):
            def __init__(self, filename, content_lines): # Pass content_lines
                self.filename = filename
                self.content_lines = content_lines # Store content_lines
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
                # Check for eval() and exec()
                snippet = _get_code_snippet(self.content_lines, node.lineno) # Get snippet once per node

                if isinstance(node.func, ast.Name):
                    if node.func.id == 'eval':
                        self.issues.append({
                            'type': 'Security Vulnerability (AST)',
                            'file': self.filename,
                            'line': node.lineno,
                            'message': "Use of eval() is discouraged due to security risks.",
                            'code_snippet': snippet # ADDED
                        }) 
                    elif node.func.id == 'exec':
                        self.issues.append({
                            'type': 'Security Vulnerability (AST)',
                            'file': self.filename,
                            'line': node.lineno,
                            'message': "Use of exec() is discouraged due to security risks.",
                            'code_snippet': snippet # ADDED
                        })

                # Check for subprocess.run with shell=True
                if isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name) and node.func.value.id == 'subprocess' and node.func.attr == 'run':
                    for keyword in node.keywords:
                        if keyword.arg == 'shell' and isinstance(keyword.value, ast.Constant) and keyword.value.value is True:
                            self.issues.append({
                                'type': 'Security Vulnerability (AST)',
                                'file': self.filename,
                                'line': node.lineno,
                                'message': "subprocess.run() with shell=True is dangerous; consider shell=False and passing arguments as a list.",
                                'code_snippet': snippet # ADDED
                            })
                
                # Check for pickle.load
                if isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name) and node.func.value.id == 'pickle' and node.func.attr == 'load':
                    self.issues.append({
                        'type': 'Security Vulnerability (AST)',
                        'file': self.filename,
                        'line': node.lineno,
                        'message': "Use of pickle.load() with untrusted data is dangerous; it can execute arbitrary code.",
                        'code_snippet': snippet # ADDED
                    })
                
                # Check for os.system
                if isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name) and node.func.value.id == 'os' and node.func.attr == 'system':
                    self.issues.append({
                        'type': 'Security Vulnerability (AST)',
                        'file': self.filename,
                        'line': node.lineno,
                        'message': "Use of os.system() is discouraged; it can execute arbitrary commands and is prone to shell injection. Consider subprocess.run() with shell=False.",
                        'code_snippet': snippet # ADDED
                    })
                
                # Check for XML External Entity (XXE) vulnerability in ElementTree
                if isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name) and node.func.value.id == 'ET' and node.func.attr == 'fromstring':
                     has_parser_none = False
                     for keyword in node.keywords:
                         if keyword.arg == 'parser' and isinstance(keyword.value, ast.Constant) and keyword.value.value is None:
                             has_parser_none = True
                             break
                     if has_parser_none:
                         self.issues.append({
                             'type': 'Security Vulnerability (AST)',
                             'file': self.filename,
                             'line': node.lineno, 
                             'message': "xml.etree.ElementTree.fromstring() with parser=None is vulnerable to XML External Entity (XXE) attacks. Use a safe parser or disable DTDs.",
                             'code_snippet': snippet # ADDED
                         })
                
                    # Enhanced deserialization vulnerability detection
                if (isinstance(node.func, ast.Attribute) and
                    isinstance(node.func.value, ast.Name)):

                    # Check for pickle.loads with untrusted data
                    if (node.func.value.id == 'pickle' and node.func.attr == 'loads' and
                        self._is_potentially_untrusted_input(node)):
                        self.issues.append({
                            'type': 'Security Vulnerability (AST)',
                            'file': self.filename,
                            'line': node.lineno,
                            'message': "pickle.loads() with potentially untrusted data can execute arbitrary code. Use a safe serialization format like JSON.",
                            'code_snippet': snippet # ADDED
                        })

                    # Check for yaml.load with Loader parameter missing
                    if (node.func.value.id == 'yaml' and node.func.attr == 'load' and
                        self._has_safe_loader_parameter(node)):
                        self.issues.append({
                            'type': 'Security Vulnerability (AST)',
                            'file': self.filename,
                            'line': node.lineno,
                            'message': "yaml.load() without Loader parameter is unsafe. Use yaml.safe_load() or specify Loader=yaml.SafeLoader.",
                            'code_snippet': snippet # ADDED
                        })

                # Check for shell injection patterns in subprocess calls
                if (isinstance(node.func, ast.Attribute) and
                    isinstance(node.func.value, ast.Name) and
                    node.func.value.id == 'subprocess' and
                    node.func.attr in ['call', 'check_call', 'check_output', 'run']):

                    shell_true_arg = False
                    for keyword in node.keywords:
                        if keyword.arg == 'shell' and isinstance(keyword.value, ast.Constant) and keyword.value.value is True:
                            shell_true_arg = True
                            break

                    if shell_true_arg:
                        self.issues.append({
                            'type': 'Security Vulnerability (AST)',
                            'file': self.filename,
                            'line': node.lineno,
                            'message': "subprocess.run() with shell=True is dangerous; consider shell=False and passing arguments as a list.",
                            'code_snippet': snippet # ADDED
                        })
                    else: # Check for shell metacharacters in string arguments if shell=True is not explicit
                        for arg in node.args:
                            if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
                                if any(char in arg.value for char in [';', '|', '&', '$', '`', '>', '<', '(', ')', '#', '*']):
                                    self.issues.append({
                                        'type': 'Security Vulnerability (AST)',
                                        'file': self.filename,
                                        'line': node.lineno, 
                                        'message': f"Potential shell injection in subprocess.{node.func.attr} with string argument containing shell metacharacters. Consider passing arguments as a list.",
                                        'code_snippet': snippet # ADDED
                                    })

                self.generic_visit(node)

            def _is_potentially_untrusted_input(self, node) -> bool: # No change needed here
                """Heuristic to check if function argument might be untrusted input."""
                if node.args and isinstance(node.args[0], ast.Name):
                    arg_name = node.args[0].id
                    untrusted_keywords = ['input', 'user', 'request', 'param', 'data', 'body', 'query', 'json', 'raw']
                    return any(keyword in arg_name.lower() for keyword in untrusted_keywords)
                return True # Assume untrusted if we can't determine

            def _has_safe_loader_parameter(self, node) -> bool: # No change needed here
                """Check if yaml.load call has a safe Loader parameter."""
                for keyword in node.keywords:
                    if keyword.arg == 'Loader':
                        if (isinstance(keyword.value, ast.Attribute) and
                            keyword.value.attr in ['SafeLoader', 'CSafeLoader']):
                            return True
                        if (isinstance(keyword.value, ast.Name) and
                            keyword.value.id in ['SafeLoader', 'CSafeLoader']):
                            return True
                return False

        visitor = EnhancedSecurityPatternVisitor(filename, content_lines) # Pass content_lines
        visitor.visit(tree)
        issues.extend(visitor.issues)
    except SyntaxError as se:
        issues.append({
            'type': 'Syntax Error',
            'file': filename,
            'line': se.lineno, 
            'column': se.offset,
            'message': f"Invalid Python syntax: {se.msg}",
            'code_snippet': _get_code_snippet(content_lines, se.lineno) # ADDED
        })
    except Exception as e:
        logger.error(f"Error during AST analysis for {filename}: {e}", exc_info=True)
        issues.append({'type': 'Validation Tool Error', 'file': filename, 'message': f'Failed during AST analysis: {e}'})
    return issues

def validate_code_output(parsed_change: Dict[str, Any], original_content: str = None) -> Dict[str, Any]:
    """Validates a single code change (ADD, MODIFY, REMOVE) for syntax, style, and security."""
    file_path_str = parsed_change.get('FILE_PATH')
    action = parsed_change.get('ACTION')
    content_to_check = ""
    issues = []

    if not file_path_str or not action:
        return {'issues': [{'type': 'Validation Error', 'file': file_path_str or 'N/A', 'message': 'Missing FILE_PATH or ACTION in parsed change.'}]}

    file_path = Path(file_path_str)
    is_python = file_path.suffix.lower() == '.py'

    if action == 'ADD':
        content_to_check = parsed_change.get('FULL_CONTENT', '')
        checksum = hashlib.sha256(content_to_check.encode('utf-8')).hexdigest()
        issues.append({'type': 'Content Integrity', 'file': file_path_str, 'message': f"New file SHA256: {checksum}"})
        if is_python:
            issues.extend(_run_ruff(content_to_check, file_path_str)) # Use Ruff
            issues.extend(_run_bandit(content_to_check, file_path_str))
            issues.extend(_run_ast_security_checks(content_to_check, file_path_str))

    elif action == 'MODIFY':
        content_to_check = parsed_change.get('FULL_CONTENT', '')
        checksum_new = hashlib.sha256(content_to_check.encode('utf-8')).hexdigest()
        issues.append({'type': 'Content Integrity', 'file': file_path_str, 'message': f"Modified file (new content) SHA256: {checksum_new}"})

        if original_content is not None:
            original_checksum = hashlib.sha256(original_content.encode('utf-8')).hexdigest()
            if checksum_new == original_checksum:
                issues.append({'type': 'No Change Detected', 'file': file_path_str, 'message': 'New content is identical to original.'})
            if is_python:
                issues.extend(_run_ruff(content_to_check, file_path_str)) # Use Ruff
                issues.extend(_run_bandit(content_to_check, file_path_str))
                issues.extend(_run_ast_security_checks(content_to_check, file_path_str))
        else:
            if is_python:
                issues.extend(_run_ruff(content_to_check, file_path_str)) # Use Ruff
                issues.extend(_run_bandit(content_to_check, file_path_str))
                issues.extend(_run_ast_security_checks(content_to_check, file_path_str))

    elif action == 'REMOVE':
        if original_content is not None:
            original_lines = original_content.splitlines()
            lines_to_remove = parsed_change.get('LINES', [])
            original_lines_set = set(original_lines)

            for line_content_to_remove in lines_to_remove:
                if line_content_to_remove not in original_lines_set:
                    issues.append({
                        'type': 'Potential Removal Mismatch',
                        'file': file_path_str,
                        'message': f"Line intended for removal not found exactly in original content: '{line_content_to_remove[:80]}'"
                    })
        else:
            issues.append({'type': 'Validation Warning', 'file': file_path_str, 'message': 'Original content not provided for REMOVE action validation.'})
        return {'issues': issues}

    return {'issues': issues}

def validate_code_output_batch(parsed_data: Dict, original_contents: Dict[str, str] = None) -> Dict[str, Any]:
    """Validates a batch of code changes and aggregates issues per file."""
    if original_contents is None:
        original_contents = {}
    all_validation_results = {}

    if not isinstance(parsed_data, dict):
        logger.error(f"validate_code_output_batch received non-dictionary parsed_data: {type(parsed_data).__name__}")
        malformed_blocks_content = []
        if isinstance(parsed_data, str):
            malformed_blocks_content.append(f"Raw output that failed type check: {parsed_data[:500]}...")
        elif parsed_data is not None:
            malformed_blocks_content.append(f"Unexpected type for parsed_data: {type(parsed_data).__name__}")

        return {'issues': [{'type': 'Internal Error', 'file': 'N/A', 'message': f"Invalid input type for parsed_data: Expected dict, got {type(parsed_data).__name__}"}], 'malformed_blocks': parsed_data.get('malformed_blocks', [])}

    code_changes_list = parsed_data.get('CODE_CHANGES', [])
    if not isinstance(code_changes_list, list):
        logger.error(f"validate_code_output_batch received non-list 'CODE_CHANGES' field: {type(code_changes_list).__name__}")
        return {'issues': [{'type': 'Internal Error', 'file': 'N/A', 'message': f"Invalid type for 'CODE_CHANGES': Expected list, got {type(code_changes_list).__name__}"}], 'malformed_blocks': parsed_data.get('malformed_blocks', [])}

    for i, change_entry in enumerate(code_changes_list):
        if not isinstance(change_entry, dict):
            issue_message = f"Code change entry at index {i} is not a dictionary. Type: {type(change_entry).__name__}, Value: {str(change_entry)[:100]}"
            logger.error(issue_message)
            all_validation_results.setdefault('N/A', []).append({
                'type': 'Malformed Change Entry',
                'file': 'N/A',
                'message': issue_message
            })
            # Explicitly populate malformed_code_change_items for better reporting
            parsed_data.setdefault('malformed_code_change_items', []).append({
                'index': i,
                'original_value': str(change_entry)[:500],
                'error': 'Entry must be a dictionary'
            })
            continue

        file_path = change_entry.get('FILE_PATH')
        if file_path:
            try:
                original_content = original_contents.get(file_path)
                validation_result = validate_code_output(change_entry, original_content)

                all_validation_results[file_path] = validation_result.get('issues', [])
                logger.debug(f"Validation for {file_path} completed with {len(validation_result.get('issues', []))} issues.")
            except Exception as e:
                logger.exception(f"Error during validation of change entry {i} for file {file_path}: {e}") # Use logger.exception for full traceback
                if file_path not in all_validation_results:
                    all_validation_results[file_path] = []
                all_validation_results[file_path].append({'type': 'Validation Tool Error', 'file': file_path, 'message': f'Failed to validate: {e}'})
        else:
            logger.warning(f"Encountered a code change without a 'FILE_PATH' in output {i}. Skipping validation for this item.")
            all_validation_results.setdefault('N/A', []).append({'type': 'Validation Error', 'file': 'N/A', 'message': f'Change item at index {i} missing FILE_PATH.'})

    # --- New: Unit Test Presence Check ---
    python_files_modified_or_added = {
        change['FILE_PATH'] for change in code_changes_list
        if change.get('FILE_PATH', '').endswith('.py') and change.get('ACTION') in ['ADD', 'MODIFY']
    }
    test_files_added = {
        change['FILE_PATH'] for change in code_changes_list
        if change.get('FILE_PATH', '').startswith('tests/') and change.get('ACTION') == 'ADD'
    }

    for py_file in python_files_modified_or_added:
        expected_test_file_prefix = f"tests/test_{Path(py_file).stem}"
        if not any(test_file.startswith(expected_test_file_prefix) for test_file in test_files_added):
            all_validation_results.setdefault(py_file, []).append({'type': 'Missing Unit Test', 'file': py_file, 'message': f"No corresponding unit test file found for this Python change. Expected a file like '{expected_test_file_prefix}.py' in 'tests/'."})

    logger.info(f"Batch validation completed. Aggregated issues for {len(all_validation_results)} files.")

    # NEW: Aggregate metrics for Data-Driven Self-Improvement
    metrics = {
        "total_code_issues": 0,
        "issue_types_summary": defaultdict(int),
        "security_issues_count": 0,
        "style_issues_count": 0,
        "syntax_issues_count": 0,
        "files_with_issues_count": 0,
        "malformed_code_change_items_count": len(parsed_data.get('malformed_code_change_items', []))
    }

    for file_path, file_issues in all_validation_results.items():
        if file_path == '_aggregated_metrics': # Skip the metrics entry itself if it somehow gets here
            continue
        if file_issues:
            metrics["files_with_issues_count"] += 1
            metrics["total_code_issues"] += len(file_issues)
            for issue in file_issues:
                issue_type = issue.get("type", "Unknown")
                metrics["issue_types_summary"][issue_type] += 1
                if "security" in issue_type.lower() or "bandit" in issue_type.lower() or "vulnerability" in issue_type.lower():
                    metrics["security_issues_count"] += 1
                if "ruff" in issue_type.lower() or "style" in issue_type.lower() or "pep8" in issue_type.lower(): # Include pep8 for compatibility
                    metrics["style_issues_count"] += 1
                if "syntax" in issue_type.lower():
                    metrics["syntax_issues_count"] += 1

    # Add the aggregated metrics to the return dictionary
    all_validation_results["_aggregated_metrics"] = metrics
    return all_validation_results