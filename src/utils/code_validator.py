# src/utils/code_validator.py

import io
from typing import List, Tuple, Dict, Any, Optional
import subprocess
import sys
import os
import tempfile
import hashlib
import re
import contextlib
import logging
from pathlib import Path # Ensure Path is imported
import pycodestyle # Import pycodestyle directly
import ast # Import ast for AST-based checks

logger = logging.getLogger(__name__)

class CodeValidationError(Exception):
    """Custom exception for code validation errors."""
    pass

# --- Helper function to find project root ---
def find_project_root(start_path: Path = None) -> Path:
    """Finds the project root directory by searching for known markers.
    Starts from the directory of the current file and traverses upwards.
    """
    # Define markers to identify the project root
    PROJECT_ROOT_MARKERS = ['.git', 'config.yaml', 'pyproject.toml']

    # Start search from the directory of this file (src/utils)
    if start_path is None:
        start_path = Path(__file__).resolve().parent

    current_dir = start_path
    # Traverse upwards to find the project root
    for _ in range(10): # Limit search depth to prevent infinite loops
        if any(current_dir.joinpath(marker).exists() for marker in PROJECT_ROOT_MARKERS):
            logger.info(f"Project root identified at: {current_dir}")
            return current_dir
        
        parent_path = current_dir.parent
        if parent_path == current_dir: # Reached filesystem root
            break
        current_dir = parent_path
    
    # If no markers are found after searching, raise an error.
    # This is more robust than falling back to the current working directory,
    # as it forces the user to ensure the script is run in a project context.
    raise FileNotFoundError(f"Project root markers ({PROJECT_ROOT_MARKERS}) not found starting from {start_path}. Cannot determine project root.")

# --- Define PROJECT_ROOT dynamically ---
# This ensures that paths used by tools like Bandit or pycodestyle are relative to the project root.
PROJECT_ROOT = find_project_root()

# MODIFIED: Renamed PROJECT_BASE_DIR to PROJECT_ROOT for consistency
def is_within_base_dir(file_path: Path) -> bool:
    """Checks if a file path is safely within the project base directory.
    Handles potential exceptions during path resolution or comparison.
    """
    try:
        # Resolve the path to handle symlinks and relative paths correctly
        resolved_path = file_path.resolve()
        # Check if the resolved path is a subdirectory of the project base directory
        resolved_path.relative_to(PROJECT_ROOT) # MODIFIED: Use PROJECT_ROOT
        return True
    except ValueError:
        # Path is not relative to PROJECT_ROOT (outside the scope)
        logger.debug(f"Path '{file_path}' is outside the project base directory '{PROJECT_ROOT}'.") # MODIFIED: Use PROJECT_ROOT
        return False
    except Exception as e:
        # Catch other potential errors during path operations (e.g., permissions)
        logger.error(f"Error resolving or comparing path '{file_path}' against base directory '{PROJECT_ROOT}': {e}") # MODIFIED: Use PROJECT_ROOT
        return False

def sanitize_and_validate_file_path(raw_path: str) -> str:
    """Sanitizes and validates a file path for safety against traversal and invalid characters.
    Ensures the path is within the project's base directory.
    """
    if not raw_path:
        raise ValueError("File path cannot be empty.")

    # Basic character sanitization: remove characters invalid in most file systems
    # and control characters. This is a defense-in-depth measure.
    # Removed space from forbidden characters as it's a valid path character.
    sanitized_path_str = re.sub(r'[<>:"|?*\\\x00-\x1f]', '', raw_path)

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


def _run_pycodestyle(content: str, filename: str) -> List[Dict[str, Any]]:
    """Runs pycodestyle on the given content using its library API."""
    issues = []
    try:
        # Use StyleGuide for checking code. quiet=True suppresses non-error messages.
        style_guide = pycodestyle.StyleGuide(quiet=True, format='default')
        
        # pycodestyle's check_files expects file paths. We simulate this using a temporary file.
        # delete=True ensures the file is cleaned up automatically after use.
        with tempfile.NamedTemporaryFile(mode='w+', suffix='.py', encoding='utf-8', delete=True) as temp_file:
            temp_file.write(content)
            temp_file.flush() # Ensure content is written before pycodestyle reads it
            
            # pycodestyle.check_files expects a list of filenames.
            # We pass the temporary file's name.
            report = style_guide.check_files([temp_file.name])
            
            # Process the report. Each line typically contains: filename:line:col: code message
            for line in report.splitlines():
                # Regex to parse pycodestyle output format, capturing line, col, code, and message.
                match = re.match(r"^[^:]+:(?P<line>\d+):(?P<col>\d+): (?P<code>\w+) (?P<message>.*)", line)
                if match:
                    issues.append({
                        'type': 'PEP8 Violation',
                        'file': filename, # Use the original filename for reporting
                        'line': int(match.group('line')),
                        'column': int(match.group('col')),
                        'code': match.group('code'),
                        'message': match.group('message').strip()
                    })
        # If report is empty, no issues were found, which is correct.
    except Exception as e:
        logger.error(f"Error running pycodestyle on {filename}: {e}")
        issues.append({'type': 'Validation Tool Error', 'file': filename, 'message': f'Failed to run pycodestyle: {e}'})
    return issues

def _run_bandit(content: str, filename: str) -> List[Dict[str, Any]]:
    """Runs Bandit security analysis on the given content via subprocess."""
    issues = []
    # Bandit typically analyzes files, so we use a temporary file.
    # Using delete=True for automatic cleanup.
    try:
        with tempfile.NamedTemporaryFile(mode='w+', suffix='.py', encoding='utf-8', delete=True) as temp_file:
            temp_file.write(content)
            temp_file.flush()
            
            # Construct the Bandit command safely.
            # Use sys.executable to ensure the correct Python interpreter is used.
            # Pass the temporary file path to Bandit. The placeholder is handled by _sandbox_execution.
            command = [
                sys.executable,
                "-m", "bandit", # Module execution is safer
                "-q",  # Quiet mode
                "-f", "json", # Output format
                # "-c", "/dev/null", # Use default config, or specify a config file if needed
                temp_file.name
            ]
            
            # Execute Bandit using subprocess with shell=False for security.
            process = subprocess.run(
                command,
                capture_output=True, # Capture stdout and stderr
                text=True,
                check=False, # Don't raise exception on non-zero exit codes
                shell=False, # Crucial for security
                timeout=30 # Add a timeout to prevent hanging
            )

            # Bandit returns 0 if no issues, 1 if issues are found, >1 for errors.
            if process.returncode not in (0, 1): # Check for errors other than finding issues
                logger.error(f"Bandit execution failed for {filename} with return code {process.returncode}. Stderr: {process.stderr}")
                issues.append({'type': 'Validation Tool Error', 'file': filename, 'message': f'Bandit failed: {process.stderr}'})
            else:
                try:
                    # Parse the JSON output.
                    bandit_results = process.stdout.strip()
                    if bandit_results:
                        import json
                        data = json.loads(bandit_results)
                        for issue in data.get('results', []):
                            # Filter out 'info' level issues if desired, or include all.
                            # For security analysis, 'info' might be relevant too, but typically warnings/errors are prioritized.
                            if issue['level'] != 'info': # Example: only include non-info issues
                                issues.append({
                                    'type': 'Bandit Security Issue',
                                    'file': filename,
                                    'line': issue.get('line_number'), # Bandit provides line numbers
                                    'code': issue.get('test_id'),
                                    'message': f"[{issue.get('severity')}] {issue.get('description')}"
                                })
                except json.JSONDecodeError as jde:
                    logger.error(f"Failed to parse Bandit JSON output for {filename}: {jde}. Output: {process.stdout}")
                    issues.append({'type': 'Validation Tool Error', 'file': filename, 'message': f'Failed to parse Bandit output: {jde}'})
                except Exception as e:
                    logger.error(f"Unexpected error processing Bandit output for {filename}: {e}")
                    issues.append({'type': 'Validation Tool Error', 'file': filename, 'message': f'Error processing Bandit output: {e}'})

    except FileNotFoundError:
        logger.error("Bandit command not found. Ensure Bandit is installed and in the PATH.")
        issues.append({'type': 'Validation Tool Error', 'file': filename, 'message': 'Bandit executable not found. Please install Bandit.'})
    except subprocess.TimeoutExpired:
        logger.error(f"Bandit execution timed out for {filename}.")
        issues.append({'type': 'Validation Tool Error', 'file': filename, 'message': 'Bandit execution timed out.'}) # Added message
    except Exception as e:
        logger.error(f"Unexpected error running Bandit on {filename}: {e}")
        issues.append({'type': 'Validation Tool Error', 'file': filename, 'message': f'Failed to run Bandit: {e}'})
        
    return issues

def _run_ast_security_checks(content: str, filename: str) -> List[Dict[str, Any]]:
    """Runs AST-based security checks on Python code."""
    issues = []
    try:
        tree = ast.parse(content)
        
        class SecurityPatternVisitor(ast.NodeVisitor):
            def __init__(self, filename):
                self.filename = filename
                self.issues = []

            def visit_Call(self, node):
                # Check for eval() and exec()
                if isinstance(node.func, ast.Name):
                    if node.func.id == 'eval':
                        self.issues.append({
                            'type': 'Security Vulnerability (AST)',
                            'file': self.filename,
                            'line': node.lineno,
                            'message': "Use of eval() is discouraged due to security risks."
                        })
                    elif node.func.id == 'exec':
                        self.issues.append({
                            'type': 'Security Vulnerability (AST)',
                            'file': self.filename,
                            'line': node.lineno,
                            'message': "Use of exec() is discouraged due to security risks."
                        })
                
                # Check for subprocess.run with shell=True
                if isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name) and node.func.value.id == 'subprocess' and node.func.attr == 'run':
                    for keyword in node.keywords:
                        if keyword.arg == 'shell' and isinstance(keyword.value, ast.Constant) and keyword.value.value is True: # Check for shell=True
                            self.issues.append({
                                'type': 'Security Vulnerability (AST)',
                                'file': self.filename,
                                'line': node.lineno,
                                'message': "subprocess.run() with shell=True is dangerous; consider shell=False and passing arguments as a list."
                            })
                
                # Check for pickle.load
                if isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name) and node.func.value.id == 'pickle' and node.func.attr == 'load': # Check for pickle.load
                    self.issues.append({
                        'type': 'Security Vulnerability (AST)',
                        'file': self.filename,
                        'line': node.lineno,
                        'message': "Use of pickle.load() with untrusted data is dangerous; it can execute arbitrary code."
                    })
                
                # Check for os.system
                if isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name) and node.func.value.id == 'os' and node.func.attr == 'system':
                    self.issues.append({
                        'type': 'Security Vulnerability (AST)',
                        'file': self.filename,
                        'line': node.lineno,
                        'message': "Use of os.system() is discouraged; it can execute arbitrary commands and is prone to shell injection. Consider subprocess.run() with shell=False."
                    })
                
                # Check for XML External Entity (XXE) vulnerability in ElementTree
                # Example: ET.fromstring(xml_string, parser=None)
                if isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name) and node.func.value.id == 'ET' and node.func.attr == 'fromstring':
                     # Check if parser=None is explicitly passed
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
                             'message': "xml.etree.ElementTree.fromstring() with parser=None is vulnerable to XML External Entity (XXE) attacks. Use a safe parser or disable DTDs."
                         })
                
                self.generic_visit(node) # Continue visiting child nodes to traverse the AST

        visitor = SecurityPatternVisitor(filename)
        visitor.visit(tree)
        issues.extend(visitor.issues)
    except SyntaxError as se:
        # Capture specific syntax error details for better reporting.
        issues.append({
            'type': 'Syntax Error',
            'file': filename,
            'line': se.lineno, # Line number of the syntax error
            'column': se.offset, # Column number of the syntax error
            'message': f"Invalid Python syntax: {se.msg}"
        })
    except Exception as e:
        logger.error(f"Error during AST analysis for {filename}: {e}")
        issues.append({'type': 'Validation Tool Error', 'file': filename, 'message': f'Failed during AST analysis: {e}'})
    return issues

def validate_code_output(parsed_change: Dict[str, Any], original_content: str = None) -> Dict[str, Any]:
    """Validates a single code change (ADD, MODIFY, REMOVE) for syntax, style, and security."""
    file_path_str = parsed_change.get('file_path')
    action = parsed_change.get('action')
    content_to_check = ""
    issues = []

    if not file_path_str or not action:
        return {'issues': [{'type': 'Validation Error', 'file': file_path_str or 'N/A', 'message': 'Missing file_path or action in parsed change.'}]}
    
    file_path = Path(file_path_str)
    is_python = file_path.suffix.lower() == '.py'

    if action == 'ADD':
        content_to_check = parsed_change.get('full_content', '')
        checksum = hashlib.sha256(content_to_check.encode('utf-8')).hexdigest()
        issues.append({'type': 'Content Integrity', 'file': file_path_str, 'message': f"New file SHA256: {checksum}"})
        if is_python:
            # Run style and security checks
            issues.extend(_run_pycodestyle(content_to_check, file_path_str))
            issues.extend(_run_bandit(content_to_check, file_path_str))
            # Add AST-based security checks
            issues.extend(_run_ast_security_checks(content_to_check, file_path_str))

    elif action == 'MODIFY':
        content_to_check = parsed_change.get('full_content', '')
        checksum_new = hashlib.sha256(content_to_check.encode('utf-8')).hexdigest()
        issues.append({'type': 'Content Integrity', 'file': file_path_str, 'message': f"Modified file (new content) SHA256: {checksum_new}"})
        
        if original_content is not None:
            original_checksum = hashlib.sha256(original_content.encode('utf-8')).hexdigest()
            if checksum_new == original_checksum:
                issues.append({'type': 'No Change Detected', 'file': file_path_str, 'message': 'New content is identical to original.'})
            if is_python:
                # Run style and security checks
                issues.extend(_run_pycodestyle(content_to_check, file_path_str))
                issues.extend(_run_bandit(content_to_check, file_path_str))
                issues.extend(_run_ast_security_checks(content_to_check, file_path_str))
        else:
            # If original content is not provided for MODIFY, we can still validate the new content
            if is_python:
                issues.extend(_run_pycodestyle(content_to_check, file_path_str))
                issues.extend(_run_bandit(content_to_check, file_path_str))
                issues.extend(_run_ast_security_checks(content_to_check, file_path_str))

    elif action == 'REMOVE':
        # For REMOVE actions, we primarily check if the lines intended for removal exist.
        # This is a heuristic and might not catch all semantic issues.
        if original_content is not None:
            original_lines = original_content.splitlines()
            lines_to_remove = parsed_change.get('lines', [])
            # Use a set for efficient lookup
            original_lines_set = set(original_lines)
            
            for line_content_to_remove in lines_to_remove:
                # This check is inherently fuzzy. A more robust approach would involve diffing.
                # For now, we check for exact matches, but log a warning if not found.
                if line_content_to_remove not in original_lines_set:
                    # This is a potential issue: the LLM wants to remove a line that doesn't seem to exist exactly.
                    # It might be a slight modification before removal, or an error.
                    # We'll flag it but not necessarily fail validation unless it's critical.
                    issues.append({
                        'type': 'Potential Removal Mismatch',
                        'file': file_path_str,
                        'message': f"Line intended for removal not found exactly in original content: '{line_content_to_remove[:80]}'"
                    })
        else:
            issues.append({'type': 'Validation Warning', 'file': file_path_str, 'message': 'Original content not provided for REMOVE action validation.'})
        # Return early for REMOVE action as there's no code content to validate
        return {'issues': issues} # Return issues found for REMOVE action
        
    return {'issues': issues}

# The main validation function that orchestrates checks for all changes
def validate_code_output_batch(parsed_data: Dict, original_contents: Dict[str, str] = None) -> Dict[str, List[Dict[str, Any]]]:
    """Validates a batch of code changes and aggregates issues per file. 
    This function is called after parse_and_validate has succeeded and returned a structured dictionary.
    """
    if original_contents is None:
        original_contents = {}
    all_validation_results = {}

    # Ensure parsed_data is a dictionary and contains the 'code_changes' key as a list
    if not isinstance(parsed_data, dict):
        logger.error(f"validate_code_output_batch received non-dictionary parsed_data: {type(parsed_data).__name__}")
        # Safely handle the case where parsed_data is not a dictionary.
        malformed_blocks_content = []
        if isinstance(parsed_data, str):
            # If parsed_data is a string, capture its content as a malformed block.
            malformed_blocks_content.append(f"Raw output that failed type check: {parsed_data[:500]}...")
        elif parsed_data is not None: # If it's some other non-dict type (e.g., list, int)
            # Capture the unexpected type.
            malformed_blocks_content.append(f"Unexpected type for parsed_data: {type(parsed_data).__name__}")
        
        # Return a structured error response, ensuring 'malformed_blocks' is always a list.
        return {'issues': [{'type': 'Internal Error', 'file': 'N/A', 'message': f"Invalid input type for parsed_data: Expected dict, got {type(parsed_data).__name__}"}], 'malformed_blocks': malformed_blocks_content}

    code_changes_list = parsed_data.get('code_changes', [])
    if not isinstance(code_changes_list, list):
        logger.error(f"validate_code_output_batch received non-list 'code_changes' field: {type(code_changes_list).__name__}")
        return {'issues': [{'type': 'Internal Error', 'file': 'N/A', 'message': f"Invalid type for 'code_changes': Expected list, got {type(code_changes_list).__name__}"}], 'malformed_blocks': parsed_data.get('malformed_blocks', [])}

    for i, change_entry in enumerate(code_changes_list):
        if not isinstance(change_entry, dict):
            # This is the specific error condition: an item in the list is not a dictionary
            issue_message = f"Code change entry at index {i} is not a dictionary. Type: {type(change_entry).__name__}, Value: {str(change_entry)[:100]}"
            logger.error(issue_message) # Log the error
            # Add the issue to the 'N/A' file key in the issues list.
            all_validation_results.setdefault('N/A', []).append({'type': 'Malformed Change Entry', 'file': 'N/A', 'message': issue_message})
            continue # Skip this malformed entry and proceed to the next

        file_path = change_entry.get('file_path')
        if file_path:
            try:
                # validate_code_output expects a single change dict and original content (if available)
                # We pass original_contents which maps file_path to its content.
                original_content = original_contents.get(file_path)
                validation_result = validate_code_output(change_entry, original_content)
                
                # Store issues per file path
                all_validation_results[file_path] = validation_result.get('issues', [])
                logger.debug(f"Validation for {file_path} completed with {len(validation_result.get('issues', []))} issues.")
            except Exception as e:
                logger.error(f"Error during validation of change entry {i} for file {file_path}: {e}")
                # Add an error issue if validation itself fails
                if file_path not in all_validation_results:
                    all_validation_results[file_path] = []
                all_validation_results[file_path].append({'type': 'Validation Tool Error', 'file': file_path, 'message': f'Failed to validate: {e}'})
        else:
            # Handle changes without file_path if necessary
            logger.warning(f"Encountered a code change without a 'file_path' in output {i}. Skipping validation for this item.")
            # Add a generic issue for the batch if such items are critical.
            all_validation_results.setdefault('N/A', []).append({'type': 'Validation Error', 'file': 'N/A', 'message': f'Change item at index {i} missing file_path.'})
            
    # --- New: Unit Test Presence Check ---
    python_files_modified_or_added = {
        change['file_path'] for change in code_changes_list
        if change.get('file_path', '').endswith('.py') and change.get('action') in ['ADD', 'MODIFY']
    }
    test_files_added = {
        change['file_path'] for change in code_changes_list
        if change.get('file_path', '').startswith('tests/') and change.get('action') == 'ADD'
    }

    for py_file in python_files_modified_or_added:
        expected_test_file_prefix = f"tests/test_{Path(py_file).stem}"
        if not any(test_file.startswith(expected_test_file_prefix) for test_file in test_files_added):
            all_validation_results.setdefault(py_file, []).append({'type': 'Missing Unit Test', 'file': py_file, 'message': f"No corresponding unit test file found for this Python change. Expected a file like '{expected_test_file_prefix}.py' in 'tests/'."})
            
    logger.info(f"Batch validation completed. Aggregated issues for {len(all_validation_results)} files.")
    return all_validation_results