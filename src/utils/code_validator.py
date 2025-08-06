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
from pathlib import Path
import pycodestyle
import ast

from src.utils.path_utils import find_project_root, is_within_base_dir, sanitize_and_validate_file_path

logger = logging.getLogger(__name__)

class CodeValidationError(Exception):
    """Custom exception for code validation errors."""
    pass

def _run_pycodestyle(content: str, filename: str) -> List[Dict[str, Any]]:
    """Runs pycodestyle on the given content using its library API, avoiding temporary files."""
    issues = []
    try:
        # Use pycodestyle's API to check code directly from content.
        # This avoids the overhead of temporary files.
        # Pass filename for accurate error reporting (line numbers, etc.)
        style_guide = pycodestyle.StyleGuide(quiet=True, format='default')
        # The Checker takes a list of lines and a filename.
        # We simulate a file by passing the content as lines.
        checker = pycodestyle.Checker(
            filename=filename,
            lines=content.splitlines(keepends=True),
            # Optionally, you can specify options here if needed,
            # e.g., exclude=['E501']
        )

        # `check_all` returns a list of tuples: (line_number, column_number, error_code, text)
        errors = checker.check_all()

        for line_num, col_num, code, message in errors:
            issues.append({
                "line_number": line_num,
                "column_number": col_num,
                "code": code,
                "message": message.strip(),
                "source": "pycodestyle",
                "filename": filename,
                "type": "PEP8 Violation" # Added type for consistency with other issues
            })

    except Exception as e:
        logger.error(f"Error running pycodestyle on {filename}: {e}")
        issues.append({
            "line_number": None,
            "column_number": None,
            "code": "PYCODESTYLE_ERROR",
            "message": f"Internal error during pycodestyle check: {e}",
            "source": "pycodestyle",
            "filename": filename,
            "type": "Validation Tool Error" # Added type
        })
    return issues

def _run_bandit(content: str, filename: str) -> List[Dict[str, Any]]:
    """Runs Bandit security analysis on the given content via subprocess."""
    issues = []
    # Bandit typically analyzes files, so we use a temporary file.
    # Using delete=True for automatic cleanup.
    try:
        with tempfile.NamedTemporaryFile(
            mode='w+', suffix='.py', encoding='utf-8', delete=True
        ) as temp_file:
            temp_file.write(content)
            temp_file.flush()
            
            # Construct the Bandit command safely.
            # Use sys.executable to ensure the correct Python interpreter is used.
            # Pass the temporary file path to Bandit.
            command = [
                sys.executable,
                "-m", "bandit", # Module execution is safer
                "-q",  # Quiet mode
                "-f", "json", # Output format
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
        issues.append({'type': 'Validation Tool Error', 'file': filename, 'message': 'Bandit execution timed out.'})
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
    file_path_str = parsed_change.get('FILE_PATH') # Use uppercase key
    action = parsed_change.get('ACTION') # Use uppercase key
    content_to_check = ""
    issues = []

    if not file_path_str or not action:
        return {'issues': [{'type': 'Validation Error', 'file': file_path_str or 'N/A', 'message': 'Missing FILE_PATH or ACTION in parsed change.'}]} # Use uppercase key
    
    file_path = Path(file_path_str)
    is_python = file_path.suffix.lower() == '.py'

    if action == 'ADD':
        content_to_check = parsed_change.get('FULL_CONTENT', '') # Use uppercase key
        checksum = hashlib.sha256(content_to_check.encode('utf-8')).hexdigest()
        issues.append({'type': 'Content Integrity', 'file': file_path_str, 'message': f"New file SHA256: {checksum}"})
        if is_python:
            # Run style and security checks
            issues.extend(_run_pycodestyle(content_to_check, file_path_str))
            issues.extend(_run_bandit(content_to_check, file_path_str))
            # Add AST-based security checks
            issues.extend(_run_ast_security_checks(content_to_check, file_path_str))

    elif action == 'MODIFY':
        content_to_check = parsed_change.get('FULL_CONTENT', '') # Use uppercase key
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
        if original_content is not None:
            original_lines = original_content.splitlines()
            lines_to_remove = parsed_change.get('LINES', []) # Use uppercase key
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

def validate_code_output_batch(parsed_data: Dict, original_contents: Dict[str, str] = None) -> Dict[str, List[Dict[str, Any]]]:
    """Validates a batch of code changes and aggregates issues per file. 
    This function is called after parse_and_validate has succeeded and returned a structured dictionary.
    """
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
        
        return {'issues': [{'type': 'Internal Error', 'file': 'N/A', 'message': f"Invalid input type for parsed_data: Expected dict, got {type(parsed_data).__name__}"}], 'malformed_blocks': malformed_blocks_content}

    code_changes_list = parsed_data.get('CODE_CHANGES', [])
    if not isinstance(code_changes_list, list):
        logger.error(f"validate_code_output_batch received non-list 'CODE_CHANGES' field: {type(code_changes_list).__name__}")
        return {'issues': [{'type': 'Internal Error', 'file': 'N/A', 'message': f"Invalid type for 'CODE_CHANGES': Expected list, got {type(code_changes_list).__name__}"}], 'malformed_blocks': parsed_data.get('malformed_blocks', [])}

    for i, change_entry in enumerate(code_changes_list):
        if not isinstance(change_entry, dict):
            issue_message = f"Code change entry at index {i} is not a dictionary. Type: {type(change_entry).__name__}, Value: {str(change_entry)[:100]}"
            logger.error(issue_message)
            all_validation_results.setdefault('N/A', []).append({'type': 'Malformed Change Entry', 'file': 'N/A', 'message': issue_message})
            continue

        file_path = change_entry.get('FILE_PATH')
        if file_path:
            try:
                original_content = original_contents.get(file_path)
                validation_result = validate_code_output(change_entry, original_content)
                
                all_validation_results[file_path] = validation_result.get('issues', [])
                logger.debug(f"Validation for {file_path} completed with {len(validation_result.get('issues', []))} issues.")
            except Exception as e:
                logger.error(f"Error during validation of change entry {i} for file {file_path}: {e}")
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
    return all_validation_results