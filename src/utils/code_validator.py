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
import json # Added for Bandit output parsing
import yaml # Added for YAML security checks
from collections import defaultdict # Added for metrics aggregation

from src.utils.path_utils import is_within_base_dir, sanitize_and_validate_file_path

logger = logging.getLogger(__name__)

class CodeValidationError(Exception):
    """Custom exception for code validation errors."""
    pass

def _run_pycodestyle(content: str, filename: str) -> List[Dict[str, Any]]:
    """Runs pycodestyle on the given content using its library API."""
    issues = []
    try:
        # pycodestyle.Checker expects an options object, even if empty.
        # We'll use a custom reporter to capture messages.
        options = pycodestyle.parse_options([])[0]
        
        class CustomReporter(pycodestyle.BaseReport):
            def __init__(self, options):
                super().__init__(options)
                self.messages = []
            
            def error(self, line_number, column_number, text, check):
                # Extract the error code (e.g., 'E501') from the text
                code = text.split(' ')[0] 
                self.messages.append((line_number, column_number, code, text))
        
        # Assign our custom reporter to the options
        options.reporter = CustomReporter
        
        checker = pycodestyle.Checker(
            filename=filename,
            lines=content.splitlines(keepends=True),
            options=options # Pass the options with our custom reporter
        )

        # Run checks. The errors will be collected by CustomReporter.
        checker.check_all()

        # Access the collected messages from the reporter instance
        for line_num, col_num, code, message in checker.report.messages:
            issues.append({
                "line_number": line_num,
                "column_number": col_num,
                "code": code,
                "message": message.strip(),
                "source": "pycodestyle",
                "filename": filename,
                "type": "PEP8 Violation"
            })

    except Exception as e:
        logger.error(f"Error running pycodestyle on {filename}: {e}", exc_info=True)
        issues.append({
            "line_number": None,
            "column_number": None,
            "code": "PYCODESTYLE_ERROR",
            "message": f"Internal error during pycodestyle check: {e}",
            "source": "pycodestyle",
            "filename": filename,
            "type": "Validation Tool Error"
        })
    return issues

def _run_bandit(content: str, filename: str) -> List[Dict[str, Any]]:
    """Runs Bandit security analysis on the given content via subprocess."""
    issues = []
    tmp_file_path = None # Initialize to None
    try:
        # FIX: Set delete=False for NamedTemporaryFile and add explicit cleanup
        with tempfile.NamedTemporaryFile(
            mode='w+', suffix='.py', encoding='utf-8', delete=False 
        ) as temp_file:
            temp_file.write(content)
            temp_file.flush()
            tmp_file_path = Path(temp_file.name) # Store path for explicit unlink
            
            command = [
                sys.executable,
                "-m", "bandit",
                "-q",
                "-f", "json",
                str(tmp_file_path) # Use str(Path)
            ]
            
            process = subprocess.run(
                command,
                capture_output=True,
                text=True,
                check=False,
                shell=False,
                timeout=30
            )

            if process.returncode not in (0, 1):
                logger.error(f"Bandit execution failed for {filename} with return code {process.returncode}. Stderr: {process.stderr}")
                issues.append({'type': 'Validation Tool Error', 'file': filename, 'message': f'Bandit failed: {process.stderr}'})
            else:
                try:
                    bandit_results = process.stdout.strip()
                    if bandit_results:
                        data = json.loads(bandit_results)
                        for issue in data.get('results', []):
                            # FIX: Removed the problematic 'if issue['level'] != 'info':'.
                            # Bandit's JSON output uses 'severity' and 'description', not 'level'.
                            issues.append({
                                'type': 'Bandit Security Issue',
                                'file': filename,
                                'line': issue.get('line_number'),
                                'code': issue.get('test_id'),
                                'message': f"[{issue.get('severity')}] {issue.get('description')}"
                            })
                except json.JSONDecodeError as jde:
                    logger.error(f"Failed to parse Bandit JSON output for {filename}: {jde}. Output: {process.stdout}", exc_info=True)
                    issues.append({'type': 'Validation Tool Error', 'file': filename, 'message': f'Failed to parse Bandit output: {jde}'})
                except Exception as e:
                    logger.error(f"Unexpected error processing Bandit output for {filename}: {e}", exc_info=True)
                    issues.append({'type': 'Validation Tool Error', 'file': filename, 'message': f'Error processing Bandit output: {e}'})

    except FileNotFoundError:
        logger.error("Bandit command not found. Ensure Bandit is installed and in the PATH.")
        issues.append({'type': 'Validation Tool Error', 'file': filename, 'message': 'Bandit executable not found. Please install Bandit.'})
    except subprocess.TimeoutExpired:
        logger.error(f"Bandit execution timed out for {filename}.")
        issues.append({'type': 'Validation Tool Error', 'file': filename, 'message': 'Bandit execution timed out.'})
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
    try:
        tree = ast.parse(content)
        
        class EnhancedSecurityPatternVisitor(ast.NodeVisitor):
            def __init__(self, filename):
                self.filename = filename
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
                        if keyword.arg == 'shell' and isinstance(keyword.value, ast.Constant) and keyword.value.value is True:
                            self.issues.append({
                                'type': 'Security Vulnerability (AST)',
                                'file': self.filename,
                                'line': node.lineno,
                                'message': "subprocess.run() with shell=True is dangerous; consider shell=False and passing arguments as a list."
                            })
                
                # Check for pickle.load
                if isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name) and node.func.value.id == 'pickle' and node.func.attr == 'load':
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
                             'message': "xml.etree.ElementTree.fromstring() with parser=None is vulnerable to XML External Entity (XXE) attacks. Use a safe parser or disable DTDs."
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
                            'message': "pickle.loads() with potentially untrusted data can execute arbitrary code. Use a safe serialization format like JSON."
                        })
                    
                    # Check for yaml.load with Loader parameter missing
                    if (node.func.value.id == 'yaml' and node.func.attr == 'load' and
                        not self._has_safe_loader_parameter(node)):
                        self.issues.append({
                            'type': 'Security Vulnerability (AST)',
                            'file': self.filename,
                            'line': node.lineno,
                            'message': "yaml.load() without Loader parameter is unsafe. Use yaml.safe_load() or specify Loader=yaml.SafeLoader."
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
                            'message': "subprocess.run() with shell=True is dangerous; consider shell=False and passing arguments as a list."
                        })
                    else: # Check for shell metacharacters in string arguments if shell=True is not explicit
                        for arg in node.args:
                            if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
                                if any(char in arg.value for char in [';', '|', '&', '$', '`', '>', '<', '(', ')', '#', '*']):
                                    self.issues.append({
                                        'type': 'Security Vulnerability (AST)',
                                        'file': self.filename,
                                        'line': node.lineno,
                                        'message': f"Potential shell injection in subprocess.{node.func.attr} with string argument containing shell metacharacters. Consider passing arguments as a list."
                                    })
                
                self.generic_visit(node)
            
            def _is_potentially_untrusted_input(self, node) -> bool:
                """Heuristic to check if function argument might be untrusted input."""
                if node.args and isinstance(node.args[0], ast.Name):
                    arg_name = node.args[0].id
                    untrusted_keywords = ['input', 'user', 'request', 'param', 'data', 'body', 'query', 'json', 'raw']
                    return any(keyword in arg_name.lower() for keyword in untrusted_keywords)
                return True # Assume untrusted if we can't determine
            
            def _has_safe_loader_parameter(self, node) -> bool:
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

        visitor = EnhancedSecurityPatternVisitor(filename)
        visitor.visit(tree)
        issues.extend(visitor.issues)
    except SyntaxError as se:
        issues.append({
            'type': 'Syntax Error',
            'file': filename,
            'line': se.lineno,
            'column': se.offset,
            'message': f"Invalid Python syntax: {se.msg}"
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

    # MODIFIED: Removed 'APPEND' action as it's not defined in the schema/model
    if action == 'ADD':
        content_to_check = parsed_change.get('FULL_CONTENT', '')
        checksum = hashlib.sha256(content_to_check.encode('utf-8')).hexdigest()
        issues.append({'type': 'Content Integrity', 'file': file_path_str, 'message': f"New file SHA256: {checksum}"})
        if is_python:
            issues.extend(_run_pycodestyle(content_to_check, file_path_str))
            issues.extend(_run_bandit(content_to_check, file_path_str))
            # Add AST-based security checks
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
                issues.extend(_run_pycodestyle(content_to_check, file_path_str))
                issues.extend(_run_bandit(content_to_check, file_path_str))
                issues.extend(_run_ast_security_checks(content_to_check, file_path_str))
        else:
            if is_python:
                issues.extend(_run_pycodestyle(content_to_check, file_path_str))
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
                if "pep8" in issue_type.lower() or "style" in issue_type.lower():
                    metrics["style_issues_count"] += 1
                if "syntax" in issue_type.lower():
                    metrics["syntax_issues_count"] += 1
    
    # Add the aggregated metrics to the return dictionary
    all_validation_results["_aggregated_metrics"] = metrics
    return all_validation_results