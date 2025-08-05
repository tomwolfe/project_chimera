# src/utils/code_validator.py

import pycodestyle
import io
from typing import List, Tuple, Dict, Any
import subprocess
import sys
import os
import tempfile
import hashlib
import re
import contextlib
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

class CodeValidationError(Exception):
    """Custom exception for code validation errors."""
    pass

# --- Helper function to find project root ---
def find_project_root(start_path: Path = None) -> Path:
    """
    Finds the project root by searching upwards for a marker file (e.g., config.yaml, .git, pyproject.toml).
    Defaults to the current working directory if no root is found.
    """
    if start_path is None:
        # Start search from the directory of this file
        start_path = Path(__file__).resolve().parent

    current_path = start_path
    # Search upwards for a marker file
    for _ in range(10): # Limit search depth to prevent infinite loops
        if (current_path / "config.yaml").exists() or \
           (current_path / ".git").exists() or \
           (current_path / "pyproject.toml").exists():
            logger.debug(f"Project root found at: {current_path}")
            return current_path
        
        # Move up one directory
        parent_path = current_path.parent
        if parent_path == current_path: # Reached filesystem root
            break
        current_path = parent_path
    
    # Fallback: If no marker found, assume the current working directory is the root.
    # This is common for Streamlit apps launched from the project root.
    logger.warning(
        "Project root marker file (config.yaml, .git, pyproject.toml) not found. "
        "Falling back to current working directory."
    )
    return Path(".").resolve()

# --- Define PROJECT_ROOT dynamically ---
# This ensures that paths used by tools like Bandit or pycodestyle are relative to the project root
# if needed, though in this implementation, we use temporary files directly.
PROJECT_ROOT = find_project_root()

# --- Sandbox Execution Helper ---
@contextlib.contextmanager
def _sandbox_execution(command: List[str], content: str, timeout: int = 10):
    """
    Executes a command in a sandboxed environment using a temporary file.
    Yields the command to execute and the temporary file path.
    This helper is crucial for safely passing code content to external tools.
    """
    temp_file_path = None
    try:
        # Create a temporary file to hold the content.
        # delete=False is used because the file needs to exist when the subprocess runs.
        # We manage cleanup manually in the finally block.
        with tempfile.NamedTemporaryFile(mode='w+', delete=False, suffix='.py', encoding='utf-8') as temp_file:
            temp_file.write(content)
            temp_file_path = temp_file.name # Store the path for later use
        
        # Replace a placeholder filename (if used in the command) with the actual temp file path.
        # This allows commands like `python -m pycodestyle TEMP_FILE_PLACEHOLDER` to work.
        cmd_with_file = [arg.replace("TEMP_FILE_PLACEHOLDER", temp_file_path) for arg in command]
        
        # Ensure the correct Python executable is used, especially in virtual environments.
        if cmd_with_file[0] == "python" and sys.executable:
            cmd_with_file[0] = sys.executable
            
        yield cmd_with_file, temp_file_path
    finally:
        # Clean up the temporary file if it was created and still exists.
        if temp_file_path and os.path.exists(temp_file_path):
            try:
                os.remove(temp_file_path)
            except OSError as e:
                logger.error(f"Error removing temporary file {temp_file_path}: {e}")

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
            # Pass the temporary file path to Bandit.
            command = [
                sys.executable,
                "-m", "bandit",
                "-q",  # Quiet mode
                "-f", "json", # Output format
                # "-c", "/dev/null", # Use default config, or specify a config file if needed
                temp_file.name
            ]
            
            # Execute Bandit using subprocess with shell=False for security.
            process = subprocess.run(
                command,
                capture_output=True,
                text=True,
                check=False, # Don't raise exception on non-zero exit codes
                shell=False, # Crucial for security
                timeout=30 # Add a timeout to prevent hanging
            )

            # Bandit returns 0 if no issues, 1 if issues are found, >1 for errors.
            if process.returncode not in (0, 1): 
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
        issues.append({'type': 'Validation Tool Error', 'file': filename, 'message': 'Bandit execution timed out.'})
    except Exception as e:
        logger.error(f"Unexpected error running Bandit on {filename}: {e}")
        issues.append({'type': 'Validation Tool Error', 'file': filename, 'message': f'Failed to run Bandit: {e}'})
        
    return issues

def validate_code_output(parsed_change: Dict[str, Any], original_content: str = None) -> Dict[str, Any]:
    """Validates a single code change (ADD, MODIFY, REMOVE) for syntax, style, and security.

    Args:
        parsed_change: A dictionary representing a single code change from parse_llm_code_output.
                       Expected keys: 'file_path', 'action', 'full_content' (for ADD/MODIFY) or 'lines' (for REMOVE).
        original_content: The original content of the file if the action is 'MODIFY' or 'REMOVE'.

    Returns:
        A dictionary containing validation results, including issues.
    """
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
            issues.extend(_run_pycodestyle(content_to_check, file_path_str))
            issues.extend(_run_bandit(content_to_check, file_path_str))

    elif action == 'MODIFY':
        content_to_check = parsed_change.get('full_content', '')
        checksum_new = hashlib.sha256(content_to_check.encode('utf-8')).hexdigest()
        issues.append({'type': 'Content Integrity', 'file': file_path_str, 'message': f"Modified file (new content) SHA256: {checksum_new}"})
        
        if original_content is not None:
            original_checksum = hashlib.sha256(original_content.encode('utf-8')).hexdigest()
            if checksum_new == original_checksum:
                issues.append({'type': 'No Change Detected', 'file': file_path_str, 'message': 'New content is identical to original.'})
            if is_python:
                issues.extend(_run_pycodestyle(content_to_check, file_path_str))
                issues.extend(_run_bandit(content_to_check, file_path_str))
        else:
            # If original content is not provided for MODIFY, we can still validate the new content
            if is_python:
                issues.extend(_run_pycodestyle(content_to_check, file_path_str))
                issues.extend(_run_bandit(content_to_check, file_path_str))

    elif action == 'REMOVE':
        # For REMOVE actions, we primarily check if the lines intended for removal exist.
        # This is a heuristic and might not catch all semantic issues.
        if original_content is not None:
            original_lines = original_content.splitlines()
            lines_to_remove = parsed_change.get('lines', [])
            # Use a set for efficient lookup
            original_lines_set = set(original_lines)
            
            for line_content_to_remove in lines_to_remove:
                # Check if the line (or a close approximation) exists in the original content.
                # This check is inherently fuzzy. A more robust approach would involve diffing.
                # For now, we check for exact matches, but log a warning if not found.
                if line_content_to_remove not in original_lines_set:
                    # This is a potential issue: the LLM wants to remove a line that doesn't seem to exist.
                    # It might be a slight modification before removal, or an error.
                    # We'll flag it but not necessarily fail validation unless it's critical.
                    issues.append({
                        'type': 'Potential Removal Mismatch',
                        'file': file_path_str,
                        'message': f"Line intended for removal not found exactly in original content: '{line_content_to_remove[:80]}'"
                    })
        else:
            issues.append({'type': 'Validation Warning', 'file': file_path_str, 'message': 'Original content not provided for REMOVE action validation.'})
        # Return early for REMOVE action as there's no code content to validate for syntax/style
        return {'issues': issues} 

    # Add a general syntax check for Python files if content is available and action is ADD/MODIFY
    if is_python and (action == 'ADD' or action == 'MODIFY') and content_to_check:
        try:
            # Use compile() for a quick syntax check. It raises SyntaxError on failure.
            compile(content_to_check, file_path_str, 'exec')
        except SyntaxError as se:
            # Capture specific syntax error details for better reporting.
            issues.append({
                'type': 'Syntax Error',
                'file': file_path_str,
                'line': se.lineno, # Line number of the syntax error
                'column': se.offset, # Column number of the syntax error
                'message': f"Invalid Python syntax: {se.msg}"
            })
        except Exception as e:
             # Catch any other unexpected errors during compilation.
             issues.append({'type': 'Syntax Error', 'file': file_path_str, 'message': f'Error during compilation: {e}'})

    return {'issues': issues}

def validate_code_output_batch(parsed_changes: List[Dict[str, Any]], original_contents: Dict[str, str] = None) -> Dict[str, List[Dict[str, Any]]]:
    """Validates a batch of code changes.

    Args:
        parsed_changes: A list of dictionaries, where each dictionary represents a code change.
        original_contents: A dictionary mapping file paths to their original content.

    Returns:
        A dictionary where keys are file paths and values are lists of validation issues for that file.
    """
    if original_contents is None:
        original_contents = {}
    all_validation_results = {}
    for change in parsed_changes:
        file_path = change.get('file_path')
        if file_path:
            original_content = original_contents.get(file_path)
            validation_result = validate_code_output(change, original_content)
            # Store issues per file path
            all_validation_results[file_path] = validation_result.get('issues', [])
        else:
            # Handle changes without file_path if necessary, though ideally all should have one.
            # For now, we'll just log a warning if a change lacks a file_path.
            logger.warning(f"Encountered a code change without a 'file_path': {change}. Skipping validation for this item.")
            # Optionally, add a generic issue for the batch if such items are critical.
            # all_validation_results['<unspecified_file>'] = [{'type': 'Validation Error', 'message': 'Change item missing file_path'}]
            
    return all_validation_results