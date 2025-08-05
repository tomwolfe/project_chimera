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

class CodeValidationError(Exception):
    """Custom exception for code validation errors."""
    pass

@contextlib.contextmanager
def _sandbox_execution(command: List[str], content: str, timeout: int = 10):
    """
    Executes a command in a sandboxed environment using a temporary file.
    Yields the command to execute and the temporary file path.
    """
    temp_file_path = None
    try:
        with tempfile.NamedTemporaryFile(mode='w+', delete=False, suffix='.py') as temp_file:
            temp_file.write(content)
            temp_file_path = temp_file.name

        # Replace placeholder with actual temp file path
        cmd_with_file = [arg.replace("TEMP_FILE_PLACEHOLDER", temp_file_path) for arg in command]

        if cmd_with_file[0] == "python" and sys.executable:
            cmd_with_file[0] = sys.executable

        yield cmd_with_file, temp_file_path
    finally:
        if temp_file_path and os.path.exists(temp_file_path):
            os.remove(temp_file_path)

logger = logging.getLogger(__name__)

def validate_code_output(parsed_change: Dict[str, Any], original_content: str = None) -> Dict[str, Any]:
    """Validates a single code change (ADD, MODIFY, REMOVE) for syntax, style, and consistency.

    Args:
        parsed_change: A dictionary representing a single code change from parse_llm_code_output.
                       Expected keys: 'file_path', 'action', 'full_content' (for ADD/MODIFY) or 'lines' (for REMOVE).
        original_content: The original content of the file if the action is 'MODIFY' or 'REMOVE'.

    Returns:
        A dictionary containing validation results, including issues and malformed blocks.
    """
    file_path = parsed_change.get('file_path')
    action = parsed_change.get('action')
    content_to_check = ""
    issues = []

    if not file_path or not action:
        return {'issues': [{'type': 'Validation Error', 'file': file_path or 'N/A', 'message': 'Missing file_path or action in parsed change.'}]}

    is_python = file_path.endswith('.py')

    if action == 'ADD':
        content_to_check = parsed_change.get('full_content', '')
        checksum = hashlib.sha256(content_to_check.encode('utf-8')).hexdigest()
        issues.append({'type': 'Content Integrity', 'file': file_path, 'message': f"New file SHA256: {checksum}"})

    elif action == 'MODIFY':
        content_to_check = parsed_change.get('full_content', '')
        checksum_new = hashlib.sha256(content_to_check.encode('utf-8')).hexdigest()
        issues.append({'type': 'Content Integrity', 'file': file_path, 'message': f"Modified file (new content) SHA256: {checksum_new}"})

        if original_content:
            original_checksum = hashlib.sha256(original_content.encode('utf-8')).hexdigest()
            if checksum_new == original_checksum:
                issues.append({'type': 'No Change Detected', 'file': file_path, 'message': 'New content is identical to original.'})

    elif action == 'REMOVE':
        if original_content:
            original_lines = original_content.splitlines()
            lines_to_remove = parsed_change.get('lines', [])
            for line_to_remove in lines_to_remove:
                if line_to_remove not in original_lines:
                    issues.append({'type': 'Diff Inconsistency', 'file': file_path, 'message': f"Line to remove '{line_to_remove}' not found in original file.", 'line': 'N/A'})
        else:
            issues.append({'type': 'Missing Context', 'file': file_path, 'message': 'Original content not provided for REMOVE action validation.'})
        return {'issues': issues}

    if is_python and content_to_check:
        # 1. Syntax Validation (sandboxed)
        ast_command = [sys.executable, "-m", "compileall", "-q", "TEMP_FILE_PLACEHOLDER"]
        try:
            with _sandbox_execution(ast_command, content_to_check) as (cmd, temp_path):
                final_cmd = [arg.replace("TEMP_FILE_PLACEHOLDER", temp_path) for arg in cmd]
                process = subprocess.run(
                    final_cmd,
                    capture_output=True,
                    text=True,
                    check=False,
                    timeout=10,
                    env={"PYTHONPATH": os.getcwd()}
                )
                if process.returncode != 0:
                    error_output = process.stdout + process.stderr
                    issues.append({'type': 'Syntax Error', 'file': file_path, 'message': error_output.strip()})
        except subprocess.TimeoutExpired:
            issues.append({'type': 'Syntax Error', 'file': file_path, 'message': "Syntax validation timed out."})
        except Exception as e:
            issues.append({'type': 'Syntax Error', 'file': file_path, 'message': f"Error running syntax validation: {e}"})

        # 2. Style Compliance (PEP8) (sandboxed)
        pep8_command = [sys.executable, "-m", "pycodestyle", "--format=default", "TEMP_FILE_PLACEHOLDER"]
        try:
            with _sandbox_execution(pep8_command, content_to_check) as (cmd, temp_path):
                final_cmd = [arg.replace("TEMP_FILE_PLACEHOLDER", temp_path) for arg in cmd]
                process = subprocess.run(
                    final_cmd,
                    capture_output=True,
                    text=True,
                    check=False,
                    timeout=10,
                    env={"PYTHONPATH": os.getcwd()}
                )
                if process.returncode != 0:
                    for line in process.stdout.splitlines():
                        escaped_temp_path = re.escape(temp_path)
                        match = re.match(rf'{escaped_temp_path}:(\d+):(\d+): (.*)', line)
                        if match:
                            issues.append({'type': 'PEP8 Violation', 'file': file_path, 'message': match.group(3), 'line': int(match.group(1)), 'column': int(match.group(2))})
                        else:
                            issues.append({'type': 'PEP8 Violation', 'file': file_path, 'message': line.strip()})
        except subprocess.TimeoutExpired:
            issues.append({'type': 'PEP8 Violation', 'file': file_path, 'message': "Style validation timed out."})
        except Exception as e:
            issues.append({'type': 'PEP8 Violation', 'file': file_path, 'message': f"Error running style validation: {e}"})

    return {'issues': issues}

def validate_code_output_batch(parsed_data: Dict, original_context: Dict) -> Dict:
    """Validates all proposed code changes in a batch."""
    report = {'issues': [], 'malformed_blocks': parsed_data.get('malformed_blocks', [])}

    for change_item in parsed_data.get('code_changes', []): # Iterate over the list of changes
        file_path = change_item.get('file_path')
        original_content = original_context.get(file_path, "") if file_path else ""

        validation_result = validate_code_output(change_item, original_content)
        report['issues'].extend(validation_result.get('issues', []))

    return report
