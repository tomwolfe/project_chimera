# utils.py
import json
import re
import subprocess
import tempfile
import os
import sys
import hashlib
import pycodestyle
import difflib
import ast
import io
import contextlib
from typing import Dict, Any, Optional, List, Tuple
import streamlit as st # For st.cache_data

# --- Post-Processing and Validation Functions ---

@st.cache_data(ttl=3600) # Cache validation to avoid re-running slow subprocesses on every rerun.
def _run_validation_in_sandbox(command: List[str], content: str, timeout: int = 10) -> Tuple[int, str, str]:
    """
    Executes a command in a sandboxed environment using a temporary file.
    Returns (return_code, stdout_stderr_output, temp_file_path).
    """
    temp_file_path = None # Initialize to None
    try:
        # Create a temporary file to hold the content
        with tempfile.NamedTemporaryFile(mode='w+', delete=False, suffix='.py') as temp_file:
            temp_file.write(content)
            temp_file_path = temp_file.name # Store the path for later use
        
        # Replace a placeholder filename with the actual temp file path if present in the command
        cmd_with_file = [arg.replace("TEMP_FILE_PLACEHOLDER", temp_file_path) for arg in command]
        
        # Ensure the Python executable is used explicitly for Python commands
        if cmd_with_file[0] == "python" and sys.executable:
            cmd_with_file[0] = sys.executable
            
        process = subprocess.run(
            cmd_with_file,
            capture_output=True,
            text=True,
            check=False, # Don't raise CalledProcessError for non-zero exit codes
            timeout=timeout,
            env={"PYTHONPATH": os.getcwd()} # Ensure Python can find local modules if needed
        )
        # Return the return code, combined stdout/stderr, and the path to the temp file
        return process.returncode, process.stdout + process.stderr, temp_file_path
    except subprocess.TimeoutExpired:
        # Ensure temp_file_path is defined before returning
        return 1, f"Validation timed out after {timeout} seconds.", temp_file_path if 'temp_file_path' in locals() else None
    except Exception as e:
        # Ensure temp_file_path is defined before returning
        return 1, f"Error running validation command: {e}", temp_file_path if 'temp_file_path' in locals() else None
    finally:
        # Clean up the temporary file if it was created
        if 'temp_file_path' in locals() and temp_file_path and os.path.exists(temp_file_path):
            os.remove(temp_file_path)

def repair_json(json_str: str) -> str:
    """Attempts to repair JSON strings by fixing common LLM errors like missing commas or keywords."""
    repaired_str = json_str
    
    # Note: The LLM is instructed to escape quotes in full_content.
    # This function focuses on structural JSON fixes and common LLM-introduced syntax errors.
    # If full_content itself has unescaped quotes, json.loads will fail,
    # and it will be reported as a malformed block.

    # Additional structural fixes for common LLM JSON errors.
    # Fix missing commas between JSON objects
    repaired_str = re.sub(r'}\s*{', '}, {', repaired_str)
    # Fix missing commas after array elements (e.g., "line1" "line2")
    repaired_str = re.sub(r'"\s*"', '", "', repaired_str)
    # Fix missing commas after array elements that are numbers or booleans
    repaired_str = re.sub(r'(\d|true|false)\s*(?="|\{|\[)', r'\1, ', repaired_str)
    
    # Fix for Python keywords that might be concatenated without spaces (e.g., `importos`)
    # This is applied *after* quote escaping to avoid issues with escaped quotes.
    repaired_str = re.sub(r'import([a-zA-Z_])', r'import \1', repaired_str)
    repaired_str = re.sub(r'def([a-zA-Z_])', r'def \1', repaired_str)
    repaired_str = re.sub(r'class([a-zA-Z_])', r'class \1', repaired_str)
    repaired_str = re.sub(r'from([a-zA-Z_])', r'from \1', repaired_str)
    repaired_str = re.sub(r'return([^{(=])', r'return \1', repaired_str)
    repaired_str = re.sub(r'raise([^{(])', r'raise \1', repaired_str)
    repaired_str = re.sub(r'assert([^{(])', r'assert \1', repaired_str)
    
    return repaired_str

@st.cache_data(ttl=3600) # Cache the parsing of LLM output to speed up UI rerenders.
def parse_llm_code_output(llm_output: str) -> Dict[str, Any]:
    """Parses the structured JSON output from the LLM into a dictionary."""
    # Initialize output structure
    output = {
        'summary': {'commit_message': '', 'rationale': '', 'conflict_resolution': '', 'unresolved_conflict': ''},
        'changes': {},
        'malformed_blocks': [],
    }
    
    llm_output_stripped = llm_output.strip()
    llm_output_cleaned = ""

    # 1. Try to extract content from ```json ... ``` block
    json_block_match = re.search(r'```json\s*(.*?)\s*```', llm_output_stripped, re.DOTALL)
    if json_block_match:
        llm_output_cleaned = json_block_match.group(1).strip()
    else:
        # 2. Try to extract content from generic ``` ... ``` block
        generic_block_match = re.search(r'```\s*(.*?)\s*```', llm_output_stripped, re.DOTALL)
        if generic_block_match:
            llm_output_cleaned = generic_block_match.group(1).strip()
        else:
            # 3. If no markdown block found, try to find the first '{' and last '}' as a fallback
            first_brace = llm_output_stripped.find('{')
            last_brace = llm_output_stripped.rfind('}')
            
            if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
                llm_output_cleaned = llm_output_stripped[first_brace : last_brace + 1]
                # Add a warning if we had to use this fallback, as it might indicate malformed LLM output
                if not (llm_output_stripped.startswith('```') and llm_output_stripped.endswith('```')):
                     output['malformed_blocks'].append(f"Warning: No markdown code block fences found. Attempting to parse content between first '{{' and last '}}'. Original output:\n{llm_output_stripped}")
            else:
                # If no braces or malformed, use the entire stripped output and mark as malformed
                llm_output_cleaned = llm_output_stripped
                output['malformed_blocks'].append(f"No valid JSON structure (braces or markdown fences) detected. Attempting to parse raw output. Original output:\n{llm_output_stripped}")

    # Check if the final cleaned output is empty before attempting to parse
    if not llm_output_cleaned:
        output['malformed_blocks'].append(f"LLM output was empty or contained only whitespace after stripping. Original raw output:\n{llm_output}")
        return output # Return early if empty

    try:
        # Attempt to parse the cleaned output
        json_data = json.loads(llm_output_cleaned)
        if not isinstance(json_data, dict):
            raise ValueError("LLM output is not a JSON object.")

    except (json.JSONDecodeError, ValueError) as e:
        # If parsing fails, attempt to repair the JSON
        try:
            repaired_json = repair_json(llm_output_cleaned)
            json_data = json.loads(repaired_json)
            if not isinstance(json_data, dict):
                raise ValueError("LLM output is not a JSON object after heuristic fix.")
            output['malformed_blocks'].append(f"LLM output was initially malformed but was heuristically fixed. Original error: {e}")
        except (json.JSONDecodeError, ValueError) as fix_e:
            output['malformed_blocks'].append(f"LLM output is not valid JSON: {e}\nAttempted heuristic fix failed: {fix_e}\nRaw output:\n{llm_output_cleaned}")
            return output # Return early if fix also fails

    # Extract summary fields
    output['summary']['commit_message'] = json_data.get('COMMIT_MESSAGE', '').strip()
    rationale_content = json_data.get('RATIONALE', '').strip()
    output['summary']['rationale'] = rationale_content

    # Extract conflict resolution/unresolved conflict from rationale within JSON
    if rationale_content:
        conflict_res_match = re.search(r"CONFLICT RESOLUTION:\s*(.*?)(?=\nUNRESOLVED CONFLICT:|\n\n|$)", rationale_content, re.DOTALL)
        if conflict_res_match:
            output['summary']['conflict_resolution'] = conflict_res_match.group(1).strip()
        
        unresolved_conflict_match = re.search(r"UNRESOLVED CONFLICT:\s*(.*?)(?=\n\n|$)", rationale_content, re.DOTALL)
        if unresolved_conflict_match:
            output['summary']['unresolved_conflict'] = unresolved_conflict_match.group(1).strip()

    # Extract code changes
    code_changes_list = json_data.get('CODE_CHANGES', [])
    if not isinstance(code_changes_list, list):
        output['malformed_blocks'].append(f"CODE_CHANGES is not a list: {code_changes_list}")
        return output # Exit early if CODE_CHANGES is malformed

    for change_item in code_changes_list:
        if not isinstance(change_item, dict) or 'file_path' not in change_item or 'action' not in change_item:
            output['malformed_blocks'].append(f"Malformed change item: {change_item}")
            continue
        
        file_path = change_item['file_path']
        action = change_item['action']

        if action == 'ADD':
            if 'full_content' in change_item:
                output['changes'][file_path] = {'type': 'ADD', 'content': change_item['full_content'].strip()}
            else:
                output['malformed_blocks'].append(f"ADD action missing 'full_content' for {file_path}: {change_item}")
        elif action == 'MODIFY':
            if 'full_content' in change_item:
                output['changes'][file_path] = {'type': 'MODIFY', 'new_content': change_item['full_content'].strip()}
            else:
                output['malformed_blocks'].append(f"MODIFY action missing 'full_content' for {file_path}: {change_item}")
        elif action == 'REMOVE':
            if 'lines' in change_item and isinstance(change_item['lines'], list):
                output['changes'][file_path] = {'type': 'REMOVE', 'lines': change_item['lines']}
            else:
                output['malformed_blocks'].append(f"REMOVE action missing 'lines' or 'lines' not a list for {file_path}: {change_item}")
        else:
            output['malformed_blocks'].append(f"Unknown action type '{action}' for {file_path}")
    
    return output

@st.cache_data(ttl=3600) # Cache validation report generation.
def validate_code_output(parsed_data: Dict, original_context: Dict) -> Dict:
    """Validates parsed code for syntax, style, and consistency using sandboxed execution."""
    report = {'issues': [], 'malformed_blocks': parsed_data.get('malformed_blocks', [])}

    for file_path, change in parsed_data.get('changes', {}).items():
        content_to_check = ""
        is_python = file_path.endswith('.py')

        if change['type'] == 'ADD':
            content_to_check = change['content']
            checksum = hashlib.sha256(content_to_check.encode('utf-8')).hexdigest()
            report['issues'].append({'type': 'Content Integrity', 'file': file_path, 'message': f"New file SHA256: {checksum}"})

        elif change['type'] == 'MODIFY':
            content_to_check = change['new_content']
            # Post-Diff Consistency: Calculate checksum of expected new content
            checksum_new = hashlib.sha256(content_to_check.encode('utf-8')).hexdigest()
            report['issues'].append({'type': 'Content Integrity', 'file': file_path, 'message': f"Modified file (new content) SHA256: {checksum_new}"})

        elif change['type'] == 'REMOVE':
            # No content to check for syntax/style for REMOVE, but can check if lines exist in original
            original_lines = original_context.get(file_path, "").splitlines()
            for line_to_remove in change['lines']:
                if line_to_remove not in original_lines:
                    report['issues'].append({'type': 'Diff Inconsistency', 'file': file_path, 'message': f"Line to remove '{line_to_remove}' not found in original file.", 'line': 'N/A'})
            continue # No syntax/style checks for REMOVE

        if is_python and content_to_check:
            # 1. Syntax Validation (sandboxed)
            # Use py_compile for robust syntax validation
            ast_command = [sys.executable, "-m", "py_compile", "TEMP_FILE_PLACEHOLDER"]
            # Capture the temp_file_path returned by the sandbox function
            ast_returncode, ast_output, _ = _run_validation_in_sandbox(ast_command, content_to_check)
            if ast_returncode != 0:
                report['issues'].append({'type': 'Syntax Error', 'file': file_path, 'message': ast_output.strip()})
            
            # 2. Style Compliance (PEP8) (sandboxed)
            # Use pycodestyle on a temporary file
            pep8_command = [sys.executable, "-m", "pycodestyle", "--format=default", "TEMP_FILE_PLACEHOLDER"]
            # Capture the temp_file_path returned by the sandbox function
            pep8_returncode, pep8_output, temp_file_for_pep8 = _run_validation_in_sandbox(pep8_command, content_to_check)
            if pep8_returncode != 0:
                # Parse pycodestyle output to get individual errors
                for line in pep8_output.splitlines():
                    # Use the actual temporary file path in the regex, escaping it for special characters
                    escaped_temp_path = re.escape(temp_file_for_pep8)
                    match = re.match(rf'{escaped_temp_path}:(\d+):\d+: (.*)', line)
                    if match:
                        report['issues'].append({'type': 'PEP8 Violation', 'file': file_path, 'message': match.group(2), 'line': int(match.group(1))})
                    else:
                        report['issues'].append({'type': 'PEP8 Violation', 'file': file_path, 'message': line})
                
    return report

@st.cache_data(ttl=3600) # Cache git diff generation.
def format_git_diff(original_content: str, new_content: str) -> str:
    """Creates a git-style unified diff from original and new content."""
    original_lines = original_content.splitlines(keepends=True)
    new_lines = new_content.splitlines(keepends=True)
    
    diff = difflib.unified_diff(
        original_lines,
        new_lines,
        fromfile='a/original', tofile='b/modified',
        lineterm='' # Prevent adding extra newlines if already present
    )
    # Skip the '--- a/original' and '+++ b/modified' headers
    return "".join(list(diff)[2:])