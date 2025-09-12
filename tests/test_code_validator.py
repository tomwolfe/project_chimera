# tests/test_code_validator.py

import pytest
from unittest.mock import MagicMock, patch  # NEW: Import patch and MagicMock

# Assuming src/utils/code_validator.py contains functions like validate_code_output
from src.utils.code_validator import (
    validate_code_output,
    _run_ruff,
    _run_bandit,
    _run_ast_security_checks,
)
from src.utils.command_executor import (
    execute_command_safely,
)  # NEW: Import execute_command_safely
from src.utils.code_utils import _get_code_snippet  # NEW: Import _get_code_snippet


# Mock execute_command_safely for _run_ruff and _run_bandit
@pytest.fixture(autouse=True)
def mock_execute_command_safely():
    with patch("src.utils.code_validator.execute_command_safely") as mock_exec:
        # Default successful return for ruff and bandit
        mock_exec.return_value = (0, '{"results": []}', "")  # No issues found
        yield mock_exec


# Mock _get_code_snippet for _run_ruff, _run_bandit, _run_ast_security_checks
@pytest.fixture(autouse=True)
def mock_get_code_snippet():
    with patch("src.utils.code_validator._get_code_snippet") as mock_snippet:
        mock_snippet.return_value = "mock_code_snippet"
        yield mock_snippet


def test_validate_code_output_add_action():
    """Test ADD action with basic content."""
    change = {
        "FILE_PATH": "new_file.py",
        "ACTION": "ADD",
        "FULL_CONTENT": "def hello():\n    print('Hello, world!')\n",
    }
    result = validate_code_output(change)
    assert not any(issue["type"] == "Ruff Linting Issue" for issue in result["issues"])
    assert not any(
        issue["type"] == "Bandit Security Issue" for issue in result["issues"]
    )
    assert not any(
        issue["type"] == "Security Vulnerability (AST)" for issue in result["issues"]
    )


def test_validate_code_output_modify_action_no_change():
    """Test MODIFY action where content is identical."""
    original_content = "def func(): pass"
    change = {
        "FILE_PATH": "existing_file.py",
        "ACTION": "MODIFY",
        "FULL_CONTENT": "def func(): pass",
    }
    result = validate_code_output(change, original_content)
    assert any(issue["type"] == "No Change Detected" for issue in result["issues"])


def test_validate_code_output_remove_action_line_not_found():
    """Test REMOVE action where a line is not in original content."""
    original_content = "line1\nline2\nline3"
    change = {"FILE_PATH": "existing_file.py", "ACTION": "REMOVE", "LINES": ["line4"]}
    result = validate_code_output(change, original_content)
    assert any(
        issue["type"] == "Potential Removal Mismatch" for issue in result["issues"]
    )


def test_run_ruff_detects_linting_issue(mock_execute_command_safely):
    """Test _run_ruff with a known linting issue (e.g., unused import)."""
    content = "import os\ndef func():\n    pass\n"
    filename = "test_lint.py"

    # Configure mock_execute_command_safely to simulate Ruff finding an issue
    mock_execute_command_safely.side_effect = [
        (
            1,
            '[{"code": "F401", "message": "unused-import", "location": {"row": 1, "column": 8}}]',
            "",
        ),  # Linting issue
        (
            1,
            "Would reformat: /tmp/tmp_file.py\n1 file would be reformatted",
            "",
        ),  # Formatting issue
    ]

    issues = _run_ruff(content, filename)
    assert any(issue["code"] == "F401" for issue in issues)  # F401: unused-import
    assert any(issue["type"] == "Ruff Formatting Issue" for issue in issues)


def test_run_bandit_detects_security_issue(mock_execute_command_safely):
    """Test _run_bandit with a known security issue (e.g., hardcoded password)."""
    content = "password = 'hardcoded_secret'\n"
    filename = "test_bandit.py"

    # Configure mock_execute_command_safely to simulate Bandit finding an issue
    mock_execute_command_safely.return_value = (
        1,  # Bandit returns 1 for issues found
        '{"results": [{"test_id": "B105", "severity": "MEDIUM", "description": "Hardcoded password string", "line_number": 1}]}',
        "",
    )

    issues = _run_bandit(content, filename)
    assert any(
        issue["code"] == "B105" for issue in issues
    )  # B105: hardcoded_password_string


def test_run_ast_security_checks_detects_eval():
    """Test _run_ast_security_checks with eval()."""
    content = "eval('1+1')\n"
    filename = "test_ast_eval.py"
    issues = _run_ast_security_checks(content, filename)
    assert any("Use of eval() is discouraged" in issue["message"] for issue in issues)
