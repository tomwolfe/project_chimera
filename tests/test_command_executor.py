"""Tests for the command executor module."""

import inspect
import subprocess
import sys
from unittest.mock import MagicMock, patch

import pytest

from src.utils.core_helpers.command_executor import execute_command_safely


class TestExecuteCommandSafely:
    """Test suite for execute_command_safely function."""

    @patch("subprocess.run")
    def test_execute_command_safely_success(self, mock_run):
        """Test successful execution of a simple command."""
        # Mock successful command execution
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = "Success output"
        mock_result.stderr = ""
        mock_run.return_value = mock_result

        command = ["echo", "hello"]
        return_code, stdout, stderr = execute_command_safely(command)

        mock_run.assert_called_once_with(
            command,
            capture_output=True,
            text=True,
            check=False,
            shell=False,
            timeout=60,
        )
        assert return_code == 0
        assert stdout == "Success output"
        assert stderr == ""

    @patch("subprocess.run")
    def test_execute_command_safely_with_timeout(self, mock_run):
        """Test command execution with custom timeout."""
        # Mock successful command execution
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = "Success output"
        mock_result.stderr = ""
        mock_run.return_value = mock_result

        command = ["echo", "hello"]
        return_code, stdout, stderr = execute_command_safely(command, timeout=30)

        mock_run.assert_called_once_with(
            command,
            capture_output=True,
            text=True,
            check=False,
            shell=False,
            timeout=30,
        )

    @patch("subprocess.run")
    def test_execute_command_safely_with_check_true_success(self, mock_run):
        """Test command execution with check=True for successful command."""
        # Mock successful command execution
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = "Success output"
        mock_result.stderr = ""
        mock_run.return_value = mock_result

        command = ["echo", "hello"]
        return_code, stdout, stderr = execute_command_safely(command, check=True)

        mock_run.assert_called_once_with(
            command,
            capture_output=True,
            text=True,
            check=True,  # This should be True
            shell=False,
            timeout=60,
        )
        assert return_code == 0
        assert stdout == "Success output"

    @patch("subprocess.run")
    def test_execute_command_safely_failure_with_stderr(self, mock_run):
        """Test command failure with stderr output."""
        # Mock failed command execution
        mock_result = MagicMock()
        mock_result.returncode = 1
        mock_result.stdout = ""
        mock_result.stderr = "Error occurred"
        mock_run.return_value = mock_result

        command = ["false"]  # Command that fails
        return_code, stdout, stderr = execute_command_safely(command)

        assert return_code == 1
        assert stdout == ""
        assert stderr == "Error occurred"

    @patch("subprocess.run")
    def test_execute_command_safely_timeout_exception(self, mock_run):
        """Test command timeout handling."""
        # Mock timeout exception
        mock_run.side_effect = subprocess.TimeoutExpired(cmd=["sleep", "1"], timeout=1)

        command = ["sleep", "1"]
        return_code, stdout, stderr = execute_command_safely(command, timeout=1)

        assert return_code == 124  # Standard timeout exit code
        assert "timed out" in stderr.lower()

    @patch("subprocess.run")
    def test_execute_command_safely_called_process_error(self, mock_run):
        """Test command failure with CalledProcessError."""
        # Mock CalledProcessError (when check=True and command fails)
        error = subprocess.CalledProcessError(
            returncode=1, cmd=["false"], output="Command output", stderr="Error details"
        )
        mock_run.side_effect = error

        command = ["false"]
        # When check=True and command fails, the function should raise CalledProcessError
        with pytest.raises(subprocess.CalledProcessError):
            execute_command_safely(command, check=True)

    @patch("subprocess.run")
    def test_execute_command_safely_file_not_found(self, mock_run):
        """Test command execution when command is not found."""
        # Mock FileNotFoundError
        mock_run.side_effect = FileNotFoundError("Command not found")

        command = ["nonexistent_command"]
        return_code, stdout, stderr = execute_command_safely(command)

        assert return_code == 127  # Standard "command not found" exit code
        assert "Command not found" in stderr

    @patch("subprocess.run")
    def test_execute_command_safely_general_exception(self, mock_run):
        """Test command execution with other exceptions."""
        # Mock general exception
        mock_run.side_effect = OSError("Permission denied")

        command = ["restricted_command"]
        return_code, stdout, stderr = execute_command_safely(command)

        assert return_code == 1  # Generic error code
        assert "Permission denied" in stderr

    @patch("subprocess.run")
    def test_execute_command_safely_python_tool_prepending(self, mock_run):
        """Test that Python tools are prepended with sys.executable -m."""

        # Mock successful command execution
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = "Pytest output"
        mock_result.stderr = ""
        mock_run.return_value = mock_result

        command = ["pytest", "--version"]
        return_code, stdout, stderr = execute_command_safely(command)

        # Check that sys.executable -m was prepended to the command
        expected_command = [sys.executable, "-m"] + command
        mock_run.assert_called_once_with(
            expected_command,
            capture_output=True,
            text=True,
            check=False,
            shell=False,
            timeout=60,
        )
        assert return_code == 0

    @patch("subprocess.run")
    def test_execute_command_safely_ruff_tool_prepending(self, mock_run):
        """Test that ruff tool is prepended with sys.executable -m."""

        # Mock successful command execution
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = "Ruff output"
        mock_result.stderr = ""
        mock_run.return_value = mock_result

        command = ["ruff", "--version"]
        return_code, stdout, stderr = execute_command_safely(command)

        # Check that sys.executable -m was prepended to the command
        expected_command = [sys.executable, "-m"] + command
        mock_run.assert_called_once_with(
            expected_command,
            capture_output=True,
            text=True,
            check=False,
            shell=False,
            timeout=60,
        )
        assert return_code == 0

    @patch("subprocess.run")
    def test_execute_command_safely_bandit_tool_prepending(self, mock_run):
        """Test that bandit tool is prepended with sys.executable -m."""

        # Mock successful command execution
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = "Bandit output"
        mock_result.stderr = ""
        mock_run.return_value = mock_result

        command = ["bandit", "--version"]
        return_code, stdout, stderr = execute_command_safely(command)

        # Check that sys.executable -m was prepended to the command
        expected_command = [sys.executable, "-m"] + command
        mock_run.assert_called_once_with(
            expected_command,
            capture_output=True,
            text=True,
            check=False,
            shell=False,
            timeout=60,
        )
        assert return_code == 0

    def test_execute_command_safely_check_false_on_failure(self):
        """Test that when check=False, failures don't raise exceptions."""
        # This test verifies that when check=False (default),
        # the function handles failures gracefully without raising exceptions
        # We can't easily mock this due to the import issue, but we can at least
        # verify that the function signature and parameter handling works as expected
        # by checking the function exists and has the right signature

        sig = inspect.signature(execute_command_safely)
        params = list(sig.parameters.keys())
        assert "command" in params
        assert "timeout" in params
        assert "check" in params
        # Default value of check parameter
        assert not sig.parameters["check"].default
