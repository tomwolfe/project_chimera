# tests/test_command_executor.py

import unittest
from unittest.mock import patch, MagicMock
import subprocess
import sys
from pathlib import Path

# Assuming src/utils/command_executor.py contains the execute_command_safely function
from src.utils.command_executor import execute_command_safely

class TestCommandExecutor(unittest.TestCase):
    """Unit tests for the execute_command_safely utility function."""

    @patch('sys.executable', sys.executable) # Ensure sys.executable is available for patching
    @patch('subprocess.run')
    def setUp(self, mock_run):
        """Set up mocks for subprocess.run and sys.executable for all tests."""
        self.mock_run = mock_run
        # Mock sys.executable to a predictable path for testing the command adjustment logic
        self.original_sys_executable = sys.executable
        sys.executable = '/fake/python/executable' # Use a fake path for consistent testing

        # Mock the return value of subprocess.run for successful calls
        self.mock_run.return_value = MagicMock(returncode=0, stdout="Success output", stderr="")

    def tearDown(self):
        """Restore sys.executable after tests."""
        sys.executable = self.original_sys_executable

    def test_execute_command_safely_success(self):
        """Test successful execution of a simple command."""
        command = ["echo", "hello"]
        return_code, stdout, stderr = execute_command_safely(command)

        self.mock_run.assert_called_once_with(
            command,
            capture_output=True,
            text=True,
            check=False,
            shell=False,
            timeout=60
        )
        self.assertEqual(return_code, 0)
        self.assertEqual(stdout, "Success output")
        self.assertEqual(stderr, "")

    def test_execute_command_safely_failure_no_stderr(self):
        """Test command failure with no stderr output."""
        command = ["false_command"]
        self.mock_run.return_value = MagicMock(returncode=1, stdout="", stderr="")

        return_code, stdout, stderr = execute_command_safely(command)

        self.mock_run.assert_called_once_with(
            command,
            capture_output=True,
            text=True,
            check=False,
            shell=False,
            timeout=60
        )
        self.assertEqual(return_code, 1)
        self.assertEqual(stdout, "")
        self.assertEqual(stderr, "")

    def test_execute_command_safely_failure_with_stderr(self):
        """Test command failure with stderr output."""
        command = ["error_command"]
        error_message = "This is an error message."
        self.mock_run.return_value = MagicMock(returncode=1, stdout="", stderr=error_message)

        return_code, stdout, stderr = execute_command_safely(command)

        self.mock_run.assert_called_once_with(
            command,
            capture_output=True,
            text=True,
            check=False,
            shell=False,
            timeout=60
        )
        self.assertEqual(return_code, 1)
        self.assertEqual(stdout, "")
        self.assertEqual(stderr, error_message)

    def test_execute_command_safely_timeout(self):
        """Test command timeout."""
        command = ["sleep", "10"]
        # Mock subprocess.run to raise TimeoutExpired
        self.mock_run.side_effect = subprocess.TimeoutExpired(cmd=command, timeout=1)

        with pytest.raises(subprocess.TimeoutExpired):
            execute_command_safely(command, timeout=1)
        self.mock_run.assert_called_once_with(
            command,
            capture_output=True,
            text=True,
            check=False,
            shell=False,
            timeout=1
        )

    def test_execute_command_safely_check_true_success(self):
        """Test check=True with a successful command."""
        command = ["echo", "success"]
        self.mock_run.return_value = MagicMock(returncode=0, stdout="Success", stderr="")

        return_code, stdout, stderr = execute_command_safely(command, check=True)

        self.mock_run.assert_called_once_with(
            command,
            capture_output=True,
            text=True,
            check=True,
            shell=False,
            timeout=60
        )
        self.assertEqual(return_code, 0)
        self.assertEqual(stdout, "Success")

    def test_execute_command_safely_check_true_failure(self):
        """Test check=True with a failing command."""
        command = ["false_command"]
        # Mock subprocess.run to raise CalledProcessError
        self.mock_run.side_effect = subprocess.CalledProcessError(
            returncode=1, cmd=command, stderr="Error message"
        )

        with pytest.raises(subprocess.CalledProcessError):
            execute_command_safely(command, check=True)
        self.mock_run.assert_called_once_with(
            command,
            capture_output=True,
            text=True,
            check=True,
            shell=False,
            timeout=60
        )

    def test_execute_command_safely_python_tool_invocation(self):
        """Test that python tools are correctly prepended with `sys.executable -m`."""
        # Test with 'pytest'
        with patch('subprocess.run') as mock_run:
            mock_run.return_value = MagicMock(returncode=0, stdout="Pytest output", stderr="")
            execute_command_safely(["pytest", "tests/"])
            mock_run.assert_called_once_with(
                [sys.executable, "-m", "pytest", "tests/"],
                capture_output=True, text=True, check=False, shell=False, timeout=60
            )

        # Test with 'ruff'
        with patch('subprocess.run') as mock_run:
            mock_run.return_value = MagicMock(returncode=0, stdout="Ruff output", stderr="")
            execute_command_safely(["ruff", "check", "."])
            mock_run.assert_called_once_with(
                [sys.executable, "-m", "ruff", "check", "."],
                capture_output=True, text=True, check=False, shell=False, timeout=60
            )

        # Test with 'bandit'
        with patch('subprocess.run') as mock_run:
            mock_run.return_value = MagicMock(returncode=0, stdout="Bandit output", stderr="")
            execute_command_safely(["bandit", "-r", "."])
            mock_run.assert_called_once_with(
                [sys.executable, "-m", "bandit", "-r", "."],
                capture_output=True, text=True, check=False, shell=False, timeout=60
            )

    def test_execute_command_safely_already_python_m(self):
        """Test that `python -m` is not double-prepended."""
        command = [sys.executable, "-m", "ruff", "check", "."]
        with patch('subprocess.run') as mock_run:
            mock_run.return_value = MagicMock(returncode=0, stdout="Ruff output", stderr="")
            execute_command_safely(command)
            mock_run.assert_called_once_with(
                command, # Should be called with the original command, not modified
                capture_output=True, text=True, check=False, shell=False, timeout=60
            )

    def test_execute_command_safely_non_python_command(self):
        """Test that non-Python commands are not prepended with sys.executable -m."""
        command = ["echo", "hello"]
        with patch('subprocess.run') as mock_run:
            mock_run.return_value = MagicMock(returncode=0, stdout="Echo output", stderr="")
            execute_command_safely(command)
            mock_run.assert_called_once_with(
                command, # Should be called with the original command
                capture_output=True, text=True, check=False, shell=False, timeout=60
            )

    def test_execute_command_safely_error_handling_for_non_subprocess_errors(self):
        """Test that other exceptions during subprocess.run are caught and logged."""
        command = ["some_command"]
        # Mock subprocess.run to raise a different exception
        self.mock_run.side_effect = OSError("Permission denied")

        with patch('logging.getLogger') as mock_logger:
            mock_logger_instance = MagicMock()
            mock_logger.return_value = mock_logger_instance
            
            with pytest.raises(OSError): # Expect the original exception to be re-raised
                execute_command_safely(command)
            
            mock_logger_instance.error.assert_called_once_with(
                f"An error occurred while executing command: Permission denied",
                exc_info=True # Ensure exc_info is passed
            )
            self.mock_run.assert_called_once_with(
                command,
                capture_output=True,
                text=True,
                check=False,
                shell=False,
                timeout=60
            )
