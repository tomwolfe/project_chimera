# src/utils/command_executor.py
import subprocess
import logging
import sys # NEW: Import sys

logger = logging.getLogger(__name__)


# ADDED: 'check' parameter to the function signature
def execute_command_safely(
    command: list[str], timeout: int = 60, check: bool = False
) -> tuple[int, str, str]:
    """Executes a shell command safely, capturing output and errors.

    Args:
        command: A list of strings representing the command and its arguments.
        timeout: The maximum time in seconds to wait for the command to complete.
        check: If True, raise a CalledProcessError if the command returns a non-zero exit code.

    Returns:
        A tuple containing the return code, stdout, and stderr.

    Raises:
        subprocess.TimeoutExpired: If the command exceeds the timeout.
        subprocess.CalledProcessError: If check is True and the command returns a non-zero exit code.
        Exception: For other errors during command execution.
    """
    try:
        # NEW: Prepend sys.executable -m if the command starts with a Python module name.
        # This ensures the command runs with the same Python interpreter as the current process,
        # which is crucial for virtual environments and consistent tool execution.
        # MODIFIED: Added check `and not command[0].startswith(sys.executable)` to prevent double-prepending
        if command and command[0] in ['pytest', 'ruff', 'bandit'] and not command[0].startswith(sys.executable):
            command.insert(0, '-m')
            command.insert(0, sys.executable)
            logger.debug(f"Adjusted command to use sys.executable: {command}")
        # Ensure shell=False for security when passing a list of arguments
        logger.info(f"Executing command: {command}")
        process = subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=check,  # Pass the 'check' argument to subprocess.run
            shell=False,  # Explicitly set to False for security
            timeout=timeout,
        )

        if process.returncode != 0:
            logger.error(f"Command failed: {command}")
            if process.stderr:
                logger.error(f"Stderr: {process.stderr}")
            else:
                # NEW: Added more context when stderr is empty but command failed
                logger.error(f"Command failed with non-zero exit code {process.returncode}, but stderr was empty. This might indicate an environment or execution path issue.")
        else:
            logger.info(f"Command executed successfully. STDOUT:\n{process.stdout}")

        return process.returncode, process.stdout, process.stderr

    except subprocess.TimeoutExpired:
        logger.error(f"Command timed out after {timeout} seconds: {' '.join(command)}")
        raise
    except subprocess.CalledProcessError:  # Added this specific exception handler
        logger.error(
            f"Command failed with non-zero exit code (check=True): {' '.join(command)}"
        )
        raise
    except Exception as e:
        logger.error(f"An error occurred while executing command: {e}", exc_info=True)
        raise