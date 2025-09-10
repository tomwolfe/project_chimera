# src/utils/command_executor.py
import subprocess
import logging
import sys # NEW: Import sys

logger = logging.getLogger(__name__)


def execute_command_safely(
    command: list[str], timeout: int = 60, check: bool = False
) -> tuple[int, str, str]:
    stdout_output = ""
    stderr_output = ""
    return_code = -1 # Default to an error code
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
        # FIX: Prepend sys.executable -m if the command is a known Python tool.
        # This ensures the command runs with the same Python interpreter as the current process,
        # which is crucial for virtual environments and consistent tool execution.
        # The check `not command[0] == sys.executable` prevents double-prepending.
        if command and command[0] in ['pytest', 'ruff', 'bandit'] and not command[0] == sys.executable:
            command.insert(0, '-m')
            command.insert(0, sys.executable)
            logger.debug(f"Adjusted command to use sys.executable: {command}")
        # Ensure shell=False for security when passing a list of arguments
        logger.info(f"Executing command: {command}")
        process = subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=False, # Always set to False here, handle check logic manually below
            shell=False,  # Explicitly set to False for security
            timeout=timeout,
        )
        return_code = process.returncode
        stdout_output = process.stdout.strip()
        stderr_output = process.stderr.strip()

        if return_code != 0:
            logger.error(f"Command failed with exit code {return_code}: {command}. Stderr: {stderr_output}. Stdout: {stdout_output}")
            if check: # If check was True, re-raise as CalledProcessError
                raise subprocess.CalledProcessError(return_code, command, stdout=stdout_output, stderr=stderr_output)
        else:
            logger.info(f"Command executed successfully. STDOUT:\n{stdout_output}")

    except subprocess.TimeoutExpired as e:
        logger.error(f"Command timed out after {timeout} seconds: {' '.join(command)}")
        if check:
            raise # Re-raise if check is True
        return_code = 124 # Standard timeout exit code
        stderr_output = f"Command timed out: {e}"
    except subprocess.CalledProcessError as e:
        # This block is now primarily for when check=True was passed to subprocess.run directly,
        # but we've set check=False above. So this block might not be hit.
        # However, if it is, we handle it.
        logger.error(f"Command failed with non-zero exit code (check=True): {e.returncode}. Stderr: {e.stderr.strip()}")
        if check:
            raise
        return_code = e.returncode
        stdout_output = e.stdout.strip()
        stderr_output = e.stderr.strip()
    except FileNotFoundError as e:
        logger.error(f"Command not found: {command[0]}. Error: {e}", exc_info=True)
        if check:
            raise
        return_code = 127 # Standard command not found exit code
        stderr_output = f"Command not found: {command[0]}. Error: {e}"
    except Exception as e:
        logger.error(f"An error occurred while executing command: {e}", exc_info=True)
        if check:
            raise
        return_code = 1 # Generic error
        stderr_output = f"Unexpected error: {e}"
    
    return return_code, stdout_output, stderr_output