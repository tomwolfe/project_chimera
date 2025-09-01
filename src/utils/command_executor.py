# src/utils/command_executor.py
import subprocess
import logging
import sys # Added for sys.executable if needed, though not used in the provided snippet

logger = logging.getLogger(__name__)

def execute_command_safely(command: list[str], timeout: int = 60) -> tuple[int, str, str]:
    """Executes a shell command safely, capturing output and errors.

    Args:
        command: A list of strings representing the command and its arguments.
        timeout: The maximum time in seconds to wait for the command to complete.

    Returns:
        A tuple containing the return code, stdout, and stderr.

    Raises:
        subprocess.TimeoutExpired: If the command exceeds the timeout.
        Exception: For other errors during command execution.
    """
    try:
        logger.info(f"Executing command: {' '.join(command)}")
        process = subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=False,  # Do not raise CalledProcessError automatically
            timeout=timeout
        )

        if process.returncode != 0:
            logger.error(
                f"Command failed with exit code {process.returncode}:\nSTDOUT:\n{process.stdout}\nSTDERR:\n{process.stderr}"
            )
        else:
            logger.info(
                f"Command executed successfully. STDOUT:\n{process.stdout}"
            )

        return process.returncode, process.stdout, process.stderr

    except subprocess.TimeoutExpired:
        logger.error(f"Command timed out after {timeout} seconds: {' '.join(command)}")
        raise
    except Exception as e:
        logger.error(f"An error occurred while executing command: {e}")
        raise