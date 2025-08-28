# src/utils/command_executor.py
import subprocess
import shlex
import logging
from typing import Union, List, Tuple, Optional

logger = logging.getLogger(__name__)

def execute_command_safely(
    command: Union[str, List[str]],
    timeout: int = 60,
    cwd: Optional[str] = None,
    check: bool = True, # Added 'check' parameter
    **kwargs
) -> Tuple[str, str]:
    """
    Executes an external command safely using subprocess.run.

    - If `command` is a string, it uses `shlex.split` to properly tokenize it,
      preventing shell injection. `shell=False` is enforced.
    - If `command` is a list, it's passed directly. `shell=False` is enforced.
    - Captures stdout and stderr.
    - Raises subprocess.CalledProcessError if the command returns a non-zero exit code
      and `check` is True.
    - Raises subprocess.TimeoutExpired if the command exceeds the timeout.

    Args:
        command: The command to execute, either as a string or a list of arguments.
        timeout: Maximum time in seconds to wait for the command to complete.
        cwd: The current working directory for the command.
        check: If True, raise CalledProcessError on non-zero exit codes.
               If False, the stdout/stderr will still be returned, and the caller
               is responsible for checking the return code (if needed).
        **kwargs: Additional keyword arguments to pass to `subprocess.run`.

    Returns:
        A tuple (stdout, stderr) of the command's output.

    Raises:
        subprocess.CalledProcessError: If the command returns a non-zero exit code and `check` is True.
        subprocess.TimeoutExpired: If the command times out.
        ValueError: If the command is empty or invalid.
        FileNotFoundError: If the command executable is not found.
    """
    if not command:
        raise ValueError("Command cannot be empty.")

    if isinstance(command, str):
        # Use shlex.split to safely tokenize the command string
        # This prevents shell injection vulnerabilities
        processed_command = shlex.split(command)
        logger.debug(f"Safely split command string: '{command}' into {processed_command}")
    elif isinstance(command, list):
        processed_command = command
        logger.debug(f"Using command as list: {processed_command}")
    else:
        raise ValueError("Command must be a string or a list of strings.")

    try:
        result = subprocess.run(
            processed_command,
            capture_output=True,
            text=True,
            check=check,  # Use the 'check' parameter
            shell=False, # Explicitly disable shell for security
            timeout=timeout,
            cwd=cwd,
            **kwargs
        )
        logger.info(f"Command '{' '.join(processed_command)}' executed successfully (check={check}).")
        return result.stdout.strip(), result.stderr.strip()
    except subprocess.CalledProcessError as e:
        logger.error(f"Command '{' '.join(processed_command)}' failed with exit code {e.returncode}. Stderr: {e.stderr.strip()}")
        raise e
    except subprocess.TimeoutExpired as e:
        logger.error(f"Command '{' '.join(processed_command)}' timed out after {timeout} seconds.")
        raise e
    except FileNotFoundError:
        logger.error(f"Command '{processed_command[0]}' not found. Ensure it's in PATH.")
        raise FileNotFoundError(f"Command '{processed_command[0]}' not found.")
    except Exception as e:
        logger.error(f"An unexpected error occurred while executing command '{' '.join(processed_command)}': {e}", exc_info=True)
        raise e
