# src/utils/command_executor.py
import logging
import subprocess
# shlex is no longer needed if command_parts is always a list
from typing import List, Tuple, Union # Import for type hints

logger = logging.getLogger(__name__)

def execute_system_command( # Renamed from execute_command_safely
    command_parts: List[str], # Changed type from str to List[str]
    timeout: int = 60, # Added timeout parameter with default
    check: bool = True # Added check parameter with default
) -> Tuple[str, str]: # Returns stdout, stderr
    """
    Executes a system command given as a list of parts, safely.
    
    Args:
        command_parts: The command and its arguments as a list of strings.
        timeout: The maximum time in seconds to wait for the command to complete.
        check: If True, raises subprocess.CalledProcessError if the command returns a non-zero exit code.
        
    Returns:
        A tuple containing the standard output and standard error of the command.
        
    Raises:
        subprocess.CalledProcessError: If the command returns a non-zero exit code and check is True.
        FileNotFoundError: If the command executable is not found.
        Exception: For other unexpected errors during execution.
    """
    try:
        # command_parts is already a list, so no need for shlex.split
        result = subprocess.run(
            command_parts,
            capture_output=True,
            text=True,
            check=check,
            shell=False, # Always use shell=False for security when command_parts is a list
            timeout=timeout
        )
        
        logger.info(f"Successfully executed command: {' '.join(command_parts)}")
        return result.stdout, result.stderr
        
    except FileNotFoundError:
        logger.error(f"Command not found: {command_parts[0]}")
        raise FileNotFoundError(f"The command '{command_parts[0]}' was not found. Ensure it is installed and in your system's PATH.")
    except subprocess.TimeoutExpired as e:
        logger.error(f"Command execution timed out after {e.timeout} seconds: {' '.join(command_parts)}. Stderr: {e.stderr.strip()}")
        raise e
    except subprocess.CalledProcessError as e:
        logger.error(f"Command execution failed with exit code {e.returncode}: {' '.join(command_parts)}. Stderr: {e.stderr.strip()}")
        raise e
    except Exception as e:
        logger.error(f"An unexpected error occurred during command execution: {e}", exc_info=True)
        raise e