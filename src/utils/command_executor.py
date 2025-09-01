# src/utils/command_executor.py
import logging
import subprocess
import shlex # Import shlex for safe command splitting

logger = logging.getLogger(__name__)

def execute_command_safely(command_string: str) -> str:
    """
    Executes a system command safely, avoiding shell injection.
    
    Args:
        command_string: The command to execute as a single string.
        
    Returns:
        The standard output of the command.
        
    Raises:
        subprocess.CalledProcessError: If the command returns a non-zero exit code.
        FileNotFoundError: If the command executable is not found.
        Exception: For other unexpected errors during execution.
    """
    try:
        # WARNING: Using shell=True can be a security risk if command_string is not properly sanitized.
        # The LLM's suggestion was to avoid shell=True and split the command.
        # We will implement that suggestion here.
        
        # Split the command string into a list of arguments safely
        # This is crucial to prevent shell injection vulnerabilities.
        command_parts = shlex.split(command_string) 
        
        # Execute the command using subprocess.run with shell=False
        # capture_output=True captures stdout and stderr
        # text=True decodes stdout and stderr as text
        # check=True raises CalledProcessError if the command returns a non-zero exit code
        result = subprocess.run(command_parts, capture_output=True, text=True, check=True, shell=False)
        
        logger.info(f"Successfully executed command: {' '.join(command_parts)}")
        return result.stdout
        
    except FileNotFoundError:
        logger.error(f"Command not found: {command_parts[0]}")
        raise FileNotFoundError(f"The command '{command_parts[0]}' was not found. Ensure it is installed and in your system's PATH.")
    except subprocess.CalledProcessError as e:
        logger.error(f"Command execution failed: {e}. Stderr: {e.stderr.strip()}")
        # Re-raise the exception to be handled by the caller
        raise e
    except Exception as e:
        logger.error(f"An unexpected error occurred during command execution: {e}", exc_info=True)
        # Re-raise any other unexpected exceptions
        raise e