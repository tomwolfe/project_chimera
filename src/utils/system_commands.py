# src/utils/system_commands.py
import logging
import subprocess
from typing import Optional

logger = logging.getLogger(__name__)


def execute_command(command_list) -> Optional[subprocess.CompletedProcess]:
    """
    Safely execute a command with error handling.
    Returns None on failure.
    """
    try:
        result = subprocess.run(
            command_list,
            capture_output=True,
            text=True,
            shell=False,
            check=True,  # Raises CalledProcessError if command fails
        )
        return result
    except subprocess.CalledProcessError as e:
        print(f"Command failed: {e}")
        return None
    except Exception as e:
        print(f"Unexpected error executing command: {e}")
        return None


def ping_host(host: str) -> Optional[str]:
    """Ping a host with sanitized input and safe subprocess execution."""
    # Basic sanitization: remove potentially harmful characters
    sanitized_host = (
        host.replace("`", "").replace(";", "").replace("&", "").replace("|", "")
    )
    command = ["ping", "-c", "1", sanitized_host]
    result = execute_command(command)
    return result.stdout if result else None
