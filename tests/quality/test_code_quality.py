import subprocess
import sys

from src.utils.core_helpers.path_utils import PROJECT_ROOT


def test_ruff_linting_passes():
    """Ensures the entire codebase passes ruff linting checks."""
    command = [sys.executable, "-m", "ruff", "check", str(PROJECT_ROOT)]
    result = subprocess.run(command, check=False, capture_output=True, text=True)
    assert result.returncode == 0, (
        f"Ruff linting failed with errors:\\n{result.stdout}\\n{result.stderr}"
    )


def test_ruff_formatting_passes():
    """Ensures the entire codebase passes ruff formatting checks."""
    command = [sys.executable, "-m", "ruff", "format", "--check", str(PROJECT_ROOT)]
    result = subprocess.run(command, check=False, capture_output=True, text=True)
    assert result.returncode == 0, (
        f"Ruff formatting check failed:\\n{result.stdout}\\n{result.stderr}"
    )
