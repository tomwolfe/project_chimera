# src/utils/git_diff_formatter.py
import difflib
from typing import List, Dict, Any

class GitDiffError(Exception):
    """Custom exception for Git diff formatting errors."""
    pass

def format_git_diff(original_content: str, new_content: str) -> str:
    """Creates a git-style unified diff from original and new content.

    Args:
        original_content: The original content of the file.
        new_content: The new content of the file.

    Returns:
        A formatted string representing the Git diff.

    Raises:
        GitDiffError: If formatting fails.
    """
    original_lines = original_content.splitlines(keepends=True)
    new_lines = new_content.splitlines(keepends=True)

    # Use difflib.unified_diff to generate the diff
    diff = difflib.unified_diff(
        original_lines,
        new_lines,
        fromfile='a/original', # Standard git diff headers
        tofile='b/modified',
        lineterm='' # Prevent adding extra newlines if already present in lines
    )

    return "".join(list(diff))
