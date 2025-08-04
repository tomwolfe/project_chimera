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
    
    # The unified_diff generator yields lines starting with '---', '+++', '@@', '+', '-', ' '.
    # We want to return the diff content, typically excluding the header lines for direct display.
    # However, for a git-like representation, keeping them might be useful.
    # Let's return the full diff including headers for clarity.
    return "".join(list(diff))

# Example usage (for demonstration):
# if __name__ == '__main__':
#     original = "line 1\nline 2\nline 3"
#     modified = "line 1\nline 2 modified\nline 3\nnew line 4"
#     print(format_git_diff(original, modified))
