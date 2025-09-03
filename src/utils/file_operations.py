# src/utils/file_operations.py
"""
Utility functions for file operations like backup and applying code changes.
"""

import logging
import shutil
import os
import subprocess
import re
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple

logger = logging.getLogger(__name__)


def _create_file_backup(file_path: Path) -> Optional[Path]:
    """
    Creates a timestamped backup of a file in a '.chimera_backups' directory
    within the file's parent directory.

    Args:
        file_path: The Path object of the file to back up.

    Returns:
        The Path object of the created backup file, or None if the file doesn't exist
        or backup creation fails.
    """
    if not file_path.exists():
        logger.warning(f"Backup skipped: File not found at {file_path}")
        return None

    try:
        backup_dir = file_path.parent / ".chimera_backups"
        backup_dir.mkdir(exist_ok=True)  # Create backup directory if it doesn't exist

        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        backup_filename = f"{file_path.name}.{timestamp}.bak"
        backup_path = backup_dir / backup_filename

        shutil.copy2(file_path, backup_path)  # Use copy2 to preserve metadata
        logger.info(f"Created backup of {file_path} at {backup_path}")
        return backup_path
    except OSError as e:
        logger.error(f"Failed to create backup for {file_path}: {e}")
        return None
    except Exception as e:
        logger.error(f"An unexpected error occurred during backup of {file_path}: {e}")
        return None


def _apply_code_change(change: Dict[str, Any], codebase_path: Path):
    """
    Applies a single code change (ADD, MODIFY, REMOVE) to the codebase.

    Args:
        change: A dictionary describing the change, containing 'FILE_PATH', 'ACTION',
                and either 'FULL_CONTENT' or 'LINES'/'DIFF_CONTENT' depending on action.
        codebase_path: The root path of the codebase where changes should be applied.
    """
    file_path = codebase_path / change["FILE_PATH"]
    action = change["ACTION"]

    if action == "ADD":
        try:
            file_path.parent.mkdir(parents=True, exist_ok=True)
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(change["FULL_CONTENT"])
            logger.info(f"Added file: {file_path}")
        except OSError as e:
            logger.error(f"Failed to add file {file_path}: {e}")
            raise  # Re-raise to indicate failure
    elif action == "MODIFY":
        if file_path.exists():
            _create_file_backup(file_path)  # Backup before modification
            if change.get("FULL_CONTENT") is not None:
                try:
                    with open(file_path, "w", encoding="utf-8") as f:
                        f.write(change["FULL_CONTENT"])
                    logger.info(f"Modified file: {file_path} with FULL_CONTENT.")
                except OSError as e:
                    logger.error(f"Failed to write FULL_CONTENT to {file_path}: {e}")
                    raise
            elif change.get("DIFF_CONTENT"):
                # Applying diff content requires a patch utility.
                # This is a placeholder; a real implementation would use subprocess
                # to call 'patch' or a Python library like 'patch'.
                # For now, we log the intent and skip actual diff application.
                if change["DIFF_CONTENT"].strip():
                    logger.info(
                        f"Applying diff content to file: {file_path} (requires patch utility implementation)."
                    )
                    # Example placeholder for applying diff:
                    # try:
                    #     original_content = file_path.read_text()
                    #     patched_content = apply_unified_diff(original_content, change["DIFF_CONTENT"]) # Assuming apply_unified_diff exists
                    #     with open(file_path, "w", encoding="utf-8") as f:
                    #         f.write(patched_content)
                    #     logger.info(f"Successfully applied diff to {file_path}")
                    # except Exception as e:
                    #     logger.error(f"Failed to apply diff to {file_path}: {e}")
                    #     raise
                else:
                    logger.warning(
                        f"MODIFY action for {file_path} provided DIFF_CONTENT but it was empty or whitespace."
                    )
            else:
                logger.warning(
                    f"MODIFY action for {file_path} provided neither FULL_CONTENT nor DIFF_CONTENT. No change applied."
                )
        else:
            logger.warning(f"Attempted to modify non-existent file: {file_path}")
            # Optionally raise an error here if modifying non-existent files should be critical
            # raise FileNotFoundError(f"Attempted to modify non-existent file: {file_path}")
    elif action == "REMOVE":
        if file_path.exists():
            _create_file_backup(file_path)  # Backup before removal
            try:
                file_path.unlink()
                logger.info(f"Removed file: {file_path}")
            except OSError as e:
                logger.error(f"Failed to remove file {file_path}: {e}")
                raise  # Re-raise to indicate failure
        else:
            logger.warning(f"Attempted to remove non-existent file: {file_path}")
            # Optionally raise an error here if removing non-existent files should be critical
            # raise FileNotFoundError(f"Attempted to remove non-existent file: {file_path}")
    else:
        logger.error(f"Unknown action type '{action}' for file {file_path}.")
        raise ValueError(f"Unknown action type: {action}")
