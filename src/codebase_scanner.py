# src/codebase_scanner.py
import os
import logging
from pathlib import Path
from typing import Dict, Any, List
import fnmatch

logger = logging.getLogger(__name__)


class CodebaseScanner:
    """Scans and analyzes the project's codebase to provide context for self-improvement."""

    def __init__(self, project_root: str = "."):
        self.project_root = Path(project_root).resolve()
        if not self.project_root.exists():
            raise ValueError(f"Project root {self.project_root} does not exist")
        logger.info(
            f"CodebaseScanner initialized for project root: {self.project_root}"
        )

    def load_own_codebase_context(self) -> Dict[str, Any]:
        """Load the entire codebase context for self-analysis."""
        try:
            file_structure = {}
            raw_file_contents = {}

            exclude_patterns = [
                ".git/",
                "__pycache__/",
                "venv/",
                ".venv/",
                "node_modules/",
                "*.pyc",
                "*.log",
                "*.sqlite3",
                "*.db",
                "*.DS_Store",
                "data/",
                "repo_contents.txt",
                "repo_to_single_file.sh",
                ".env",
                "*.bak",
            ]

            for root, dirs, files in os.walk(self.project_root, topdown=True):
                # Modify dirs in-place to prune the search
                dirs[:] = [
                    d
                    for d in dirs
                    if not any(
                        p.endswith("/") and p.strip("/") in d for p in exclude_patterns
                    )
                ]

                root_path = Path(root)

                for file in files:
                    relative_file_path = root_path / file
                    if any(
                        fnmatch.fnmatch(
                            str(relative_file_path.relative_to(self.project_root)), p
                        )
                        for p in exclude_patterns
                    ):
                        continue

                    try:
                        with open(
                            relative_file_path, "r", encoding="utf-8", errors="ignore"
                        ) as f:
                            content = f.read()
                            if len(content) > 1000000:  # 1MB limit
                                logger.debug(
                                    f"Skipping large file {relative_file_path}"
                                )
                                continue

                            relative_path_str = str(
                                relative_file_path.relative_to(self.project_root)
                            )
                            raw_file_contents[relative_path_str] = content

                            relative_dir = str(root_path.relative_to(self.project_root))
                            if relative_dir == ".":
                                relative_dir = ""

                            if relative_dir not in file_structure:
                                file_structure[relative_dir] = []
                            file_structure[relative_dir].append(file)

                    except (FileNotFoundError, PermissionError) as e:
                        logger.debug(f"Skipping file {relative_file_path}: {str(e)}")

            return {
                "file_structure": file_structure,
                "raw_file_contents": raw_file_contents,
            }
        except Exception as e:
            logger.error(f"Error loading codebase context: {str(e)}", exc_info=True)
            raise


def get_codebase_scanner_instance() -> CodebaseScanner:
    """Factory function to get a properly configured CodebaseScanner instance."""
    from src.utils.path_utils import PROJECT_ROOT

    return CodebaseScanner(str(PROJECT_ROOT))
