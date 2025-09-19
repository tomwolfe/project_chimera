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
        """Loads Project Chimera's own codebase context for self-analysis, including raw file contents."""
        project_root_path = Path(self.project_root)
        if not project_root_path.exists():
            logger.error(
                f"Project root directory not found at {project_root_path} for self-analysis"
            )
            raise RuntimeError(
                "Project root not found. Self-analysis requires access to the codebase. "
                "Ensure the application is running from within the Project Chimera directory."
            )

        self.logger.info(
            f"CodebaseScanner.load_own_codebase_context: Resolved project root to {project_root_path}"
        )
        self._validate_project_structure(project_root_path)

        # Call scan_codebase to get the full structured context including raw file contents
        full_context = self.scan_codebase()
        num_files = len(full_context.get("raw_file_contents", {}))
        self.logger.info(
            f"CodebaseScanner.load_own_codebase_context: Scanned {num_files} files for self-analysis."
        )

        return full_context

    def _find_project_root(self) -> Optional[Path]:
        """Determine the root directory of the current Project Chimera instance."""
        # Use the dynamically determined PROJECT_ROOT
        return PROJECT_ROOT

    @staticmethod
    def _validate_project_structure(project_root: Path) -> None:
        """Validates critical project structure elements for self-analysis."""
        required_files = [
            "pyproject.toml",
            "personas.yaml",
            "src/__init__.py",
            "core.py",
        ]

        missing = []
        for file in required_files:
            if not (project_root / file).exists():
                logger.warning(f"Missing critical file for self-analysis: {file}")
                missing.append(file)

        if missing:
            logger.warning(
                f"Missing critical files for self-analysis: {', '.join(missing)}"
            )

    def _collect_raw_file_contents(self) -> Dict[str, str]:
        """
        Collects the raw string content of relevant files in the project.
        Filters out binary files, large files, and common ignore patterns.
        """
        raw_contents: Dict[str, str] = {}
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
            "data/",  # Exclude data directory contents by default
            # "docs/", # Exclude docs directory contents by default, unless explicitly needed (now included)
            "repo_contents.txt",
            "repo_to_single_file.sh",  # Specific files
            ".env",  # Exclude environment files
            "*.bak",  # Exclude backup files
        ]
        include_extensions = [
            ".py",
            ".md",
            ".yaml",
            ".yml",
            ".json",
            ".toml",
            ".txt",
            ".sh",
            ".dockerignore",
            ".gitignore",
            ".pre-commit-config.yaml",
            "Dockerfile",
            ".github/workflows/*.yml",  # Include GitHub Actions workflows
            "requirements.txt",
            "requirements-prod.txt",
            "LICENSE",
            "README.md",
        ]

        for root, dirs, files in os.walk(self.project_root):
            # Filter out excluded directories
            dirs[:] = [
                d
                for d in dirs
                if not any(fnmatch.fnmatch(d, p.strip("/")) for p in exclude_patterns)
            ]

            for file in files:
                relative_file_path = Path(root).relative_to(self.project_root) / file
                full_file_path = Path(root) / file

                # Apply exclude patterns to the relative path
                if any(
                    fnmatch.fnmatch(str(relative_file_path), p)
                    for p in exclude_patterns
                ):
                    continue
                # Apply include extensions/patterns
                if not any(
                    str(relative_file_path).endswith(ext)
                    or fnmatch.fnmatch(str(relative_file_path), ext)
                    for ext in include_extensions
                ):
                    continue

                try:
                    # Limit file size to avoid reading huge binary files or logs
                    if full_file_path.stat().st_size > 1 * 1024 * 1024:  # 1MB limit
                        self.logger.warning(
                            f"Skipping large file: {relative_file_path} (>{1}MB)"
                        )
                        continue

                    with open(
                        full_file_path, "r", encoding="utf-8", errors="ignore"
                    ) as f:
                        raw_contents[str(relative_file_path)] = f.read()
                except Exception as e:
                    self.logger.warning(
                        f"Could not read file {relative_file_path}: {e}"
                    )
        return raw_contents

    def _scan_file_structure(self) -> Dict[str, Any]:
        """Scan and document the file structure of the project."""
        file_structure = {}
        try:
            for root, dirs, files in os.walk(self.project_root):
                # Filter out excluded directories and hidden directories (except .github)
                dirs[:] = [d for d in dirs if not d.startswith(".") or d == ".github"]
                # Remove common excluded directories from traversal
                for excluded_dir in [
                    ".git",
                    "__pycache__",
                    "venv",
                    ".venv",
                    "node_modules",
                    "data",
                    "docs",
                ]:
                    if excluded_dir in dirs:
                        dirs.remove(excluded_dir)

                rel_path = os.path.relpath(root, self.project_root)

                dir_key = rel_path if rel_path != "." else "."

                file_structure[dir_key] = {
                    "subdirectories": dirs,
                    "files": files,
                    "file_count": len(files),
                    "subdir_count": len(dirs),
                }
        except Exception as e:
            logger.error(f"Error walking directory structure: {e}", exc_info=True)
            return {
                "error": f"Failed to scan file structure: {e}"
            }  # Return error in a structured way

        # Add preview of critical files
        critical_files = [
            "core.py",
            "src/llm_provider.py",
            "src/config/settings.py",
            "app.py",
            "personas.yaml",
        ]
        file_structure["critical_files_preview"] = {}
        for filename in critical_files:
            file_path = Path(self.project_root) / filename
            if file_path.exists():
                try:
                    with open(file_path, "r", encoding="utf-8") as f:
                        lines = f.readlines()[:50]  # Read first 50 lines
                        file_structure["critical_files_preview"][filename] = "".join(
                            lines
                        )
                except Exception as e:
                    logger.error(f"Error reading critical file {filename}: {str(e)}")
            else:
                logger.warning(f"Critical file not found: {filename}")

        return file_structure


def get_codebase_scanner_instance() -> CodebaseScanner:
    """Factory function to get a properly configured CodebaseScanner instance."""
    from src.utils.path_utils import PROJECT_ROOT

    return CodebaseScanner(str(PROJECT_ROOT))
