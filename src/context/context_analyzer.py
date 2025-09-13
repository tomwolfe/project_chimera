# src/context/context_analyzer.py
import os
import logging
from pathlib import Path
import fnmatch
from typing import Dict, Any, List, Tuple, Optional
from sentence_transformers import SentenceTransformer
import re
import json
import numpy as np

logger = logging.getLogger(__name__)

# --- Constants for Project Root Markers ---
# These markers help in identifying the project's root directory.
PROJECT_ROOT_MARKERS = [
    ".git",
    "config.yaml",
    "pyproject.toml",
    "Dockerfile",
    "README.md",
    "src/",
    ".github/",
    "app.py",  # ADDED
    "core.py",  # ADDED
]


def _find_project_root_internal(start_path: Path) -> Optional[Path]:
    """Internal helper to find the project root without raising an error."""
    current_dir = start_path
    # Limit search depth to prevent infinite loops in unusual file structures
    for _ in range(15):
        if any(
            current_dir.joinpath(marker).exists() for marker in PROJECT_ROOT_MARKERS
        ):
            return current_dir

        parent_path = current_dir.parent
        if parent_path == current_dir:  # Reached the filesystem root
            break
        current_dir = parent_path
    return None  # Return None if not found


# --- Define PROJECT_ROOT dynamically ---
# This ensures that the project root is correctly identified regardless of where the script is run from.
_initial_start_path = Path(__file__).resolve().parent
_found_root = _find_project_root_internal(_initial_start_path)

if _found_root:
    PROJECT_ROOT = _found_root
    logger.info(f"Project root identified at: {PROJECT_ROOT}")
else:
    PROJECT_ROOT = Path.cwd()  # Fallback to current working directory
    logger.warning(
        f"Project root markers ({PROJECT_ROOT_MARKERS}) not found after searching up to 15 levels from {_initial_start_path}. Falling back to CWD: {PROJECT_ROOT}. Path validation might be less effective."
    )


def is_within_base_dir(file_path: Path) -> bool:
    """Checks if a file path is safely within the project base directory."""
    try:
        resolved_path = file_path.resolve()
        resolved_path.relative_to(
            PROJECT_ROOT
        )  # This will raise ValueError if not a subpath
        return True
    except ValueError:
        logger.debug(
            f"Path '{file_path}' (resolved: '{resolved_path}') is outside the project base directory '{PROJECT_ROOT}'."
        )
        return False
    except FileNotFoundError:
        logger.debug(
            f"Path '{file_path}' (resolved: '{resolved_path}') does not exist."
        )
        return False
    except Exception as e:
        logger.error(
            f"Error validating path '{file_path}' against base directory '{PROJECT_ROOT}': {e}"
        )
        return False


def sanitize_and_validate_file_path(raw_path: str) -> str:
    """Sanitizes and validates a file path for safety against traversal and invalid characters.
    Ensures the path is within the project's base directory and returns it relative to PROJECT_ROOT.
    """
    if not raw_path:
        raise ValueError("File path cannot be empty.")

    # MODIFIED: Allow forward slashes in sanitized_path_str as they are valid path separators
    sanitized_path_str = re.sub(
        r'[<>:"\\|?*\x00-\x1f\x7f]', "", raw_path
    )  # Removed '/' from invalid chars
    sanitized_path_str = re.sub(
        r"\.\./", "", sanitized_path_str
    )  # Remove parent directory traversal
    sanitized_path_str = re.sub(
        r"//+", "/", sanitized_path_str
    )  # Normalize multiple slashes

    path_obj = Path(sanitized_path_str)

    # Resolve the path to get its absolute, canonical form
    try:
        resolved_path = path_obj.resolve()
    except Exception as e:
        raise ValueError(f"Failed to resolve path '{sanitized_path_str}': {e}") from e

    # Check if the resolved path is within the project root
    if not is_within_base_dir(resolved_path):
        raise ValueError(
            f"File path '{raw_path}' (sanitized: '{sanitized_path_str}') resolves to a location outside the allowed project directory."
        )

    # Return path relative to PROJECT_ROOT
    try:
        return str(resolved_path.relative_to(PROJECT_ROOT))
    except ValueError:
        # This should ideally not happen if is_within_base_dir returned True,
        # but as a safeguard, return the absolute path if relative_to fails.
        logger.warning(
            f"Could not get relative path for '{resolved_path}' from '{PROJECT_ROOT}'. Returning absolute path."
        )
        return str(resolved_path)


# --- CodebaseScanner Class ---
class CodebaseScanner:
    """Scans and analyzes the project's codebase to provide context for self-improvement."""

    def __init__(self, project_root: str = None):
        """Initialize with optional project root path."""
        if project_root is None:
            # Use the dynamically determined PROJECT_ROOT
            project_root = str(PROJECT_ROOT)
            logger.info(f"Using dynamically determined project root: {project_root}")

        self.project_root = project_root
        self.codebase_path = Path(self.project_root)
        self.logger = logger
        self.logger.info(
            f"CodebaseScanner initialized with project root: {self.project_root}"
        )

    def scan_codebase(self) -> Dict[str, Any]:
        """Scan the entire codebase and return structured context, including raw file contents."""
        context = {
            "file_structure": {},
            "raw_file_contents": {},  # NEW: Add raw file contents here
        }

        try:
            context["file_structure"] = self._scan_file_structure()
            context["raw_file_contents"] = (
                self._collect_raw_file_contents()
            )  # NEW: Collect raw file contents
            context["project_root"] = (
                self.project_root
            )  # Add project root to the context

            return context
        except Exception as e:
            logger.error(f"Error scanning codebase: {str(e)}", exc_info=True)
            return {"error": str(e)}

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

        self._validate_project_structure(project_root_path)

        # Call scan_codebase to get the full structured context including raw file contents
        return self.scan_codebase()

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


# --- ContextRelevanceAnalyzer Class ---
class ContextRelevanceAnalyzer:
    """
    Analyzes the relevance of codebase context to the prompt and personas,
    using semantic search and keyword matching.
    """

    def __init__(
        self,
        cache_dir: str,
        raw_file_contents: Optional[
            Dict[str, str]
        ] = None,  # MODIFIED: Renamed codebase_context to raw_file_contents
        max_file_content_size: int = 100000, # NEW: Add max_file_content_size (100KB default)
    ):
        """
        Initializes the analyzer.
        """
        self.cache_dir = cache_dir
        self.max_file_content_size = max_file_content_size # NEW: Store max_file_content_size
        
        # NEW: Filter raw_file_contents based on size during initialization
        if raw_file_contents is not None:
            self.raw_file_contents = {
                k: v for k, v in raw_file_contents.items()
                if len(v) < self.max_file_content_size
            }
            if len(raw_file_contents) != len(self.raw_file_contents):
                self.logger.warning(f"Filtered out {len(raw_file_contents) - len(self.raw_file_contents)} large files from initial raw_file_contents in ContextRelevanceAnalyzer init.")
        else:
            self.raw_file_contents = {}

        self.logger = logger
        self.persona_router = None
        self.file_embeddings: Dict[str, Any] = {}  # Initialize empty
        self._last_raw_file_contents_hash: Optional[int] = (
            None  # NEW: Store hash of last processed content
        )

        try:
            # Ensure the cache directory exists
            Path(self.cache_dir).mkdir(parents=True, exist_ok=True)
            # Initialize SentenceTransformer model
            # This will download the model if not already cached in self.cache_dir

            self.model = SentenceTransformer(
                "all-MiniLM-L6-v2", cache_folder=self.cache_dir
            )
            self.logger.info(f"SentenceTransformer model loaded from {self.cache_dir}")
        except Exception as e:
            self.logger.error(
                f"Failed to load SentenceTransformer model: {e}", exc_info=True
            )
            raise RuntimeError(f"Failed to initialize SentenceTransformer: {e}") from e

        # NEW: Compute initial embeddings and store hash if raw_file_contents are provided
        if self.raw_file_contents:
            self.file_embeddings = self._compute_file_embeddings(self.raw_file_contents)
            self._last_raw_file_contents_hash = hash(
                frozenset(self.raw_file_contents.items())
            )

    def compute_file_embeddings(self, context: Dict[str, str]) -> Dict[str, Any]:
        """
        Public method to compute embeddings for files in the codebase context.
        Includes a hash-based check to skip re-computation if the content hasn't changed.
        """
        if not context:
            self.logger.warning(
                "No file content provided for embedding. Clearing existing embeddings."
            )
            self.file_embeddings = {}  # Clear old embeddings if context is empty
            self._last_raw_file_contents_hash = None  # Reset hash
            return {}

        # Calculate hash of current context to check for changes
        current_context_hash = hash(frozenset(context.items()))

        # If the context hasn't changed, return existing embeddings
        if (
            hasattr(self, "_last_raw_file_contents_hash")
            and self._last_raw_file_contents_hash == current_context_hash
        ):
            self.logger.info("File embeddings are up-to-date. Skipping re-computation.")
            return self.file_embeddings

        # If context changed or no embeddings exist, re-compute
        self.file_embeddings = {}  # Clear existing embeddings before re-computing
        self.logger.info("Computing embeddings for files...")

        embeddings = {}
        try:
            # Filter out empty file contents and large files before encoding
            files_to_encode = {}
            for k, v in context.items():
                if v and len(v) < self.max_file_content_size: # NEW: Filter by max_file_content_size
                    files_to_encode[k] = v
                elif v:
                    self.logger.warning(f"Skipping embedding for large file: {k} ({len(v)} bytes > {self.max_file_content_size} bytes)")

            if not files_to_encode:
                self.logger.warning(
                    "No non-empty or appropriately sized file content found in context for embedding. Clearing existing embeddings."
                )
                self.file_embeddings = {}
                self._last_raw_file_contents_hash = None
                return {}

            # NEW: Log if the model.encode call fails
            if not hasattr(self, "model") or self.model is None:
                raise RuntimeError(
                    "SentenceTransformer model not loaded for embedding."
                )

            file_paths = list(files_to_encode.keys())
            file_contents = list(files_to_encode.values())

            self.logger.info(f"Computing embeddings for {len(file_paths)} files...")
            # Ensure file_contents are not empty before encoding
            if not file_contents:
                self.logger.warning("No file contents to encode for embeddings.")
            file_embeddings_list = self.model.encode(file_contents)

            embeddings = dict(zip(file_paths, file_embeddings_list))
            self.logger.info(f"Computed embeddings for {len(embeddings)} files.")

        except Exception as e:
            self.logger.error(f"Error computing file embeddings: {e}", exc_info=True)
            embeddings = {}  # Return empty dict on error

        # Store the newly computed embeddings and the hash of the content that generated them
        self.file_embeddings = embeddings
        self._last_raw_file_contents_hash = current_context_hash
        return self.file_embeddings

    def set_persona_router(self, persona_router: Any):
        """Sets the persona router for context relevance scoring."""
        self.persona_router = persona_router
        self.logger.info("Persona router set for context relevance analysis.")

    def _compute_file_embeddings(self, context: Dict[str, str]) -> Dict[str, Any]:
        """
        Internal method to compute embeddings. Delegates to the public `compute_file_embeddings`
        to ensure the caching logic is always applied.
        """
        return self.compute_file_embeddings(context)

    def find_relevant_files(
        self, prompt: str, max_context_tokens: int, active_personas: List[str] = []
    ) -> List[Tuple[str, float]]:
        """
        Finds relevant files based on prompt and persona relevance using semantic search.
        Returns a list of (file_path, relevance_score) tuples.
        """
        if not self.file_embeddings:
            self.logger.warning(
                "No file embeddings available. Cannot perform semantic search."
            )
            return []

        try:
            prompt_embedding = self.model.encode([prompt])[0]
        except Exception as e:
            self.logger.error(
                f"Failed to encode prompt for semantic search: {e}", exc_info=True
            )
            return []

        relevance_scores = {}

        for file_path, embedding in self.file_embeddings.items():
            try:
                # Calculate cosine similarity
                similarity = np.dot(prompt_embedding, embedding) / (
                    np.linalg.norm(prompt_embedding) * np.linalg.norm(embedding)
                )
                relevance_scores[file_path] = similarity
            except Exception as e:
                self.logger.warning(
                    f"Could not calculate similarity for {file_path}: {e}"
                )
                relevance_scores[
                    file_path
                ] = -1.0  # Assign a low score if calculation fails

        sorted_files = sorted(
            relevance_scores.items(), key=lambda item: item[1], reverse=True
        )

        relevant_files = []
        current_tokens = 0
        # Estimate average tokens per file for budget calculation. This is a heuristic.
        # A more accurate approach would be to count tokens for each file before adding.
        avg_file_tokens = 500

        for file_path, score in sorted_files:
            if score < 0:  # Skip files where similarity calculation failed
                continue

            # Estimate tokens for the file content. A more precise method would count tokens.
            file_tokens = avg_file_tokens  # Using a fixed average for simplicity

            if current_tokens + file_tokens <= max_context_tokens:
                relevant_files.append((file_path, score))
                current_tokens += file_tokens
            else:
                break  # Stop if adding this file exceeds the token budget

        self.logger.info(
            f"Found {len(relevant_files)} relevant files within token budget ({current_tokens}/{max_context_tokens} tokens estimated)."
        )
        return relevant_files

    def generate_context_summary(
        self, relevant_files: List[str], max_tokens: int, prompt: str = ""
    ) -> str:
        """
        Generates a concise summary of the relevant codebase context.
        Includes previews of critical files.
        """
        summary = f"Context Summary for prompt: '{prompt[:100]}...'\n\n"
        summary += f"Relevant files ({len(relevant_files)}):\n"
        for file_path in relevant_files:
            summary += f"- {file_path}\n"

        # Use the raw_file_contents to get snippets for critical files
        if self.raw_file_contents:
            summary += "\nCritical Files Preview:\n"
            # Define critical files to preview
            critical_files_to_preview = [
                "core.py",
                "src/llm_provider.py",
                "src/config/settings.py",
                "app.py",
                "personas.yaml",
            ]
            for filename in critical_files_to_preview:
                snippet = self.raw_file_contents.get(filename, "")
                if snippet:
                    lines = snippet.splitlines()[:50]  # Get first 50 lines
                    summary += f"\n--- {filename} (first 50 lines) ---\n{''.join(lines)}\n--------------------\n"
                else:
                    summary += f"\n--- {filename} (not found or empty) ---\n"
        else:
            summary += "\nNo critical files preview available.\n"

        summary += "\n(Detailed content summarization is a placeholder.)"

        # Estimate characters per token for truncation
        chars_per_token_estimate = 4
        # Truncate summary if it exceeds the token budget (approximated by characters)
        if len(summary) > max_tokens * chars_per_token_estimate:
            summary = summary[: max_tokens * chars_per_token_estimate] + "..."

        return summary

    def get_context_summary(self) -> str:
        """Returns a summary of the raw file contents available."""
        if self.raw_file_contents:  # MODIFIED: Check raw_file_contents
            return f"Raw file contents available ({len(self.raw_file_contents)} files). See details in intermediate steps."
        return "No raw file contents provided or scanned."