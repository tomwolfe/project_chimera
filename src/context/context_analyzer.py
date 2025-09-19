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

# --- Import PROJECT_ROOT and related utilities from src.utils.path_utils ---
# This ensures a single source of truth for project root detection.
from src.utils.path_utils import (
    PROJECT_ROOT,
    is_within_base_dir,
    sanitize_and_validate_file_path,
)

# REMOVED: Duplicated PROJECT_ROOT_MARKERS, _find_project_root_internal,
# and dynamic PROJECT_ROOT definition as they are now imported from src.utils.path_utils.
# REMOVED: Duplicated is_within_base_dir and sanitize_and_validate_file_path
# as they are now imported from src.utils.path_utils.


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
            # REMOVED: "data/",  # Exclude data directory contents by default
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
                    # REMOVED: "data",
                    # REMOVED: "docs",
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

    _model_instance = None  # Singleton instance for the model
    _model_name_cached = None  # To ensure we don't load different models
    _cache_dir_cached = None  # To ensure consistent cache dir

    def __init__(
        self,
        cache_dir: str,
        raw_file_contents: Optional[Dict[str, str]] = None,
        max_file_content_size: int = 500000,
        codebase_scanner: Optional[CodebaseScanner] = None,
        model_name: str = "all-MiniLM-L6-v2",  # NEW: Make model_name configurable
    ):
        """
        Initializes the analyzer.
        """
        self.cache_dir = cache_dir
        self.max_file_content_size = max_file_content_size
        self.codebase_scanner = codebase_scanner
        self.model_name = model_name  # Store model name

        if raw_file_contents is not None:
            self.raw_file_contents = {
                k: v
                for k, v in raw_file_contents.items()
                if len(v) < self.max_file_content_size
            }
            if len(raw_file_contents) != len(self.raw_file_contents):
                self.logger.warning(
                    f"Filtered out {len(raw_file_contents) - len(self.raw_file_contents)} large files from initial raw_file_contents in ContextRelevanceAnalyzer init."
                )
        else:
            self.raw_file_contents = {}

        self.logger = logger
        self.persona_router = None
        self.file_embeddings: Dict[str, Any] = {}
        self._last_raw_file_contents_hash: Optional[int] = None

        self._load_model()  # Call the method to load/get the singleton model

        if self.raw_file_contents:
            self.file_embeddings = self._compute_file_embeddings(self.raw_file_contents)
            self._last_raw_file_contents_hash = hash(
                frozenset(self.raw_file_contents.items())
            )

    def _load_model(self):
        """Loads the SentenceTransformer model using the singleton pattern."""
        try:
            # Use class method to get/load the singleton model
            self.model = ContextRelevanceAnalyzer._get_model_instance(
                self.model_name, self.cache_dir
            )
        except Exception as e:
            self.logger.error(
                f"Failed to load SentenceTransformer model: {e}", exc_info=True
            )
            raise RuntimeError(f"Failed to initialize SentenceTransformer: {e}") from e

    @classmethod
    def _get_model_instance(cls, model_name: str, cache_dir: str):
        """
        Provides a singleton instance of the SentenceTransformer model.
        Ensures the model is loaded only once and from the specified cache directory.
        """
        if (
            cls._model_instance is None
            or cls._model_name_cached != model_name
            or cls._cache_dir_cached != cache_dir
        ):
            # Ensure the cache directory exists before loading
            Path(cache_dir).mkdir(parents=True, exist_ok=True)
            cls._model_instance = SentenceTransformer(
                model_name, cache_folder=cache_dir
            )
            cls._model_name_cached = model_name
            cls._cache_dir_cached = cache_dir
            logger.info(
                f"SentenceTransformer model loaded (singleton) from {cache_dir}"
            )
        return cls._model_instance

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
                if v and len(v) < self.max_file_content_size:
                    files_to_encode[k] = v
                elif v:
                    self.logger.warning(
                        f"Skipping embedding for large file: {k} ({len(v)} bytes > {self.max_file_content_size} bytes)"
                    )

            if not files_to_encode:
                self.logger.warning(
                    "No non-empty or appropriately sized file content found in context for embedding. Clearing existing embeddings."
                )
                self.file_embeddings = {}
                self._last_raw_file_contents_hash = None
                return {}

            if not hasattr(self, "model") or self.model is None:
                raise RuntimeError(
                    "SentenceTransformer model not loaded for embedding."
                )

            file_paths = list(files_to_encode.keys())
            file_contents = list(files_to_encode.values())

            self.logger.info(f"Computing embeddings for {len(file_paths)} files...")
            if not file_contents:
                self.logger.warning("No file contents to encode for embeddings.")

            # --- START FIX: Chunk large documents before embedding ---
            processed_contents = []
            # Heuristic: A token is roughly 4 chars. The model max sequence length is ~256-512 tokens.
            # We'll use a character limit of 1000 as a safe chunk size.
            CHUNK_SIZE = 1000
            for content in file_contents:
                if len(content) > CHUNK_SIZE:
                    # For large files, we take the beginning and end chunks to capture imports and key logic.
                    processed_contents.append(
                        content[:CHUNK_SIZE] + "\n...\n" + content[-CHUNK_SIZE:]
                    )
                else:
                    processed_contents.append(content)
            # --- END FIX ---

            file_embeddings_list = self.model.encode(processed_contents)

            embeddings = dict(zip(file_paths, file_embeddings_list))
            self.logger.info(f"Computed embeddings for {len(embeddings)} files.")

        except Exception as e:
            self.logger.error(f"Error computing file embeddings: {e}", exc_info=True)
            embeddings = {}

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
            if score < 0:  # Skip files where similarity calculation fails
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

    def _count_tokens_robustly(self, text: str) -> int:
        """Robustly counts tokens using available tokenizer methods on the SentenceTransformer's internal tokenizer."""
        if hasattr(self.model.tokenizer, "count_tokens"):
            return self.model.tokenizer.count_tokens(text)
        elif hasattr(self.model.tokenizer, "encode"):
            return len(self.model.tokenizer.encode(text))
        else:
            # Fallback for other tokenizer types, or raise an error if no known method
            logger.warning(
                f"Unknown tokenizer type for {type(self.model.tokenizer).__name__}. Falling back to character count / 4 estimate."
            )
            return len(text) // 4  # Rough estimate

    def generate_context_summary(
        self, relevant_files: List[Tuple[str, float]], max_tokens: int, prompt: str = ""
    ) -> str:
        """Generates a detailed summary of the relevant codebase context, including actual file contents."""
        current_summary_parts = [
            f"Codebase Context for prompt: '{prompt[:100]}...'\n\n"
        ]
        current_tokens = self._count_tokens_robustly(current_summary_parts[0])

        # Diagnostic print to inspect relevant_files before the loop
        self.logger.debug(f"Relevant files received for summary: {relevant_files}")

        for i, item in enumerate(relevant_files):
            file_path = None
            score = None
            try:
                # Explicitly try to unpack, catching the ValueError if it occurs
                file_path, score = item
            except (ValueError, TypeError) as e:
                self.logger.warning(
                    f"Skipping malformed item at index {i} in relevant_files: {item}. Error: {e}"
                )
                continue

            # Additional defensive checks for type
            if not isinstance(file_path, str):
                self.logger.warning(
                    f"Skipping item with non-string file path at index {i}: {item}"
                )
                continue

            file_content = self.raw_file_contents.get(file_path, "")
            if not file_content:
                self.logger.debug(
                    f"Skipping empty or non-existent content for file: {file_path}"
                )
                continue

            remaining_tokens_for_content = max_tokens - current_tokens - 50
            if remaining_tokens_for_content <= 0:
                self.logger.info(
                    f"Context token budget exhausted. Stopping at file: {file_path}"
                )
                break

            # --- START FIX ---
            # Replace the incorrect truncate_to_token_limit with the correct transformers library pattern.
            token_ids = self.model.tokenizer.encode(
                file_content, max_length=remaining_tokens_for_content, truncation=True
            )
            truncated_content = self.model.tokenizer.decode(
                token_ids, skip_special_tokens=True
            )
            if len(token_ids) >= remaining_tokens_for_content:
                truncated_content += "\n... (truncated)"
            # --- END FIX ---

            file_block = f"### File: {file_path}\n```\n{truncated_content}\n```\n\n"
            file_block_tokens = self._count_tokens_robustly(file_block)

            if current_tokens + file_block_tokens <= max_tokens:
                current_summary_parts.append(file_block)
                current_tokens += file_block_tokens
            else:
                if (
                    self._count_tokens_robustly(f"- {file_path}\n")
                    <= max_tokens - current_tokens
                ):
                    current_summary_parts.append(
                        f"- {file_path} (content omitted due to token limits)\n"
                    )
                    current_tokens += self._count_tokens_robustly(f"- {file_path}\n")
                self.logger.info(
                    f"Context token budget exhausted. Stopping at file: {file_path}"
                )
                break

        if current_tokens < max_tokens:
            current_summary_parts.append(
                f"Remaining token budget for context: {max_tokens - current_tokens} tokens.\n"
            )

        final_summary = "".join(current_summary_parts)
        return final_summary

    def get_context_summary(self) -> str:
        """Returns a summary of the raw file contents available."""
        if self.raw_file_contents:
            return f"Raw file contents available ({len(self.raw_file_contents)} files). See details in intermediate steps."
        return "No raw file contents provided or scanned."

    def analyze_codebase(self) -> Tuple[Dict[str, Any], Dict[str, str]]:
        """
        Scans the codebase using the associated CodebaseScanner and updates internal context.
        Returns structured context and raw file contents.
        """
        if not self.codebase_scanner:
            logger.error("CodebaseScanner not initialized in ContextRelevanceAnalyzer.")
            return {}, {}

        full_codebase_analysis = self.codebase_scanner.scan_codebase()
        structured_context = full_codebase_analysis.get("file_structure", {})
        raw_contents = full_codebase_analysis.get("raw_file_contents", {})

        if raw_contents:
            current_files_hash = hash(frozenset(raw_contents.items()))
            if (
                not hasattr(self, "_last_raw_file_contents_hash")
                or self._last_raw_file_contents_hash != current_files_hash
            ):
                self.raw_file_contents = raw_contents
                self.compute_file_embeddings(self.raw_file_contents)
                self._last_raw_file_contents_hash = current_files_hash
                logger.info(
                    "ContextRelevanceAnalyzer updated with new codebase scan results."
                )

        return structured_context, raw_contents
