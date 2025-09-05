# src/context/context_analyzer.py
import os # Used for os.walk, os.path.relpath
import logging # Used for logger
from pathlib import Path # Used for Path objects
from typing import Dict, Any, List, Tuple, Optional
from sentence_transformers import SentenceTransformer # Needed for embeddings
import re # For keyword matching
import json # For potential JSON handling
import numpy as np # Needed for semantic similarity calculation

logger = logging.getLogger(__name__)


# --- CodebaseScanner Class ---
class CodebaseScanner:
    """Scans and analyzes the project's codebase to provide context for self-improvement."""

    def __init__(self, project_root: str = None):
        """Initialize with optional project root path."""
        if project_root is None:
            found_root = self._find_project_root()
            if found_root:
                project_root = str(found_root)
            else:
                project_root = str(
                    Path(__file__).resolve().parent.parent.parent
                )
                logger.warning(
                    f"Project root markers not found. Falling back to default path: {project_root}"
                )

        self.project_root = project_root
        self.codebase_path = Path(self.project_root)
        self.logger = logging.getLogger(__name__)
        self.logger.info(
            f"CodebaseScanner initialized with project root: {self.project_root}"
        )

    def scan_codebase(self) -> Dict[str, Any]:
        """Scan the entire codebase and return structured context."""
        context = {
            "file_structure": {},
            "code_quality_metrics": {},
            "security_issues": [],
            "test_coverage": {},
            "dependencies": {},
        }

        try:
            context["file_structure"] = self._scan_file_structure()
            context["code_quality_metrics"] = self._analyze_code_quality()
            context["security_issues"] = self._check_security_issues()
            context["test_coverage"] = self._analyze_test_coverage()
            context["dependencies"] = self._gather_dependencies()

            return context
        except Exception as e:
            logger.error(f"Error scanning codebase: {str(e)}", exc_info=True)
            return {"error": str(e)}

    def load_own_codebase_context(self) -> Dict[str, Any]:
        """Loads Project Chimera's own codebase context for self-analysis."""
        project_root = self._find_project_root()
        if not project_root:
            logger.error("Could not locate Project Chimera root directory for self-analysis")
            raise RuntimeError(
                "Project root not found. Self-analysis requires access to the codebase. "
                "Ensure the application is running from within the Project Chimera directory."
            )
        
        self._validate_project_structure(project_root)
        
        # Collect context from relevant directories
        context = {
            "project_root": str(project_root),
            "file_structure": self._scan_file_structure(),
            "code_quality_metrics": self._analyze_code_quality(),
            "security_issues": self._check_security_issues(),
            "test_coverage": self._analyze_test_coverage(),
            "dependencies": self._gather_dependencies()
        }

        return context

    def _find_project_root(self) -> Optional[Path]:
        """Determine the root directory of the current Project Chimera instance."""
        current_path = Path(__file__).resolve().parent

        markers = ['pyproject.toml', '.git', 'README.md', 'src/']

        for parent in [current_path] + list(current_path.parents):
            if any((parent / marker).exists() for marker in markers):
                return parent

        return None

    @staticmethod
    def _validate_project_structure(project_root: Path) -> None:
        """Validates critical project structure elements for self-analysis."""
        required_files = [
            "pyproject.toml",
            "personas.yaml",
            "src/__init__.py",
            "src/core.py"
        ]
        
        missing = []
        for file in required_files:
            if not (project_root / file).exists():
                missing.append(file)
        
        if missing:
            logger.warning(f"Missing critical files for self-analysis: {', '.join(missing)}")
    
    def _gather_dependencies(self) -> Dict[str, Any]:
        """Placeholder to gather project dependencies."""
        logger.info("Gathering dependencies (placeholder).")
        return {
            "python_dependencies": ["streamlit", "google-genai", "pydantic"],
            "system_dependencies": [],
            "package_manager": "pip",
        }

    def _scan_file_structure(self) -> Dict[str, Any]:
        """Scan and document the file structure of the project."""
        file_structure = {}
        try:
            for root, dirs, files in os.walk(self.project_root):
                if "venv" in dirs:
                    dirs.remove("venv")
                if ".git" in dirs:
                    dirs.remove(".git")
                if "__pycache__" in dirs:
                    dirs.remove("__pycache__")
                if "node_modules" in dirs:
                    dirs.remove("node_modules")

                dirs[:] = [d for d in dirs if not d.startswith(".")]

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
            return {"error": f"Failed to scan file structure: {e}"}

        critical_files = ["core.py", "src/llm_provider.py", "src/config/settings.py"]

        file_structure["critical_files_preview"] = {}
        for filename in critical_files:
            file_path = Path(self.project_root) / filename
            if file_path.exists():
                try:
                    with open(file_path, "r", encoding="utf-8") as f:
                        lines = f.readlines()[:50]
                        file_structure["critical_files_preview"][filename] = "".join(
                            lines
                        )
                except Exception as e:
                    logger.error(f"Error reading critical file {filename}: {str(e)}")
            else:
                logger.warning(f"Critical file not found: {filename}")

        return file_structure

    def _analyze_code_quality(self) -> Dict[str, Any]:
        """Analyze code quality metrics (placeholder for Ruff, complexity, etc.)."""
        logger.info("Running placeholder code quality analysis.")
        return {
            "ruff_issues_count": 0,
            "complexity_metrics": {
                "avg_cyclomatic_complexity": 2.5,
                "avg_loc_per_function": 30,
                "avg_num_arguments": 3,
                "avg_max_nesting_depth": 2,
            },
            "code_smells_count": 5,
            "detailed_issues": [],
            "test_coverage_summary": {
                "overall_coverage_percentage": 75.5,
                "untested_files": ["src/utils/some_util.py"],
                "critical_paths_uncovered": ["auth_flow"],
            },
        }

    def _check_security_issues(self) -> List[Dict[str, Any]]:
        """Check for security issues in the codebase (placeholder for Bandit, AST checks)."""
        logger.info("Running placeholder security analysis.")
        return [
            {
                "type": "Bandit Security Issue",
                "file": "src/llm_provider.py",
                "line": 42,
                "code": "B105",
                "message": "[MEDIUM] Hardcoded password string",
            },
            {
                "type": "Security Vulnerability (AST)",
                "file": "src/database_operations.py",
                "line": 150,
                "message": "Use of eval() is discouraged.",
            },
        ]

    def _analyze_test_coverage(self) -> Dict[str, Any]:
        """Analyze test coverage metrics (placeholder for coverage.py)."""
        logger.info("Running placeholder test coverage analysis.")
        return {
            "overall_coverage_percentage": 85.0,
            "files_covered": 50,
            "total_files": 60,
            "coverage_details": "High coverage in core modules, lower in utility scripts.",
            "untested_files": ["src/utils/legacy_helper.py"],
            "critical_paths_uncovered": [],
        }


# --- ContextRelevanceAnalyzer Class ---
class ContextRelevanceAnalyzer:
    """
    Analyzes the relevance of codebase context to the prompt and personas,
    using semantic search and keyword matching.
    """

    def __init__(
        self, cache_dir: str, codebase_context: Optional[Dict[str, str]] = None
    ):
        """
        Initializes the analyzer.
        """
        self.cache_dir = cache_dir
        self.codebase_context = codebase_context if codebase_context is not None else {}
        self.logger = logger
        self.persona_router = None

        try:
            Path(self.cache_dir).mkdir(parents=True, exist_ok=True)
            self.model = SentenceTransformer(
                "all-MiniLM-L6-v2",
                cache_folder=self.cache_dir,
            )
            self.logger.info(f"SentenceTransformer model loaded from {self.cache_dir}")
        except Exception as e:
            self.logger.error(
                f"Failed to load SentenceTransformer model: {e}", exc_info=True
            )
            raise RuntimeError(f"Failed to initialize SentenceTransformer: {e}") from e

        if self.codebase_context:
            self.file_embeddings = self._compute_file_embeddings(self.codebase_context)
        else:
            self.file_embeddings = {}

    def set_persona_router(self, persona_router: Any):
        """Sets the persona router for context relevance scoring."""
        self.persona_router = persona_router
        self.logger.info("Persona router set for context relevance analysis.")

    def _compute_file_embeddings(self, context: Dict[str, str]) -> Dict[str, Any]:
        """Computes embeddings for files in the codebase context."""
        if not context:
            return {}

        embeddings = {}
        try:
            files_with_content = {k: v for k, v in context.items() if v}
            if not files_with_content:
                self.logger.warning("No file content found in context for embedding.")
                return {}

            file_paths = list(files_with_content.keys())
            file_contents = list(files_with_content.values())

            self.logger.info(f"Computing embeddings for {len(file_paths)} files...")
            if not hasattr(self, "model") or self.model is None:
                raise RuntimeError("SentenceTransformer model not loaded.")

            file_embeddings_list = self.model.encode(file_contents)

            embeddings = dict(zip(file_paths, file_embeddings_list))
            self.logger.info(f"Computed embeddings for {len(embeddings)} files.")

        except Exception as e:
            self.logger.error(f"Error computing file embeddings: {e}", exc_info=True)
            return {}
        return embeddings

    def find_relevant_files(
        self,
        prompt: str,
        max_context_tokens: int,
        active_personas: List[str] = [],
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
                similarity = np.dot(prompt_embedding, embedding) / (
                    np.linalg.norm(prompt_embedding) * np.linalg.norm(embedding)
                )
                relevance_scores[file_path] = similarity
            except Exception as e:
                self.logger.warning(
                    f"Could not calculate similarity for {file_path}: {e}"
                )
                relevance_scores[file_path] = -1.0

        sorted_files = sorted(
            relevance_scores.items(), key=lambda item: item[1], reverse=True
        )

        relevant_files = []
        current_tokens = 0
        avg_file_tokens = 500

        for file_path, score in sorted_files:
            if score < 0:
                continue
            file_tokens = avg_file_tokens
            if current_tokens + file_tokens <= max_context_tokens:
                relevant_files.append((file_path, score))
                current_tokens += file_tokens
            else:
                break

        self.logger.info(
            f"Found {len(relevant_files)} relevant files within token budget."
        )
        return relevant_files

    def generate_context_summary(
        self,
        relevant_files: List[str],
        max_tokens: int,
        prompt: str = "",
    ) -> str:
        """
        Generates a concise summary of the relevant codebase context.
        """
        summary = f"Context Summary for prompt: '{prompt[:100]}...'\n\n"
        summary += f"Relevant files ({len(relevant_files)}):\n"
        for file_path in relevant_files:
            summary += f"- {file_path}\n"

        if self.codebase_context and "critical_files_preview" in self.codebase_context:
            summary += "\nCritical Files Preview:\n"
            for filename, snippet in self.codebase_context[
                "critical_files_preview"
            ].items():
                summary += f"\n--- {filename} (first 50 lines) ---\n{snippet}\n--------------------\n"
        else:
            summary += "\nNo critical files preview available.\n"

        summary += "\n(Detailed content summarization is a placeholder.)"

        chars_per_token_estimate = 4
        if len(summary) > max_tokens * chars_per_token_estimate:
            summary = summary[: max_tokens * chars_per_token_estimate] + "..."

        return summary

    def get_context_summary(self) -> str:
        """Returns the pre-computed or scanned context summary."""
        if self.codebase_context:
            return f"Codebase context available ({len(self.codebase_context)} files). See details in intermediate steps."
        return "No codebase context provided or scanned."