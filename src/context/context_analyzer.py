# src/context/context_analyzer.py
import os
import logging
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
from sentence_transformers import SentenceTransformer # Needed for embeddings
import re # For keyword matching
import json # For potential JSON handling
import numpy as np # Needed for semantic similarity calculation

# Assuming SentenceTransformer is available and cache_dir is used for it.

logger = logging.getLogger(__name__)

# --- CodebaseScanner Class (as provided in the dump) ---
class CodebaseScanner:
    """Scans and analyzes the project's codebase to provide context for self-improvement."""
    
    def __init__(self, project_root: str = None):
        """Initialize with optional project root path."""
        # Determine project root dynamically if not provided
        if project_root is None:
            # Try to find project root using common markers
            current_path = Path(__file__).resolve().parent
            for _ in range(10): # Limit search depth
                if any(current_path.joinpath(marker).exists() for marker in [".git", "pyproject.toml", "Dockerfile"]):
                    project_root = str(current_path)
                    break
                parent_path = current_path.parent
                if parent_path == current_path: # Reached filesystem root
                    break
                current_path = parent_path
            
            # Fallback if markers not found
            if project_root is None:
                project_root = str(Path(__file__).resolve().parent.parent.parent) # Default fallback to project root based on file location
                logger.warning(f"Project root markers not found. Falling back to default path: {project_root}")

        self.project_root = project_root
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"CodebaseScanner initialized with project root: {self.project_root}")
    
    def scan_codebase(self) -> Dict[str, Any]:
        """Scan the entire codebase and return structured context."""
        context = {
            "file_structure": {},
            "code_quality_metrics": {},
            "security_issues": [],
            "test_coverage": {}
        }
        
        try:
            # Scan file structure
            context["file_structure"] = self._scan_file_structure()
            
            # Analyze code quality (this would integrate with tools like Ruff/Bandit)
            # Placeholder implementation:
            context["code_quality_metrics"] = self._analyze_code_quality()
            
            # Check security issues
            # Placeholder implementation:
            context["security_issues"] = self._check_security_issues()
            
            # Analyze test coverage
            # Placeholder implementation:
            context["test_coverage"] = self._analyze_test_coverage()
            
            return context
        except Exception as e:
            logger.error(f"Error scanning codebase: {str(e)}", exc_info=True)
            return {"error": str(e)}
    
    def _scan_file_structure(self) -> Dict[str, Any]:
        """Scan and document the file structure of the project."""
        file_structure = {}
        try:
            for root, dirs, files in os.walk(self.project_root):
                # Skip virtual environments and other non-project directories
                if "venv" in dirs:
                    dirs.remove("venv") # Do not descend into venv
                if ".git" in dirs:
                    dirs.remove(".git") # Do not descend into .git
                if "__pycache__" in dirs:
                    dirs.remove("__pycache__") # Do not descend into __pycache__
                if "node_modules" in dirs:
                    dirs.remove("node_modules") # Skip node_modules

                # Skip hidden directories (like .vscode, .idea)
                dirs[:] = [d for d in dirs if not d.startswith('.')]

                rel_path = os.path.relpath(root, self.project_root)
                
                # Skip the root directory itself if it's empty or just contains skipped dirs
                if rel_path == ".":
                    if not files and not dirs: # If root is empty after skips
                        continue
                    # Use '.' as key for root directory structure if it has content
                    dir_key = "."
                else:
                    dir_key = rel_path

                file_structure[dir_key] = {
                    "subdirectories": dirs,
                    "files": files,
                    "file_count": len(files),
                    "subdir_count": len(dirs)
                }
        except Exception as e:
            logger.error(f"Error walking directory structure: {e}", exc_info=True)
            return {"error": f"Failed to scan file structure: {e}"}

        # Add code snippets for critical files
        # CORRECTED critical_files list based on feedback
        critical_files = [
            "core.py",
            "src/llm_provider.py",
            "src/config/settings.py"
        ]
        
        file_structure["critical_files_preview"] = {}
        for filename in critical_files:
            file_path = os.path.join(self.project_root, filename)
            if os.path.exists(file_path):
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        # Only take first 50 lines to avoid token explosion
                        lines = f.readlines()[:50]
                        file_structure["critical_files_preview"][filename] = "".join(lines)
                except Exception as e:
                    logger.error(f"Error reading critical file {filename}: {str(e)}")
            else:
                logger.warning(f"Critical file not found: {filename}")
        
        return file_structure
    
    def _analyze_code_quality(self) -> Dict[str, Any]:
        """Analyze code quality metrics (placeholder for Ruff, complexity, etc.)."""
        # This would normally involve running tools like Ruff, Radon, etc.
        # For now, return placeholder data.
        logger.info("Running placeholder code quality analysis.")
        return {
            "ruff_issues_count": 0, # Placeholder
            "complexity_metrics": { # Placeholder
                "avg_cyclomatic_complexity": 2.5,
                "avg_loc_per_function": 30,
                "avg_num_arguments": 3,
                "avg_max_nesting_depth": 2,
            },
            "code_smells_count": 5, # Placeholder
            "detailed_issues": [], # Placeholder for specific issues
            "test_coverage_summary": { # Placeholder
                "overall_coverage_percentage": 75.5,
                "untested_files": ["src/utils/some_util.py"],
                "critical_paths_uncovered": ["auth_flow"]
            }
        }
    
    def _check_security_issues(self) -> List[Dict[str, Any]]:
        """Check for security issues in the codebase (placeholder for Bandit, AST checks)."""
        # This would normally involve running security scanners like Bandit.
        # For now, return placeholder data.
        logger.info("Running placeholder security analysis.")
        return [
            {"type": "Bandit Security Issue", "file": "src/llm_provider.py", "line": 42, "code": "B105", "message": "[MEDIUM] Hardcoded password string"},
            {"type": "Security Vulnerability (AST)", "file": "src/database_operations.py", "line": 150, "message": "Use of eval() is discouraged."},
        ]
    
    def _analyze_test_coverage(self) -> Dict[str, Any]:
        """Analyze test coverage metrics (placeholder for coverage.py)."""
        # This would normally involve running coverage tools.
        # For now, return placeholder data.
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
# This class is expected by core.py and app.py.
# It likely uses CodebaseScanner internally or for specific tasks.
class ContextRelevanceAnalyzer:
    """
    Analyzes the relevance of codebase context to the prompt and personas,
    using semantic search and keyword matching.
    """
    def __init__(self, cache_dir: str, codebase_context: Optional[Dict[str, str]] = None):
        """
        Initializes the analyzer.

        Args:
            cache_dir: Directory for caching SentenceTransformer models.
            codebase_context: Pre-loaded codebase context (e.g., from files).
        """
        self.cache_dir = cache_dir
        self.codebase_context = codebase_context if codebase_context is not None else {}
        self.logger = logger
        self.persona_router = None # Will be set by set_persona_router

        # Initialize SentenceTransformer model
        try:
            # Ensure the cache directory exists
            Path(self.cache_dir).mkdir(parents=True, exist_ok=True)
            self.model = SentenceTransformer(
                'all-MiniLM-L6-v2', # Standard model for semantic similarity
                cache_folder=self.cache_dir
            )
            self.logger.info(f"SentenceTransformer model loaded from {self.cache_dir}")
        except Exception as e:
            self.logger.error(f"Failed to load SentenceTransformer model: {e}", exc_info=True)
            # Raise a more specific error or handle gracefully if SentenceTransformer is critical
            raise RuntimeError(f"Failed to initialize SentenceTransformer: {e}") from e

        # Compute embeddings if context is provided
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
            # Process only files with content
            files_with_content = {k: v for k, v in context.items() if v}
            if not files_with_content:
                self.logger.warning("No file content found in context for embedding.")
                return {}

            # Encode file contents
            # SentenceTransformer expects a list of strings
            file_paths = list(files_with_content.keys())
            file_contents = list(files_with_content.values())
            
            self.logger.info(f"Computing embeddings for {len(file_paths)} files...")
            # Ensure the model is loaded before encoding
            if not hasattr(self, 'model') or self.model is None:
                 raise RuntimeError("SentenceTransformer model not loaded.")
            
            file_embeddings_list = self.model.encode(file_contents)
            
            # Map embeddings back to file paths
            embeddings = dict(zip(file_paths, file_embeddings_list))
            self.logger.info(f"Computed embeddings for {len(embeddings)} files.")
            
        except Exception as e:
            self.logger.error(f"Error computing file embeddings: {e}", exc_info=True)
            # Return empty dict or raise error depending on desired behavior
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
            self.logger.warning("No file embeddings available. Cannot perform semantic search.")
            return []

        try:
            prompt_embedding = self.model.encode([prompt])[0]
        except Exception as e:
            self.logger.error(f"Failed to encode prompt for semantic search: {e}", exc_info=True)
            return []

        relevance_scores = {}

        # Calculate semantic similarity between prompt and file embeddings
        for file_path, embedding in self.file_embeddings.items():
            # Cosine similarity calculation
            # Ensure embeddings are numpy arrays for dot product and norm
            try:
                similarity = np.dot(prompt_embedding, embedding) / (
                    np.linalg.norm(prompt_embedding) * np.linalg.norm(embedding)
                )
                relevance_scores[file_path] = similarity
            except Exception as e:
                self.logger.warning(f"Could not calculate similarity for {file_path}: {e}")
                relevance_scores[file_path] = -1.0 # Assign low score on error

        # Sort files by relevance score
        sorted_files = sorted(
            relevance_scores.items(), key=lambda item: item[1], reverse=True
        )

        # Filter based on relevance score and token budget (simplified)
        relevant_files = []
        current_tokens = 0
        # Placeholder token estimation - replace with actual tokenizer if available
        # For now, assume average file size or use a simple heuristic
        avg_file_tokens = 500 # Rough estimate, adjust as needed

        for file_path, score in sorted_files:
            if score < 0: # Skip files that failed embedding calculation
                continue
            file_tokens = avg_file_tokens # Use estimated tokens
            if current_tokens + file_tokens <= max_context_tokens:
                relevant_files.append((file_path, score))
                current_tokens += file_tokens
            else:
                break # Stop if token budget is exceeded

        self.logger.info(f"Found {len(relevant_files)} relevant files within token budget.")
        return relevant_files

    def generate_context_summary(
        self,
        relevant_files: List[str],
        max_tokens: int,
        prompt: str = "",
    ) -> str:
        """
        Generates a concise summary of the relevant codebase context.
        This is a placeholder; a real implementation would summarize file contents.
        """
        summary = f"Context Summary for prompt: '{prompt[:100]}...'\n\n"
        summary += f"Relevant files ({len(relevant_files)}):\n"
        for file_path in relevant_files:
            summary += f"- {file_path}\n"

        # Add snippets of critical files if available from CodebaseScanner
        if self.codebase_context and "critical_files_preview" in self.codebase_context:
            summary += "\nCritical Files Preview:\n"
            for filename, snippet in self.codebase_context["critical_files_preview"].items():
                summary += f"\n--- {filename} ---\n{snippet}\n--------------------\n"

        # Placeholder for actual content summarization logic
        # This would involve reading files and summarizing them, potentially using another LLM call
        # or a summarization model.
        # For now, we return a basic summary.
        summary += "\n(Detailed content summarization is a placeholder.)"

        # Trim summary to fit max_tokens (using a simple character-based heuristic if tokenizer is not available)
        # In a real scenario, use the provided tokenizer.
        # Estimate characters per token (e.g., 4 chars/token)
        chars_per_token_estimate = 4
        if len(summary) > max_tokens * chars_per_token_estimate:
            summary = summary[:max_tokens * chars_per_token_estimate] + "..."

        return summary

    def get_context_summary(self) -> str:
        """Returns the pre-computed or scanned context summary."""
        # This method might return a summary generated during __init__ or via scan_codebase
        # For now, it returns a simple indicator if context is available.
        if self.codebase_context:
            return f"Codebase context available ({len(self.codebase_context)} files). See details in intermediate steps."
        return "No codebase context provided or scanned."