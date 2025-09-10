# src/context/context_analyzer.py
import os # Used for os.walk, os.path.relpath
import logging # Used for logger
from pathlib import Path # Used for Path objects
from typing import Dict, Any, List, Tuple, Optional
from sentence_transformers import SentenceTransformer # Needed for embeddings
import re # For keyword matching
import json # For potential JSON handling
import numpy as np # Needed for semantic similarity calculation
import toml # NEW: For parsing pyproject.toml
import yaml # NEW: For parsing YAML files
import subprocess # NEW: For running external tools
import sys # NEW: For sys.executable

from src.utils.command_executor import execute_command_safely # NEW: Import execute_command_safely
from src.utils.code_validator import _run_ruff, _run_bandit, _run_ast_security_checks # NEW: Import code validation tools
from src.utils.code_utils import _get_code_snippet, ComplexityVisitor # NEW: Import code utility functions

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
        """
        Loads Project Chimera's own codebase context for self-analysis.
        This method performs a comprehensive scan and collects various metrics.
        """
        project_root = self._find_project_root()
        if not project_root:
            logger.error("Could not locate Project Chimera root directory for self-analysis")
            raise RuntimeError(
                "Project root not found. Self-analysis requires access to the codebase. "
                "Ensure the application is running from within the Project Chimera directory."
            )
        
        self._validate_project_structure(project_root)
        
        file_contents: Dict[str, str] = {}
        file_structure: Dict[str, Any] = {}
        all_ruff_issues: List[Dict[str, Any]] = []
        all_bandit_issues: List[Dict[str, Any]] = []
        all_ast_security_issues: List[Dict[str, Any]] = []
        all_complexity_metrics: List[Dict[str, Any]] = []

        # 1. Scan file structure and read all relevant file contents
        for root, dirs, files in os.walk(project_root):
            # Exclude common non-code directories
            dirs[:] = [d for d in dirs if d not in ['.git', '__pycache__', 'venv', '.venv', 'node_modules', 'data', 'docs', 'custom_frameworks']]
            
            for file in files:
                # Include relevant code and config files
                if file.endswith(('.py', '.yaml', '.yml', '.json', '.toml', '.md', 'Dockerfile', 'requirements.txt', 'requirements-prod.txt')):
                    file_path = Path(root) / file
                    relative_file_path = str(file_path.relative_to(project_root))
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            content = f.read()
                            file_contents[relative_file_path] = content
                    except Exception as e:
                        logger.warning(f"Could not read file {relative_file_path}: {e}")
            
            # Build file structure representation
            rel_path = str(Path(root).relative_to(project_root))
            if rel_path == '.': rel_path = '/'
            file_structure[rel_path] = {
                "subdirectories": [d for d in dirs],
                "files": [f for f in files if f.endswith(('.py', '.yaml', '.yml', '.json', '.toml', '.md', 'Dockerfile', 'requirements.txt', 'requirements-prod.txt'))],
            }

        # 2. Run static analysis tools on collected file contents
        for relative_file_path, content in file_contents.items():
            if relative_file_path.endswith('.py'):
                # Run Ruff
                ruff_issues = _run_ruff(content, relative_file_path)
                all_ruff_issues.extend(ruff_issues)

                # Run Bandit
                bandit_issues = _run_bandit(content, relative_file_path)
                all_bandit_issues.extend(bandit_issues)

                # Run AST-based security checks
                ast_security_issues = _run_ast_security_checks(content, relative_file_path)
                all_ast_security_issues.extend(ast_security_issues)

                # Run complexity analysis
                content_lines = content.splitlines()
                file_function_metrics = ComplexityVisitor(content_lines).function_metrics
                all_complexity_metrics.extend(file_function_metrics)

        # 3. Collect other metrics (test coverage, dependencies, config analysis, deployment analysis)
        test_coverage_summary = self._assess_test_coverage() # This will run pytest with coverage
        dependencies_info = self._gather_dependencies()
        config_analysis = self._collect_configuration_analysis(str(project_root))
        deployment_analysis = self._collect_deployment_robustness_metrics(str(project_root))

        # Consolidate all collected data
        context = {
            "project_root": str(project_root),
            "file_contents": file_contents, # Raw file contents
            "file_structure": file_structure, # Structured directory/file list
            "static_analysis_results": {
                "ruff_issues": all_ruff_issues,
                "bandit_issues": all_bandit_issues,
                "ast_security_issues": all_ast_security_issues,
                "complexity_metrics": all_complexity_metrics,
            },
            "test_coverage_summary": test_coverage_summary,
            "dependencies_info": dependencies_info,
            "configuration_analysis": config_analysis.model_dump(by_alias=True),
            "deployment_analysis": deployment_analysis.model_dump(by_alias=True),
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
            "core.py" # core.py is in the root
        ]
        
        missing = []
        for file in required_files:
            if not (project_root / file).exists():
                missing.append(file)
        
        if missing:
            logger.warning(f"Missing critical files for self-analysis: {', '.join(missing)}")
    
    def _gather_dependencies(self) -> Dict[str, Any]:
        """Gathers project dependencies from requirements.txt and requirements-prod.txt."""
        dependencies = {
            "requirements_txt": [],
            "requirements_prod_txt": [],
            "dev_prod_overlap": [],
        }

        req_path = self.codebase_path / "requirements.txt"
        prod_req_path = self.codebase_path / "requirements-prod.txt"

        dev_deps = set()
        if req_path.exists():
            try:
                with open(req_path, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if line and not line.startswith("#"):
                            dependencies["requirements_txt"].append(line)
                            dev_deps.add(re.split(r'[=~><]', line)[0].lower())
            except OSError as e:
                logger.warning(f"Could not read requirements.txt: {e}")

        prod_deps = set()
        if prod_req_path.exists():
            try:
                with open(prod_req_path, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if line and not line.startswith("#"):
                            dependencies["requirements_prod_txt"].append(line)
                            prod_deps.add(re.split(r'[=~><]', line)[0].lower())
            except OSError as e:
                logger.warning(f"Could not read requirements-prod.txt: {e}")
        
        dependencies["dev_prod_overlap"] = list(dev_deps.intersection(prod_deps))
        return dependencies

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

    def _assess_test_coverage(self) -> Dict[str, Any]:
        """
        Assesses test coverage for the codebase.
        Executes pytest with coverage and parses the generated JSON report.
        """
        coverage_data = {
            "overall_coverage_percentage": 0.0,
            "coverage_details": "Failed to run coverage tool.",
        }
        try:
            # Run pytest with coverage and generate a JSON report
            command = [
                "pytest", "-v", "tests/", "--cov=src", "--cov-report=json:coverage.json"
            ]
            # Use execute_command_safely for robustness
            return_code, stdout, stderr = execute_command_safely(command, timeout=120, check=False)

            # Pytest returns 0 for success, 1 for failed tests, 2 for internal errors/usage errors.
            # Only consider exit code 0 or 1 as valid execution for coverage reporting.
            if return_code not in (0, 1):
                logger.warning(f"Pytest coverage command failed with return code {return_code}. Stderr: {stderr}")
                # Provide more detailed error info, including stdout for debugging.
                coverage_data["coverage_details"] = f"Pytest command failed with exit code {return_code}. Stderr: {stderr or 'Not available'}. Stdout: {stdout or 'Not available'}."
                return coverage_data

            coverage_json_path = Path("coverage.json")
            # Check if the command actually produced the coverage.json file
            # and if the return code indicates a successful or partially successful run (0 or 1 for pytest)
            if coverage_json_path.exists() and return_code in (0, 1):
                with open(coverage_json_path, "r", encoding="utf-8") as f:
                    report = json.load(f)
                
                coverage_data["overall_coverage_percentage"] = report.get("totals", {}).get("percent_covered", 0.0)
                coverage_data["covered_statements"] = report.get("totals", {}).get("covered_statements", 0)
                coverage_data["total_files"] = report.get("totals", {}).get("num_statements", 0)
                coverage_data["total_python_files_analyzed"] = len(report.get("files", {}))
                coverage_data["files_covered_count"] = sum(1 for file_report in report.get("files", {}).values() if file_report.get("percent_covered", 0) > 0)

                coverage_data["coverage_details"] = "Coverage report generated successfully."
                # NEW: Add a note if tests failed, even if coverage command ran
                if return_code == 1:
                    coverage_data["coverage_details"] += " Note: Some tests failed during coverage collection."
                coverage_json_path.unlink()
            elif return_code not in (0, 1): # If command failed with unexpected code
                coverage_data["coverage_details"] = "Coverage JSON report not found."

        except Exception as e:
            logger.error(f"Error assessing test coverage: {e}", exc_info=True)
            coverage_data["coverage_details"] = f"Error during coverage assessment: {e}"

        return coverage_data

    @classmethod
    def _analyze_python_file_ast(
        cls, content: str, content_lines: List[str], file_path: str
    ) -> List[Dict[str, Any]]:
        """
        Analyzes a Python file's AST for complexity, lines of code in functions,
        number of functions, code smells, and potential bottlenecks.
        """
        try:
            tree = ast.parse(content)
            visitor = ComplexityVisitor(content_lines)
            visitor.visit(tree)
            return visitor.function_metrics
        except SyntaxError as e:
            logger.error(f"Syntax error in {file_path} during AST analysis: {e}")
            return []
        except Exception as e:
            logger.error(
                f"Unexpected error during AST analysis for {file_path}: {e}",
                exc_info=True,
            )
            return []

    def _collect_configuration_analysis(self, codebase_path: str):
        """Collects structured information about existing tool configurations."""
        # This method is already implemented in FocusedMetricsCollector,
        # but CodebaseScanner needs to call it to gather the data.
        # We'll call the static method from FocusedMetricsCollector.
        from src.self_improvement.metrics_collector import FocusedMetricsCollector
        return FocusedMetricsCollector._collect_configuration_analysis(codebase_path)

    def _collect_deployment_robustness_metrics(self, codebase_path: str):
        """Collects metrics related to deployment robustness."""
        # This method is already implemented in FocusedMetricsCollector,
        # but CodebaseScanner needs to call it to gather the data.
        # We'll call the static method from FocusedMetricsCollector.
        from src.self_improvement.metrics_collector import FocusedMetricsCollector
        return FocusedMetricsCollector._collect_deployment_robustness_metrics(codebase_path)


# --- ContextRelevanceAnalyzer Class ---
class ContextRelevanceAnalyzer:
    """
    Analyzes the relevance of codebase context to the prompt and personas,
    using semantic search and keyword matching.
    """

    def __init__(
        self, codebase_context: Dict[str, Any], cache_dir: str
    ):
        """
        Initializes the analyzer.
        MODIFIED: codebase_context now expects the richer dict from CodebaseScanner.
        """
        self.cache_dir = cache_dir
        self.codebase_context = codebase_context if codebase_context is not None else {}
        # NEW: Extract file_contents for embedding computation
        self.file_contents = self.codebase_context.get("file_contents", {})
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

        # MODIFIED: Compute embeddings using self.file_contents
        if self.file_contents:
            self.file_embeddings = self._compute_file_embeddings(self.file_contents)
        else:
            self.file_embeddings = {}

    def set_persona_router(self, persona_router: Any):
        """Sets the persona router for context relevance scoring."""
        self.persona_router = persona_router
        self.logger.info("Persona router set for context relevance analysis.")

    def _compute_file_embeddings(self, file_contents: Dict[str, str]) -> Dict[str, Any]:
        """Computes embeddings for files in the codebase context."""
        if not file_contents:
            return {}

        embeddings = {}
        try:
            files_with_content = {k: v for k, v in file_contents.items() if v}
            if not files_with_content:
                self.logger.warning("No file content found in context for embedding.")
                return {}

            file_paths = list(files_with_content.keys())
            file_contents_list = list(files_with_content.values())

            self.logger.info(f"Computing embeddings for {len(file_paths)} files...")
            if not hasattr(self, "model") or self.model is None:
                raise RuntimeError("SentenceTransformer model not loaded.")

            file_embeddings_list = self.model.encode(file_contents_list)

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

        # MODIFIED: Access file_structure from self.codebase_context
        if self.codebase_context.get("file_structure") and "critical_files_preview" in self.codebase_context["file_structure"]:
            summary += "\nCritical Files Preview:\n"
            for filename, snippet in self.codebase_context["file_structure"][
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
        # MODIFIED: Check for 'file_contents' key in codebase_context
        if self.codebase_context.get("file_contents"):
            return f"Codebase context available ({len(self.codebase_context.get('file_contents', {}))} files). See details in intermediate steps."
        return "No codebase context provided or scanned."