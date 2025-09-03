# src/context/context_analyzer.py
import os
import logging
from pathlib import Path
from typing import Dict, Any, List

logger = logging.getLogger(__name__)

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