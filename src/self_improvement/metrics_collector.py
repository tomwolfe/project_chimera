# src/self_improvement/metrics_collector.py
import os
import json
import subprocess
import ast
import logging
from typing import Dict, Any, List, Tuple
from collections import defaultdict
from pathlib import Path

# Import existing validation functions to reuse their logic
from src.utils.code_validator import _run_pycodestyle, _run_bandit, _run_ast_security_checks

logger = logging.getLogger(__name__)

class ImprovementMetricsCollector:
    """Collects objective metrics for self-improvement analysis."""
    
    @classmethod
    def collect_all_metrics(cls, codebase_path: str, debate_intermediate_steps: Dict[str, Any]) -> Dict[str, Any]:
        """
        Collect all relevant metrics from the codebase and debate history for self-improvement analysis.
        """
        metrics = {
            "code_quality": {
                "pep8_issues_count": 0,
                "complexity_metrics": {"avg_cyclomatic_complexity": 0.0, "avg_loc_per_function": 0.0},
                "code_smells_count": 0,
                "detailed_issues": [] # To store all collected issues for detailed analysis
            },
            "security": {
                "bandit_issues_count": 0,
                "ast_security_issues_count": 0,
            },
            "performance_efficiency": {
                "token_usage_stats": cls._collect_token_usage_stats(debate_intermediate_steps),
                "debate_efficiency_summary": cls._analyze_debate_efficiency(debate_intermediate_steps),
                "potential_bottlenecks_count": 0 # Count of detected potential bottlenecks
            },
            "robustness": {
                "schema_validation_failures_count": len(debate_intermediate_steps.get("malformed_blocks", [])),
                "unresolved_conflict_present": bool(debate_intermediate_steps.get("Unresolved_Conflict")),
                "conflict_resolution_attempted": bool(debate_intermediate_steps.get("Conflict_Resolution_Attempt"))
            },
            "maintainability": {
                "test_coverage_summary": cls._assess_test_coverage(codebase_path)
            }
        }

        total_functions = 0
        total_loc_in_functions = 0
        total_complexity = 0
        
        # Collect code-specific metrics by iterating through Python files
        for root, _, files in os.walk(codebase_path):
            for file in files:
                if file.endswith('.py'):
                    file_path = os.path.join(root, file)
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            content = f.read()
                        
                        # Reuse existing code_validator functions
                        pep8_issues = _run_pycodestyle(content, file_path)
                        if pep8_issues:
                            metrics["code_quality"]["pep8_issues_count"] += len(pep8_issues)
                            metrics["code_quality"]["detailed_issues"].extend(pep8_issues)
                        
                        bandit_issues = _run_bandit(content, file_path)
                        if bandit_issues:
                            metrics["security"]["bandit_issues_count"] += len(bandit_issues)
                            metrics["code_quality"]["detailed_issues"].extend(bandit_issues) # Add to detailed issues for full context
                        
                        ast_issues = _run_ast_security_checks(content, file_path)
                        if ast_issues:
                            metrics["security"]["ast_security_issues_count"] += len(ast_issues)
                            metrics["code_quality"]["detailed_issues"].extend(ast_issues) # Add to detailed issues

                        # Collect complexity and code smell metrics
                        file_complexity, file_loc, file_functions, file_smells, file_bottlenecks = cls._analyze_python_file_ast(content, file_path)
                        total_complexity += file_complexity
                        total_loc_in_functions += file_loc
                        total_functions += file_functions
                        metrics["code_quality"]["code_smells_count"] += file_smells
                        metrics["performance_efficiency"]["potential_bottlenecks_count"] += file_bottlenecks

                    except Exception as e:
                        logger.error(f"Error collecting code metrics for {file_path}: {e}")
        
        if total_functions > 0:
            metrics["code