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
                        # Placeholder for now, as _analyze_python_file_ast is not provided in the original snippet
                        file_complexity, file_loc, file_functions, file_smells, file_bottlenecks = (0, 0, 0, 0, 0) # Default values
                        try:
                            file_complexity, file_loc, file_functions, file_smells, file_bottlenecks = cls._analyze_python_file_ast(content, file_path)
                        except NotImplementedError:
                            logger.warning(f"'_analyze_python_file_ast' not fully implemented, skipping detailed AST analysis for {file_path}.")
                        except Exception as ast_e:
                            logger.error(f"Error during AST analysis for {file_path}: {ast_e}")


                        total_complexity += file_complexity
                        total_loc_in_functions += file_loc
                        total_functions += file_functions
                        metrics["code_quality"]["code_smells_count"] += file_smells
                        metrics["performance_efficiency"]["potential_bottlenecks_count"] += file_bottlenecks

                    except Exception as e:
                        logger.error(f"Error collecting code metrics for {file_path}: {e}")
        
        if total_functions > 0:
            # This was the truncated line. Completing it based on the metrics dictionary structure.
            metrics["code_quality"]["complexity_metrics"]["avg_cyclomatic_complexity"] = total_complexity / total_functions
            metrics["code_quality"]["complexity_metrics"]["avg_loc_per_function"] = total_loc_in_functions / total_functions
        
        return metrics

    @classmethod
    def _collect_token_usage_stats(cls, debate_intermediate_steps: Dict[str, Any]) -> Dict[str, Any]:
        """
        Collects token usage statistics from debate intermediate steps.
        Placeholder implementation.
        """
        total_tokens = debate_intermediate_steps.get("Total_Tokens_Used", 0)
        total_cost = debate_intermediate_steps.get("Total_Estimated_Cost_USD", 0.0)
        
        # Example: breakdown by persona/phase if available in intermediate_steps
        phase_token_usage = {}
        for key, value in debate_intermediate_steps.items():
            if key.endswith("_Tokens_Used") and not key.startswith("Total_"):
                phase_name = key.replace("_Tokens_Used", "")
                phase_token_usage[phase_name] = value

        return {
            "total_tokens": total_tokens,
            "total_cost_usd": total_cost,
            "phase_token_usage": phase_token_usage
        }

    @classmethod
    def _analyze_debate_efficiency(cls, debate_intermediate_steps: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyzes the efficiency of the debate process.
        Placeholder implementation.
        """
        efficiency_summary = {
            "num_turns": len(debate_intermediate_steps.get("Debate_History", [])),
            "malformed_blocks_count": len(debate_intermediate_steps.get("malformed_blocks", [])),
            "conflict_resolution_attempts": 1 if debate_intermediate_steps.get("Conflict_Resolution_Attempt") else 0,
            "unresolved_conflict": bool(debate_intermediate_steps.get("Unresolved_Conflict")),
            "average_turn_tokens": 0.0
        }
        
        total_debate_tokens = debate_intermediate_steps.get("debate_Tokens_Used", 0)
        num_turns = efficiency_summary["num_turns"]
        if num_turns > 0:
            efficiency_summary["average_turn_tokens"] = total_debate_tokens / num_turns

        return efficiency_summary

    @classmethod
    def _assess_test_coverage(cls, codebase_path: str) -> Dict[str, Any]:
        """
        Assesses test coverage for the codebase.
        Placeholder implementation.
        """
        # In a real scenario, this would run a tool like 'coverage.py'
        # For now, return dummy data.
        return {
            "overall_coverage_percentage": 0.0, # Cannot calculate without running tests
            "files_covered": 0,
            "total_files": 0,
            "coverage_details": "Automated test coverage assessment not implemented."
        }

    @classmethod
    def _analyze_python_file_ast(cls, content: str, file_path: str) -> Tuple[int, int, int, int, int]:
        """
        Analyzes a Python file's AST for complexity, lines of code in functions,
        number of functions, code smells, and potential bottlenecks.
        Placeholder implementation.
        """
        # This is a complex task requiring AST traversal.
        # For a placeholder, we'll return zeros or simple counts.
        
        # Basic LOC count (excluding comments and blank lines)
        loc = sum(1 for line in content.splitlines() if line.strip() and not line.strip().startswith('#'))
        
        functions = 0
        complexity = 0 # Cyclomatic complexity
        code_smells = 0
        bottlenecks = 0

        try:
            tree = ast.parse(content)
            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    functions += 1
                    # Simple heuristic for complexity (e.g., number of if/for/while/try/except)
                    complexity += 1 + sum(isinstance(sub_node, (ast.If, ast.For, ast.While, ast.Try)) for sub_node in ast.walk(node))
                    
                    # Simple heuristic for code smells (e.g., long functions, too many arguments)
                    if len(node.body) > 50: # Arbitrary long function threshold
                        code_smells += 1
                    if len(node.args.args) > 5: # Arbitrary too many arguments threshold
                        code_smells += 1
                    
                    # Simple heuristic for potential bottlenecks (e.g., nested loops)
                    nested_loops = 0
                    for sub_node in ast.walk(node):
                        if isinstance(sub_node, (ast.For, ast.While)):
                            nested_loops += 1
                    if nested_loops >= 2: # Nested loops
                        bottlenecks += 1

        except SyntaxError as e:
            logger.error(f"Syntax error in {file_path} during AST analysis: {e}")
            # Return default values if syntax error prevents AST parsing
            return 0, 0, 0, 0, 0
        except Exception as e:
            logger.error(f"Unexpected error during AST analysis for {file_path}: {e}")
            return 0, 0, 0, 0, 0

        return complexity, loc, functions, code_smells, bottlenecks