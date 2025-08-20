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
            metrics["code_quality"]["complexity_metrics"]["avg_cyclomatic_complexity"] = round(total_complexity / total_functions, 2)
            metrics["code_quality"]["complexity_metrics"]["avg_loc_per_function"] = round(total_loc_in_functions / total_functions, 2)
        
        return metrics
    
    @staticmethod
    def _collect_token_usage_stats(intermediate_steps: Dict[str, Any]) -> Dict[str, Any]:
        """Collects token usage statistics from debate intermediate steps."""
        return {
            "total_tokens_consumed": intermediate_steps.get("Total_Tokens_Used", 0),
            "total_estimated_cost_usd": intermediate_steps.get("Total_Estimated_Cost_USD", 0.0),
            "context_phase_tokens": intermediate_steps.get("context_Tokens_Used", 0),
            "debate_phase_tokens": intermediate_steps.get("debate_Tokens_Used", 0),
            "synthesis_phase_tokens": intermediate_steps.get("synthesis_Tokens_Used", 0),
        }

    @staticmethod
    def _analyze_debate_efficiency(intermediate_steps: Dict[str, Any]) -> Dict[str, Any]:
        """Analyzes debate efficiency based on turns and outcomes."""
        debate_history = intermediate_steps.get("Debate_History", [])
        total_turns = len(debate_history)
        
        # Count turns with errors or malformed outputs
        error_turns = sum(1 for turn in debate_history if turn.get("error") or turn.get("output", {}).get("malformed_blocks"))
        
        return {
            "total_debate_turns": total_turns,
            "error_prone_turns": error_turns,
            "error_rate": round((error_turns / total_turns), 2) if total_turns > 0 else 0.0
        }

    @staticmethod
    def _assess_test_coverage(codebase_path: str) -> Dict[str, Any]:
        """Assesses test coverage by counting test files and test functions."""
        test_files_found = 0
        test_functions_found = 0
        
        for root, _, files in os.walk(codebase_path):
            for file in files:
                if file.startswith('test_') and file.endswith('.py'):
                    test_files_found += 1
                    file_path = os.path.join(root, file)
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            tree = ast.parse(f.read())
                        for node in ast.walk(tree):
                            if isinstance(node, ast.FunctionDef) and node.name.startswith('test_'):
                                test_functions_found += 1
                    except Exception as e:
                        logger.warning(f"Error parsing test file {file_path}: {e}")
        
        return {
            "test_files_found": test_files_found,
            "test_functions_found": test_functions_found,
            "note": "This is a heuristic based on file/function naming conventions, not actual execution coverage."
        }

    @staticmethod
    def _analyze_python_file_ast(content: str, file_path: str) -> Tuple[int, int, int, int, int]:
        """Analyzes a single Python file's AST for complexity, LOC, functions, smells, and bottlenecks."""
        try:
            tree = ast.parse(content)
            
            total_complexity = 0
            total_loc = 0
            total_functions = 0
            code_smells = 0
            potential_bottlenecks = 0

            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef) or isinstance(node, ast.AsyncFunctionDef):
                    total_functions += 1
                    # Basic LOC for function
                    func_loc = node.end_lineno - node.lineno + 1
                    total_loc += func_loc

                    # Basic Cyclomatic Complexity (counting decision points)
                    complexity = 1 # for the function itself
                    for sub_node in ast.walk(node):
                        if sub_node != node and isinstance(sub_node, (ast.If, ast.For, ast.While, ast.AsyncFor, ast.With, ast.AsyncWith, ast.ExceptHandler, ast.And, ast.Or)):
                            complexity += 1
                    total_complexity += complexity

                    # Code Smells: Long functions (heuristic)
                    if func_loc > 50: # Arbitrary threshold for long function
                        code_smells += 1
                        logger.debug(f"Code smell: Long function '{node.name}' in {file_path} (LOC: {func_loc})")
                    
                    # Code Smells: Too many arguments
                    if len(node.args.args) > 5: # Arbitrary threshold for too many arguments
                        code_smells += 1
                        logger.debug(f"Code smell: Too many arguments in '{node.name}' in {file_path} (Args: {len(node.args.args)})")

                # Potential Bottlenecks: Deeply nested loops (heuristic)
                if isinstance(node, (ast.For, ast.While, ast.AsyncFor)):
                    # Check nesting level (simplified: just count nested loops)
                    nested_loops = 0
                    for sub_node in ast.walk(node):
                        if sub_node != node and isinstance(sub_node, (ast.For, ast.While, ast.AsyncFor)):
                            nested_loops += 1
                    if nested_loops >= 2: # Two or more nested loops
                        potential_bottlenecks += 1
                        logger.debug(f"Potential bottleneck: Deeply nested loops in {file_path} at line {node.lineno}")

            return total_complexity, total_loc, total_functions, code_smells, potential_bottlenecks

        except SyntaxError as se:
            logger.warning(f"Syntax error in {file_path}, skipping AST analysis: {se}")
            return 0, 0, 0, 0, 0
        except Exception as e:
            logger.error(f"Unexpected error during AST analysis of {file_path}: {e}")
            return 0, 0, 0, 0, 0
