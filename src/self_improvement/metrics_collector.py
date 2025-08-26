# src/self_improvement/metrics_collector.py
import os
import json
import subprocess
import ast
import logging
from typing import Dict, Any, List, Tuple, Union # ADDED Union import
from collections import defaultdict
from pathlib import Path

# Import existing validation functions to reuse their logic
from src.utils.code_validator import _run_pycodestyle, _run_bandit, _run_ast_security_checks

logger = logging.getLogger(__name__)

# --- NEW: AST Visitor for detailed code metrics ---
class ComplexityVisitor(ast.NodeVisitor):
    """
    AST visitor to calculate various code metrics for functions and methods,
    including cyclomatic complexity, lines of code, nesting depth, and code smells.
    """
    def __init__(self, content_lines: List[str]):
        self.content_lines = content_lines
        self.function_metrics = [] # Stores metrics for each function/method
        self.current_function_name = None
        self.current_function_start_line = None

    def _calculate_loc(self, node: ast.AST) -> int:
        """Calculates non-blank, non-comment lines of code within a node's body."""
        if not hasattr(node, 'body') or not node.body:
            return 0
        
        # Ensure node has lineno and end_lineno (available in Python 3.8+)
        if not hasattr(node.body[0], 'lineno') or not hasattr(node.body[-1], 'end_lineno'):
            # Fallback for older Python versions or nodes without line info
            return 0 
            
        start_line = node.body[0].lineno
        end_line = node.body[-1].end_lineno
        
        loc_count = 0
        # Iterate through lines within the function's body
        for i in range(start_line - 1, end_line):
            if i < len(self.content_lines): # Ensure index is within bounds
                line = self.content_lines[i].strip()
                if line and not line.startswith('#'): # Count non-blank, non-comment lines
                    loc_count += 1
        return loc_count

    def visit_FunctionDef(self, node: ast.FunctionDef):
        """Visits a synchronous function definition."""
        self._analyze_function(node)
        self.generic_visit(node) # Continue traversal to nested nodes

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef):
        """Visits an asynchronous function definition."""
        self._analyze_function(node)
        self.generic_visit(node) # Continue traversal to nested nodes

    def _analyze_function(self, node: Union[ast.FunctionDef, ast.AsyncFunctionDef]):
        """
        Performs detailed analysis for a given function or async function node.
        Calculates cyclomatic complexity, LOC, argument count, nesting depth,
        and identifies basic code smells and potential bottlenecks.
        """
        function_name = node.name
        start_line = node.lineno
        end_line = node.end_lineno # Python 3.8+
        
        complexity = 1 # Start with 1 for the function's entry point (standard for cyclomatic complexity)
        max_nesting_depth = 0
        
        nested_loops_count = 0
        
        # Stack to track block-level nodes for nesting depth and nested loop detection
        stack = []

        for sub_node in ast.walk(node):
            # Cyclomatic Complexity points (each decision point adds 1)
            if isinstance(sub_node, (ast.If, ast.For, ast.While, ast.AsyncFor, ast.With, ast.AsyncWith, ast.ExceptHandler)):
                complexity += 1
            elif isinstance(sub_node, ast.BoolOp): # 'and', 'or' operators in conditions
                complexity += len(sub_node.values) - 1
            elif isinstance(sub_node, ast.comprehension) and sub_node.ifs: # Conditional comprehensions (e.g., [x for x in y if x > 0])
                complexity += len(sub_node.ifs)

            # Nesting depth calculation
            # Increment depth when entering a new block-level node within the current function
            if isinstance(sub_node, (ast.If, ast.For, ast.While, ast.AsyncFor, ast.With, ast.AsyncWith, ast.ExceptHandler, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                # Only consider nodes that are children of the current function node
                # and are not the function node itself.
                if sub_node != node and sub_node not in stack:
                    stack.append(sub_node)
                    current_nesting_depth = len(stack)
                    max_nesting_depth = max(max_nesting_depth, current_nesting_depth)
            
            # Nested loops detection
            if isinstance(sub_node, (ast.For, ast.While, ast.AsyncFor)):
                # Check if this loop is inside another loop (i.e., there's another loop in the stack before it)
                if any(isinstance(s, (ast.For, ast.While, ast.AsyncFor)) for s in stack[:-1]):
                    nested_loops_count += 1
        
        # After walking the function's subtree, clear the stack for this function's context
        stack.clear()

        loc = self._calculate_loc(node)
        # Count arguments including positional-only, keyword-only, and regular arguments
        num_args = len(node.args.args) + len(node.args.posonlyargs) + len(node.args.kwonlyargs)

        # Code Smells (illustrative thresholds, can be configured externally)
        code_smells = 0
        if loc > 50: # Long function
            code_smells += 1
        if num_args > 5: # Too many arguments
            code_smells += 1
        if max_nesting_depth > 3: # Deep nesting
            code_smells += 1
        
        # Potential Bottlenecks (illustrative)
        bottlenecks = 0
        if nested_loops_count > 0: # Any nested loops are a potential bottleneck
            bottlenecks += 1
        # Further checks could include:
        # - Excessive recursion (requires more complex call graph analysis)
        # - Large list/dict comprehensions that might be inefficient
        
        self.function_metrics.append({
            "name": function_name,
            "start_line": start_line,
            "end_line": end_line,
            "loc": loc,
            "cyclomatic_complexity": complexity,
            "num_arguments": num_args,
            "max_nesting_depth": max_nesting_depth,
            "nested_loops_count": nested_loops_count,
            "code_smells": code_smells,
            "potential_bottlenecks": bottlenecks
        })

# --- END NEW: AST Visitor for detailed code metrics ---

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
                "complexity_metrics": {
                    "avg_cyclomatic_complexity": 0.0,
                    "avg_loc_per_function": 0.0,
                    "avg_num_arguments": 0.0,
                    "avg_max_nesting_depth": 0.0
                },
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

        total_functions_across_codebase = 0
        total_loc_across_functions = 0
        total_complexity_across_functions = 0
        total_args_across_functions = 0
        total_nesting_depth_across_functions = 0
        
        # Collect code-specific metrics by iterating through Python files
        for root, _, files in os.walk(codebase_path):
            for file in files:
                if file.endswith('.py'):
                    file_path = os.path.join(root, file)
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            content = f.read()
                            content_lines = content.splitlines() # Pass lines for LOC calculation
                        
                        # Reuse existing code_validator functions
                        pep8_issues = _run_pycodestyle(content, file_path)
                        if pep8_issues:
                            metrics["code_quality"]["pep8_issues_count"] += len(pep8_issues)
                            metrics["code_quality"]["detailed_issues"].extend(pep8_issues)
                        
                        bandit_issues = _run_bandit(content, file_path)
                        if bandit_issues:
                            metrics["security"]["bandit_issues_count"] += len(bandit_issues)
                            metrics["code_quality"]["detailed_issues"].extend(bandit_issues) # Add to detailed issues for full context
                        
                        ast_security_issues = _run_ast_security_checks(content, file_path)
                        if ast_security_issues:
                            metrics["security"]["ast_security_issues_count"] += len(ast_security_issues)
                            metrics["code_quality"]["detailed_issues"].extend(ast_security_issues) # Add to detailed issues

                        # Collect complexity and code smell metrics using the new AST visitor
                        file_function_metrics = cls._analyze_python_file_ast(content, content_lines, file_path)
                        
                        for func_metric in file_function_metrics:
                            total_functions_across_codebase += 1
                            total_complexity_across_functions += func_metric["cyclomatic_complexity"]
                            total_loc_across_functions += func_metric["loc"]
                            total_args_across_functions += func_metric["num_arguments"]
                            total_nesting_depth_across_functions += func_metric["max_nesting_depth"]
                            metrics["code_quality"]["code_smells_count"] += func_metric["code_smells"]
                            metrics["performance_efficiency"]["potential_bottlenecks_count"] += func_metric["potential_bottlenecks"]

                    except Exception as e:
                        logger.error(f"Error collecting code metrics for {file_path}: {e}", exc_info=True)
        
        if total_functions_across_codebase > 0:
            metrics["code_quality"]["complexity_metrics"]["avg_cyclomatic_complexity"] = total_complexity_across_functions / total_functions_across_codebase
            metrics["code_quality"]["complexity_metrics"]["avg_loc_per_function"] = total_loc_across_functions / total_functions_across_codebase
            metrics["code_quality"]["complexity_metrics"]["avg_num_arguments"] = total_args_across_functions / total_functions_across_codebase
            metrics["code_quality"]["complexity_metrics"]["avg_max_nesting_depth"] = total_nesting_depth_across_functions / total_functions_across_codebase
        
        return metrics

    @classmethod
    def _collect_token_usage_stats(cls, debate_intermediate_steps: Dict[str, Any]) -> Dict[str, Any]:
        """
        Collects token usage statistics from debate intermediate steps.
        """
        total_tokens = debate_intermediate_steps.get("Total_Tokens_Used", 0)
        total_cost = debate_intermediate_steps.get("Total_Estimated_Cost_USD", 0.0)
        
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
        return {
            "overall_coverage_percentage": 0.0, # Cannot calculate without running tests
            "files_covered": 0,
            "total_files": 0,
            "coverage_details": "Automated test coverage assessment not implemented."
        }

    @classmethod
    def _analyze_python_file_ast(cls, content: str, content_lines: List[str], file_path: str) -> List[Dict[str, Any]]:
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
            logger.error(f"Unexpected error during AST analysis for {file_path}: {e}", exc_info=True)
            return []