# src/utils/code_utils.py
"""
Utility functions related to code manipulation and analysis.
"""

import logging
import re
import ast
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any, Union

logger = logging.getLogger(__name__)


# --- Helper to get a snippet around a line number ---
def _get_code_snippet(
    content_lines: List[str], line_number: Optional[int], context_lines: int = 2
) -> str:  # Changed return type from Optional[str] to str
    """
    Retrieves a snippet of code around a specific line number from a list of lines.

    Args:
        content_lines: A list of strings, where each string is a line of code.
        line_number: The 1-based line number to center the snippet around.
        context_lines: The number of lines before and after the target line to include.

    Returns:
        A formatted string snippet of the code, or an empty string if input is invalid.
    """
    # Ensure line_number is valid and within bounds
    if (
        line_number is None
        or not content_lines
        or line_number < 1
        or line_number > len(content_lines)
    ):
        return ""  # Return empty string instead of None

    # Adjust line_number to be 0-indexed for list access
    actual_line_idx = line_number - 1

    # Determine the start and end indices for the snippet
    start_idx = max(0, actual_line_idx - context_lines)
    # Ensure end_idx is within bounds and includes the target line + context lines
    end_idx = min(len(content_lines), actual_line_idx + context_lines + 1)

    snippet_lines = []
    for i in range(start_idx, end_idx):
        # Add 1 to i to display 1-indexed line numbers for clarity
        snippet_lines.append(
            f"{i + 1}: {content_lines[i].rstrip()}"
        )  # rstrip to remove trailing newlines

    return "\n".join(snippet_lines)


# --- AST Visitor for detailed code metrics (if needed elsewhere, otherwise can be removed) ---
class ComplexityVisitor(ast.NodeVisitor):
    """
    AST visitor to calculate various code metrics for functions and methods,
    including cyclomatic complexity, lines of code, nesting depth, and code smells.
    """

    def __init__(self, content_lines: List[str]):
        self.content_lines = content_lines
        self.function_metrics = []  # Stores metrics for each function/method
        self.current_function_name = None
        self.current_function_start_line = None

    def _calculate_loc(self, node: ast.AST) -> int:
        """Calculates non-blank, non-comment lines of code within a node's body."""
        if not hasattr(node, "body") or not node.body:
            return 0

        # AST nodes might not always have end_lineno, especially for simple statements.
        # Fallback to lineno if end_lineno is missing.
        start_line = node.lineno
        end_line = getattr(node.body[-1], "end_lineno", node.body[-1].lineno)

        loc_count = 0
        for i in range(start_line - 1, end_line):
            if i < len(self.content_lines):
                line = self.content_lines[i].strip()
                if line and not line.startswith("#"):
                    loc_count += 1
        return loc_count

    def visit_FunctionDef(self, node: ast.FunctionDef):
        self._analyze_function(node)
        self.generic_visit(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef):
        self._analyze_function(node)
        self.generic_visit(node)

    def _analyze_function(self, node: Union[ast.FunctionDef, ast.AsyncFunctionDef]):
        function_name = node.name
        start_line = node.lineno
        end_line = getattr(
            node, "end_lineno", node.lineno
        )  # Use lineno if end_lineno is missing

        complexity = 1
        max_nesting_depth = 0
        nested_loops_count = 0
        stack = []

        for sub_node in ast.walk(node):
            # Check for control flow statements that increase complexity
            if isinstance(
                sub_node,
                (
                    ast.If,
                    ast.For,
                    ast.While,
                    ast.AsyncFor,
                    ast.With,
                    ast.AsyncWith,
                    ast.ExceptHandler,
                ),
            ):
                complexity += 1
            # Check for boolean operations (e.g., 'and', 'or') that increase complexity
            elif isinstance(sub_node, ast.BoolOp):
                complexity += len(sub_node.values) - 1
            # Check for comprehensions with conditional clauses
            elif isinstance(sub_node, ast.comprehension) and sub_node.ifs:
                complexity += len(sub_node.ifs)

            # Track nesting depth
            if isinstance(
                sub_node,
                (
                    ast.If,
                    ast.For,
                    ast.While,
                    ast.AsyncFor,
                    ast.With,
                    ast.AsyncWith,
                    ast.ExceptHandler,
                    ast.FunctionDef,
                    ast.AsyncFunctionDef,
                    ast.ClassDef,
                ),
            ):
                # Avoid counting the current node itself in depth calculation
                if sub_node != node and sub_node not in stack:
                    stack.append(sub_node)
                    current_nesting_depth = len(stack)
                    max_nesting_depth = max(max_nesting_depth, current_nesting_depth)

            # Count nested loops
            if isinstance(sub_node, (ast.For, ast.While, ast.AsyncFor)):
                # Check if any parent node in the stack is also a loop/conditional
                if any(
                    isinstance(s, (ast.For, ast.While, ast.AsyncFor))
                    for s in stack[:-1]
                ):
                    nested_loops_count += 1

        stack.clear()  # Clear stack after visiting the function node
        loc = self._calculate_loc(node)
        num_args = (
            len(node.args.args)
            + len(node.args.posonlyargs)
            + len(node.args.kwonlyargs)
            + (1 if node.args.vararg else 0)
            + (1 if node.args.kwarg else 0)
        )

        # Simple code smell heuristics
        code_smells = 0
        if loc > 50:  # Long function
            code_smells += 1
        if num_args > 5:  # High number of arguments
            code_smells += 1
        if max_nesting_depth > 3:  # Deeply nested logic
            code_smells += 1

        # Placeholder for potential performance bottlenecks (e.g., nested loops)
        potential_bottlenecks = 0
        if nested_loops_count > 0:
            potential_bottlenecks += 1

        self.function_metrics.append(
            {
                "name": function_name,
                "start_line": start_line,
                "end_line": end_line,
                "loc": loc,
                "cyclomatic_complexity": complexity,
                "num_arguments": num_args,
                "max_nesting_depth": max_nesting_depth,
                "nested_loops_count": nested_loops_count,
                "code_smells": code_smells,
                "potential_bottlenecks": potential_bottlenecks,
            }
        )
