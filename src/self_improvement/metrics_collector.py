# src/self_improvement/metrics_collector.py
import os
import json
import subprocess
import ast
import logging
from typing import Dict, Any, List, Tuple, Union
from collections import defaultdict
from pathlib import Path

# Import existing validation functions to reuse their logic
from src.utils.code_validator import _run_ruff, _run_bandit, _run_ast_security_checks
from src.models import ConfigurationAnalysisOutput, CiWorkflowConfig, CiWorkflowJob, CiWorkflowStep, PreCommitHook, PyprojectTomlConfig, RuffConfig, BanditConfig, PydanticSettingsConfig, DeploymentAnalysisOutput # NEW IMPORTS: DeploymentAnalysisOutput
import yaml # Added for YAML parsing
import toml # Added for TOML parsing
from pydantic import ValidationError # Added for Pydantic validation in parsing
from src.utils.command_executor import execute_command_safely # Re-import for clarity
from src.utils.path_utils import PROJECT_ROOT # Re-import for clarity

logger = logging.getLogger(__name__)

# Placeholder for PEP8 descriptions. In a real scenario, this would be a comprehensive mapping.
# Keeping this for now, but Ruff's messages are often more descriptive directly.
PEP8_DESCRIPTIONS = {
    "E101": "Indentation contains mixed spaces and tabs",
    "E111": "Indentation is not a multiple of four",
    "E114": "Indentation is not a multiple of four (comment)",
    "E117": "Over-indented",
    "E121": "Continuation line under-indented for hanging indent",
    "E122": "Continuation line missing indentation or outdented",
    "E123": "Closing bracket does not match indentation of opening bracket's line",
    "E124": "Closing bracket does not match visual indentation",
    "E125": "Continuation line with same indent as next logical line",
    "E126": "Continuation line over-indented for hanging indent",
    "E127": "Continuation line over-indented for visual indent",
    "E128": "Continuation line under-indented for visual indent",
    "E129": "Visual indentation is not a multiple of four",
    "E131": "Continuation line unaligned for hanging indent",
    "E133": "First argument on line not indented",
    "E201": "Whitespace after '('",
    "E202": "Whitespace before ')'",
    "E203": "Whitespace before ':'",
    "E211": "Whitespace before '['",
    "E221": "Multiple spaces before operator",
    "E222": "Multiple spaces after operator",
    "E225": "Missing whitespace around operator",
    "E226": "Missing whitespace around arithmetic operator",
    "E227": "Missing whitespace around bitwise or shift operator",
    "E228": "Missing whitespace around modulo operator",
    "E231": "Missing whitespace after ','",
    "E251": "Unexpected whitespace around keyword / parameter equals",
    "E261": "At least two spaces before inline comment",
    "E262": "Inline comment should start with '# '",
    "E265": "Block comment should start with '# '",
    "E266": "Too many leading '#' for block comment",
    "E271": "Multiple spaces after keyword",
    "E272": "Multiple spaces before keyword",
    "E301": "Expected 1 blank line, found 0 (before class/def)",
    "E302": "Expected 2 blank lines, found 0 (before class/def)",
    "E303": "Too many blank lines (3 or more)",
    "E304": "Blank lines found after function decorator",
    "E305": "Expected 2 blank lines after class or function definition, found 0",
    "E306": "Expected 1 blank line before nested class or function definition, found 0",
    "E401": "Multiple imports on one line",
    "E402": "Module level import not at top of file",
    "E501": "Line too long",
    "E502": "The backslash is redundant between brackets",
    "E701": "Multiple statements on one line (colon)",
    "E702": "Multiple statements on one line (semicolon)",
    "E703": "Statement ends with a semicolon",
    "E711": "Comparison to None should be 'if cond is None:'",
    "E712": "Comparison to True should be 'if cond is True:' or 'if cond:'",
    "E713": "Test for membership should be 'not in'",
    "E714": "Test for object identity should be 'is not'",
    "E721": "Do not compare types, use isinstance()",
    "E722": "Do not use bare 'except:'",
    "E731": "Do not assign a lambda expression, use a def",
    "E741": "Ambiguous variable name 'l', 'O', or 'I'",
    "E742": "Ambiguous class name 'l', 'O', or 'I'",
    "E743": "Ambiguous function name 'l', 'O', or 'I'",
    "E901": "SyntaxError or IndentationError",
    "E902": "IOError",
    "W191": "Visual indentation contains mixed spaces and tabs",
    "W291": "Trailing whitespace",
    "W292": "No newline at end of file",
    "W293": "Blank line contains whitespace",
    "W391": "Blank line at end of file",
    "W503": "Line break before binary operator",
    "W504": "Line break after binary operator",
    "W601": "Invalid escape sequence 'x'",
    "W602": "Deprecated form of raising exception",
    "W603": "Invalid comparison with '== None'",
    "W604": "Backticks are deprecated",
    "W605": "Invalid escape sequence 'x'",
    "W606": "f-string contains backslash",
    "W607": "Invalid escape sequence 'x'"
}


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
    def _collect_configuration_analysis(cls, codebase_path: str) -> ConfigurationAnalysisOutput: # Return type is now the Pydantic model
        """
        Collects structured information about existing tool configurations from
        critical project configuration files.
        """
        config_analysis_data = {
            "ci_workflow": {},
            "pre_commit_hooks": [],
            "pyproject_toml": {}
        }
        malformed_blocks = []

        # 1. Analyze .github/workflows/ci.yml
        ci_yml_path = Path(codebase_path) / ".github/workflows/ci.yml"
        if ci_yml_path.exists():
            try:
                with open(ci_yml_path, 'r', encoding='utf-8') as f:
                    ci_config_raw = yaml.safe_load(f)
                    ci_workflow_jobs = {}
                    for job_name, job_details in ci_config_raw.get("jobs", {}).items():
                        steps_summary = []
                        for step in job_details.get("steps", []):
                            step_name = step.get("name", "Unnamed Step")
                            step_run = step.get("run")
                            step_uses = step.get("uses")
                            
                            summary_item_data = {"name": step_name}
                            if step_uses:
                                summary_item_data["uses"] = step_uses
                            if step_run:
                                commands = [cmd.strip() for cmd in step_run.split('\n') if cmd.strip()]
                                summary_item_data["runs_commands"] = commands
                            steps_summary.append(CiWorkflowStep(**summary_item_data))
                        ci_workflow_jobs[job_name] = CiWorkflowJob(steps_summary=steps_summary)
                    
                    config_analysis_data["ci_workflow"] = CiWorkflowConfig(
                        name=ci_config_raw.get("name"),
                        on_triggers=ci_config_raw.get("on"),
                        jobs=ci_workflow_jobs
                    )
            except (yaml.YAMLError, OSError, ValidationError) as e:
                logger.error(f"Error parsing CI workflow file {ci_yml_path}: {e}")
                malformed_blocks.append({"type": "CI_CONFIG_PARSE_ERROR", "message": str(e), "file": str(ci_yml_path)})

        # 2. Analyze .pre-commit-config.yaml
        pre_commit_path = Path(codebase_path) / ".pre-commit-config.yaml"
        if pre_commit_path.exists():
            try:
                with open(pre_commit_path, 'r', encoding='utf-8') as f:
                    pre_commit_config_raw = yaml.safe_load(f)
                    for repo_config in pre_commit_config_raw.get("repos", []):
                        repo_url = repo_config.get("repo")
                        repo_rev = repo_config.get("rev")
                        for hook in repo_config.get("hooks", []):
                            hook_id = hook.get("id")
                            hook_args = hook.get("args", [])
                            config_analysis_data["pre_commit_hooks"].append(
                                PreCommitHook(repo=repo_url, rev=repo_rev, id=hook_id, args=hook_args)
                            )
            except (yaml.YAMLError, OSError, ValidationError) as e:
                logger.error(f"Error parsing pre-commit config file {pre_commit_path}: {e}")
                malformed_blocks.append({"type": "PRE_COMMIT_CONFIG_PARSE_ERROR", "message": str(e), "file": str(pre_commit_path)})

        # 3. Analyze pyproject.toml
        pyproject_path = Path(codebase_path) / "pyproject.toml"
        if pyproject_path.exists():
            try:
                with open(pyproject_path, 'r', encoding='utf-8') as f:
                    pyproject_config_raw = toml.load(f)
                    pyproject_toml_data = {}

                    # Extract Ruff settings
                    ruff_tool_config = pyproject_config_raw.get("tool", {}).get("ruff", {})
                    if ruff_tool_config:
                        pyproject_toml_data["ruff"] = RuffConfig(
                            line_length=ruff_tool_config.get("line-length"),
                            target_version=ruff_tool_config.get("target-version"),
                            lint_select=ruff_tool_config.get("lint", {}).get("select"),
                            lint_ignore=ruff_tool_config.get("lint", {}).get("ignore"),
                            format_settings=ruff_tool_config.get("format")
                        )
                    # Extract Bandit settings
                    bandit_tool_config = pyproject_config_raw.get("tool", {}).get("bandit", {})
                    if bandit_tool_config:
                        pyproject_toml_data["bandit"] = BanditConfig(
                            exclude_dirs=bandit_tool_config.get("exclude_dirs"),
                            severity_level=bandit_tool_config.get("severity_level"),
                            confidence_level=bandit_tool_config.get("confidence_level"),
                            skip_checks=bandit_tool_config.get("skip_checks")
                        )
                    # Extract Pydantic settings if present
                    pydantic_settings_config = pyproject_config_raw.get("tool", {}).get("pydantic-settings", {})
                    if pydantic_settings_config:
                        pyproject_toml_data["pydantic_settings"] = PydanticSettingsConfig(**pydantic_settings_config)
                    
                    config_analysis_data["pyproject_toml"] = PyprojectTomlConfig(**pyproject_toml_data)

            except (toml.TomlDecodeError, OSError, ValidationError) as e:
                logger.error(f"Error parsing pyproject.toml file {pyproject_path}: {e}")
                malformed_blocks.append({"type": "PYPROJECT_CONFIG_PARSE_ERROR", "message": str(e), "file": str(pyproject_path)})

        return ConfigurationAnalysisOutput(
            ci_workflow=config_analysis_data["ci_workflow"],
            pre_commit_hooks=config_analysis_data["pre_commit_hooks"],
            pyproject_toml=config_analysis_data["pyproject_toml"],
            malformed_blocks=malformed_blocks
        )

    # NEW METHOD: Collect Deployment Robustness Metrics
    @classmethod
    def _collect_deployment_robustness_metrics(cls, codebase_path: str) -> DeploymentAnalysisOutput:
        """
        Collects metrics related to deployment robustness by analyzing Dockerfile
        and production requirements.
        """
        deployment_metrics_data = {
            "dockerfile_present": False,
            "dockerfile_healthcheck_present": False,
            "dockerfile_non_root_user": False,
            "dockerfile_exposed_ports": [],
            "dockerfile_multi_stage_build": False,
            "prod_requirements_present": False,
            "prod_dependency_count": 0,
            "dev_dependency_overlap_count": 0,
            "malformed_blocks": []
        }
        
        # 1. Analyze Dockerfile
        dockerfile_path = Path(codebase_path) / "Dockerfile"
        if dockerfile_path.exists():
            deployment_metrics_data["dockerfile_present"] = True
            try:
                with open(dockerfile_path, 'r', encoding='utf-8') as f:
                    dockerfile_content = f.read()
                
                if "HEALTHCHECK" in dockerfile_content:
                    deployment_metrics_data["dockerfile_healthcheck_present"] = True
                if re.search(r"USER\s+(?!root)", dockerfile_content, re.IGNORECASE):
                    deployment_metrics_data["dockerfile_non_root_user"] = True
                
                exposed_ports = re.findall(r"EXPOSE\s+(\d+)", dockerfile_content)
                deployment_metrics_data["dockerfile_exposed_ports"] = [int(p) for p in exposed_ports]

                if re.search(r"FROM\s+.*?AS\s+.*?\nFROM", dockerfile_content, re.DOTALL | re.IGNORECASE):
                    deployment_metrics_data["dockerfile_multi_stage_build"] = True

            except OSError as e:
                logger.error(f"Error reading Dockerfile {dockerfile_path}: {e}")
                deployment_metrics_data["malformed_blocks"].append({"type": "DOCKERFILE_READ_ERROR", "message": str(e), "file": str(dockerfile_path)})
        
        # 2. Analyze requirements-prod.txt and requirements.txt
        prod_req_path = Path(codebase_path) / "requirements-prod.txt"
        dev_req_path = Path(codebase_path) / "requirements.txt"

        prod_deps = set()
        if prod_req_path.exists():
            deployment_metrics_data["prod_requirements_present"] = True
            try:
                with open(prod_req_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        line = line.strip()
                        if line and not line.startswith('#'):
                            prod_deps.add(line.split('==')[0].split('>=')[0].split('~=')[0].lower())
                deployment_metrics_data["prod_dependency_count"] = len(prod_deps)
            except OSError as e:
                logger.error(f"Error reading requirements-prod.txt {prod_req_path}: {e}")
                deployment_metrics_data["malformed_blocks"].append({"type": "PROD_REQ_READ_ERROR", "message": str(e), "file": str(prod_req_path)})

        if dev_req_path.exists() and prod_req_path.exists(): # Only check overlap if both exist
            dev_deps = set()
            try:
                with open(dev_req_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        line = line.strip()
                        if line and not line.startswith('#'):
                            dev_deps.add(line.split('==')[0].split('>=')[0].split('~=')[0].lower())
                
                overlap = prod_deps.intersection(dev_deps)
                deployment_metrics_data["dev_dependency_overlap_count"] = len(overlap)
            except OSError as e:
                logger.error(f"Error reading requirements.txt {dev_req_path}: {e}")
                deployment_metrics_data["malformed_blocks"].append({"type": "DEV_REQ_READ_ERROR", "message": str(e), "file": str(dev_req_path)})

        return DeploymentAnalysisOutput(**deployment_metrics_data)

    # NEW METHOD: Collect Reasoning Quality Metrics
    @classmethod
    def _collect_reasoning_quality_metrics(cls, debate_intermediate_steps: Dict[str, Any]) -> Dict[str, Any]:
        """
        Collects metrics related to the quality of the Socratic debate process itself.
        Assumes `core.py` has been enhanced to log more granular data.
        """
        reasoning_metrics = {
            "total_debate_turns": 0,
            "unique_personas_involved": 0,
            "schema_validation_failures_count": 0, # From malformed_blocks
            "content_misalignment_warnings": 0, # From malformed_blocks
            "debate_turn_errors": 0, # From malformed_blocks
            "conflict_resolution_attempts": 0,
            "conflict_resolution_successes": 0,
            "unresolved_conflict_present": False,
            "average_persona_output_tokens": 0.0,
            "persona_specific_performance": defaultdict(lambda: {"success_rate": 0.0, "schema_failures": 0, "truncations": 0, "total_turns": 0}),
            "prompt_verbosity_score": 0.0, # Placeholder for future analysis
            "malformed_blocks_summary": defaultdict(int)
        }

        # Total debate turns (excluding context/synthesis setup)
        debate_history = debate_intermediate_steps.get("Debate_History", [])
        reasoning_metrics["total_debate_turns"] = len(debate_history)

        # Unique personas involved
        unique_personas = set()
        for turn in debate_history:
            if "persona" in turn:
                unique_personas.add(turn["persona"])
        reasoning_metrics["unique_personas_involved"] = len(unique_personas)

        # Malformed blocks analysis
        all_malformed_blocks = debate_intermediate_steps.get("malformed_blocks", [])
        reasoning_metrics["schema_validation_failures_count"] = sum(1 for b in all_malformed_blocks if b.get("type") == "SCHEMA_VALIDATION_ERROR")
        reasoning_metrics["content_misalignment_warnings"] = sum(1 for b in all_malformed_blocks if b.get("type") == "CONTENT_MISALIGNMENT")
        reasoning_metrics["debate_turn_errors"] = sum(1 for b in all_malformed_blocks if b.get("type") == "DEBATE_TURN_ERROR")
        
        for block in all_malformed_blocks:
            reasoning_metrics["malformed_blocks_summary"][block.get("type", "UNKNOWN_MALFORMED_BLOCK")] += 1

        # Conflict resolution
        if debate_intermediate_steps.get("Conflict_Resolution_Attempt"):
            reasoning_metrics["conflict_resolution_attempts"] = 1
            if debate_intermediate_steps["Conflict_Resolution_Attempt"].get("conflict_resolved"):
                reasoning_metrics["conflict_resolution_successes"] = 1
        reasoning_metrics["unresolved_conflict_present"] = bool(debate_intermediate_steps.get("Unresolved_Conflict"))

        # Average persona output tokens (sum of all persona outputs / total turns)
        total_output_tokens = 0
        for key, value in debate_intermediate_steps.items():
            if key.endswith("_Tokens_Used") and not key.startswith(("Total_", "context_", "synthesis_")):
                total_output_tokens += value
        
        if reasoning_metrics["total_debate_turns"] > 0:
            reasoning_metrics["average_persona_output_tokens"] = total_output_tokens / reasoning_metrics["total_debate_turns"]

        # Persona-specific performance (assuming core.py logs this, or using a simplified heuristic)
        # This would ideally come from `persona_manager.persona_performance_metrics` if passed through
        # or from detailed logs in `debate_intermediate_steps`.
        for persona_name in unique_personas:
            # Placeholder: In a real scenario, `core.py` would log `persona_name_Success`, `persona_name_SchemaFailures`, etc.
            # For now, we'll just count malformed blocks associated with a persona.
            persona_malformed_blocks = [b for b in all_malformed_blocks if b.get("persona") == persona_name]
            schema_failures = sum(1 for b in persona_malformed_blocks if b.get("type") == "SCHEMA_VALIDATION_ERROR")
            content_misalignments = sum(1 for b in persona_malformed_blocks if b.get("type") == "CONTENT_MISALIGNMENT")
            
            # Assuming each persona in `debate_history` represents a turn
            persona_turns = sum(1 for turn in debate_history if turn.get("persona") == persona_name)
            
            reasoning_metrics["persona_specific_performance"][persona_name]["total_turns"] = persona_turns
            reasoning_metrics["persona_specific_performance"][persona_name]["schema_failures"] = schema_failures
            reasoning_metrics["persona_specific_performance"][persona_name]["truncations"] = 0 # Not easily derivable from current intermediate_steps
            
            if persona_turns > 0:
                reasoning_metrics["persona_specific_performance"][persona_name]["success_rate"] = \
                    (persona_turns - schema_failures - content_misalignments) / persona_turns
            else:
                reasoning_metrics["persona_specific_performance"][persona_name]["success_rate"] = 0.0

        return reasoning_metrics


    @classmethod
    def collect_all_metrics(cls, codebase_path: str, debate_intermediate_steps: Dict[str, Any]) -> Dict[str, Any]:
        """
        Collect all relevant metrics from the codebase and debate history for self-improvement analysis.
        """
        metrics = {
            "code_quality": {
                "ruff_issues_count": 0,
                "complexity_metrics": {
                    "avg_cyclomatic_complexity": 0.0,
                    "avg_loc_per_function": 0.0,
                    "avg_num_arguments": 0.0,
                    "avg_max_nesting_depth": 0.0
                },
                "code_smells_count": 0,
                "detailed_issues": [], # To store all collected issues for detailed analysis
                "ruff_violations": []
            },
            "security": {
                "bandit_issues_count": 0,
                "ast_security_issues_count": 0,
            },
            "performance_efficiency": {
                "token_usage_stats": cls._collect_token_usage_stats(debate_intermediate_steps),
                "debate_efficiency_summary": cls._analyze_debate_efficiency(debate_intermediate_steps),
                "potential_bottlenecks_count": 0
            },
            "robustness": {
                "schema_validation_failures_count": len(debate_intermediate_steps.get("malformed_blocks", [])),
                "unresolved_conflict_present": bool(debate_intermediate_steps.get("Unresolved_Conflict")),
                "conflict_resolution_attempted": bool(debate_intermediate_steps.get("Conflict_Resolution_Attempt"))
            },
            "maintainability": {
                "test_coverage_summary": cls._assess_test_coverage(codebase_path)
            },
            "configuration_analysis": cls._collect_configuration_analysis(codebase_path).model_dump(by_alias=True), # NEW: Add configuration analysis
            "deployment_robustness": cls._collect_deployment_robustness_metrics(codebase_path).model_dump(by_alias=True), # NEW: Add deployment robustness analysis
            "reasoning_quality": cls._collect_reasoning_quality_metrics(debate_intermediate_steps) # NEW: Add reasoning quality metrics
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
                        ruff_issues = _run_ruff(content, file_path)
                        if ruff_issues:
                            metrics["code_quality"]["ruff_issues_count"] += len(ruff_issues)
                            metrics["code_quality"]["detailed_issues"].extend(ruff_issues)
                            metrics["code_quality"]["ruff_violations"].extend(ruff_issues) # NEW: Add to dedicated list

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
            "average_turn_tokens": 0.0,
            "persona_token_breakdown": {} # NEW: Per-persona token usage
        }

        total_debate_tokens = debate_intermediate_steps.get("debate_Tokens_Used", 0)
        num_turns = efficiency_summary["num_turns"]
        if num_turns > 0:
            efficiency_summary["average_turn_tokens"] = total_debate_tokens / num_turns

        # NEW: Collect per-persona token usage
        for key, value in debate_intermediate_steps.items():
            if key.endswith("_Tokens_Used") and not key.startswith(("Total_", "context_", "synthesis_", "debate_")):
                persona_name = key.replace("_Tokens_Used", "")
                efficiency_summary["persona_token_breakdown"][persona_name] = value

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

# MODIFIED: Renamed to be more generic for linter output (Ruff)
def analyze_linter_patterns(violations: List[Dict]) -> Dict:
    """Analyze linter violations to identify the most impactful patterns."""
    # Count frequency of each error code
    error_counts = {}
    file_impact = {}

    for violation in violations:
        code = violation['code']
        file_path = violation['file'] # Ruff output uses 'file', not 'filename'

        error_counts[code] = error_counts.get(code, 0) + 1

        if file_path not in file_impact:
            file_impact[file_path] = set()
        file_impact[file_path].add(code)

    # Calculate files impacted per error code
    code_file_counts = {}
    for file_path, codes in file_impact.items():
        for code in codes:
            code_file_counts[code] = code_file_counts.get(code, 0) + 1

    # Identify top patterns by both frequency and file impact
    top_patterns = []
    for code, count in sorted(error_counts.items(), key=lambda x: x[1], reverse=True)[:5]:
        files_affected = code_file_counts.get(code, 0)
        severity = count * files_affected
        # Ruff messages are often self-contained, so we might not need a separate description map
        # For now, use a generic description if not found in PEP8_DESCRIPTIONS
        description = next((v['message'] for v in violations if v['code'] == code), 'No specific description available from Ruff output.')
        top_patterns.append({
            'code': code,
            'count': count,
            'files_affected': files_affected,
            'severity': severity,
            'description': description # Use the message directly from Ruff output
        })

    return {
        'total_violations': len(violations),
        'top_patterns': top_patterns
    }

def prioritize_maintenance_tasks(metrics: Dict) -> List[Dict]:
    """Prioritize maintenance tasks using the 80/20 principle."""
    # Focus on the top linter patterns first
    # Use the dedicated 'ruff_violations' list from the metrics
    linter_analysis = analyze_linter_patterns(metrics['code_quality']['ruff_violations'])

    # Create actionable tasks for the most impactful patterns
    tasks = []
    for i, pattern in enumerate(linter_analysis['top_patterns']):
        # Calculate impact score (combining frequency and file spread)
        impact_score = pattern['count'] * pattern['files_affected']

        # Create specific remediation task
        tasks.append({
            'id': f'MAINT-00{i+1}',
            'title': f'Fix {pattern["code"]} violations ({pattern["count"]} occurrences)',
            'description': f'Resolve {pattern["code"]} ({pattern["description"]}) violations affecting {pattern["files_affected"]} files',
            'impact_score': impact_score,
            'estimated_effort': min(5, pattern['count'] // 100 + 1), # Simple heuristic for effort
            'priority': i + 1
        })

    # Sort by priority (lowest number = highest priority)
    return sorted(tasks, key=lambda x: x['priority'])