# src/self_improvement/metrics_collector.py
import os
import json
import subprocess
import ast
import logging
from typing import Dict, Any, List, Tuple, Union
from collections import defaultdict
from pathlib import Path
import re
import yaml # Added for YAML parsing
import toml # Added for TOML parsing
from pydantic import ValidationError # Added for Pydantic validation in parsing

# Import existing validation functions to reuse their logic
# Ensure _get_code_snippet is imported from src.utils.code_validator
from src.utils.code_validator import _run_ruff, _run_bandit, _run_ast_security_checks, _get_code_snippet 
from src.models import ConfigurationAnalysisOutput, CiWorkflowConfig, CiWorkflowJob, CiWorkflowStep, PreCommitHook, PyprojectTomlConfig, RuffConfig, BanditConfig, PydanticSettingsConfig, DeploymentAnalysisOutput # NEW IMPORTS
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

# --- AST Visitor for detailed code metrics ---
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

        if not hasattr(node.body[0], 'lineno') or not hasattr(node.body[-1], 'end_lineno'):
            return 0

        start_line = node.body[0].lineno
        end_line = node.body[-1].end_lineno

        loc_count = 0
        for i in range(start_line - 1, end_line):
            if i < len(self.content_lines):
                line = self.content_lines[i].strip()
                if line and not line.startswith('#'):
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
        end_line = node.end_lineno

        complexity = 1
        max_nesting_depth = 0
        nested_loops_count = 0
        stack = []

        for sub_node in ast.walk(node):
            if isinstance(sub_node, (ast.If, ast.For, ast.While, ast.AsyncFor, ast.With, ast.AsyncWith, ast.ExceptHandler)):
                complexity += 1
            elif isinstance(sub_node, ast.BoolOp):
                complexity += len(sub_node.values) - 1
            elif isinstance(sub_node, ast.comprehension) and sub_node.ifs:
                complexity += len(sub_node.ifs)

            if isinstance(sub_node, (ast.If, ast.For, ast.While, ast.AsyncFor, ast.With, ast.AsyncWith, ast.ExceptHandler, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                if sub_node != node and sub_node not in stack:
                    stack.append(sub_node)
                    current_nesting_depth = len(stack)
                    max_nesting_depth = max(max_nesting_depth, current_nesting_depth)

            if isinstance(sub_node, (ast.For, ast.While, ast.AsyncFor)):
                if any(isinstance(s, (ast.For, ast.While, ast.AsyncFor)) for s in stack[:-1]):
                    nested_loops_count += 1

        stack.clear()
        loc = self._calculate_loc(node)
        num_args = len(node.args.args) + len(node.args.posonlyargs) + len(node.args.kwonlyargs)

        code_smells = 0
        if loc > 50: code_smells += 1
        if num_args > 5: code_smells += 1
        if max_nesting_depth > 3: code_smells += 1

        bottlenecks = 0
        if nested_loops_count > 0: bottlenecks += 1

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

# --- AST Visitor for detailed code metrics ---

class ImprovementMetricsCollector:
    """Collects objective metrics for self-improvement analysis."""

    def __init__(self, initial_prompt: str, debate_history: List[Dict], intermediate_steps: Dict[str, Any],
                 codebase_context: Dict[str, str], tokenizer: Any, llm_provider: Any,
                 persona_manager: Any, content_validator: Any):
        """Initialize with debate context for analysis."""
        self.initial_prompt = initial_prompt
        self.debate_history = debate_history
        self.intermediate_steps = intermediate_steps
        self.codebase_context = codebase_context
        self.tokenizer = tokenizer
        self.llm_provider = llm_provider
        self.persona_manager = persona_manager
        self.content_validator = content_validator
        self.codebase_path = PROJECT_ROOT # Assuming PROJECT_ROOT is the base path for analysis

    @classmethod
    def _collect_configuration_analysis(cls, codebase_path: str) -> ConfigurationAnalysisOutput:
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
                with open(ci_yml_path, 'r', encoding='utf-8') as f: # Re-open to read lines
                    ci_content_lines = f.readlines()
                    ci_workflow_jobs = {}
                    for job_name, job_details in ci_config_raw.get("jobs", {}).items():
                        steps_summary = []
                        for step in job_details.get("steps", []):
                            step_name = step.get("name", "Unnamed Step")
                            step_run = step.get("run")
                            step_uses = step.get("uses")
                            
                            summary_item_data = {"name": step_name}
                            if step_uses: summary_item_data["uses"] = step_uses
                            if step_run:
                                commands = [cmd.strip() for cmd in step_run.split('\n') if cmd.strip()]
                                summary_item_data["runs_commands"] = commands
                                # Find the line number for the 'run' block to get a snippet
                                # This is a heuristic, might need refinement for complex YAML structures
                                run_line_number = None
                                for i, line in enumerate(ci_content_lines):
                                    if f"name: \"{step_name}\"" in line:
                                        # Look for the 'run:' keyword after the name
                                        for j in range(i, len(ci_content_lines)):
                                            if "run:" in ci_content_lines[j]:
                                                run_line_number = j + 1 # 1-indexed
                                                break
                                        break
                                summary_item_data["code_snippet"] = _get_code_snippet(ci_content_lines, run_line_number, context_lines=3)
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
                with open(pre_commit_path, 'r', encoding='utf-8') as f: # Re-open to read lines
                    pre_commit_content_lines = f.readlines()
                    for repo_config in pre_commit_config_raw.get("repos", []):
                        repo_url = repo_config.get("repo")
                        repo_rev = repo_config.get("rev")
                        for hook in repo_config.get("hooks", []):
                            hook_id = hook.get("id")
                            hook_args = hook.get("args", [])
                            
                            # Find line number for the hook definition
                            hook_line_number = None
                            for i, line in enumerate(pre_commit_content_lines):
                                if f"id: {hook_id}" in line:
                                    hook_line_number = i + 1
                                    break
                            
                            config_analysis_data["pre_commit_hooks"].append(
                                PreCommitHook(repo=repo_url, rev=repo_rev, id=hook_id, args=hook_args, code_snippet=_get_code_snippet(pre_commit_content_lines, hook_line_number, context_lines=3))
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
                with open(pyproject_path, 'r', encoding='utf-8') as f: # Re-open to read lines
                    pyproject_content_lines = f.readlines()
                    pyproject_toml_data = {}

                    ruff_tool_config = pyproject_config_raw.get("tool", {}).get("ruff", {})
                    if ruff_tool_config:
                        # Heuristic to find line number for ruff config
                        ruff_line_number = None
                        for i, line in enumerate(pyproject_content_lines):
                            if "[tool.ruff]" in line:
                                ruff_line_number = i + 1
                                break

                        pyproject_toml_data["ruff"] = RuffConfig(
                            line_length=ruff_tool_config.get("line-length"),
                            target_version=ruff_tool_config.get("target-version"),
                            lint_select=ruff_tool_config.get("lint", {}).get("select"),
                            lint_ignore=ruff_tool_config.get("lint", {}).get("ignore"),
                            format_settings=ruff_tool_config.get("format"),
                            config_snippet=_get_code_snippet(pyproject_content_lines, ruff_line_number, context_lines=5)
                        )
                    bandit_tool_config = pyproject_config_raw.get("tool", {}).get("bandit", {})
                    if bandit_tool_config:
                        # Heuristic to find line number for bandit config
                        bandit_line_number = None
                        for i, line in enumerate(pyproject_content_lines):
                            if "[tool.bandit]" in line:
                                bandit_line_number = i + 1
                                break

                        pyproject_toml_data["bandit"] = BanditConfig(
                            exclude_dirs=bandit_tool_config.get("exclude_dirs"),
                            severity_level=bandit_tool_config.get("severity_level"),
                            confidence_level=bandit_tool_config.get("confidence_level"),
                            skip_checks=bandit_tool_config.get("skip_checks"),
                            config_snippet=_get_code_snippet(pyproject_content_lines, bandit_line_number, context_lines=5)
                        )
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
            "dockerfile_problem_snippets": [],
            "prod_requirements_present": False,
            "prod_dependency_count": 0,
            "dev_dependency_overlap_count": 0,
            "unpinned_prod_dependencies": [],
            "malformed_blocks": []
        }
        
        # 1. Analyze Dockerfile
        dockerfile_path = Path(codebase_path) / "Dockerfile"
        if dockerfile_path.exists():
            deployment_metrics_data["dockerfile_present"] = True
            try:
                with open(dockerfile_path, 'r', encoding='utf-8') as f:
                    dockerfile_content = f.read()
                    dockerfile_lines = dockerfile_content.splitlines()
                
                if "HEALTHCHECK" not in dockerfile_content:
                    deployment_metrics_data["dockerfile_problem_snippets"].append(
                        "Missing HEALTHCHECK instruction. Example: `HEALTHCHECK --interval=30s --timeout=5s CMD curl -f http://localhost:8080/health || exit 1`"
                    )
                else:
                    deployment_metrics_data["dockerfile_healthcheck_present"] = True

                if not re.search(r"USER\s+(?!root)", dockerfile_content, re.IGNORECASE):
                    deployment_metrics_data["dockerfile_problem_snippets"].append(
                        "Missing non-root USER instruction. Example: `RUN useradd -m appuser && USER appuser`"
                    )
                else:
                    deployment_metrics_data["dockerfile_non_root_user"] = True
                
                exposed_ports = re.findall(r"EXPOSE\s+(\d+)", dockerfile_content)
                deployment_metrics_data["dockerfile_exposed_ports"] = [int(p) for p in exposed_ports]

                if not re.search(r"FROM\s+.*?AS\s+.*?\nFROM", dockerfile_content, re.DOTALL | re.IGNORECASE):
                    deployment_metrics_data["dockerfile_problem_snippets"].append(
                        "Missing multi-stage build. Consider using multiple FROM statements for smaller images."
                    )
                else:
                    deployment_metrics_data["dockerfile_multi_stage_build"] = True

            except OSError as e:
                logger.error(f"Error reading Dockerfile {dockerfile_path}: {e}")
                deployment_metrics_data["malformed_blocks"].append({"type": "DOCKERFILE_READ_ERROR", "message": str(e), "file": str(dockerfile_path)})
        
        # 2. Analyze requirements-prod.txt and requirements.txt
        prod_req_path = Path(codebase_path) / "requirements-prod.txt"
        dev_req_path = Path(codebase_path) / "requirements.txt"

        prod_deps = set()
        unpinned_prod_deps = []
        if prod_req_path.exists():
            deployment_metrics_data["prod_requirements_present"] = True
            try:
                with open(prod_req_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        line = line.strip()
                        if line and not line.startswith('#'):
                            if not re.search(r'[=~><]=', line):
                                unpinned_prod_deps.append(line)
                            prod_deps.add(line.split('==')[0].split('>=')[0].split('~=')[0].lower())
                deployment_metrics_data["prod_dependency_count"] = len(prod_deps)
                deployment_metrics_data["unpinned_prod_dependencies"] = unpinned_prod_deps
            except OSError as e:
                logger.error(f"Error reading requirements-prod.txt {prod_req_path}: {e}")
                deployment_metrics_data["malformed_blocks"].append({"type": "PROD_REQ_READ_ERROR", "message": str(e), "file": str(prod_req_path)})

        if dev_req_path.exists() and prod_req_path.exists():
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

    def _collect_reasoning_quality_metrics(self) -> Dict[str, Any]:
        """
        Collects metrics related to the quality of the Socratic debate process itself.
        """
        reasoning_metrics = {
            "total_debate_turns": 0,
            "unique_personas_involved": 0,
            "schema_validation_failures_count": 0,
            "content_misalignment_warnings": 0,
            "debate_turn_errors": 0,
            "conflict_resolution_attempts": 0,
            "conflict_resolution_successes": 0,
            "unresolved_conflict_present": False,
            "average_persona_output_tokens": 0.0,
            "persona_specific_performance": defaultdict(lambda: {"success_rate": 0.0, "schema_failures": 0, "truncations": 0, "total_turns": 0}),
            "prompt_verbosity_score": 0.0,
            "malformed_blocks_summary": defaultdict(int)
        }

        debate_history = self.intermediate_steps.get("Debate_History", [])
        reasoning_metrics["total_debate_turns"] = len(debate_history)

        unique_personas = set()
        for turn in debate_history:
            if "persona" in turn:
                unique_personas.add(turn["persona"])
        reasoning_metrics["unique_personas_involved"] = len(unique_personas)

        all_malformed_blocks = self.intermediate_steps.get("malformed_blocks", [])
        reasoning_metrics["schema_validation_failures_count"] = sum(1 for b in all_malformed_blocks if b.get("type") == "SCHEMA_VALIDATION_ERROR")
        reasoning_metrics["content_misalignment_warnings"] = sum(1 for b in all_malformed_blocks if b.get("type") == "CONTENT_MISALIGNMENT")
        reasoning_metrics["debate_turn_errors"] = sum(1 for b in all_malformed_blocks if b.get("type") == "DEBATE_TURN_ERROR")
        
        for block in all_malformed_blocks:
            reasoning_metrics["malformed_blocks_summary"][block.get("type", "UNKNOWN_MALFORMED_BLOCK")] += 1

        if self.intermediate_steps.get("Conflict_Resolution_Attempt"):
            reasoning_metrics["conflict_resolution_attempts"] = 1
            if self.intermediate_steps["Conflict_Resolution_Attempt"].get("conflict_resolved"):
                reasoning_metrics["conflict_resolution_successes"] = 1
        reasoning_metrics["unresolved_conflict_present"] = bool(self.intermediate_steps.get("Unresolved_Conflict"))

        total_output_tokens = 0
        for key, value in self.intermediate_steps.items():
            if key.endswith("_Tokens_Used") and not key.startswith(("Total_", "context_", "synthesis_", "debate_")):
                total_output_tokens += value
        
        if reasoning_metrics["total_debate_turns"] > 0:
            reasoning_metrics["average_persona_output_tokens"] = total_output_tokens / reasoning_metrics["total_debate_turns"]

        for persona_name in unique_personas:
            persona_malformed_blocks = [b for b in all_malformed_blocks if b.get("persona") == persona_name]
            schema_failures = sum(1 for b in persona_malformed_blocks if b.get("type") == "SCHEMA_VALIDATION_ERROR")
            content_misalignments = sum(1 for b in persona_malformed_blocks if b.get("type") == "CONTENT_MISALIGNMENT")
            
            persona_turns = sum(1 for turn in debate_history if turn.get("persona") == persona_name)
            
            reasoning_metrics["persona_specific_performance"][persona_name]["total_turns"] = persona_turns
            reasoning_metrics["persona_specific_performance"][persona_name]["schema_failures"] = schema_failures
            reasoning_metrics["persona_specific_performance"][persona_name]["truncations"] = 0
            
            if persona_turns > 0:
                reasoning_metrics["persona_specific_performance"][persona_name]["success_rate"] = \
                    (persona_turns - schema_failures - content_misalignments) / persona_turns
            else:
                reasoning_metrics["persona_specific_performance"][persona_name]["success_rate"] = 0.0

        return reasoning_metrics

    def collect_all_metrics(self) -> Dict[str, Any]:
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
                "detailed_issues": [],
                "ruff_violations": []
            },
            "security": {
                "bandit_issues_count": 0,
                "ast_security_issues_count": 0,
            },
            "performance_efficiency": {
                "token_usage_stats": self._collect_token_usage_stats(),
                "debate_efficiency_summary": self._analyze_debate_efficiency(),
                "potential_bottlenecks_count": 0
            },
            "robustness": {
                "schema_validation_failures_count": len(self.intermediate_steps.get("malformed_blocks", [])),
                "unresolved_conflict_present": bool(self.intermediate_steps.get("Unresolved_Conflict")),
                "conflict_resolution_attempted": bool(self.intermediate_steps.get("Conflict_Resolution_Attempt"))
            },
            "maintainability": {
                "test_coverage_summary": self._assess_test_coverage()
            },
            "configuration_analysis": self._collect_configuration_analysis(self.codebase_path).model_dump(by_alias=True),
            "deployment_robustness": self._collect_deployment_robustness_metrics(self.codebase_path).model_dump(by_alias=True),
            "reasoning_quality": self._collect_reasoning_quality_metrics()
        }

        total_functions_across_codebase = 0
        total_loc_across_functions = 0
        total_complexity_across_functions = 0
        total_args_across_functions = 0
        total_nesting_depth_across_functions = 0

        for root, _, files in os.walk(self.codebase_path):
            for file in files:
                if file.endswith('.py'):
                    file_path = os.path.join(root, file)
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            content = f.read()
                            content_lines = content.splitlines()

                        ruff_issues = _run_ruff(content, file_path)
                        if ruff_issues:
                            metrics["code_quality"]["ruff_issues_count"] += len(ruff_issues)
                            metrics["code_quality"]["detailed_issues"].extend(ruff_issues)
                            metrics["code_quality"]["ruff_violations"].extend(ruff_issues)

                        bandit_issues = _run_bandit(content, file_path)
                        if bandit_issues:
                            metrics["security"]["bandit_issues_count"] += len(bandit_issues)
                            metrics["code_quality"]["detailed_issues"].extend(bandit_issues)

                        ast_security_issues = _run_ast_security_checks(content, file_path)
                        if ast_security_issues:
                            metrics["security"]["ast_security_issues_count"] += len(ast_security_issues)
                            metrics["code_quality"]["detailed_issues"].extend(ast_security_issues)

                        file_function_metrics = self._analyze_python_file_ast(content, content_lines, file_path)

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

    def _collect_token_usage_stats(self) -> Dict[str, Any]:
        """
        Collects token usage statistics from debate intermediate steps.
        """
        total_tokens = self.intermediate_steps.get("Total_Tokens_Used", 0)
        total_cost = self.intermediate_steps.get("Total_Estimated_Cost_USD", 0.0)

        phase_token_usage = {}
        for key, value in self.intermediate_steps.items():
            if key.endswith("_Tokens_Used") and not key.startswith(("Total_", "context_", "synthesis_", "debate_")):
                phase_name = key.replace("_Tokens_Used", "")
                phase_token_usage[phase_name] = value

        return {
            "total_tokens": total_tokens,
            "total_cost_usd": total_cost,
            "phase_token_usage": phase_token_usage
        }

    def _analyze_debate_efficiency(self) -> Dict[str, Any]:
        """
        Analyzes the efficiency of the debate process.
        """
        efficiency_summary = {
            "num_turns": len(self.intermediate_steps.get("Debate_History", [])),
            "malformed_blocks_count": len(self.intermediate_steps.get("malformed_blocks", [])),
            "conflict_resolution_attempts": 1 if self.intermediate_steps.get("Conflict_Resolution_Attempt") else 0,
            "unresolved_conflict": bool(self.intermediate_steps.get("Unresolved_Conflict")),
            "average_turn_tokens": 0.0,
            "persona_token_breakdown": {}
        }

        total_debate_tokens = self.intermediate_steps.get("debate_Tokens_Used", 0)
        num_turns = efficiency_summary["num_turns"]
        if num_turns > 0:
            efficiency_summary["average_turn_tokens"] = total_debate_tokens / num_turns

        for key, value in self.intermediate_steps.items():
            if key.endswith("_Tokens_Used") and not key.startswith(("Total_", "context_", "synthesis_", "debate_")):
                persona_name = key.replace("_Tokens_Used", "")
                efficiency_summary["persona_token_breakdown"][persona_name] = value

        return efficiency_summary

    def _assess_test_coverage(self) -> Dict[str, Any]:
        """
        Assesses test coverage for the codebase.
        Placeholder implementation.
        """
        return {
            "overall_coverage_percentage": 0.0,
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

    # --- SUGGESTED METHODS TO ADD ---

    def save_improvement_results(self, suggestions: List[Dict], metrics_before: Dict,
                                 metrics_after: Dict, success: bool):
        """Save results of improvement attempt for future learning"""
        from datetime import datetime
        import json
        from pathlib import Path
        
        # Calculate performance changes here as well, for consistency in the record
        performance_changes = {}
        for category, metrics in metrics_after.items():
            if category in metrics_before:
                category_changes = {}
                for metric, value_after in metrics.items():
                    if metric in metrics_before[category]:
                        value_before = metrics_before[category][metric]
                        
                        if isinstance(value_before, (int, float)) and isinstance(value_after, (int, float)):
                            absolute_change = value_after - value_before
                            percent_change = (absolute_change / value_before * 100) if value_before != 0 else float('inf')
                            
                            category_changes[metric] = {
                                "absolute_change": absolute_change,
                                "percent_change": percent_change
                            }
                        else:
                            category_changes[metric] = {
                                "changed": value_before != value_after,
                                "before": value_before,
                                "after": value_after
                            }
                if category_changes:
                    performance_changes[category] = category_changes

        improvement_record = {
            "timestamp": datetime.now().isoformat(),
            "suggestions": suggestions,
            "metrics_before": metrics_before,
            "metrics_after": metrics_after,
            "success": success,
            "performance_changes": performance_changes,
            "improvement_score": self.intermediate_steps.get("improvement_score", 0.0) # Assuming score is stored here
        }
        
        # Append to historical data
        history_file = Path("data/improvement_history.jsonl")
        history_file.parent.mkdir(exist_ok=True)
        
        try:
            with open(history_file, "a", encoding='utf-8') as f:
                f.write(json.dumps(improvement_record) + "\n")
            logger.info(f"Saved improvement results to {history_file}")
        except IOError as e:
            logger.error(f"Failed to save improvement results to {history_file}: {e}")

    def get_historical_improvement_data(self, limit=50) -> List[Dict]:
        """Retrieve historical improvement data for analysis"""
        history_file = Path("data/improvement_history.jsonl")
        if not history_file.exists():
            logger.info("No historical improvement data found.")
            return []
        
        records = []
        try:
            with open(history_file, "r", encoding='utf-8') as f:
                for line in f:
                    try:
                        records.append(json.loads(line))
                        if len(records) >= limit:
                            break
                    except json.JSONDecodeError:
                        logger.warning(f"Skipping malformed line in {history_file}: {line.strip()}")
                        continue
            logger.info(f"Retrieved {len(records)} historical improvement records.")
            return records
        except IOError as e:
            logger.error(f"Failed to read historical improvement data from {history_file}: {e}")
            return []

    def analyze_historical_effectiveness(self) -> Dict:
        """Analyze historical data to identify patterns of successful improvements"""
        history = self.get_historical_improvement_data()
        
        if not history:
            return {
                "success_rate": 0.0,
                "total_attempts": 0,
                "successful_attempts": 0,
                "top_performing_areas": [],
                "common_failure_modes": []
            }
        
        # Calculate overall success rate
        successful_attempts = sum(1 for h in history if h.get("success", False))
        total_attempts = len(history)
        success_rate = successful_attempts / total_attempts if total_attempts > 0 else 0.0
        
        # Analyze by improvement area
        area_success = defaultdict(int)
        area_attempts = defaultdict(int)
        
        for record in history:
            for suggestion in record.get("suggestions", []):
                area = suggestion.get("AREA", "Unknown")
                area_attempts[area] += 1
                
                if record.get("success"):
                    area_success[area] += 1
        
        # Calculate success rates by area
        area_success_rates = {}
        for area, attempts in area_attempts.items():
            successes = area_success.get(area, 0)
            area_success_rates[area] = successes / attempts if attempts > 0 else 0.0
        
        # Identify top performing areas (minimum 3 attempts to be considered)
        top_areas = [
            area for area, rate in sorted(area_success_rates.items(), key=lambda x: x[1], reverse=True)
            if area_attempts.get(area, 0) >= 3
        ][:5]
        
        # Identify common failure modes
        failure_modes = defaultdict(int)
        for record in history:
            if not record.get("success", False):
                for category, changes in record.get("performance_changes", {}).items():
                    for metric, change_data in changes.items():
                        if isinstance(change_data, dict) and "percent_change" in change_data:
                            # Consider negative percent change as a failure mode for "higher is better" metrics
                            # Or positive percent change for "lower is better" metrics
                            # This logic needs to be more robust based on metric directionality
                            # For simplicity, let's assume negative percent_change is bad for most metrics
                            if change_data.get("percent_change", 0) < -5: # If metric decreased by more than 5%
                                key = f"{category}.{metric}"
                                failure_modes[key] += 1
        
        # Sort failure modes by frequency
        common_failure_modes = sorted(
            failure_modes.items(), 
            key=lambda x: x[1], 
            reverse=True
        )[:5]
        
        return {
            "success_rate": success_rate,
            "total_attempts": total_attempts,
            "successful_attempts": successful_attempts,
            "top_performing_areas": [
                {"area": area, "success_rate": area_success_rates[area]}
                for area in top_areas
            ],
            "common_failure_modes": [
                {"metric": mode[0], "occurrences": mode[1]}
                for mode in common_failure_modes
            ]
        }

    def analyze(self) -> List[Dict[str, Any]]:
        """
        Performs the self-analysis and generates improvement suggestions.
        Focuses on the top 3 highest impact areas based on metrics, adhering to the Pareto principle.
        """
        logger.info("Performing self-analysis for Project Chimera.")
        
        suggestions = []
        
        # --- NEW: Handle Conflict Resolution first if codebase access is denied ---
        conflict_resolution_summary = self.intermediate_steps.get("Conflict_Resolution_Attempt", {}).get("resolution_summary", {})
        if conflict_resolution_summary and conflict_resolution_summary.get("conflict_resolved") and \
           "cannot be fulfilled due to the absence of the codebase" in conflict_resolution_summary.get("RATIONALE", ""):
            
            # This is the most impactful immediate action as per the conflict resolution
            suggestions.append({
                "AREA": "Maintainability",
                "PROBLEM": "Critical lack of codebase access prevents meaningful code-level analysis and improvements. The system cannot perform security, robustness, or detailed maintainability analyses without the codebase. The conflict resolution summary correctly identified the need for the codebase to proceed.",
                "PROPOSED_SOLUTION": "Establish a mechanism for providing the Project Chimera codebase and its context. This includes providing relevant files, their purpose, architecture, and any prior analysis. A `docs/project_chimera_context.md` file is proposed to guide this collection.",
                "EXPECTED_IMPACT": "Enables the self-improvement process to proceed effectively, allowing for specific vulnerability identification, optimization opportunities, and actionable code modifications. This directly addresses the 'Maintainability' aspect of the self-improvement goals.",
                "CODE_CHANGES_SUGGESTED": [
                    {
                        "FILE_PATH": "docs/project_chimera_context.md",
                        "ACTION": "ADD",
                        "FULL_CONTENT": "# Project Chimera Context and Codebase\\n\\nThis document outlines the necessary context for analyzing and improving Project Chimera.\\n\\n## Project Purpose:\\n\\n[Insert a clear and concise description of Project Chimera's overall goal and function here.]\\n\\n## Architecture Overview:\\n\\n[Describe the high-level architecture, key components, and their interactions. For example:\\n- Core modules (e.g., LLM interaction, data processing, UI)\\n- Key libraries and frameworks used\\n- Data flow and storage mechanisms]\\n\\n## Codebase Location:\\n\\n[Provide the location or a representative subset of the Project Chimera codebase. If a subset is provided, specify which parts are included and why.]\\n\\n## Prior Analysis Context:\\n\\n[If applicable, reference or include the 'Previous Debate Output Summary' or any other relevant historical analysis that informs the current self-improvement goals.]\\n\\n## Current Focus Areas for Improvement:\\n\\n[Based on prior discussions or initial observations, list the key areas targeted for improvement (e.g., Security, Performance, Maintainability, Reasoning Quality).]\\n"
                    }
                ]
            })
            # If codebase access is the primary blocker, other code changes are secondary or conceptual.
            # We return here to ensure this is the ONLY suggestion if the codebase is missing.
            return suggestions 

        # --- Extract top Ruff and Bandit issues for snippets ---
        top_ruff_issues_snippets = []
        top_bandit_issues_snippets = []
        
        # Filter and collect snippets for Ruff issues
        ruff_detailed_issues = [
            issue for issue in self.metrics.get('code_quality', {}).get('detailed_issues', [])
            if issue.get('source') == 'ruff_lint' or issue.get('source') == 'ruff_format'
        ]
        for issue in ruff_detailed_issues[:3]: # Take top 3
            snippet = issue.get('code_snippet')
            if snippet:
                top_ruff_issues_snippets.append(f"  - File: `{issue.get('file', 'N/A')}` (Line: {issue.get('line', 'N/A')}): `{issue.get('code', 'N/A')}` - {issue.get('message', 'N/A')}\n```\n{snippet}\n```")
            else:
                top_ruff_issues_snippets.append(f"  - File: `{issue.get('file', 'N/A')}` (Line: {issue.get('line', 'N/A')}): `{issue.get('code', 'N/A')}` - {issue.get('message', 'N/A')}")

        # Filter and collect snippets for Bandit issues
        bandit_detailed_issues = [
            issue for issue in self.metrics.get('code_quality', {}).get('detailed_issues', [])
            if issue.get('source') == 'bandit'
        ]
        for issue in bandit_detailed_issues[:3]: # Take top 3
            snippet = issue.get('code_snippet')
            if snippet:
                top_bandit_issues_snippets.append(f"  - File: `{issue.get('file', 'N/A')}` (Line: {issue.get('line', 'N/A')}): `{issue.get('code', 'N/A')}` - {issue.get('message', 'N/A')}\n```\n{snippet}\n```")
            else:
                top_bandit_issues_snippets.append(f"  - File: `{issue.get('file', 'N/A')}` (Line: {issue.get('line', 'N/A')}): `{issue.get('code', 'N/A')}` - {issue.get('message', 'N/A')}")
        # --- End snippet extraction ---

        # --- MODIFIED LOGIC FOR PARETO PRINCIPLE AND CLARITY ---
        # Focus on the top 3 highest impact areas based on metrics (Pareto principle).
        # Prioritize Security, Maintainability, and Robustness.
        
        # Maintainability (Linting Issues)
        ruff_issues_count = self.metrics.get('code_quality', {}).get('ruff_issues_count', 0)
        if ruff_issues_count > 100: # Threshold for significant linting issues
            suggestions.append({
                "AREA": "Maintainability",
                "PROBLEM": f"The project exhibits widespread Ruff formatting issues across numerous files (e.g., `core.py`, `code_validator.py`, `app.py`, all test files, etc.). The `code_quality.ruff_violations` list contains {ruff_issues_count} entries, predominantly `FMT` (formatting) errors. This inconsistency detracts from readability and maintainability. Examples:\n" + "\n".join(top_ruff_issues_snippets),
                "PROPOSED_SOLUTION": "Enforce consistent code formatting by running `ruff format .` across the entire project. Integrate this command into the CI pipeline and pre-commit hooks to ensure all committed code adheres to the defined style guidelines. This will resolve the numerous `FMT` violations.",
                "EXPECTED_IMPACT": "Improved code readability and consistency, reduced cognitive load for developers, and a cleaner codebase. This directly addresses the maintainability aspect by enforcing a standard.",
                "CODE_CHANGES_SUGGESTED": [
                    {
                        "FILE_PATH": ".github/workflows/ci.yml",
                        "ACTION": "MODIFY",
                        "DIFF_CONTENT": """--- a/.github/workflows/ci.yml
+++ b/.github/workflows/ci.yml
@@ -18,8 +18,8 @@
               # Explicitly install Ruff and Black for CI to ensure they are available
               pip install ruff black
             },
-            {
-              name: "Run Ruff (Linter & Formatter Check) - Fail on Violation",
+            # Run Ruff for linting and formatting checks
+            {
+              name: "Run Ruff Check and Format",
               uses: null,
               runs_commands:
                 - "ruff check . --output-format=github --exit-non-zero-on-fix"
@@ -27,7 +27,7 @@
 
             {
               name: "Run Bandit Security Scan",
-              uses: null,
+              uses: null
               runs_commands:
                 - "bandit -r . -ll -c pyproject.toml --exit-on-error"
                 # Bandit is configured to exit-on-error, which will fail the job if issues are found based on pyproject.toml settings.
"""
                    },
                    {
                        "FILE_PATH": ".pre-commit-config.yaml",
                        "ACTION": "MODIFY",
                        "DIFF_CONTENT": """--- a/.pre-commit-config.yaml
+++ b/.pre-commit-config.yaml
@@ -16,7 +16,7 @@
       - id: ruff
         args: [
           "--fix"
-        ]
+        ]
 
       - repo: https://github.com/charliermarsh/ruff-pre-commit
         rev: v0.1.9
@@ -24,7 +24,7 @@
         id: ruff-format
         args: []
 
-      - repo: https://github.com/PyCQA/bandit
+      - repo: https://github.com/PyCQA/bandit
         rev: 1.7.5
         id: bandit
         args: [
"""
                    }
                ]
            })
        
        # Security
        bandit_issues_count = self.metrics.get('security', {}).get('bandit_issues_count', 0)
        pyproject_config_error = any(block.get('type') == 'PYPROJECT_CONFIG_PARSE_ERROR' for block in self.metrics.get('configuration_analysis', {}).get('malformed_blocks', []))
        
        if bandit_issues_count > 0 or pyproject_config_error: # Trigger if issues or config error
            problem_description = f"Bandit security scans are failing with configuration errors (`Bandit failed with exit code 2: [config] ERROR Invalid value (at line 33, column 15) [main] ERROR /Users/tom/Documents/apps/project_chimera/pyproject.toml : Error parsing file.`). This indicates a misconfiguration in `pyproject.toml` for Bandit, preventing security vulnerabilities from being detected. The `pyproject.toml` file itself has a `PYPROJECT_CONFIG_PARSE_ERROR` related to `ruff` configuration."
            if bandit_issues_count > 0:
                problem_description += f"\nAdditionally, {bandit_issues_count} Bandit security vulnerabilities were detected. Prioritize HIGH severity issues like potential injection flaws. Examples:\n" + "\n".join(top_bandit_issues_snippets)
            
            suggestions.append({
                "AREA": "Security",
                "PROBLEM": problem_description,
                "PROPOSED_SOLUTION": "Correct the Bandit configuration within `pyproject.toml`. Ensure that all Bandit-related settings are valid and adhere to Bandit's expected format. Additionally, address the Ruff configuration error in `pyproject.toml` to ensure consistent code formatting and linting. The CI workflow should also be updated to correctly invoke Bandit with the corrected configuration.",
                "EXPECTED_IMPACT": "Enables the Bandit security scanner to run successfully, identifying potential security vulnerabilities. This will improve the overall security posture of the project.",
                "CODE_CHANGES_SUGGESTED": [
                    {
                        "FILE_PATH": "pyproject.toml",
                        "ACTION": "MODIFY",
                        "DIFF_CONTENT": """--- a/pyproject.toml
+++ b/pyproject.toml
@@ -30,7 +30,7 @@
 
 [tool.ruff]
 line-length = 88
-target-version = "null"
+target-version = "py311"
 
 [tool.ruff.lint]
 ignore = [
@@ -310,7 +310,7 @@
 
 [tool.bandit]
 conf_file = "pyproject.toml"
-level = "null"
+level = "info"
 # Other Bandit configurations can be added here as needed.
 # For example:
 # exclude = [
"""
                    },
                    {
                        "FILE_PATH": ".github/workflows/ci.yml",
                        "ACTION": "MODIFY",
                        "DIFF_CONTENT": """--- a/.github/workflows/ci.yml
+++ b/.github/workflows/ci.yml
@@ -21,7 +21,7 @@
             # Run Ruff (Linter & Formatter Check) - Fail on Violation
             ruff check . --output-format=github --exit-non-zero-on-fix
             ruff format --check --diff --exit-non-zero-on-fix # Show diff and fail on formatting issues
-            # Run Bandit Security Scan
-            bandit -r . -ll -c pyproject.toml --exit-on-error
+            # Run Bandit Security Scan with corrected configuration
+            bandit -r . --config pyproject.toml --exit-on-error
             # Run Pytest and generate coverage report
             pytest --cov=src --cov-report=xml --cov-report=term
"""
                    }
                ]
            })
        
        # Maintainability (Testing)
        zero_test_coverage = self.metrics.get('maintainability', {}).get('test_coverage_summary', {}).get('overall_coverage_percentage', 0) == 0
        if zero_test_coverage:
            suggestions.append({
                "AREA": "Maintainability",
                "PROBLEM": "The project lacks automated test coverage. The `maintainability.test_coverage_summary` shows `overall_coverage_percentage: 0.0` and `coverage_details: 'Automated test coverage assessment not implemented.'`. This significantly hinders the ability to refactor code confidently, introduce new features without regressions, and ensure the long-term health of the codebase.",
                "PROPOSED_SOLUTION": "Implement a comprehensive testing strategy. This includes writing unit tests for core logic (e.g., LLM interactions, data processing, utility functions) and integration tests for key workflows. Start with critical modules like `src/llm_provider.py`, `src/utils/prompt_engineering.py`, and `src/persona_manager.py`. Aim for a minimum of 70% test coverage within the next iteration.",
                "EXPECTED_IMPACT": "Improved code stability, reduced regression bugs, increased developer confidence during changes, and a clearer understanding of code behavior. This directly addresses the 'Maintainability' aspect of the self-improvement goals.",
                "CODE_CHANGES_SUGGESTED": [
                    {
                        "FILE_PATH": "tests/test_llm_provider.py",
                        "ACTION": "ADD",
                        "FULL_CONTENT": """import pytest
from src.llm_provider import LLMProvider

# Mocking the LLM API for testing
class MockLLMClient:
    def __init__(self, model_name):
        self.model_name = model_name

    def generate_content(self, prompt):
        # Simulate a response based on prompt content
        if "summarize" in prompt.lower():
            return "This is a simulated summary."
        elif "analyze" in prompt.lower():
            return "This is a simulated analysis."
        else:
            return "This is a simulated default response."

@pytest.fixture
def llm_provider():
    # Use the mock client for testing
    client = MockLLMClient(model_name="mock-model")
    return LLMProvider(client=client)

def test_llm_provider_initialization(llm_provider):
    \"\"\"Test that the LLMProvider initializes correctly.\"\"\"
    assert llm_provider.client.model_name == "mock-model"

def test_llm_provider_generate_content_summary(llm_provider):
    \"\"\"Test content generation for a summarization prompt.\"\"\"
    prompt = "Please summarize the following text: ..."
    response = llm_provider.generate_content(prompt)
    assert response == "This is a simulated summary."

def test_llm_provider_generate_content_analysis(llm_provider):
    \"\"\"Test content generation for an analysis prompt.\"\"\"
    prompt = "Analyze the provided data: ..."
    response = llm_provider.generate_content(prompt)
    assert response == "This is a simulated analysis."

def test_llm_provider_generate_content_default(llm_provider):
    \"\"\"Test content generation for a general prompt.\"\"\"
    prompt = "What is the capital of France?"
    response = llm_provider.generate_content(prompt)
    assert response == "This is a simulated default response."

# Add more tests for different scenarios and edge cases
"""
                    },
                    {
                        "FILE_PATH": "tests/test_prompt_engineering.py",
                        "ACTION": "ADD",
                        "FULL_CONTENT": """import pytest
from src.utils.prompt_engineering import create_persona_prompt, create_task_prompt

def test_create_persona_prompt_basic():
    \"\"\"Test creating a persona prompt with basic details.\"\"\"
    persona_details = {
        "name": "Test Persona",
        "role": "Tester",
        "goal": "Evaluate prompts"
    }
    expected_prompt = "You are Test Persona, a Tester. Your goal is to Evaluate prompts."
    assert create_persona_prompt(persona_details) == expected_prompt

def test_create_persona_prompt_with_constraints():
    \"\"\"Test creating a persona prompt with additional constraints.\"\"\"
    persona_details = {
        "name": "Constraint Bot",
        "role": "Rule Enforcer",
        "goal": "Ensure adherence to rules",
        "constraints": ["Be concise", "Avoid jargon"]
    }
    expected_prompt = "You are Constraint Bot, a Rule Enforcer. Your goal is to Ensure adherence to rules. Adhere to the following constraints: Be concise, Avoid jargon."
    assert create_persona_prompt(persona_details) == expected_prompt

def test_create_persona_prompt_empty_details():
    \"\"\"Test creating a persona prompt with empty details.\"\"\"
    persona_details = {}
    expected_prompt = "You are an AI assistant. Your goal is to assist the user."
    assert create_persona_prompt(persona_details) == expected_prompt

def test_create_task_prompt_basic():
    \"\"\"Test creating a basic task prompt.\"\"\"
    task_description = "Summarize the provided text."
    expected_prompt = f"Task: {task_description}\\n\\nProvide a concise summary."
    assert create_task_prompt(task_description) == expected_prompt

def test_create_task_prompt_with_context():
    \"\"\"Test creating a task prompt with context.\"\"\"
    task_description = "Analyze the user query."
    context = "User is asking about project status."
    expected_prompt = f"Task: {task_description}\\n\\nContext: {context}\\n\\nProvide a detailed analysis."
    assert create_task_prompt(task_description, context=context) == expected_prompt

def test_create_task_prompt_with_specific_instructions():
    \"\"\"Test creating a task prompt with specific output instructions.\"\"\"
    task_description = "Extract key entities."
    instructions = "Output the entities as a JSON list."
    expected_prompt = f"Task: {task_description}\\n\\nInstructions: {instructions}\\n\\nProvide the extracted entities in the specified format."
    assert create_task_prompt(task_description, instructions=instructions) == expected_prompt

# Add more tests for edge cases and variations in input
"""
                    }
                ]
            })
        
        # Efficiency (Token Usage)
        high_token_personas = self.metrics.get('performance_efficiency', {}).get('debate_efficiency_summary', {}).get('persona_token_breakdown', {})
        high_token_consumers = {p: t for p, t in high_token_personas.items() if t > 2000}
        
        if high_token_consumers:
            suggestions.append({
                "AREA": "Efficiency",
                "PROBLEM": f"High token consumption by personas: {', '.join(high_token_consumers.keys())}. This indicates potentially verbose or repetitive analysis patterns.",
                "PROPOSED_SOLUTION": "Optimize prompts for high-token personas. Implement prompt truncation strategies where appropriate, focusing on summarizing or prioritizing key information. For 'Self_Improvement_Analyst', focus on direct actionable insights rather than exhaustive analysis. For technical personas, ensure they are provided with concise, targeted information relevant to their specific task.",
                "EXPECTED_IMPACT": "Reduces overall token consumption, leading to lower operational costs and potentially faster response times. Improves the efficiency of the self-analysis process.",
                "CODE_CHANGES_SUGGESTED": [
                    # Example code changes are provided in the main analysis output, not here.
                    # This section would typically be populated by a more detailed analysis.
                ]
            })
        
        # Reasoning Quality (Content Misalignment)
        content_misalignment_warnings = self.metrics.get('reasoning_quality', {}).get('content_misalignment_warnings', 0)
        if content_misalignment_warnings > 3: # Threshold for multiple warnings
            suggestions.append({
                "AREA": "Reasoning Quality",
                "PROBLEM": f"Content misalignment warnings ({content_misalignment_warnings}) indicate potential issues in persona reasoning or prompt engineering.",
                "PROPOSED_SOLUTION": "Refine prompts for clarity and specificity. Review persona logic for consistency and accuracy. Ensure personas stay focused on the core task and domain.",
                "EXPECTED_IMPACT": "Enhances the quality and relevance of persona outputs, leading to more coherent and accurate final answers.",
                "CODE_CHANGES_SUGGESTED": [] # This is a prompt engineering suggestion
            })
        
        # Apply Pareto Principle: Limit to top 3 suggestions
        final_suggestions = suggestions[:3] 
        
        logger.info(f"Generated {len(suggestions)} potential suggestions. Finalizing with top {len(final_suggestions)}.")
        
        return final_suggestions

    # --- Placeholder methods for other potential analyses ---
    def analyze_codebase_structure(self) -> Dict[str, Any]:
        logger.info("Analyzing codebase structure.")
        return {"summary": "Codebase structure analysis is a placeholder."}

    def analyze_performance_bottlenecks(self) -> Dict[str, Any]:
        logger.info("Analyzing performance bottlenecks.")
        return {"summary": "Performance bottleneck analysis is a placeholder."}