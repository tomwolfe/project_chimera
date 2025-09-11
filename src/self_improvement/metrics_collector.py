# src/self_improvement/metrics_collector.py
import os
import json
import ast
import logging
from typing import Dict, Any, List, Tuple, Union, Optional
from collections import defaultdict
from pathlib import Path
import re
import yaml
import toml
from pydantic import ValidationError
from datetime import datetime
import sys
import difflib # NEW: Import difflib for regenerating diffs
import tempfile # NEW: Import tempfile for _validate_and_fix_code_suggestion

from src.utils.code_utils import _get_code_snippet, ComplexityVisitor
from src.utils.code_validator import (
    _run_ruff, _run_bandit, _run_ast_security_checks,
)
from src.models import (
    ConfigurationAnalysisOutput,
    CiWorkflowConfig,
    CiWorkflowJob,
    CiWorkflowStep,
    PreCommitHook,
    PyprojectTomlConfig,
    RuffConfig,
    BanditConfig,
    PydanticSettingsConfig,
    DeploymentAnalysisOutput,
)
from src.utils.command_executor import execute_command_safely
from src.utils.path_utils import PROJECT_ROOT

logger = logging.getLogger(__name__)


class FocusedMetricsCollector:
    """Collects objective metrics for self-improvement analysis."""

    CRITICAL_METRICS = {
        "token_efficiency": {
            "description": "Tokens per meaningful suggestion",
            "threshold": 2000,
            "priority": 1,
        },
        "impact_potential": {
            "description": "Estimated impact of suggested changes (0-100)",
            "threshold": 40,
            "priority": 2,
        },
        "fix_confidence": {
            "description": "Confidence in fix correctness (0-100)",
            "threshold": 70,
            "priority": 3,
        },
    }

    def __init__(
        self,
        initial_prompt: str,
        debate_history: List[Dict],
        intermediate_steps: Dict[str, Any],
        codebase_raw_file_contents: Dict[str, str], # MODIFIED: Accept raw file contents
        tokenizer: Any,
        llm_provider: Any,
        persona_manager: Any,
        content_validator: Any,
    ):
        """Initializes the analyst with collected metrics and context."""
        self.initial_prompt = initial_prompt
        self.debate_history = debate_history
        self.intermediate_steps = intermediate_steps
        self.raw_file_contents = codebase_raw_file_contents # MODIFIED: Store raw file contents
        self.tokenizer = tokenizer
        self.llm_provider = llm_provider
        self.persona_manager = persona_manager
        self.content_validator = content_validator
        self.codebase_path = (
            PROJECT_ROOT
        )  # Assuming the analyst operates from the project root
        self.collected_metrics: Dict[str, Any] = {} # Stores the final collected metrics
        self.reasoning_quality_metrics: Dict[str, Any] = {} # Initialized here, populated by analyze_reasoning_quality
        self.file_analysis_cache: Dict[str, Dict[str, Any]] = {} # Cache for file analysis results
        
        # NEW: Raw counts for tracking current run's outcome (to be saved historically)
        self._current_run_total_suggestions_processed: int = 0
        self._current_run_successful_suggestions: int = 0
        self._current_run_schema_validation_failures: Dict[str, int] = defaultdict(int)

        # These will hold aggregated historical data, populated by analyze_historical_effectiveness
        self._historical_total_suggestions_processed: int = 0
        self._historical_successful_suggestions: int = 0
        self._historical_schema_validation_failures: Dict[str, int] = defaultdict(int)

        self.critical_metric: Optional[str] = None # Initialized here, populated by _identify_critical_metric

        # Load historical data at initialization
        historical_summary = self.analyze_historical_effectiveness()
        self._historical_total_suggestions_processed = historical_summary.get("historical_total_suggestions_processed", 0)
        self._historical_successful_suggestions = historical_summary.get("historical_successful_suggestions", 0)
        self._historical_schema_validation_failures = defaultdict(int, historical_summary.get("historical_schema_validation_failures", {}))


    def _identify_critical_metric(self):
        """Identify the single most critical metric that's furthest from threshold."""
        critical_metric = None
        max_deviation = -1

        for metric_name, config in self.CRITICAL_METRICS.items():
            value = self.collected_metrics.get(metric_name, 0) # Use self.collected_metrics
            threshold = config["threshold"]

            if metric_name == "token_efficiency":
                deviation = value - threshold
            else:
                deviation = threshold - value

            if deviation > max_deviation:
                max_deviation = deviation
                critical_metric = metric_name

        self.critical_metric = critical_metric

    def analyze_reasoning_quality( # MODIFIED: Removed debate_history from arguments
        self, analysis_output: Dict[str, Any]
    ):
        """Analyzes the quality of reasoning in the debate process and final output."""
        # Use self.debate_history which is already available in the instance
        debate_history = self.debate_history
        
        self.reasoning_quality_metrics = {
            "argument_strength_score": 0.0,
            "debate_effectiveness": 0.0,
            "conflict_resolution_quality": 0.0,
            "80_20_adherence_score": 0.0,
            "reasoning_depth": 0,
            "critical_thinking_indicators": {
                "counter_arguments": 0,
                "evidence_citations": 0,
                "assumption_challenges": 0,
            },
            # NEW: Include historical self-improvement process quality metrics here
            "self_improvement_suggestion_success_rate_historical": self._get_historical_self_improvement_success_rate(),
            "schema_validation_failures_historical": dict(self._get_historical_schema_validation_failures()),
        }

        for turn in debate_history:
            content = ""
            if isinstance(turn.get("output"), dict):
                content = (
                    turn["output"].get("general_output", "")
                    or turn["output"].get("CRITIQUE_SUMMARY", "")
                    or turn["output"].get("ANALYSIS_SUMMARY", "")
                    or turn["output"].get("summary", "")
                )
            elif isinstance(turn.get("output"), str):
                content = turn["output"]

            content_lower = content.lower()
            self.reasoning_quality_metrics["critical_thinking_indicators"][
                "counter_arguments"
            ] += (
                content_lower.count("however")
                + content_lower.count("but")
                + content_lower.count("counterpoint")
            )
            self.reasoning_quality_metrics["critical_thinking_indicators"][
                "evidence_citations"
            ] += (
                content_lower.count("evidence")
                + content_lower.count("data shows")
                + content_lower.count("metrics indicate")
            )
            self.reasoning_quality_metrics["critical_thinking_indicators"][
                "assumption_challenges"
            ] += (
                content_lower.count("assumption")
                + content_lower.count("presumes")
                + content_lower.count("challenging the assumption")
            )

        analysis_text = str(analysis_output).lower()
        self.reasoning_quality_metrics["80_20_adherence_score"] = (
            0.8 if ("80/20" in analysis_text or "pareto" in analysis_text) else 0.3
        )

        ct_indicators = self.reasoning_quality_metrics["critical_thinking_indicators"]
        total_indicators = sum(ct_indicators.values())
        self.reasoning_quality_metrics["reasoning_depth"] = min(
            5, total_indicators // 3
        )
        
        # Store reasoning quality metrics in the main collected_metrics dictionary
        self.collected_metrics["reasoning_quality"] = self.reasoning_quality_metrics

    @classmethod
    def _collect_configuration_analysis(
        cls, codebase_path: str
    ) -> ConfigurationAnalysisOutput:
        """
        Collects structured information about existing tool configurations from
        critical project configuration files.
        """
        config_analysis_data = {
            "ci_workflow": {},
            "pre_commit_hooks": [],
            "pyproject_toml": {},
        }
        malformed_blocks = []

        # 1. Analyze .github/workflows/ci.yml
        ci_yml_path = Path(codebase_path) / ".github/workflows/ci.yml"
        if ci_yml_path.exists():
            try:
                with open(ci_yml_path, "r", encoding="utf-8") as f:
                    ci_config_raw = yaml.safe_load(f) or {}
                with open(
                    ci_yml_path, "r", encoding="utf-8"
                ) as f:
                    ci_content_lines = f.readlines()
                    ci_workflow_jobs = {}
                    
                    jobs_section = ci_config_raw.get("jobs")
                    if isinstance(jobs_section, dict):
                        for job_name, job_details in jobs_section.items():
                            if not isinstance(job_details, dict):
                                logger.warning(f"Job '{job_name}' in CI workflow is malformed (not a dictionary). Skipping.")
                                malformed_blocks.append({
                                    "type": "CI_JOB_MALFORMED",
                                    "message": f"Job '{job_name}' is not a dictionary.",
                                    "file": str(ci_yml_path),
                                    "job_name": job_name
                                })
                                continue

                            steps_summary = []
                            steps_section = job_details.get("steps")
                            if isinstance(steps_section, list):
                                for step in steps_section:
                                    if not isinstance(step, dict):
                                        logger.warning(f"Step in job '{job_name}' in CI workflow is malformed (not a dictionary). Skipping.")
                                        malformed_blocks.append({
                                            "type": "CI_STEP_MALFORMED",
                                            "message": f"Step in job '{job_name}' is not a dictionary.",
                                            "file": str(ci_yml_path),
                                            "job_name": job_name
                                        })
                                        continue

                                    step_name = step.get("name", "Unnamed Step")
                                    step_run = step.get("run")
                                    step_uses = step.get("uses")

                                    summary_item_data = {"name": step_name}
                                    if step_uses:
                                        summary_item_data["uses"] = step_uses
                                    if step_run:
                                        commands = [
                                            cmd.strip()
                                            for cmd in step_run.split("\n")
                                            if cmd.strip()
                                        ]
                                        summary_item_data["runs_commands"] = commands
                                        run_line_number = None
                                        for i, line in enumerate(ci_content_lines):
                                            if f'name: "{step_name}"' in line:
                                                for j in range(i, len(ci_content_lines)):
                                                    if "run:" in ci_content_lines[j]:
                                                        run_line_number = j + 1
                                                        break
                                                break
                                        summary_item_data["code_snippet"] = _get_code_snippet(
                                            ci_content_lines, run_line_number, context_lines=3
                                        )
                                    steps_summary.append(CiWorkflowStep(**summary_item_data))
                            else:
                                logger.warning(f"Steps section for job '{job_name}' in CI workflow is malformed (not a list). Skipping steps processing.")
                                malformed_blocks.append({
                                    "type": "CI_STEPS_SECTION_MALFORMED",
                                    "message": f"Steps section for job '{job_name}' is not a list.",
                                    "file": str(ci_yml_path),
                                    "job_name": job_name
                                })
                            ci_workflow_jobs[job_name] = CiWorkflowJob(
                                steps_summary=steps_summary
                            )
                    else:
                        logger.warning(f"Jobs section in CI workflow is malformed (not a dictionary). Skipping jobs processing.")
                        malformed_blocks.append({
                            "type": "CI_JOBS_SECTION_MALFORMED",
                            "message": "Jobs section is not a dictionary.",
                            "file": str(ci_yml_path)
                        })

                    config_analysis_data["ci_workflow"] = CiWorkflowConfig(
                        name=ci_config_raw.get("name"),
                        on_triggers=ci_config_raw.get("on"),
                        jobs=ci_workflow_jobs,
                    )
            except (yaml.YAMLError, OSError, ValidationError) as e:
                logger.error(f"Error parsing CI workflow file {ci_yml_path}: {e}")
                malformed_blocks.append(
                    {
                        "type": "CI_CONFIG_PARSE_ERROR",
                        "message": str(e),
                        "file": str(ci_yml_path),
                    }
                )

        # 2. Analyze .pre-commit-config.yaml
        pre_commit_path = Path(codebase_path) / ".pre-commit-config.yaml"
        if pre_commit_path.exists():
            try:
                with open(pre_commit_path, "r", encoding="utf-8") as f:
                    pre_commit_config_raw = yaml.safe_load(f) or {}
                with open(
                    pre_commit_path, "r", encoding="utf-8"
                ) as f:
                    pre_commit_content_lines = f.readlines()
                    
                    repos_section = pre_commit_config_raw.get("repos")
                    if isinstance(repos_section, list):
                        for repo_config in repos_section:
                            if not isinstance(repo_config, dict):
                                logger.warning(f"Repo config in pre-commit is malformed (not a dictionary). Skipping.")
                                malformed_blocks.append({
                                    "type": "PRE_COMMIT_REPO_MALFORMED",
                                    "message": "Repo config is not a dictionary.",
                                    "file": str(pre_commit_path)
                                })
                                continue

                            repo_url = repo_config.get("repo")
                            repo_rev = repo_config.get("rev")
                            
                            hooks_section = repo_config.get("hooks")
                            if isinstance(hooks_section, list):
                                for hook in hooks_section:
                                    if not isinstance(hook, dict):
                                        logger.warning(f"Hook config in pre-commit repo '{repo_url}' is malformed (not a dictionary). Skipping.")
                                        malformed_blocks.append({
                                            "type": "PRE_COMMIT_HOOK_MALFORMED",
                                            "message": f"Hook in repo '{repo_url}' is not a dictionary.",
                                            "file": str(pre_commit_path),
                                            "repo": repo_url
                                        })
                                        continue

                                    hook_id = hook.get("id")
                                    hook_args = hook.get("args", [])

                                    hook_line_number = None
                                    for i, line in enumerate(pre_commit_content_lines):
                                        if f"id: {hook_id}" in line:
                                            hook_line_number = i + 1
                                            break

                                    config_analysis_data["pre_commit_hooks"].append(
                                        PreCommitHook(
                                            repo=repo_url,
                                            rev=repo_rev,
                                            id=hook_id,
                                            args=hook_args,
                                            code_snippet=_get_code_snippet(
                                                pre_commit_content_lines,
                                                hook_line_number,
                                                context_lines=3,
                                            ),
                                        )
                                    )
                            else:
                                logger.warning(f"Hooks section for repo '{repo_url}' in pre-commit is malformed (not a list). Skipping hooks processing.")
                                malformed_blocks.append({
                                    "type": "PRE_COMMIT_HOOKS_SECTION_MALFORMED",
                                    "message": f"Hooks section for repo '{repo_url}' is not a list.",
                                    "file": str(pre_commit_path),
                                    "repo": repo_url
                                })
                    else:
                        logger.warning(f"Repos section in pre-commit is malformed (not a list). Skipping repos processing.")
                        malformed_blocks.append({
                            "type": "PRE_COMMIT_REPOS_SECTION_MALFORMED",
                            "message": "Repos section is not a dictionary.",
                            "file": str(pre_commit_path)
                        })

            except (yaml.YAMLError, OSError, ValidationError) as e:
                logger.error(
                    f"Error parsing pre-commit config file {pre_commit_path}: {e}"
                )
                malformed_blocks.append(
                    {
                        "type": "PRE_COMMIT_CONFIG_PARSE_ERROR",
                        "message": str(e),
                        "file": str(pre_commit_path),
                    }
                )

        # 3. Analyze pyproject.toml
        pyproject_path = Path(codebase_path) / "pyproject.toml"
        if pyproject_path.exists():
            try:
                with open(pyproject_path, "r", encoding="utf-8") as f:
                    pyproject_config_raw = toml.load(f) or {}
                with open(
                    pyproject_path, "r", encoding="utf-8"
                ) as f:
                    pyproject_content_lines = f.readlines()
                    pyproject_toml_data = {}

                    tool_section = pyproject_config_raw.get("tool")
                    if isinstance(tool_section, dict):
                        ruff_tool_config = tool_section.get("ruff")
                        if isinstance(ruff_tool_config, dict):
                            ruff_line_number = None
                            for i, line in enumerate(pyproject_content_lines):
                                if "[tool.ruff]" in line:
                                    ruff_line_number = i + 1
                                    break

                            lint_section_data = ruff_tool_config.get("lint")
                            if not isinstance(lint_section_data, dict):
                                lint_section_data = {}

                            pyproject_toml_data["ruff"] = RuffConfig(
                                line_length=ruff_tool_config.get("line-length"),
                                target_version=ruff_tool_config.get("target-version"),
                                lint_select=lint_section_data.get("select"),
                                lint_ignore=lint_section_data.get("ignore"),
                                format_settings=ruff_tool_config.get("format"),
                                config_snippet=_get_code_snippet(
                                    pyproject_content_lines,
                                    ruff_line_number,
                                    context_lines=5,
                                ),
                            )
                        elif ruff_tool_config is not None:
                            logger.warning(f"Ruff config in pyproject.toml is malformed (not a dictionary). Skipping.")
                            malformed_blocks.append({
                                "type": "PYPROJECT_RUFF_MALFORMED",
                                "message": "Ruff config is not a dictionary.",
                                "file": str(pyproject_path)
                            })

                        bandit_tool_config = tool_section.get("bandit")
                        if isinstance(bandit_tool_config, dict):
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
                                config_snippet=_get_code_snippet(
                                    pyproject_content_lines,
                                    bandit_line_number,
                                    context_lines=5,
                                ),
                            )
                        elif bandit_tool_config is not None:
                            logger.warning(f"Bandit config in pyproject.toml is malformed (not a dictionary). Skipping.")
                            malformed_blocks.append({
                                "type": "PYPROJECT_BANDIT_MALFORMED",
                                "message": "Bandit config is not a dictionary.",
                                "file": str(pyproject_path)
                            })

                        pydantic_settings_config = tool_section.get("pydantic-settings")
                        if isinstance(pydantic_settings_config, dict):
                            pyproject_toml_data["pydantic_settings"] = (
                                PydanticSettingsConfig(**pydantic_settings_config)
                            )
                        elif pydantic_settings_config is not None:
                            logger.warning(f"Pydantic-settings config in pyproject.toml is malformed (not a dictionary). Skipping.")
                            malformed_blocks.append({
                                "type": "PYPROJECT_PYDANTIC_SETTINGS_MALFORMED",
                                "message": "Pydantic-settings config is not a dictionary.",
                                "file": str(pyproject_path)
                            })
                    else:
                        logger.warning(f"Tool section in pyproject.toml is malformed (not a dictionary). Skipping tool processing.")
                        malformed_blocks.append({
                            "type": "PYPROJECT_TOOL_SECTION_MALFORMED",
                            "message": "Tool section is not a dictionary.",
                            "file": str(pyproject_path)
                        })

                    config_analysis_data["pyproject_toml"] = PyprojectTomlConfig(
                        **pyproject_toml_data
                    )

            except (toml.TomlDecodeError, OSError, ValidationError) as e:
                logger.error(f"Error parsing pyproject.toml file {pyproject_path}: {e}")
                malformed_blocks.append(
                    {
                        "type": "PYPROJECT_CONFIG_PARSE_ERROR",
                        "message": str(e),
                        "file": str(pyproject_path),
                    }
                )

        return ConfigurationAnalysisOutput(
            ci_workflow=config_analysis_data["ci_workflow"],
            pre_commit_hooks=config_analysis_data["pre_commit_hooks"],
            pyproject_toml=config_analysis_data["pyproject_toml"],
            malformed_blocks=malformed_blocks,
        )

    @classmethod
    def _collect_deployment_robustness_metrics(
        cls, codebase_path: str
    ) -> DeploymentAnalysisOutput:
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
            "malformed_blocks": [],
        }

        # 1. Analyze Dockerfile
        dockerfile_path = Path(codebase_path) / "Dockerfile"
        if dockerfile_path.exists():
            deployment_metrics_data["dockerfile_present"] = True
            try:
                with open(dockerfile_path, "r", encoding="utf-8") as f:
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
                deployment_metrics_data["dockerfile_exposed_ports"] = [
                    int(p) for p in exposed_ports
                ]

                if not re.search(
                    r"FROM\s+.*?AS\s+.*?\nFROM",
                    dockerfile_content,
                    re.DOTALL | re.IGNORECASE,
                ):
                    deployment_metrics_data["dockerfile_problem_snippets"].append(
                        "Missing multi-stage build. Consider using multiple FROM statements for smaller images."
                    )
                else:
                    deployment_metrics_data["dockerfile_multi_stage_build"] = True

            except OSError as e:
                logger.error(f"Error reading Dockerfile {dockerfile_path}: {e}")
                deployment_metrics_data["malformed_blocks"].append(
                    {
                        "type": "DOCKERFILE_READ_ERROR",
                        "message": str(e),
                        "file": str(dockerfile_path),
                    }
                )

        # 2. Analyze requirements-prod.txt and requirements.txt
        prod_req_path = Path(codebase_path) / "requirements-prod.txt"
        dev_req_path = Path(codebase_path) / "requirements.txt"

        prod_deps = set()
        unpinned_prod_deps = []
        if prod_req_path.exists():
            deployment_metrics_data["prod_requirements_present"] = True
            try:
                with open(prod_req_path, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if line and not line.startswith("#"):
                            if not re.search(r"[=~><]=", line):
                                unpinned_prod_deps.append(line)
                            prod_deps.add(
                                line.split("==")[0]
                                .split(">=")[0]
                                .split("~=")[0]
                                .lower()
                            )
                deployment_metrics_data["prod_dependency_count"] = len(prod_deps)
                deployment_metrics_data["unpinned_prod_dependencies"] = (
                    unpinned_prod_deps
                )
            except OSError as e:
                logger.error(
                    f"Error reading requirements-prod.txt {prod_req_path}: {e}"
                )
                deployment_metrics_data["malformed_blocks"].append(
                    {
                        "type": "PROD_REQ_READ_ERROR",
                        "message": str(e),
                        "file": str(prod_req_path),
                    }
                )

        if dev_req_path.exists() and prod_req_path.exists():
            dev_deps = set()
            try:
                with open(dev_req_path, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if line and not line.startswith("#"):
                            dev_deps.add(
                                line.split("==")[0]
                                .split(">=")[0]
                                .split("~=")[0]
                                .lower()
                            )

                overlap = prod_deps.intersection(dev_deps)
                deployment_metrics_data["dev_dependency_overlap_count"] = len(overlap)
            except OSError as e:
                logger.error(f"Error reading requirements.txt {dev_req_path}: {e}")
                deployment_metrics_data["malformed_blocks"].append(
                    {
                        "type": "DEV_REQ_READ_ERROR",
                        "message": str(e),
                        "file": str(dev_req_path),
                    }
                )

        return DeploymentAnalysisOutput(**deployment_metrics_data)

    def _collect_token_usage_stats(self) -> Dict[str, Any]:
        """
        Collects token usage statistics from debate intermediate steps.
        """
        total_tokens = self.intermediate_steps.get("Total_Tokens_Used", 0)
        total_cost = self.intermediate_steps.get("Total_Estimated_Cost_USD", 0.0)
        
        phase_token_usage = {}
        for key, value in self.intermediate_steps.items():
            if key.endswith("_Tokens_Used") and not key.startswith(
                ("Total_", "context_", "synthesis_", "debate_")
            ):
                persona_name = key.replace("_Tokens_Used", "")
                phase_token_usage[persona_name] = value
        
        # Calculate token efficiency (this was in _collect_core_metrics)
        suggestions_count = 0
        try:
            # This part is problematic as debate_history[-1]["output"] is the *last persona's output*
            # not necessarily the final synthesis output with suggestions.
            # The `token_efficiency` metric should probably be calculated *after* synthesis.
            # For now, we'll use a placeholder or calculate based on available suggestions in intermediate_steps.
            final_synthesis_output = self.intermediate_steps.get("Final_Synthesis_Output", {})
            if final_synthesis_output.get("version") == "1.0" and "data" in final_synthesis_output:
                suggestions_count = len(final_synthesis_output["data"].get("IMPACTFUL_SUGGESTIONS", []))
            elif "IMPACTFUL_SUGGESTIONS" in final_synthesis_output: # Direct V1 output
                suggestions_count = len(final_synthesis_output.get("IMPACTFUL_SUGGESTIONS", []))
        except Exception as e:
            logger.warning(f"Failed to parse final synthesis output for suggestions count in token stats: {e}")
            pass

        token_efficiency = (
            total_tokens / max(1, suggestions_count)
            if suggestions_count > 0
            else total_tokens # If no suggestions, efficiency is just total tokens
        )
        
        return {
            "total_tokens": total_tokens,
            "total_cost_usd": total_cost,
            "persona_token_usage": phase_token_usage,
            "token_efficiency": token_efficiency, # NEW: Add token efficiency here
        }

    def _analyze_debate_efficiency(self) -> Dict[str, Any]:
        """
        Analyzes the efficiency of the debate process.
        """
        efficiency_summary = {
            "num_turns": len(self.intermediate_steps.get("Debate_History", [])),
            "malformed_blocks_count": len(
                self.intermediate_steps.get("malformed_blocks", [])
            ),
            "conflict_resolution_attempts": 1
            if self.intermediate_steps.get("Conflict_Resolution_Attempt")
            else 0,
            "unresolved_conflict": bool(
                self.intermediate_steps.get("Unresolved_Conflict")
            ),
            "average_turn_tokens": 0.0,
            "persona_token_breakdown": {},
        }

        total_debate_tokens = self.intermediate_steps.get("debate_Tokens_Used", 0)
        num_turns = efficiency_summary["num_turns"]
        if num_turns > 0:
            efficiency_summary["average_turn_tokens"] = total_debate_tokens / num_turns

        for key, value in self.intermediate_steps.items():
            if key.endswith("_Tokens_Used") and not key.startswith(
                ("Total_", "context_", "synthesis_", "debate_")
            ):
                persona_name = key.replace("_Tokens_Used", "")
                efficiency_summary["persona_token_breakdown"][persona_name] = value

        return efficiency_summary

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

    @staticmethod
    def _generate_suggestion_id(suggestion: Dict) -> str:
        """Generates a consistent ID for a suggestion to track its impact over time."""
        import hashlib
        
        hash_input = (
            suggestion.get("AREA", "") + 
            suggestion.get("PROBLEM", "") + 
            suggestion.get("EXPECTED_IMPACT", "")
        )
        return hashlib.md5(hash_input.encode()).hexdigest()[:8]
    
    def analyze_historical_effectiveness(self) -> Dict[str, Any]:
        """Analyzes historical improvement data to identify patterns of success."""
        history_file = Path("data/improvement_history.jsonl")
        if not history_file.exists():
            return {
                "total_attempts": 0,
                "success_rate": 0.0,
                "top_performing_areas": [],
                "common_failure_modes": {},
                "historical_total_suggestions_processed": 0, # NEW: Return raw counts
                "historical_successful_suggestions": 0,
                "historical_schema_validation_failures": {},
            }
        
        try:
            with open(history_file, "r", encoding="utf-8") as f:
                records = [json.loads(line) for line in f if line.strip()]
            
            total = len(records)
            successful = sum(1 for r in records if r.get("success", False))

            # NEW: Aggregate raw counts from historical records
            total_suggestions_across_history = 0
            successful_suggestions_across_history = 0
            schema_validation_failures_across_history = defaultdict(int)
            for record in records:
                outcome = record.get("current_run_outcome", {})
                total_suggestions_across_history += outcome.get("total_suggestions_processed", 0)
                successful_suggestions_across_history += outcome.get("successful_suggestions", 0)
                for persona, count in outcome.get("schema_validation_failures", {}).items():
                    schema_validation_failures_across_history[persona] += count
            
            area_success = {}
            for record in records:
                for suggestion in record.get("suggestions", []):
                    area = suggestion.get("AREA", "Unknown")
                    if area not in area_success:
                        area_success[area] = {"attempts": 0, "successes": 0}
                    area_success[area]["attempts"] += 1
                    if record.get("success", False):
                        area_success[area]["successes"] += 1
            
            top_areas = [
                {
                    "area": area,
                    "success_rate": data["successes"] / data["attempts"],
                    "attempts": data["attempts"]
                }
                for area, data in area_success.items()
                if data["attempts"] > 2
            ]
            top_areas.sort(key=lambda x: x["success_rate"], reverse=True)
            
            return {
                "total_attempts": total,
                "success_rate": successful / total if total > 0 else 0.0,
                "top_performing_areas": top_areas[:3],
                "common_failure_modes": self._identify_common_failure_modes(records),
                "historical_total_suggestions_processed": total_suggestions_across_history,
                "historical_successful_suggestions": successful_suggestions_across_history,
                "historical_schema_validation_failures": dict(schema_validation_failures_across_history),
            }
        except Exception as e:
            logger.error(f"Error analyzing historical data: {e}")
            return {
                "total_attempts": 0,
                "success_rate": 0.0,
                "top_performing_areas": [],
                "common_failure_modes": {},
                "historical_total_suggestions_processed": 0,
                "historical_successful_suggestions": 0,
                "historical_schema_validation_failures": {},
            }
    
    @staticmethod
    def _identify_common_failure_modes(records: List[Dict]) -> Dict[str, int]:
        """Identifies common patterns in failed improvements by analyzing malformed_blocks and error types."""
        failure_modes_count = defaultdict(int)
        
        for record in records:
            if not record.get("success", False):
                for suggestion in record.get("suggestions", []):
                    for block in suggestion.get("malformed_blocks", []):
                        failure_modes_count[block.get("type", "UNKNOWN_MALFORMED_BLOCK")] += 1
                
                for category, changes in record.get("performance_changes", {}).items():
                    if "schema_validation_failures_count" in changes and changes["schema_validation_failures_count"].get("after", 0) > changes["schema_validation_failures_count"].get("before", 0):
                        failure_modes_count["schema_validation_failures_count"] += 1
                    if "token_budget_exceeded_count" in changes and changes["token_budget_exceeded_count"].get("after", 0) > changes["token_budget_exceeded_count"].get("before", 0):
                        failure_modes_count["token_budget_exceeded_count"] += 1

        return dict(failure_modes_count)

    # NEW: Method to record self-improvement suggestion outcomes
    def record_self_improvement_suggestion_outcome(
        self, persona_name: str, is_successful: bool, schema_failed: bool
    ):
        """
        Records the outcome of a self-improvement suggestion generated by a persona
        for the *current run*. This data will be saved historically.
        """
        self._current_run_total_suggestions_processed += 1
        if is_successful:
            self._current_run_successful_suggestions += 1
        if schema_failed:
            self._current_run_schema_validation_failures[persona_name] += 1

        logger.info(
            f"Recorded current run's self-improvement suggestion outcome for {persona_name}: "
            f"Successful={is_successful}, SchemaFailed={schema_failed}. "
            f"Current run total processed: {self._current_run_total_suggestions_processed}, "
            f"Successful: {self._current_run_successful_suggestions}"
        )

    def _get_critical_metric_info(self): # NEW: Renamed from get_critical_metric_info
        """Get information about the critical metric for prompt engineering."""
        if not self.critical_metric:
            return None

        config = self.CRITICAL_METRICS[self.critical_metric]
        value = self.collected_metrics.get(self.critical_metric, 0)
        threshold = config["threshold"]

        return {
            "name": self.critical_metric,
            "value": value,
            "threshold": threshold,
            "description": config["description"],
            "status": "CRITICAL"
            if (self.critical_metric == "token_efficiency" and value > threshold)
            or (self.critical_metric != "token_efficiency" and value < threshold)
            else "OK",
        }

    def _get_historical_self_improvement_success_rate(self) -> float:
        """Calculates the historical overall success rate of self-improvement suggestions."""
        if self._historical_total_suggestions_processed > 0:
            return (
                self._historical_successful_suggestions
                / self._historical_total_suggestions_processed
            )
        return 0.0

    def _get_historical_schema_validation_failures(self) -> Dict[str, int]:
        """Returns the historical counts of schema validation failures for self-improvement suggestions."""
        return self._historical_schema_validation_failures

    def _validate_and_fix_code_suggestion(self, code_change: Dict[str, Any]) -> Dict[str, Any]:
        """
        Internally validates a code suggestion using Ruff and Bandit.
        If Ruff formatting issues are found, it attempts to auto-fix them.
        Returns the (potentially fixed) code change and any remaining issues.
        """
        file_path_str = code_change.get("FILE_PATH")
        action = code_change.get("ACTION")
        content = code_change.get("FULL_CONTENT") or ""
        diff_content = code_change.get("DIFF_CONTENT")

        if not file_path_str or not action or not content:
            return code_change # Cannot validate malformed change

        if not file_path_str.endswith(".py"):
            return code_change # Only validate Python files

        # Create a temporary file for validation
        with tempfile.NamedTemporaryFile(mode="w+", suffix=".py", encoding="utf-8", delete=False) as temp_file:
            temp_file.write(content)
            temp_file.flush()
            tmp_file_path = Path(temp_file.name)

        try:
            # 1. Run Ruff Format (and auto-fix)
            format_command = [
                "ruff", "format", str(tmp_file_path)
            ]
            execute_command_safely(format_command, timeout=30, check=False)
            
            # Read the (potentially fixed) content back
            fixed_content = tmp_file_path.read_text(encoding="utf-8")
            
            # 2. Run Ruff Check (linting) and Bandit (security) on the fixed content
            ruff_issues = _run_ruff(fixed_content, file_path_str)
            bandit_issues = _run_bandit(fixed_content, file_path_str)
            ast_issues = _run_ast_security_checks(fixed_content, file_path_str)

            all_issues = ruff_issues + bandit_issues + ast_issues

            if fixed_content != content:
                # If content was fixed, update the code_change
                code_change["FULL_CONTENT"] = fixed_content
                # If it was a MODIFY action with DIFF_CONTENT, regenerate the diff
                if action == "MODIFY" and diff_content:
                    original_file_content = self.raw_file_contents.get(file_path_str, "")
                    code_change["DIFF_CONTENT"] = difflib.unified_diff(
                        original_file_content.splitlines(keepends=True),
                        fixed_content.splitlines(keepends=True),
                        fromfile=f"a/{file_path_str}",
                        tofile=f"b/{file_path_str}",
                        lineterm="",
                    )
                logger.info(f"Auto-fixed formatting for {file_path_str}. Remaining issues: {len(all_issues)}")
            
            if all_issues:
                # Attach remaining issues to the code_change for reporting
                code_change.setdefault("validation_issues", []).extend(all_issues)

        except Exception as e:
            logger.error(f"Error during internal validation/fix for {file_path_str}: {e}", exc_info=True)
            code_change.setdefault("validation_issues", []).append({
                "type": "Internal Validation Error",
                "message": f"Failed to internally validate/fix: {e}"
            })
        finally:
            if tmp_file_path and tmp_file_path.exists():
                os.unlink(tmp_file_path)
        
        return code_change

    def _process_suggestions_for_quality(self, suggestions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Iterates through suggestions and applies internal validation and auto-fixing
        to code changes.
        """
        processed_suggestions = []
        for suggestion in suggestions:
            if "CODE_CHANGES_SUGGESTED" in suggestion and suggestion["CODE_CHANGES_SUGGESTED"]:
                processed_code_changes = []
                for code_change in suggestion["CODE_CHANGES_SUGGESTED"]:
                    processed_code_changes.append(self._validate_and_fix_code_suggestion(code_change))
                suggestion["CODE_CHANGES_SUGGESTED"] = processed_code_changes
            processed_suggestions.append(suggestion)
        return processed_suggestions

    def _collect_code_quality_and_security_metrics(self):
        """
        Collects code quality (Ruff) and security (Bandit, AST) metrics for all Python files
        in the codebase. Stores results in self.collected_metrics and self.file_analysis_cache.
        """
        ruff_issues_count = 0
        bandit_issues_count = 0
        ast_security_issues_count = 0
        complexity_metrics = []
        code_smells_count = 0
        detailed_issues = []
        ruff_violations = [] # Specific list for Ruff violations

        for file_path_str, content in self.raw_file_contents.items():
            if file_path_str.endswith(".py"):
                content_lines = content.splitlines()
                
                # Cache for this file
                file_cache = self.file_analysis_cache.setdefault(file_path_str, {})

                # Run Ruff
                ruff_file_issues = _run_ruff(content, file_path_str)
                ruff_issues_count += len(ruff_file_issues)
                detailed_issues.extend(ruff_file_issues)
                ruff_violations.extend([issue for issue in ruff_file_issues if issue["type"] == "Ruff Linting Issue" or issue["type"] == "Ruff Formatting Issue"])
                file_cache["ruff_issues"] = ruff_file_issues

                # Run Bandit
                bandit_file_issues = _run_bandit(content, file_path_str)
                bandit_issues_count += len(bandit_file_issues)
                detailed_issues.extend(bandit_file_issues)
                file_cache["bandit_issues"] = bandit_file_issues

                # Run AST-based security checks
                ast_file_issues = _run_ast_security_checks(content, file_path_str)
                ast_security_issues_count += len(ast_file_issues)
                detailed_issues.extend(ast_file_issues)
                file_cache["ast_security_issues"] = ast_file_issues

                # Analyze AST for complexity and code smells
                function_metrics = self._analyze_python_file_ast(
                    content, content_lines, file_path_str
                )
                complexity_metrics.extend(function_metrics)
                code_smells_count += sum(
                    m.get("code_smells", 0) for m in function_metrics
                )
                file_cache["function_metrics"] = function_metrics

        self.collected_metrics["code_quality"] = {
            "ruff_issues_count": ruff_issues_count,
            "bandit_issues_count": bandit_issues_count, # Also include in code_quality for overview
            "ast_security_issues_count": ast_security_issues_count, # Also include in code_quality for overview
            "complexity_metrics": complexity_metrics,
            "code_smells_count": code_smells_count,
            "detailed_issues": detailed_issues,
            "ruff_violations": ruff_violations,
        }
        self.collected_metrics["security"] = { # Separate security section for clarity
            "bandit_issues_count": bandit_issues_count,
            "ast_security_issues_count": ast_security_issues_count,
            "detailed_security_issues": [issue for issue in detailed_issues if issue["type"] == "Bandit Security Issue" or issue["type"] == "Security Vulnerability (AST)"],
        }
        logger.info(f"Collected code quality and security metrics for {len(self.raw_file_contents)} files.")

    def collect_all_metrics(self) -> Dict[str, Any]: # NEW: Public method to collect all metrics
        """
        Collects all objective metrics that are available *before* the synthesis persona runs.
        This is the main entry point for `core.py` to get metrics for the synthesis prompt.
        """
        logger.info("Performing self-analysis for Project Chimera.")

        # Reset collected_metrics to ensure a fresh collection for each run
        self.collected_metrics = {}
        self.file_analysis_cache = {} # Clear cache for fresh run

        # Retrieve Conflict_Resolution_Attempt safely, handling potential None values
        # FIX: Ensure that .get() is not called on None.
        conflict_resolution_attempt_data = self.intermediate_steps.get("Conflict_Resolution_Attempt")
        
        # Initialize conflict_resolution_summary as an empty dict if attempt data is None or empty
        conflict_resolution_summary = {}
        if conflict_resolution_attempt_data is not None:
            conflict_resolution_summary = conflict_resolution_attempt_data.get("resolution_summary", {})

        # Initialize suggestions list, which will be part of collected_metrics
        initial_suggestions_from_conflict = [] # Renamed to avoid confusion with the 'suggestions' list in generate_suggestions

        if (
            conflict_resolution_summary
            and conflict_resolution_summary.get("conflict_resolved")
            and "cannot be fulfilled due to the absence of the codebase"
            in conflict_resolution_summary.get("RATIONALE", "")
        ):
            initial_suggestions_from_conflict.append(
                {
                    "AREA": "Maintainability",
                    "PROBLEM": "Critical lack of codebase access prevents meaningful code-level analysis and improvements. The system cannot perform security, robustness, or detailed maintainability analyses without the codebase. The conflict resolution summary correctly identified the need for the codebase to proceed.",
                    "PROPOSED_SOLUTION": "Establish a mechanism for providing the Project Chimera codebase and its context. This includes providing relevant files, their purpose, architecture, and any prior analysis. A `docs/project_chimera_context.md` file is proposed to guide this collection.",
                    "EXPECTED_IMPACT": "Enables the self-improvement process to proceed effectively, allowing for specific vulnerability identification, optimization opportunities, and actionable code modifications. This directly addresses the 'Maintainability' aspect of the self-improvement goals.",
                    "CODE_CHANGES_SUGGESTED": [
                        {
                            "FILE_PATH": "docs/project_chimera_context.md",
                            "ACTION": "ADD",
                            "FULL_CONTENT": """# Project Chimera Self-Improvement Methodology

This document outlines the refined methodology for identifying and implementing self-improvement strategies for Project Chimera. Recognizing that AI self-improvement is fundamentally different from traditional software refactoring, this methodology prioritizes experimental interventions and data-driven optimizations.

## Core Principles:

1.  **AI-Centric Optimization:** Improvements are driven by adjustments to the AI model's architecture, training data, hyperparameters, and inference strategies, not solely by static code modifications.
2.  **Objective Metrics:** All proposed improvements must be tied to measurable metrics that quantify improvements in:
    *   **Reasoning Quality:** Accuracy on specific benchmarks, logical consistency, coherence, factual correctness.
    *   **Robustness:** Performance under noisy or adversarial inputs, graceful degradation.
    *   **Efficiency:** Inference latency, token usage per query, computational cost.
3.  **Experimental Interventions:** Suggestions will be framed as experiments. Each suggestion will propose a specific intervention (e.g., \"fine-tune on dataset X\", \"adjust temperature parameter to Y\", \"implement retrieval-augmented generation with source Z\") and the metrics to evaluate its its success.
4.  **80/20 Principle Applied to Experiments:** Identify interventions with the highest potential impact on the defined metrics, prioritizing those that address core AI capabilities.

## Process:

1.  **Identify Weakness:** Analyze AI performance against defined metrics to pinpoint areas for improvement.
2.  **Propose Experiment:** Formulate a specific, testable intervention targeting the identified weakness.
3.  **Define Metrics:** Specify the objective metrics that will be used to evaluate the experiment's success.
4.  **Implement & Measure:** Execute the experiment and collect data on the defined metrics.
5.  **Iterate:** Based on results, refine the intervention or propose new experiments.

## Example Suggestion Format:

*   **AREA:** Reasoning Quality
*   **PROBLEM:** The AI exhibits logical inconsistencies in complex multi-turn debates.
*   **PROPOSED_SOLUTION:** Experiment with fine-tuning the LLM on a curated dataset of high-quality Socratic dialogues, focusing on logical argumentation and refutation. Measure improvements using a custom benchmark assessing logical fallacies and argument coherence.
*   **EXPECTED_IMPACT:** Enhanced logical consistency and reduced instances of fallacious reasoning in debates.
*   **CODE_CHANGES_SUGGESTED:** [] (As the change is algorithmic/data-driven, direct code changes may not be applicable or the primary focus. If code is involved, it would be in data processing or training scripts, e.g., `src/data/prepare_socratic_dialogues.py`)""",
                        }
                    ],
                }
            )
            # FIX: Store these suggestions in collected_metrics, do not return early.
            self.collected_metrics["initial_suggestions_from_conflict"] = initial_suggestions_from_conflict
        
        # Collect token usage stats
        self.collected_metrics["performance_efficiency"] = self._collect_token_usage_stats()
        
        # Collect debate efficiency metrics
        self.collected_metrics["debate_efficiency"] = self._analyze_debate_efficiency()

        # Assess test coverage
        self.collected_metrics["maintainability"] = {
            "test_coverage_summary": self._assess_test_coverage()
        }

        # Collect configuration analysis
        self.collected_metrics["configuration_analysis"] = self._collect_configuration_analysis(str(self.codebase_path))

        # Collect deployment robustness metrics
        self.collected_metrics["deployment_robustness"] = self._collect_deployment_robustness_metrics(str(self.codebase_path))

        # Collect code quality and security metrics
        self._collect_code_quality_and_security_metrics()

        # Identify the critical metric based on collected data
        # This should be called *after* all relevant metrics are in self.collected_metrics
        self._identify_critical_metric() # <-- This call is important here

        logger.info("Finished collecting all pre-synthesis metrics.")
        return self.collected_metrics

    # REMOVED: The entire generate_suggestions method (lines 1090-1293 in the original file)
    # Its logic is now handled by the Self_Improvement_Analyst persona via LLM.

    def analyze_codebase_structure(self) -> Dict[str, Any]:
        logger.info("Analyzing codebase structure.")
        return {"summary": "Codebase structure analysis is a placeholder."}

    def analyze_performance_bottlenecks(self) -> Dict[str, Any]:
        logger.info("Analyzing performance bottlenecks.")
        return {"summary": "Performance bottleneck analysis is a placeholder."}