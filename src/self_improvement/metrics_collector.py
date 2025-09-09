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
        codebase_context: Dict[str, str],
        tokenizer: Any,
        llm_provider: Any,
        persona_manager: Any,
        content_validator: Any,
        # REMOVED: metrics_collector: Any, # This argument was incorrectly added and caused the TypeError
    ):
        """Initialize with debate context for analysis."""
        self.initial_prompt = initial_prompt # FIXED: Corrected from 'self.metrics = metrics'
        self.debate_history = debate_history
        self.intermediate_steps = intermediate_steps
        self.codebase_context = codebase_context
        self.tokenizer = tokenizer
        self.llm_provider = llm_provider
        self.persona_manager = persona_manager
        self.content_validator = content_validator
        # REMOVED: self.metrics_collector = metrics_collector # This was incorrectly stored
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


    def _collect_core_metrics(self, tokenizer, llm_provider):
        """Collect core metrics and identify the single most critical bottleneck."""
        input_tokens = tokenizer.count_tokens(self.initial_prompt)
        output_tokens = 0
        if self.debate_history and len(self.debate_history) > 0:
            last_response = self.debate_history[-1].get("response", "")
            output_tokens = tokenizer.count_tokens(last_response)

        suggestions_count = 0
        try:
            analysis = json.loads(
                self.debate_history[-1]["output"]
            )
            suggestions_count = len(analysis.get("IMPACTFUL_SUGGESTIONS", []))
        except Exception as e:
            logger.warning(f"Failed to parse debate history for suggestions count: {e}")
            pass

        self.collected_metrics["token_efficiency"] = ( # Use self.collected_metrics
            output_tokens / max(1, suggestions_count)
            if suggestions_count > 0
            else output_tokens
        )

        self._identify_critical_metric()

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

    def get_critical_metric_info(self):
        """Get information about the critical metric for prompt engineering."""
        if not self.critical_metric:
            return None

        config = self.CRITICAL_METRICS[self.critical_metric]
        value = self.collected_metrics.get(self.critical_metric, 0) # Use self.collected_metrics
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

    def analyze_reasoning_quality(
        self, debate_history: List[Dict[str, Any]], analysis_output: Dict[str, Any]
    ):
        """Analyzes the quality of reasoning in the debate process and final output."""
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

        # No need to assign to self.metrics here, as self.reasoning_quality_metrics is already updated.

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

        return {
            "total_tokens": total_tokens,
            "total_cost_usd": total_cost,
            "persona_token_usage": phase_token_usage,
        }

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
            "persona_specific_performance": defaultdict(
                lambda: {
                    "success_rate": 0.0,
                    "schema_failures": 0,
                    "truncations": 0,
                    "total_turns": 0,
                }
            ),
            "prompt_verbosity_score": 0.0,
            "malformed_blocks_summary": defaultdict(int),
        }

        debate_history = self.intermediate_steps.get("Debate_History", [])
        reasoning_metrics["total_debate_turns"] = len(debate_history)

        unique_personas = set()
        for turn in debate_history:
            if "persona" in turn:
                unique_personas.add(turn["persona"])
        reasoning_metrics["unique_personas_involved"] = len(unique_personas)

        all_malformed_blocks = self.intermediate_steps.get("malformed_blocks", [])
        reasoning_metrics["schema_validation_failures_count"] = sum(
            1
            for b in all_malformed_blocks
            if b.get("type") == "SCHEMA_VALIDATION_ERROR"
        )
        reasoning_metrics["content_misalignment_warnings"] = sum(
            1 for b in all_malformed_blocks if b.get("type") == "CONTENT_MISALIGNMENT"
        )
        reasoning_metrics["debate_turn_errors"] = sum(
            1 for b in all_malformed_blocks if b.get("type") == "DEBATE_TURN_ERROR"
        )

        for block in all_malformed_blocks:
            reasoning_metrics["malformed_blocks_summary"][
                block.get("type", "UNKNOWN_MALFORMED_BLOCK")
            ] += 1

        if self.intermediate_steps.get("Conflict_Resolution_Attempt"):
            reasoning_metrics["conflict_resolution_attempts"] = 1
            if self.intermediate_steps["Conflict_Resolution_Attempt"].get(
                "conflict_resolved"
            ):
                reasoning_metrics["conflict_resolution_successes"] = 1
        reasoning_metrics["unresolved_conflict_present"] = bool(
            self.intermediate_steps.get("Unresolved_Conflict")
        )

        total_output_tokens = 0
        for key, value in self.intermediate_steps.items():
            if key.endswith("_Tokens_Used") and not key.startswith(
                ("Total_", "context_", "synthesis_", "debate_")
            ):
                total_output_tokens += value

        if reasoning_metrics["total_debate_turns"] > 0:
            reasoning_metrics["average_persona_output_tokens"] = (
                total_output_tokens / reasoning_metrics["total_debate_turns"]
            )

        for persona_name in unique_personas:
            persona_malformed_blocks = [
                b for b in all_malformed_blocks if b.get("persona") == persona_name
            ]
            schema_failures = sum(
                1
                for b in persona_malformed_blocks
                if b.get("type") == "SCHEMA_VALIDATION_ERROR"
            )
            content_misalignments = sum(
                1
                for b in persona_malformed_blocks
                if b.get("type") == "CONTENT_MISALIGNMENT"
            )

            persona_turns = sum(
                1 for turn in debate_history if turn.get("persona") == persona_name
            )

            reasoning_metrics["persona_specific_performance"][persona_name][
                "total_turns"
            ] = persona_turns
            reasoning_metrics["persona_specific_performance"][persona_name][
                "schema_failures"
            ] = schema_failures
            reasoning_metrics["persona_specific_performance"][persona_name][
                "truncations"
            ] = 0

            if persona_turns > 0:
                reasoning_metrics["persona_specific_performance"][persona_name][
                    "success_rate"
                ] = (
                    persona_turns - schema_failures - content_misalignments
                ) / persona_turns
            else:
                reasoning_metrics["persona_specific_performance"][persona_name][
                    "success_rate"
                ] = 0.0

        return reasoning_metrics

    def collect_all_metrics(self) -> Dict[str, Any]:
        """
        Collect all relevant metrics from the codebase and debate history for self-improvement analysis.
        """
        # Ensure reasoning_quality_metrics is populated before being used in the 'metrics' dict
        self.analyze_reasoning_quality(
            self.debate_history, self.intermediate_steps.get("Final_Synthesis_Output", {})
        )

        metrics = {
            "code_quality": {
                "ruff_issues_count": 0,
                "complexity_metrics": {
                    "avg_cyclomatic_complexity": 0.0,
                    "avg_loc_per_function": 0.0,
                    "avg_num_arguments": 0.0,
                    "avg_max_nesting_depth": 0.0,
                },
                "code_smells_count": 0,
                "detailed_issues": [],
                "ruff_violations": [],
            },
            "security": {
                "bandit_issues_count": 0,
                "ast_security_issues_count": 0,
            },
            "performance_efficiency": {
                "token_usage_stats": self._collect_token_usage_stats(),
                "debate_efficiency_summary": self._analyze_debate_efficiency(),
                "potential_bottlenecks_count": 0,
            },
            "robustness": {
                "schema_validation_failures_count": len(
                    self.intermediate_steps.get("malformed_blocks", [])
                ),
                "unresolved_conflict_present": bool(
                    self.intermediate_steps.get("Unresolved_Conflict")
                ),
                "conflict_resolution_attempted": bool(
                    self.intermediate_steps.get("Conflict_Resolution_Attempt")
                ),
            },
            "maintainability": {"test_coverage_summary": self._assess_test_coverage()},
            "configuration_analysis": self._collect_configuration_analysis(
                self.codebase_path
            ).model_dump(by_alias=True),
            "deployment_robustness": self._collect_deployment_robustness_metrics(
                self.codebase_path
            ).model_dump(by_alias=True),
            "reasoning_quality": self.reasoning_quality_metrics, # Now this is safe
            "historical_analysis": self.analyze_historical_effectiveness(),
        }

        total_functions_across_codebase = 0 # NEW: Initialize
        total_loc_across_functions = 0 # NEW: Initialize
        total_complexity_across_functions = 0 # NEW: Initialize
        total_args_across_functions = 0 # NEW: Initialize
        total_nesting_depth_across_codebase = 0 # NEW: Initialize

        for root, _, files in os.walk(self.codebase_path):
            for file in files:
                if file.endswith(".py"):
                    file_path = os.path.join(root, file)
                    try:
                        with open(file_path, "r", encoding="utf-8") as f:
                            content = f.read()
                            content_lines = content.splitlines()

                        if file_path not in self.file_analysis_cache:
                            self.file_analysis_cache[file_path] = {}

                        ruff_issues = _run_ruff(content, file_path)
                        if ruff_issues:
                            metrics["code_quality"]["ruff_issues_count"] += len(
                                ruff_issues
                            )
                            metrics["code_quality"]["detailed_issues"].extend(
                                ruff_issues
                            )
                            metrics["code_quality"]["ruff_violations"].extend(
                                ruff_issues
                            )
                            self.file_analysis_cache[file_path]["ruff_issues"] = ruff_issues

                        bandit_issues = _run_bandit(content, file_path)
                        if bandit_issues:
                            metrics["security"]["bandit_issues_count"] += len(
                                bandit_issues
                            )
                            metrics["code_quality"]["detailed_issues"].extend(
                                bandit_issues
                            )
                            self.file_analysis_cache[file_path]["bandit_issues"] = bandit_issues

                        ast_security_issues = _run_ast_security_checks(
                            content, file_path
                        )
                        if ast_security_issues:
                            metrics["security"]["ast_security_issues_count"] += len(
                                ast_security_issues
                            )
                            self.file_analysis_cache[file_path]["ast_security_issues"] = ast_security_issues
                            metrics["code_quality"]["detailed_issues"].extend(
                                ast_security_issues
                            )

                        file_function_metrics = self._analyze_python_file_ast(
                            content, content_lines, file_path
                        )

                        for func_metric in file_function_metrics:
                            total_functions_across_codebase += 1
                            total_complexity_across_functions += func_metric[
                                "cyclomatic_complexity"
                            ]
                            total_loc_across_functions += func_metric["loc"]
                            total_args_across_functions += func_metric["num_arguments"]
                            total_nesting_depth_across_codebase += func_metric[
                                "max_nesting_depth"
                            ]
                            metrics["code_quality"]["code_smells_count"] += func_metric[
                                "code_smells"
                            ]
                            metrics["performance_efficiency"][
                                "potential_bottlenecks_count"
                            ] += func_metric["potential_bottlenecks"]

                    except Exception as e:
                        logger.error(
                            f"Error collecting code metrics for {file_path}: {e}",
                            exc_info=True,
                        )

        if total_functions_across_codebase > 0:
            metrics["code_quality"]["complexity_metrics"]["avg_cyclomatic_complexity"] = (
                total_complexity_across_functions / total_functions_across_codebase
            )
            metrics["code_quality"]["complexity_metrics"]["avg_loc_per_function"] = (
                total_loc_across_functions / total_functions_across_codebase
            )
            metrics["code_quality"]["complexity_metrics"]["avg_num_arguments"] = (
                total_args_across_functions / total_functions_across_codebase
            )
            metrics["code_quality"]["complexity_metrics"]["avg_max_nesting_depth"] = (
                total_nesting_depth_across_codebase / total_functions_across_codebase
            )

        return metrics

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

        return {
            "total_tokens": total_tokens,
            "total_cost_usd": total_cost,
            "persona_token_usage": phase_token_usage,
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
            # FIX: Construct the command without the python executable,
            # allowing execute_command_safely to handle it.
            command = [
                "pytest", "-v", "tests/", "--cov=src", "--cov-report=json:coverage.json"
            ]
            # Use execute_command_safely for robustness
            return_code, stdout, stderr = execute_command_safely(command, timeout=120, check=False)

            # Pytest returns 0 for success, 1 for failed tests, 2 for internal errors/usage errors.
            # FIX: Only consider exit code 0 as full success for the command itself.
            # Test failures (exit code 1) are still a valid execution for coverage reporting.
            if return_code not in (0, 1): # Keep 0 or 1 as valid for coverage report generation
                logger.warning(f"Pytest coverage command failed with return code {return_code}. Stderr: {stderr}")
                # Provide more detailed error info, including stdout for debugging.
                coverage_data["coverage_details"] = f"Pytest command failed with exit code {return_code}. Stderr: {stderr or 'Not available'}. Stdout: {stdout or 'Not available'}."
                return coverage_data

            coverage_json_path = Path("coverage.json")
            if coverage_json_path.exists():
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
            else:
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

    def save_improvement_results(
        self,
        suggestions: List[Dict],
        metrics_before: Dict,
        metrics_after: Dict,
        success: bool,
    ):
        """Save results of improvement attempt for future learning"""
        from datetime import datetime
        import json
        from pathlib import Path

        performance_changes = {}
        for category, metrics in metrics_after.items():
            if category in metrics_before:
                category_changes = {}
                for metric, value_after in metrics.items():
                    if metric in metrics_before[category]:
                        value_before = metrics_before[category][metric]

                        if isinstance(value_before, (int, float)) and isinstance(
                            value_after, (int, float)
                        ):
                            absolute_change = value_after - value_before
                            percent_change = (
                                (absolute_change / value_before * 100)
                                if value_before != 0
                                else float("inf")
                            )

                            category_changes[metric] = {
                                "absolute_change": absolute_change,
                                "percent_change": percent_change,
                            }
                        else:
                            category_changes[metric] = {
                                "changed": value_before != value_after,
                                "before": value_before,
                                "after": value_after,
                            }
                if category_changes:
                    performance_changes[category] = category_changes

        improvement_record = {
            "timestamp": datetime.now().isoformat(),
            "suggestions": suggestions,
            "suggestion_ids": [self._generate_suggestion_id(s) for s in suggestions],
            "metrics_before": metrics_before,
            "metrics_after": metrics_after,
            "success": success,
            "performance_changes": performance_changes,
            "improvement_score": self.intermediate_steps.get("improvement_score", 0.0),
            "current_run_outcome": { # NEW: Add current run's outcome to the historical record
                "total_suggestions_processed": self._current_run_total_suggestions_processed,
                "successful_suggestions": self._current_run_successful_suggestions,
                "schema_validation_failures": dict(self._current_run_schema_validation_failures),
            },
        }
        history_file = Path("data/improvement_history.jsonl")
        history_file.parent.mkdir(exist_ok=True)
        try:
            with open(history_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(improvement_record) + "\n")
            logger.info(f"Saved improvement results to {history_file}")
        except IOError as e:
            logger.error(f"Failed to save improvement results to {history_file}: {e}")
    
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

    def analyze(self) -> List[Dict[str, Any]]:
        """
        Performs the self-analysis and generates improvement suggestions.
        Focuses on the top 3 highest impact areas based on metrics, adhering to the Pareto principle.
        """
        logger.info("Performing self-analysis for Project Chimera.")

        suggestions = []

        conflict_resolution_summary = self.intermediate_steps.get(
            "Conflict_Resolution_Attempt", {}
        ).get("resolution_summary", {})
        if (
            conflict_resolution_summary
            and conflict_resolution_summary.get("conflict_resolved")
            and "cannot be fulfilled due to the absence of the codebase"
            in conflict_resolution_summary.get("RATIONALE", "")
        ):
            suggestions.append(
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
3.  **Experimental Interventions:** Suggestions will be framed as experiments. Each suggestion will propose a specific intervention (e.g., \"fine-tune on dataset X\", \"adjust temperature parameter to Y\", \"implement retrieval-augmented generation with source Z\") and the metrics to evaluate its success.
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
            return suggestions

        top_ruff_issues_snippets = []
        top_bandit_issues_snippets = []

        ruff_detailed_issues = [
            issue
            for issue in self.collected_metrics.get("code_quality", {}).get("detailed_issues", []) # Use self.collected_metrics
            if issue.get("source") == "ruff_lint"
            or issue.get("source") == "ruff_format"
        ]
        for issue in ruff_detailed_issues[:3]:
            snippet = issue.get("code_snippet")
            if snippet:
                top_ruff_issues_snippets.append(
                    f"  - File: `{issue.get('file', 'N/A')}` (Line: {issue.get('line', 'N/A')}): `{issue.get('code', 'N/A')}` - {issue.get('message', 'N/A')}\n```\n{snippet}\n```"
                )
            else:
                top_ruff_issues_snippets.append(
                    f"  - File: `{issue.get('file', 'N/A')}` (Line: {issue.get('line', 'N/A')}): `{issue.get('code', 'N/A')}` - {issue.get('message', 'N/A')}"
                )

        bandit_detailed_issues = [
            issue
            for issue in self.collected_metrics.get("code_quality", {}).get("detailed_issues", []) # Use self.collected_metrics
            if issue.get("source") == "bandit"
        ]
        for issue in bandit_detailed_issues[:3]:
            snippet = issue.get("code_snippet")
            if snippet:
                top_bandit_issues_snippets.append(
                    f"  - File: `{issue.get('file', 'N/A')}` (Line: {issue.get('line', 'N/A')}): `{issue.get('code', 'N/A')}` - {issue.get('message', 'N/A')}\n```\n{snippet}\n```"
                )
            else:
                top_bandit_issues_snippets.append(
                    f"  - File: `{issue.get('file', 'N/A')}` (Line: {issue.get('line', 'N/A')}): `{issue.get('code', 'N/A')}` - {issue.get('message', 'N/A')}"
                )

        ruff_issues_count = self.collected_metrics.get("code_quality", {}).get( # Use self.collected_metrics
            "ruff_issues_count", 0
        )
        if ruff_issues_count > 100:
            suggestions.append(
                {
                    "AREA": "Maintainability",
                    "PROBLEM": f"The project exhibits widespread Ruff formatting issues across numerous files (e.g., `core.py`, `code_validator.py`, `app.py`, all test files, etc.). The `code_quality.ruff_violations` list contains {ruff_issues_count} entries, predominantly `FMT` (formatting) errors. This inconsistency detracts from readability and maintainability. Examples:\n"
                    + "\n".join(top_ruff_issues_snippets),
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
""",
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
""",
                        },
                    ],
                }
            )

        bandit_issues_count = self.collected_metrics.get("security", {}).get( # Use self.collected_metrics
            "bandit_issues_count", 0
        )
        pyproject_config_error = any(
            block.get("type") == "PYPROJECT_CONFIG_PARSE_ERROR"
            for block in self.collected_metrics.get("configuration_analysis", {}).get( # Use self.collected_metrics
                "malformed_blocks", []
            )
        )

        if (
            bandit_issues_count > 0 or pyproject_config_error
        ):
            problem_description = f"Bandit security scans are failing with configuration errors (`Bandit failed with exit code 2: [config] ERROR Invalid value (at line 33, column 15) [main] ERROR /Users/tom/Documents/apps/project_chimera/pyproject.toml : Error parsing file.`). This indicates a misconfiguration in `pyproject.toml` for Bandit, preventing security vulnerabilities from being detected. The `pyproject.toml` file itself has a `PYPROJECT_CONFIG_PARSE_ERROR` related to `ruff` configuration."
            if bandit_issues_count > 0:
                problem_description += (
                    f"\nAdditionally, {bandit_issues_count} Bandit security vulnerabilities were detected. Prioritize HIGH severity issues like potential injection flaws. Examples:\n"
                    + "\n".join(top_bandit_issues_snippets)
                )

            suggestions.append(
                {
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
""",
                        },
                        {
                            "FILE_PATH": ".github/workflows/ci.yml",
                            "ACTION": "MODIFY",
                            "DIFF_CONTENT": """--- a/.github/workflows/ci.yml
+++ /.github/workflows/ci.yml
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
""",
                        },
                    ],
                }
            )

        zero_test_coverage = (
            self.collected_metrics.get("maintainability", {}) # Use self.collected_metrics
            .get("test_coverage_summary", {})
            .get("overall_coverage_percentage", 0)
            == 0
        )
        if zero_test_coverage:
            suggestions.append(
                {
                    "AREA": "Maintainability",
                    "PROBLEM": "The project lacks automated test coverage. The `maintainability.test_coverage_summary` shows `overall_coverage_percentage: 0.0` and `coverage_details: 'Automated test coverage assessment not implemented.'`. This significantly hinders the ability to refactor code confidently, introduce new features without regressions, and ensure the long-term health of the codebase.",
                    "PROPOSED_SOLUTION": "Implement a comprehensive testing strategy. This includes writing unit tests for core logic (e.g., LLM interactions, data processing, utility functions) and integration tests for key workflows. Start with critical modules like `src/llm_provider.py`, `src/utils/prompt_engineering.py`, and `src/persona_manager.py`. Aim for a minimum of 70% test coverage within the next iteration.",
                    "EXPECTED_IMPACT": "Improved code stability, reduced regression bugs, increased developer confidence during changes, and a clearer understanding of code behavior. This directly addresses the 'Maintainability' aspect of the self-improvement goals.",
                    "CODE_CHANGES_SUGGESTED": [
                        {
                            "FILE_PATH": "tests/test_llm_provider.py",
                            "ACTION": "ADD",
                            "FULL_CONTENT": """import pytest
from src.llm_provider import GeminiProvider # Corrected import

# Mocking the LLM API for testing
class MockLLMClient:
    def __init__(self, model_name):
        self.model_name = model_name

    def generate_content(self, contents, config): # Updated signature to match genai.Client
        # Simulate a response based on prompt content
        prompt_content = contents # Assuming 'contents' is the prompt string
        if "summarize" in prompt_content.lower():
            return MagicMock(candidates=[MagicMock(content=MagicMock(parts=[MagicMock(text="This is a simulated summary.")]))])
        elif "analyze" in prompt_content.lower():
            return MagicMock(candidates=[MagicMock(content=MagicMock(parts=[MagicMock(text="this is a simulated analysis.")]))])
        else:
            return MagicMock(candidates=[MagicMock(content=MagicMock(parts=[MagicMock(text="This is a simulated default response.")]))])

    def count_tokens(self, model, contents): # Updated signature to match genai.Client
        return MagicMock(total_tokens=len(contents) // 4) # Simulate token count

@pytest.fixture
def llm_provider_instance():
    # Use the mock client for testing
    mock_client = MockLLMClient(model_name="mock-model")
    mock_tokenizer = MagicMock() # Mock tokenizer
    mock_tokenizer.count_tokens.side_effect = lambda text: len(text) // 4 # Simple token count
    mock_tokenizer.max_output_tokens = 8192 # Set a default max_output_tokens
    
    # Patch genai.Client during fixture creation
    with patch('src.llm_provider.genai.Client', return_value=mock_client):
        provider = GeminiProvider(
            api_key="mock-key",
            model_name="mock-model",
            tokenizer=mock_tokenizer,
            settings=MagicMock() # Mock settings
        )
        return provider

def test_llm_provider_initialization(llm_provider_instance):
    \"\"\"Test that the GeminiProvider initializes correctly.\"\"\"
    assert llm_provider_instance.model_name == "mock-model"

def test_llm_provider_generate_content_summary(llm_provider_instance):
    \"\"\"Test content generation for a summarization prompt.\"\"\"
    prompt = "Please summarize the following text: ..."
    response_text, input_tokens, output_tokens, is_truncated = llm_provider_instance.generate(
        prompt=prompt,
        system_prompt="You are a helpful assistant.",
        temperature=0.7,
        max_tokens=100,
    )
    assert response_text == "This is a simulated summary."
    assert input_tokens > 0
    assert output_tokens > 0
    assert is_truncated == False

def test_llm_provider_generate_content_analysis(llm_provider_instance):
    \"\"\"Test content generation for an analysis prompt.\"\"\"
    prompt = "Analyze the provided data: ..."
    response_text, input_tokens, output_tokens, is_truncated = llm_provider_instance.generate(
        prompt=prompt,
        system_prompt="You are a helpful assistant.",
        temperature=0.7,
        max_tokens=100,
    )
    assert response_text == "this is a simulated analysis."
    assert input_tokens > 0
    assert output_tokens > 0
    assert is_truncated == False

def test_llm_provider_generate_content_default(llm_provider_instance):
    \"\"\"Test content generation for a general prompt.\"\"\"
    prompt = "What is the capital of France?"
    response_text, input_tokens, output_tokens, is_truncated = llm_provider_instance.generate(
        prompt=prompt,
        system_prompt="You are a helpful assistant.",
        temperature=0.7,
        max_tokens=100,
    )
    assert response_text == "This is a simulated default response."
    assert input_tokens > 0
    assert output_tokens > 0
    assert is_truncated == False

# Add more tests for different scenarios and edge cases
""",
                        },
                        {
                            "FILE_PATH": "tests/test_prompt_engineering.py",
                            "ACTION": "ADD",
                            "FULL_CONTENT": """import pytest
from src.utils.prompt_engineering import format_prompt
from src.persona_manager import PersonaManager # Needed for mocking in session_manager
from unittest.mock import MagicMock # Import MagicMock

# Mock app_config and EXAMPLE_PROMPTS for session_manager initialization
@pytest.fixture
def mock_app_config():
    return MagicMock(
        total_budget=2000000,
        context_token_budget_ratio=0.25,
        domain_keywords={"General": ["general"], "Software Engineering": ["code"]},
        example_prompts={
            "Coding & Implementation": {
                "Implement Python API Endpoint": {
                    "prompt": "Implement a new FastAPI endpoint.",
                    "description": "Generate an API endpoint.",
                    "framework_hint": "Software Engineering",
                }
            }
        }
    )

def test_format_prompt_basic(prompt_manager):
    \"\"\"Test format_prompt with basic variable substitution.\"\"\"
    template = "Hello, {name}!"
    kwargs = {"name": "World"}
    result = prompt_manager.format_prompt(template, **kwargs) # Use prompt_manager instance
    assert result == "Hello, World!"

def test_format_prompt_with_codebase_context_self_analysis(prompt_manager):
    \"\"\"Test format_prompt with codebase context for self-analysis.\"\"\"
    template = "Analyze this: {issue}"
    codebase_context = {
        "file_structure": {
            "critical_files_preview": {
                "file1.py": "def func(): pass"
            }
        }
    }
    kwargs = {"issue": "bug"}
    result = prompt_manager.format_prompt(template, codebase_context=codebase_context, is_self_analysis=True, **kwargs) # Use prompt_manager instance
    assert "CODEBASE CONTEXT" in result
    assert "file1.py" in result
    assert "bug" in result

def test_format_prompt_missing_key(prompt_manager):
    \"\"\"Test format_prompt handles missing keys gracefully.\"\"\"
    template = "Hello, {name}!"
    kwargs = {"age": 30} # Missing 'name'
    result = prompt_manager.format_prompt(template, **kwargs) # Use prompt_manager instance
    assert "Missing key for prompt formatting" in result # Check for warning message
    assert "{name}" in result # The placeholder should remain if not formatted
""",
                        },
                    ],
                }
            )

        high_token_personas = (
            self.collected_metrics.get("performance_efficiency", {}) # Use self.collected_metrics
            .get("debate_efficiency_summary", {})
            .get("persona_token_breakdown", {})
        )
        high_token_consumers = {
            p: t for p, t in high_token_personas.items() if t > 2000
        }

        if high_token_consumers:
            suggestions.append(
                {
                    "AREA": "Efficiency",
                    "PROBLEM": f"High token consumption by personas: {', '.join(high_token_consumers.keys())}. This indicates potentially verbose or repetitive analysis patterns.",
                    "PROPOSED_SOLUTION": "Optimize prompts for high-token personas. Implement prompt truncation strategies where appropriate, focusing on summarizing or prioritizing key information. For 'Self_Improvement_Analyst', focus on direct actionable insights rather than exhaustive analysis. For technical personas, ensure they are provided with concise, targeted information relevant to their specific task.",
                    "EXPECTED_IMPACT": "Reduces overall token consumption, leading to lower operational costs and potentially faster response times. Improves the efficiency of the self-analysis process.",
                    "CODE_CHANGES_SUGGESTED": [],
                }
            )

        content_misalignment_warnings = self.collected_metrics.get("reasoning_quality", {}).get( # Use self.collected_metrics
            "content_misalignment_warnings", 0
        )
        if content_misalignment_warnings > 3:
            suggestions.append(
                {
                    "AREA": "Reasoning Quality",
                    "PROBLEM": f"Content misalignment warnings ({content_misalignment_warnings}) indicate potential issues in persona reasoning or prompt engineering.",
                    "PROPOSED_SOLUTION": "Refine prompts for clarity and specificity. Review persona logic for consistency and accuracy. Ensure personas stay focused on the core task and domain.",
                    "EXPECTED_IMPACT": "Enhances the quality and relevance of persona outputs, leading to more coherent and accurate final answers.",
                    "CODE_CHANGES_SUGGESTED": [],
                }
            )

        final_suggestions = suggestions[:3]

        logger.info(
            f"Generated {len(suggestions)} potential suggestions. Finalizing with top {len(final_suggestions)}."
        )

        return suggestions

    def analyze_codebase_structure(self) -> Dict[str, Any]:
        logger.info("Analyzing codebase structure.")
        return {"summary": "Codebase structure analysis is a placeholder."}

    def analyze_performance_bottlenecks(self) -> Dict[str, Any]:
        logger.info("Analyzing performance bottlenecks.")
        return {"summary": "Performance bottleneck analysis is a placeholder."}