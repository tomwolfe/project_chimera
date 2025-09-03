# src/personas/self_improvement_analyst.py
import logging
from typing import Dict, Any, List  # Added List
from datetime import datetime  # Added for _create_file_backup
import shutil  # Added for _create_file_backup
import os  # Added for _create_file_backup, _run_targeted_tests, _get_relevant_test_files
import subprocess  # Added for _run_targeted_tests
import re  # Added for _get_relevant_test_files
import json  # Added for _calculate_improvement_score, save_improvement_results
from pathlib import Path  # Added for Path.cwd()

# Assuming FocusedMetricsCollector and other necessary classes/functions are importable
# from src.self_improvement.metrics_collector import FocusedMetricsCollector
# from src.utils.prompt_engineering import create_self_improvement_prompt # Not directly needed here, but for context
# from src.models import LLMOutput # Assuming this might be relevant for return types, though not explicitly in suggestions


# Mock classes from original file for context, but they will be replaced by actual logic
class MockModel:
    """A mock AI model for demonstration purposes."""

    def __init__(self):
        self.params = {}  # Placeholder for model parameters

    def train(self, data: Any, learning_rate: float):
        logger.debug(
            f"Mock model training with data: {data} and learning rate: {learning_rate}"
        )
        pass

    def update(self, learning_rate: float, adaptability: float, robustness: float):
        logger.debug(
            f"Mock model updating with learning_rate={learning_rate}, adaptability={adaptability}, robustness={robustness}"
        )
        pass


class MockLogger:
    """A mock logger for demonstration purposes."""

    def log_metrics(
        self,
        evaluation_results: Dict,
        adaptability_score: float,
        robustness_score: float,
    ):
        logger.info(
            f"Metrics logged: Evaluation={evaluation_results}, Adaptability={adaptability_score:.2f}, Robustness={robustness_score:.2f}"
        )


# Placeholder functions for adaptability and robustness calculation
# In a real system, these would involve complex logic, potentially
# interacting with a dedicated testing harness or evaluation module.
def calculate_adaptability(model: Any, novel_data: Any) -> float:
    """
    Calculates a score indicating the model's adaptability to novel data.
    This is a placeholder.
    """
    logger.debug("Calculating adaptability score (placeholder).")
    return 0.75  # Placeholder value


def calculate_robustness(model: Any, adversarial_data: Any) -> float:
    """
    Calculates a score indicating the model's robustness to adversarial data.
    This is a placeholder.
    """
    logger.debug("Calculating robustness score (placeholder).")
    return 0.82  # Placeholder value


logger = logging.getLogger(__name__)


class SelfImprovementAnalyst:
    """
    Analyzes metrics and suggests code improvements based on the 80/20 principle.
    Focuses on reasoning quality, robustness, efficiency, and maintainability.
    """

    def __init__(
        self,
        metrics: Dict[str, Any],
        debate_history: List[Dict],
        intermediate_steps: Dict[str, Any],
        codebase_context: Dict[str, str],
        tokenizer: Any,
        llm_provider: Any,
        persona_manager: Any,
        content_validator: Any,
        metrics_collector: Any, # NEW: Add metrics_collector to init
    ):
        """
        Initializes the analyst with collected metrics and context.
        """
        self.metrics = metrics
        self.debate_history = debate_history
        self.intermediate_steps = intermediate_steps
        self.codebase_context = codebase_context
        self.tokenizer = tokenizer
        self.llm_provider = llm_provider
        self.persona_manager = persona_manager
        self.content_validator = content_validator
        self.metrics_collector = metrics_collector # NEW: Store metrics_collector
        self.codebase_path = (
            Path.cwd()
        )  # Assuming the analyst operates from the project root

    def get_prompt(self, context: dict) -> str:
        """
        Generates the prompt for the Self-Improvement Analyst persona.
        This prompt is designed to elicit specific, actionable self-improvement suggestions
        based on the provided metrics and adhering to the 80/20 principle.
        """
        # --- MODIFIED PROMPT FOR CONCISENESS AND FOCUS ---
        # The prompt is refined to be more directive about the 80/20 principle,
        # actionable code changes, and specific focus areas.
        # It also includes a directive to summarize findings concisely.

        # Ensure context is properly formatted, handling potential missing keys gracefully.
        metrics_context = context.get("metrics", "No metrics provided.")
        reasoning_quality_context = context.get("reasoning_quality", "N/A")
        historical_analysis_context = context.get("historical_analysis", "N/A") # NEW: Add historical context

        # Construct the prompt using f-string for clarity
        self_improvement_prompt = f"""Analyze Project Chimera for high-impact self-improvement (80/20).
Prioritize: reasoning quality, robustness, efficiency, maintainability.
Provide clear rationale and actionable code modifications.

**CRITICAL: Focus on improving the AI's own capabilities (reasoning, robustness, efficiency) and the self-improvement process itself. Frame suggestions as experiments or methodological adjustments where appropriate. Code changes can include modifications to persona system prompts, prompt engineering logic, data processing scripts, or conceptual documentation outlining new AI strategies.**

SECURITY ANALYSIS:
- Prioritize HIGH severity Bandit issues (SQLi, command injection, hardcoded secrets)
- Group similar issues together rather than listing individually
- Provide specific examples of the MOST critical 3-5 vulnerabilities, **referencing the provided `code_snippet` for each issue directly within the `PROBLEM` field.**

TOKEN OPTIMIZATION:
- Analyze which personas consume disproportionate tokens
- Identify repetitive or redundant analysis patterns
- Suggest specific prompt truncation strategies for high-token personas

TESTING STRATEGY:
- Prioritize testing core logic (SocraticDebate, LLM interaction) before UI components
- Focus on areas with highest bug density per historical data
- Implement targeted smoke tests for critical paths first, **providing example test code in `CODE_CHANGES_SUGGESTED` (FULL_CONTENT for ADD actions).**

SELF-REFLECTION:
- What aspects of previous self-improvement analyses were most/least effective?
- How can the self-analysis framework be enhanced to produce better recommendations?
- What metrics would best measure the effectiveness of self-improvement changes?

---

Metrics:
{metrics_context}

Reasoning Quality Analysis:
{reasoning_quality_context}

Historical Self-Improvement Effectiveness:
{historical_analysis_context}

Summarize findings concisely.
"""
        return self_improvement_prompt
        # --- END MODIFIED PROMPT ---

    def analyze(self) -> List[Dict[str, Any]]:
        """
        Performs the self-analysis and generates improvement suggestions.
        Focuses on the top 3 highest impact areas based on metrics, adhering to the Pareto principle.
        """
        logger.info("Performing self-analysis for Project Chimera.")

        suggestions = []

        # --- MODIFICATION START ---
        # Retrieve historical analysis data
        historical_data = self.metrics_collector.analyze_historical_effectiveness()
        top_performing_areas = {item['area']: item['success_rate'] for item in historical_data.get('top_performing_areas', [])}
        common_failure_modes = {item['metric']: item['occurrences'] for item in historical_data.get('common_failure_modes', [])}

        # Add a suggestion based on historical data if available
        if historical_data.get("total_attempts", 0) > 0:
            if historical_data.get("success_rate", 0) < 0.5: # If overall success rate is low
                suggestions.append({
                    "AREA": "Reasoning Quality",
                    "PROBLEM": f"Overall self-improvement success rate is low ({historical_data['success_rate']:.1%}). This indicates a need to refine the self-improvement methodology or prompt engineering.",
                    "PROPOSED_SOLUTION": "Review the `Self_Improvement_Analyst`'s system prompt and the overall debate flow. Prioritize clarity in instructions and ensure the AI is focused on actionable, AI-centric improvements. Consider A/B testing different prompt versions.",
                    "EXPECTED_IMPACT": "Improved reliability and effectiveness of the self-improvement loop, leading to more successful code changes and AI enhancements.",
                    "CODE_CHANGES_SUGGESTED": [
                        {
                            "FILE_PATH": "personas.yaml",
                            "ACTION": "MODIFY",
                            "DIFF_CONTENT": """--- a/personas.yaml
+++ b/personas.yaml
@@ -100,7 +100,10 @@
       You are Project Chimera's Self-Improvement Analyst. Your core mission is to identify the most impactful improvements for Project Chimera, strictly adhering to the 80/20 Pareto principle.
       Prioritize enhancements in the following order: **reasoning quality, robustness, efficiency, and maintainability.**
       Your analysis must be concise, evidence-based, and directly actionable. For each suggestion, provide a clear rationale and specific, actionable code modifications.
-      Identify ONLY the top 1-3 most impactful changes.
+      Identify ONLY the top 1-3 most impactful changes.
+
+      **CRITICAL: Focus on improving the AI's own capabilities (reasoning, robustness, efficiency) and the self-improvement process itself. Frame suggestions as experiments or methodological adjustments where appropriate. Code changes can include modifications to persona system prompts, prompt engineering logic, data processing scripts, or conceptual documentation outlining new AI strategies.**
+
 
       ---
       **CRITICAL INSTRUCTION: ABSOLUTE ADHERENCE TO CONFLICT RESOLUTION** If the provided `Conflict Resolution Summary` explicitly states that specific code modifications cannot be provided due to lack of direct codebase access or other methodological limitations, you MUST **ABSOLUTELY AND WITHOUT EXCEPTION** adhere to that resolution. In such cases: - Your `IMPACTFUL_SUGGESTIONS` should contain **ONLY** suggestions focused on resolving the lack of codebase context (e.g., suggesting a `docs/project_chimera_context.md` file). - For any such suggestions, the `CODE_CHANGES_SUGGESTED` array MUST be EMPTY for items that would normally require direct codebase access. - If a conceptual change is needed, suggest an 'ADD' action to a new documentation file (e.g., `docs/security_guidance.md`) and put the conceptual content in `FULL_CONTENT`. - If the conflict resolution dictates no code changes, then `CODE_CHANGES_SUGGESTED` for *all* other suggestions MUST be an empty array `[]`. ---
"""
                        }
                    ]
                })
            elif top_performing_areas:
                # Suggest leveraging a top-performing area
                best_area = max(top_performing_areas, key=top_performing_areas.get)
                suggestions.append({
                    "AREA": "Reasoning Quality",
                    "PROBLEM": f"Historical analysis shows '{best_area}' is a top-performing area for self-improvement (success rate: {top_performing_areas[best_area]:.1%}).",
                    "PROPOSED_SOLUTION": f"Double down on strategies that have proven effective in '{best_area}'. Analyze successful past suggestions in this area to extract common patterns and apply them to new problems.",
                    "EXPECTED_IMPACT": "Increased efficiency and success rate of self-improvement by focusing on proven methods.",
                    "CODE_CHANGES_SUGGESTED": []
                })
            elif common_failure_modes:
                # Suggest addressing a common failure mode
                most_common_failure = max(common_failure_modes, key=common_failure_modes.get)
                suggestions.append({
                    "AREA": "Robustness",
                    "PROBLEM": f"Historical analysis identifies '{most_common_failure}' as a common failure mode ({common_failure_modes[most_common_failure]} occurrences).",
                    "PROPOSED_SOLUTION": f"Implement specific safeguards or prompt adjustments to mitigate '{most_common_failure}'. For example, if it's 'schema_validation_failures_count', refine JSON output instructions or use more robust parsing.",
                    "EXPECTED_IMPACT": "Reduced recurrence of known issues, improving the overall reliability of the self-improvement process.",
                    "CODE_CHANGES_SUGGESTED": []
                })
        # --- MODIFICATION END ---


        # --- NEW: Handle Conflict Resolution first if codebase access is denied ---
        conflict_resolution_summary = self.intermediate_steps.get(
            "Conflict_Resolution_Attempt", {}
        ).get("resolution_summary", {})
        if (
            conflict_resolution_summary
            and conflict_resolution_summary.get("conflict_resolved")
            and "cannot be fulfilled due to the absence of the codebase"
            in conflict_resolution_summary.get("RATIONALE", "")
        ):
            # This is the most impactful immediate action as per the conflict resolution
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
            # If codebase access is the primary blocker, other code changes are secondary or conceptual.
            # We return here to ensure this is the ONLY suggestion if the codebase is missing.
            return suggestions

        # --- Extract top Ruff and Bandit issues for snippets ---
        top_ruff_issues_snippets = []
        top_bandit_issues_snippets = []

        # Filter and collect snippets for Ruff issues
        ruff_detailed_issues = [
            issue
            for issue in self.metrics.get("code_quality", {}).get("detailed_issues", [])
            if issue.get("source") == "ruff_lint"
            or issue.get("source") == "ruff_format"
        ]
        for issue in ruff_detailed_issues[:3]:  # Take top 3
            snippet = issue.get("code_snippet")
            if snippet:
                top_ruff_issues_snippets.append(
                    f"  - File: `{issue.get('file', 'N/A')}` (Line: {issue.get('line', 'N/A')}): `{issue.get('code', 'N/A')}` - {issue.get('message', 'N/A')}\n```\n{snippet}\n```"
                )
            else:
                top_ruff_issues_snippets.append(
                    f"  - File: `{issue.get('file', 'N/A')}` (Line: {issue.get('line', 'N/A')}): `{issue.get('code', 'N/A')}` - {issue.get('message', 'N/A')}"
                )

        # Filter and collect snippets for Bandit issues
        bandit_detailed_issues = [
            issue
            for issue in self.metrics.get("code_quality", {}).get("detailed_issues", [])
            if issue.get("source") == "bandit"
        ]
        for issue in bandit_detailed_issues[:3]:  # Take top 3
            snippet = issue.get("code_snippet")
            if snippet:
                top_bandit_issues_snippets.append(
                    f"  - File: `{issue.get('file', 'N/A')}` (Line: {issue.get('line', 'N/A')}): `{issue.get('code', 'N/A')}` - {issue.get('message', 'N/A')}\n```\n{snippet}\n```"
                )
            else:
                top_bandit_issues_snippets.append(
                    f"  - File: `{issue.get('file', 'N/A')}` (Line: {issue.get('line', 'N/A')}): `{issue.get('code', 'N/A')}` - {issue.get('message', 'N/A')}"
                )
        # --- End snippet extraction ---

        # --- MODIFIED LOGIC FOR PARETO PRINCIPLE AND CLARITY ---
        # Focus on the top 3 highest impact areas based on metrics (Pareto principle).
        # Prioritize Security, Maintainability, and Robustness.

        # Maintainability (Linting Issues)
        ruff_issues_count = self.metrics.get("code_quality", {}).get(
            "ruff_issues_count", 0
        )
        if ruff_issues_count > 100:  # Threshold for significant linting issues
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

        # Security
        bandit_issues_count = self.metrics.get("security", {}).get(
            "bandit_issues_count", 0
        )
        pyproject_config_error = any(
            block.get("type") == "PYPROJECT_CONFIG_PARSE_ERROR"
            for block in self.metrics.get("configuration_analysis", {}).get(
                "malformed_blocks", []
            )
        )

        if (
            bandit_issues_count > 0 or pyproject_config_error
        ):  # Trigger if issues or config error
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
""",
                        },
                    ],
                }
            )

        # Maintainability (Testing)
        zero_test_coverage = (
            self.metrics.get("maintainability", {})
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
""",
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
""",
                        },
                    ],
                }
            )

        # Efficiency (Token Usage)
        high_token_personas = (
            self.metrics.get("performance_efficiency", {})
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
                    "CODE_CHANGES_SUGGESTED": [
                        # Example code changes are provided in the main analysis output, not here.
                        # This section would typically be populated by a more detailed analysis.
                    ],
                }
            )

        # Reasoning Quality (Content Misalignment)
        content_misalignment_warnings = self.metrics.get("reasoning_quality", {}).get(
            "content_misalignment_warnings", 0
        )
        if content_misalignment_warnings > 3:  # Threshold for multiple warnings
            suggestions.append(
                {
                    "AREA": "Reasoning Quality",
                    "PROBLEM": f"Content misalignment warnings ({content_misalignment_warnings}) indicate potential issues in persona reasoning or prompt engineering.",
                    "PROPOSED_SOLUTION": "Refine prompts for clarity and specificity. Review persona logic for consistency and accuracy. Ensure personas stay focused on the core task and domain.",
                    "EXPECTED_IMPACT": "Enhances the quality and relevance of persona outputs, leading to more coherent and accurate final answers.",
                    "CODE_CHANGES_SUGGESTED": [],
                }
            )

        # Apply Pareto Principle: Limit to top 3 suggestions
        final_suggestions = suggestions[:3]

        logger.info(
            f"Generated {len(suggestions)} potential suggestions. Finalizing with top {len(final_suggestions)}."
        )

        return final_suggestions

    # --- Placeholder methods for other potential analyses ---
    def analyze_codebase_structure(self) -> Dict[str, Any]:
        logger.info("Analyzing codebase structure.")
        return {"summary": "Codebase structure analysis is a placeholder."}

    def analyze_performance_bottlenecks(self) -> Dict[str, Any]:
        logger.info("Analyzing performance bottlenecks.")
        return {"summary": "Performance bottleneck analysis is a placeholder."}