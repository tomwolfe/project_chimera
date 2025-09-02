# ai_core/self_improvement_loop.py
import logging
from typing import Any, Dict, List  # Added List
from datetime import datetime  # Added for _create_file_backup
import shutil  # Added for _create_file_backup
import os  # Added for _create_file_backup, _run_targeted_tests, _get_relevant_test_files
import subprocess  # Added for _run_targeted_tests
import re  # Added for _get_relevant_test_files
import json  # Added for _calculate_improvement_score, save_improvement_results

# Assuming ImprovementMetricsCollector and other necessary classes/functions are importable
# from src.self_improvement.metrics_collector import ImprovementMetricsCollector
# from src.utils.prompt_engineering import create_self_improvement_prompt # Not directly needed here, but for context
# from src.models import LLMOutput # Assuming this might be relevant for return types, though not explicitly in suggestions


# Mock classes from original file for context, but they will be replaced by actual logic
class MockAIModel:
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


class SelfImprovementLoop:
    """
    Orchestrates Project Chimera's self-improvement loop, including evaluation,
    change application, and learning from results.
    """

    # FIX: Added necessary parameters to __init__ for ImprovementMetricsCollector
    def __init__(
        self,
        model: Any,
        training_data: Any,
        validation_data: Any,
        novel_data: Any,
        adversarial_data: Any,
        learning_rate: float = 0.01,
        initial_prompt: str = "",
        debate_history: List[Dict] = None,
        intermediate_steps: Dict[str, Any] = None,
        codebase_context: Dict[str, str] = None,
        tokenizer: Any = None,
        llm_provider: Any = None,
        persona_manager: Any = None,
        content_validator: Any = None,
    ):
        """
        Initializes the self-improvement loop with model, data, and context for analysis.
        """
        self.model = model
        self.training_data = training_data
        self.validation_data = validation_data
        self.novel_data = novel_data
        self.adversarial_data = adversarial_data
        self.learning_rate = learning_rate
        self.logger = (
            MockLogger()
        )  # Using a mock logger for this example, but ideally inject a real logger

        # Store context for metrics collection and learning
        self.initial_prompt = initial_prompt
        self.debate_history = debate_history if debate_history is not None else []
        self.intermediate_steps = (
            intermediate_steps if intermediate_steps is not None else {}
        )
        self.codebase_context = codebase_context if codebase_context is not None else {}
        self.tokenizer = tokenizer
        self.llm_provider = llm_provider
        self.persona_manager = persona_manager
        self.content_validator = content_validator

        # Store the project root for test execution context
        self.codebase_path = (
            Path.cwd()
        )  # Assuming the loop is run from project root or context is provided

    def _run_targeted_tests(self, repo_path: Path, command: List[str]) -> Tuple[str, str]:
        """
        Runs a specific test command within the repository path.
        This is a placeholder for actual test execution logic.
        """
        try:
            # Use a more secure way to execute commands if possible, e.g., passing args as a list
            # and ensuring shell=False if not strictly necessary.
            # For now, assuming commands are trusted or sanitized elsewhere.
            result = subprocess.run(command, shell=False, capture_output=True, text=True, check=True, cwd=repo_path)

            return result.stdout, result.stderr
        except subprocess.CalledProcessError as e:
            logging.error(f"Error running command: {e}")
            return "", e.stderr
        except FileNotFoundError:
            logging.error(f"Command not found: {command[0]}")
            return "", f"Command not found: {command[0]}"
        except Exception as e:
            logging.error(f"An unexpected error occurred: {e}")
            return "", str(e)

    def _create_file_backup(self, file_path: Path) -> Optional[Path]:
        """Creates a timestamped backup of a file."""
        if not file_path.exists():
            return None
        backup_dir = file_path.parent / ".chimera_backups"
        backup_dir.mkdir(exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        backup_path = backup_dir / f"{file_path.name}.{timestamp}.bak"
        shutil.copy(file_path, backup_path)
        logger.info(f"Created backup of {file_path} at {backup_path}")
        return backup_path

    def _apply_code_change(self, change: Dict[str, Any]):
        """Applies a single code change (ADD, MODIFY, REMOVE)."""
        file_path = self.codebase_path / change["FILE_PATH"]
        action = change["ACTION"]

        if action == "ADD":
            file_path.parent.mkdir(parents=True, exist_ok=True)
            with open(file_path, "w") as f:
                f.write(change["FULL_CONTENT"])
            logger.info(f"Added file: {file_path}")
        elif action == "MODIFY":
            if file_path.exists():
                self._create_file_backup(file_path)
                if change.get("FULL_CONTENT"):
                    with open(file_path, "w") as f:
                        f.write(change["FULL_CONTENT"])
                    logger.info(f"Modified file: {file_path} with FULL_CONTENT.")
                elif change.get("DIFF_CONTENT"):
                    # Apply diff content (requires a diff utility or manual parsing)
                    # This is a simplified placeholder; real diff application is complex.
                    original_content = file_path.read_text()
                    patched_content = self._apply_unified_diff(original_content, change["DIFF_CONTENT"])
                    with open(file_path, "w") as f:
                        f.write(patched_content)
                    logger.info(f"Modified file: {file_path} with DIFF_CONTENT.")
            else:
                logger.warning(f"Attempted to modify non-existent file: {file_path}")
        elif action == "REMOVE":
            if file_path.exists():
                self._create_file_backup(file_path)
                file_path.unlink()
                logger.info(f"Removed file: {file_path}")
            else:
                logger.warning(f"Attempted to remove non-existent file: {file_path}")

    def _apply_unified_diff(self, original_content: str, diff_content: str) -> str:
        """
        Applies a unified diff to the original content.
        This is a simplified implementation and might not handle all diff complexities.
        For production, consider a dedicated patch library.
        """
        original_lines = original_content.splitlines(keepends=True)
        diff_lines = diff_content.splitlines(keepends=True)
        
        patched_lines = []
        original_idx = 0
        diff_idx = 0

        while diff_idx < len(diff_lines):
            line = diff_lines[diff_idx]
            if line.startswith('---') or line.startswith('+++') or line.startswith('@@'):
                # Skip diff headers
                diff_idx += 1
                continue
            elif line.startswith('-'):
                # Line removed, skip in original
                original_idx += 1
            elif line.startswith('+'):
                # Line added
                patched_lines.append(line[1:])
            else:
                # Context line or unchanged line
                if original_idx < len(original_lines) and original_lines[original_idx].strip() == line.strip():
                    patched_lines.append(original_lines[original_idx])
                    original_idx += 1
                else:
                    # If context line doesn't match, it's a more complex diff or an error
                    # For simplicity, we'll just add the diff line as is, but this is risky
                    patched_lines.append(line[1:]) # Add the line from diff, removing the space
            diff_idx += 1
        
        # Add any remaining lines from the original if the diff ended prematurely
        while original_idx < len(original_lines):
            patched_lines.append(original_lines[original_idx])
            original_idx += 1

        return "".join(patched_lines)

    def _get_relevant_test_files(self, changed_files: List[str]) -> List[Path]:
        """
        Identifies relevant test files for a given set of changed source files.
        This is a heuristic and might need refinement.
        """
        relevant_tests = set()
        for changed_file in changed_files:
            path_obj = Path(changed_file)
            if path_obj.suffix == ".py":
                # Look for tests/test_module.py for src/module.py
                test_file_name = f"test_{path_obj.stem}.py"
                potential_test_path = self.codebase_path / "tests" / test_file_name
                if potential_test_path.exists():
                    relevant_tests.add(potential_test_path)
                
                # Also check for integration tests if applicable
                potential_integration_test_path = self.codebase_path / "tests" / "integration" / test_file_name
                if potential_integration_test_path.exists():
                    relevant_tests.add(potential_integration_test_path)

        return list(relevant_tests)

    def _calculate_improvement_score(self, metrics_before: Dict, metrics_after: Dict) -> float:
        """
        Calculates an improvement score based on changes in key metrics.
        This is a simplified scoring mechanism.
        """
        score = 0.0
        
        # Example: Reward for reduced security issues
        bandit_before = metrics_before.get("security", {}).get("bandit_issues_count", 0)
        bandit_after = metrics_after.get("security", {}).get("bandit_issues_count", 0)
        if bandit_before > bandit_after:
            score += (bandit_before - bandit_after) * 0.1 # 0.1 point per issue reduced

        # Example: Reward for improved test coverage (placeholder)
        coverage_before = metrics_before.get("maintainability", {}).get("test_coverage_summary", {}).get("overall_coverage_percentage", 0)
        coverage_after = metrics_after.get("maintainability", {}).get("test_coverage_summary", {}).get("overall_coverage_percentage", 0)
        if coverage_after > coverage_before:
            score += (coverage_after - coverage_before) * 0.5 # 0.5 point per % coverage increase

        # Example: Reward for reduced token usage (efficiency)
        tokens_before = metrics_before.get("performance_efficiency", {}).get("token_usage_stats", {}).get("total_tokens", 0)
        tokens_after = metrics_after.get("performance_efficiency", {}).get("token_usage_stats", {}).get("total_tokens", 0)
        if tokens_before > tokens_after:
            score += (tokens_before - tokens_after) * 0.00001 # Small reward for token reduction

        return score

    def run_self_improvement(self, analysis_output: Dict[str, Any]) -> Dict[str, Any]:
        """
        Orchestrates the self-improvement process: applies changes, runs tests,
        and evaluates the impact.
        """
        logger.info("Starting self-improvement application phase.")
        
        metrics_collector = ImprovementMetricsCollector(
            initial_prompt=self.initial_prompt,
            debate_history=self.debate_history,
            intermediate_steps=self.intermediate_steps,
            codebase_context=self.codebase_context,
            tokenizer=self.tokenizer,
            llm_provider=self.llm_provider,
            persona_manager=self.persona_manager,
            content_validator=self.content_validator,
        )
        metrics_before = metrics_collector.collect_all_metrics()

        successful_changes = []
        failed_changes = []
        changed_files = []

        for suggestion in analysis_output.get("IMPACTFUL_SUGGESTIONS", []):
            for change in suggestion.get("CODE_CHANGES_SUGGESTED", []):
                try:
                    self._apply_code_change(change)
                    successful_changes.append(change)
                    changed_files.append(change["FILE_PATH"])
                except Exception as e:
                    logger.error(f"Failed to apply change {change}: {e}")
                    failed_changes.append({"change": change, "error": str(e)})

        # Run targeted tests
        test_results = {}
        if changed_files:
            relevant_test_files = self._get_relevant_test_files(changed_files)
            if relevant_test_files:
                for test_file in relevant_test_files:
                    test_command = [sys.executable, "-m", "pytest", str(test_file)]
                    stdout, stderr = self._run_targeted_tests(self.codebase_path, test_command)
                    test_results[str(test_file)] = {"stdout": stdout, "stderr": stderr}
                    if "failed" in stdout.lower() or stderr:
                        logger.warning(f"Tests failed for {test_file}: {stderr or stdout}")
                        # Rollback changes if tests fail (simplified)
                        # For a real system, this would involve more robust rollback logic
                        # For now, we'll just log the failure.
            else:
                logger.info("No relevant test files found for changed code.")
        else:
            logger.info("No code changes applied, skipping test execution.")

        metrics_after = metrics_collector.collect_all_metrics()
        improvement_score = self._calculate_improvement_score(metrics_before, metrics_after)

        self.intermediate_steps["improvement_score"] = improvement_score
        self.intermediate_steps["metrics_before_improvement"] = metrics_before
        self.intermediate_steps["metrics_after_improvement"] = metrics_after
        self.intermediate_steps["applied_changes"] = successful_changes
        self.intermediate_steps["failed_changes"] = failed_changes
        self.intermediate_steps["test_results_after_changes"] = test_results

        # Save historical results
        metrics_collector.save_improvement_results(
            analysis_output.get("IMPACTFUL_SUGGESTIONS", []),
            metrics_before,
            metrics_after,
            success=not bool(failed_changes) and not any("failed" in res.get("stdout", "").lower() for res in test_results.values())
        )

        logger.info(f"Self-improvement application phase completed. Score: {improvement_score:.2f}")
        return {"status": "completed", "improvement_score": improvement_score, "test_results": test_results}