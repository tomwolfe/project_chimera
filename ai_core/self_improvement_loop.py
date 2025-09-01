# ai_core/self_improvement_loop.py
import logging
from typing import Any, Dict, List # Added List
from datetime import datetime # Added for _create_file_backup
import shutil # Added for _create_file_backup
import os # Added for _create_file_backup, _run_targeted_tests, _get_relevant_test_files
import subprocess # Added for _run_targeted_tests
import re # Added for _get_relevant_test_files
import json # Added for _calculate_improvement_score, save_improvement_results

# Assuming ImprovementMetricsCollector and other necessary classes/functions are importable
# from src.self_improvement.metrics_collector import ImprovementMetricsCollector
# from src.utils.prompt_engineering import create_self_improvement_prompt # Not directly needed here, but for context
# from src.models import LLMOutput # Assuming this might be relevant for return types, though not explicitly in suggestions

# Mock classes from original file for context, but they will be replaced by actual logic
class MockModel:
    """A mock AI model for demonstration purposes."""
    def __init__(self):
_        self.params = {} # Placeholder for model parameters

    def train(self, data: Any, learning_rate: float):
        logger.debug(f"Mock model training with data: {data} and learning rate: {learning_rate}")
        pass

    def update(self, learning_rate: float, adaptability: float, robustness: float):
        logger.debug(f"Mock model updating with learning_rate={learning_rate}, adaptability={adaptability}, robustness={robustness}")
        pass

class MockLogger:
    """A mock logger for demonstration purposes."""
    def log_metrics(self, evaluation_results: Dict, adaptability_score: float, robustness_score: float):
        logger.info(f"Metrics logged: Evaluation={evaluation_results}, Adaptability={adaptability_score:.2f}, Robustness={robustness_score:.2f}")

# Placeholder functions for adaptability and robustness calculation
# In a real system, these would involve complex logic, potentially
# interacting with a dedicated testing harness or evaluation module.
def calculate_adaptability(model: Any, novel_data: Any) -> float:
    """
    Calculates a score indicating the model's adaptability to novel data.
    This is a placeholder.
    """
    logger.debug("Calculating adaptability score (placeholder).")
    return 0.75 # Placeholder value

def calculate_robustness(model: Any, adversarial_data: Any) -> float:
    """
    Calculates a score indicating the model's robustness to adversarial data.
    This is a placeholder.
    """
    logger.debug("Calculating robustness score (placeholder).")
    return 0.82 # Placeholder value


logger = logging.getLogger(__name__)

class SelfImprovementLoop:
    """
    Orchestrates Project Chimera's self-improvement loop, including evaluation,
    change application, and learning from results.
    """
    # FIX: Added necessary parameters to __init__ for ImprovementMetricsCollector
    def __init__(self, model: Any, training_data: Any, validation_data: Any,
                 novel_data: Any, adversarial_data: Any, learning_rate: float = 0.01,
                 initial_prompt: str = "", debate_history: List[Dict] = None,
                 intermediate_steps: Dict[str, Any] = None, codebase_context: Dict[str, str] = None,
                 tokenizer: Any = None, llm_provider: Any = None,
                 persona_manager: Any = None, content_validator: Any = None):
        """
        Initializes the self-improvement loop with model, data, and context for analysis.
        """
        self.model = model
        self.training_data = training_data
        self.validation_data = validation_data
        self.novel_data = novel_data
        self.adversarial_data = adversarial_data
        self.learning_rate = learning_rate
        self.logger = MockLogger() # Using a mock logger for this example, but ideally inject a real logger

        # Store context for metrics collection and learning
        self.initial_prompt = initial_prompt
        self.debate_history = debate_history if debate_history is not None else []
        self.intermediate_steps = intermediate_steps if intermediate_steps is not None else {}
        self.codebase_context = codebase_context if codebase_context is not None else {}
        self.tokenizer = tokenizer
        self.llm_provider = llm_provider
        self.persona_manager = persona_manager
        self.content_validator = content_validator
        
        # Store the project root for test execution context
        self.codebase_path = Path.cwd() # Assuming the loop is run from project root or context is provided

    def evaluate_performance(self, metrics_before: Dict) -> Dict[str, Any]:
        """
        Evaluates the model's performance with before/after comparison.
        Returns detailed metrics showing impact of changes.
        """
        logger.info("Evaluating model performance with before/after comparison.")
        
        # Collect current metrics
        metrics_after = self._collect_current_metrics()
        
        # Compare metrics
        performance_changes = {}
        for category, metrics in metrics_after.items():
            if category in metrics_before:
                for metric_name, value_after in metrics.items():
                    if metric_name in metrics_before[category]:
                        value_before = metrics_before[category][metric_name]
                        
                        # Ensure values are numeric for comparison
                        if isinstance(value_before, (int, float)) and isinstance(value_after, (int, float)):
                            change = value_after - value_before
                            # Handle division by zero for percent change
                            percent_change = (change / value_before * 100) if value_before != 0 else float('inf')
                            
                            performance_changes.setdefault(category, {})[metric_name] = {
                                "before": value_before,
                                "after": value_after,
                                "absolute_change": change,
                                "percent_change": percent_change
                            }
                        else:
                            # Handle non-numeric metrics or just note if changed
                            performance_changes.setdefault(category, {})[metric_name] = {
                                "changed": value_before != value_after,
                                "before": value_before,
                                "after": value_after
                            }
        
        # Determine if overall improvement occurred
        improvement_score = self._calculate_improvement_score(metrics_before, metrics_after)
        
        # Log results
        logger.info(f"Performance evaluation complete. Improvement score: {improvement_score:.2f}")
        
        return {
            "metrics_before": metrics_before,
            "metrics_after": metrics_after,
            "performance_changes": performance_changes,
            "improvement_score": improvement_score,
            "successful": improvement_score > 0 # Define success based on score
        }

    def _collect_current_metrics(self) -> Dict:
        """Collect all relevant metrics for current state."""
        from src.self_improvement.metrics_collector import ImprovementMetricsCollector
        
        # Ensure all required context is available
        if not all([self.initial_prompt, self.tokenizer, self.llm_provider, self.persona_manager, self.content_validator]):
            logger.error("Missing critical context for ImprovementMetricsCollector. Cannot collect metrics.")
            return {"error": "Missing context for metrics collection"}

        metrics_collector = ImprovementMetricsCollector(
            initial_prompt=self.initial_prompt,
            debate_history=self.debate_history,
            intermediate_steps=self.intermediate_steps,
            codebase_context=self.codebase_context,
            tokenizer=self.tokenizer,
            llm_provider=self.llm_provider,
            persona_manager=self.persona_manager,
            content_validator=self.content_validator
        )
        return metrics_collector.collect_all_metrics()

    def _calculate_improvement_score(self, metrics_before: Dict, metrics_after: Dict) -> float:
        """Calculate a weighted score representing overall improvement."""
        score = 0.0
        # Define weights for different categories. These should align with the metrics collected.
        weights = {
            "security": 0.25,
            "robustness": 0.20,
            "efficiency": 0.15,
            "maintainability": 0.20,
            "test_coverage": 0.10, # Assuming test_coverage metrics are collected
            "ci_cd": 0.10 # Assuming CI/CD metrics are collected
        }
        
        for category, weight in weights.items():
            if category in metrics_before and category in metrics_after:
                # For each category, calculate improvement (higher score = better)
                # Note: Some metrics might be "lower is better" (like errors), so we need to account for direction
                for metric_name, value_after in metrics_after[category].items():
                    if metric_name in metrics_before[category]:
                        value_before = metrics_before[category][metric_name]
                        
                        # Determine if higher value is better. This logic needs to be robust.
                        # For now, assume common metrics: lower is better for errors/violations, higher for performance/coverage.
                        higher_is_better = metric_name not in ["error_count", "violations", "failures", "complexity", "nesting_depth", "malformed_blocks_count"]
                        
                        if isinstance(value_before, (int, float)) and isinstance(value_after, (int, float)):
                            if value_before == 0:
                                # Avoid division by zero. If before is 0:
                                # If after is also 0, change is 0.
                                # If after is positive, it's an infinite improvement (or large positive change).
                                # If after is negative (e.g., error count), it's an infinite negative change.
                                if value_after == 0:
                                    change_ratio = 0.0
                                else:
                                    change_ratio = 1.0 if higher_is_better else -1.0 # Represent infinite change directionally
                            else:
                                change = value_after - value_before
                                change_ratio = change / value_before
                            
                            # Adjust direction if lower is better
                            if not higher_is_better:
                                change_ratio = -change_ratio # Invert the ratio for "lower is better" metrics
                                
                            score += change_ratio * weight
                        elif isinstance(value_before, dict) and isinstance(value_after, dict):
                            # Handle nested metrics if necessary, e.g., for complex objects
                            pass # Placeholder for more complex metric structures
        
        # Ensure score is within a reasonable range, e.g., -1 to 1 or 0 to 1
        # For simplicity, let's just return the raw score, which might be unbounded.
        # A more sophisticated approach would normalize this score.
        return score

    def apply_suggested_changes(self, suggestions: List[Dict]) -> bool:
        """
        Apply suggested code changes with safety mechanisms.
        Returns True if changes were successfully applied and validated.
        """
        logger.info(f"Attempting to apply {len(suggestions)} suggested changes.")
        
        all_successful = True
        
        for idx, suggestion in enumerate(suggestions):
            suggestion_area = suggestion.get("AREA", "Unknown area")
            logger.info(f"Processing suggestion #{idx+1}/{len(suggestions)}: {suggestion_area}")
            
            # Skip if no code changes suggested for this suggestion item
            if not suggestion.get("CODE_CHANGES_SUGGESTED"):
                logger.debug("No code changes to apply for this suggestion.")
                continue
                
            # Apply each code change with safety checks
            for code_change_idx, code_change in enumerate(suggestion["CODE_CHANGES_SUGGESTED"]):
                file_path = code_change.get("FILE_PATH")
                action = code_change.get("ACTION")
                content = code_change.get("FULL_CONTENT")
                lines_to_remove = code_change.get("LINES")
                
                if not file_path or not action:
                    logger.warning(f"Skipping invalid code change entry #{code_change_idx+1} in suggestion {idx+1}: missing FILE_PATH or ACTION.")
                    all_successful = False
                    continue
                
                backup_path = None # Initialize backup_path
                original_content = None # Initialize original_content

                try:
                    # Ensure file path is safe and within project bounds
                    # Assuming sanitize_and_validate_file_path is available in the scope or imported
                    safe_file_path = sanitize_and_validate_file_path(file_path) 
                    
                    # Read original content for validation and rollback
                    if action in ["MODIFY", "REMOVE"]:
                        if os.path.exists(safe_file_path):
                            with open(safe_file_path, 'r', encoding='utf-8') as f:
                                original_content = f.read()
                        else:
                            # If MODIFY/REMOVE is suggested for a non-existent file, it's an error
                            if action == "MODIFY":
                                raise FileNotFoundError(f"File not found for MODIFY action: {safe_file_path}")
                            # For REMOVE, if file doesn't exist, it's effectively removed, so we can skip.
                            elif action == "REMOVE":
                                logger.warning(f"File not found for REMOVE action: {safe_file_path}. Skipping.")
                                continue # Skip to next code change

                    # Create backup before applying change
                    backup_path = self._create_file_backup(safe_file_path)
                    
                    # Apply change based on action type
                    if action == "ADD":
                        # For ADD, ensure the directory exists
                        os.makedirs(os.path.dirname(safe_file_path), exist_ok=True)
                        with open(safe_file_path, 'w', encoding='utf-8') as f:
                            f.write(content if content is not None else "")
                        logger.info(f"Added file: {safe_file_path}")
                    elif action == "MODIFY":
                        if content is None and lines_to_remove is None:
                            logger.warning(f"MODIFY action for {safe_file_path} has no FULL_CONTENT or LINES. Skipping.")
                            continue
                        
                        # If DIFF_CONTENT is provided, apply it. Otherwise, use FULL_CONTENT.
                        # A proper diff application would be more complex. For simplicity,
                        # we'll assume FULL_CONTENT is the primary source for modification.
                        if content is not None:
                            with open(safe_file_path, 'w', encoding='utf-8') as f:
                                f.write(content)
                            logger.info(f"Modified file: {safe_file_path}")
                        else:
                            # If only lines to remove are provided for MODIFY, it's ambiguous.
                            # Log warning and skip if no content is provided.
                            logger.warning(f"MODIFY action for {safe_file_path} has no FULL_CONTENT. Skipping.")
                            continue
                            
                    elif action == "REMOVE":
                        os.remove(safe_file_path)
                        logger.info(f"Removed file: {safe_file_path}")
                    else:
                        logger.warning(f"Unknown action type: {action} for file {safe_file_path}")
                        continue
                    
                    # Run targeted tests
                    test_result = self._run_targeted_tests(safe_file_path)
                    
                    if not test_result["all_passed"]:
                        # Revert change if tests fail
                        self._restore_from_backup(safe_file_path, backup_path)
                        logger.error(f"Tests failed after applying change to {safe_file_path}. Reverted.")
                        all_successful = False
                    else:
                        logger.info(f"Successfully applied and validated change to {safe_file_path}")
                        # Optionally, remove the backup file after successful application and validation
                        # if os.path.exists(backup_path):
                        #     os.remove(backup_path)
                        
                except FileNotFoundError as e:
                    logger.error(f"Error applying code change: {e}")
                    if backup_path and os.path.exists(backup_path):
                        self._restore_from_backup(file_path, backup_path)
                    all_successful = False
                except Exception as e:
                    logger.exception(f"Error applying code change: {str(e)}") # Use logger.exception for full traceback
                    # Attempt to restore from backup if available
                    if backup_path and os.path.exists(backup_path):
                        self._restore_from_backup(file_path, backup_path)
                    all_successful = False
        
        return all_successful

    def _create_file_backup(self, file_path: str) -> str:
        """Create a timestamped backup of the file."""
        if not os.path.exists(file_path):
            # Return a placeholder path if the file doesn't exist (e.g., for ADD action)
            # This path won't be used for restoration if the file was never created.
            return f"{file_path}.bak.nonexistent.{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = f"{file_path}.bak.{timestamp}"
        try:
            shutil.copy2(file_path, backup_path)
            logger.info(f"Created backup for {file_path} at {backup_path}")
            return backup_path
        except Exception as e:
            logger.error(f"Failed to create backup for {file_path} at {backup_path}: {e}")
            raise # Re-raise the exception to halt the process if backup fails

    def _restore_from_backup(self, file_path: str, backup_path: str):
        """Restore file from backup."""
        if not backup_path or not os.path.exists(backup_path):
            logger.error(f"Backup file not found or backup_path is invalid: {backup_path}. Cannot restore {file_path}.")
            return

        try:
            shutil.copy2(backup_path, file_path)
            logger.info(f"Restored {file_path} from backup {backup_path}")
            # Optionally remove the backup after successful restore
            # if os.path.exists(backup_path):
            #     os.remove(backup_path)
        except Exception as e:
            logger.error(f"Failed to restore {file_path} from backup {backup_path}: {e}")

    def _run_targeted_tests(self, file_path: str) -> Dict:
        """Run tests relevant to the modified file."""
        logger.info(f"Determining and running tests relevant to: {file_path}")
        
        test_files = self._get_relevant_test_files(file_path)
        
        if not test_files:
            logger.info(f"No specific tests found for {file_path}. Falling back to running all tests.")
            # In a real implementation, you'd have a more sophisticated test selection strategy
            # For now, we'll run all tests in the 'tests/' directory.
            test_files = ["tests/"] # Assuming 'tests/' is the root test directory
        
        results = {
            "tests_run": 0,
            "tests_passed": 0,
            "all_passed": True,
            "details": []
        }
        
        for test_target in test_files:
            try:
                # In a real implementation, you'd use the actual test runner and parse its output robustly.
                # This example uses pytest.
                cmd = ["pytest", test_target, "-v", "--tb=no"] # -v for verbose, --tb=no to reduce traceback noise in output
                logger.info(f"Running command: {' '.join(cmd)}")
                
                process = subprocess.run(
                    cmd, 
                    capture_output=True, 
                    text=True, 
                    cwd=str(self.codebase_path), # Run from project root
                    timeout=300  # 5 minute timeout per test target
                )
                
                output = process.stdout
                stderr = process.stderr
                
                if process.returncode != 0 and not output: # If command failed and produced no stdout
                    logger.error(f"Test execution failed for {test_target}. Return code: {process.returncode}. Stderr: {stderr}")
                    results["all_passed"] = False
                    results["details"].append({
                        "test_target": test_target,
                        "status": "ERROR",
                        "message": f"Command failed: {stderr}"
                    })
                    continue # Move to next test target

                # Parse results (simplified for example)
                tests_run_match = re.search(r"collected (\d+) items", output)
                tests_passed_match = re.search(r"(\d+) passed", output)
                
                current_tests_run = int(tests_run_match.group(1)) if tests_run_match else 0
                current_tests_passed = int(tests_passed_match.group(1)) if tests_passed_match else 0
                
                results["tests_run"] += current_tests_run
                results["tests_passed"] += current_tests_passed
                
                test_result_detail = {
                    "test_target": test_target,
                    "tests_run": current_tests_run,
                    "tests_passed": current_tests_passed,
                    "status": "PASSED" if current_tests_run == current_tests_passed else "FAILED",
                    "output": output # Include full output for debugging
                }
                results["details"].append(test_result_detail)
                
                if current_tests_run != current_tests_passed:
                    results["all_passed"] = False
                    
            except FileNotFoundError:
                logger.error(f"Pytest command not found. Ensure pytest is installed and in PATH.")
                results["all_passed"] = False
                results["details"].append({"test_target": test_target, "status": "ERROR", "message": "Pytest command not found."})
            except subprocess.TimeoutExpired:
                logger.error(f"Test execution timed out for {test_target}.")
                results["all_passed"] = False
                results["details"].append({"test_target": test_target, "status": "ERROR", "message": "Test execution timed out."})
            except Exception as e:
                logger.exception(f"Unexpected error running tests for {test_target}: {str(e)}")
                results["all_passed"] = False
                results["details"].append({"test_target": test_target, "status": "ERROR", "message": f"Unexpected error: {str(e)}"})
        
        logger.info(f"Test run summary for {file_path}: Total run={results['tests_run']}, Passed={results['tests_passed']}, All Passed={results['all_passed']}")
        return results

    def _get_relevant_test_files(self, file_path: str) -> List[str]:
        """Determine which test files are relevant to the modified file."""
        logger.debug(f"Finding relevant tests for: {file_path}")
        
        base_name = os.path.basename(file_path)
        dir_name = os.path.dirname(file_path)
        
        module_name = os.path.splitext(base_name)[0]
        
        test_candidates = []
        
        # Standard test location: tests/test_module_name.py
        test_file_candidate = os.path.join("tests", f"test_{module_name}.py")
        if os.path.exists(test_file_candidate):
            test_candidates.append(test_file_candidate)
            logger.debug(f"Found test file candidate: {test_file_candidate}")
        
        # Alternative test location: tests/module_name/
        # Example: src/utils/path_utils.py -> tests/utils/
        relative_path_from_src = os.path.relpath(file_path, "src")
        if relative_path_from_src.startswith(".."): # If not under src/, use module name logic
            relative_path_from_src = module_name

        test_dir_candidate = os.path.join("tests", relative_path_from_src)
        if os.path.isdir(test_dir_candidate):
            test_candidates.append(test_dir_candidate)
            logger.debug(f"Found test directory candidate: {test_dir_candidate}")
        
        if not test_candidates:
            logger.debug(f"No specific tests found for {file_path}.")
        
        return test_candidates

# Example usage (if this file were run directly)
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    dummy_model = MockModel()
    dummy_training_data = {"data": "train"}
    dummy_validation_data = {"data": "validate"}
    dummy_novel_data = {"data": "novel"}
    dummy_adversarial_data = {"data": "adversarial"}

    mock_initial_prompt = "Analyze the performance of the self-improvement loop."
    mock_debate_history = [{"persona": "TestPersona", "output": {"metrics": {"accuracy": 0.8}}}]
    mock_intermediate_steps = {"Total_Tokens_Used": 1000, "improvement_score": 0.5}
    mock_codebase_context = {"src/ai_core/self_improvement_loop.py": "def evaluate_performance(self): pass"}
    mock_tokenizer = MagicMock()
    mock_llm_provider = MagicMock()
    mock_persona_manager = MagicMock()
    mock_content_validator = MagicMock()

    loop = SelfImprovementLoop(
        model=dummy_model,
        training_data=dummy_training_data,
        validation_data=dummy_validation_data,
        novel_data=dummy_novel_data,
        adversarial_data=dummy_adversarial_data,
        initial_prompt=mock_initial_prompt,
        debate_history=mock_debate_history,
        intermediate_steps=mock_intermediate_steps,
        codebase_context=mock_codebase_context,
        tokenizer=mock_tokenizer,
        llm_provider=mock_llm_provider,
        persona_manager=mock_persona_manager,
        content_validator=mock_content_validator
    )

    # Example of evaluating performance
    # metrics_before_change = loop._collect_current_metrics() # This would be called before a change
    # print("Metrics before change:", metrics_before_change)
    
    # Example of applying changes (requires suggestions)
    # suggestions_to_apply = [
    #     {
    #         "AREA": "Maintainability",
    #         "PROBLEM": "Long function",
    #         "PROPOSED_SOLUTION": "Split function",
    #         "EXPECTED_IMPACT": "Improved readability",
    #         "CODE_CHANGES_SUGGESTED": [
    #             {"FILE_PATH": "example_file.py", "ACTION": "MODIFY", "FULL_CONTENT": "def new_func(): pass"}
    #         ]
    #     }
    # ]
    # loop.apply_suggested_changes(suggestions_to_apply)