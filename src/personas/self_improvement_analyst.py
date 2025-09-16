import logging
from typing import Dict, Any, List, Optional
from datetime import datetime
import shutil
import os
import subprocess
import re
import json
from pathlib import Path

# Assuming FocusedMetricsCollector and other necessary classes/functions are importable

logger = logging.getLogger(__name__)


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


# Placeholder functions for adaptability and robustness calculation
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


class SelfImprovementAnalyst:
    """
    Analyzes metrics and suggests code improvements based on the 80/20 principle.
    Focuses on reasoning quality, robustness, efficiency, and maintainability.
    """

    def __init__(
        self,
        initial_prompt: str,
        metrics: Dict[str, Any],
        debate_history: List[Dict],
        intermediate_steps: Dict[str, Any],
        codebase_raw_file_contents: Dict[str, str],
        tokenizer: Any,
        llm_provider: Any,
        persona_manager: Any,
        content_validator: Any,
        metrics_collector: Any,
    ):
        """
        Initializes the analyst with collected metrics and context.
        """
        self.initial_prompt = initial_prompt
        self.metrics = metrics
        self.debate_history = debate_history
        self.intermediate_steps = intermediate_steps
        self.codebase_raw_file_contents = codebase_raw_file_contents
        self.tokenizer = tokenizer
        self.llm_provider = llm_provider
        self.persona_manager = persona_manager
        self.content_validator = content_validator
        self.metrics_collector = metrics_collector
        self.codebase_path = Path.cwd()

    def get_prompt(self, context: dict) -> str:
        """
        Generates the prompt for the Self-Improvement Analyst persona.
        This prompt is designed to elicit specific, actionable self-improvement suggestions
        based on the provided metrics and adhering to the 80/20 principle.
        """
        # --- MODIFIED PROMPT FOR CONCISENESS AND FOCUS ---
        # The prompt is refined to be more directive about the 80/20 principle,
        # actionable code changes, and specific focus areas.
        # It also includes a directive to summarize findings concisely and leverage historical data.

        # Ensure context is properly formatted, handling potential missing keys gracefully.
        metrics_context = context.get("metrics", "No metrics provided.")
        reasoning_quality_context = context.get("reasoning_quality", "N/A")
        historical_analysis_context = context.get("historical_analysis", "N/A")

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

**CRITICAL: Use the historical analysis to identify patterns of success or common failure modes. Prioritize suggestions that leverage past successes or directly address recurring issues.**

Summarize findings concisely.
"""
        return self_improvement_prompt

    def analyze(self) -> List[Dict[str, Any]]:
        """
        Performs the self-analysis and generates improvement suggestions.
        Focuses on the top 3 highest impact areas based on metrics, adhering to the Pareto principle.
        """
        logger.info("Performing self-analysis for Project Chimera.")

        suggestions = []

        # --- NEW: Incorporate Historical Analysis ---
        # Retrieve historical analysis data
        historical_data = self.metrics_collector.analyze_historical_effectiveness()
        top_performing_areas = {
            item["area"]: item["success_rate"]
            for item in historical_data.get("top_performing_areas", [])
        }
        common_failure_modes = {
            item["metric"]: item["occurrences"]
            for item in historical_data.get("common_failure_modes", [])
        }

        # Add a suggestion based on historical data if available
        if historical_data.get("total_attempts", 0) > 0:
            if (
                historical_data.get("success_rate", 0) < 0.5
                and historical_data.get("total_attempts", 0) > 5
            ):
                suggestions.append(
                    {
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
 
       ---
       **CRITICAL INSTRUCTION: ABSOLUTE ADHERENCE TO CONFLICT RESOLUTION** If the provided `Conflict Resolution Summary` explicitly states that specific code modifications cannot be provided due to lack of direct codebase access or other methodological limitations, you MUST **ABSOLUTELY AND WITHOUT EXCEPTION** adhere to that resolution. In such cases: - Your `IMPACTFUL_SUGGESTIONS` should contain **ONLY** suggestions focused on resolving the lack of codebase context (e.g., suggesting a `docs/project_chimera_context.md` file). - For any such suggestions, the `CODE_CHANGES_SUGGESTED` array MUST be EMPTY for items that would normally require direct codebase access. - If a conceptual change is needed, suggest an 'ADD' action to a new documentation file (e.g., `docs/security_guidance.md`) and put the conceptual content in `FULL_CONTENT`. - If the conflict resolution dictates no code changes, then `CODE_CHANGES_SUGGESTED` for *all* other suggestions MUST be an empty array `[]`. ---
""",
                            }
                        ],
                    }
                )
            elif top_performing_areas and historical_data.get("total_attempts", 0) > 5:
                # Suggest leveraging a top-performing area
                best_area = max(top_performing_areas, key=top_performing_areas.get)
                suggestions.append(
                    {
                        "AREA": "Reasoning Quality",
                        "PROBLEM": f"Historical analysis shows '{best_area}' is a top-performing area for self-improvement (success rate: {top_performing_areas[best_area]:.1%}).",
                        "PROPOSED_SOLUTION": f"Double down on strategies that have proven effective in '{best_area}'. Analyze successful past suggestions in this area to extract common patterns and apply them to new problems.",
                        "EXPECTED_IMPACT": "Increased efficiency and success rate of self-improvement by focusing on proven methods.",
                        "CODE_CHANGES_SUGGESTED": [],
                    }
                )
            elif common_failure_modes and historical_data.get("total_attempts", 0) > 5:
                # Suggest addressing a common failure mode
                most_common_failure = max(
                    common_failure_modes, key=common_failure_modes.get
                )
                suggestions.append(
                    {
                        "AREA": "Robustness",
                        "PROBLEM": f"Historical analysis identifies '{most_common_failure}' as a common failure mode ({common_failure_modes[most_common_failure]} occurrences).",
                        "PROPOSED_SOLUTION": f"Implement specific safeguards or prompt adjustments to mitigate '{most_common_failure}'. For example, if it's 'schema_validation_failures_count', refine JSON output instructions or use more robust parsing.",
                        "EXPECTED_IMPACT": "Reduced recurrence of known issues, improving the overall reliability of the self-improvement process.",
                        "CODE_CHANGES_SUGGESTED": [
                            {
                                "FILE_PATH": "personas.yaml",
                                "ACTION": "MODIFY",
                                "DIFF_CONTENT": """--- a/personas.yaml
+++ b/personas.yaml
@@ -200,7 +200,7 @@
       **CRITICAL JSON OUTPUT INSTRUCTIONS: ABSOLUTELY MUST BE FOLLOWED**
       **1. YOUR RESPONSE MUST BE A SINGLE, VALID JSON OBJECT. IT MUST START WITH '{' AND END WITH '}'. DO NOT RETURN A JSON ARRAY.**
       2. DO NOT USE NUMBERED ARRAY ELEMENTS (e.g., "0:{...}" is INVALID).
-      3. DO NOT INCLUDE ANY CONVERSATIONAL TEXT, MARKDOWN FENCES (```json), OR EXPLANATIONS OUTSIDE THE JSON OBJECT.
+      3. ABSOLUTELY NO CONVERSATIONAL TEXT, MARKDOWN FENCES (```json), OR EXPLANATIONS OUTSIDE THE JSON OBJECT.
       4. STRICTLY ADHERE TO THE PROVIDED JSON SCHEMA BELOW.**
       5. USE ONLY DOUBLE QUOTES for all keys and string values.
       6. ENSURE COMMAS separate all properties in objects and elements in arrays.
""",
                            }
                        ],
                    }
                )

        # --- START SUGGESTED CHANGES INTEGRATION ---
        # Validate and correct file paths against actual codebase
        validated_suggestions_temp = []
        for suggestion in suggestions:
            processed_suggestion = suggestion.copy()
            processed_code_changes = []
            for change in processed_suggestion.get("CODE_CHANGES_SUGGESTED", []):
                if change["ACTION"] in ["MODIFY", "REMOVE"]:
                    # Check if file exists in codebase context
                    if change["FILE_PATH"] not in self.codebase_raw_file_contents:
                        # Convert common path patterns to actual paths
                        corrected_path = self._validate_file_path(change["FILE_PATH"])

                        if (
                            corrected_path
                            and corrected_path in self.codebase_raw_file_contents
                        ):
                            change["FILE_PATH"] = corrected_path
                        else:
                            # File doesn't exist - convert to CREATE action
                            change["ACTION"] = "CREATE"
                            change["FULL_CONTENT"] = (
                                self._generate_default_file_content(change["FILE_PATH"])
                            )
                            change["DIFF_CONTENT"] = None
                            change["LINES"] = None
                            logger.warning(
                                f"File '{change['FILE_PATH']}' doesn't exist. Converting to CREATE action."
                            )
                processed_code_changes.append(change)

            processed_suggestion["CODE_CHANGES_SUGGESTED"] = processed_code_changes
            validated_suggestions_temp.append(processed_suggestion)
        # --- END SUGGESTED CHANGES INTEGRATION ---

        final_suggestions = validated_suggestions_temp[
            :3
        ]  # Apply slicing after validation

        logger.info(
            f"Generated {len(suggestions)} potential suggestions. Finalizing with top {len(final_suggestions)}."
        )

        return self.metrics_collector._process_suggestions_for_quality(
            final_suggestions
        )

    # --- START NEW HELPER METHODS ---
    def _validate_file_path(self, file_path: str) -> Optional[str]:
        """Validate and correct file path based on actual codebase structure."""
        normalized_path = file_path.strip()

        # Handle common path patterns
        if normalized_path.startswith("services/"):
            # Example: services/reasoning_engine.py -> src/core.py
            # Example: services/token_manager.py -> src/token_tracker.py
            if "reasoning_engine.py" in normalized_path:
                return "src/core.py"
            elif "token_manager.py" in normalized_path:
                return "src/token_tracker.py"
            else:  # General mapping for other files in 'services/'
                corrected_path = f"src/{normalized_path.split('/', 1)[1]}"
                if corrected_path in self.codebase_raw_file_contents:
                    return corrected_path
        elif normalized_path.startswith("api/"):
            # Example: api/routes.py -> app.py
            if "routes.py" in normalized_path:
                return "app.py"
            else:  # General mapping for other files in 'api/'
                # Assuming 'api/' maps to 'src/' for other files, adjust if needed
                corrected_path = f"src/{normalized_path.split('/', 1)[1]}"
                if corrected_path in self.codebase_raw_file_contents:
                    return corrected_path
        # Direct file name mappings
        elif normalized_path == "reasoning_engine.py":
            if "src/core.py" in self.codebase_raw_file_contents:
                return "src/core.py"
        elif normalized_path == "token_manager.py":
            if "src/token_tracker.py" in self.codebase_raw_file_contents:
                return "src/token_tracker.py"
        elif normalized_path == "routes.py":
            if "app.py" in self.codebase_raw_file_contents:
                return "app.py"

        # If no specific mapping, check if the path exists as is
        if normalized_path in self.codebase_raw_file_contents:
            return normalized_path

        return None

    def _generate_default_file_content(self, file_path: str) -> str:
        """Generates default content for a new file based on type."""
        if file_path.endswith(".py"):
            return f"# Automatically generated file: {file_path}\n\n"
        elif file_path.endswith(".json"):
            return "{}"
        elif file_path.endswith(".yaml") or file_path.endswith(".yml"):
            return "# Automatically generated configuration file\n\n"
        elif file_path.endswith(".md"):
            return f"# {file_path}\n\n"
        return ""

    # --- END NEW HELPER METHODS ---

    def analyze_codebase_structure(self) -> Dict[str, Any]:
        logger.info("Analyzing codebase structure.")
        return {"summary": "Codebase structure analysis is a placeholder."}

    def analyze_performance_bottlenecks(self) -> Dict[str, Any]:
        logger.info("Analyzing performance bottlenecks.")
        return {"summary": "Performance bottleneck analysis is a placeholder."}
