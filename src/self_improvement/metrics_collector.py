# src/self_improvement/metrics_collector.py
import logging
import re
import json
from pathlib import Path
import subprocess
import os
import datetime
import shutil
from typing import Dict, Any, List, Tuple, Union, Optional
from collections import defaultdict

# Assuming necessary imports from other modules are available
# from src.utils.code_validator import ... # Potentially needed for detailed analysis
# from src.llm_provider import ... # Potentially needed for token counting if not passed via tokenizer

logger = logging.getLogger(__name__)

# --- NEW: FocusedMetricsCollector Class ---
class FocusedMetricsCollector:
    """Collects and prioritizes metrics with strict 80/20 focus."""
    
    CRITICAL_METRICS = {
        "token_efficiency": {
            "description": "Tokens per meaningful suggestion",
            "threshold": 2000,
            "priority": 1
        },
        "impact_potential": {
            "description": "Estimated impact of suggested changes (0-100)",
            "threshold": 40,
            "priority": 2
        },
        "fix_confidence": {
            "description": "Confidence in fix correctness (0-100)",
            "threshold": 70,
            "priority": 3
        }
    }

    def __init__(self, initial_prompt: str, debate_history: List[Dict], intermediate_steps: Dict[str, Any],
                 codebase_context: Dict[str, str], tokenizer: Any, llm_provider: Any, persona_manager: Any, content_validator: Any):
        """
        Initializes the FocusedMetricsCollector with context for metric collection.
        """
        self.initial_prompt = initial_prompt
        self.debate_history = debate_history
        self.intermediate_steps = intermediate_steps
        self.codebase_context = codebase_context
        self.tokenizer = tokenizer
        self.llm_provider = llm_provider
        self.persona_manager = persona_manager
        self.content_validator = content_validator
        self.metrics = {}
        self.critical_metric = None
        
        # Call the core metric collection upon initialization
        self._collect_core_metrics(tokenizer, llm_provider)

    def _collect_core_metrics(self, tokenizer, llm_provider):
        """Collect core metrics and identify the single most critical bottleneck."""
        # Token efficiency calculation
        # Ensure tokenizer and initial_prompt are available
        input_tokens = 0
        if tokenizer and self.initial_prompt:
            input_tokens = tokenizer.count_tokens(self.initial_prompt)
        
        output_tokens = 0
        if self.debate_history and len(self.debate_history) > 0:
            # Assuming the last turn's response contains the relevant output tokens
            # This might need adjustment based on how debate_history is structured
            last_turn = self.debate_history[-1]
            if isinstance(last_turn, dict):
                last_response = last_turn.get("output", "") # Use 'output' if it's the final synthesized response
                if isinstance(last_response, dict): # If output is a dict, try to get a relevant string
                    last_response = json.dumps(last_response) # Convert dict to string for token counting
                
                if last_response and tokenizer:
                    output_tokens = tokenizer.count_tokens(last_response)
            else:
                logger.warning(f"Unexpected format for debate_history item: {type(last_turn)}")

        # Count valid suggestions from the final analysis output
        suggestions_count = 0
        try:
            # Assuming the final analysis output is stored in intermediate_steps
            # and is accessible via a key like 'Final_Synthesis_Output' or similar.
            # If the structure is different, this part needs adjustment.
            final_analysis_output = self.intermediate_steps.get("Final_Synthesis_Output", {})
            if isinstance(final_analysis_output, dict):
                # For Self-Improvement Analysis, suggestions are in 'IMPACTFUL_SUGGESTIONS'
                suggestions_count = len(final_analysis_output.get("IMPACTFUL_SUGGESTIONS", []))
            elif isinstance(final_analysis_output, list): # Handle cases where output might be a list directly
                 suggestions_count = len(final_analysis_output)

        except Exception as e:
            logger.warning(f"Could not parse final analysis output for suggestion count: {e}")
            suggestions_count = 0 # Default to 0 if parsing fails
        
        # Calculate token efficiency: tokens per suggestion
        # Avoid division by zero if no suggestions are made
        if suggestions_count > 0:
            self.metrics["token_efficiency"] = (input_tokens + output_tokens) / suggestions_count
        else:
            # If no suggestions, use total tokens as a proxy for overall process efficiency
            self.metrics["token_efficiency"] = input_tokens + output_tokens
        
        # Identify the critical metric based on deviation from thresholds
        self._identify_critical_metric()
        
    def _identify_critical_metric(self):
        """Identify the single most critical metric that's furthest from its threshold."""
        critical_metric = None
        max_deviation = -float('inf') # Initialize with negative infinity to find the largest deviation
        
        for metric_name, config in self.CRITICAL_METRICS.items():
            value = self.metrics.get(metric_name, 0)
            threshold = config["threshold"]
            
            # Calculate deviation:
            # For token_efficiency, higher value is worse (further from ideal low value)
            # For impact_potential and fix_confidence, higher value is better (further from ideal low value)
            if metric_name == "token_efficiency":
                # Deviation is how much the value exceeds the threshold
                deviation = value - threshold
            else:
                # Deviation is how much the value is below the threshold
                deviation = threshold - value
            
            # We want the metric that is MOST problematic.
            # For token_efficiency, this means the largest positive deviation (value > threshold).
            # For impact_potential/fix_confidence, this means the largest positive deviation (threshold > value).
            # In both cases, we are looking for the largest positive 'deviation' value.
            if deviation > max_deviation:
                max_deviation = deviation
                critical_metric = metric_name
        
        self.critical_metric = critical_metric
        logger.debug(f"Identified critical metric: {self.critical_metric} with deviation {max_deviation}")
        
    def get_critical_metric_info(self):
        """Get information about the critical metric for prompt engineering."""
        if not self.critical_metric:
            return None
        
        config = self.CRITICAL_METRICS[self.critical_metric]
        value = self.metrics.get(self.critical_metric, 0)
        threshold = config["threshold"]
        
        # Determine status based on whether the metric is meeting its goal
        is_critical = False
        if self.critical_metric == "token_efficiency":
            # Token efficiency is critical if it's *above* the threshold
            is_critical = value > threshold
        else:
            # Impact potential and fix confidence are critical if they are *below* the threshold
            is_critical = value < threshold
            
        status = "CRITICAL" if is_critical else "OK"
        
        return {
            "name": self.critical_metric,
            "value": value,
            "threshold": threshold,
            "description": config["description"],
            "status": status,
        }

    # --- Placeholder methods called by other parts of the system ---
    # These methods would contain the actual logic for collecting all metrics,
    # recording auto-remediation results, and saving improvement history.
    # Their implementation details are not provided in the diff, so placeholders are used.

    def collect_all_metrics(self) -> Dict[str, Any]:
        """
        Collects all relevant metrics from the codebase and debate history.
        This is a placeholder; the actual implementation would involve
        calling various analysis tools (Ruff, Bandit, AST, etc.) and
        aggregating their results.
        """
        logger.info("Collecting all metrics (placeholder).")
        # In a real implementation, this would gather metrics like:
        # - Code quality (linting errors, complexity)
        # - Security vulnerabilities (Bandit findings)
        # - Test coverage
        # - Token usage per persona/turn
        # - Reasoning quality metrics (schema adherence, content alignment)
        # - Performance metrics
        # - Configuration analysis
        # - Deployment robustness
        
        # For now, return the core metrics collected during init and add placeholders
        return {
            **self.metrics, # Include critical metrics identified during init
            "code_quality": {"ruff_issues_count": 0, "bandit_issues_count": 0, "ast_issues_count": 0},
            "performance_efficiency": {"total_tokens": self.metrics.get("token_efficiency", 0)}, # Simplified
            "maintainability": {"test_coverage": 0.0},
            "reasoning_quality": {"schema_validation_failures": 0, "content_misalignment": 0},
            "configuration_analysis": {},
            "deployment_robustness": {},
            "historical_analysis": self.persona_manager.analyze_historical_effectiveness() if self.persona_manager else {}, # Example of using other components
        }

    def record_auto_remediation(self, results: Dict[str, Any]):
        """Records the outcome of auto-remediation attempts."""
        logger.info(f"Recording auto-remediation results: {results}")
        # This method would typically store the results in self.intermediate_steps
        # or a dedicated history log for tracking automated fixes.
        self.intermediate_steps["auto_remediation_results"] = results

    def save_improvement_results(self, suggestions: List[Dict], metrics_before: Dict, metrics_after: Dict, success: bool):
        """Saves the results of an improvement cycle for historical analysis."""
        logger.info(f"Saving improvement results: Success={success}, Suggestions={len(suggestions)}")
        # This method would typically append a record to a JSONL file or database
        # containing the suggestions, metrics before/after, success status, etc.
        # For demonstration, we'll just log the action.
        pass