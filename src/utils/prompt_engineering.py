# src/utils/prompt_engineering.py
import streamlit as st  # Assuming this is used in the app context, but not strictly needed for the function itself
import json
import os
import io
import contextlib
import re
import datetime
import time
from typing import Dict, Any, List, Optional
import yaml
import logging
from rich.console import Console
from core import SocraticDebate  # Assuming SocraticDebate is available for context

# --- MODIFIED IMPORTS ---
# Import necessary classes for historical analysis
from src.self_improvement.metrics_collector import ImprovementMetricsCollector
# --- END MODIFIED IMPORTS ---

# Assuming other necessary imports like PersonaConfig, LLMOutput, etc., are present
# from src.models import ...
# from src.persona_manager import PersonaManager
# from src.exceptions import ...
# from src.constants import ...
# from src.logging_config import setup_structured_logging

# Setup logging if this module is run standalone or needs its own logger
logger = logging.getLogger(__name__)
if not logger.handlers:  # Basic setup if not already configured
    logging.basicConfig(level=logging.INFO)


def create_self_improvement_prompt(
    metrics: Dict[str, Any], previous_analyses: Optional[List[Dict]] = None
) -> str:
    """
    Enhanced prompt that guides more targeted, actionable self-analysis with historical context.
    This version is more concise and structured, reflecting token optimization goals.
    """
    # Get historical effectiveness data
    # Instantiate ImprovementMetricsCollector to access analysis methods
    # Note: This instantiation requires context that might not be available here if called in isolation.
    # In a real application, this would likely be passed down from a higher-level orchestrator
    # that has access to the necessary components (like LLMProvider, Tokenizer, etc.).
    # For this function's purpose, we'll assume a minimal instantiation is possible or mock it.

    # Placeholder/Mock objects for instantiation if not available in this scope
    mock_tokenizer = None  # Replace with actual tokenizer if available
    mock_llm_provider = None  # Replace with actual LLM provider if available
    mock_persona_manager = None  # Replace with actual PersonaManager if available
    mock_content_validator = None  # Replace with actual ContentValidator if available

    try:
        # Assuming ImprovementMetricsCollector can load data from disk without full context
        metrics_collector_instance = ImprovementMetricsCollector(
            initial_prompt="Historical effectiveness analysis",  # Dummy prompt
            debate_history=[],  # Dummy data
            intermediate_steps={},  # Dummy data
            codebase_context={},  # Dummy data
            tokenizer=mock_tokenizer,
            llm_provider=mock_llm_provider,
            persona_manager=mock_persona_manager,
            content_validator=mock_content_validator,
        )
        historical_analysis = (
            metrics_collector_instance.analyze_historical_effectiveness()
        )

    except Exception as e:
        logger.error(
            f"Could not perform historical analysis for prompt generation: {e}"
        )
        historical_analysis = {
            "success_rate": 0,
            "total_attempts": 0,
            "top_performing_areas": [],
            "common_failure_modes": [],
        }

    # Format historical data for the prompt
    historical_effectiveness_str = ""
    if historical_analysis.get("total_attempts", 0) > 0:
        success_rate = historical_analysis.get("success_rate", 0) * 100
        historical_effectiveness_str = (
            f"\n\n--- HISTORICAL IMPROVEMENT EFFECTIVENESS ---\n"
            f"- Total attempts: {historical_analysis['total_attempts']}\n"
            f"- Success rate: {success_rate:.1f}%\n"
        )

        top_areas = historical_analysis.get("top_performing_areas", [])
        if top_areas:
            top_areas_formatted = [
                f"{area['area']} ({area['success_rate'] * 100:.1f}% success)"
                for area in top_areas
            ]
            historical_effectiveness_str += (
                f"- Most successful areas: {', '.join(top_areas_formatted)}\n"
            )

        failure_modes = historical_analysis.get("common_failure_modes", [])
        if failure_modes:
            failure_modes_formatted = [
                f"{mode['metric']} (failed {mode['occurrences']} times)"
                for mode in failure_modes
            ]
            historical_effectiveness_str += (
                f"- Common failure patterns: {', '.join(failure_modes_formatted)}\n"
            )
        historical_effectiveness_str += "-------------------------------------------\n"

    # Refined guidance sections for conciseness
    security_guidance = (
        "SECURITY: Prioritize HIGH severity Bandit issues (SQLi, command injection, hardcoded secrets). "
        "Group similar issues. Provide specific examples (3-5) with `code_snippet` in `PROBLEM` field. "
        "Analyze Python security pitfalls (deserialization, subprocess.run shell=True, XXE). "
        "Evaluate CI/CD security and unpinned prod dependencies."
    )

    token_optimization_guidance = (
        "EFFICIENCY: Analyze persona token consumption and redundant patterns. "
        "Suggest prompt truncation strategies for high-token personas. "
        "Evaluate current token budget allocation effectiveness."
    )

    maintainability_and_testing_guidance = (
        "MAINTAINABILITY & TESTING: Prioritize testing core logic (SocraticDebate, LLM interaction, metrics). "
        "Focus on high bug density/complexity areas. Implement targeted smoke tests with example code. "
        "Evaluate PEP8 adherence and overall maintainability (docs, structure)."
    )

    self_reflection_guidance = (
        "SELF-REFLECTION: Evaluate past analyses effectiveness. "
        "Enhance framework (personas, routing, prompt engineering) for better recommendations. "
        "Identify metrics for self-improvement changes. "
        "Ensure 80/20 principle adherence in recommendations."
    )

    # Combine all sections into a concise prompt
    prompt_sections = [
        "You are Project Chimera's Self-Improvement Analyst. Critically analyze the provided metrics and identify the most impactful improvements (80/20 principle) across reasoning quality, robustness, efficiency, and maintainability.",
        security_guidance,
        token_optimization_guidance,
        maintainability_and_testing_guidance,
        self_reflection_guidance,
    ]

    # Add historical context if available
    if historical_effectiveness_str:
        prompt_sections.append(historical_effectiveness_str)

    # Add current metrics
    prompt_sections.append(
        f"\n\n--- CURRENT METRICS ---\n{json.dumps(metrics, indent=2)}\n-----------------------\n"
    )

    # Add previous analyses if provided (though historical_analysis is more direct)
    if previous_analyses:
        prompt_sections.append(
            f"\nPrevious analyses provided for context:\n{json.dumps(previous_analyses, indent=2)}\n"
        )

    # Final instructions for output format
    output_instructions = (
        "\n\nProvide your analysis with:\n"
        "1.  **Specific, prioritized recommendations** (top 3-5) following the 80/20 principle.\n"
        "2.  **Concrete code examples** (FILE_PATH, ACTION, FULL_CONTENT/DIFF_CONTENT/LINES).\n"
        "3.  **Expected impact metrics** for each change.\n"
        "4.  **Clear rationale** for each recommendation.\n"
        "5.  **Adherence to JSON Schema**: Strictly follow `SelfImprovementAnalysisOutputV1` schema, including `malformed_blocks`."
    )
    prompt_sections.append(output_instructions)

    final_prompt = "\n".join(prompt_sections)

    return final_prompt


# --- Placeholder for other functions if they exist in the original file ---
# def generate_socratic_question(topic): ...
# def refine_prompt(original_prompt, refinement): ...
# def SYSTEM_PROMPT_SECURITY_AUDITOR(): ...