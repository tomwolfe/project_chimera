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
# from core import SocraticDebate  # Assuming SocraticDebate is available for context - removed as it's not needed here and causes circular dependency

# --- MODIFIED IMPORTS ---
# Import necessary classes for historical analysis
# from src.self_improvement.metrics_collector import ImprovementMetricsCollector # This import is problematic as it requires full context
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


def optimize_reasoning_prompt(original_prompt: str, reasoning_quality_score: float = None) -> str:
    """
    Optimizes prompts specifically for reasoning quality, focusing on conciseness and
    adherence to the 80/20 principle.

    Args:
        original_prompt: The original prompt to optimize
        reasoning_quality_score: Optional pre-calculated reasoning quality score

    Returns:
        Optimized prompt with improved reasoning focus
    """
    # If no score provided, create a basic assessment
    if reasoning_quality_score is None:
        indicators = {
            "contains_80_20_language": "80/20" in original_prompt or "Pareto" in original_prompt.lower(),
            "explicit_focus_areas": any(area in original_prompt.lower() for area in
                ["reasoning quality", "robustness", "efficiency", "maintainability"]),
            "token_usage_warning": "token usage" in original_prompt.lower() or "cost" in original_prompt.lower(),
            "structured_output_request": "JSON" in original_prompt or "schema" in original_prompt.lower()
        }
        reasoning_quality_score = (
            0.25 * indicators["contains_80_20_language"] +
            0.25 * indicators["explicit_focus_areas"] +
            0.25 * indicators["token_usage_warning"] +
            0.25 * indicators["structured_output_request"]
        )
    else:
        # Re-derive indicators if score is provided, for conditional appending
        indicators = {
            "contains_80_20_language": "80/20" in original_prompt or "Pareto" in original_prompt.lower(),
            "explicit_focus_areas": any(area in original_prompt.lower() for area in
                ["reasoning quality", "robustness", "efficiency", "maintainability"]),
            "token_usage_warning": "token usage" in original_prompt.lower() or "cost" in original_prompt.lower(),
            "structured_output_request": "JSON" in original_prompt or "schema" in original_prompt.lower()
        }

    # Create optimized prompt with reasoning quality enhancements
    optimized_prompt = original_prompt

    # Add 80/20 principle reminder if missing
    if not indicators.get("contains_80_20_language", False):
        optimized_prompt += "\n\nCRITICAL: Apply the 80/20 Pareto principle - focus ONLY on the top 20% of issues that will yield 80% of potential improvements."

    # Add explicit focus areas if missing
    if not indicators.get("explicit_focus_areas", False):
        optimized_prompt += "\n\nPRIORITIZE: reasoning quality, robustness, efficiency, and maintainability in that order."

    # Add token consciousness directive
    if not indicators.get("token_usage_warning", False):
        optimized_prompt += "\n\nIMPORTANT: Be concise. Prioritize high-impact insights over comprehensiveness. Target <2000 tokens for your response."

    # Add structured output requirement
    if not indicators.get("structured_output_request", False):
        optimized_prompt += "\n\nFORMAT: Your response MUST follow the SelfImprovementAnalysisOutputV1 JSON schema with clear rationale and actionable code modifications."

    return optimized_prompt


def create_reasoning_quality_metrics_prompt(metrics: Dict[str, Any]) -> str:
    """
    Creates a specialized prompt for analyzing and improving reasoning quality.

    Args:
        metrics: Current system metrics including token usage, debate quality, etc.

    Returns:
        A prompt specifically designed to analyze and improve reasoning quality
    """
    token_stats = metrics.get("performance_efficiency", {}).get("token_usage_stats", {})
    total_tokens = token_stats.get("total_tokens", 0)
    high_token_personas = []

    # Identify personas with high token usage
    if "persona_token_usage" in token_stats:
        for persona, tokens in token_stats["persona_token_usage"].items():
            if tokens > 2000:  # Threshold for "high" token usage
                high_token_personas.append(f"{persona} ({tokens} tokens)")

    # Construct the reasoning quality analysis prompt
    prompt = f"""Analyze Project Chimera's reasoning quality with specific focus on debate effectiveness and token efficiency.

CURRENT METRICS:
- Total tokens used: {total_tokens}
- High-token personas: {', '.join(high_token_personas) if high_token_personas else 'None'}
- Reasoning quality score: {metrics.get('reasoning_quality', {}).get('overall_score', 'N/A')}

ANALYSIS FOCUS:
1. Identify specific patterns causing excessive token usage in debate contributions
2. Evaluate the effectiveness of the Socratic debate process in generating high-quality insights
3. Assess whether the 80/20 principle is being properly applied in analysis outputs
4. Determine if conflict resolution is effectively synthesizing diverse perspectives

RECOMMENDATIONS MUST:
- Target concrete improvements to reasoning quality (not just general code improvements)
- Include specific persona prompt modifications
- Address debate structure and conflict resolution mechanisms
- Provide measurable outcomes for implemented changes

CRITICAL: Focus ONLY on the top 1-2 most impactful changes for reasoning quality (80/20 principle)."""

    return prompt

# --- Placeholder for other functions if they exist in the original file ---
# def generate_socratic_question(topic): ...
# def refine_prompt(original_prompt, refinement): ...
# def SYSTEM_PROMPT_SECURITY_AUDITOR(): ...