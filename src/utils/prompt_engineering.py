# src/utils/prompt_engineering.py
import streamlit as st  # Assuming this is used in the app context, but not strictly needed for the function itself
import json
import os
import io
import contextlib
import re
import datetime
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple  # Added Tuple

# Assuming necessary models and constants are available via imports
# from src.models import PersonaConfig, LLMOutput, ...
# from src.persona_manager import PersonaManager
# from src.constants import SELF_ANALYSIS_KEYWORDS, NEGATION_PATTERNS, THRESHOLD
# from src.tokenizers.base import Tokenizer # If needed for token counting within prompt engineering

logger = logging.getLogger(__name__)

# --- Placeholder for PromptAnalyzer if it's used here ---
# If PromptAnalyzer is intended to be used within this file, ensure it's imported correctly.
# For the purpose of providing just this file's code, we'll assume it's handled elsewhere
# or that the logic below doesn't strictly depend on it being instantiated here.

# --- Placeholder for domain_keywords if needed ---
# If domain_keywords are needed here, they should be passed or loaded.
# For example:
# DOMAIN_KEYWORDS = { ... } # Load from config or pass as argument

# --- Placeholder for is_self_analysis_prompt if needed ---
# If is_self_analysis_prompt is used here, ensure it's imported from src.constants
# from src.constants import is_self_analysis_prompt


# --- MODIFICATION: Add format_prompt function ---
def format_prompt(
    template: str,
    codebase_context: Optional[Dict[str, Any]] = None,
    is_self_analysis: bool = False,
    **kwargs,
) -> str:
    """
    Format a prompt with variables, adding codebase context when relevant for self-analysis.

    Args:
        template: The base prompt template string.
        codebase_context: Dictionary containing codebase information (file structure, snippets).
        is_self_analysis: Boolean indicating if the current context is a self-analysis task.
        **kwargs: Additional variables to format the template with.

    Returns:
        The formatted prompt string.
    """
    formatted_prompt = template

    # Add codebase context to the prompt if it's a self-analysis task and context is available
    if is_self_analysis and codebase_context:
        try:
            file_structure = codebase_context.get("file_structure", {})
            critical_files_preview = file_structure.get("critical_files_preview", {})

            if file_structure or critical_files_preview:
                context_summary = "\n\nCODEBASE CONTEXT:\n"
                context_summary += (
                    f"Project scanned: {len(file_structure)} directories found.\n"
                )

                if critical_files_preview:
                    context_summary += "Preview of critical files analyzed:\n"
                    for filename, content in critical_files_preview.items():
                        context_summary += f"\n--- {filename} (first 50 lines) ---\n"
                        context_summary += (
                            content.strip()
                        )  # Use strip() to remove leading/trailing whitespace
                        context_summary += "\n--------------------------------\n"
                else:
                    context_summary += "No critical files preview available.\n"

                # Append the context summary to the prompt
                # Ensure it doesn't exceed token limits (though this function doesn't manage token limits directly)
                formatted_prompt += context_summary
                kwargs["codebase_context_summary"] = (
                    context_summary  # Add to kwargs if needed elsewhere
                )

        except Exception as e:
            logger.error(f"Error formatting prompt with codebase context: {str(e)}")
            # Optionally add an error message to the prompt or log it
            formatted_prompt += "\n\n[Error: Could not include codebase context.]"

    # Format the prompt with any additional keyword arguments
    try:
        # Ensure all kwargs are strings or JSON serializable if needed
        formatted_prompt = formatted_prompt.format(**kwargs)
    except KeyError as e:
        logger.warning(
            f"Missing key for prompt formatting: {e}. Prompt might be incomplete."
        )
    except Exception as e:
        logger.error(f"Error during final prompt formatting: {str(e)}")

    return formatted_prompt


# --- Placeholder for other prompt engineering functions ---
# Example: A function to dynamically select personas based on prompt analysis
# This logic is likely handled elsewhere (e.g., PersonaRouter), but could be centralized here.
def select_personas_based_on_prompt(
    prompt: str,
    domain: str,
    available_personas: Dict[str, Any],
    persona_sets: Dict[str, List[str]],
    prompt_analyzer: Any,  # Assuming PromptAnalyzer is available
) -> List[str]:
    """
    Selects an initial persona sequence based on prompt analysis, domain, and available personas.
    This is a simplified example; actual logic might be more complex.
    """
    if not prompt:
        return persona_sets.get("General", [])

    # Use PromptAnalyzer to determine domain and self-analysis likelihood
    # Note: This assumes PromptAnalyzer is instantiated and available.
    # If not, this logic would need to be adapted or moved.
    domain_recommendation = prompt_analyzer.recommend_domain_from_keywords(prompt)
    is_self_analysis = prompt_analyzer.is_self_analysis_prompt(prompt)

    selected_domain = domain
    if is_self_analysis:
        selected_domain = "Self-Improvement"
    elif domain_recommendation and domain_recommendation != domain:
        # If prompt analysis suggests a different domain, consider using it
        # For now, we'll stick to the explicitly selected domain unless it's self-analysis
        pass

    if selected_domain not in persona_sets:
        logger.warning(
            f"Domain '{selected_domain}' not found in persona sets. Falling back to 'General'."
        )
        selected_domain = "General"

    base_sequence = persona_sets.get(selected_domain, persona_sets.get("General", []))

    # Add specific personas based on keywords if not already included
    final_sequence = base_sequence.copy()
    prompt_lower = prompt.lower()

    if "security" in prompt_lower or "vulnerability" in prompt_lower:
        if (
            "Security_Auditor" not in final_sequence
            and "Security_Auditor" in available_personas
        ):
            final_sequence.insert(0, "Security_Auditor")  # Prioritize security

    if "test" in prompt_lower or "coverage" in prompt_lower or "bug" in prompt_lower:
        if (
            "Test_Engineer" not in final_sequence
            and "Test_Engineer" in available_personas
        ):
            # Insert before the final synthesizer
            synth_index = len(final_sequence)
            if "Impartial_Arbitrator" in final_sequence:
                synth_index = final_sequence.index("Impartial_Arbitrator")
            elif "General_Synthesizer" in final_sequence:
                synth_index = final_sequence.index("General_Synthesizer")
            final_sequence.insert(synth_index, "Test_Engineer")

    # Ensure uniqueness and order
    seen = set()
    unique_sequence = []
    for persona in final_sequence:
        if persona not in seen:
            unique_sequence.append(persona)
            seen.add(persona)

    return unique_sequence
