# src/utils/prompt_engineering.py
import logging
from typing import Any, Optional  # Added Tuple

# Assuming necessary models and constants are available via imports
# from src.models import PersonaConfig, LLMOutput, ...
# from src.persona_manager import PersonaManager
# from src.constants import SELF_ANALYSIS_KEYWORDS, NEGATION_PATTERNS, THRESHOLD
# from src.tokenizers.base import Tokenizer # If needed for token counting within prompt engineering

logger = logging.getLogger(__name__)


# --- MODIFICATION: Add format_prompt function ---
def format_prompt(
    template: str,
    codebase_context: Optional[dict[str, Any]] = None,
    is_self_analysis: bool = False,
    **kwargs,
) -> str:
    """Format a prompt with variables, adding codebase context when relevant for self-analysis.

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
                        context_summary += content.strip()
                        context_summary += "\n--------------------------------\n"
                else:
                    context_summary += "No critical files preview available.\n"

                formatted_prompt += context_summary
                kwargs["codebase_context_summary"] = context_summary

        except Exception as e:
            logger.error(f"Error formatting prompt with codebase context: {str(e)}")
            formatted_prompt += "\n\n[Error: Could not include codebase context.]"

    try:
        formatted_prompt = formatted_prompt.format(**kwargs)
    except KeyError as e:
        logger.warning(
            f"Missing key for prompt formatting: {e}. Prompt might be incomplete."
        )
    except Exception as e:
        logger.error(f"Error during final prompt formatting: {str(e)}")

    return formatted_prompt


# REMOVED: select_personas_based_on_prompt function as it is not used.
# def select_personas_based_on_prompt(...):
#     """
#     Selects an initial persona sequence based on prompt analysis, domain, and available personas.
#     """
#     ...

# REMOVED: create_persona_prompt and create_task_prompt as they are not used.
# def create_persona_prompt(persona_details: Dict[str, Any]) -> str:
#     """Creates a detailed prompt for a persona."""
#     ...
# def create_task_prompt(task_description: str, context: str = "", instructions: str = "") -> str:
#     """Creates a task-specific prompt."""
#     ...
