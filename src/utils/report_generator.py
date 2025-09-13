# src/utils/report_generator.py

import datetime
import json
import re
from typing import Any, Dict, List, Optional
from pathlib import Path
import numpy as np
import numbers
import logging
from src.utils.json_utils import convert_to_json_friendly # NEW: Import the shared utility

logger = logging.getLogger(__name__)


def strip_ansi_codes(text: str) -> str:
    """Removes ANSI escape codes from text."""
    ansi_escape_re = re.compile(r"\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])")
    return ansi_escape_re.sub("", text)


def generate_markdown_report(
    user_prompt: str,
    final_answer: Any,
    intermediate_steps: Dict[str, Any],
    process_log_output: str,
    config_params: Dict[str, Any],
    persona_audit_log: List[Dict[str, Any]],
) -> str:
    """
    Generates a comprehensive Markdown report from the analysis results.
    """
    report_date = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    md_content = f"# Project Chimera Socratic Debate Report\n\n"
    md_content += f"**Date:** {report_date}\n"
    md_content += f"**Original Prompt:** {user_prompt}\n\n"
    md_content += "---\n\n"
    md_content += "## Configuration\n\n"
    md_content += f"*   **Model:** {config_params.get('model_name', 'N/A')}\n"
    md_content += f"*   **Max Total Tokens Budget:** {config_params.get('max_tokens_budget', 'N/A')}\n"
    md_content += f"*   **Intermediate Steps Shown in UI:** {'Yes' if config_params.get('show_intermediate_steps', False) else 'No'}\n"
    md_content += f"*   **Reasoning Framework:** {config_params.get('domain', 'N/A')}\n"
    md_content += "---\n\n"

    if persona_audit_log:
        md_content += "## Persona Configuration Audit Trail (Current Session)\n\n"
        md_content += "| Timestamp | Persona | Parameter | Old Value | New Value |\n"
        md_content += "|---|---|---|---|---|\n"

        escaped_newline_for_display = "\\n"

        for entry in persona_audit_log:
            old_val_str = str(entry.get("old_value", ""))
            new_val_str = str(entry.get("new_value", ""))
            old_val_display = (
                (old_val_str[:50] + "...") if len(old_val_str) > 50 else old_val_str
            )
            new_val_display = (
                (new_val_str[:50] + "...") if len(new_val_str) > 50 else new_val_str
            )

            old_val_display_escaped = old_val_display.replace(
                "\n", escaped_newline_for_display
            )
            new_val_display_escaped = new_val_display.replace(
                "\n", escaped_newline_for_display
            )
            md_content += f"| {entry.get('timestamp')} | {entry.get('persona')} | {entry.get('parameter')} | `{old_val_display_escaped}` | `{new_val_display_escaped}` |\n"
        md_content += "\n---\n\n"

    md_content += "## Process Log\n\n"
    md_content += "```text\n"
    md_content += strip_ansi_codes(process_log_output)
    md_content += "\n```\n\n"

    if config_params.get("show_intermediate_steps", True):
        md_content += "---\n\n"
        md_content += "## Intermediate Reasoning Steps\n\n"
        step_keys_to_process = sorted(
            [
                k
                for k in intermediate_steps.keys()
                if not k.endswith("_Tokens_Used")
                and not k.endswith("_Estimated_Cost_USD")
                and k != "Total_Tokens_Used"
                and k != "Total_Estimated_Cost_USD"
                and k != "debate_history"
                and not k.startswith("malformed_blocks")
            ],
            key=lambda x: (x.split("_")[0] if "_" in x else "", x),
        )

        for step_key in step_keys_to_process:
            display_name = (
                step_key.replace("_Output", "")
                .replace("_Critique", "")
                .replace("_Feedback", "")
                .replace("_", " ")
                .title()
            )
            content = intermediate_steps.get(step_key, "N/A")
            token_base_name = (
                step_key.replace("_Output", "")
                .replace("_Critique", "")
                .replace("_Feedback", "")
            )
            token_count_key = f"{token_base_name}_Tokens_Used"
            tokens_used = intermediate_steps.get(token_count_key, "N/A")

            md_content += f"### {display_name}\n\n"
            if isinstance(content, dict): # Check if it's a dict before dumping
                md_content += "```json\n"
                md_content += json.dumps(content, indent=2, default=convert_to_json_friendly)
                md_content += "\n```\n"
            else:
                # If content is not a dict, convert it to a serializable string for markdown display
                md_content += f"```markdown\n{convert_to_json_friendly(content)}\n```\n"
            md_content += f"**Tokens Used for this step:** {tokens_used}\n\n"
    md_content += "---\n\n"
    md_content += "## Final Synthesized Answer\n\n"

    if isinstance(final_answer, dict): # Check if it's a dict before dumping
        md_content += "```json\n"
        md_content += json.dumps(final_answer, indent=2, default=convert_to_json_friendly)
    else:
        # If final_answer is not a dict, convert it to a serializable string for markdown display
        md_content += f"{convert_to_json_friendly(final_answer)}\n\n"

    md_content += "---\n\n"
    md_content += "## Summary\n\n"
    md_content += f"**Total Tokens Consumed:** {intermediate_steps.get('Total_Tokens_Used', 0):,}\n"
    md_content += f"**Total Estimated Cost:** ${intermediate_steps.get('Total_Estimated_Cost_USD', 0.0):.4f}\n"
    return md_content