# src/utils/output_formatter.py
import json  # Used for json.dumps
import logging  # Used for logger
from datetime import datetime  # Used for datetime.now
from typing import Dict, Any, List, Optional  # Used for type hints
from pathlib import Path  # Used for Path(change.get('FILE_PATH', 'N/A')).name

from src.models import (  # Used for type hints in format_findings_list, format_complete_report
    CodeChange,
    SelfImprovementFinding,
    QuantitativeImpactMetrics,
    SelfImprovementAnalysisOutput,
)

logger = logging.getLogger(__name__)


class OutputFormatter:
    """Utility class for formatting analysis results into various outputs (e.g., Markdown)."""

    @staticmethod
    def format_suggestion(suggestion: Dict[str, Any]) -> str:
        """Formats a single self-improvement suggestion."""
        markdown = f"### AREA: {suggestion.get('AREA', 'N/A')}\n\n"
        markdown += f"**Problem:** {suggestion.get('PROBLEM', 'N/A')}\n\n"
        markdown += (
            f"**Proposed Solution:** {suggestion.get('PROPOSED_SOLUTION', 'N/A')}\n\n"
        )
        markdown += (
            f"**Expected Impact:** {suggestion.get('EXPECTED_IMPACT', 'N/A')}\n\n"
        )

        code_changes = suggestion.get("CODE_CHANGES_SUGGESTED", [])
        if code_changes:
            markdown += "**Suggested Code Changes:**\n"
            for change in code_changes:
                action = change.get("ACTION", "N/A")
                file_path = change.get("FILE_PATH", "N/A")
                markdown += f"- **{action}:** `{file_path}`\n"
                if action == "ADD":
                    content = change.get("FULL_CONTENT", "")
                    markdown += f"  ```python\n  {content[:100]}...\n  ```\n"
                elif action == "MODIFY":
                    diff = change.get("DIFF_CONTENT", "")
                    if diff:
                        markdown += f"  ```diff\n  {diff[:200]}...\n  ```\n"
                    else:
                        content = change.get("FULL_CONTENT", "")
                        markdown += f"  ```python\n  {content[:100]}...\n  ```\n"
                elif action == "REMOVE":
                    lines = change.get("LINES", [])
                    markdown += f"  Lines to remove: {len(lines)}\n"
        else:
            markdown += "**Suggested Code Changes:** None specified.\n"

        return markdown

    @staticmethod
    def format_findings_list(findings: List[SelfImprovementFinding]) -> str:
        """Formats a list of findings into Markdown."""
        markdown = "## Detailed Analysis Findings\n\n"
        if not findings:
            return markdown + "No specific findings to report.\n"

        # Sort findings by priority score (descending)
        sorted_findings = sorted(findings, key=lambda x: x.priority_score, reverse=True)

        for i, finding in enumerate(sorted_findings):
            markdown += f"### #{i + 1} Priority ({finding.priority_score:.2f}): {finding.area}\n\n"
            markdown += f"**Problem:** {finding.problem}\n\n"
            markdown += f"**Solution:** {finding.solution}\n\n"
            markdown += f"**Impact:** {finding.impact}\n\n"
            if finding.metrics:
                markdown += "**Quantitative Impact:**\n"
                markdown += (
                    f"- Estimated Effort: {finding.metrics.estimated_effort}/10\n"
                )
                markdown += f"- Expected Quality Improvement: {finding.metrics.expected_quality_improvement * 100:.1f}%\n"
                if finding.metrics.token_savings_percent is not None:
                    markdown += f"- Token Savings: {finding.metrics.token_savings_percent * 100:.1f}%\n"
                markdown += "\n"
            if finding.code_changes:
                markdown += "**Suggested Code Changes:**\n"
                for change in finding.code_changes:
                    markdown += f"- **{change.action}:** `{change.file_path}`\n"
                    if change.diff_content:
                        markdown += (
                            f"  ```diff\n  {change.diff_content[:200]}...\n  ```\n"
                        )
                    elif change.full_content:
                        markdown += (
                            f"  ```python\n  {change.full_content[:100]}...\n  ```\n"
                        )
                    elif change.lines:
                        markdown += f"  Lines to remove: {len(change.lines)}\n"
            markdown += "---\n\n"

        return markdown

    @staticmethod
    def format_pareto_prioritized_findings(
        findings: List[SelfImprovementFinding],
    ) -> str:
        """Formats findings with explicit 80/20 prioritization and quantitative metrics."""
        # Sort by Pareto score (impact/effort) descending
        sorted_findings = sorted(findings, key=lambda x: x.pareto_score, reverse=True)

        markdown = "## Highest Impact Improvements (80/20 Analysis)\n\n"
        markdown += "The following improvements were prioritized using quantitative impact metrics "
        markdown += "to identify the 20% of changes that deliver 80% of potential improvements:\n\n"

        # Limit to top 3 findings for the summary section
        for i, finding in enumerate(sorted_findings[:3], 1):
            markdown += f"### #{i} Priority: {finding.area} (Pareto Score: {finding.pareto_score:.2f})\n\n"
            markdown += f"**Problem:** {finding.problem}\n\n"
            markdown += f"**Solution:** {finding.solution}\n\n"

            # Safely access metrics, providing defaults if missing
            metrics_str = "**Quantitative Impact:**\n"
            quality_improvement = (
                finding.metrics.expected_quality_improvement if finding.metrics else 0.0
            )
            token_savings = (
                finding.metrics.token_savings_percent if finding.metrics else 0.0
            )

            metrics_str += f"- Quality Improvement: {quality_improvement * 100:.1f}%\n"
            # Handle potential None for token savings
            if finding.metrics and finding.metrics.token_savings_percent is not None:
                metrics_str += f"- Token Savings: {token_savings * 100:.1f}%\n"
            else:
                metrics_str += "- Token Savings: N/A\n"

            markdown += metrics_str + "\n"

        return markdown

    @staticmethod
    def format_complete_report(analysis: SelfImprovementAnalysisOutput) -> str:
        """Formats the complete analysis report in Markdown."""
        markdown = f"# Project Chimera Self-Improvement Analysis Report\n\n"
        markdown += f"**Analysis ID:** {analysis.metadata.get('analysis_id', 'N/A')}\n"
        markdown += f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        markdown += f"**Original Prompt:** {analysis.original_prompt}\n\n"

        # NEW: Start with Pareto-prioritized findings
        markdown += OutputFormatter.format_pareto_prioritized_findings(
            analysis.findings
        )
        markdown += "\n---\n\n"
        markdown += "## Detailed Analysis\n\n"
        markdown += f"{analysis.summary}\n\n"

        markdown += "## Impactful Suggestions\n\n"
        if not analysis.findings:
            markdown += "No specific suggestions were generated.\n"
        else:
            # Use the detailed findings formatter for the full list
            markdown += OutputFormatter.format_findings_list(analysis.findings)

        return markdown
