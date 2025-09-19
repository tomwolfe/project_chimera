import json
import logging
from typing import Dict, Any, List, Optional
from pathlib import Path
import re

logger = logging.getLogger(__name__)


class StrategyManager:
    """Manages the overall self-improvement strategy, including phases, metrics, and goals."""

    def __init__(self, strategy_file: str = "docs/system_improvement_strategy.md"):
        self.strategy_file = Path(strategy_file)
        self.current_strategy = self._load_strategy()

    def _load_strategy(self) -> Dict[str, Any]:
        """Loads the self-improvement strategy from the documentation file."""
        if self.strategy_file.exists():
            try:
                # In a real implementation, this would parse the markdown document
                # to extract structured strategy. For now, return a placeholder.
                logger.info(f"Loading strategy from {self.strategy_file}")
                # Simulate parsing a structured section from the markdown
                content = self.strategy_file.read_text(encoding="utf-8")

                # Example: Extracting framework and key areas from the markdown
                framework_match = re.search(
                    r"## Self-Improvement Framework\n\n(.*?)\n## Prioritization Criteria",
                    content,
                    re.DOTALL,
                )
                key_areas_match = re.search(
                    r"## Key Areas for Improvement\n\n(.*?)\n", content, re.DOTALL
                )

                framework_text = (
                    framework_match.group(1).strip() if framework_match else ""
                )
                key_areas_text = (
                    key_areas_match.group(1).strip() if key_areas_match else ""
                )

                framework_goals = {}
                for line in framework_text.split("\n"):
                    step_match = re.match(r"(\d+\.\s*\*\*([A-Za-z-]+):\*\*.*)", line)
                    if step_match:
                        step_name = step_match.group(2).lower().replace("-", "_")
                        framework_goals[step_name] = {"goal": line.strip()}

                key_areas = [
                    line.strip("* ").strip()
                    for line in key_areas_text.split("\n")
                    if line.strip().startswith("*")
                ]

                return {"framework": framework_goals, "key_areas": key_areas}
            except Exception as e:
                logger.error(
                    f"Error parsing strategy file {self.strategy_file}: {e}",
                    exc_info=True,
                )
        logger.warning(
            f"Strategy file {self.strategy_file} not found or malformed. Returning default strategy."
        )
        return {
            "framework": {
                "assessment": {"goal": "Identify areas for improvement"},
                "development": {"goal": "Formulate targeted strategies"},
                "implementation": {"goal": "Apply strategies and measure impact"},
                "evaluation": {"goal": "Evaluate effectiveness and iterate"},
            },
            "key_areas": [
                "Reasoning Quality",
                "Robustness",
                "Efficiency",
                "Maintainability",
            ],
        }

    def get_current_strategy(self) -> Dict[str, Any]:
        return self.current_strategy

    def update_strategy(self, new_strategy: Dict[str, Any]):
        # Logic to update the strategy document and internal state
        logger.info("Updating strategy (in-memory only for now)...")
        self.current_strategy = new_strategy
        # TODO: Implement saving the strategy back to a file (e.g., by modifying the markdown)

    def get_improvement_goals(self) -> List[str]:
        return self.current_strategy.get("key_areas", [])

    def get_framework_step_goal(self, step: str) -> str:
        return self.current_strategy.get("framework", {}).get(
            step, {"goal": "No goal defined"}
        )["goal"]
