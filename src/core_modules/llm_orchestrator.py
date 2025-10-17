"""
Modular LLM Orchestrator for Project Chimera
Separated from core.py to reduce complexity
"""

from typing import Any

from src.llm.orchestrator import LLMOrchestrator


class EnhancedLLMOrchestrator:
    """
    Enhanced LLM orchestration logic that was previously in core.py
    """

    def __init__(self, base_orchestrator: LLMOrchestrator):
        self.base_orchestrator = base_orchestrator

    def execute_request(
        self,
        prompt: str,
        persona_config: dict[str, Any],
        context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """
        Execute an LLM request with enhanced logic
        """
        # Placeholder for complex orchestration logic that would be moved from core.py
        # This is a simplified version to demonstrate the separation of concerns
        return self.base_orchestrator.execute_request(prompt, persona_config, context)
