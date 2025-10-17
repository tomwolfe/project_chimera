"""
Modular Persona Router for Project Chimera
Separated from core.py to reduce complexity
"""

from typing import Any

from src.models import PersonaConfig


class PersonaRouter:
    """
    Handles persona routing logic that was previously in core.py
    """

    def __init__(self, persona_configs: dict[str, PersonaConfig]):
        self.persona_configs = persona_configs or {}

    def route_persona(self, prompt: str, context: dict[str, Any]) -> str:
        """
        Determine the appropriate persona for a given prompt and context
        """
        # Simplified routing logic that was extracted from core.py
        if not prompt or not context:
            return "Creative"  # Default persona

        # Placeholder for complex routing logic that would be moved from core.py
        # This is a simplified version to demonstrate the separation of concerns
        return self._determine_persona(prompt, context)

    def _determine_persona(self, prompt: str, context: dict[str, Any]) -> str:
        """Internal method to determine persona"""
        # In a real implementation, this would contain the complex logic
        # that was previously in the core.py routing functions
        return "Creative"
