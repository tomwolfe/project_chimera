import pytest

# Assuming src/persona/routing.py exists and contains the routing logic
# Adjust import based on actual file structure
from src.persona.routing import select_next_persona, get_initial_personas


def test_get_initial_personas_basic():
    """Test that initial personas are selected correctly."""
    # Mocking PersonaManager or providing a simplified structure for testing
    class MockPersonaManager:
        def get_active_personas(self):
            # Return a list of mock persona objects or names
            return ["Analyst", "Critic", "SelfImprovementAnalyst"]

    mock_manager = MockPersonaManager()
    initial_personas = get_initial_personas(mock_manager)
    
    assert isinstance(initial_personas, list)
    assert len(initial_personas) > 0
    # Add more specific assertions based on expected initial personas
    assert "Analyst" in initial_personas

def test_select_next_persona_round_robin():
    """Test a simple round-robin persona selection logic."""
    # Assuming a round-robin strategy for demonstration
    active_personas = ["Analyst", "Critic", "SelfImprovementAnalyst"]
    current_persona_index = 0 # Analyst is current
    
    next_persona = select_next_persona(active_personas, current_persona_index)
    
    assert next_persona == "Critic"

    current_persona_index = 1 # Critic is current
    next_persona = select_next_persona(active_personas, current_persona_index)
    assert next_persona == "SelfImprovementAnalyst"

    current_persona_index = 2 # SelfImprovementAnalyst is current
    next_persona = select_next_persona(active_personas, current_persona_index)
    assert next_persona == "Analyst" # Wraps around

def test_select_next_persona_no_personas():
    """Test behavior when no active personas are available."""
    active_personas = []
    current_persona_index = 0
    
    next_persona = select_next_persona(active_personas, current_persona_index)
    assert next_persona is None

def test_select_next_persona_single_persona():
    """Test behavior with only one active persona."""
    active_personas = ["Analyst"]
    current_persona_index = 0
    
    next_persona = select_next_persona(active_personas, current_persona_index)
    assert next_persona == "Analyst" # Should loop back to itself

# Add more tests for:
# - Complex routing logic (e.g., based on debate state, previous responses)
# - Handling of persona availability changes during a debate
# - Integration with the DebateOrchestrator to ensure correct state updates