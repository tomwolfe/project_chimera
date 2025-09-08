import pytest
from unittest.mock import MagicMock

# Assuming core.py and persona_manager.py are in the project root or accessible via path
# Adjust imports as necessary based on actual project structure
from src.core import DebateOrchestrator
from src.persona_manager import PersonaManager


def test_debate_orchestrator_initialization():
    """Test that the DebateOrchestrator initializes correctly."""
    mock_persona_manager = MagicMock(spec=PersonaManager)
    orchestrator = DebateOrchestrator(persona_manager=mock_persona_manager)
    assert orchestrator.persona_manager == mock_persona_manager
    assert orchestrator.debate_history == []
    assert orchestrator.current_turn == 0

def test_debate_orchestrator_run_debate_basic():
    """Test a basic debate flow with mock personas."""
    mock_persona_manager = MagicMock(spec=PersonaManager)
    
    # Mock personas and their responses
    mock_analyst = MagicMock()
    mock_analyst.name = "Analyst"
    mock_analyst.system_prompt = "You are an analyst."
    mock_analyst.get_response.return_value = "Analysis result."
    
    mock_critic = MagicMock()
    mock_critic.name = "Critic"
    mock_critic.system_prompt = "You are a critic."
    mock_critic.get_response.return_value = "Criticism."
    
    mock_persona_manager.get_persona.side_effect = [mock_analyst, mock_critic]
    mock_persona_manager.get_active_personas.return_value = [mock_analyst, mock_critic]

    orchestrator = DebateOrchestrator(persona_manager=mock_persona_manager)
    
    # Simulate a few turns
    orchestrator.run_debate(max_turns=2)

    # Assertions
    assert len(orchestrator.debate_history) == 4 # 2 turns * 2 personas per turn
    assert mock_analyst.get_response.call_count == 2
    assert mock_critic.get_response.call_count == 2
    
    # Check history content (basic check)
    for entry in orchestrator.debate_history:
        assert 'persona_name' in entry
        assert 'response' in entry
        assert 'turn' in entry

def test_debate_orchestrator_persona_selection_logic():
    """Test if the orchestrator correctly selects personas based on some logic (simplified)."""
    mock_persona_manager = MagicMock(spec=PersonaManager)
    
    mock_persona1 = MagicMock()
    mock_persona1.name = "Persona1"
    mock_persona1.get_response.return_value = "Response 1"
    
    mock_persona2 = MagicMock()
    mock_persona2.name = "Persona2"
    mock_persona2.get_response.return_value = "Response 2"

    # Simulate a routing logic where Persona1 is chosen first, then Persona2
    mock_persona_manager.get_persona.side_effect = [mock_persona1, mock_persona2]
    mock_persona_manager.get_active_personas.return_value = [mock_persona1, mock_persona2]

    orchestrator = DebateOrchestrator(persona_manager=mock_persona_manager)
    
    # Run a single turn to test persona selection
    orchestrator.run_debate(max_turns=1)

    # Assert that the correct personas were called in sequence
    # This test assumes a simple round-robin or sequential selection for demonstration
    # Actual routing logic in core.py would need to be reflected here.
    assert mock_persona1.get_response.call_count == 1
    assert mock_persona2.get_response.call_count == 1
    
    # A more sophisticated test would inspect the order of calls or the history entries
    # to verify the sequence of persona engagement.

# Add more tests for:
# - Handling of errors during persona response generation
# - Different debate termination conditions
# - Interaction with the SelfImprovementAnalyst persona
# - Specific routing logic defined in src/persona/routing.py