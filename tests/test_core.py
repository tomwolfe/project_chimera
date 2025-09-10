import pytest
from unittest.mock import MagicMock

# Assuming core.py and persona_manager.py are in the project root or accessible via path
# Adjust imports as necessary based on actual project structure
from src.core import DebateOrchestrator
from src.persona_manager import PersonaManager


def test_debate_orchestrator_initialization():
    """Test that the DebateOrchestrator initializes correctly."""
    mock_persona_manager = MagicMock(spec=PersonaManager)
    orchestrator = DebateOrchestrator(persona_manager=mock_persona_manager) # nosec B101
    if orchestrator.persona_manager != mock_persona_manager:
        raise AssertionError("Persona manager not correctly initialized")
    if orchestrator.debate_history != []:
        raise AssertionError("Debate history not empty")
    if orchestrator.current_turn != 0:
        raise AssertionError("Current turn not 0")

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
    if len(orchestrator.debate_history) != 4: # 2 turns * 2 personas per turn
        raise AssertionError("Debate history length mismatch")
    if mock_analyst.get_response.call_count != 2:
        raise AssertionError("Analyst response call count mismatch")
    if mock_critic.get_response.call_count != 2:
        raise AssertionError("Critic response call count mismatch")
    
    # Check history content (basic check)
    for entry in orchestrator.debate_history:
        if 'persona_name' not in entry: raise AssertionError("Missing 'persona_name' in history entry")
        if 'response' not in entry: raise AssertionError("Missing 'response' in history entry")
        if 'turn' not in entry: raise AssertionError("Missing 'turn' in history entry")

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
    
    if mock_persona1.get_response.call_count != 1:
        raise AssertionError("Persona1 response call count mismatch")
    if mock_persona2.get_response.call_count != 1:
        raise AssertionError("Persona2 response call count mismatch")

# Add more tests for:
# - Handling of errors during persona response generation
# - Different debate termination conditions
# - Interaction with the SelfImprovementAnalyst persona
# - Specific routing logic defined in src/persona/routing.py