import pytest
from unittest.mock import MagicMock, patch
from core import SocraticDebate
from src.persona_manager import PersonaManager
from src.config.settings import ChimeraSettings

@pytest.fixture
def mock_persona_manager():
    pm = MagicMock(spec=PersonaManager)
    pm.all_personas = {
        "General_Synthesizer": MagicMock(name="General_Synthesizer", system_prompt="sys", temperature=0.5, max_tokens=1024)
    }
    pm.persona_sets = {"General": ["General_Synthesizer"]}
    pm.prompt_analyzer = MagicMock()
    pm.prompt_analyzer.analyze_complexity.return_value = {'complexity_score': 0.1, 'primary_domain': 'General', 'domain_scores': {}, 'word_count': 2, 'sentence_count': 1}
    pm.prompt_analyzer.is_self_analysis_prompt.return_value = False
    pm.persona_router = MagicMock()
    pm.persona_router.determine_persona_sequence.return_value = ["General_Synthesizer"]
    return pm

def test_socratic_debate_initialization_success(mock_persona_manager):
    # Arrange
    mock_api_key = "mock_key"
    mock_prompt = "Test prompt for initialization"
    settings = ChimeraSettings()

    # Act
    debate_instance = SocraticDebate(
        initial_prompt=mock_prompt,
        api_key=mock_api_key,
        persona_manager=mock_persona_manager,
        settings=settings
    )

    # Assert
    assert debate_instance.initial_prompt == mock_prompt
    assert debate_instance.model_name == "gemini-2.5-flash-lite"
    assert debate_instance.tokens_used == 0
    assert "General" in debate_instance.persona_sets

def test_socratic_debate_token_budget_calculation(mock_persona_manager):
    # Arrange
    mock_api_key = "mock_key"
    mock_prompt = "Short prompt"
    settings = ChimeraSettings(total_budget=10000)

    # Act
    debate_instance = SocraticDebate(
        initial_prompt=mock_prompt,
        api_key=mock_api_key,
        persona_manager=mock_persona_manager,
        settings=settings
    )

    # Assert that budgets are calculated and non-zero (within reasonable bounds)
    assert debate_instance.phase_budgets["context"] > 0
    assert debate_instance.phase_budgets["debate"] > 0
    assert debate_instance.phase_budgets["synthesis"] > 0
    assert sum(debate_instance.phase_budgets[p] for p in ["context", "debate", "synthesis"]) <= settings.total_budget