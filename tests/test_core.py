# tests/test_core.py

import pytest
from unittest.mock import MagicMock
from src.core import ProjectChimera

# Mocking necessary components for core logic testing
@pytest.fixture
def mock_llm_provider():
    mock_provider = MagicMock()
    mock_provider.generate_content.return_value = "Mocked LLM response."
    return mock_provider

@pytest.fixture # Corrected: Removed duplicate pytest.
def mock_persona_manager():
    mock_manager = MagicMock()
    mock_manager.get_persona_by_name.return_value = MagicMock()
    return mock_manager

@pytest.fixture # Corrected: Removed duplicate pytest.
def mock_context_analyzer():
    mock_analyzer = MagicMock()
    mock_analyzer.analyze_context.return_value = {"summary": "Mocked context analysis."}
    return mock_analyzer

@pytest.fixture # Corrected: Removed duplicate pytest.
def mock_output_parser():
    mock_parser = MagicMock()
    mock_parser.parse_output.return_value = {"action": "continue", "reasoning": "Mocked parsing."}
    return mock_parser

def test_project_chimera_initialization(mock_llm_provider, mock_persona_manager, mock_context_analyzer, mock_output_parser):
    chimera = ProjectChimera(
        llm_provider=mock_llm_provider,
        persona_manager=mock_persona_manager,
        context_analyzer=mock_context_analyzer,
        output_parser=mock_output_parser
    )
    assert chimera.llm_provider == mock_llm_provider
    assert chimera.persona_manager == mock_persona_manager
    assert chimera.context_analyzer == mock_context_analyzer
    assert chimera.output_parser == mock_output_parser
    assert chimera.current_persona is None

def test_project_chimera_run_debate_simple_flow(mock_llm_provider, mock_persona_manager, mock_context_analyzer, mock_output_parser):
    # Mock persona manager to return a simple sequence
    mock_persona_manager.get_persona_sequence_for_framework.return_value = ["Visionary_Generator", "Impartial_Arbitrator"]
    mock_persona_manager.all_personas = {
        "Visionary_Generator": MagicMock(name="Visionary_Generator", system_prompt="Visionary", temperature=0.7, max_tokens=100),
        "Impartial_Arbitrator": MagicMock(name="Impartial_Arbitrator", system_prompt="Arbitrator", temperature=0.2, max_tokens=100)
    }
    mock_persona_manager.get_adjusted_persona_config.side_effect = lambda name: mock_persona_manager.all_personas[name]

    chimera = ProjectChimera(
        llm_provider=mock_llm_provider,
        persona_manager=mock_persona_manager,
        context_analyzer=mock_context_analyzer,
        output_parser=mock_output_parser,
        initial_prompt="Test prompt",
        api_key="mock_api_key",
        model_name="mock-model",
        domain="General"
    )

    # Mock the internal _execute_llm_turn to return structured output
    with patch.object(chimera, '_execute_llm_turn', side_effect=[
        {"general_output": "Visionary idea"}, # Visionary_Generator output
        {"general_output": "Final synthesis"}  # Impartial_Arbitrator output
    ]) as mock_execute_llm_turn:
        final_answer, intermediate_steps = chimera.run_debate()

        assert "Final synthesis" in final_answer.get("general_output", "")
        assert "Visionary_Generator_Output" in intermediate_steps
        assert "Impartial_Arbitrator_Output" in intermediate_steps
        mock_execute_llm_turn.call_count == 2 # Should be called twice for two personas

def test_project_chimera_run_debate_with_context_analysis(mock_llm_provider, mock_persona_manager, mock_context_analyzer, mock_output_parser):
    mock_persona_manager.get_persona_sequence_for_framework.return_value = ["Context_Aware_Assistant", "Impartial_Arbitrator"]
    mock_persona_manager.all_personas = {
        "Context_Aware_Assistant": MagicMock(name="Context_Aware_Assistant", system_prompt="Context", temperature=0.1, max_tokens=200),
        "Impartial_Arbitrator": MagicMock(name="Impartial_Arbitrator", system_prompt="Arbitrator", temperature=0.2, max_tokens=100)
    }
    mock_persona_manager.get_adjusted_persona_config.side_effect = lambda name: mock_persona_manager.all_personas[name]

    mock_context_analyzer.find_relevant_files.return_value = [("file1.py", 0.9)]
    mock_context_analyzer.generate_context_summary.return_value = "Summary of file1.py"

    chimera = ProjectChimera(
        llm_provider=mock_llm_provider,
        persona_manager=mock_persona_manager,
        context_analyzer=mock_context_analyzer,
        output_parser=mock_output_parser,
        initial_prompt="Analyze code",
        api_key="mock_api_key",
        model_name="mock-model",
        domain="Software Engineering",
        codebase_context={"file1.py": "print('hello')"}
    )

    with patch.object(chimera, '_execute_llm_turn', side_effect=[
        {"general_overview": "Context analysis done"}, # Context_Aware_Assistant output
        {"general_output": "Final answer with context"} # Impartial_Arbitrator output
    ]) as mock_execute_llm_turn:
        final_answer, intermediate_steps = chimera.run_debate()

        assert "Final answer with context" in final_answer.get("general_output", "")
        assert "Context_Analysis_Output" in intermediate_steps
        assert "Context_Aware_Assistant_Output" in intermediate_steps
        mock_context_analyzer.find_relevant_files.assert_called_once()
        mock_context_analyzer.generate_context_summary.assert_called_once()
        mock_execute_llm_turn.call_count == 2