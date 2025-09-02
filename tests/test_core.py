import pytest
from unittest.mock import MagicMock, patch
import json

# Assuming Chimera, MockLLMProvider, MockPersonaManager, MockOutputParser are importable
# Adjust imports based on actual project structure
from core import Chimera
from src.models import LLMOutput, CritiqueOutput, SelfImprovementAnalysisOutput


@pytest.fixture
def chimera_instance():
    """Provides a mock Chimera instance for testing."""
    mock_llm_provider = MagicMock()
    mock_persona_manager = MagicMock()
    mock_context_analyzer = MagicMock()
    mock_output_parser = MagicMock()

    # Configure mock return values for methods called by Chimera
    mock_llm_provider.generate.return_value = "Mock LLM response"
    mock_persona_manager.get_persona.side_effect = lambda name: {"name": name, "description": f"Description for {name}"} if name == "default" else None
    mock_persona_manager.list_personas.return_value = ["default", "expert"]
    mock_context_analyzer.analyze.return_value = {"key_modules": [], "general_overview": "Mock overview"}
    mock_output_parser.parse_and_validate.side_effect = [
        LLMOutput(COMMIT_MESSAGE="Feat: Initial commit", CODE_CHANGES=[], RATIONALE="", MALFORMED_BLOCKS=[]),
        CritiqueOutput(CRITIQUE_SUMMARY="Good", CRITIQUE_POINTS=[], SUGGESTIONS=[], MALFORMED_BLOCKS=[]),
        SelfImprovementAnalysisOutput(version="1.0", data={"ANALYSIS_SUMMARY": "OK", "IMPACTFUL_SUGGESTIONS": []}, MALFORMED_BLOCKS=[])
    ]

    chimera = Chimera(
        llm_provider=mock_llm_provider,
        persona_manager=mock_persona_manager,
        context_analyzer=mock_context_analyzer,
        output_parser=mock_output_parser,
        initial_prompt="Analyze this code.",
        domain="Software Engineering",
        codebase_context={"file.py": "print('hello')"}
    )
    return chimera

def test_chimera_initialization(chimera_instance):
    """Tests the initialization of the Chimera class."""
    assert chimera_instance.llm_provider is not None
    assert chimera_instance.persona_manager is not None
    assert chimera_instance.context_analyzer is not None
    assert chimera_instance.output_parser is not None
    assert chimera_instance.initial_prompt == "Analyze this code."
    assert chimera_instance.domain == "Software Engineering"
    assert chimera_instance.codebase_context == {"file.py": "print('hello')"}

def test_chimera_run_debate_success(chimera_instance):
    """Tests the successful execution of the run_debate method."""
    # Mock the LLM provider to return a structured response that the parser can handle
    mock_llm_provider = chimera_instance.llm_provider
    mock_llm_provider.generate.return_value = (json.dumps({
        "COMMIT_MESSAGE": "Feat: Implement debate logic",
        "CODE_CHANGES": [
            {
                "FILE_PATH": "src/core.py",
                "ACTION": "MODIFY",
                "DIFF_CONTENT": "+    print('debate')"
            }
        ],
        "RATIONALE": "Implementing the core debate loop.",
        "MALFORMED_BLOCKS": []
    }), 100, 50) # Simulate return tuple (text, input_tokens, output_tokens)

    # Mock the output parser to return a valid LLMOutput object
    mock_output_parser = chimera_instance.output_parser
    mock_output_parser.parse_and_validate.side_effect = [
        LLMOutput(
            COMMIT_MESSAGE="Feat: Implement debate logic",
            CODE_CHANGES=[
                {
                    "FILE_PATH": "src/core.py",
                    "ACTION": "MODIFY",
                    "DIFF_CONTENT": "+    print('debate')"
                }
            ],
            RATIONALE="Implementing the core debate loop.",
            MALFORMED_BLOCKS=[]
        ),
        CritiqueOutput(
            CRITIQUE_SUMMARY="Debate logic is sound.",
            CRITIQUE_POINTS=[],
            SUGGESTIONS=[],
            MALFORMED_BLOCKS=[]
        ),
        SelfImprovementAnalysisOutput(
            version="1.0",
            data={"ANALYSIS_SUMMARY": "Debate successful.", "IMPACTFUL_SUGGESTIONS": []},
            MALFORMED_BLOCKS=[]
        )
    ]

    final_answer, intermediate_steps = chimera_instance.run_debate()

    assert isinstance(final_answer, dict)
    assert "general_output" in final_answer
    assert "Parsed output" in final_answer["general_output"]
    assert "MockPersona1_Output" in intermediate_steps
    assert "MockPersona2_Output" in intermediate_steps
    mock_llm_provider.generate.assert_called_once()
    mock_output_parser.parse_and_validate.assert_called()

def test_chimera_run_debate_with_malformed_output(chimera_instance):
    """Tests run_debate when the LLM output is malformed."""
    mock_llm_provider = chimera_instance.llm_provider
    mock_llm_provider.generate.return_value = ("Invalid JSON output", 10, 5) # Simulate malformed output

    mock_output_parser = chimera_instance.output_parser
    mock_output_parser.parse_and_validate.side_effect = [
        LLMOutput(COMMIT_MESSAGE="LLM_OUTPUT_ERROR", CODE_CHANGES=[], RATIONALE="Failed to parse LLM output.", MALFORMED_BLOCKS=[{"type": "JSON_DECODE_ERROR", "message": "Invalid JSON"}]),
        CritiqueOutput(CRITIQUE_SUMMARY="Error during debate.", CRITIQUE_POINTS=[], SUGGESTIONS=[], MALFORMED_BLOCKS=[]),
        SelfImprovementAnalysisOutput(version="1.0", data={"ANALYSIS_SUMMARY": "Debate failed due to LLM output error.", "IMPACTFUL_SUGGESTIONS": []}, MALFORMED_BLOCKS=[])
    ]

    final_answer, intermediate_steps = chimera_instance.run_debate()

    assert isinstance(final_answer, dict)
    assert "general_output" in final_answer
    assert "LLM_OUTPUT_ERROR" in final_answer["general_output"]
    assert "MockPersona1_Output" in intermediate_steps
    assert "MockPersona2_Output" in intermediate_steps
    mock_llm_provider.generate.assert_called_once()
    mock_output_parser.parse_and_validate.assert_called()

# Add more tests for different scenarios, e.g., empty codebase context, specific persona interactions, etc.