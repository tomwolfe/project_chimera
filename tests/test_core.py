import pytest
from unittest.mock import MagicMock, patch
import json

# Assuming Chimera, MockLLMProvider, MockPersonaManager, MockOutputParser are importable
# Adjust imports based on actual project structure
from core import Chimera
from src.models import LLMOutput, CritiqueOutput, SelfImprovementAnalysisOutput, GeneralOutput # Added GeneralOutput for fallback


@pytest.fixture
def chimera_instance():
    """Provides a mock Chimera instance for testing."""
    mock_llm_provider = MagicMock()
    mock_persona_manager = MagicMock()
    mock_context_analyzer = MagicMock()
    mock_output_parser = MagicMock()

    # Configure mock return values for methods called by Chimera
    mock_llm_provider.generate.return_value = ("Mock LLM response", 100, 50) # Simulate return tuple (text, input_tokens, output_tokens)
    mock_persona_manager.get_persona.side_effect = lambda name: {"name": name, "description": f"Description for {name}"} if name == "default" else None
    mock_persona_manager.list_personas.return_value = ["default", "expert"]
    mock_context_analyzer.analyze.return_value = {"key_modules": [], "general_overview": "Mock overview"}
    
    # Initial setup for parse_and_validate, will be overridden in specific tests
    mock_output_parser.parse_and_validate.side_effect = [
        GeneralOutput(general_output="MockPersona1_Output", malformed_blocks=[]).model_dump(by_alias=True),
        GeneralOutput(general_output="MockPersona2_Output", malformed_blocks=[]).model_dump(by_alias=True),
        SelfImprovementAnalysisOutput(version="1.0", data={"ANALYSIS_SUMMARY": "OK", "IMPACTFUL_SUGGESTIONS": []}, malformed_blocks=[]).model_dump(by_alias=True)
    ]

    # Mock the TokenUsageTracker as it's a dependency
    mock_token_tracker = MagicMock()
    mock_token_tracker.current_usage = 0
    mock_token_tracker.budget = 2000000
    mock_token_tracker.record_usage.side_effect = lambda tokens, persona: setattr(mock_token_tracker, 'current_usage', mock_token_tracker.current_usage + tokens)
    mock_token_tracker.reset.side_effect = lambda: setattr(mock_token_tracker, 'current_usage', 0)
    mock_token_tracker.get_consumption_rate.return_value = 0.0

    chimera = Chimera(
        api_key="dummy-api-key", # SocraticDebate expects API key
        model_name="gemini-2.5-flash-lite",
        initial_prompt="Analyze this code.",
        domain="Software Engineering",
        codebase_context={"file.py": "print('hello')"},
        # Pass mocks to SocraticDebate
        llm_provider=mock_llm_provider, # This will be overridden by SocraticDebate's internal init
        persona_manager=mock_persona_manager,
        context_analyzer=mock_context_analyzer,
        output_parser=mock_output_parser,
        token_tracker=mock_token_tracker,
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

@patch('core.GeminiProvider') # Mock the GeminiProvider during SocraticDebate init
@patch('core.PersonaManager') # Mock PersonaManager during SocraticDebate init
@patch('core.ContextRelevanceAnalyzer') # Mock ContextRelevanceAnalyzer during SocraticDebate init
@patch('core.LLMOutputParser') # Mock LLMOutputParser during SocraticDebate init
@patch('core.TokenUsageTracker') # Mock TokenUsageTracker during SocraticDebate init
def test_chimera_run_debate_success(mock_token_tracker_cls, mock_output_parser_cls, mock_context_analyzer_cls, mock_persona_manager_cls, mock_gemini_provider_cls):
    """Tests a successful run of the Chimera debate."""
    # Setup mocks for a successful debate scenario
    mock_llm_provider_instance = MagicMock()
    mock_llm_provider_instance.generate.return_value = (
        json.dumps({
            "COMMIT_MESSAGE": "Feat: Implement debate logic",
            "CODE_CHANGES": [
                {
                    "FILE_PATH": "src/core.py",
                    "ACTION": "MODIFY",
                    "DIFF_CONTENT": "+    print('debate')"
                }
            ],
            "RATIONALE": "Implementing the core debate loop.",
            "malformed_blocks": []
        }), 100, 50
    )
    mock_llm_provider_instance.calculate_usd_cost.return_value = 0.0001

    mock_gemini_provider_cls.return_value = mock_llm_provider_instance

    mock_persona_manager_instance = MagicMock()
    mock_persona_manager_instance.get_adjusted_persona_config.side_effect = lambda name: MagicMock(name=name, system_prompt=f"System prompt for {name}", temperature=0.5, max_tokens=1024)
    mock_persona_manager_instance.persona_sets = {"Software Engineering": ["Code_Architect", "Impartial_Arbitrator"]}
    mock_persona_manager_instance.persona_router.determine_persona_sequence.return_value = ["Code_Architect", "Impartial_Arbitrator"]
    mock_persona_manager_instance.get_token_optimized_persona_sequence.side_effect = lambda seq: seq # No optimization for this test
    mock_persona_manager_instance._analyze_prompt_complexity.return_value = {"complexity_score": 0.5, "primary_domain": "Software Engineering", "domain_scores": {}, "word_count": 10, "sentence_count": 1}
    mock_persona_manager_instance.record_persona_performance.return_value = None

    mock_persona_manager_cls.return_value = mock_persona_manager_instance

    mock_context_analyzer_instance = MagicMock()
    mock_context_analyzer_instance.find_relevant_files.return_value = []
    mock_context_analyzer_instance.generate_context_summary.return_value = ""
    mock_context_analyzer_cls.return_value = mock_context_analyzer_instance

    mock_output_parser_instance = MagicMock()
    mock_output_parser_instance.parse_and_validate.side_effect = [
        CritiqueOutput(CRITIQUE_SUMMARY="Architectural critique", CRITIQUE_POINTS=[], SUGGESTIONS=[], malformed_blocks=[]).model_dump(by_alias=True),
        LLMOutput(COMMIT_MESSAGE="Final commit", RATIONALE="Final rationale", CODE_CHANGES=[], malformed_blocks=[]).model_dump(by_alias=True)
    ]
    mock_output_parser_cls.return_value = mock_output_parser_instance

    mock_token_tracker_instance = MagicMock()
    mock_token_tracker_instance.current_usage = 0
    mock_token_tracker_instance.budget = 2000000
    mock_token_tracker_instance.record_usage.side_effect = lambda tokens, persona: setattr(mock_token_tracker_instance, 'current_usage', mock_token_tracker_instance.current_usage + tokens)
    mock_token_tracker_instance.reset.side_effect = lambda: setattr(mock_token_tracker_instance, 'current_usage', 0)
    mock_token_tracker_instance.get_consumption_rate.return_value = 0.0
    mock_token_tracker_cls.return_value = mock_token_tracker_instance


    chimera = Chimera(
        api_key="dummy-api-key",
        model_name="gemini-2.5-flash-lite",
        initial_prompt="Analyze this code.",
        domain="Software Engineering",
        codebase_context={"file.py": "print('hello')"},
        status_callback=MagicMock(),
        rich_console=MagicMock(),
    )

    final_answer, intermediate_steps = chimera.run_debate()

    assert isinstance(final_answer, dict)
    assert "COMMIT_MESSAGE" in final_answer
    assert final_answer["COMMIT_MESSAGE"] == "Final commit"
    assert "Code_Architect_Output" in intermediate_steps
    assert "Final_Synthesis_Output" in intermediate_steps
    mock_llm_provider_instance.generate.assert_called()
    mock_output_parser_instance.parse_and_validate.assert_called()
    mock_token_tracker_instance.record_usage.assert_called()


@patch('core.GeminiProvider')
@patch('core.PersonaManager')
@patch('core.ContextRelevanceAnalyzer')
@patch('core.LLMOutputParser')
@patch('core.TokenUsageTracker')
def test_chimera_run_debate_llm_error(mock_token_tracker_cls, mock_output_parser_cls, mock_context_analyzer_cls, mock_persona_manager_cls, mock_gemini_provider_cls):
    """Tests Chimera run_debate when LLM returns an error."""
    mock_llm_provider_instance = MagicMock()
    mock_llm_provider_instance.generate.side_effect = Exception("LLM Generation Failed")
    mock_gemini_provider_cls.return_value = mock_llm_provider_instance

    mock_persona_manager_instance = MagicMock()
    mock_persona_manager_instance.get_adjusted_persona_config.side_effect = lambda name: MagicMock(name=name, system_prompt=f"System prompt for {name}", temperature=0.5, max_tokens=1024)
    mock_persona_manager_instance.persona_sets = {"Software Engineering": ["Code_Architect", "Impartial_Arbitrator"]}
    mock_persona_manager_instance.persona_router.determine_persona_sequence.return_value = ["Code_Architect", "Impartial_Arbitrator"]
    mock_persona_manager_instance.get_token_optimized_persona_sequence.side_effect = lambda seq: seq
    mock_persona_manager_instance._analyze_prompt_complexity.return_value = {"complexity_score": 0.5, "primary_domain": "Software Engineering", "domain_scores": {}, "word_count": 10, "sentence_count": 1}
    mock_persona_manager_instance.record_persona_performance.return_value = None
    mock_persona_manager_cls.return_value = mock_persona_manager_instance

    mock_context_analyzer_instance = MagicMock()
    mock_context_analyzer_instance.find_relevant_files.return_value = []
    mock_context_analyzer_instance.generate_context_summary.return_value = ""
    mock_context_analyzer_cls.return_value = mock_context_analyzer_instance

    mock_output_parser_instance = MagicMock()
    mock_output_parser_instance.parse_and_validate.return_value = {} # Should not be called if LLM fails early
    mock_output_parser_cls.return_value = mock_output_parser_instance

    mock_token_tracker_instance = MagicMock()
    mock_token_tracker_instance.current_usage = 0
    mock_token_tracker_instance.budget = 2000000
    mock_token_tracker_instance.record_usage.side_effect = lambda tokens, persona: setattr(mock_token_tracker_instance, 'current_usage', mock_token_tracker_instance.current_usage + tokens)
    mock_token_tracker_instance.reset.side_effect = lambda: setattr(mock_token_tracker_instance, 'current_usage', 0)
    mock_token_tracker_instance.get_consumption_rate.return_value = 0.0
    mock_token_tracker_cls.return_value = mock_token_tracker_instance

    chimera = Chimera(
        api_key="dummy-api-key",
        model_name="gemini-2.5-flash-lite",
        initial_prompt="Analyze this code.",
        domain="Software Engineering",
        codebase_context={"file.py": "print('hello')"},
        status_callback=MagicMock(),
        rich_console=MagicMock(),
    )

    with pytest.raises(Exception, match="LLM Generation Failed"):
        chimera.run_debate()

    mock_llm_provider_instance.generate.assert_called_once()
    mock_output_parser_instance.parse_and_validate.assert_not_called() # Parser should not be called if LLM fails before generating output
    mock_token_tracker_instance.record_usage.assert_called() # Tokens for prompt might still be recorded

@patch('core.GeminiProvider')
@patch('core.PersonaManager')
@patch('core.ContextRelevanceAnalyzer')
@patch('core.LLMOutputParser')
@patch('core.TokenUsageTracker')
def test_chimera_run_debate_parsing_error(mock_token_tracker_cls, mock_output_parser_cls, mock_context_analyzer_cls, mock_persona_manager_cls, mock_gemini_provider_cls):
    """Tests Chimera run_debate when output parsing fails."""
    mock_llm_provider_instance = MagicMock()
    mock_llm_provider_instance.generate.return_value = ("This is not valid JSON.", 100, 50) # Simulate unparseable output
    mock_llm_provider_instance.calculate_usd_cost.return_value = 0.0001
    mock_gemini_provider_cls.return_value = mock_llm_provider_instance

    mock_persona_manager_instance = MagicMock()
    mock_persona_manager_instance.get_adjusted_persona_config.side_effect = lambda name: MagicMock(name=name, system_prompt=f"System prompt for {name}", temperature=0.5, max_tokens=1024)
    mock_persona_manager_instance.persona_sets = {"Software Engineering": ["Code_Architect", "Impartial_Arbitrator"]}
    mock_persona_manager_instance.persona_router.determine_persona_sequence.return_value = ["Code_Architect", "Impartial_Arbitrator"]
    mock_persona_manager_instance.get_token_optimized_persona_sequence.side_effect = lambda seq: seq
    mock_persona_manager_instance._analyze_prompt_complexity.return_value = {"complexity_score": 0.5, "primary_domain": "Software Engineering", "domain_scores": {}, "word_count": 10, "sentence_count": 1}
    mock_persona_manager_instance.record_persona_performance.return_value = None
    mock_persona_manager_cls.return_value = mock_persona_manager_instance

    mock_context_analyzer_instance = MagicMock()
    mock_context_analyzer_instance.find_relevant_files.return_value = []
    mock_context_analyzer_instance.generate_context_summary.return_value = ""
    mock_context_analyzer_cls.return_value = mock_context_analyzer_instance

    mock_output_parser_instance = MagicMock()
    # Mocking output parser to indicate a parsing failure
    mock_output_parser_instance.parse_and_validate.side_effect = [
        CritiqueOutput(CRITIQUE_SUMMARY="LLM_OUTPUT_ERROR", CRITIQUE_POINTS=[], SUGGESTIONS=[], malformed_blocks=[{"type": "JSON_EXTRACTION_FAILED", "message": "Could not parse JSON."}]).model_dump(by_alias=True),
        LLMOutput(COMMIT_MESSAGE="LLM_OUTPUT_ERROR", RATIONALE="No valid JSON data could be extracted or parsed.", CODE_CHANGES=[], malformed_blocks=[{"type": "JSON_EXTRACTION_FAILED", "message": "Could not parse JSON."}]).model_dump(by_alias=True)
    ]
    mock_output_parser_cls.return_value = mock_output_parser_instance

    mock_token_tracker_instance = MagicMock()
    mock_token_tracker_instance.current_usage = 0
    mock_token_tracker_instance.budget = 2000000
    mock_token_tracker_instance.record_usage.side_effect = lambda tokens, persona: setattr(mock_token_tracker_instance, 'current_usage', mock_token_tracker_instance.current_usage + tokens)
    mock_token_tracker_instance.reset.side_effect = lambda: setattr(mock_token_tracker_instance, 'current_usage', 0)
    mock_token_tracker_instance.get_consumption_rate.return_value = 0.0
    mock_token_tracker_cls.return_value = mock_token_tracker_instance

    chimera = Chimera(
        api_key="dummy-api-key",
        model_name="gemini-2.5-flash-lite",
        initial_prompt="Analyze this code.",
        domain="Software Engineering",
        codebase_context={"file.py": "print('hello')"},
        status_callback=MagicMock(),
        rich_console=MagicMock(),
    )

    final_answer, intermediate_steps = chimera.run_debate()

    assert isinstance(final_answer, dict)
    assert "COMMIT_MESSAGE" in final_answer
    assert final_answer["COMMIT_MESSAGE"] == "LLM_OUTPUT_ERROR"
    assert "Code_Architect_Output" in intermediate_steps # First persona still runs
    assert "Final_Synthesis_Output" in intermediate_steps
    mock_llm_provider_instance.generate.assert_called()
    mock_output_parser_instance.parse_and_validate.assert_called()
    mock_token_tracker_instance.record_usage.assert_called()