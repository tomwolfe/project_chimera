import pytest
from unittest.mock import MagicMock, patch

# Assuming core.py is in the project root
from core import SocraticDebate
from src.persona_manager import PersonaManager
from src.llm_provider import GeminiProvider
from src.token_tracker import TokenUsageTracker
from src.config.settings import ChimeraSettings
from src.context.context_analyzer import ContextRelevanceAnalyzer
from src.models import PersonaConfig, GeneralOutput, CritiqueOutput, ConflictReport


@pytest.fixture
def mock_llm_provider():
    provider = MagicMock(spec=GeminiProvider)
    provider.tokenizer = MagicMock()
    provider.tokenizer.count_tokens.side_effect = lambda text: len(text) // 4
    provider.tokenizer.max_output_tokens = 8192
    provider.calculate_usd_cost.return_value = 0.001
    # Default LLM response for a persona turn
    provider.generate.return_value = (
        '{"general_output": "Mock response"}',
        100,
        50,
        False,
    )
    return provider


@pytest.fixture
def mock_token_tracker():
    tracker = MagicMock(spec=TokenUsageTracker)
    tracker.current_usage = 0
    tracker.budget = 1000000
    tracker.record_usage.side_effect = lambda tokens, persona=None: setattr(
        tracker, "current_usage", tracker.current_usage + tokens
    )
    tracker.get_consumption_rate.return_value = 0.1
    tracker.reset.return_value = None
    return tracker


@pytest.fixture
def mock_settings():
    settings = MagicMock(spec=ChimeraSettings)
    settings.total_budget = 1000000
    settings.context_token_budget_ratio = 0.25
    settings.debate_token_budget_ratio = 0.65
    settings.synthesis_token_budget_ratio = 0.10
    settings.self_analysis_context_ratio = 0.45
    settings.self_analysis_debate_ratio = 0.30
    settings.self_analysis_synthesis_ratio = 0.25
    settings.max_retries = 2
    settings.max_backoff_seconds = 1
    settings.default_max_input_tokens_per_persona = 4000
    settings.max_tokens_per_persona = {
        "Constructive_Critic": 3500,
        "Devils_Advocate": 4096,
        "Conflict_Resolution_Manager": 4096,  # New persona
        "Impartial_Arbitrator": 4096,
        "Visionary_Generator": 1024,
        "Skeptical_Generator": 1024,
    }
    return settings


@pytest.fixture
def mock_persona_manager(mock_token_tracker, mock_settings):
    pm = MagicMock(spec=PersonaManager)
    pm.all_personas = {
        "Constructive_Critic": PersonaConfig(
            name="Constructive_Critic",
            system_prompt="Critic",
            temperature=0.15,
            max_tokens=8192,
        ),
        "Devils_Advocate": PersonaConfig(
            name="Devils_Advocate",
            system_prompt="Devils Advocate",
            temperature=0.1,
            max_tokens=4096,
        ),
        "Conflict_Resolution_Manager": PersonaConfig(
            name="Conflict_Resolution_Manager",
            system_prompt="Mediator",
            temperature=0.1,
            max_tokens=4096,
        ),
        "Impartial_Arbitrator": PersonaConfig(
            name="Impartial_Arbitrator",
            system_prompt="Arbitrator",
            temperature=0.1,
            max_tokens=4096,
        ),
        "Visionary_Generator": PersonaConfig(
            name="Visionary_Generator",
            system_prompt="Visionary",
            temperature=0.7,
            max_tokens=1024,
        ),
        "Skeptical_Generator": PersonaConfig(
            name="Skeptical_Generator",
            system_prompt="Skeptical",
            temperature=0.3,
            max_tokens=1024,
        ),
    }
    pm.persona_sets = {
        "General": [
            "Visionary_Generator",
            "Skeptical_Generator",
            "Constructive_Critic",
            "Devils_Advocate",
            "Conflict_Resolution_Manager",
            "Impartial_Arbitrator",
        ]
    }
    pm.get_adjusted_persona_config.side_effect = lambda name: pm.all_personas.get(
        name.replace("_TRUNCATED", ""),
        MagicMock(
            spec=PersonaConfig,
            name=name,
            system_prompt="Fallback",
            temperature=0.5,
            max_tokens=1024,
        ),
    )
    pm.prompt_analyzer = MagicMock()
    pm.prompt_analyzer.is_self_analysis_prompt.return_value = False
    pm.persona_router = MagicMock()
    pm.persona_router.determine_persona_sequence.return_value = [
        "Constructive_Critic",
        "Devils_Advocate",
        "Conflict_Resolution_Manager",
        "Impartial_Arbitrator",
    ]
    pm.token_tracker = mock_token_tracker
    pm.settings = mock_settings
    pm.PERSONA_OUTPUT_SCHEMAS = {
        "Constructive_Critic": CritiqueOutput,
        "Devils_Advocate": ConflictReport,
        "Conflict_Resolution_Manager": GeneralOutput,  # ConflictReport is for Devils_Advocate, Manager gives general output
        "Impartial_Arbitrator": GeneralOutput,
        "Visionary_Generator": GeneralOutput,
        "Skeptical_Generator": GeneralOutput,
    }
    return pm


@pytest.fixture
def mock_context_analyzer():
    analyzer = MagicMock(spec=ContextRelevanceAnalyzer)
    analyzer.file_embeddings = {
        "mock_file.py": [0.1, 0.2]
    }  # Ensure embeddings are present
    analyzer.find_relevant_files.return_value = []
    analyzer.generate_context_summary.return_value = "No context summary."
    analyzer.compute_file_embeddings.return_value = {}
    return analyzer


@pytest.fixture
def mock_output_parser():
    parser = MagicMock()
    parser.parse_and_validate.side_effect = [
        {
            "CRITIQUE_SUMMARY": "Mock critique",
            "CRITIQUE_POINTS": [],
            "SUGGESTIONS": [],
            "malformed_blocks": [],
        },
        {
            "conflict_type": "NO_CONFLICT",
            "summary": "No conflict reported",
            "involved_personas": [],
            "conflicting_outputs_snippet": "",
            "proposed_resolution_paths": [],
            "conflict_found": False,
            "malformed_blocks": [],
        },
        {"general_output": "Mock resolution", "malformed_blocks": []},
        {"general_output": "Mock final answer", "malformed_blocks": []},
    ]
    return parser


@pytest.fixture
def mock_conflict_manager():
    manager = MagicMock()
    manager.resolve_conflict.return_value = {
        "resolution_strategy": "mock_resolution",
        "resolved_output": {
            "general_output": "Mock resolved output from conflict",
            "malformed_blocks": [],
        },
        "resolution_summary": "Mock conflict resolved.",
        "malformed_blocks": [],
    }
    return manager


@pytest.fixture
def socratic_debate_instance(
    mock_llm_provider,
    mock_persona_manager,
    mock_token_tracker,
    mock_settings,
    mock_context_analyzer,
    mock_output_parser,
    mock_conflict_manager,
):
    with (
        patch("core.LLMOutputParser", return_value=mock_output_parser),
        patch("core.ConflictResolutionManager", return_value=mock_conflict_manager),
    ):
        debate = SocraticDebate(
            initial_prompt="Test prompt",
            api_key="mock_api_key",
            model_name="gemini-2.5-flash-lite",
            domain="General",
            persona_manager=mock_persona_manager,
            structured_codebase_context={},
            raw_file_contents={},
            settings=mock_settings,
            context_analyzer=mock_context_analyzer,
            token_tracker=mock_token_tracker,
            status_callback=MagicMock(),
            rich_console=MagicMock(),
        )
        # Manually set the mocks that SocraticDebate might re-initialize or use directly
        debate.llm_provider = mock_llm_provider
        debate.output_parser = mock_output_parser
        debate.conflict_manager = mock_conflict_manager
        return debate


def test_socratic_debate_initialization(socratic_debate_instance):
    assert socratic_debate_instance is not None
    assert socratic_debate_instance.initial_prompt == "Test prompt"
    assert socratic_debate_instance.persona_manager is not None
    assert socratic_debate_instance.llm_provider is not None


def test_socratic_debate_run_flow_simple(
    socratic_debate_instance, mock_llm_provider, mock_output_parser
):
    # Mock LLM responses for the sequence: Critic, Devils_Advocate, Conflict_Resolution_Manager, Arbitrator
    mock_llm_provider.generate.side_effect = [
        # Constructive_Critic
        (
            '{"CRITIQUE_SUMMARY": "Critique output", "CRITIQUE_POINTS": [], "SUGGESTIONS": []}',
            100,
            50,
            False,
        ),
        # Devils_Advocate (no conflict)
        (
            '{"conflict_type": "NO_CONFLICT", "summary": "No conflict reported", "involved_personas": [], "conflicting_outputs_snippet": "", "proposed_resolution_paths": [], "conflict_found": false}',
            100,
            50,
            False,
        ),
        # Conflict_Resolution_Manager (not called if no conflict)
        # Impartial_Arbitrator
        ('{"general_output": "Final synthesized answer"}', 100, 50, False),
    ]
    # Reset mock_output_parser.parse_and_validate for this specific test
    mock_output_parser.parse_and_validate.side_effect = [
        {
            "CRITIQUE_SUMMARY": "Critique output",
            "CRITIQUE_POINTS": [],
            "SUGGESTIONS": [],
            "malformed_blocks": [],
        },
        {
            "conflict_type": "NO_CONFLICT",
            "summary": "No conflict reported",
            "involved_personas": [],
            "conflicting_outputs_snippet": "",
            "proposed_resolution_paths": [],
            "conflict_found": False,
            "malformed_blocks": [],
        },
        {"general_output": "Final synthesized answer", "malformed_blocks": []},
    ]

    final_answer, intermediate_steps = socratic_debate_instance.run_debate()

    assert final_answer is not None
    assert "Final synthesized answer" in final_answer.get("general_output", "")
    assert len(intermediate_steps.get("Debate_History", [])) > 0
    assert "Constructive_Critic" in [
        t["persona"] for t in intermediate_steps["Debate_History"]
    ]
    assert "Devils_Advocate" in [
        t["persona"] for t in intermediate_steps["Debate_History"]
    ]
    assert "Impartial_Arbitrator" in [
        t["persona"] for t in intermediate_steps["Debate_History"]
    ]
    assert "Conflict_Resolution_Manager" not in [
        t["persona"] for t in intermediate_steps["Debate_History"]
    ]  # Should not be called if no conflict


def test_socratic_debate_conflict_resolution_flow(
    socratic_debate_instance,
    mock_llm_provider,
    mock_output_parser,
    mock_conflict_manager,
):
    # Mock LLM responses for the sequence: Critic (problematic), Devils_Advocate (reports conflict), Conflict_Resolution_Manager (resolves), Arbitrator
    mock_llm_provider.generate.side_effect = [
        # Constructive_Critic (problematic output)
        (
            '{"CRITIQUE_SUMMARY": "Problematic critique", "malformed_blocks": [{"type": "SCHEMA_VALIDATION_ERROR"}]}',
            100,
            50,
            False,
        ),
        # Devils_Advocate (reports conflict)
        (
            '{"conflict_type": "LOGICAL_INCONSISTENCY", "summary": "Conflict detected", "involved_personas": ["Constructive_Critic"], "conflicting_outputs_snippet": "...", "proposed_resolution_paths": ["Mediate"], "conflict_found": true}',
            100,
            50,
            False,
        ),
        # Impartial_Arbitrator (receives resolved output from mock_conflict_manager)
        (
            '{"general_output": "Final answer after conflict resolution"}',
            100,
            50,
            False,
        ),
    ]
    # Mock parser to return problematic for critic, then conflict report, then final answer
    mock_output_parser.parse_and_validate.side_effect = [
        {
            "CRITIQUE_SUMMARY": "Problematic critique",
            "CRITIQUE_POINTS": [],
            "SUGGESTIONS": [],
            "malformed_blocks": [{"type": "SCHEMA_VALIDATION_ERROR"}],
        },
        {
            "conflict_type": "LOGICAL_INCONSISTENCY",
            "summary": "Conflict detected",
            "involved_personas": ["Constructive_Critic"],
            "conflicting_outputs_snippet": "...",
            "proposed_resolution_paths": ["Mediate"],
            "conflict_found": True,
            "malformed_blocks": [],
        },
        {
            "general_output": "Final answer after conflict resolution",
            "malformed_blocks": [],
        },
    ]

    final_answer, intermediate_steps = socratic_debate_instance.run_debate()

    assert final_answer is not None
    assert "Final answer after conflict resolution" in final_answer.get(
        "general_output", ""
    )
    assert "Conflict_Resolution_Attempt" in intermediate_steps
    assert (
        intermediate_steps["Conflict_Resolution_Attempt"]["conflict_resolved"] is True
    )
    assert (
        "Mock conflict resolved."
        in intermediate_steps["Conflict_Resolution_Attempt"]["resolution_summary"]
    )
    assert "Conflict_Resolution_Manager" in [
        t["persona"] for t in intermediate_steps["Debate_History"]
    ]  # Should be called
