# tests/test_reasoning.py
from unittest.mock import MagicMock, patch

import pytest

from core import SocraticDebate  # Core reasoning logic is in SocraticDebate
from src.config.settings import ChimeraSettings
from src.conflict_resolution import ConflictResolutionManager
from src.context.context_analyzer import ContextRelevanceAnalyzer
from src.llm_provider import GeminiProvider
from src.persona_manager import PersonaManager
from src.self_improvement.content_validator import ContentAlignmentValidator
from src.self_improvement.metrics_collector import FocusedMetricsCollector
from src.token_tracker import TokenUsageTracker
from src.utils.prompting.prompt_optimizer import PromptOptimizer
from src.utils.reporting.output_parser import LLMOutputParser
from src.llm.orchestrator import LLMOrchestrator # ADD THIS LINE


@pytest.fixture
def mock_socratic_debate_instance():
    """Provides a mocked SocraticDebate instance for testing core reasoning paths.
    This fixture sets up minimal mocks for SocraticDebate's dependencies
    to allow focused testing of its internal logic without full external calls.
    """
    # Mock core dependencies
    mock_settings = MagicMock(spec=ChimeraSettings)
    mock_settings.total_budget = 1000000
    mock_settings.context_token_budget_ratio = 0.25
    mock_settings.debate_token_budget_ratio = 0.65
    mock_settings.synthesis_token_budget_ratio = 0.10
    mock_settings.self_analysis_context_ratio = 0.45
    mock_settings.self_analysis_debate_ratio = 0.30
    mock_settings.self_analysis_synthesis_ratio = 0.25
    mock_settings.max_retries = 2
    mock_settings.max_backoff_seconds = 1
    mock_settings.default_max_input_tokens_per_persona = 4000
    mock_settings.max_tokens_per_persona = {}  # Empty for this mock

    mock_persona_manager = MagicMock(spec=PersonaManager)
    mock_persona_manager.prompt_analyzer.is_self_analysis_prompt.return_value = False
    mock_persona_manager.prompt_analyzer.analyze_complexity.return_value = {
        "complexity_score": 0.5
    }
    mock_persona_manager.persona_router.determine_persona_sequence.return_value = [
        "Visionary_Generator",
        "Impartial_Arbitrator",
    ]
    mock_persona_manager.get_adjusted_persona_config.return_value = MagicMock(
        system_prompt="Mock system prompt", temperature=0.1, max_tokens=1024
    )
    mock_persona_manager.PERSONA_OUTPUT_SCHEMAS = {
        "Visionary_Generator": MagicMock(),
        "Impartial_Arbitrator": MagicMock(),
    }

    mock_token_tracker = MagicMock(spec=TokenUsageTracker)
    mock_token_tracker.current_usage = 0
    mock_token_tracker.budget = 1000000
    mock_token_tracker.record_usage.return_value = None
    mock_token_tracker.get_consumption_rate.return_value = 0.1
    mock_token_tracker.reset.return_value = None

    mock_context_analyzer = MagicMock(spec=ContextRelevanceAnalyzer)
    mock_context_analyzer.file_embeddings = {}
    mock_context_analyzer.find_relevant_files.return_value = []
    mock_context_analyzer.generate_context_summary.return_value = "No context summary."
    mock_context_analyzer.raw_file_contents = {}
    mock_context_analyzer.compute_file_embeddings.return_value = {}

    mock_llm_provider = MagicMock(spec=GeminiProvider)
    mock_llm_provider.tokenizer.count_tokens.return_value = 10  # Mock token count
    mock_llm_provider.tokenizer.max_output_tokens = 8192
    mock_llm_provider.calculate_usd_cost.return_value = 0.001
    mock_llm_provider.generate.return_value = (
        '{"general_output": "Mock LLM output"}',
        10,
        10,
        False,
    )

    mock_output_parser = MagicMock(spec=LLMOutputParser)
    mock_output_parser.parse_and_validate.return_value = {
        "general_output": "Parsed mock output",
        "malformed_blocks": [],
    }
    mock_output_parser._create_fallback_output.return_value = {
        "general_output": "Fallback mock output",
        "malformed_blocks": [],
    }

    mock_conflict_manager = MagicMock(spec=ConflictResolutionManager)
    mock_conflict_manager.resolve_conflict.return_value = None

    mock_metrics_collector = MagicMock(spec=FocusedMetricsCollector)
    mock_metrics_collector.collect_all_metrics.return_value = {}
    mock_metrics_collector.analyze_historical_effectiveness.return_value = {}
    mock_metrics_collector.record_self_improvement_suggestion_outcome.return_value = (
        None
    )
    mock_metrics_collector.file_analysis_cache = {}

    mock_prompt_optimizer = MagicMock(spec=PromptOptimizer)
    mock_prompt_optimizer.optimize_prompt.side_effect = lambda p, pn, mot, sm, is_self_analysis_prompt: p # ADD is_self_analysis_prompt
    mock_prompt_optimizer.optimize_debate_history.side_effect = lambda h, mt: h
    mock_prompt_optimizer.tokenizer = mock_llm_provider.tokenizer

    mock_content_validator = MagicMock(spec=ContentAlignmentValidator)
    mock_content_validator.validate.return_value = (True, "Content aligned.", {})

    mock_summarizer_pipeline = MagicMock()
    mock_summarizer_pipeline.return_value = [{"summary_text": "Mock summary."}]
    mock_summarizer_pipeline.tokenizer.model_max_length = 1024

    # ADD THIS BLOCK
    mock_llm_orchestrator = MagicMock(spec=LLMOrchestrator)
    mock_llm_orchestrator.call_llm.return_value = {
        "text": '{"general_output": "Mock LLM output from orchestrator"}',
        "usage": {"prompt_tokens": 10, "completion_tokens": 10, "total_tokens": 20},
        "is_truncated": False,
    }
    mock_llm_orchestrator.close.return_value = None
    # END ADD THIS BLOCK

    # Patch the SocraticDebate constructor's dependencies to use our mocks
    with (
        patch("core.GeminiProvider", return_value=mock_llm_provider),
        patch("core.PersonaManager", return_value=mock_persona_manager),
        patch("core.TokenUsageTracker", return_value=mock_token_tracker),
        patch("core.ContextRelevanceAnalyzer", return_value=mock_context_analyzer),
        patch("core.LLMOutputParser", return_value=mock_output_parser),
        patch("core.ConflictResolutionManager", return_value=mock_conflict_manager),
        patch("core.FocusedMetricsCollector", return_value=mock_metrics_collector),
        patch("core.PromptOptimizer", return_value=mock_prompt_optimizer),
        patch("core.ContentAlignmentValidator", return_value=mock_content_validator),
        patch("core.LLMOrchestrator", return_value=mock_llm_orchestrator), # ADD THIS LINE
    ):
        debate_instance = SocraticDebate(
            initial_prompt="Test prompt",
            api_key="AIza_mock-key-for-testing-purposes-1234567890",  # Valid-looking API key
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
            summarizer_pipeline_instance=mock_summarizer_pipeline,
            llm_orchestrator=mock_llm_orchestrator, # ADD THIS LINE
        )
        # Ensure the instance's internal references are set to our mocks
        debate_instance.llm_provider = mock_llm_provider
        debate_instance.output_parser = mock_output_parser
        debate_instance.conflict_manager = mock_conflict_manager
        debate_instance.metrics_collector = mock_metrics_collector
        debate_instance.prompt_optimizer = mock_prompt_optimizer
        debate_instance.content_validator = mock_content_validator
        debate_instance.llm_orchestrator = mock_llm_orchestrator # ADD THIS LINE

        yield debate_instance


def test_complex_prompt_handling(mock_socratic_debate_instance):
    """Tests the SocraticDebate's ability to handle complex, multi-step prompts.
    This test should verify that the debate orchestrator correctly sequences
    persona interactions and integrates their outputs for a coherent final answer.
    """
    # Example: Mock the persona sequence to be more complex
    mock_socratic_debate_instance.persona_manager.persona_router.determine_persona_sequence.return_value = [
        "Visionary_Generator",
        "Skeptical_Generator",
        "Constructive_Critic",
        "Impartial_Arbitrator",
    ]

    # Mock LLM responses for each step in the complex sequence
    mock_socratic_debate_instance.llm_orchestrator.call_llm.side_effect = [ # MODIFIED: Use orchestrator
        {
            "text": '{"general_output": "Initial idea for complex prompt"}',
            "usage": {"prompt_tokens": 10, "completion_tokens": 10, "total_tokens": 20},
            "is_truncated": False,
        },
        {
            "text": '{"general_output": "Critique of initial idea"}',
            "usage": {"prompt_tokens": 10, "completion_tokens": 10, "total_tokens": 20},
            "is_truncated": False,
        },
        {
            "text": '{"CRITIQUE_SUMMARY": "Constructive feedback", "CRITIQUE_POINTS": [], "SUGGESTIONS": []}',
            "usage": {"prompt_tokens": 10, "completion_tokens": 10, "total_tokens": 20},
            "is_truncated": False,
        },
        {
            "text": '{"general_output": "Final synthesis for complex prompt"}',
            "usage": {"prompt_tokens": 10, "completion_tokens": 10, "total_tokens": 20},
            "is_truncated": False,
        },
    ]
    mock_socratic_debate_instance.output_parser.parse_and_validate.side_effect = [
        {"general_output": "Initial idea for complex prompt", "malformed_blocks": []},
        {"general_output": "Critique of initial idea", "malformed_blocks": []},
        {
            "CRITIQUE_SUMMARY": "Constructive feedback",
            "CRITIQUE_POINTS": [],
            "SUGGESTIONS": [],
            "malformed_blocks": [],
        },
        {
            "general_output": "Final synthesis for complex prompt",
            "malformed_blocks": [],
        },
    ]

    mock_socratic_debate_instance.initial_prompt = (
        "Design a new feature, then critique its scalability and propose improvements."
    )
    final_answer, intermediate_steps = mock_socratic_debate_instance.run_debate()

    assert "Final synthesis for complex prompt" in final_answer.get(
        "general_output", ""
    )
    assert "Visionary_Generator_Output" in intermediate_steps
    assert "Skeptical_Generator_Output" in intermediate_steps
    assert "Constructive_Critic_Output" in intermediate_steps
    assert "Impartial_Arbitrator_Output" in intermediate_steps
    assert mock_socratic_debate_instance.llm_orchestrator.call_llm.call_count == 4 # MODIFIED: Check orchestrator call count


def test_error_case_handling(mock_socratic_debate_instance):
    """Tests the SocraticDebate's robustness in handling various error conditions,
    such as LLM response validation failures and unexpected exceptions.
    """
    # Simulate an LLM response that consistently fails schema validation
    mock_socratic_debate_instance.llm_orchestrator.call_llm.side_effect = [ # MODIFIED: Use orchestrator
        {
            "text": '{"invalid_json": "malformed"}',
            "usage": {"prompt_tokens": 10, "completion_tokens": 10, "total_tokens": 20},
            "is_truncated": False,
        },  # First attempt
        {
            "text": '{"invalid_json": "malformed_retry"}',
            "usage": {"prompt_tokens": 10, "completion_tokens": 10, "total_tokens": 20},
            "is_truncated": False,
        },  # Second attempt (retry)
        {
            "text": '{"invalid_json": "malformed_final"}',
            "usage": {"prompt_tokens": 10, "completion_tokens": 10, "total_tokens": 20},
            "is_truncated": False,
        },  # Third attempt (max retries)
        {
            "text": '{"general_output": "Fallback after errors"}',
            "usage": {"prompt_tokens": 10, "completion_tokens": 10, "total_tokens": 20},
            "is_truncated": False,
        },  # Final synthesis
    ]
    # Mock the parser to always return malformed blocks for the persona
    mock_socratic_debate_instance.output_parser.parse_and_validate.side_effect = [
        {
            "invalid_json": "malformed",
            "malformed_blocks": [{"type": "SCHEMA_VALIDATION_ERROR"}],
        },
        {
            "invalid_json": "malformed_retry",
            "malformed_blocks": [{"type": "SCHEMA_VALIDATION_ERROR"}],
        },
        {
            "invalid_json": "malformed_final",
            "malformed_blocks": [{"type": "SCHEMA_VALIDATION_ERROR"}],
        },
        {"general_output": "Fallback after errors", "malformed_blocks": []},
    ]

    mock_socratic_debate_instance.initial_prompt = "Generate a perfect JSON output."
    final_answer, intermediate_steps = mock_socratic_debate_instance.run_debate()

    # Expect the final answer to reflect the fallback mechanism
    assert "Fallback after errors" in final_answer.get("general_output", "")
    # Verify that schema validation errors were recorded
    assert any(
        block["type"] == "SCHEMA_VALIDATION_ERROR"
        for block in intermediate_steps.get("malformed_blocks", [])
    )
    # Verify that the LLM was called multiple times due to retries
    assert (
        mock_socratic_debate_instance.llm_orchestrator.call_llm.call_count >= 3 # MODIFIED: Check orchestrator call count
    )  # At least 3 calls for the failing persona + 1 for synthesis


def test_edge_cases(mock_socratic_debate_instance):
    """Tests various edge cases in the SocraticDebate process,
    such as empty prompts, minimal context, or unusual persona sequences.
    """
    # Edge Case 1: Empty initial prompt
    mock_socratic_debate_instance.initial_prompt = ""
    final_answer, intermediate_steps = mock_socratic_debate_instance.run_debate()
    assert "Fallback mock output" in final_answer.get(
        "general_output", ""
    )  # Should return fallback due to empty prompt

    # Edge Case 2: No relevant files found for context
    mock_socratic_debate_instance.initial_prompt = "Analyze a simple problem."
    mock_socratic_debate_instance.context_analyzer.find_relevant_files.return_value = []
    final_answer, intermediate_steps = mock_socratic_debate_instance.run_debate()
    assert "Parsed mock output" in final_answer.get(
        "general_output", ""
    )  # Should still proceed without context

    # Edge Case 3: Persona sequence with only one persona
    mock_socratic_debate_instance.persona_manager.persona_router.determine_persona_sequence.return_value = [
        "Impartial_Arbitrator"
    ]
    mock_socratic_debate_instance.initial_prompt = "Synthesize this."
    final_answer, intermediate_steps = mock_socratic_debate_instance.run_debate()
    assert "Parsed mock output" in final_answer.get(
        "general_output", ""
    )  # Should still work with single persona
    assert (
        mock_socratic_debate_instance.llm_orchestrator.call_llm.call_count >= 1 # MODIFIED: Check orchestrator call count
    )  # At least one call for the arbitrator