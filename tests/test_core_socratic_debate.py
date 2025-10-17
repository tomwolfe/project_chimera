# tests/test_core_socratic_debate.py
from unittest.mock import MagicMock, patch

import pytest

from core import SocraticDebate
from src.config.settings import ChimeraSettings  # Import ChimeraSettings
from src.context.context_analyzer import ContextRelevanceAnalyzer

# Import FocusedMetricsCollector
from src.exceptions import TokenBudgetExceededError
from src.llm.orchestrator import LLMOrchestrator  # ADD THIS LINE

# Assuming these imports are correct based on your project structure
# Adjust if your core.py or llm_provider.py are in different locations
from src.llm_provider import GeminiProvider

# Import ContextRelevanceAnalyzer
# Import specific exceptions
from src.models import (
    ConflictReport,  # Added for mock_output_parser fixture
    CritiqueOutput,
    GeneralOutput,
    PersonaConfig,  # Import PersonaConfig
    SelfImprovementAnalysisOutputV1,
)

# Assuming GeminiProvider is in src/llm_provider.py
from src.persona_manager import PersonaManager  # Import PersonaManager
from src.self_improvement.metrics_collector import FocusedMetricsCollector
from src.token_tracker import TokenUsageTracker  # Import TokenUsageTracker
from src.utils.reporting.output_parser import LLMOutputParser  # Import LLMOutputParser

# --- CONSTANTS FOR PLR2004 FIXES ---
EXPECTED_DEBATE_TURNS_3 = 3
EXPECTED_RETRY_COUNT_2 = 2
# --- END CONSTANTS ---


@pytest.fixture
def mock_summarizer_pipeline():
    """Provides a mock Hugging Face summarization pipeline."""
    mock_pipeline = MagicMock()
    mock_pipeline.return_value = [{"summary_text": "Mock summary."}]
    mock_pipeline.tokenizer.model_max_length = 1024  # Simulate distilbart's max input
    return mock_pipeline


@pytest.fixture
def mock_settings():
    """Provides a mock ChimeraSettings instance."""
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
        "Self_Improvement_Analyst": 4000,
        "Context_Aware_Assistant": 3072,
        "Impartial_Arbitrator": 4096,
        "Devils_Advocate": 4096,
        "Visionary_Generator": 1024,
        "Skeptical_Generator": 1024,
    }
    return settings


@pytest.fixture
def mock_token_tracker():
    """Provides a mock TokenUsageTracker instance."""
    tracker = MagicMock(spec=TokenUsageTracker)
    tracker.current_usage = 0
    tracker.budget = 1000000
    tracker.record_usage.side_effect = (
        lambda prompt_tokens,
        completion_tokens,
        persona=None,
        is_successful_turn=True: setattr(
            tracker,
            "current_usage",
            tracker.current_usage + prompt_tokens + completion_tokens,
        )
    )
    tracker.get_consumption_rate.return_value = 0.1
    tracker.reset.return_value = None
    return tracker


@pytest.fixture
def mock_context_analyzer():
    """Provides a mock ContextRelevanceAnalyzer instance."""
    analyzer = MagicMock(spec=ContextRelevanceAnalyzer)
    analyzer.file_embeddings = {"file1.py": [0.1, 0.2]}
    analyzer.find_relevant_files.return_value = [("file1.py", 0.9)]
    analyzer.generate_context_summary.return_value = "Summary of relevant files."
    analyzer.compute_file_embeddings.return_value = {"file1.py": [0.1, 0.2]}
    return analyzer


@pytest.fixture
def mock_persona_manager(mock_token_tracker, mock_settings):
    """Provides a mock PersonaManager instance."""
    pm = MagicMock(spec=PersonaManager)
    pm.all_personas = {
        "Visionary_Generator": PersonaConfig(
            name="Visionary_Generator",
            system_prompt_template="Visionary",  # Changed to template
            temperature=0.7,
            max_tokens=1024,
            description="Visionary persona",  # Added description
        ),
        "Skeptical_Generator": PersonaConfig(
            name="Skeptical_Generator",
            system_prompt_template="Skeptical",  # Changed to template
            temperature=0.3,
            max_tokens=1024,
            description="Skeptical persona",  # Added description
        ),
        "Constructive_Critic": PersonaConfig(
            name="Constructive_Critic",
            system_prompt_template="Critic",  # Changed to template
            temperature=0.15,
            max_tokens=8192,
            description="Constructive Critic persona",  # Added description
        ),
        "Impartial_Arbitrator": PersonaConfig(
            name="Impartial_Arbitrator",
            system_prompt_template="Arbitrator",  # Changed to template
            temperature=0.1,
            max_tokens=4096,
            description="Impartial Arbitrator persona",  # Added description
        ),
        "Devils_Advocate": PersonaConfig(
            name="Devils_Advocate",
            system_prompt_template="Devils Advocate",  # Changed to template
            temperature=0.1,
            max_tokens=4096,
            description="Devils Advocate persona",  # Added description
        ),
        "Self_Improvement_Analyst": PersonaConfig(
            name="Self_Improvement_Analyst",
            system_prompt_template="Self-Improvement Analyst",  # Changed to template
            temperature=0.1,
            max_tokens=8192,
            description="Self-Improvement Analyst persona",  # Added description
        ),
        "Context_Aware_Assistant": PersonaConfig(
            name="Context_Aware_Assistant",
            system_prompt_template="Context Assistant",  # Changed to template
            temperature=0.1,
            max_tokens=3072,
            description="Context Aware Assistant persona",  # Added description
        ),
    }
    pm.persona_sets = {
        "General": [
            "Visionary_Generator",
            "Skeptical_Generator",
            "Impartial_Arbitrator",
        ],
        "Self-Improvement": [
            "Self_Improvement_Analyst",
            "Code_Architect",
            "Security_Auditor",
            "DevOps_Engineer",
            "Test_Engineer",
            "Constructive_Critic",
            "Devils_Advocate",
            "Impartial_Arbitrator",
        ],
    }
    pm.get_adjusted_persona_config.side_effect = lambda name: pm.all_personas.get(
        name.replace("_TRUNCATED", ""),
        MagicMock(
            spec=PersonaConfig,
            name=name,  # Ensure mock PersonaConfig has a name attribute
            system_prompt_template="Fallback",  # Changed to template
            temperature=0.5,
            max_tokens=1024,
            description="Fallback persona",  # Added description
        ),
    )
    pm.prompt_analyzer = MagicMock()
    pm.prompt_analyzer.is_self_analysis_prompt.return_value = False
    pm.prompt_analyzer.analyze_complexity.return_value = {"complexity_score": 0.5}
    pm.persona_router = MagicMock()
    pm.persona_router.determine_persona_sequence.return_value = [
        "Visionary_Generator",
        "Skeptical_Generator",
        "Impartial_Arbitrator",
    ]
    pm.token_tracker = mock_token_tracker
    pm.settings = mock_settings
    pm.PERSONA_OUTPUT_SCHEMAS = {
        "Impartial_Arbitrator": GeneralOutput,
        "Context_Aware_Assistant": GeneralOutput,
        "Constructive_Critic": CritiqueOutput,
        "Self_Improvement_Analyst": SelfImprovementAnalysisOutputV1,  # Corrected schema for Self_Improvement_Analyst
        "Devils_Advocate": GeneralOutput,
        "Visionary_Generator": GeneralOutput,
        "Skeptical_Generator": GeneralOutput,
    }
    return pm


@pytest.fixture
def mock_gemini_provider(mock_settings):
    """Provides a mock GeminiProvider instance."""
    provider = MagicMock(spec=GeminiProvider)
    provider.tokenizer = MagicMock()
    provider.tokenizer.count_tokens.side_effect = lambda text: max(
        1, len(text) // 4
    )  # Ensure at least 1 token
    provider.tokenizer.max_output_tokens = 8192
    provider.calculate_usd_cost.return_value = 0.001
    provider.generate.return_value = (
        "{}",
        100,
        50,
        False,
    )  # Default successful response
    provider.settings = mock_settings
    return provider


@pytest.fixture
def mock_output_parser():
    """Provides a mock LLMOutputParser instance."""

    parser = MagicMock(spec=LLMOutputParser)
    parser.parse_and_validate.return_value = {
        "general_output": "Parsed output",
        "malformed_blocks": [],
    }
    parser._create_fallback_output.return_value = {
        "general_output": "Fallback output",
        "malformed_blocks": [{"type": "FALLBACK_TRIGGERED"}],
    }
    # Mock the _get_schema_class_from_name method to return actual schema classes
    # which have the model_json_schema method that returns a serializable dict
    parser._get_schema_class_from_name.side_effect = lambda schema_name: {
        "GeneralOutput": GeneralOutput,
        "CritiqueOutput": CritiqueOutput,
        "ConflictReport": ConflictReport,
        "SelfImprovementAnalysisOutputV1": SelfImprovementAnalysisOutputV1,
    }.get(schema_name, GeneralOutput)
    return parser


@pytest.fixture
def mock_conflict_manager():
    """Provides a mock ConflictResolutionManager instance."""
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
def mock_metrics_collector():
    """Provides a mock FocusedMetricsCollector instance."""
    collector = MagicMock(spec=FocusedMetricsCollector)
    # MODIFIED: Mock the full structure of the return value to prevent KeyError
    collector.collect_all_metrics.return_value = {
        "code_quality": {
            "ruff_issues_count": 0,
            "complexity_metrics": {},
            "code_smells_count": 0,
            "detailed_issues": [],
            "ruff_violations": [],
        },
        "security": {"bandit_issues_count": 0, "ast_security_issues_count": 0},
        "performance_efficiency": {
            "token_usage_stats": {},
            "debate_efficiency_summary": {},
            "potential_bottlenecks_count": 0,
        },
        "robustness": {
            "schema_validation_failures_count": 0,
            "unresolved_conflict_present": False,
            "conflict_resolution_attempted": False,
        },
        "maintainability": {"test_coverage_summary": {}},
        "reasoning_quality": {
            "argument_strength_score": 0.0,
            "debate_effectiveness": 0.0,
            "conflict_resolution_quality": 0.0,
            "80_20_adherence_score": 0.0,
            "reasoning_depth": 0,
            "critical_thinking_indicators": {},
            "self_improvement_suggestion_success_rate_historical": 0.0,
            "schema_validation_failures_historical": {},
        },
        "historical_analysis": {
            "total_attempts": 0,
            "success_rate": 0.0,
            "top_performing_areas": [],
            "common_failure_modes": {},
            "historical_total_suggestions_processed": 0,
            "historical_successful_suggestions": 0,
            "historical_schema_validation_failures": {},
        },
        "configuration_analysis": {
            "ci_workflow": {},
            "pre_commit_hooks": [],
            "pyproject_toml": {},
        },
        "deployment_robustness": {},
    }
    collector.analyze_historical_effectiveness.return_value = {
        "total_attempts": 0,
        "success_rate": 0.0,
    }
    collector.record_self_improvement_suggestion_outcome.return_value = None
    collector.file_analysis_cache = {}  # Ensure this attribute exists
    return collector


# ADD THIS FIXTURE
@pytest.fixture
def mock_llm_orchestrator():
    """Provides a mock LLMOrchestrator instance."""
    orchestrator = MagicMock(spec=LLMOrchestrator)
    orchestrator.call_llm.return_value = {
        "text": '{"general_output": "Mock LLM output from orchestrator"}',
        "usage": {"prompt_tokens": 10, "completion_tokens": 10, "total_tokens": 20},
        "is_truncated": False,
    }
    orchestrator.close.return_value = None
    return orchestrator


# END ADD THIS FIXTURE


@pytest.fixture
def socratic_debate_instance(  # noqa: F811
    mock_gemini_provider,
    mock_persona_manager,
    mock_token_tracker,
    mock_context_analyzer,
    mock_settings,
    mock_output_parser,
    mock_conflict_manager,
    mock_metrics_collector,
    mock_summarizer_pipeline,  # NEW: Add mock_summarizer_pipeline
    mock_llm_orchestrator,  # ADD THIS LINE
):
    """Provides a SocraticDebate instance with mocked dependencies."""
    with (
        patch("core.GeminiProvider", return_value=mock_gemini_provider),
        patch("core.PersonaManager", return_value=mock_persona_manager),
        patch("core.TokenUsageTracker", return_value=mock_token_tracker),
        patch("core.ContextRelevanceAnalyzer", return_value=mock_context_analyzer),
        patch("core.LLMOutputParser", return_value=mock_output_parser),
        patch("core.ConflictResolutionManager", return_value=mock_conflict_manager),
        patch("core.FocusedMetricsCollector", return_value=mock_metrics_collector),
        patch(
            "core.PromptOptimizer"
        ) as MockPromptOptimizer,  # MODIFIED: Patch the class directly
        # NEW: Patch additional injected dependencies
        patch("core.CritiqueEngine") as MockCritiqueEngine,
        patch("core.ImprovementApplicator") as MockImprovementApplicator,
        patch("core.ContentAlignmentValidator") as MockContentAlignmentValidator,
        patch(
            "core.LLMOrchestrator", return_value=mock_llm_orchestrator
        ),  # ADD THIS LINE
    ):
        # Mock the PromptOptimizer constructor to return a mock instance
        MockPromptOptimizer.return_value = (
            MagicMock()
        )  # MODIFIED: Use MockPromptOptimizer
        MockPromptOptimizer.return_value.optimize_prompt.side_effect = (
            lambda user_prompt_text,
            persona_config,
            max_output_tokens_for_turn,
            system_message_for_token_count,
            is_self_analysis_prompt: user_prompt_text
        )  # Default to no-op
        MockPromptOptimizer.return_value.optimize_debate_history.side_effect = (
            lambda h, mt: h
        )  # Default to no-op
        MockPromptOptimizer.return_value.tokenizer = (
            mock_gemini_provider.tokenizer
        )  # Ensure tokenizer is set
        MockPromptOptimizer.return_value.generate_prompt.side_effect = (
            lambda template_name, context: template_name
        )  # Mock generate_prompt

        # NEW: Mock the injected dependencies
        mock_critique_engine = MockCritiqueEngine.return_value
        mock_improvement_applicator = MockImprovementApplicator.return_value
        mock_content_alignment_validator = MockContentAlignmentValidator.return_value
        mock_content_alignment_validator.validate.return_value = (
            True,
            "Content aligned.",
            {},
        )

        debate = SocraticDebate(
            initial_prompt="Test prompt",
            api_key="AIza_mock-key-for-testing-purposes-1234567890",  # FIX: Long enough API key
            model_name="gemini-2.5-flash-lite-preview-09-2025",  # Use a light model for tests
            domain="General",  # Use 'General' for simple questions
            persona_manager=mock_persona_manager,  # Pass the persona manager
            context_analyzer=mock_context_analyzer,  # Pass the mock context analyzer
            token_tracker=mock_token_tracker,  # Pass the token tracker
            settings=mock_settings,  # Pass mock_settings
            structured_codebase_context={},  # NEW: Add structured_codebase_context
            raw_file_contents={"file1.py": "content"},  # NEW: Add raw_file_contents
            status_callback=MagicMock(),  # FIX: Ensure status_callback is a callable MagicMock
            rich_console=MagicMock(),
            summarizer_pipeline_instance=mock_summarizer_pipeline,  # NEW: Pass the mock summarizer
            # NEW: Pass the mocked injected dependencies
            prompt_optimizer=MockPromptOptimizer.return_value,
            critique_engine=mock_critique_engine,
            improvement_applicator=mock_improvement_applicator,
            content_validator=mock_content_alignment_validator,
            llm_orchestrator=mock_llm_orchestrator,  # ADD THIS LINE
        )
        # Ensure the conflict manager mock is assigned to the instance
        debate.conflict_manager = mock_conflict_manager
        # Ensure the metrics collector mock is assigned to the instance
        debate.metrics_collector = mock_metrics_collector
        # FIX: Ensure output_parser is set on the instance
        debate.output_parser = mock_output_parser
        return debate


def test_socratic_debate_initialization(socratic_debate_instance):  # noqa: F811
    """Tests that the SocraticDebate initializes correctly."""
    assert socratic_debate_instance is not None
    assert socratic_debate_instance.initial_prompt == "Test prompt"  # noqa: F841
    assert (
        socratic_debate_instance.model_name
        == "gemini-2.5-flash-lite-preview-09-2025"  # Corrected model name
    )
    assert socratic_debate_instance.llm_provider is not None
    assert socratic_debate_instance.persona_manager is not None
    assert socratic_debate_instance.token_tracker is not None
    assert socratic_debate_instance.context_analyzer is not None
    assert socratic_debate_instance.settings is not None
    assert socratic_debate_instance.llm_orchestrator is not None  # ADD THIS ASSERTION
    assert (
        socratic_debate_instance.output_parser is not None
    )  # FIX: Assert output_parser is set


def test_socratic_debate_run_debate_success(  # noqa: F811
    socratic_debate_instance,
    mock_gemini_provider,
    mock_output_parser,
    mock_llm_orchestrator,  # ADD mock_llm_orchestrator
):
    """Tests a successful end-to-end debate run."""
    # Reset mock_output_parser.parse_and_validate for this specific test
    mock_output_parser.parse_and_validate.side_effect = [
        {
            "general_output": "Visionary idea",
            "malformed_blocks": [],
        },  # Visionary_Generator
        {
            "general_output": "Skeptical idea",
            "malformed_blocks": [],
        },  # Skeptical_Generator
        {
            "general_output": "Final Answer",
            "malformed_blocks": [],
        },  # Impartial_Arbitrator
    ]
    # The mock_llm_orchestrator.call_llm will be called instead of mock_gemini_provider.generate directly
    mock_llm_orchestrator.call_llm.side_effect = [
        {
            "text": '{"general_output": "Visionary idea"}',
            "usage": {
                "prompt_tokens": 100,
                "completion_tokens": 50,
                "total_tokens": 150,
            },
            "is_truncated": False,
        },  # Visionary_Generator
        {
            "text": '{"general_output": "Skeptical idea"}',
            "usage": {
                "prompt_tokens": 100,
                "completion_tokens": 50,
                "total_tokens": 150,
            },
            "is_truncated": False,
        },  # Skeptical_Generator
        {
            "text": '{"general_output": "Final Answer"}',
            "usage": {
                "prompt_tokens": 100,
                "completion_tokens": 50,
                "total_tokens": 150,
            },
            "is_truncated": False,
        },  # Impartial_Arbitrator
    ]

    final_answer, intermediate_steps = socratic_debate_instance.run_debate()

    assert final_answer["general_output"] == "Final Answer"  # noqa: F841
    assert "Total_Tokens_Used" in intermediate_steps
    assert "Total_Estimated_Cost_USD" in intermediate_steps
    assert (
        mock_llm_orchestrator.call_llm.call_count
        == EXPECTED_DEBATE_TURNS_3  # MODIFIED: Check orchestrator call count
    )
    assert (
        mock_output_parser.parse_and_validate.call_count == EXPECTED_DEBATE_TURNS_3
    )  # FIX: PLR2004 (3)


def test_socratic_debate_malformed_output_triggers_conflict_manager(  # noqa: F811
    socratic_debate_instance,
    mock_gemini_provider,
    mock_conflict_manager,
    mock_llm_orchestrator,  # ADD mock_llm_orchestrator
):
    """Tests that malformed output triggers the conflict manager and that resolution is handled."""
    socratic_debate_instance.persona_manager.persona_router.determine_persona_sequence.return_value = [
        "Visionary_Generator",
        "Constructive_Critic",
        "Impartial_Arbitrator",
    ]

    # The mock_llm_orchestrator.call_llm will be called instead of mock_gemini_provider.generate directly
    mock_llm_orchestrator.call_llm.side_effect = [
        # Visionary_Generator (valid)
        {
            "text": '{"general_output": "Visionary idea"}',
            "usage": {
                "prompt_tokens": 100,
                "completion_tokens": 50,
                "total_tokens": 150,
            },
            "is_truncated": False,
        },
        # Constructive_Critic (malformed)
        {
            "text": '{"CRITIQUE_SUMMARY": "Malformed output", "malformed_blocks": [{"type": "SCHEMA_VALIDATION_ERROR"}]}',
            "usage": {
                "prompt_tokens": 150,
                "completion_tokens": 70,
                "total_tokens": 220,
            },
            "is_truncated": False,
        },
        # Impartial_Arbitrator (will receive resolved output from conflict manager)
        {
            "text": '{"general_output": "Final synthesis from resolved conflict"}',
            "usage": {
                "prompt_tokens": 200,
                "completion_tokens": 100,
                "total_tokens": 300,
            },
            "is_truncated": False,
        },
    ]

    # Mock the parser to return malformed for the critic, then conflict report, then final answer
    socratic_debate_instance.output_parser.parse_and_validate.side_effect = [
        {"general_output": "Visionary idea", "malformed_blocks": []},  # Visionary
        {
            "CRITIQUE_SUMMARY": "Malformed output",
            "malformed_blocks": [
                {"type": "SCHEMA_VALIDATION_ERROR", "message": "Invalid field"}
            ],
        },  # Critic
        {
            "general_output": "Final synthesis from resolved conflict",
            "malformed_blocks": [],
        },  # Arbitrator
    ]

    final_answer, intermediate_steps = socratic_debate_instance.run_debate()

    assert final_answer["general_output"] == "Final synthesis from resolved conflict"  # noqa: F841
    assert "Conflict_Resolution_Attempt" in intermediate_steps
    assert (
        intermediate_steps["Conflict_Resolution_Attempt"]["conflict_resolved"] is True
    )
    assert (
        "Mock conflict resolved."
        in intermediate_steps["Conflict_Resolution_Attempt"]["resolution_summary"]
    )
    # Assert that the resolved output from the conflict manager was used in the history
    # The history should contain an entry for Conflict_Resolution_Manager
    conflict_manager_turn = next(
        (
            t
            for t in intermediate_steps.get("Debate_History", [])
            if t.get("persona") == "Conflict_Resolution_Manager"
        ),
        None,
    )
    assert conflict_manager_turn is not None, (
        "Conflict resolution manager turn not found in history"
    )
    assert "Mock resolved output from conflict" in conflict_manager_turn["output"].get(
        "resolved_output", {}
    ).get("general_output", "")


def test_socratic_debate_token_budget_exceeded(  # noqa: F811
    socratic_debate_instance,
    mock_gemini_provider,
    mock_token_tracker,
    mock_llm_orchestrator,  # ADD mock_llm_orchestrator
):
    """Tests that a TokenBudgetExceededError is raised when budget is exceeded."""
    mock_token_tracker.current_usage = 990000  # Near budget limit
    mock_token_tracker.budget = 1000000
    # Mock the orchestrator to return a response that exceeds the budget
    mock_llm_orchestrator.call_llm.side_effect = TokenBudgetExceededError(
        current_tokens=990000,
        budget=1000000,
        details={"phase": "debate", "step_name": "TestPersona", "tokens_needed": 50000},
    )

    with pytest.raises(TokenBudgetExceededError):
        socratic_debate_instance.run_debate()
    # Ensure orchestrator was called at least once before the budget was exceeded
    assert mock_llm_orchestrator.call_llm.call_count >= 1


def test_execute_llm_turn_schema_validation_retry(  # noqa: F811
    socratic_debate_instance,
    mock_gemini_provider,
    mock_llm_orchestrator,  # ADD mock_llm_orchestrator
):
    """Tests that _execute_llm_turn retries on SchemaValidationError."""
    persona_name = "Constructive_Critic"
    persona_config = (
        socratic_debate_instance.persona_manager.get_adjusted_persona_config(
            persona_name
        )
    )

    # Mock orchestrator to first return invalid, then valid JSON
    mock_llm_orchestrator.call_llm.side_effect = [
        {
            "text": '{"invalid_field": "Malformed output"}',
            "usage": {
                "prompt_tokens": 100,
                "completion_tokens": 50,
                "total_tokens": 150,
            },
            "is_truncated": False,
        },  # First attempt: invalid
        {
            "text": '{"CRITIQUE_SUMMARY": "Valid critique", "CRITIQUE_POINTS": [], "SUGGESTIONS": []}',
            "usage": {
                "prompt_tokens": 120,
                "completion_tokens": 60,
                "total_tokens": 180,
            },
            "is_truncated": False,
        },  # Second attempt: valid
    ]

    # Mock parser to first fail, then succeed
    socratic_debate_instance.output_parser.parse_and_validate.side_effect = [
        {
            "CRITIQUE_SUMMARY": "Malformed output",
            "malformed_blocks": [
                {"type": "SCHEMA_VALIDATION_ERROR", "message": "Invalid field"}
            ],
        },
        {
            "CRITIQUE_SUMMARY": "Valid critique",
            "CRITIQUE_POINTS": [],
            "SUGGESTIONS": [],
            "malformed_blocks": [],
        },
    ]

    output = socratic_debate_instance._execute_llm_turn(
        persona_name, "Test prompt for critic", "debate", persona_config.max_tokens
    )

    assert (
        mock_llm_orchestrator.call_llm.call_count
        == EXPECTED_RETRY_COUNT_2  # MODIFIED: Check orchestrator call count
    )
    assert (
        socratic_debate_instance.output_parser.parse_and_validate.call_count
        == EXPECTED_RETRY_COUNT_2
    )  # FIX: PLR2004 (2)
    # Assert that the final output is the valid one
    assert output["CRITIQUE_SUMMARY"] == "Valid critique"
    # Assert that a malformed block for retry was recorded (if applicable, depends on mock behavior)
    assert any(
        block["type"] == "RETRYABLE_VALIDATION_ERROR"
        for block in socratic_debate_instance.intermediate_steps.get(
            "malformed_blocks", []
        )
    )


def test_socratic_debate_self_analysis_flow(  # noqa: F811
    socratic_debate_instance,
    mock_gemini_provider,
    mock_persona_manager,
    mock_metrics_collector,
    mock_llm_orchestrator,  # ADD mock_llm_orchestrator
):
    """Tests the Self-Improvement Analyst flow, ensuring metrics are collected and passed."""
    mock_metrics_collector.collect_all_metrics.return_value = {
        "reasoning_quality": {"score": 0.8}
    }
    mock_metrics_collector.analyze_historical_effectiveness.return_value = {
        "total_attempts": 1,
        "success_rate": 1.0,
    }
    mock_persona_manager.prompt_analyzer.is_self_analysis_prompt.return_value = True
    mock_persona_manager.persona_router.determine_persona_sequence.return_value = [
        "Self_Improvement_Analyst"
    ]

    # Mock LLM response for Self_Improvement_Analyst
    mock_llm_orchestrator.call_llm.return_value = {
        "text": """{"ANALYSIS_SUMMARY": "Self-analysis complete.", "IMPACTFUL_SUGGESTIONS": []}""",
        "usage": {"prompt_tokens": 200, "completion_tokens": 100, "total_tokens": 300},
        "is_truncated": False,
    }
    socratic_debate_instance.output_parser.parse_and_validate.side_effect = [
        # Self_Improvement_Analyst (final synthesis)
        (
            {
                "ANALYSIS_SUMMARY": "Self-analysis complete.",
                "IMPACTFUL_SUGGESTIONS": [],
                "malformed_blocks": [],
            }
        )
    ]

    final_answer, intermediate_steps = socratic_debate_instance.run_debate()

    assert final_answer["ANALYSIS_SUMMARY"] == "Self-analysis complete."  # noqa: F841
    assert "Self_Improvement_Analyst_Output" in intermediate_steps
    mock_metrics_collector.collect_all_metrics.assert_called_once()
    mock_metrics_collector.record_self_improvement_suggestion_outcome.assert_called_once_with(
        "Self_Improvement_Analyst", True, False
    )


def test_socratic_debate_context_aware_assistant_turn(  # noqa: F811
    socratic_debate_instance,
    mock_gemini_provider,
    mock_context_analyzer,
    mock_llm_orchestrator,  # ADD mock_llm_orchestrator
):
    """Tests the Context_Aware_Assistant turn when present in the sequence, ensuring context is passed."""
    socratic_debate_instance.persona_router.determine_persona_sequence.return_value = [
        "Context_Aware_Assistant",
        "Impartial_Arbitrator",
    ]

    mock_llm_orchestrator.call_llm.side_effect = [
        {
            "text": '{"general_overview": "Context analysis output"}',
            "usage": {
                "prompt_tokens": 100,
                "completion_tokens": 50,
                "total_tokens": 150,
            },
            "is_truncated": False,
        },  # Context_Aware_Assistant
        {
            "text": '{"general_output": "Final answer from arbitrator"}',
            "usage": {
                "prompt_tokens": 100,
                "completion_tokens": 50,
                "total_tokens": 150,
            },
            "is_truncated": False,
        },  # Impartial_Arbitrator
    ]
    socratic_debate_instance.output_parser.parse_and_validate.side_effect = [
        {"general_overview": "Context analysis output", "malformed_blocks": []},
        {"general_output": "Final answer from arbitrator", "malformed_blocks": []},
    ]

    final_answer, intermediate_steps = socratic_debate_instance.run_debate()

    assert "Context_Aware_Assistant_Output" in intermediate_steps
    assert (
        intermediate_steps["Context_Aware_Assistant_Output"].get("general_overview")
        == "Context analysis output"
    )
    mock_context_analyzer.find_relevant_files.assert_called_once()
    mock_context_analyzer.generate_context_summary.assert_called_once()
