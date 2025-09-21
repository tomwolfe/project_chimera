from unittest.mock import MagicMock, patch

import pytest

# Assuming core.py is in the project root
from core import SocraticDebate
from src.config.settings import ChimeraSettings
from src.context.context_analyzer import ContextRelevanceAnalyzer
from src.llm_provider import GeminiProvider
from src.models import ConflictReport, CritiqueOutput, GeneralOutput, PersonaConfig
from src.persona_manager import PersonaManager
from src.token_tracker import TokenUsageTracker

# NEW IMPORTS for full dependency injection
from src.self_improvement.content_validator import ContentAlignmentValidator
from src.self_improvement.critique_engine import CritiqueEngine
from src.self_improvement.improvement_applicator import ImprovementApplicator
from src.self_improvement.metrics_collector import FocusedMetricsCollector
from src.utils.prompting.prompt_optimizer import PromptOptimizer
from src.utils.reporting.output_parser import LLMOutputParser


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
    tracker.record_usage.side_effect = lambda tokens, persona=None: setattr(
        tracker, "current_usage", tracker.current_usage + tokens
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


@pytest.fixture
def socratic_debate_instance(
    mock_gemini_provider,
    mock_persona_manager,
    mock_token_tracker,
    mock_context_analyzer,
    mock_settings,
    mock_output_parser,
    mock_conflict_manager,
    mock_metrics_collector,
    mock_summarizer_pipeline,  # NEW: Add mock_summarizer_pipeline
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
            model_name="gemini-2.5-flash-lite",  # Use a light model for tests
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
    assert socratic_debate_instance.model_name == "gemini-2.5-flash-lite"
    assert socratic_debate_instance.llm_provider is not None
    assert socratic_debate_instance.persona_manager is not None
    assert socratic_debate_instance.token_tracker is not None
    assert socratic_debate_instance.context_analyzer is not None
    assert socratic_debate_instance.settings is not None
    assert (
        socratic_debate_instance.output_parser is not None
    )  # FIX: Assert output_parser is set


def test_socratic_debate_run_debate_success(  # noqa: F811
    socratic_debate_instance, mock_gemini_provider, mock_output_parser
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
    mock_gemini_provider.generate.side_effect = [
        ('{"general_output": "Visionary idea"}', 100, 50, False),  # Visionary_Generator
        ('{"general_output": "Skeptical idea"}', 100, 50, False),  # Skeptical_Generator
        ('{"general_output": "Final Answer"}', 100, 50, False),  # Impartial_Arbitrator
    ]

    final_answer, intermediate_steps = socratic_debate_instance.run_debate()

    assert final_answer["general_output"] == "Final Answer"  # noqa: F841
    assert "Total_Tokens_Used" in intermediate_steps
    assert "Total_Estimated_Cost_USD" in intermediate_steps
    assert mock_gemini_provider.generate.call_count == 3  # 3 personas in sequence
    assert mock_output_parser.parse_and_validate.call_count == 3


def test_socratic_debate_malformed_output_triggers_conflict_manager(  # noqa: F811
    socratic_debate_instance, mock_gemini_provider, mock_conflict_manager
):
    """Tests that malformed output triggers the conflict manager and that resolution is handled."""
    socratic_debate_instance.persona_manager.persona_router.determine_persona_sequence.return_value = [
        "Visionary_Generator",
        "Constructive_Critic",
        "Impartial_Arbitrator",
    ]

    # Mock the Constructive_Critic to return malformed output
    mock_gemini_provider.generate.side_effect = [
        # Visionary_Generator (valid)
        ('{"general_output": "Visionary idea"}', 100, 50, False),
        # Constructive_Critic (malformed)
        (
            '{"CRITIQUE_SUMMARY": "Malformed output", "malformed_blocks": [{"type": "SCHEMA_VALIDATION_ERROR"}]}',
            150,
            70,
            False,
        ),
        # Impartial_Arbitrator (will receive resolved output from conflict manager)
        (
            '{"general_output": "Final synthesis from resolved conflict"}',
            200,
            100,
            False,
        ),
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
    socratic_debate_instance, mock_gemini_provider, mock_token_tracker
):
    """Tests that a TokenBudgetExceededError is raised when budget is exceeded."""
    mock_token_tracker.current_usage = 990000  # Near budget limit
    mock_token_tracker.budget = 1000000
    mock_gemini_provider.generate.return_value = (
        '{"general_output": "Exceeded"}',
        50000,
        10000,
        False,
    )  # This will exceed budget

    with pytest.raises(TokenBudgetExceededError):
        socratic_debate_instance.run_debate()
    # Ensure generate was called at least once before the budget was exceeded
    assert mock_gemini_provider.generate.call_count >= 1


def test_execute_llm_turn_schema_validation_retry(  # noqa: F811
    socratic_debate_instance, mock_gemini_provider
):
    """Tests that _execute_llm_turn retries on SchemaValidationError."""
    persona_name = "Constructive_Critic"
    persona_config = (
        socratic_debate_instance.persona_manager.get_adjusted_persona_config(
            persona_name
        )
    )

    # Mock LLM to first return invalid, then valid JSON
    mock_gemini_provider.generate.side_effect = [
        (
            '{"invalid_field": "Malformed output"}',
            100,
            50,
            False,
        ),  # First attempt: invalid
        (
            '{"CRITIQUE_SUMMARY": "Valid critique", "CRITIQUE_POINTS": [], "SUGGESTIONS": []}',
            120,
            60,
            False,
        ),  # Second attempt: valid
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

    assert mock_gemini_provider.generate.call_count == 2
    assert socratic_debate_instance.output_parser.parse_and_validate.call_count == 2
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
    mock_gemini_provider.generate.return_value = (
        """{"ANALYSIS_SUMMARY": "Self-analysis complete.", "IMPACTFUL_SUGGESTIONS": []}""",
        200,
        100,
        False,
    )
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
    socratic_debate_instance, mock_gemini_provider, mock_context_analyzer
):
    """Tests the Context_Aware_Assistant turn when present in the sequence, ensuring context is passed."""
    socratic_debate_instance.persona_manager.persona_router.determine_persona_sequence.return_value = [
        "Context_Aware_Assistant",
        "Impartial_Arbitrator",
    ]

    mock_gemini_provider.generate.side_effect = [
        (
            '{"general_overview": "Context analysis output"}',
            100,
            50,
            False,
        ),  # Context_Aware_Assistant
        (
            '{"general_output": "Final answer from arbitrator"}',
            100,
            50,
            False,
        ),  # Impartial_Arbitrator
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


# NEW TESTS for core reasoning logic (from suggested changes, adapted)


class TestProjectChimeraCoreLogic:  # Renamed to avoid conflict with existing TestProjectChimera
    def setup_method(self):
        # Mock all dependencies for SocraticDebate
        self.mock_llm_provider = MagicMock(spec=GeminiProvider)
        self.mock_llm_provider.tokenizer = MagicMock()
        self.mock_llm_provider.tokenizer.count_tokens.side_effect = (
            lambda text: len(text) // 4
        )
        self.mock_llm_provider.tokenizer.max_output_tokens = 8192
        self.mock_llm_provider.calculate_usd_cost.return_value = 0.001
        self.mock_llm_provider.generate.return_value = (
            '{"general_output": "Mock LLM response"}',
            100,
            50,
            False,
        )

        self.mock_token_tracker = MagicMock(spec=TokenUsageTracker)
        self.mock_token_tracker.current_usage = 0
        self.mock_token_tracker.budget = 1000000
        self.mock_token_tracker.record_usage.side_effect = (
            lambda tokens, persona=None: setattr(
                self.mock_token_tracker,
                "current_usage",
                self.mock_token_tracker.current_usage + tokens,
            )
        )
        self.mock_token_tracker.get_consumption_rate.return_value = 0.1
        self.mock_token_tracker.reset.return_value = None

        self.mock_settings = MagicMock(spec=ChimeraSettings)
        self.mock_settings.total_budget = 1000000
        self.mock_settings.context_token_budget_ratio = 0.25
        self.mock_settings.debate_token_budget_ratio = 0.65
        self.mock_settings.synthesis_token_budget_ratio = 0.10
        self.mock_settings.self_analysis_context_ratio = 0.45
        self.mock_settings.self_analysis_debate_ratio = 0.30
        self.mock_settings.self_analysis_synthesis_ratio = 0.25
        self.mock_settings.max_retries = 2
        self.mock_settings.max_backoff_seconds = 1
        self.mock_settings.default_max_input_tokens_per_persona = 4000
        self.mock_settings.max_tokens_per_persona = {}

        self.mock_persona_manager = MagicMock(spec=PersonaManager)
        self.mock_persona_manager.prompt_analyzer.is_self_analysis_prompt.return_value = False
        self.mock_persona_manager.prompt_analyzer.analyze_complexity.return_value = {
            "complexity_score": 0.5
        }
        self.mock_persona_manager.persona_router.determine_persona_sequence.return_value = [
            "Visionary_Generator",
            "Impartial_Arbitrator",
        ]
        self.mock_persona_manager.get_adjusted_persona_config.return_value = MagicMock(
            system_prompt_template="Mock system prompt",
            temperature=0.1,
            max_tokens=1024,
            output_schema="GeneralOutput",
        )
        self.mock_persona_manager.PERSONA_OUTPUT_SCHEMAS = {
            "Visionary_Generator": GeneralOutput,
            "Impartial_Arbitrator": GeneralOutput,
        }

        self.mock_context_analyzer = MagicMock(spec=ContextRelevanceAnalyzer)
        self.mock_context_analyzer.file_embeddings = {}
        self.mock_context_analyzer.find_relevant_files.return_value = []
        self.mock_context_analyzer.generate_context_summary.return_value = (
            "No context summary."
        )
        self.mock_context_analyzer.raw_file_contents = {}
        self.mock_context_analyzer.compute_file_embeddings.return_value = {}

        self.mock_output_parser = MagicMock(spec=LLMOutputParser)
        self.mock_output_parser.parse_and_validate.return_value = {
            "general_output": "Parsed mock output",
            "malformed_blocks": [],
        }
        self.mock_output_parser._create_fallback_output.return_value = {
            "general_output": "Fallback mock output",
            "malformed_blocks": [],
        }
        self.mock_output_parser._get_schema_class_from_name.return_value = (
            GeneralOutput  # Ensure this is mocked
        )

        self.mock_conflict_manager = MagicMock()
        self.mock_conflict_manager.resolve_conflict.return_value = None

        self.mock_metrics_collector = MagicMock(spec=FocusedMetricsCollector)
        self.mock_metrics_collector.collect_all_metrics.return_value = {}
        self.mock_metrics_collector.analyze_historical_effectiveness.return_value = {}
        self.mock_metrics_collector.record_self_improvement_suggestion_outcome.return_value = None
        self.mock_metrics_collector.file_analysis_cache = {}

        self.mock_prompt_optimizer = MagicMock(spec=PromptOptimizer)
        self.mock_prompt_optimizer.optimize_prompt.side_effect = (
            lambda p, pc, mot, sm, isa: p
        )
        self.mock_prompt_optimizer.optimize_debate_history.side_effect = lambda h, mt: h
        self.mock_prompt_optimizer.tokenizer = self.mock_llm_provider.tokenizer
        self.mock_prompt_optimizer.generate_prompt.side_effect = (
            lambda template_name, context: template_name
        )

        self.mock_content_validator = MagicMock(spec=ContentAlignmentValidator)
        self.mock_content_validator.validate.return_value = (
            True,
            "Content aligned.",
            {},
        )

        self.mock_critique_engine = MagicMock(spec=CritiqueEngine)
        self.mock_improvement_applicator = MagicMock(spec=ImprovementApplicator)

        self.mock_summarizer_pipeline = MagicMock()
        self.mock_summarizer_pipeline.return_value = [{"summary_text": "Mock summary."}]
        self.mock_summarizer_pipeline.tokenizer.model_max_length = 1024

        # Instantiate SocraticDebate with all mocked dependencies
        self.socratic_debate = SocraticDebate(
            initial_prompt="Analyze this reasoning problem: What is the capital of France?",
            api_key="AIza_mock-key-for-testing-purposes-1234567890",
            model_name="gemini-2.5-flash-lite",
            domain="General",
            persona_manager=self.mock_persona_manager,
            structured_codebase_context={},
            raw_file_contents={},
            settings=self.mock_settings,
            context_analyzer=self.mock_context_analyzer,
            token_tracker=self.mock_token_tracker,
            status_callback=MagicMock(),
            rich_console=MagicMock(),
            summarizer_pipeline_instance=self.mock_summarizer_pipeline,
            prompt_optimizer=self.mock_prompt_optimizer,
            critique_engine=self.mock_critique_engine,
            improvement_applicator=self.mock_improvement_applicator,
            content_validator=self.mock_content_validator,
            conflict_resolution_manager=self.mock_conflict_manager,
        )
        # Ensure internal references are also set to mocks
        self.socratic_debate.llm_provider = self.mock_llm_provider
        self.socratic_debate.output_parser = self.mock_output_parser
        self.socratic_debate.metrics_collector = self.mock_metrics_collector

    def test_reasoning_task(self):
        # Mock LLM to return a specific answer for the reasoning task
        self.mock_llm_provider.generate.side_effect = [
            ('{"general_output": "Paris"}', 100, 50, False),  # Visionary
            (
                '{"general_output": "Paris is the capital of France."}',
                100,
                50,
                False,
            ),  # Arbitrator
        ]
        self.mock_output_parser.parse_and_validate.side_effect = [
            {"general_output": "Paris", "malformed_blocks": []},
            {
                "general_output": "Paris is the capital of France.",
                "malformed_blocks": [],
            },
        ]

        final_answer, _ = self.socratic_debate.run_debate()
        assert "Paris" in final_answer.get("general_output", "")

    def test_code_generation(self):
        # Mock LLM to return Python code
        self.mock_llm_provider.generate.side_effect = [
            (
                '{"general_output": "def add(a, b):\\n    return a + b"}',
                100,
                50,
                False,
            ),  # Visionary
            (
                '{"general_output": "def add(a, b):\\n    return a + b"}',
                100,
                50,
                False,
            ),  # Arbitrator
        ]
        self.mock_output_parser.parse_and_validate.side_effect = [
            {
                "general_output": "def add(a, b):\\n    return a + b",
                "malformed_blocks": [],
            },
            {
                "general_output": "def add(a, b):\\n    return a + b",
                "malformed_blocks": [],
            },
        ]

        self.socratic_debate.initial_prompt = (
            "Generate a Python function that adds two numbers"
        )
        final_answer, _ = self.socratic_debate.run_debate()
        assert "def add" in final_answer.get("general_output", "")
        assert "return a + b" in final_answer.get("general_output", "")

    def test_error_handling(self):
        # Mock LLM to raise an exception
        self.mock_llm_provider.generate.side_effect = Exception("Simulated LLM error")

        with pytest.raises(Exception, match="Simulated LLM error"):
            self.socratic_debate.run_debate()

    def test_token_usage_tracking(self):
        # Reset token tracker for this test
        self.mock_token_tracker.reset()
        self.mock_token_tracker.current_usage = 0

        self.mock_llm_provider.generate.side_effect = [
            ('{"general_output": "Output 1"}', 100, 50, False),  # Visionary
            ('{"general_output": "Output 2"}', 100, 50, False),  # Arbitrator
        ]
        self.mock_output_parser.parse_and_validate.side_effect = [
            {"general_output": "Output 1", "malformed_blocks": []},
            {"general_output": "Output 2", "malformed_blocks": []},
        ]

        _, intermediate_steps = self.socratic_debate.run_debate()
        # Expect total tokens to be sum of input+output from 2 LLM calls + initial prompt/context tokens
        # Mocked generate returns (100 input + 50 output) * 2 = 300
        # Plus initial prompt/context tokens (mocked tokenizer returns len(text)//4)
        # Initial prompt is "Analyze this reasoning problem: What is the capital of France?" (approx 15 words * 4 = 60 tokens)
        # So total should be around 360.
        assert self.mock_token_tracker.current_usage > 300
        assert (
            intermediate_steps.get("Total_Tokens_Used")
            == self.mock_token_tracker.current_usage
        )

    def test_prompt_optimization(self):
        original_prompt = "Analyze this reasoning problem"
        optimized_prompt_text = "Optimized: Analyze this reasoning problem"
        self.mock_prompt_optimizer.optimize_prompt.side_effect = (
            lambda p, pc, mot, sm, isa: optimized_prompt_text
        )

        # Mock LLM to return a specific answer
        self.mock_llm_provider.generate.side_effect = [
            (
                '{"general_output": "Result from optimized prompt"}',
                100,
                50,
                False,
            ),  # Visionary
            ('{"general_output": "Final result"}', 100, 50, False),  # Arbitrator
        ]
        self.mock_output_parser.parse_and_validate.side_effect = [
            {"general_output": "Result from optimized prompt", "malformed_blocks": []},
            {"general_output": "Final result", "malformed_blocks": []},
        ]

        _, _ = self.socratic_debate.run_debate()
        self.mock_prompt_optimizer.optimize_prompt.assert_called()
        # Check that the optimized prompt was passed to the LLM (this is an indirect check)
        # The mock_llm_provider.generate is called with the optimized prompt.
        # We can assert on the arguments of the last call to generate.
        assert (
            self.mock_llm_provider.generate.call_args_list[0].kwargs["prompt"]
            == optimized_prompt_text
        )

    def test_complex_reasoning(self):
        self.socratic_debate.initial_prompt = "Solve this complex problem: If a train leaves New York at 60mph and another leaves Los Angeles at 80mph, when will they meet?"
        self.mock_llm_provider.generate.side_effect = [
            (
                '{"general_output": "The trains will meet at a certain time and distance."}',
                100,
                50,
                False,
            ),  # Visionary
            (
                '{"general_output": "The trains will meet at a certain time and distance."}',
                100,
                50,
                False,
            ),  # Arbitrator
        ]
        self.mock_output_parser.parse_and_validate.side_effect = [
            {
                "general_output": "The trains will meet at a certain time and distance.",
                "malformed_blocks": [],
            },
            {
                "general_output": "The trains will meet at a certain time and distance.",
                "malformed_blocks": [],
            },
        ]

        final_answer, _ = self.socratic_debate.run_debate()
        assert "time" in final_answer.get(
            "general_output", ""
        ) or "distance" in final_answer.get("general_output", "")

    def test_invalid_code_generation(self):
        # Mock LLM to return invalid code (e.g., missing 'def')
        self.mock_llm_provider.generate.side_effect = [
            ('{"general_output": "invalid code snippet"}', 100, 50, False),  # Visionary
            (
                '{"general_output": "invalid code snippet"}',
                100,
                50,
                False,
            ),  # Arbitrator
        ]
        self.mock_output_parser.parse_and_validate.side_effect = [
            {"general_output": "invalid code snippet", "malformed_blocks": []},
            {"general_output": "invalid code snippet", "malformed_blocks": []},
        ]

        self.socratic_debate.initial_prompt = "Generate invalid code"
        final_answer, _ = self.socratic_debate.run_debate()
        # The current SocraticDebate doesn't have a code_validator in its core loop for general output.
        # It would only validate if the output schema explicitly included code changes.
        # So, for a general output, it would just return the "invalid code snippet".
        assert "invalid code snippet" in final_answer.get("general_output", "")
        # To properly test code validation, the output schema would need to be LLMOutput or similar.
        # This test as written in the suggested diff is not directly applicable to SocraticDebate's current structure.
        # I will keep the test but acknowledge this limitation.
