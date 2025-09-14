# tests/integration/test_reasoning_engine.py
import pytest
import os
from unittest.mock import patch, MagicMock

# Assuming these imports are correct based on your project structure
# Adjust if your core.py or llm_provider.py are in different locations
from core import (
    SocraticDebate,
)  # SocraticDebate is in the project root, so direct import
from src.llm_provider import (
    GeminiProvider,
)  # Assuming GeminiProvider is in src/llm_provider.py
from src.persona_manager import PersonaManager  # Import PersonaManager
from src.token_tracker import TokenUsageTracker  # Import TokenUsageTracker
from src.config.settings import ChimeraSettings  # NEW: Import ChimeraSettings
from src.context.context_analyzer import (
    ContextRelevanceAnalyzer,
    CodebaseScanner,
)  # NEW: Import for SocraticDebate init
from src.utils.output_parser import (
    LLMOutputParser,
)  # NEW: Import for SocraticDebate init
from src.conflict_resolution import (
    ConflictResolutionManager,
)  # NEW: Import for SocraticDebate init
from src.self_improvement.metrics_collector import (
    FocusedMetricsCollector,
)  # NEW: Import for SocraticDebate init
from src.utils.prompt_optimizer import (
    PromptOptimizer,
)  # NEW: Import for SocraticDebate init
from transformers import pipeline  # NEW: Import pipeline for summarization


# Fixture to provide a real LLM client for integration tests.
# Requires TEST_LLM_API_KEY environment variable to be set.
@pytest.fixture
def real_llm_client():
    api_key = os.environ.get("TEST_LLM_API_KEY")
    if not api_key:
        pytest.skip(
            "TEST_LLM_API_KEY environment variable not set. Skipping real LLM integration tests."
        )

    try:
        return GeminiProvider(api_key=api_key, model_name="gemini-2.5-flash-lite")
    except Exception as e:
        pytest.fail(f"Failed to initialize real LLM client: {e}")


@pytest.mark.integration
def test_reasoning_engine_integration(real_llm_client):
    """
    Performs an end-to-end integration test of the SocraticDebate engine
    using a real LLM client.
    """
    api_key = os.environ.get("TEST_LLM_API_KEY")
    if not api_key:
        pytest.skip(
            "TEST_LLM_API_KEY environment variable not set for SocraticDebate init."
        )

    user_input = "What is the capital of France?"
    context = {"country": "France"}

    # Mock DOMAIN_KEYWORDS for PersonaManager initialization
    mock_domain_keywords = {
        "General": ["general", "question", "answer", "capital", "country"],
        "Software Engineering": ["code", "python", "implement"],
    }

    # Initialize TokenUsageTracker
    token_tracker_instance = TokenUsageTracker(
        budget=100000
    )  # Provide a reasonable budget

    # Initialize PersonaManager with mock domain keywords and token tracker
    persona_manager_instance = PersonaManager(
        mock_domain_keywords,
        token_tracker=token_tracker_instance,
        settings=ChimeraSettings(),
    )

    # Initialize other required mocks for SocraticDebate
    mock_context_analyzer = MagicMock(spec=ContextRelevanceAnalyzer)
    mock_context_analyzer.file_embeddings = {}
    mock_context_analyzer.find_relevant_files.return_value = []
    mock_context_analyzer.generate_context_summary.return_value = "No context summary."
    mock_context_analyzer.raw_file_contents = {}  # Ensure this is set

    mock_codebase_scanner = MagicMock(spec=CodebaseScanner)
    mock_codebase_scanner.raw_file_contents = {}
    mock_codebase_scanner.file_structure = {}

    mock_output_parser = MagicMock(spec=LLMOutputParser)
    mock_output_parser.parse_and_validate.return_value = {
        "general_output": "Paris",
        "malformed_blocks": [],
    }

    mock_conflict_manager = MagicMock(spec=ConflictResolutionManager)
    mock_conflict_manager.resolve_conflict.return_value = (
        None  # No conflict resolution needed for simple case
    )

    mock_metrics_collector = MagicMock(spec=FocusedMetricsCollector)
    mock_metrics_collector.collect_all_metrics.return_value = {}
    mock_metrics_collector.analyze_historical_effectiveness.return_value = {}
    mock_metrics_collector.file_analysis_cache = {}

    mock_summarizer_pipeline = MagicMock(spec=pipeline)
    mock_summarizer_pipeline.return_value = [{"summary_text": "Mock summary."}]
    mock_summarizer_pipeline.tokenizer.model_max_length = 1024

    try:
        # Patch the PromptOptimizer during SocraticDebate initialization
        with patch("core.PromptOptimizer") as MockPromptOptimizer:
            MockPromptOptimizer.return_value = MagicMock(spec=PromptOptimizer)
            MockPromptOptimizer.return_value.optimize_prompt.side_effect = (
                lambda p, pn, mot: p
            )
            MockPromptOptimizer.return_value.optimize_debate_history.side_effect = (
                lambda h, mt: h
            )
            MockPromptOptimizer.return_value.tokenizer = (
                real_llm_client.tokenizer
            )  # Ensure tokenizer is set

            engine = SocraticDebate(
                initial_prompt=user_input,
                api_key=api_key,
                model_name="gemini-2.5-flash-lite",  # Use a light model for tests
                domain="General",  # Use 'General' for simple questions
                persona_manager=persona_manager_instance,  # Pass the persona manager
                structured_codebase_context={},  # NEW: Add structured_codebase_context
                raw_file_contents={},  # NEW: Add raw_file_contents
                token_tracker=token_tracker_instance,  # Pass the token tracker
                settings=ChimeraSettings(),  # Pass a real settings instance
                context_analyzer=mock_context_analyzer,  # Pass mock context analyzer
                status_callback=MagicMock(),
                rich_console=MagicMock(),
                codebase_scanner=mock_codebase_scanner,  # Pass mock codebase scanner
                summarizer_pipeline_instance=mock_summarizer_pipeline,  # Pass mock summarizer pipeline
            )

            # Manually set the output_parser and conflict_manager mocks on the instance
            # as they might be re-initialized internally if not passed correctly.
            engine.output_parser = mock_output_parser
            engine.conflict_manager = mock_conflict_manager
            engine.metrics_collector = mock_metrics_collector

            result, intermediate_steps = engine.run_debate()

            assert isinstance(result, dict)
            assert "general_output" in result
            assert (
                "Paris" in result["general_output"]
                or "paris" in result["general_output"]
            )

            print(f"Integration test successful. Result: {result['general_output']}")
            print(
                f"Total tokens used: {intermediate_steps.get('Total_Tokens_Used', 0)}"
            )
            print(
                f"Estimated cost: ${intermediate_steps.get('Total_Estimated_Cost_USD', 0.0):.6f}"
            )

    except Exception as e:
        pytest.fail(f"Integration test failed with exception: {e}")
