from core import SocraticDebate # Corrected import path
import pytest
from unittest.mock import patch, MagicMock

# Assuming these imports are correct based on your project structure
# Adjust if your core.py or llm_provider.py are in different locations
from src.llm_provider import (
    GeminiProvider,
)  # Assuming GeminiProvider is in src/llm_provider.py
from src.persona_manager import PersonaManager # Import PersonaManager
from src.token_tracker import TokenUsageTracker # Import TokenUsageTracker
from src.config.settings import ChimeraSettings # Import ChimeraSettings
from src.context.context_analyzer import ContextRelevanceAnalyzer # Import ContextRelevanceAnalyzer
from src.utils.output_parser import LLMOutputParser # Import LLMOutputParser
from src.self_improvement.metrics_collector import FocusedMetricsCollector # Import FocusedMetricsCollector
from src.exceptions import TokenBudgetExceededError, SchemaValidationError # Import specific exceptions
from src.models import GeneralOutput, CritiqueOutput # Import specific models

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
    tracker.record_usage.side_effect = lambda tokens, persona=None: setattr(tracker, 'current_usage', tracker.current_usage + tokens)
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
        "Visionary_Generator": PersonaConfig(name="Visionary_Generator", system_prompt="Visionary", temperature=0.7, max_tokens=1024),
        "Skeptical_Generator": PersonaConfig(name="Skeptical_Generator", system_prompt="Skeptical", temperature=0.3, max_tokens=1024),
        "Constructive_Critic": PersonaConfig(name="Constructive_Critic", system_prompt="Critic", temperature=0.15, max_tokens=8192),
        "Impartial_Arbitrator": PersonaConfig(name="Impartial_Arbitrator", system_prompt="Arbitrator", temperature=0.1, max_tokens=4096),
        "Devils_Advocate": PersonaConfig(name="Devils_Advocate", system_prompt="Devils Advocate", temperature=0.1, max_tokens=4096),
        "Self_Improvement_Analyst": PersonaConfig(name="Self_Improvement_Analyst", system_prompt="Self-Improvement Analyst", temperature=0.1, max_tokens=8192),
        "Context_Aware_Assistant": PersonaConfig(name="Context_Aware_Assistant", system_prompt="Context Assistant", temperature=0.1, max_tokens=3072),
    }
    pm.persona_sets = {
        "General": ["Visionary_Generator", "Skeptical_Generator", "Impartial_Arbitrator"],
        "Self-Improvement": ["Self_Improvement_Analyst", "Code_Architect", "Security_Auditor", "DevOps_Engineer", "Test_Engineer", "Constructive_Critic", "Devils_Advocate", "Impartial_Arbitrator"],
    }
    pm.get_adjusted_persona_config.side_effect = lambda name: pm.all_personas.get(name.replace("_TRUNCATED", ""), MagicMock(spec=PersonaConfig, name=name, system_prompt="Fallback", temperature=0.5, max_tokens=1024))
    pm.prompt_analyzer = MagicMock()
    pm.prompt_analyzer.is_self_analysis_prompt.return_value = False
    pm.prompt_analyzer.analyze_complexity.return_value = {"complexity_score": 0.5}
    pm.persona_router = MagicMock()
    pm.persona_router.determine_persona_sequence.return_value = ["Visionary_Generator", "Skeptical_Generator", "Impartial_Arbitrator"]
    pm.token_tracker = mock_token_tracker
    pm.settings = mock_settings
    pm.PERSONA_OUTPUT_SCHEMAS = {
        "Impartial_Arbitrator": GeneralOutput,
        "Context_Aware_Assistant": GeneralOutput,
        "Constructive_Critic": CritiqueOutput,
        "Self_Improvement_Analyst": GeneralOutput,
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
    provider.tokenizer.count_tokens.side_effect = lambda text: len(text) // 4
    provider.tokenizer.max_output_tokens = 8192
    provider.calculate_usd_cost.return_value = 0.001
    provider.generate.return_value = ("{}", 100, 50, False) # Default successful response
    provider.settings = mock_settings
    return provider

@pytest.fixture
def mock_output_parser():
    """Provides a mock LLMOutputParser instance."""
    parser = MagicMock(spec=LLMOutputParser)
    parser.parse_and_validate.return_value = {"general_output": "Parsed output", "malformed_blocks": []}
    parser._create_fallback_output.return_value = {"general_output": "Fallback output", "malformed_blocks": [{"type": "FALLBACK_TRIGGERED"}]}
    return parser

@pytest.fixture
def mock_conflict_manager():
    """Provides a mock ConflictResolutionManager instance."""
    manager = MagicMock()
    manager.resolve_conflict.return_value = {
        "resolution_strategy": "mock_resolution",
        "resolved_output": {"general_output": "Mock resolved output from conflict", "malformed_blocks": []},
        "resolution_summary": "Mock conflict resolved."
    }
    return manager

@pytest.fixture
def mock_metrics_collector():
    """Provides a mock FocusedMetricsCollector instance."""
    collector = MagicMock(spec=FocusedMetricsCollector)
    collector.collect_all_metrics.return_value = {"code_quality": {"ruff_issues_count": 0}}
    collector.analyze_historical_effectiveness.return_value = {"total_attempts": 0, "success_rate": 0.0}
    collector.record_self_improvement_suggestion_outcome.return_value = None
    collector.file_analysis_cache = {} # Ensure this attribute exists
    return collector

@pytest.fixture
def socratic_debate_instance(mock_gemini_provider, mock_persona_manager, mock_token_tracker, mock_context_analyzer, mock_settings, mock_output_parser, mock_conflict_manager, mock_metrics_collector):
    """Provides a SocraticDebate instance with mocked dependencies."""
    with patch('core.GeminiProvider', return_value=mock_gemini_provider), \
         patch('core.PersonaManager', return_value=mock_persona_manager), \
         patch('core.TokenUsageTracker', return_value=mock_token_tracker), \
         patch('core.ContextRelevanceAnalyzer', return_value=mock_context_analyzer), \
         patch('core.LLMOutputParser', return_value=mock_output_parser), \
         patch('core.ConflictResolutionManager', return_value=mock_conflict_manager), \
         patch('core.FocusedMetricsCollector', return_value=mock_metrics_collector):
        
        debate = SocraticDebate(
            initial_prompt="Test prompt",
            api_key="mock_api_key",
            model_name="gemini-2.5-flash-lite",  # Use a light model for tests
            domain="General",  # Use 'General' for simple questions
            persona_manager=mock_persona_manager,  # Pass the persona manager
            context_analyzer=mock_context_analyzer, # Pass the mock context analyzer
            token_tracker=mock_token_tracker, # Pass the mock token tracker
            settings=mock_settings, # Pass mock_settings
            structured_codebase_context={}, # NEW: Add structured_codebase_context
            raw_file_contents={"file1.py": "content"}, # NEW: Add raw_file_contents
        )
        # Ensure the conflict manager mock is assigned to the instance
        debate.conflict_manager = mock_conflict_manager
        # Ensure the metrics collector mock is assigned to the instance
        debate.metrics_collector = mock_metrics_collector
        return debate

def test_socratic_debate_initialization(socratic_debate_instance):
    """Tests that the SocraticDebate initializes correctly."""
    assert socratic_debate_instance.initial_prompt == "Test prompt"
    assert socratic_debate_instance.model_name == "gemini-2.5-flash-lite"
    assert socratic_debate_instance.llm_provider is not None
    assert socratic_debate_instance.persona_manager is not None
    assert socratic_debate_instance.token_tracker is not None
    assert socratic_debate_instance.context_analyzer is not None
    assert socratic_debate_instance.settings is not None

def test_socratic_debate_run_debate_success(socratic_debate_instance, mock_gemini_provider, mock_output_parser):
    """Tests a successful end-to-end debate run."""
    mock_gemini_provider.generate.return_value = ("{}", 100, 50, False) # Mock LLM output
    mock_output_parser.parse_and_validate.return_value = {"general_output": "Final Answer", "malformed_blocks": []}

    final_answer, intermediate_steps = socratic_debate_instance.run_debate()

    assert final_answer["general_output"] == "Final Answer"
    assert "Total_Tokens_Used" in intermediate_steps
    assert "Total_Estimated_Cost_USD" in intermediate_steps
    assert mock_gemini_provider.generate.call_count > 0
    assert mock_output_parser.parse_and_validate.call_count > 0

def test_socratic_debate_malformed_output_triggers_conflict_manager(socratic_debate_instance, mock_gemini_provider, mock_conflict_manager):
    """Tests that malformed output triggers the conflict manager and that resolution is handled."""
    socratic_debate_instance.persona_router.determine_persona_sequence.return_value = [
        "Visionary_Generator", "Constructive_Critic", "Impartial_Arbitrator"
    ]
    
    # Mock the Constructive_Critic to return malformed output
    mock_gemini_provider.generate.side_effect = [
        # Visionary_Generator (valid)
        ('{"general_output": "Visionary idea"}', 100, 50, False),
        # Constructive_Critic (malformed)
        ('{"CRITIQUE_SUMMARY": "Malformed output", "malformed_blocks": [{"type": "SCHEMA_VALIDATION_ERROR"}]}', 150, 70, False),
        # Impartial_Arbitrator (will receive resolved output from conflict manager)
        ('{"general_output": "Final synthesis from resolved conflict"}', 200, 100, False),
    ]
    
    # Mock the parser to return malformed for the critic, then valid for others
    socratic_debate_instance.output_parser.parse_and_validate.side_effect = [
        {"general_output": "Visionary idea", "malformed_blocks": []}, # Visionary
        {"CRITIQUE_SUMMARY": "Malformed output", "malformed_blocks": [{"type": "SCHEMA_VALIDATION_ERROR"}]}, # Critic
        {"general_output": "Final synthesis from resolved conflict", "malformed_blocks": []}, # Arbitrator
    ]

    final_answer, intermediate_steps = socratic_debate_instance.run_debate()

    assert final_answer["general_output"] == "Final synthesis from resolved conflict"
    assert "Conflict_Resolution_Attempt" in intermediate_steps
    assert intermediate_steps["Conflict_Resolution_Attempt"]["conflict_resolved"] is True
    assert "Mock conflict resolved." in intermediate_steps["Conflict_Resolution_Attempt"]["resolution_summary"]
    # Assert that the resolved output from the conflict manager was used in the history
    # The history should contain an entry for Conflict_Resolution_Manager
    conflict_manager_turn = next((t for t in intermediate_steps.get('Debate_History', []) if t.get('persona') == 'Conflict_Resolution_Manager'), None)
    assert conflict_manager_turn is not None, "Conflict resolution manager turn not found in history"
    assert "Mock resolved output from conflict" in conflict_manager_turn['output'].get('resolved_output', {}).get('general_output', '')

def test_socratic_debate_token_budget_exceeded(socratic_debate_instance, mock_gemini_provider, mock_token_tracker):
    """Tests that a TokenBudgetExceededError is raised when budget is exceeded."""
    mock_token_tracker.current_usage = 990000 # Near budget limit
    mock_token_tracker.budget = 1000000
    mock_gemini_provider.generate.return_value = ("{}", 50000, 10000, False) # This will exceed budget

    with pytest.raises(TokenBudgetExceededError):
        socratic_debate_instance.run_debate()

def test_execute_llm_turn_schema_validation_retry(socratic_debate_instance, mock_gemini_provider):
    """Tests that _execute_llm_turn retries on SchemaValidationError."""
    persona_name = "Constructive_Critic"
    persona_config = socratic_debate_instance.persona_manager.get_adjusted_persona_config(persona_name)
    
    # Mock LLM to first return invalid, then valid JSON
    mock_gemini_provider.generate.side_effect = [
        ('{"invalid_field": "Malformed output"}', 100, 50, False), # First attempt: invalid
        ('{"CRITIQUE_SUMMARY": "Valid critique", "CRITIQUE_POINTS": [], "SUGGESTIONS": []}', 120, 60, False), # Second attempt: valid
    ]
    
    # Mock parser to first fail, then succeed
    socratic_debate_instance.output_parser.parse_and_validate.side_effect = [
        {"CRITIQUE_SUMMARY": "Malformed output", "malformed_blocks": [{"type": "SCHEMA_VALIDATION_ERROR", "message": "Invalid field"}]},
        {"CRITIQUE_SUMMARY": "Valid critique", "CRITIQUE_POINTS": [], "SUGGESTIONS": [], "malformed_blocks": []},
    ]

    output = socratic_debate_instance._execute_llm_turn(
        persona_name,
        "Test prompt for critic",
        "debate",
        persona_config.max_tokens,
    )

    assert mock_gemini_provider.generate.call_count == 2
    assert socratic_debate_instance.output_parser.parse_and_validate.call_count == 2
    # Assert that the final output is the valid one
    assert output["CRITIQUE_SUMMARY"] == "Valid critique"
    # Assert that a malformed block for retry was recorded (if applicable, depends on mock behavior)
    # The core logic should handle recording this, not necessarily the test directly asserting it.
    # For this test, we primarily care that the retry mechanism works and a valid output is eventually produced.

def test_socratic_debate_self_analysis_flow(socratic_debate_instance, mock_gemini_provider, mock_persona_manager, mock_metrics_collector):
    """Tests the Self-Improvement Analyst flow, ensuring metrics are collected and passed."""
    mock_metrics_collector.collect_all_metrics.return_value = {"reasoning_quality": {"score": 0.8}}
    mock_metrics_collector.analyze_historical_effectiveness.return_value = {"total_attempts": 1, "success_rate": 1.0}
    mock_persona_manager.prompt_analyzer.is_self_analysis_prompt.return_value = True
    mock_persona_manager.persona_router.determine_persona_sequence.return_value = ["Self_Improvement_Analyst"]

    # Mock LLM response for Self_Improvement_Analyst
    mock_gemini_provider.generate.return_value = (
        """{"ANALYSIS_SUMMARY": "Self-analysis complete.", "IMPACTFUL_SUGGESTIONS": []}""",
        200, 100, False
    )
    socratic_debate_instance.output_parser.parse_and_validate.side_effect = [
        # Context_Aware_Assistant (if in sequence)
        ({"general_overview": "Context overview", "malformed_blocks": []}),
        # Self_Improvement_Analyst (final synthesis)
        ({"ANALYSIS_SUMMARY": "Self-analysis complete.", "IMPACTFUL_SUGGESTIONS": [], "malformed_blocks": []}),
    ]

    final_answer, intermediate_steps = socratic_debate_instance.run_debate()

    assert final_answer["ANALYSIS_SUMMARY"] == "Self-analysis complete."
    assert "Self_Improvement_Analyst_Output" in intermediate_steps
    mock_metrics_collector.collect_all_metrics.assert_called_once()
    mock_metrics_collector.record_self_improvement_suggestion_outcome.assert_called_once_with(
        "Self_Improvement_Analyst", True, False
    )

def test_socratic_debate_context_aware_assistant_turn(socratic_debate_instance, mock_gemini_provider, mock_context_analyzer):
    """Tests the Context_Aware_Assistant turn when present in the sequence, ensuring context is passed."""
    socratic_debate_instance.persona_router.determine_persona_sequence.return_value = [
        "Context_Aware_Assistant", "Impartial_Arbitrator"
    ]
    
    mock_gemini_provider.generate.side_effect = [
        ('{"general_overview": "Context analysis output"}', 100, 50, False), # Context_Aware_Assistant
        ('{"general_output": "Final answer from arbitrator"}', 100, 50, False), # Impartial_Arbitrator
    ]
    socratic_debate_instance.output_parser.parse_and_validate.side_effect = [
        {"general_overview": "Context analysis output", "malformed_blocks": []},
        {"general_output": "Final answer from arbitrator", "malformed_blocks": []},
    ]

    final_answer, intermediate_steps = socratic_debate_instance.run_debate()

    assert "Context_Aware_Assistant_Output" in intermediate_steps
    assert intermediate_steps["Context_Aware_Assistant_Output"].get("general_overview") == "Context analysis output"
    mock_context_analyzer.find_relevant_files.assert_called_once()
    mock_context_analyzer.generate_context_summary.assert_called_once()