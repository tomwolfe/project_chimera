import pytest
from unittest.mock import MagicMock, patch
from core import SocraticDebate
from src.models import PersonaConfig, SelfImprovementAnalysisOutputV1 # Import SelfImprovementAnalysisOutputV1
from src.config.settings import ChimeraSettings
from src.token_tracker import TokenUsageTracker
from src.persona_manager import PersonaManager
from src.context.context_analyzer import ContextRelevanceAnalyzer
from src.llm_provider import GeminiProvider
from src.conflict_resolution import ConflictResolutionManager
from src.utils.output_parser import LLMOutputParser # Keep LLMOutputParser
from src.self_improvement.metrics_collector import FocusedMetricsCollector # Import FocusedMetricsCollector
from src.exceptions import SchemaValidationError # Import SchemaValidationError
from src.self_improvement.content_validator import ContentAlignmentValidator # Import ContentAlignmentValidator
from src.utils.prompt_optimizer import PromptOptimizer # Import PromptOptimizer
from src.config.model_registry import ModelRegistry, ModelSpecification # NEW: Import ModelRegistry

@pytest.fixture
def mock_persona_manager():
    """Provides a mock PersonaManager instance."""
    pm = MagicMock(spec=PersonaManager)
    # Mocking essential attributes and methods used by SocraticDebate
    pm.all_personas = {
        "Visionary_Generator": PersonaConfig(name="Visionary_Generator", system_prompt="Visionary", temperature=0.7, max_tokens=1024, description="Generates innovative solutions."),
        "Skeptical_Generator": PersonaConfig(name="Skeptical_Generator", system_prompt="Skeptical", temperature=0.3, max_tokens=1024, description="Identifies flaws."),
        "Impartial_Arbitrator": PersonaConfig(name="Impartial_Arbitrator", system_prompt="Arbitrator", temperature=0.1, max_tokens=4096, description="Synthesizes outcomes."),
        "Self_Improvement_Analyst": PersonaConfig(name="Self_Improvement_Analyst", system_prompt="Analyst", temperature=0.1, max_tokens=4096, description="Analyzes for improvements."),
        "Constructive_Critic": PersonaConfig(name="Constructive_Critic", system_prompt="Critic", temperature=0.15, max_tokens=8192, description="Provides constructive feedback."),
        "Devils_Advocate": PersonaConfig(name="Devils_Advocate", system_prompt="Devils Advocate", temperature=0.15, max_tokens=4096, description="Challenges proposals."),
        "Context_Aware_Assistant": PersonaConfig(name="Context_Aware_Assistant", system_prompt="Context Assistant", temperature=0.1, max_tokens=3072, description="Analyzes context."),
    }
    pm.persona_sets = {
        "General": ["Visionary_Generator", "Skeptical_Generator", "Impartial_Arbitrator"],
        "Self-Improvement": ["Self_Improvement_Analyst", "Constructive_Critic", "Devils_Advocate", "Impartial_Arbitrator"],
    }
    # Mock methods called by SocraticDebate
    pm.get_adjusted_persona_config.side_effect = lambda name: pm.all_personas.get(name)
    pm.get_persona_sequence_for_framework.return_value = ["Visionary_Generator", "Skeptical_Generator", "Impartial_Arbitrator"]
    pm.prompt_analyzer = MagicMock()
    pm.prompt_analyzer.is_self_analysis_prompt.return_value = False
    pm.get_token_optimized_persona_sequence.side_effect = lambda seq: seq # Return sequence as is for simplicity
    # NEW: Mock PERSONA_OUTPUT_SCHEMAS
    pm.PERSONA_OUTPUT_SCHEMAS = {
        "Impartial_Arbitrator": MagicMock(return_value={"general_output": "Mocked LLM response"}),
        "Constructive_Critic": MagicMock(return_value={"CRITIQUE_SUMMARY": "Mocked critique"}),
        "Self_Improvement_Analyst": MagicMock(return_value={"ANALYSIS_SUMMARY": "Mocked analysis", "IMPACTFUL_SUGGESTIONS": []}),
        "Devils_Advocate": MagicMock(return_value={"conflict_type": "NO_CONFLICT", "summary": "No conflict"}),
        "Visionary_Generator": MagicMock(return_value={"general_output": "Visionary idea"}),
        "Skeptical_Generator": MagicMock(return_value={"general_output": "Skeptical critique"}),
        "Context_Aware_Assistant": MagicMock(return_value={"general_overview": "Context overview"}),
    }
    return pm

@pytest.fixture
def mock_gemini_provider(mock_settings): # MODIFIED: Accept mock_settings
    """Provides a mock GeminiProvider instance."""
    gp = MagicMock(spec=GeminiProvider)
    gp.tokenizer = MagicMock(spec=GeminiTokenizer) # Specify spec for tokenizer
    # Simple token count approximation for testing
    gp.tokenizer.count_tokens.side_effect = lambda text: len(text) // 4 if text else 0
    # Simple truncation simulation
    gp.tokenizer.truncate_to_token_limit.side_effect = lambda text, max_tokens, indicator="": text[:max_tokens*4] + indicator if len(text) > max_tokens*4 else text
    gp.tokenizer.max_output_tokens = 65536
    # Mock generate method to return a tuple: (raw_output, input_tokens, output_tokens, is_truncated)
    gp.generate.return_value = ("{'general_output': 'Mocked LLM response'}", 50, 100, False)
    gp.calculate_usd_cost.return_value = 0.001
    # NEW: Mock model_registry and get_model_specification
    gp.model_registry = MagicMock(spec=ModelRegistry)
    gp.model_registry.get_model.return_value = ModelSpecification(
        name="gemini-2.5-flash-lite",
        max_input_tokens=1048576,
        max_output_tokens=8192,
        cost_per_1k_input=0.075,
        cost_per_1k_output=0.30,
        capabilities=["reasoning", "coding"]
    )
    gp.get_model_specification.return_value = gp.model_registry.get_model.return_value
    # NEW: Set MAX_RETRIES and MAX_BACKOFF_SECONDS from settings
    gp.MAX_RETRIES = mock_settings.max_retries
    gp.MAX_BACKOFF_SECONDS = mock_settings.max_backoff_seconds
    return gp

@pytest.fixture
def mock_context_analyzer():
    """Provides a mock ContextRelevanceAnalyzer instance."""
    ca = MagicMock(spec=ContextRelevanceAnalyzer)
    ca.codebase_context = {"file1.py": "content"} # Provide some context
    ca.file_embeddings = {}
    ca.find_relevant_files.return_value = []
    ca.generate_context_summary.return_value = "Mock context summary."
    ca.set_persona_router = MagicMock() # Mock this method as it's called in SocraticDebate init
    ca.compute_file_embeddings.return_value = {} # Ensure this is mocked
    return ca

@pytest.fixture
def mock_token_tracker():
    """Provides a mock TokenUsageTracker instance."""
    tt = MagicMock(spec=TokenUsageTracker)
    tt.budget = 100000
    tt.current_usage = 0
    # Mock record_usage to update current_usage
    tt.record_usage.side_effect = lambda tokens, persona=None: setattr(tt, 'current_usage', tt.current_usage + tokens)
    tt.reset.side_effect = lambda: setattr(tt, 'current_usage', 0)
    tt.set_current_stage = MagicMock()
    return tt

@pytest.fixture
def mock_settings(): # NEW: Fixture for ChimeraSettings
    return ChimeraSettings(total_budget=100000, max_retries=2, max_backoff_seconds=10)

@pytest.fixture
def mock_conflict_manager(mock_settings): # MODIFIED: Accept mock_settings
    """Provides a mock ConflictResolutionManager instance."""
    cm = MagicMock(spec=ConflictResolutionManager)
    # Configure resolve_conflict to return a successful mock resolution
    cm.resolve_conflict.return_value = {
        "resolution_strategy": "mock_resolved",
        "resolved_output": {"general_output": "Mock resolved output"},
        "resolution_summary": "Mock resolution summary.",
        "malformed_blocks": [{"type": "MOCK_RESOLUTION", "message": "Mock resolution applied."}]
    }
    # NEW: Set max_self_correction_retries from settings
    cm.max_self_correction_retries = mock_settings.max_retries
    return cm

@pytest.fixture
def mock_content_validator():
    """Provides a mock ContentAlignmentValidator instance."""
    cv = MagicMock(spec=ContentAlignmentValidator)
    cv.validate.return_value = (True, "Content aligned.", {})
    return cv

@pytest.fixture
def mock_metrics_collector():
    """Provides a mock FocusedMetricsCollector instance."""
    mc = MagicMock(spec=FocusedMetricsCollector)
    mc.collect_all_metrics.return_value = {"reasoning_quality": {"schema_validation_failures_count": 0}}
    mc.analyze_historical_effectiveness.return_value = {"total_attempts": 0}
    return mc

@pytest.fixture
def mock_prompt_optimizer():
    """Provides a mock PromptOptimizer instance."""
    po = MagicMock(spec=PromptOptimizer)
    po.optimize_prompt.side_effect = lambda prompt, persona_name, max_output_tokens_for_turn: prompt # Return prompt as is for simplicity
    return po

@pytest.fixture
def socratic_debate_instance(
    mock_persona_manager, mock_gemini_provider, mock_context_analyzer, mock_token_tracker, mock_conflict_manager,
    mock_content_validator, mock_metrics_collector, mock_prompt_optimizer, mock_settings # MODIFIED: Add mock_settings
):
    """Provides a SocraticDebate instance with mocked dependencies."""
    # Patch the constructors of dependencies to return our mocks
    with patch('core.GeminiProvider', return_value=mock_gemini_provider), \
         patch('core.PersonaManager', return_value=mock_persona_manager), \
         patch('core.ContextRelevanceAnalyzer', return_value=mock_context_analyzer), \
         patch('core.TokenUsageTracker', return_value=mock_token_tracker), \
         patch('core.ConflictResolutionManager', return_value=mock_conflict_manager), \
         patch('core.ContentAlignmentValidator', return_value=mock_content_validator), \
         patch('src.self_improvement.metrics_collector.FocusedMetricsCollector', return_value=mock_metrics_collector), \
         patch('core.PromptOptimizer', return_value=mock_prompt_optimizer):
        
        debate = SocraticDebate(
            initial_prompt="Test prompt",
            api_key="mock_api_key",
            model_name="gemini-2.5-flash-lite",
            domain="General",
            persona_manager=mock_persona_manager,
            context_analyzer=mock_context_analyzer,
            token_tracker=mock_token_tracker,
            settings=mock_settings, # MODIFIED: Pass mock_settings
        )
        # Ensure the conflict manager mock is assigned to the instance
        debate.conflict_manager = mock_conflict_manager
        debate.content_validator = mock_content_validator
        debate.prompt_optimizer = mock_prompt_optimizer
        # NEW: Ensure metrics_collector is assigned for self-analysis tests
        debate.metrics_collector = mock_metrics_collector
        # NEW: Mock output_parser on the debate instance
        debate.output_parser = MagicMock(spec=LLMOutputParser)
        debate.output_parser.parse_and_validate.side_effect = lambda raw, schema: {"general_output": raw, "malformed_blocks": []}
        debate.output_parser._create_fallback_output.side_effect = lambda schema, blocks, raw, partial, extracted: {"general_output": "Fallback output", "malformed_blocks": blocks}

        return debate

def test_socratic_debate_initialization(socratic_debate_instance):
    """Tests the basic initialization of the SocraticDebate instance."""
    assert socratic_debate_instance.initial_prompt == "Test prompt"
    assert socratic_debate_instance.model_name == "gemini-2.5-flash-lite"
    assert socratic_debate_instance.max_total_tokens_budget == 100000
    assert socratic_debate_instance.token_tracker.current_usage == 0
    assert isinstance(socratic_debate_instance.conflict_manager, MagicMock)
    assert isinstance(socratic_debate_instance.prompt_optimizer, MagicMock)
    assert isinstance(socratic_debate_instance.settings, ChimeraSettings)
    assert socratic_debate_instance.persona_manager.get_adjusted_persona_config.call_count == 0 # Should not be called during init unless explicitly needed

def test_socratic_debate_run_debate_basic_flow(socratic_debate_instance, mock_gemini_provider, mock_token_tracker):
    """Tests the basic flow of run_debate with mocked LLM responses."""
    # Mock the persona sequence to be simple for this test
    socratic_debate_instance.persona_router.determine_persona_sequence.return_value = [
        "Visionary_Generator", "Skeptical_Generator", "Impartial_Arbitrator"
    ]
    
    # Mock LLM responses for each persona turn, including is_truncated
    mock_gemini_provider.generate.side_effect = [
        ("{'general_output': 'Visionary idea'}", 50, 100, False), # Turn 1
        ("{'general_output': 'Skeptical critique'}", 60, 110, False), # Turn 2
        ("{'general_output': 'Final synthesis'}", 70, 120, False), # Turn 3 (Synthesis)
    ]

    # Mock the parser to return valid output for each turn
    socratic_debate_instance.output_parser.parse_and_validate.side_effect = [
        ({'general_output': 'Visionary idea', 'malformed_blocks': []}),
        ({'general_output': 'Skeptical critique', 'malformed_blocks': []}),
        ({'general_output': 'Final synthesis', 'malformed_blocks': []}),
    ]

    final_answer, intermediate_steps = socratic_debate_instance.run_debate()

    assert isinstance(final_answer, dict)
    assert (
        "general_output" in final_answer
    )  # Adjust based on expected output structure for General domain
    assert (
        "Final synthesis" in final_answer["general_output"]
    )  # Basic check

    # Verify token usage tracking
    assert intermediate_steps["Total_Tokens_Used"] > 0
    assert mock_token_tracker.current_usage > 0
    
    # Verify LLM calls and parsing calls
    assert mock_gemini_provider.generate.call_count == 3 # 3 personas in sequence
    assert socratic_debate_instance.output_parser.parse_and_validate.call_count == 3 # Parser called for each turn

def test_socratic_debate_malformed_output_triggers_conflict_manager(socratic_debate_instance, mock_gemini_provider, mock_conflict_manager):
    """Tests that malformed output triggers the conflict manager."""
    socratic_debate_instance.persona_router.determine_persona_sequence.return_value = [
        "Visionary_Generator", "Constructive_Critic", "Impartial_Arbitrator"
    ]

    # Simulate a malformed output from Constructive_Critic
    malformed_output_str = "This is not valid JSON output from critic"
    mock_gemini_provider.generate.side_effect = [
        ("{'general_output': 'Visionary idea'}", 50, 100, False), # Turn 1 (Valid)
        (malformed_output_str, 60, 110, False), # Turn 2 (Malformed)
        ("{'general_output': 'Final synthesis'}", 70, 120, False), # Turn 3 (Synthesis)
    ]
    
    # Mock the parser to return malformed_blocks for the bad output
    socratic_debate_instance.output_parser.parse_and_validate.side_effect = [
        ({'general_output': 'Visionary idea', 'malformed_blocks': []}), # Valid output
        ({'malformed_blocks': [{'type': 'JSON_DECODE_ERROR', 'message': 'Invalid JSON'}]}), # Malformed output
        ({'general_output': 'Final synthesis', 'malformed_blocks': []}), # Valid output
    ]

    # Configure the mock conflict manager to return a successful resolution
    mock_conflict_manager.resolve_conflict.return_value = {
        "resolution_strategy": "mock_resolved",
        "resolved_output": {"general_output": "Mock resolved output from conflict"},
        "resolution_summary": "Successfully resolved conflict.",
        "malformed_blocks": [{"type": "MOCK_RESOLUTION", "message": "Mock resolution applied."}]
    }

    final_answer, intermediate_steps = socratic_debate_instance.run_debate()

    # Assert that conflict manager was called exactly once
    mock_conflict_manager.resolve_conflict.assert_called_once()
    
    # Assert that the malformed block was recorded
    assert any(block['type'] == 'JSON_DECODE_ERROR' for block in intermediate_steps.get('malformed_blocks', [])), "Malformed block not recorded"
    
    # Assert that the resolved output from the conflict manager was used in the history
    # The history should contain an entry for Conflict_Resolution_Manager
    conflict_manager_turn = next((t for t in intermediate_steps.get('Debate_History', []) if t.get('persona') == 'Conflict_Resolution_Manager'), None)
    assert conflict_manager_turn is not None, "Conflict resolution turn not found in history"
    assert "Mock resolved output from conflict" in conflict_manager_turn['output'].get('resolved_output', {}).get('general_output', '')

def test_socratic_debate_token_budget_exceeded(socratic_debate_instance, mock_gemini_provider, mock_token_tracker):
    """Tests that TokenBudgetExceededError is raised when the budget is exceeded."""
    socratic_debate_instance.persona_router.determine_persona_sequence.return_value = [
        "Visionary_Generator", "Skeptical_Generator", "Impartial_Arbitrator"
    ]
    
    # Set a very small budget to trigger the error quickly
    socratic_debate_instance.max_total_tokens_budget = 150
    mock_token_tracker.budget = 150
    mock_token_tracker.current_usage = 0 # Ensure it starts at 0

    # Simulate LLM responses that will exceed the budget
    mock_gemini_provider.generate.side_effect = [
        ("{'general_output': 'Visionary idea'}", 50, 40, False), # Turn 1: 90 tokens used
        ("{'general_output': 'Skeptical critique'}", 60, 50, False), # Turn 2: Tries to use 110 tokens (90 + 110 = 200 > 150)
    ]
    
    # Mock the parser to return valid output for each turn
    socratic_debate_instance.output_parser.parse_and_validate.side_effect = [
        ({'general_output': 'Visionary idea', 'malformed_blocks': []}),
        ({'general_output': 'Skeptical critique', 'malformed_blocks': []}),
    ]

    # Expecting a TokenBudgetExceededError to be raised
    with pytest.raises(TokenBudgetExceededError) as excinfo:
        socratic_debate_instance.run_debate()
    
    # Assert that the error message contains relevant details
    assert "Token budget exceeded" in str(excinfo.value)
    assert "90/150 tokens used" in str(excinfo.value) # Check current usage before failure
    assert "step_name='Skeptical_Generator'" in str(excinfo.value) # Check context

    # Verify that the token tracker recorded usage up to the point of failure
    assert mock_token_tracker.current_usage == 90 # Usage from the first turn

    # Ensure the generate call was made for the second turn before the exception
    assert mock_gemini_provider.generate.call_count == 2

def test_execute_llm_turn_schema_validation_retry(socratic_debate_instance, mock_gemini_provider):
    """Tests that _execute_llm_turn retries on SchemaValidationError."""
    persona_name = "Constructive_Critic"
    prompt_for_llm = "Critique this."
    phase = "debate"
    max_output_tokens_for_turn = 1000

    # Simulate initial malformed output, then a valid one on retry
    mock_gemini_provider.generate.side_effect = [
        ("Malformed JSON string", 50, 50, False), # First attempt: malformed
        ("{'CRITIQUE_SUMMARY': 'Valid critique', 'CRITIQUE_POINTS': [], 'SUGGESTIONS': []}", 60, 60, False), # Second attempt: valid
    ]

    # Mock the parser to raise SchemaValidationError on first call, then succeed
    socratic_debate_instance.output_parser.parse_and_validate.side_effect = [
        # First call: simulate schema validation failure
        SchemaValidationError(
            error_type="validation_error",
            field_path="CRITIQUE_SUMMARY",
            invalid_value="missing",
            details={"message": "Missing required field"}
        ),
        # Second call: simulate success
        ({'CRITIQUE_SUMMARY': 'Valid critique', 'CRITIQUE_POINTS': [], 'SUGGESTIONS': [], 'malformed_blocks': []}),
    ]

    output = socratic_debate_instance._execute_llm_turn(
        persona_name, prompt_for_llm, phase, max_output_tokens_for_turn, max_retries=1
    )

    # Assert that generate was called twice (initial + 1 retry)
    assert mock_gemini_provider.generate.call_count == 2
    # Assert that parse_and_validate was called twice
    assert socratic_debate_instance.output_parser.parse_and_validate.call_count == 2
    # Assert that the final output is the valid one
    assert output["CRITIQUE_SUMMARY"] == "Valid critique"
    # Assert that a malformed block for retry was recorded
    assert any(block['type'] == 'RETRYABLE_VALIDATION_ERROR' for block in socratic_debate_instance.intermediate_steps.get('malformed_blocks', []))

def test_socratic_debate_self_analysis_flow(socratic_debate_instance, mock_gemini_provider, mock_persona_manager, mock_metrics_collector):
    """Tests the self-analysis flow, ensuring Self_Improvement_Analyst is used."""
    socratic_debate_instance.is_self_analysis = True
    mock_persona_manager.prompt_analyzer.is_self_analysis_prompt.return_value = True
    mock_persona_manager.persona_router.determine_persona_sequence.return_value = ["Self_Improvement_Analyst"]

    # Mock LLM response for Self_Improvement_Analyst
    mock_gemini_provider.generate.return_value = (
        """{"ANALYSIS_SUMMARY": "Self-analysis complete.", "IMPACTFUL_SUGGESTIONS": []}""",
        100, 150, False
    )
    # Mock the parser for SelfImprovementAnalysisOutputV1
    socratic_debate_instance.output_parser.parse_and_validate.side_effect = [
        # Context_Aware_Assistant (if in sequence)
        ({'general_overview': 'Context overview', 'malformed_blocks': []}),
        # Self_Improvement_Analyst
        ({'ANALYSIS_SUMMARY': 'Self-analysis complete.', 'IMPACTFUL_SUGGESTIONS': [], 'malformed_blocks': []}),
    ]

    final_answer, intermediate_steps = socratic_debate_instance.run_debate()

    assert isinstance(final_answer, dict)
    assert final_answer.get("version") == "1.0"
    assert final_answer.get("data", {}).get("ANALYSIS_SUMMARY") == "Self-analysis complete."
    assert "Self_Improvement_Metrics" in intermediate_steps
    mock_metrics_collector.collect_all_metrics.assert_called_once()
    # Ensure the correct synthesis persona was used
    assert "Self_Improvement_Analyst_Output" in intermediate_steps

def test_socratic_debate_context_aware_assistant_turn(socratic_debate_instance, mock_gemini_provider, mock_context_analyzer):
    """Tests the Context_Aware_Assistant turn when present in the sequence."""
    socratic_debate_instance.persona_router.determine_persona_sequence.return_value = [
        "Context_Aware_Assistant", "Impartial_Arbitrator"
    ]
    mock_context_analyzer.find_relevant_files.return_value = [("file1.py", 0.8)]
    mock_context_analyzer.generate_context_summary.return_value = "Detailed context for file1.py"
    socratic_debate_instance.codebase_context = {"file1.py": "def func(): pass"}

    # Mock LLM response for Context_Aware_Assistant
    mock_gemini_provider.generate.side_effect = [
        ("""{"general_overview": "Context analysis output"}""", 80, 120, False), # Context_Aware_Assistant
        ("""{"general_output": "Final synthesis"}""", 70, 120, False), # Impartial_Arbitrator
    ]
    socratic_debate_instance.output_parser.parse_and_validate.side_effect = [
        ({'general_overview': 'Context analysis output', 'malformed_blocks': []}),
        ({'general_output': 'Final synthesis', 'malformed_blocks': []}),
    ]

    final_answer, intermediate_steps = socratic_debate_instance.run_debate()

    assert "Context_Aware_Assistant_Output" in intermediate_steps
    assert intermediate_steps["Context_Aware_Assistant_Output"].get("general_overview") == "Context analysis output"
    mock_context_analyzer.find_relevant_files.assert_called_once()
    mock_context_analyzer.generate_context_summary.assert_called_once()