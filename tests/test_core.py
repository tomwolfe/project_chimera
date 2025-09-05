# tests/test_core.py

import pytest
from unittest.mock import MagicMock, patch

# Assuming SocraticDebate is in core.py
from core import SocraticDebate
from src.models import PersonaConfig
from src.config.settings import ChimeraSettings
from src.token_tracker import TokenUsageTracker
from src.persona_manager import PersonaManager
from src.context.context_analyzer import ContextRelevanceAnalyzer
from src.llm_provider import GeminiProvider
from src.conflict_resolution import ConflictResolutionManager # NEW: Import ConflictResolutionManager
from src.utils.output_parser import LLMOutputParser # Import LLMOutputParser for mocking

# Mock necessary dependencies
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
    return pm

@pytest.fixture
def mock_gemini_provider():
    """Provides a mock GeminiProvider instance."""
    gp = MagicMock(spec=GeminiProvider)
    gp.tokenizer = MagicMock()
    # Simple token count approximation for testing
    gp.tokenizer.count_tokens.side_effect = lambda text: len(text) // 4 if text else 0
    # Simple truncation simulation
    gp.tokenizer.truncate_to_token_limit.side_effect = lambda text, max_tokens, indicator="": text[:max_tokens*4] + indicator if len(text) > max_tokens*4 else text
    gp.tokenizer.max_output_tokens = 65536
    # Mock generate method to return a tuple: (raw_output, input_tokens, output_tokens, is_truncated)
    gp.generate.return_value = ("{'general_output': 'Mocked LLM response'}", 50, 100, False)
    gp.calculate_usd_cost.return_value = 0.001
    return gp

@pytest.fixture
def mock_context_analyzer():
    """Provides a mock ContextRelevanceAnalyzer instance."""
    ca = MagicMock(spec=ContextRelevanceAnalyzer)
    ca.codebase_context = {}
    ca.file_embeddings = {}
    ca.find_relevant_files.return_value = []
    ca.generate_context_summary.return_value = "Mock context summary."
    ca.set_persona_router = MagicMock() # Mock this method as it's called in SocraticDebate init
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
def mock_conflict_manager():
    """Provides a mock ConflictResolutionManager instance."""
    cm = MagicMock(spec=ConflictResolutionManager)
    # Configure resolve_conflict to return a successful mock resolution
    cm.resolve_conflict.return_value = {
        "resolution_strategy": "mock_resolved",
        "resolved_output": {"general_output": "Mock resolved output"},
        "resolution_summary": "Mock resolution summary.",
        "malformed_blocks": [{"type": "MOCK_RESOLUTION", "message": "Mock resolution applied."}]
    }
    return cm

@pytest.fixture
def socratic_debate_instance(
    mock_persona_manager,
    mock_gemini_provider,
    mock_context_analyzer,
    mock_token_tracker,
    mock_conflict_manager,
):
    """Provides a SocraticDebate instance with mocked dependencies."""
    # Patch the constructors of dependencies to return our mocks
    with patch('core.GeminiProvider', return_value=mock_gemini_provider), \
         patch('core.PersonaManager', return_value=mock_persona_manager), \
         patch('core.ContextRelevanceAnalyzer', return_value=mock_context_analyzer), \
         patch('core.TokenUsageTracker', return_value=mock_token_tracker), \
         patch('core.ConflictResolutionManager', return_value=mock_conflict_manager): # Patch the new manager
        
        settings = ChimeraSettings(total_budget=100000)
        debate = SocraticDebate(
            initial_prompt="Test prompt",
            api_key="mock_api_key",
            model_name="gemini-2.5-flash-lite",
            domain="General",
            persona_manager=mock_persona_manager,
            context_analyzer=mock_context_analyzer,
            token_tracker=mock_token_tracker,
            settings=settings, # Pass settings
        )
        # Ensure the conflict manager mock is assigned to the instance
        debate.conflict_manager = mock_conflict_manager
        return debate

def test_socratic_debate_initialization(socratic_debate_instance):
    """Tests the basic initialization of the SocraticDebate instance."""
    assert socratic_debate_instance.initial_prompt == "Test prompt"
    assert socratic_debate_instance.model_name == "gemini-2.5-flash-lite"
    assert socratic_debate_instance.max_total_tokens_budget == 100000
    assert socratic_debate_instance.token_tracker.current_usage == 0
    assert isinstance(socratic_debate_instance.conflict_manager, MagicMock)
    assert socratic_debate_instance.persona_manager.get_adjusted_persona_config.call_count == 0 # Should not be called during init unless explicitly needed

def test_socratic_debate_run_debate_basic_flow(socratic_debate_instance, mock_gemini_provider, mock_token_tracker):
    """Tests the basic flow of run_debate with mocked LLM responses."""
    # Mock the persona sequence to be simple for this test
    socratic_debate_instance.persona_router.determine_persona_sequence.return_value = [
        "Visionary_Generator", "Skeptical_Generator", "Impartial_Arbitrator"
    ]
    
    # Mock LLM responses for each persona turn
    mock_gemini_provider.generate.side_effect = [
        ("{'general_output': 'Visionary idea'}", 50, 100, False), # Turn 1
        ("{'general_output': 'Skeptical critique'}", 60, 110, False), # Turn 2
        ("{'general_output': 'Final synthesis'}", 70, 120, False), # Turn 3 (Synthesis)
    ]

    # Mock the parser to return valid output for each turn
    with patch('src.utils.output_parser.LLMOutputParser.parse_and_validate') as mock_parse:
        mock_parse.side_effect = [
            ({'general_output': 'Visionary idea', 'malformed_blocks': []}),
            ({'general_output': 'Skeptical critique', 'malformed_blocks': []}),
            ({'general_output': 'Final synthesis', 'malformed_blocks': []}),
        ]

        final_answer, intermediate_steps = socratic_debate_instance.run_debate()

        assert isinstance(final_answer, dict)
        assert "general_output" in final_answer
        assert "Final synthesis" in final_answer["general_output"]
        assert intermediate_steps["Total_Tokens_Used"] > 0
        assert mock_token_tracker.current_usage > 0
        assert mock_gemini_provider.generate.call_count == 3 # 3 personas in sequence
        assert mock_parse.call_count == 3 # Parser called for each turn

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
    with patch('src.utils.output_parser.LLMOutputParser.parse_and_validate') as mock_parse:
        mock_parse.side_effect = [
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

        # Assert that the final answer reflects the resolution (if it was used in subsequent steps)
        # In this mock, the final synthesis uses the output of the last persona turn, which should be the resolved one if it replaced the problematic one.
        # The current logic appends the resolution, so the final answer might not directly reflect it unless the synthesis persona uses the history.
        # For this test, we check that the conflict manager's output was recorded.

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
    with patch('src.utils.output_parser.LLMOutputParser.parse_and_validate') as mock_parse:
        mock_parse.side_effect = [
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