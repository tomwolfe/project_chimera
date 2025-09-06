# tests/test_persona_routing.py
import pytest
from unittest.mock import MagicMock, patch
import numpy as np
from src.persona.routing import PersonaRouter
from src.models import PersonaConfig
from src.utils.prompt_analyzer import PromptAnalyzer
from src.persona_manager import PersonaManager # Import PersonaManager for the new persona_manager argument

@pytest.fixture
def mock_prompt_analyzer():
    """Provides a mock PromptAnalyzer instance."""
    pa = MagicMock(spec=PromptAnalyzer)
    pa.is_self_analysis_prompt.return_value = False
    pa.recommend_domain_from_keywords.return_value = "General"
    return pa

@pytest.fixture
def mock_persona_manager():
    """Provides a mock PersonaManager instance."""
    pm = MagicMock(spec=PersonaManager)
    pm.all_personas = {
        "Visionary_Generator": PersonaConfig(name="Visionary_Generator", system_prompt="Visionary", temperature=0.7, max_tokens=1024, description="Generates innovative solutions."),
        "Skeptical_Generator": PersonaConfig(name="Skeptical_Generator", system_prompt="Skeptical", temperature=0.3, max_tokens=1024, description="Identifies flaws."),
        "Impartial_Arbitrator": PersonaConfig(name="Impartial_Arbitrator", system_prompt="Arbitrator", temperature=0.1, max_tokens=4096, description="Synthesizes outcomes."),
        "Self_Improvement_Analyst": PersonaConfig(name="Self_Improvement_Analyst", system_prompt="Analyst", temperature=0.1, max_tokens=4096, description="Analyzes for improvements."),
        "Constructive_Critic": PersonaConfig(name="Constructive_Critic", system_prompt="Critic", temperature=0.15, max_tokens=8192, description="Provides constructive feedback."),
        "Devils_Advocate": PersonaConfig(name="Devils_Advocate", system_prompt="Devils Advocate", temperature=0.15, max_tokens=4096, description="Challenges proposals."),
        "Code_Architect": PersonaConfig(name="Code_Architect", system_prompt="Architect", temperature=0.4, max_tokens=4096, description="Analyzes architecture."),
        "Security_Auditor": PersonaConfig(name="Security_Auditor", system_prompt="Security", temperature=0.2, max_tokens=4096, description="Identifies security vulnerabilities."),
        "DevOps_Engineer": PersonaConfig(name="DevOps_Engineer", system_prompt="DevOps", temperature=0.3, max_tokens=4096, description="Focuses on operations."),
        "Test_Engineer": PersonaConfig(name="Test_Engineer", system_prompt="Test", temperature=0.3, max_tokens=4096, description="Ensures quality through testing."),
    }
    pm.persona_sets = {
        "General": ["Visionary_Generator", "Skeptical_Generator", "Impartial_Arbitrator"],
        "Software Engineering": ["Visionary_Generator", "Skeptical_Generator", "Code_Architect", "Security_Auditor", "DevOps_Engineer", "Test_Engineer", "Constructive_Critic", "Devils_Advocate", "Impartial_Arbitrator"],
        "Self-Improvement": ["Self_Improvement_Analyst", "Code_Architect", "Security_Auditor", "DevOps_Engineer", "Test_Engineer", "Constructive_Critic", "Devils_Advocate", "Impartial_Arbitrator"],
    }
    # Mock performance metrics for dynamic re-ordering tests
    pm.persona_performance_metrics = {
        "Visionary_Generator": {"total_turns": 10, "schema_failures": 0, "truncation_failures": 0, "last_adjustment_timestamp": time.time() - 600}, # Good performance
        "Skeptical_Generator": {"total_turns": 10, "schema_failures": 5, "truncation_failures": 0, "last_adjustment_timestamp": time.time() - 600}, # Bad performance
        "Constructive_Critic": {"total_turns": 10, "schema_failures": 0, "truncation_failures": 0, "last_adjustment_timestamp": time.time() - 600}, # Good performance
        "Devils_Advocate": {"total_turns": 10, "schema_failures": 0, "truncation_failures": 0, "last_adjustment_timestamp": time.time() - 600}, # Good performance
        "Impartial_Arbitrator": {"total_turns": 10, "schema_failures": 0, "truncation_failures": 0, "last_adjustment_timestamp": time.time() - 600}, # Good performance
        "Self_Improvement_Analyst": {"total_turns": 10, "schema_failures": 0, "truncation_failures": 0, "last_adjustment_timestamp": time.time() - 600}, # Good performance
        "Code_Architect": {"total_turns": 10, "schema_failures": 0, "truncation_failures": 0, "last_adjustment_timestamp": time.time() - 600}, # Good performance
        "Security_Auditor": {"total_turns": 10, "schema_failures": 0, "truncation_failures": 0, "last_adjustment_timestamp": time.time() - 600}, # Good performance
        "DevOps_Engineer": {"total_turns": 10, "schema_failures": 0, "truncation_failures": 0, "last_adjustment_timestamp": time.time() - 600}, # Good performance
        "Test_Engineer": {"total_turns": 10, "schema_failures": 0, "truncation_failures": 0, "last_adjustment_timestamp": time.time() - 600}, # Good performance
    }
    pm.min_turns_for_adjustment = 5
    pm.adjustment_cooldown_seconds = 300
    return pm


@pytest.fixture
def persona_router_instance(mock_prompt_analyzer, mock_persona_manager):
    """Provides a PersonaRouter instance with mocked dependencies."""
    # Patch SentenceTransformer to avoid actual model loading during tests
    with patch('src.persona.routing.SentenceTransformer') as MockSentenceTransformer:
        # Configure the mock to return a dummy encoder that just returns a fixed vector
        MockSentenceTransformer.return_value.encode.return_value = np.array([0.1, 0.2, 0.3])
        
        router = PersonaRouter(
            all_personas=mock_persona_manager.all_personas,
            persona_sets=mock_persona_manager.persona_sets,
            prompt_analyzer=mock_prompt_analyzer,
            persona_manager=mock_persona_manager # Pass the mock persona manager
        )
        # Manually set persona embeddings to avoid complex mocking of encode for each persona
        router.persona_embeddings = {
            name: np.array([i * 0.1, (i + 1) * 0.1, (i + 2) * 0.1])
            for i, name in enumerate(mock_persona_manager.all_personas.keys())
        }
        return router

def test_determine_persona_sequence_general_domain(persona_router_instance):
    """Tests persona sequence determination for a general domain prompt."""
    prompt = "Tell me about general knowledge."
    domain = "General"
    
    # Mock prompt_analyzer to return General domain
    persona_router_instance.prompt_analyzer.recommend_domain_from_keywords.return_value = "General"
    persona_router_instance.prompt_analyzer.is_self_analysis_prompt.return_value = False

    sequence = persona_router_instance.determine_persona_sequence(prompt, domain)
    expected_sequence = ["Visionary_Generator", "Skeptical_Generator", "Impartial_Arbitrator"]
    assert all(p in sequence for p in expected_sequence)
    assert len(sequence) == len(expected_sequence)

def test_determine_persona_sequence_self_analysis_domain(persona_router_instance):
    """Tests persona sequence determination for a self-analysis prompt."""
    prompt = "Critically analyze the entire Project Chimera codebase for self-improvement."
    domain = "Self-Improvement"
    
    # Mock prompt_analyzer to detect self-analysis
    persona_router_instance.prompt_analyzer.is_self_analysis_prompt.return_value = True
    persona_router_instance.prompt_analyzer.recommend_domain_from_keywords.return_value = "Self-Improvement"

    sequence = persona_router_instance.determine_persona_sequence(prompt, domain)
    
    # Expected personas for self-improvement, with Self_Improvement_Analyst at the end
    expected_core_personas = [
        "Self_Improvement_Analyst", "Code_Architect", "Security_Auditor",
        "DevOps_Engineer", "Test_Engineer", "Constructive_Critic",
        "Devils_Advocate", "Impartial_Arbitrator"
    ]
    
    # Ensure all expected personas are present
    assert all(p in sequence for p in expected_core_personas)
    # Ensure Self_Improvement_Analyst is the last one (or near the end for synthesis)
    assert sequence[-1] == "Self_Improvement_Analyst"
    # Ensure Impartial_Arbitrator is before Self_Improvement_Analyst
    assert sequence.index("Impartial_Arbitrator") < sequence.index("Self_Improvement_Analyst")
    # Ensure Devils_Advocate is before Impartial_Arbitrator
    assert sequence.index("Devils_Advocate") < sequence.index("Impartial_Arbitrator")


def test_determine_persona_sequence_software_engineering_domain_with_keywords(persona_router_instance):
    """Tests persona sequence for software engineering with specific keywords."""
    prompt = "Implement a new API endpoint with robust security and comprehensive tests."
    domain = "Software Engineering"
    
    persona_router_instance.prompt_analyzer.is_self_analysis_prompt.return_value = False
    persona_router_instance.prompt_analyzer.recommend_domain_from_keywords.return_value = "Software Engineering"

    sequence = persona_router_instance.determine_persona_sequence(prompt, domain)
    
    # Expect Security_Auditor and Test_Engineer to be prioritized
    assert "Security_Auditor" in sequence
    assert "Test_Engineer" in sequence
    assert sequence.index("Security_Auditor") < sequence.index("Impartial_Arbitrator")
    assert sequence.index("Test_Engineer") < sequence.index("Impartial_Arbitrator")
    assert sequence.index("Code_Architect") < sequence.index("Impartial_Arbitrator") # Code_Architect is also core to SE

def test_apply_dynamic_adjustment_security_concerns(persona_router_instance):
    """Tests dynamic adjustment when security concerns are present in context."""
    prompt = "Analyze the code."
    domain = "Software Engineering"
    context_analysis_results = {
        "security_concerns": ["SQL Injection"],
        "key_modules": []
    }
    
    persona_router_instance.prompt_analyzer.is_self_analysis_prompt.return_value = False
    persona_router_instance.prompt_analyzer.recommend_domain_from_keywords.return_value = "Software Engineering"

    initial_sequence = persona_router_instance.persona_sets["Software Engineering"].copy()
    adjusted_sequence = persona_router_instance._apply_dynamic_adjustment(
        initial_sequence, {}, prompt.lower(), domain, context_analysis_results
    )
    
    assert "Security_Auditor" in adjusted_sequence
    # Ensure Security_Auditor is before Impartial_Arbitrator
    assert adjusted_sequence.index("Security_Auditor") < adjusted_sequence.index("Impartial_Arbitrator")

def test_apply_dynamic_adjustment_low_code_quality(persona_router_instance):
    """Tests dynamic adjustment when low code quality is detected in context."""
    prompt = "Refactor the module."
    domain = "Software Engineering"
    context_analysis_results = {
        "security_concerns": [],
        "key_modules": [{"name": "module.py", "code_quality_score": 0.6, "complexity_score": 0.8}]
    }
    
    persona_router_instance.prompt_analyzer.is_self_analysis_prompt.return_value = False
    persona_router_instance.prompt_analyzer.recommend_domain_from_keywords.return_value = "Software Engineering"

    initial_sequence = persona_router_instance.persona_sets["Software Engineering"].copy()
    adjusted_sequence = persona_router_instance._apply_dynamic_adjustment(
        initial_sequence, {}, prompt.lower(), domain, context_analysis_results
    )
    
    assert "Code_Architect" in adjusted_sequence
    # Ensure Code_Architect is before Impartial_Arbitrator
    assert adjusted_sequence.index("Code_Architect") < adjusted_sequence.index("Impartial_Arbitrator")

def test_apply_dynamic_adjustment_test_engineer_inclusion(persona_router_instance):
    """Tests dynamic adjustment for Test_Engineer inclusion based on prompt keywords."""
    prompt = "Ensure high test coverage for the new feature."
    domain = "Software Engineering"
    
    persona_router_instance.prompt_analyzer.is_self_analysis_prompt.return_value = False
    persona_router_instance.prompt_analyzer.recommend_domain_from_keywords.return_value = "Software Engineering"

    initial_sequence = [p for p in persona_router_instance.persona_sets["Software Engineering"] if p != "Test_Engineer"] # Start without Test_Engineer
    adjusted_sequence = persona_router_instance._apply_dynamic_adjustment(
        initial_sequence, {}, prompt.lower(), domain, {}
    )
    
    assert "Test_Engineer" in adjusted_sequence
    assert adjusted_sequence.index("Test_Engineer") < adjusted_sequence.index("Impartial_Arbitrator")

def test_apply_dynamic_adjustment_test_engineer_removal(persona_router_instance):
    """Tests dynamic adjustment for Test_Engineer removal when no relevant keywords."""
    prompt = "Design a new system architecture."
    domain = "Software Engineering"
    
    persona_router_instance.prompt_analyzer.is_self_analysis_prompt.return_value = False
    persona_router_instance.prompt_analyzer.recommend_domain_from_keywords.return_value = "Software Engineering"

    initial_sequence = persona_router_instance.persona_sets["Software Engineering"].copy() # Start with Test_Engineer
    adjusted_sequence = persona_router_instance._apply_dynamic_adjustment(
        initial_sequence, {}, prompt.lower(), domain, {}
    )
    
    assert "Test_Engineer" not in adjusted_sequence # Should be removed

def test_dynamic_reordering_based_on_performance(persona_router_instance, mock_persona_manager):
    """Tests dynamic re-ordering of personas based on performance metrics."""
    prompt = "Analyze and improve the system."
    domain = "Software Engineering"

    # Simulate Skeptical_Generator having bad performance (high schema failures)
    mock_persona_manager.persona_performance_metrics["Skeptical_Generator"]["schema_failures"] = 8
    mock_persona_manager.persona_performance_metrics["Skeptical_Generator"]["total_turns"] = 10
    mock_persona_manager.persona_performance_metrics["Skeptical_Generator"]["last_adjustment_timestamp"] = time.time() - 600 # Not in cooldown

    # Simulate Visionary_Generator having good performance
    mock_persona_manager.persona_performance_metrics["Visionary_Generator"]["schema_failures"] = 0
    mock_persona_manager.persona_performance_metrics["Visionary_Generator"]["total_turns"] = 10
    mock_persona_manager.persona_performance_metrics["Visionary_Generator"]["last_adjustment_timestamp"] = time.time() - 600 # Not in cooldown

    persona_router_instance.prompt_analyzer.is_self_analysis_prompt.return_value = False
    persona_router_instance.prompt_analyzer.recommend_domain_from_keywords.return_value = "Software Engineering"

    # Get the initial sequence (should contain both Visionary and Skeptical)
    initial_sequence = persona_router_instance.persona_sets["Software Engineering"].copy()
    
    # Determine the dynamically adjusted sequence
    adjusted_sequence = persona_router_instance.determine_persona_sequence(prompt, domain)

    # Expect Visionary_Generator to be prioritized over Skeptical_Generator
    assert adjusted_sequence.index("Visionary_Generator") < adjusted_sequence.index("Skeptical_Generator")
    
    # Ensure synthesis personas remain at the end
    assert adjusted_sequence[-1] == "Impartial_Arbitrator"
    assert adjusted_sequence[-2] == "Devils_Advocate" # Devils_Advocate is before Arbitrator in SE set
