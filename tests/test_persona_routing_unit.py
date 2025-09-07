# tests/test_persona_routing_unit.py
import pytest
from unittest.mock import MagicMock, patch
import numpy as np
from src.persona.routing import PersonaRouter
from src.models import PersonaConfig
from src.utils.prompt_analyzer import PromptAnalyzer
from src.persona_manager import PersonaManager
import time

@pytest.fixture(autouse=True)
def mock_prompt_analyzer_autouse():
    """Provides a mock PromptAnalyzer instance, autoused for all tests in this module."""
    pa = MagicMock(spec=PromptAnalyzer)
    pa.is_self_analysis_prompt.return_value = False
    pa.recommend_domain_from_keywords.return_value = "General"
    return pa

@pytest.fixture
def mock_persona_manager_for_router():
    """Provides a mock PersonaManager instance for PersonaRouter tests."""
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
    pm.persona_performance_metrics = {
        "Visionary_Generator": {"total_turns": 10, "schema_failures": 0, "truncation_failures": 0, "last_adjustment_timestamp": time.time() - 600},
        "Skeptical_Generator": {"total_turns": 10, "schema_failures": 5, "truncation_failures": 0, "last_adjustment_timestamp": time.time() - 600},
        "Constructive_Critic": {"total_turns": 10, "schema_failures": 0, "truncation_failures": 0, "last_adjustment_timestamp": time.time() - 600},
        "Devils_Advocate": {"total_turns": 10, "schema_failures": 0, "truncation_failures": 0, "last_adjustment_timestamp": time.time() - 600},
        "Impartial_Arbitrator": {"total_turns": 10, "schema_failures": 0, "truncation_failures": 0, "last_adjustment_timestamp": time.time() - 600},
        "Self_Improvement_Analyst": {"total_turns": 10, "schema_failures": 0, "truncation_failures": 0, "last_adjustment_timestamp": time.time() - 600},
        "Code_Architect": {"total_turns": 10, "schema_failures": 0, "truncation_failures": 0, "last_adjustment_timestamp": time.time() - 600},
        "Security_Auditor": {"total_turns": 10, "schema_failures": 0, "truncation_failures": 0, "last_adjustment_timestamp": time.time() - 600},
        "DevOps_Engineer": {"total_turns": 10, "schema_failures": 0, "truncation_failures": 0, "last_adjustment_timestamp": time.time() - 600},
        "Test_Engineer": {"total_turns": 10, "schema_failures": 0, "truncation_failures": 0, "last_adjustment_timestamp": time.time() - 600},
    }
    pm.min_turns_for_adjustment = 5
    pm.adjustment_cooldown_seconds = 300
    return pm


@pytest.fixture
def persona_router_instance(mock_prompt_analyzer_autouse, mock_persona_manager_for_router):
    """Provides a PersonaRouter instance with mocked dependencies."""
    with patch('src.persona.routing.SentenceTransformer') as MockSentenceTransformer:
        MockSentenceTransformer.return_value.encode.return_value = np.array([0.1, 0.2, 0.3])
        
        router = PersonaRouter(
            all_personas=mock_persona_manager_for_router.all_personas,
            persona_sets=mock_persona_manager_for_router.persona_sets,
            prompt_analyzer=mock_prompt_analyzer_autouse,
            persona_manager=mock_persona_manager_for_router
        )
        router.persona_embeddings = {
            name: np.array([i * 0.1, (i + 1) * 0.1, (i + 2) * 0.1])
            for i, name in enumerate(mock_persona_manager_for_router.all_personas.keys())
        }
        return router

def test_determine_persona_sequence_general_domain(persona_router_instance):
    """Tests persona sequence determination for a general domain prompt."""
    prompt = "Tell me about general knowledge."
    domain = "General"
    sequence = persona_router_instance.determine_persona_sequence(prompt, domain)
    expected_sequence = ["Visionary_Generator", "Skeptical_Generator", "Impartial_Arbitrator"]
    assert all(p in sequence for p in expected_sequence)
    assert len(sequence) == len(expected_sequence)

def test_determine_persona_sequence_self_analysis_domain(persona_router_instance):
    """Tests persona sequence determination for a self-analysis prompt."""
    prompt = "Critically analyze the entire Project Chimera codebase for self-improvement."
    domain = "Self-Improvement"
    persona_router_instance.prompt_analyzer.is_self_analysis_prompt.return_value = True
    sequence = persona_router_instance.determine_persona_sequence(prompt, domain)
    
    expected_core_personas = [
        "Self_Improvement_Analyst", "Code_Architect", "Security_Auditor",
        "DevOps_Engineer", "Test_Engineer", "Constructive_Critic",
        "Devils_Advocate", "Impartial_Arbitrator"
    ]
    
    assert all(p in sequence for p in expected_core_personas)
    assert sequence[-1] == "Self_Improvement_Analyst"
    assert sequence.index("Impartial_Arbitrator") < sequence.index("Self_Improvement_Analyst")
    assert sequence.index("Devils_Advocate") < sequence.index("Impartial_Arbitrator")


def test_determine_persona_sequence_software_engineering_domain_with_keywords(persona_router_instance):
    """Tests persona sequence for software engineering with specific keywords."""
    prompt = "Implement a new API endpoint with robust security and comprehensive tests."
    domain = "Software Engineering"
    sequence = persona_router_instance.determine_persona_sequence(prompt, domain)
    
    assert "Security_Auditor" in sequence
    assert "Test_Engineer" in sequence
    assert sequence.index("Security_Auditor") < sequence.index("Impartial_Arbitrator")
    assert sequence.index("Test_Engineer") < sequence.index("Impartial_Arbitrator")
    assert sequence.index("Code_Architect") < sequence.index("Impartial_Arbitrator")

def test_apply_dynamic_adjustment_security_concerns(persona_router_instance):
    """Tests dynamic adjustment when security concerns are present in context."""
    prompt = "Analyze the code."
    domain = "Software Engineering"
    context_analysis_results = {"security_concerns": ["SQL Injection"], "key_modules": []}
    initial_sequence = persona_router_instance.persona_sets["Software Engineering"].copy()
    adjusted_sequence = persona_router_instance._apply_dynamic_adjustment(
        initial_sequence, {}, prompt.lower(), domain, context_analysis_results
    )
    
    assert "Security_Auditor" in adjusted_sequence
    assert adjusted_sequence.index("Security_Auditor") < adjusted_sequence.index("Impartial_Arbitrator")

def test_apply_dynamic_adjustment_low_code_quality(persona_router_instance):
    """Tests dynamic adjustment when low code quality is detected in context."""
    prompt = "Refactor the module."
    domain = "Software Engineering"
    context_analysis_results = {"security_concerns": [], "key_modules": [{"name": "module.py", "code_quality_score": 0.6, "complexity_score": 0.8}]}
    initial_sequence = persona_router_instance.persona_sets["Software Engineering"].copy()
    adjusted_sequence = persona_router_instance._apply_dynamic_adjustment(
        initial_sequence, {}, prompt.lower(), domain, context_analysis_results
    )
    
    assert "Code_Architect" in adjusted_sequence
    assert adjusted_sequence.index("Code_Architect") < adjusted_sequence.index("Impartial_Arbitrator")

def test_apply_dynamic_adjustment_test_engineer_inclusion(persona_router_instance):
    """Tests dynamic adjustment for Test_Engineer inclusion based on prompt keywords."""
    prompt = "Ensure high test coverage for the new feature."
    domain = "Software Engineering"
    initial_sequence = [p for p in persona_router_instance.persona_sets["Software Engineering"] if p != "Test_Engineer"]
    adjusted_sequence = persona_router_instance._apply_dynamic_adjustment(
        initial_sequence, {}, prompt.lower(), domain, {}
    )
    
    assert "Test_Engineer" in adjusted_sequence
    assert adjusted_sequence.index("Test_Engineer") < adjusted_sequence.index("Impartial_Arbitrator")

def test_apply_dynamic_adjustment_test_engineer_removal(persona_router_instance):
    """Tests dynamic adjustment for Test_Engineer removal when no relevant keywords."""
    prompt = "Design a new system architecture."
    domain = "Software Engineering"
    initial_sequence = persona_router_instance.persona_sets["Software Engineering"].copy()
    adjusted_sequence = persona_router_instance._apply_dynamic_adjustment(
        initial_sequence, {}, prompt.lower(), domain, {}
    )
    
    assert "Test_Engineer" not in adjusted_sequence

def test_dynamic_reordering_based_on_performance(persona_router_instance, mock_persona_manager_for_router):
    """Tests dynamic re-ordering of personas based on performance metrics."""
    prompt = "Analyze and improve the system."
    domain = "Software Engineering"

    mock_persona_manager_for_router.persona_performance_metrics["Skeptical_Generator"] = {
        "total_turns": 10, "schema_failures": 8, "truncation_failures": 0,
        "last_adjustment_timestamp": time.time() - 600
    }
    mock_persona_manager_for_router.persona_performance_metrics["Visionary_Generator"] = {
        "total_turns": 10, "schema_failures": 0, "truncation_failures": 0,
        "last_adjustment_timestamp": time.time() - 600
    }

    persona_router_instance.prompt_analyzer.is_self_analysis_prompt.return_value = False
    persona_router_instance.prompt_analyzer.recommend_domain_from_keywords.return_value = "Software Engineering"

    initial_sequence = persona_router_instance.persona_sets[domain].copy()
    adjusted_sequence = persona_router_instance.determine_persona_sequence(prompt, domain)

    assert adjusted_sequence.index("Visionary_Generator") < adjusted_sequence.index("Skeptical_Generator")
    assert adjusted_sequence[-1] == "Impartial_Arbitrator"
    assert adjusted_sequence[-2] == "Devils_Advocate"
