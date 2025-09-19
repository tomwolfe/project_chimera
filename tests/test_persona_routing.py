# tests/test_persona_routing.py
import pytest
from unittest.mock import MagicMock
from src.persona.routing import PersonaRouter
from src.persona_manager import (
    PersonaManager,
)  # Assuming PersonaManager is needed for PersonaRouter init
from src.utils.prompting.prompt_analyzer import PromptAnalyzer
from src.models import PersonaConfig


@pytest.fixture
def mock_persona_manager():
    pm = MagicMock(spec=PersonaManager)
    pm.all_personas = {
        "Analyst": PersonaConfig(
            name="Analyst",
            system_prompt="You are an analyst.",
            temperature=0.5,
            max_tokens=1024,
        ),
        "Critic": PersonaConfig(
            name="Critic",
            system_prompt="You are a critic.",
            temperature=0.5,
            max_tokens=1024,
        ),
        "Self_Improvement_Analyst": PersonaConfig(  # Standardized to snake_case
            name="Self_Improvement_Analyst",  # Standardized to snake_case
            system_prompt="You are a self-improvement analyst.",
            temperature=0.5,
            max_tokens=1024,
        ),
        "Security_Auditor": PersonaConfig(
            name="Security_Auditor",
            system_prompt="You are a security auditor.",
            temperature=0.5,
            max_tokens=1024,
        ),
        "Code_Architect": PersonaConfig(
            name="Code_Architect",
            system_prompt="You are a code architect.",
            temperature=0.5,
            max_tokens=1024,
        ),
        "Devils_Advocate": PersonaConfig(
            name="Devils_Advocate",
            system_prompt="You are a devils advocate.",
            temperature=0.5,
            max_tokens=1024,
        ),
        "Test_Engineer": PersonaConfig(
            name="Test_Engineer",
            system_prompt="You are a test engineer.",
            temperature=0.5,
            max_tokens=1024,
        ),  # Added Test_Engineer
        "Impartial_Arbitrator": PersonaConfig(
            name="Impartial_Arbitrator",
            system_prompt="You are an impartial arbitrator.",
            temperature=0.5,
            max_tokens=1024,
        ),  # Added Impartial_Arbitrator
    }
    pm.persona_sets = {
        "General": [
            "Analyst",
            "Critic",
            "Self_Improvement_Analyst",
        ],  # Standardized to snake_case
        "Software Engineering": [
            "Analyst",
            "Critic",
            "Security_Auditor",
            "Code_Architect",
            "Impartial_Arbitrator",
        ],
        "Self-Improvement": [
            "Self_Improvement_Analyst",  # Standardized to snake_case
            "Code_Architect",
            "Security_Auditor",
            "Devils_Advocate",
            "Impartial_Arbitrator",
        ],
        "Testing": ["Test_Engineer", "Critic"],  # Added a testing domain
    }
    pm.prompt_analyzer = MagicMock(spec=PromptAnalyzer)
    pm.prompt_analyzer.is_self_analysis_prompt.return_value = False
    pm.prompt_analyzer.analyze_complexity.return_value = {"complexity_score": 0.5}
    return pm


@pytest.fixture
def persona_router_instance(mock_persona_manager):
    return PersonaRouter(
        all_personas=mock_persona_manager.all_personas,
        persona_sets=mock_persona_manager.persona_sets,
        prompt_analyzer=mock_persona_manager.prompt_analyzer,
    )


def test_determine_persona_sequence_basic(persona_router_instance):
    """Test a basic persona sequence determination."""
    prompt = "Analyze a general problem."
    domain = "General"
    sequence = persona_router_instance.determine_persona_sequence(prompt, domain)
    assert isinstance(sequence, list)
    assert len(sequence) > 0
    assert "Analyst" in sequence
    assert "Critic" in sequence
    assert "Self_Improvement_Analyst" in sequence  # Standardized to snake_case


def test_determine_persona_sequence_self_analysis(
    persona_router_instance, mock_persona_manager
):
    """Test self-analysis prompt routing."""
    mock_persona_manager.prompt_analyzer.is_self_analysis_prompt.return_value = True
    mock_persona_manager.persona_sets["Self-Improvement"] = [
        "Self_Improvement_Analyst",  # Standardized to snake_case
        "Code_Architect",
        "Security_Auditor",
        "Devils_Advocate",
        "Impartial_Arbitrator",
    ]  # Simplified for test

    prompt = "Critically analyze the Project Chimera codebase."
    domain = "Self-Improvement"
    sequence = persona_router_instance.determine_persona_sequence(prompt, domain)
    assert (
        "Self_Improvement_Analyst" in sequence
    )  # Should be present # Standardized to snake_case
    assert "Devils_Advocate" in sequence  # Should be present
    assert "Code_Architect" in sequence  # Should be present
    assert "Security_Auditor" in sequence  # Should be present
    assert "Impartial_Arbitrator" in sequence  # Should be present
    # Ensure the order is generally correct for self-improvement
    assert sequence.index(
        "Self_Improvement_Analyst"
    ) < sequence.index(  # Standardized to snake_case
        "Impartial_Arbitrator"
    )
    assert sequence.index("Devils_Advocate") < sequence.index("Impartial_Arbitrator")


def test_determine_persona_sequence_with_context_analysis(persona_router_instance):
    """Test routing with context analysis results."""
    prompt = "Check security of the code."
    domain = "Software Engineering"
    context_analysis_results = {
        "security_concerns": ["SQL Injection detected"],
        "relevant_files": [
            ("src/security.py", 0.8)
        ],  # Added relevant_files for _should_include_test_engineer
        "key_modules": [],
    }
    sequence = persona_router_instance.determine_persona_sequence(
        prompt, domain, context_analysis_results=context_analysis_results
    )
    assert "Security_Auditor" in sequence
    # Code_Architect is a default for Software Engineering, but the new logic might remove it if not architectural.
    # For this specific test, the prompt "Check security of the code." is not strongly architectural,
    # so Code_Architect might be removed by _apply_dynamic_adjustment.
    assert (
        "Code_Architect" not in sequence
    )  # Expect it to be removed if not explicitly architectural


def test_should_include_test_engineer_self_improvement_domain(persona_router_instance):
    """Test that Test_Engineer is always included for 'self-improvement' domain."""
    prompt = "Analyze the codebase for general improvements."
    context_analysis_results = {}
    domain = "self-improvement"
    assert (
        persona_router_instance._should_include_test_engineer(
            prompt.lower(), context_analysis_results, domain
        )
        is True
    )


def test_should_include_test_engineer_with_testing_keywords(persona_router_instance):
    """Test that Test_Engineer is included when testing keywords are present."""
    prompt = "Improve test coverage for the new feature."
    context_analysis_results = {}
    domain = "Software Engineering"
    assert (
        persona_router_instance._should_include_test_engineer(
            prompt.lower(), context_analysis_results, domain
        )
        is True
    )


def test_should_include_test_engineer_with_test_files_in_context(
    persona_router_instance,
):
    """Test that Test_Engineer is included when test files are in context."""
    prompt = "Refactor the core logic."
    context_analysis_results = {"relevant_files": [("tests/test_core.py", 0.9)]}
    domain = "Software Engineering"
    assert (
        persona_router_instance._should_include_test_engineer(
            prompt.lower(), context_analysis_results, domain
        )
        is True
    )


def test_should_include_test_engineer_no_relevant_triggers(persona_router_instance):
    """Test that Test_Engineer is NOT included when no relevant triggers are present."""
    prompt = "Design a new database schema."
    context_analysis_results = {"relevant_files": [("src/database/schema.py", 0.9)]}
    domain = "Software Engineering"
    assert (
        persona_router_instance._should_include_test_engineer(
            prompt.lower(), context_analysis_results, domain
        )
        is False
    )
