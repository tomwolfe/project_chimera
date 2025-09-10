# tests/test_persona_routing.py
import pytest
from unittest.mock import MagicMock
from src.persona.routing import PersonaRouter
from src.persona_manager import PersonaManager # Assuming PersonaManager is needed for PersonaRouter init
from src.utils.prompt_analyzer import PromptAnalyzer
from src.models import PersonaConfig

@pytest.fixture
def mock_persona_manager():
    pm = MagicMock(spec=PersonaManager)
    pm.all_personas = {
        "Analyst": PersonaConfig(name="Analyst", system_prompt="You are an analyst.", temperature=0.5, max_tokens=1024),
        "Critic": PersonaConfig(name="Critic", system_prompt="You are a critic.", temperature=0.5, max_tokens=1024),
        "SelfImprovementAnalyst": PersonaConfig(name="SelfImprovementAnalyst", system_prompt="You are a self-improvement analyst.", temperature=0.5, max_tokens=1024),
        "Security_Auditor": PersonaConfig(name="Security_Auditor", system_prompt="You are a security auditor.", temperature=0.5, max_tokens=1024),
        "Code_Architect": PersonaConfig(name="Code_Architect", system_prompt="You are a code architect.", temperature=0.5, max_tokens=1024),
        "Devils_Advocate": PersonaConfig(name="Devils_Advocate", system_prompt="You are a devils advocate.", temperature=0.5, max_tokens=1024),
    }
    pm.persona_sets = {
        "General": ["Analyst", "Critic", "SelfImprovementAnalyst"],
        "Software Engineering": ["Analyst", "Critic", "Security_Auditor", "Code_Architect", "Impartial_Arbitrator"],
        "Self-Improvement": ["SelfImprovementAnalyst", "Code_Architect", "Security_Auditor", "Devils_Advocate", "Impartial_Arbitrator"],
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
        prompt_analyzer=mock_persona_manager.prompt_analyzer
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
    assert "SelfImprovementAnalyst" in sequence

def test_determine_persona_sequence_self_analysis(persona_router_instance, mock_persona_manager):
    """Test self-analysis prompt routing."""
    mock_persona_manager.prompt_analyzer.is_self_analysis_prompt.return_value = True
    mock_persona_manager.persona_sets["Self-Improvement"] = ["SelfImprovementAnalyst", "Code_Architect", "Security_Auditor", "Devils_Advocate", "Impartial_Arbitrator"] # Simplified for test
    
    prompt = "Critically analyze the Project Chimera codebase."
    domain = "Self-Improvement"
    sequence = persona_router_instance.determine_persona_sequence(prompt, domain)
    assert "SelfImprovementAnalyst" in sequence
    assert "Devils_Advocate" in sequence # Should be added by dynamic adjustment

def test_determine_persona_sequence_with_context_analysis(persona_router_instance):
    """Test routing with context analysis results."""
    prompt = "Check security of the code."
    domain = "Software Engineering"
    context_analysis_results = {
        "security_concerns": ["SQL Injection detected"],
        "relevant_files": [("src/security.py", 0.8)], # Added relevant_files for _should_include_test_engineer
        "key_modules": []
    }
    sequence = persona_router_instance.determine_persona_sequence(prompt, domain, context_analysis_results=context_analysis_results)
    assert "Security_Auditor" in sequence
    assert "Code_Architect" not in sequence # Should not be added without architectural concerns