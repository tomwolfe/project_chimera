import pytest
from unittest.mock import patch, MagicMock
from src.core import ProjectChimera

# Mocking necessary components
class MockLLMProvider:
    def get_llm_response(self, prompt):
        return f"Mocked response for: {prompt}"

class MockOutputParser:
    def parse(self, text):
        return {"parsed_output": text}

class MockCodeValidator:
    def validate(self, code):
        return True, ""

class MockPersonaManager:
    def get_persona_by_name(self, name):
        return MagicMock()
    
    def get_persona_sequence_for_framework(self, framework_name):
        return ["MockPersona1", "MockPersona2"] # Example sequence

    @property
    def all_personas(self):
        return {
            "MockPersona1": MagicMock(name="MockPersona1", system_prompt="Mock prompt 1", temperature=0.7, max_tokens=100),
            "MockPersona2": MagicMock(name="MockPersona2", system_prompt="Mock prompt 2", temperature=0.3, max_tokens=100)
        }
    
    def get_adjusted_persona_config(self, persona_name):
        return self.all_personas.get(persona_name)

@pytest.fixture
def chimera_instance():
    # Mocking dependencies during initialization
    with patch('src.core.GeminiProvider', return_value=MockLLMProvider()), \
         patch('src.core.PersonaManager', return_value=MockPersonaManager()), \
         patch('src.core.ContextRelevanceAnalyzer', return_value=MagicMock()), \
         patch('src.utils.output_parser.LLMOutputParser', return_value=MockOutputParser()):
        
        # Instantiate ProjectChimera with mocked dependencies
        yield ProjectChimera(
            initial_prompt="Analyze this code.",
            api_key="mock_api_key",
            model_name="gemini-2.5-flash-lite",
            domain="Software Engineering",
            codebase_context={"file.py": "print('hello')"}
        )

def test_chimera_init(chimera_instance):
    # Check if dependencies are correctly injected
    assert isinstance(chimera_instance.llm_provider, MockLLMProvider)
    assert isinstance(chimera_instance.persona_manager, MockPersonaManager)
    assert isinstance(chimera_instance.context_analyzer, MagicMock)
    assert isinstance(chimera_instance.output_parser, MockOutputParser)
    assert chimera_instance.initial_prompt == "Analyze this code."
    assert chimera_instance.domain == "Software Engineering"
    assert chimera_instance.codebase_context == {"file.py": "print('hello')"}

def test_chimera_run_debate_simple_flow(chimera_instance):
    # Mock the LLM call and persona manager sequence
    mock_llm_provider = MagicMock()
    mock_llm_provider.generate.return_value = ("Mocked LLM response.", 100, 50) # Simulate return tuple
    
    mock_persona_manager = chimera_instance.persona_manager
    mock_persona_manager.get_persona_sequence_for_framework.return_value = ["MockPersona1", "MockPersona2"]
    
    mock_output_parser = MagicMock()
    mock_output_parser.parse_and_validate.return_value = {"general_output": "Parsed output"}

    # Patch the LLMProvider and OutputParser within the ProjectChimera instance
    chimera_instance.llm_provider = mock_llm_provider
    chimera_instance.output_parser = mock_output_parser

    final_answer, intermediate_steps = chimera_instance.run_debate()

    assert "Parsed output" in final_answer.get("general_output", "")
    assert "MockPersona1_Output" in intermediate_steps
    assert "MockPersona2_Output" in intermediate_steps
    mock_llm_provider.generate.assert_called()
    mock_output_parser.parse_and_validate.assert_called()

# Add more tests for different scenarios (e.g., context analysis, error handling)