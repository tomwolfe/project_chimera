import pytest

# Assuming persona_manager.py contains functions/classes for managing personas
# Replace with actual imports from your project
# from src.persona_manager import PersonaManager, load_persona

# Mock PersonaManager for testing
class MockPersonaManager:
    def __init__(self):
        self.personas = {
            "default": {"name": "Default Persona", "description": "A standard persona."},
            "expert": {"name": "Expert Persona", "description": "An expert in a specific field."}
        }

    def get_persona(self, persona_name):
        return self.personas.get(persona_name)

    def list_personas(self):
        return list(self.personas.keys())

@pytest.fixture
def persona_manager():
    return MockPersonaManager()

def test_persona_manager_initialization():
    manager = MockPersonaManager()
    assert manager is not None
    assert manager.list_personas() == ["default", "expert"]

def test_get_persona_existing(persona_manager):
    persona = persona_manager.get_persona("default")
    assert persona is not None
    assert persona["name"] == "Default Persona"

def test_get_persona_non_existing(persona_manager):
    persona = persona_manager.get_persona("unknown")
    assert persona is None

def test_list_personas(persona_manager):
    personas = persona_manager.list_personas()
    assert "default" in personas
    assert "expert" in personas
    assert len(personas) == 2

# Add more tests for persona creation, modification, deletion, etc.
