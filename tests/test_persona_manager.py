# tests/test_persona_manager.py

import pytest
from unittest.mock import MagicMock, patch
from src.persona_manager import PersonaManager
from src.models import PersonaConfig
from src.token_tracker import TokenUsageTracker
from src.utils.prompt_analyzer import PromptAnalyzer
from src.config.persistence import ConfigPersistence # Import ConfigPersistence

# Mock dependencies for PersonaManager
@pytest.fixture
def mock_token_tracker():
    return MagicMock(spec=TokenUsageTracker)

@pytest.fixture
def mock_prompt_analyzer():
    pa = MagicMock(spec=PromptAnalyzer)
    pa.analyze_complexity.return_value = {"complexity_score": 0.5}
    pa.is_self_analysis_prompt.return_value = False
    pa.recommend_domain_from_keywords.return_value = "General"
    return pa

@pytest.fixture
def mock_config_persistence():
    cp = MagicMock(spec=ConfigPersistence)
    # Mock load_personas_config to return a consistent structure
    cp.load_personas_config.return_value = {
        "personas": [
            {"name": "Visionary_Generator", "system_prompt": "Visionary", "temperature": 0.7, "max_tokens": 1024, "description": "Generates innovative solutions."},
            {"name": "Skeptical_Generator", "system_prompt": "Skeptical", "temperature": 0.3, "max_tokens": 1024, "description": "Identifies flaws."},
            {"name": "Impartial_Arbitrator", "system_prompt": "Arbitrator", "temperature": 0.1, "max_tokens": 4096, "description": "Synthesizes outcomes."},
            {"name": "Test_Engineer", "system_prompt": "Test Engineer", "temperature": 0.3, "max_tokens": 4096, "description": "Ensures code quality."},
        ],
        "persona_sets": {
            "General": ["Visionary_Generator", "Skeptical_Generator", "Impartial_Arbitrator"],
            "Software Engineering": ["Visionary_Generator", "Skeptical_Generator", "Test_Engineer", "Impartial_Arbitrator"],
        }
    }
    # Mock _get_saved_custom_framework_names to return an empty list for most tests
    cp._get_saved_custom_framework_names.return_value = []
    # Mock _load_custom_framework_config_from_file to return None
    cp._load_custom_framework_config_from_file.return_value = None
    # Mock save_user_framework to always succeed
    cp.save_user_framework.return_value = (True, "Framework saved.")
    # Mock export_framework_for_sharing to return dummy content
    cp.export_framework_for_sharing.return_value = "framework_content"
    # Mock import_framework_from_file to always succeed
    cp.import_framework_from_file.return_value = (True, "Framework imported.", {"framework_name": "ImportedFramework", "personas": {}, "persona_sets": {"ImportedFramework": []}})
    return cp

@pytest.fixture
def persona_manager_instance(mock_token_tracker, mock_prompt_analyzer, mock_config_persistence):
    """Provides a PersonaManager instance with mocked dependencies."""
    # Patch ConfigPersistence to return our mock
    with patch('src.persona_manager.ConfigPersistence', return_value=mock_config_persistence):
        # PersonaManager's __init__ calls _load_initial_data which uses ConfigPersistence
        pm = PersonaManager(
            domain_keywords={"General": ["general"], "Software Engineering": ["code"]},
            token_tracker=mock_token_tracker
        )
        # Manually set the mock prompt_analyzer as it's passed to PersonaRouter
        pm.prompt_analyzer = mock_prompt_analyzer
        pm.persona_router.prompt_analyzer = mock_prompt_analyzer
        return pm

def test_persona_manager_initialization(persona_manager_instance):
    """Tests that PersonaManager initializes correctly and loads default personas."""
    assert "Visionary_Generator" in persona_manager_instance.all_personas
    assert "General" in persona_manager_instance.persona_sets
    assert persona_manager_instance.default_persona_set_name == "General"
    assert persona_manager_instance.available_domains == ["General", "Software Engineering"]
    assert persona_manager_instance.persona_router is not None
    assert persona_manager_instance.token_tracker is not None
    assert persona_manager_instance.prompt_analyzer is not None

def test_get_persona_sequence_for_framework(persona_manager_instance):
    """Tests retrieving a persona sequence for a known framework."""
    sequence = persona_manager_instance.get_persona_sequence_for_framework("General")
    assert sequence == ["Visionary_Generator", "Skeptical_Generator", "Impartial_Arbitrator"]

    sequence_se = persona_manager_instance.get_persona_sequence_for_framework("Software Engineering")
    assert sequence_se == ["Visionary_Generator", "Skeptical_Generator", "Test_Engineer", "Impartial_Arbitrator"]

def test_get_persona_sequence_for_unknown_framework_falls_back_to_general(persona_manager_instance):
    """Tests that an unknown framework falls back to the 'General' sequence."""
    sequence = persona_manager_instance.get_persona_sequence_for_framework("UnknownFramework")
    assert sequence == ["Visionary_Generator", "Skeptical_Generator", "Impartial_Arbitrator"]

def test_update_persona_config(persona_manager_instance):
    """Tests updating a persona's configuration."""
    original_temp = persona_manager_instance.all_personas["Visionary_Generator"].temperature
    new_temp = 0.8
    success = persona_manager_instance.update_persona_config("Visionary_Generator", "temperature", new_temp)
    assert success
    assert persona_manager_instance.all_personas["Visionary_Generator"].temperature == new_temp
    # Ensure original_personas is not directly modified
    assert persona_manager_instance._original_personas["Visionary_Generator"].temperature == original_temp

def test_reset_persona_to_default(persona_manager_instance):
    """Tests resetting a persona to its default configuration."""
    # First, modify a persona
    persona_manager_instance.update_persona_config("Visionary_Generator", "temperature", 0.9)
    persona_manager_instance.update_persona_config("Visionary_Generator", "system_prompt", "New prompt")
    
    # Then, reset it
    success = persona_manager_instance.reset_persona_to_default("Visionary_Generator")
    assert success
    assert persona_manager_instance.all_personas["Visionary_Generator"].temperature == persona_manager_instance._original_personas["Visionary_Generator"].temperature
    assert persona_manager_instance.all_personas["Visionary_Generator"].system_prompt == persona_manager_instance._original_personas["Visionary_Generator"].system_prompt

def test_reset_all_personas_for_current_framework(persona_manager_instance):
    """Tests resetting all personas in a framework to default."""
    # Modify a persona in the "General" framework
    persona_manager_instance.update_persona_config("Visionary_Generator", "temperature", 0.9)
    
    # Reset all for "General"
    success = persona_manager_instance.reset_all_personas_for_current_framework("General")
    assert success
    assert persona_manager_instance.all_personas["Visionary_Generator"].temperature == persona_manager_instance._original_personas["Visionary_Generator"].temperature

def test_save_framework(persona_manager_instance, mock_config_persistence):
    """Tests saving a custom framework."""
    framework_name = "MyCustomFramework"
    description = "A test framework"
    current_active_personas = {
        "Visionary_Generator": persona_manager_instance.all_personas["Visionary_Generator"]
    }
    success, message = persona_manager_instance.save_framework(
        framework_name, "General", current_active_personas, description
    )
    assert success
    assert "saved successfully" in message
    mock_config_persistence.save_user_framework.assert_called_once()
    assert framework_name in persona_manager_instance.available_domains

def test_load_framework_into_session_custom(persona_manager_instance, mock_config_persistence):
    """Tests loading a custom framework into the session."""
    custom_framework_data = {
        "framework_name": "LoadedCustom",
        "description": "Loaded framework",
        "personas": {
            "NewPersona": {"name": "NewPersona", "system_prompt": "New", "temperature": 0.5, "max_tokens": 512}
        },
        "persona_sets": {"LoadedCustom": ["NewPersona"]}
    }
    mock_config_persistence._load_custom_framework_config_from_file.return_value = custom_framework_data
    
    success, message, loaded_personas, loaded_persona_sets, new_framework_name = \
        persona_manager_instance.load_framework_into_session("LoadedCustom")
    
    assert success
    assert "Loaded custom framework" in message
    assert "NewPersona" in persona_manager_instance.all_personas
    assert "LoadedCustom" in persona_manager_instance.persona_sets
    assert new_framework_name == "LoadedCustom"

def test_export_framework_for_sharing(persona_manager_instance, mock_config_persistence):
    """Tests exporting a framework."""
    mock_config_persistence.export_framework_for_sharing.return_value = "exported_yaml_content"
    success, message, content = persona_manager_instance.export_framework_for_sharing("General")
    assert success
    assert "exported successfully" in message
    assert content == "exported_yaml_content"

def test_import_framework(persona_manager_instance, mock_config_persistence):
    """Tests importing a framework."""
    # The mock_config_persistence is already set to return success for import_framework_from_file
    success, message = persona_manager_instance.import_framework("file_content", "test.yaml")
    assert success
    assert "imported and saved successfully" in message
    mock_config_persistence.import_framework_from_file.assert_called_once_with("file_content", "test.yaml")
    assert "ImportedFramework" in persona_manager_instance.available_domains

def test_get_adjusted_persona_config_truncated(persona_manager_instance):
    """Tests getting a truncated persona config."""
    truncated_config = persona_manager_instance.get_adjusted_persona_config("Visionary_Generator_TRUNCATED")
    assert truncated_config.name == "Visionary_Generator" # Base name
    assert truncated_config.max_tokens < persona_manager_instance.all_personas["Visionary_Generator"].max_tokens
    assert "CRITICAL: Be extremely concise" in truncated_config.system_prompt

def test_record_persona_performance(persona_manager_instance):
    """Tests recording persona performance metrics."""
    persona_manager_instance.record_persona_performance("Visionary_Generator", 1, {}, True, "Valid output")
    metrics = persona_manager_instance.persona_performance_metrics["Visionary_Generator"]
    assert metrics["total_turns"] == 1
    assert metrics["schema_failures"] == 0
    assert metrics["truncation_failures"] == 0

    persona_manager_instance.record_persona_performance("Visionary_Generator", 2, {}, False, "Schema error")
    metrics = persona_manager_instance.persona_performance_metrics["Visionary_Generator"]
    assert metrics["total_turns"] == 2
    assert metrics["schema_failures"] == 1

    persona_manager_instance.record_persona_performance("Visionary_Generator", 3, {}, True, "Truncated", is_truncated=True)
    metrics = persona_manager_instance.persona_performance_metrics["Visionary_Generator"]
    assert metrics["total_turns"] == 3
    assert metrics["truncation_failures"] == 1

def test_get_token_optimized_persona_sequence_global_high_consumption(persona_manager_instance, mock_token_tracker):
    """Tests token optimization when global consumption is high."""
    mock_token_tracker.get_consumption_rate.return_value = 0.8 # Simulate high global consumption
    sequence = ["Visionary_Generator", "Skeptical_Generator"]
    optimized_sequence = persona_manager_instance.get_token_optimized_persona_sequence(sequence)
    assert "Visionary_Generator_TRUNCATED" in optimized_sequence
    assert "Skeptical_Generator_TRUNCATED" in optimized_sequence

def test_get_token_optimized_persona_sequence_persona_prone_to_truncation(persona_manager_instance, mock_token_tracker):
    """Tests token optimization when a specific persona is prone to truncation."""
    mock_token_tracker.get_consumption_rate.return_value = 0.1 # Low global consumption
    
    # Simulate Visionary_Generator being prone to truncation
    metrics = persona_manager_instance.persona_performance_metrics["Visionary_Generator"]
    metrics["total_turns"] = 10
    metrics["truncation_failures"] = 3 # 30% truncation rate
    
    sequence = ["Visionary_Generator", "Skeptical_Generator"]
    optimized_sequence = persona_manager_instance.get_token_optimized_persona_sequence(sequence)
    assert "Visionary_Generator_TRUNCATED" in optimized_sequence
    assert "Skeptical_Generator" in optimized_sequence # Skeptical_Generator should not be truncated