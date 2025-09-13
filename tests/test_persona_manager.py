# tests/test_persona_manager.py

import pytest
from unittest.mock import MagicMock, patch
from src.persona_manager import PersonaManager
from src.models import PersonaConfig
from src.token_tracker import TokenUsageTracker
from src.utils.prompt_analyzer import PromptAnalyzer
from src.config.persistence import ConfigPersistence
from src.config.settings import ChimeraSettings  # NEW: Import ChimeraSettings
import time


@pytest.fixture
def mock_token_tracker():
    """Provides a mock TokenUsageTracker instance."""
    return MagicMock(spec=TokenUsageTracker)


@pytest.fixture
def mock_prompt_analyzer():
    """Provides a mock PromptAnalyzer instance."""
    pa = MagicMock(spec=PromptAnalyzer)
    pa.analyze_complexity.return_value = {"complexity_score": 0.5}
    pa.is_self_analysis_prompt.return_value = False
    pa.recommend_domain_from_keywords.return_value = "General"
    return pa


@pytest.fixture
def mock_config_persistence():
    """Provides a mock ConfigPersistence instance."""
    cp = MagicMock(spec=ConfigPersistence)
    # Mock load_personas_config to return a consistent structure
    cp.load_personas_config.return_value = {
        "personas": [
            {
                "name": "Visionary_Generator",
                "system_prompt": "Visionary",
                "temperature": 0.7,
                "max_tokens": 1024,
                "description": "Generates innovative solutions.",
            },
            {
                "name": "Skeptical_Generator",
                "system_prompt": "Skeptical",
                "temperature": 0.3,
                "max_tokens": 1024,
                "description": "Identifies flaws.",
            },
            {
                "name": "Impartial_Arbitrator",
                "system_prompt": "Arbitrator",
                "temperature": 0.1,
                "max_tokens": 4096,
                "description": "Synthesizes outcomes.",
            },
            {
                "name": "Test_Engineer",
                "system_prompt": "Test Engineer",
                "temperature": 0.3,
                "max_tokens": 4096,
                "description": "Ensures code quality.",
            },
            {
                "name": "TestPersona",
                "system_prompt": "Test system prompt",
                "temperature": 0.5,
                "max_tokens": 1024,
                "description": "A persona for testing metrics.",
            },  # Added for new tests
        ],
        "persona_sets": {
            "General": [
                "Visionary_Generator",
                "Skeptical_Generator",
                "Impartial_Arbitrator",
            ],
            "Software Engineering": [
                "Visionary_Generator",
                "Skeptical_Generator",
                "Test_Engineer",
                "Impartial_Arbitrator",
            ],
        },
    }
    # Mock methods related to custom frameworks
    cp._get_saved_custom_framework_names.return_value = []
    cp._load_custom_framework_config_from_file.return_value = None
    cp.save_user_framework.return_value = (True, "Framework saved.")
    cp.export_framework_for_sharing.return_value = "exported_yaml_content"
    cp.import_framework_from_file.return_value = (
        True,
        "Framework imported.",
        {
            "framework_name": "ImportedFramework",
            "description": "A description for imported framework",  # FIX: Add description
            "personas": {},
            "persona_sets": {"ImportedFramework": []},
        },
    )
    return cp


@pytest.fixture
def mock_settings():
    """Provides a mock ChimeraSettings instance."""
    settings = MagicMock(spec=ChimeraSettings)
    settings.default_max_input_tokens_per_persona = 4000
    settings.max_tokens_per_persona = {
        "TestPersona": 1024,  # Default for TestPersona
        "Visionary_Generator": 1024,
        "Skeptical_Generator": 1024,
        "Impartial_Arbitrator": 4096,
        "Test_Engineer": 4096,
    }
    return settings


@pytest.fixture
def persona_manager_instance(
    mock_token_tracker, mock_prompt_analyzer, mock_config_persistence, mock_settings
):
    """Provides a PersonaManager instance with mocked dependencies."""
    # Patch ConfigPersistence to return our mock during PersonaManager initialization
    with patch(
        "src.persona_manager.ConfigPersistence", return_value=mock_config_persistence
    ):
        pm = PersonaManager(
            domain_keywords={"General": ["general"], "Software Engineering": ["code"]},
            token_tracker=mock_token_tracker,
            settings=mock_settings,
        )
        # Manually set the mock prompt_analyzer as it's used by PersonaRouter internally
        pm.prompt_analyzer = mock_prompt_analyzer
        # Ensure the router also uses the mocked analyzer if it's initialized separately
        if pm.persona_router:
            pm.persona_router.prompt_analyzer = mock_prompt_analyzer
        return pm


@pytest.fixture
def persona_manager_for_metrics(
    mock_token_tracker, mock_prompt_analyzer, mock_config_persistence, mock_settings
):
    """Provides a PersonaManager instance specifically for testing metrics, ensuring 'TestPersona' is available."""
    with patch(
        "src.persona_manager.ConfigPersistence", return_value=mock_config_persistence
    ):
        pm = PersonaManager(
            domain_keywords={"General": ["general"]},
            token_tracker=mock_token_tracker,
            settings=mock_settings,
        )
        # Ensure TestPersona is in all_personas for metrics tracking
        if "TestPersona" not in pm.all_personas:
            pm.all_personas["TestPersona"] = PersonaConfig(
                name="TestPersona",
                system_prompt="Test system prompt",
                temperature=0.5,
                max_tokens=1024,
                description="A persona for testing metrics.",
            )
        pm._initialize_performance_metrics()  # Re-initialize to ensure TestPersona is in metrics
        return pm


def test_persona_manager_initialization(persona_manager_instance):
    """Tests that PersonaManager initializes correctly and loads default personas."""
    assert "Visionary_Generator" in persona_manager_instance.all_personas
    assert "General" in persona_manager_instance.persona_sets
    assert persona_manager_instance.default_persona_set_name == "General"
    # The order of available_domains is sorted alphabetically
    assert persona_manager_instance.available_domains == [
        "General",
        "Software Engineering",
    ]
    assert persona_manager_instance.persona_router is not None
    assert persona_manager_instance.token_tracker is not None
    assert persona_manager_instance.prompt_analyzer is not None


def test_get_persona_sequence_for_framework(persona_manager_instance):
    """Tests retrieving a persona sequence for a known framework."""
    sequence = persona_manager_instance.get_persona_sequence_for_framework("General")
    assert sequence == [
        "Visionary_Generator",
        "Skeptical_Generator",
        "Impartial_Arbitrator",
    ]

    sequence_se = persona_manager_instance.get_persona_sequence_for_framework(
        "Software Engineering"
    )
    assert sequence_se == [
        "Visionary_Generator",
        "Skeptical_Generator",
        "Test_Engineer",
        "Impartial_Arbitrator",
    ]


def test_get_persona_sequence_for_unknown_framework_falls_back_to_general(
    persona_manager_instance,
):
    """Tests that an unknown framework falls back to the 'General' sequence."""
    sequence = persona_manager_instance.get_persona_sequence_for_framework(
        "UnknownFramework"
    )
    assert sequence == [
        "Visionary_Generator",
        "Skeptical_Generator",
        "Impartial_Arbitrator",
    ]


def test_update_persona_config(persona_manager_instance):
    """Tests updating a persona's configuration."""
    original_temp = persona_manager_instance.all_personas[
        "Visionary_Generator"
    ].temperature
    new_temp = 0.8
    success = persona_manager_instance.update_persona_config(
        "Visionary_Generator", "temperature", new_temp
    )
    assert success
    assert (
        persona_manager_instance.all_personas["Visionary_Generator"].temperature
        == new_temp
    )
    # Ensure original_personas is not directly modified
    assert (
        persona_manager_instance._original_personas["Visionary_Generator"].temperature
        == original_temp
    )


def test_reset_persona_to_default(persona_manager_instance):
    """Tests resetting a persona to its default configuration."""
    # First, modify a persona
    persona_manager_instance.update_persona_config(
        "Visionary_Generator", "temperature", 0.9
    )
    persona_manager_instance.update_persona_config(
        "Visionary_Generator", "system_prompt", "New prompt"
    )

    # Then, reset it
    success = persona_manager_instance.reset_persona_to_default("Visionary_Generator")
    assert success
    # Check against the stored original configuration
    assert (
        persona_manager_instance.all_personas["Visionary_Generator"].temperature
        == persona_manager_instance._original_personas[
            "Visionary_Generator"
        ].temperature
    )
    assert (
        persona_manager_instance.all_personas["Visionary_Generator"].system_prompt
        == persona_manager_instance._original_personas[
            "Visionary_Generator"
        ].system_prompt
    )


def test_reset_all_personas_for_current_framework(persona_manager_instance):
    """Tests resetting all personas in a framework to default."""
    # Modify a persona in the "General" framework
    persona_manager_instance.update_persona_config(
        "Visionary_Generator", "temperature", 0.9
    )

    # Reset all for "General"
    success = persona_manager_instance.reset_all_personas_for_current_framework(
        "General"
    )
    assert success
    # Verify the reset against the original configuration
    assert (
        persona_manager_instance.all_personas["Visionary_Generator"].temperature
        == persona_manager_instance._original_personas[
            "Visionary_Generator"
        ].temperature
    )


def test_save_framework(persona_manager_instance, mock_config_persistence):
    """Tests saving a custom framework."""
    framework_name = "MyCustomFramework"
    description = "A test framework"
    # Create a copy of the persona config to simulate current state
    current_active_personas = {
        p_name: PersonaConfig(**p_data.model_dump())
        for p_name, p_data in persona_manager_instance.all_personas.items()
        if p_name
        in persona_manager_instance.get_persona_sequence_for_framework("General")
    }

    success, message = persona_manager_instance.save_framework(
        framework_name, "General", current_active_personas, description
    )
    assert success
    assert "Framework saved." in message
    mock_config_persistence.save_user_framework.assert_called_once()
    # Check if the framework name was added to available domains
    assert framework_name in persona_manager_instance.available_domains


def test_load_framework_into_session_custom(
    persona_manager_instance, mock_config_persistence
):
    """Tests loading a custom framework into the session."""
    custom_framework_data = {
        "framework_name": "LoadedCustom",
        "description": "A description for loaded framework",  # FIX: Add description
        "personas": {
            "NewPersona": {
                "name": "NewPersona",
                "system_prompt": "New",
                "temperature": 0.5,
                "max_tokens": 512,
                "description": "A new persona.",
            }
        },
        "persona_sets": {"LoadedCustom": ["NewPersona"]},
    }
    # Configure the mock to return this data when loading 'LoadedCustom'
    mock_config_persistence._load_custom_framework_config_from_file.return_value = (
        custom_framework_data
    )

    success, message, loaded_personas, loaded_persona_sets, new_framework_name = (
        persona_manager_instance.load_framework_into_session("LoadedCustom")
    )

    assert success
    assert "Loaded custom framework" in message
    # Check if the new persona was added to all_personas
    assert "NewPersona" in persona_manager_instance.all_personas
    # Check if the new persona set was added
    assert "LoadedCustom" in persona_manager_instance.persona_sets
    assert new_framework_name == "LoadedCustom"


def test_export_framework_for_sharing(
    persona_manager_instance, mock_config_persistence
):
    """Tests exporting a framework."""
    # Configure the mock to return dummy content for export
    mock_config_persistence.export_framework_for_sharing.return_value = (
        "exported_yaml_content"
    )

    success, message, content = persona_manager_instance.export_framework_for_sharing(
        "General"
    )
    assert success
    assert "exported successfully" in message
    assert content == "exported_yaml_content"
    # Ensure the persistence method was called with the correct framework name
    mock_config_persistence.export_framework_for_sharing.assert_called_once_with(
        "General"
    )


def test_import_framework(persona_manager_instance, mock_config_persistence):
    """Tests importing a framework."""
    # The mock_config_persistence is already set to return success for import_framework_from_file
    mock_config_persistence.import_framework_from_file.return_value = (
        True,
        "Framework imported.",
        {
            "framework_name": "ImportedFramework",
            "description": "A description for imported framework",  # FIX: Add description
            "personas": {},
            "persona_sets": {"ImportedFramework": []},
        },
    )
    success, message = persona_manager_instance.import_framework(
        "file_content", "test.yaml"
    )
    assert success
    assert "Framework imported." in message
    # Verify that the persistence method was called correctly
    mock_config_persistence.import_framework_from_file.assert_called_once_with(
        "file_content", "test.yaml"
    )
    # Check if the imported framework's name is now available
    assert "ImportedFramework" in persona_manager_instance.available_domains


def test_get_adjusted_persona_config_truncated(persona_manager_instance):
    """Tests getting a truncated persona config."""
    truncated_config = persona_manager_instance.get_adjusted_persona_config(
        "Visionary_Generator_TRUNCATED"
    )

    # Assert that the base persona name is used correctly
    assert truncated_config.name == "Visionary_Generator"
    # Assert that max_tokens has been reduced
    assert (
        truncated_config.max_tokens
        < persona_manager_instance.all_personas["Visionary_Generator"].max_tokens
    )
    # Assert that the system prompt includes the truncation warning
    assert "CRITICAL: Be extremely concise" in truncated_config.system_prompt


def test_record_persona_performance(persona_manager_for_metrics):
    """Tests recording persona performance metrics."""
    pm = persona_manager_for_metrics
    # Record a successful turn
    pm.record_persona_performance(
        "TestPersona",
        1,
        {},
        True,
        "Valid output",
        schema_validation_failed=False,
        is_truncated=False,
    )
    metrics = pm.persona_performance_metrics["TestPersona"]
    assert metrics["total_turns"] == 1
    assert metrics["schema_failures"] == 0
    assert metrics["truncation_failures"] == 0

    # Record a turn with schema failure
    pm.record_persona_performance(
        "TestPersona",
        2,
        {},
        False,
        "Schema error",
        schema_validation_failed=True,
        is_truncated=False,
    )
    metrics = pm.persona_performance_metrics["TestPersona"]
    assert metrics["total_turns"] == 2
    assert metrics["schema_failures"] == 1

    # Record a turn with truncation
    pm.record_persona_performance(
        "TestPersona", 3, {}, True, "Truncated", is_truncated=True
    )
    metrics = pm.persona_performance_metrics["TestPersona"]
    assert metrics["total_turns"] == 3
    assert metrics["truncation_failures"] == 1


def test_get_token_optimized_persona_sequence_global_high_consumption(
    persona_manager_instance, mock_token_tracker
):
    """Tests token optimization when global consumption is high."""
    # Simulate high global token consumption rate
    mock_token_tracker.get_consumption_rate.return_value = 0.8

    sequence = ["Visionary_Generator", "Skeptical_Generator"]
    optimized_sequence = persona_manager_instance.get_token_optimized_persona_sequence(
        sequence
    )

    # Expect both personas to be truncated
    assert "Visionary_Generator_TRUNCATED" in optimized_sequence
    assert "Skeptical_Generator_TRUNCATED" in optimized_sequence


def test_get_token_optimized_persona_sequence_persona_prone_to_truncation(
    persona_manager_instance, mock_token_tracker
):
    """Tests token optimization when a specific persona is prone to truncation."""
    # Simulate low global token consumption
    mock_token_tracker.get_consumption_rate.return_value = 0.1

    # Simulate Visionary_Generator being prone to truncation by setting its metrics
    metrics = persona_manager_instance.persona_performance_metrics[
        "Visionary_Generator"
    ]
    metrics["total_turns"] = 10
    metrics["truncation_failures"] = 3  # 30% truncation rate, exceeding the threshold

    sequence = ["Visionary_Generator", "Skeptical_Generator"]
    optimized_sequence = persona_manager_instance.get_token_optimized_persona_sequence(
        sequence
    )

    # Expect only Visionary_Generator to be truncated
    assert "Visionary_Generator_TRUNCATED" in optimized_sequence
    assert "Skeptical_Generator" in optimized_sequence


def test_get_adjusted_persona_config_adaptive_temperature(persona_manager_for_metrics):
    """Test adaptive temperature adjustment based on schema failures."""
    pm = persona_manager_for_metrics
    original_temp = pm.all_personas["TestPersona"].temperature
    metrics = pm.persona_performance_metrics["TestPersona"]

    # Simulate high schema failure rate
    metrics["total_turns"] = 10
    metrics["schema_failures"] = 3  # 30% failure rate
    metrics["last_adjustment_timestamp"] = (
        time.time() - pm.adjustment_cooldown_seconds - 10
    )

    adjusted_config = pm.get_adjusted_persona_config("TestPersona")
    assert adjusted_config.temperature < original_temp
    assert metrics["last_adjusted_temp"] == adjusted_config.temperature
    assert metrics["total_turns"] == 0


def test_get_adjusted_persona_config_adaptive_max_tokens(persona_manager_for_metrics):
    """Test adaptive max_tokens adjustment based on truncation failures."""
    pm = persona_manager_for_metrics
    original_max_tokens = pm.all_personas["TestPersona"].max_tokens
    metrics = pm.persona_performance_metrics["TestPersona"]

    # Simulate high truncation failure rate
    metrics["total_turns"] = 10
    metrics["truncation_failures"] = 2  # 20% truncation rate
    metrics["last_adjustment_timestamp"] = (
        time.time() - pm.adjustment_cooldown_seconds - 10
    )

    adjusted_config = pm.get_adjusted_persona_config("TestPersona")
    assert adjusted_config.max_tokens > original_max_tokens
    assert metrics["last_adjusted_max_tokens"] == adjusted_config.max_tokens
    assert metrics["total_turns"] == 0
