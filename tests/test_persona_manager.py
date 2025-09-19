import unittest
from unittest.mock import patch, MagicMock
import yaml
from src.persona_manager import PersonaManager
from src.models import PersonaConfig
from src.token_tracker import TokenUsageTracker
from src.utils.prompting.prompt_analyzer import PromptAnalyzer
from src.config.persistence import ConfigPersistence
from src.config.settings import ChimeraSettings
import time


@patch("src.persona_manager.ConfigPersistence")
@patch("src.persona_manager.PromptAnalyzer")
@patch("src.persona_manager.TokenUsageTracker")
@patch("src.persona_manager.ChimeraSettings")
class TestPersonaManager(unittest.TestCase):
    def setUp(self):
        # Mock dependencies
        self.mock_token_tracker = MagicMock(spec=TokenUsageTracker)
        self.mock_prompt_analyzer = MagicMock(spec=PromptAnalyzer)
        self.mock_config_persistence = MagicMock(spec=ConfigPersistence)
        self.mock_settings = MagicMock(spec=ChimeraSettings)

        # Configure mock_config_persistence to return a consistent structure
        self.mock_config_persistence.load_personas_config.return_value = {
            "personas": [
                {
                    "name": "Analyst",
                    "system_prompt": "You are an Analyst.",
                    "temperature": 0.5,
                    "max_tokens": 1024,
                    "description": "Analyzes.",
                },
                {
                    "name": "Critic",
                    "system_prompt": "You are a Critic.",
                    "temperature": 0.5,
                    "max_tokens": 1024,
                    "description": "Critiques.",
                },
                {
                    "name": "TestPersona",
                    "system_prompt": "Test system prompt",
                    "temperature": 0.5,
                    "max_tokens": 1024,
                    "description": "A persona for testing metrics.",
                },
            ],
            "persona_sets": {
                "General": ["Analyst", "Critic"],
                "TestFramework": ["TestPersona"],
            },
        }
        self.mock_config_persistence._get_saved_custom_framework_names.return_value = []
        self.mock_config_persistence._load_custom_framework_config_from_file.return_value = None
        self.mock_config_persistence.save_user_framework.return_value = (
            True,
            "Framework saved.",
        )
        self.mock_config_persistence.export_framework_for_sharing.return_value = (
            "exported_yaml_content"
        )
        self.mock_config_persistence.import_framework_from_file.return_value = (
            True,
            "Framework imported.",
            {
                "framework_name": "ImportedFramework",
                "description": "A description for imported framework",
                "personas": {},
                "persona_sets": {"ImportedFramework": []},
            },
        )

        # Configure mock_settings
        self.mock_settings.default_max_input_tokens_per_persona = 4000
        self.mock_settings.max_tokens_per_persona = {
            "TestPersona": 1024,
            "Analyst": 1024,
            "Critic": 1024,
        }

        # Instantiate PersonaManager with mocks
        self.persona_manager = PersonaManager(
            domain_keywords={"General": ["general"]},
            token_tracker=self.mock_token_tracker,
            settings=self.mock_settings,
        )
        # Manually set the mock prompt_analyzer as it's used by PersonaRouter internally
        self.persona_manager.prompt_analyzer = self.mock_prompt_analyzer
        if self.persona_manager.persona_router:
            self.persona_manager.persona_router.prompt_analyzer = (
                self.mock_prompt_analyzer
            )

        # Ensure TestPersona is in all_personas for metrics tracking
        if "TestPersona" not in self.persona_manager.all_personas:
            self.persona_manager.all_personas["TestPersona"] = PersonaConfig(
                name="TestPersona",
                system_prompt="Test system prompt",
                temperature=0.5,
                max_tokens=1024,
                description="A persona for testing metrics.",
            )
        self.persona_manager._initialize_performance_metrics()  # Re-initialize to ensure TestPersona is in metrics

    def test_get_persona_success(self):
        """Test retrieving an existing persona."""
        persona = self.persona_manager.all_personas.get("Analyst")
        self.assertIsNotNone(persona)
        self.assertEqual(persona.name, "Analyst")
        self.assertEqual(persona.system_prompt, "You are an Analyst.")

    def test_get_persona_not_found(self):
        """Test retrieving a non-existent persona."""
        persona = self.persona_manager.all_personas.get("NonExistent")
        self.assertIsNone(persona)

    def test_get_all_personas(self):
        """Test retrieving all personas."""
        personas = self.persona_manager.all_personas
        self.assertEqual(len(personas), 3)  # Analyst, Critic, TestPersona
        self.assertIn("Analyst", personas)
        self.assertIn("Critic", personas)
        self.assertIn("TestPersona", personas)

    def test_get_persona_sequence_for_framework(self):
        """Test getting persona sequence for a specific framework."""
        sequence = self.persona_manager.get_persona_sequence_for_framework("General")
        self.assertEqual(sequence, ["Analyst", "Critic"])

    def test_update_persona_config(self):
        """Test updating a persona's configuration."""
        original_temp = self.persona_manager.all_personas["Analyst"].temperature
        new_temp = 0.8
        success = self.persona_manager.update_persona_config(
            "Analyst", "temperature", new_temp
        )
        self.assertTrue(success)
        self.assertEqual(
            self.persona_manager.all_personas["Analyst"].temperature, new_temp
        )
        # Ensure original_personas is not directly modified
        self.assertEqual(
            self.persona_manager._original_personas["Analyst"].temperature,
            original_temp,
        )

    def test_reset_persona_to_default(self):
        """Test resetting a persona to its default configuration."""
        # First, modify a persona
        self.persona_manager.update_persona_config("Analyst", "temperature", 0.9)
        self.persona_manager.update_persona_config(
            "Analyst", "system_prompt", "New prompt"
        )

        # Then, reset it
        success = self.persona_manager.reset_persona_to_default("Analyst")
        self.assertTrue(success)
        # Check against the stored original configuration
        self.assertEqual(
            self.persona_manager.all_personas["Analyst"].temperature,
            self.persona_manager._original_personas["Analyst"].temperature,
        )
        self.assertEqual(
            self.persona_manager.all_personas["Analyst"].system_prompt,
            self.persona_manager._original_personas["Analyst"].system_prompt,
        )

    def test_reset_all_personas_for_current_framework(self):
        """Test resetting all personas in a framework to default."""
        # Modify a persona in the "General" framework
        self.persona_manager.update_persona_config("Analyst", "temperature", 0.9)

        # Reset all for "General"
        success = self.persona_manager.reset_all_personas_for_current_framework(
            "General"
        )
        self.assertTrue(success)
        # Verify the reset against the original configuration
        self.assertEqual(
            self.persona_manager.all_personas["Analyst"].temperature,
            self.persona_manager._original_personas["Analyst"].temperature,
        )

    def test_save_framework(self):
        """Test saving a custom framework."""
        framework_name = "MyCustomFramework"
        description = "A test framework"
        current_active_personas = {
            p_name: PersonaConfig(**p_data.model_dump())
            for p_name, p_data in self.persona_manager.all_personas.items()
            if p_name
            in self.persona_manager.get_persona_sequence_for_framework("General")
        }

        success, message = self.persona_manager.save_framework(
            framework_name, "General", current_active_personas, description
        )
        self.assertTrue(success)
        self.assertIn("Framework saved.", message)
        self.mock_config_persistence.save_user_framework.assert_called_once()
        self.assertIn(framework_name, self.persona_manager.available_domains)

    def test_load_framework_into_session_custom(self):
        """Test loading a custom framework into the session."""
        custom_framework_data = {
            "framework_name": "LoadedCustom",
            "description": "A description for loaded framework",
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
        self.mock_config_persistence._load_custom_framework_config_from_file.return_value = custom_framework_data

        success, message, loaded_personas, loaded_persona_sets, new_framework_name = (
            self.persona_manager.load_framework_into_session("LoadedCustom")
        )

        self.assertTrue(success)
        self.assertIn("Loaded custom framework", message)
        self.assertIn("NewPersona", self.persona_manager.all_personas)
        self.assertIn("LoadedCustom", self.persona_manager.persona_sets)
        self.assertEqual(new_framework_name, "LoadedCustom")

    def test_export_framework_for_sharing(self):
        """Test exporting a framework."""
        self.mock_config_persistence.export_framework_for_sharing.return_value = (
            "exported_yaml_content"
        )

        success, message, content = self.persona_manager.export_framework_for_sharing(
            "General"
        )
        self.assertTrue(success)
        self.assertIn("exported successfully", message)
        self.assertEqual(content, "exported_yaml_content")
        self.mock_config_persistence.export_framework_for_sharing.assert_called_once_with(
            "General"
        )

    def test_import_framework(self):
        """Test importing a framework."""
        self.mock_config_persistence.import_framework_from_file.return_value = (
            True,
            "Framework imported.",
            {
                "framework_name": "ImportedFramework",
                "description": "A description for imported framework",
                "personas": {},
                "persona_sets": {"ImportedFramework": []},
            },
        )
        success, message = self.persona_manager.import_framework(
            "file_content", "test.yaml"
        )
        self.assertTrue(success)
        self.assertIn("Framework imported.", message)
        self.mock_config_persistence.import_framework_from_file.assert_called_once_with(
            "file_content", "test.yaml"
        )
        self.assertIn("ImportedFramework", self.persona_manager.available_domains)

    def test_get_adjusted_persona_config_truncated(self):
        """Test getting a truncated persona config."""
        truncated_config = self.persona_manager.get_adjusted_persona_config(
            "Analyst_TRUNCATED"
        )

        self.assertEqual(truncated_config.name, "Analyst")
        self.assertLess(
            truncated_config.max_tokens,
            self.persona_manager.all_personas["Analyst"].max_tokens,
        )
        self.assertIn("CRITICAL: Be extremely concise", truncated_config.system_prompt)

    def test_record_persona_performance(self):
        """Test recording persona performance metrics."""
        pm = self.persona_manager
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
        self.assertEqual(metrics["total_turns"], 1)
        self.assertEqual(metrics["schema_failures"], 0)
        self.assertEqual(metrics["truncation_failures"], 0)

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
        self.assertEqual(metrics["total_turns"], 2)
        self.assertEqual(metrics["schema_failures"], 1)

        # Record a turn with truncation
        pm.record_persona_performance(
            "TestPersona", 3, {}, True, "Truncated", is_truncated=True
        )
        metrics = pm.persona_performance_metrics["TestPersona"]
        self.assertEqual(metrics["total_turns"], 3)
        self.assertEqual(metrics["truncation_failures"], 1)

    def test_get_token_optimized_persona_sequence_global_high_consumption(self):
        """Test token optimization when global consumption is high."""
        self.mock_token_tracker.get_consumption_rate.return_value = (
            0.8  # Simulate high global token consumption rate
        )

        sequence = ["Analyst", "Critic"]
        optimized_sequence = self.persona_manager.get_token_optimized_persona_sequence(
            sequence
        )

        self.assertIn("Analyst_TRUNCATED", optimized_sequence)
        self.assertIn("Critic_TRUNCATED", optimized_sequence)

    def test_get_token_optimized_persona_sequence_persona_prone_to_truncation(self):
        """Test token optimization when a specific persona is prone to truncation."""
        self.mock_token_tracker.get_consumption_rate.return_value = (
            0.1  # Simulate low global token consumption
        )

        # Simulate Analyst being prone to truncation by setting its metrics
        metrics = self.persona_manager.persona_performance_metrics["Analyst"]
        metrics["total_turns"] = 10
        metrics["truncation_failures"] = (
            3  # 30% truncation rate, exceeding the threshold
        )
        metrics["last_adjustment_timestamp"] = (
            time.time() - self.persona_manager.adjustment_cooldown_seconds - 10
        )  # Ensure cooldown passed

        sequence = ["Analyst", "Critic"]
        optimized_sequence = self.persona_manager.get_token_optimized_persona_sequence(
            sequence
        )

        self.assertIn("Analyst_TRUNCATED", optimized_sequence)
        self.assertIn("Critic", optimized_sequence)  # Critic should not be truncated

    def test_get_adjusted_persona_config_adaptive_temperature(self):
        """Test adaptive temperature adjustment based on schema failures."""
        pm = self.persona_manager
        original_temp = pm.all_personas["TestPersona"].temperature
        metrics = pm.persona_performance_metrics["TestPersona"]

        # Simulate high schema failure rate
        metrics["total_turns"] = 10
        metrics["schema_failures"] = 3  # 30% failure rate
        metrics["last_adjustment_timestamp"] = (
            time.time() - pm.adjustment_cooldown_seconds - 10
        )  # Ensure cooldown passed

        adjusted_config = pm.get_adjusted_persona_config("TestPersona")
        self.assertLess(adjusted_config.temperature, original_temp)
        self.assertEqual(metrics["last_adjusted_temp"], adjusted_config.temperature)
        self.assertEqual(metrics["total_turns"], 0)  # Should reset after adjustment

    def test_get_adjusted_persona_config_adaptive_max_tokens(self):
        """Test adaptive max_tokens adjustment based on truncation failures."""
        pm = self.persona_manager
        original_max_tokens = pm.all_personas["TestPersona"].max_tokens
        metrics = pm.persona_performance_metrics["TestPersona"]

        # Simulate high truncation failure rate
        metrics["total_turns"] = 10
        metrics["truncation_failures"] = 2  # 20% truncation rate
        metrics["last_adjustment_timestamp"] = (
            time.time() - pm.adjustment_cooldown_seconds - 10
        )  # Ensure cooldown passed

        adjusted_config = pm.get_adjusted_persona_config("TestPersona")
        self.assertGreater(adjusted_config.max_tokens, original_max_tokens)
        self.assertEqual(
            metrics["last_adjusted_max_tokens"], adjusted_config.max_tokens
        )
        self.assertEqual(metrics["total_turns"], 0)  # Should reset after adjustment
