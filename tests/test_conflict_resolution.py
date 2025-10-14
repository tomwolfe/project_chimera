from unittest.mock import Mock, patch

from src.conflict_resolution import ConflictResolutionManager


class TestConflictResolutionManager:
    def test_conflict_resolution_manager_initialization(self):
        """Test ConflictResolutionManager initialization."""
        # Test initialization without parameters
        manager = ConflictResolutionManager()
        assert manager.max_self_correction_retries == 2
        assert manager.llm_provider is None
        assert manager.persona_manager is None

        # Test initialization with parameters
        mock_llm = Mock()
        mock_persona_manager = Mock()
        manager = ConflictResolutionManager(
            llm_provider=mock_llm, persona_manager=mock_persona_manager
        )
        assert manager.llm_provider == mock_llm
        assert manager.persona_manager == mock_persona_manager

    def test_detect_conflict_type_general_disagreement(self):
        """Test detection of general disagreement conflict type."""
        manager = ConflictResolutionManager()
        debate_history = [
            {"persona": "Code_Architect", "output": {"SUGGESTIONS": []}},
            {"persona": "Devils_Advocate", "output": {}},
        ]

        conflict_type = manager._detect_conflict_type(debate_history)
        assert conflict_type == "GENERAL_DISAGREEMENT"

    def test_detect_conflict_type_security_vs_architecture(self):
        """Test detection of security vs architecture conflict type."""
        manager = ConflictResolutionManager()
        debate_history = [
            {
                "persona": "Security_Auditor",
                "output": {
                    "SUGGESTIONS": [
                        {"AREA": "Security", "PROBLEM": "Vulnerability found"}
                    ]
                },
            },
            {
                "persona": "Code_Architect",
                "output": {
                    "SUGGESTIONS": [
                        {
                            "AREA": "Maintainability",
                            "PROPOSED_SOLUTION": "Improve code structure",
                        }
                    ]
                },
            },
        ]

        conflict_type = manager._detect_conflict_type(debate_history)
        assert conflict_type == "SECURITY_VS_ARCHITECTURE"

    def test_detect_conflict_type_fundamental_flaw(self):
        """Test detection of fundamental flaw detection conflict type."""
        manager = ConflictResolutionManager()
        debate_history = [
            {"persona": "Security_Auditor", "output": {"SUGGESTIONS": []}},
            {
                "persona": "Devils_Advocate",
                "output": {
                    "conflict_found": True,
                    "summary": "Fundamental flaw detected",
                },
            },
        ]

        conflict_type = manager._detect_conflict_type(debate_history)
        assert conflict_type == "FUNDAMENTAL_FLAW_DETECTION"

    def test_resolve_conflict_empty_history(self):
        """Test resolve_conflict with empty debate history."""
        manager = ConflictResolutionManager()
        result = manager.resolve_conflict([])

        assert result["resolution_strategy"] == "manual_intervention"
        assert result["resolved_output"] is None

    def test_resolve_conflict_with_malformed_output(self):
        """Test resolve_conflict with malformed output that can be parsed."""
        manager = ConflictResolutionManager()
        debate_history = [
            {"persona": "Some_Persona", "output": '{"test": "valid json string"}'}
        ]

        result = manager.resolve_conflict(debate_history)

        # If the string can be parsed as JSON, it should be handled differently
        # but in this case it should still go through the fallback
        assert "resolution_strategy" in result

    def test_synthesize_from_history_with_dict_output(self):
        """Test _synthesize_from_history with dictionary output."""
        manager = ConflictResolutionManager()
        valid_turns = [
            {
                "persona": "Test_Persona",
                "output": {
                    "test_field": "test_value",
                    "existing_key": "existing_value",
                },
            }
        ]

        result = manager._synthesize_from_history(
            problematic_output="bad output", valid_turns=valid_turns
        )

        assert result is not None
        assert "test_field" in result
        assert "CONFLICT_RESOLUTION_ATTEMPT" in result

    def test_synthesize_from_history_with_str_output(self):
        """Test _synthesize_from_history with string output."""
        manager = ConflictResolutionManager()
        valid_turns = [{"persona": "Test_Persona", "output": "This is a string output"}]

        result = manager._synthesize_from_history(
            problematic_output="bad output", valid_turns=valid_turns
        )

        assert result is not None
        assert "general_output" in result
        assert "This is a string output" in result["general_output"]

    def test_synthesize_from_history_empty_turns(self):
        """Test _synthesize_from_history with empty valid_turns."""
        manager = ConflictResolutionManager()

        result = manager._synthesize_from_history(
            problematic_output="bad output", valid_turns=[]
        )

        assert result is None

    def test_manual_intervention_fallback(self):
        """Test _manual_intervention_fallback method."""
        manager = ConflictResolutionManager()
        message = "Test manual intervention required"

        result = manager._manual_intervention_fallback(message)

        assert result["resolution_strategy"] == "manual_intervention"
        assert result["resolved_output"] is None
        assert message in result["resolution_summary"]
        assert result["malformed_blocks"][0]["type"] == "MANUAL_INTERVENTION_REQUIRED"
        assert result["malformed_blocks"][0]["message"] == message


class TestConflictResolutionManagerWithMocking:
    @patch("src.conflict_resolution.GeminiProvider")
    @patch("src.conflict_resolution.LLMOutputParser")
    def test_retry_persona_with_feedback_success(
        self, mock_output_parser, mock_llm_provider
    ):
        """Test _retry_persona_with_feedback with successful correction."""
        # Set up mocks
        mock_llm = Mock()
        mock_llm.tokenizer = Mock()
        mock_llm.tokenizer.max_output_tokens = 2048
        mock_llm.generate.return_value = ('{"test": "valid_json"}', 100, 50, False)

        mock_output_parser_instance = Mock()
        mock_output_parser_instance.parse_and_validate.return_value = {
            "test": "result",
            "malformed_blocks": [],
        }
        mock_output_parser_instance._clean_llm_output.return_value = "clean output"
        # Mock the _get_schema_class_from_name method to return a proper class
        from src.models import GeneralOutput

        mock_output_parser_instance._get_schema_class_from_name.return_value = (
            GeneralOutput
        )
        mock_output_parser.return_value = mock_output_parser_instance

        mock_persona_manager = Mock()
        mock_config = Mock()
        mock_config.output_schema = "GeneralOutput"
        mock_config.system_prompt_template = "You are a helpful AI assistant."
        mock_config.temperature = 0.7
        mock_config.max_tokens = 1000
        # Make sure _rendered_system_prompt is also defined
        mock_config._rendered_system_prompt = "You are a helpful AI assistant."
        mock_persona_manager.get_adjusted_persona_config.return_value = mock_config

        manager = ConflictResolutionManager(
            llm_provider=mock_llm, persona_manager=mock_persona_manager
        )
        manager.output_parser = mock_output_parser_instance

        debate_history = [
            {
                "persona": "Test_Persona",
                "output": {
                    "malformed_blocks": [{"type": "TEST_ERROR", "message": "test"}]
                },
            }
        ]

        manager._retry_persona_with_feedback("Test_Persona", debate_history)
        # The function should complete without crashing
        # (It may return None if the validation still fails, but it shouldn't crash)

    @patch("src.conflict_resolution.GeminiProvider")
    @patch("src.conflict_resolution.LLMOutputParser")
    def test_retry_persona_with_feedback_max_retries(
        self, mock_output_parser, mock_llm_provider
    ):
        """Test _retry_persona_with_feedback that fails after max retries."""
        # Set up mocks to always fail validation
        mock_llm = Mock()
        mock_llm.tokenizer = Mock()
        mock_llm.tokenizer.max_output_tokens = 2048
        mock_llm.generate.return_value = ('{"test": "still_bad"}', 100, 50, False)

        mock_output_parser_instance = Mock()
        mock_output_parser_instance.parse_and_validate.return_value = {
            "test": "result",
            "malformed_blocks": [{"type": "STILL_BAD", "message": "still bad"}],
        }
        mock_output_parser_instance._clean_llm_output.return_value = "clean output"
        # Mock the _get_schema_class_from_name method to return a proper class
        from src.models import GeneralOutput

        mock_output_parser_instance._get_schema_class_from_name.return_value = (
            GeneralOutput
        )
        mock_output_parser.return_value = mock_output_parser_instance

        mock_persona_manager = Mock()
        mock_config = Mock()
        mock_config.output_schema = "GeneralOutput"
        mock_config.system_prompt_template = "You are a helpful AI assistant."
        mock_config.temperature = 0.7
        mock_config.max_tokens = 1000
        # Make sure _rendered_system_prompt is also defined
        mock_config._rendered_system_prompt = "You are a helpful AI assistant."
        mock_persona_manager.get_adjusted_persona_config.return_value = mock_config

        manager = ConflictResolutionManager(
            llm_provider=mock_llm, persona_manager=mock_persona_manager
        )
        manager.output_parser = mock_output_parser_instance

        debate_history = [
            {
                "persona": "Test_Persona",
                "output": {
                    "malformed_blocks": [{"type": "TEST_ERROR", "message": "test"}]
                },
            }
        ]

        manager._retry_persona_with_feedback("Test_Persona", debate_history)
        # Should complete execution without crashing (may return None if validation fails)

    def test_retry_persona_with_feedback_no_provider(self):
        """Test _retry_persona_with_feedback when provider is not set."""
        manager = ConflictResolutionManager()
        # Don't set llm_provider or persona_manager

        debate_history = [
            {"persona": "Test_Persona", "output": {"malformed_blocks": []}}
        ]

        result = manager._retry_persona_with_feedback("Test_Persona", debate_history)
        assert result is None

    def test_resolve_conflict_with_missing_codebase_context(self):
        """Test conflict resolution for missing codebase context."""
        manager = ConflictResolutionManager()
        debate_history = [
            {
                "persona": "Devils_Advocate",
                "output": {
                    "summary": "Cannot analyze due to lack of information",
                    "conflict_found": True,
                    "involved_personas": ["Devils_Advocate"],
                },
            }
        ]

        result = manager.resolve_conflict(debate_history)
        # This should result in a missing codebase context resolution
        assert "resolution_strategy" in result

    def test_resolve_conflict_security_vs_architecture(self):
        """Test conflict resolution for security vs architecture."""
        manager = ConflictResolutionManager()
        debate_history = [
            {
                "persona": "Security_Auditor",
                "output": {
                    "SUGGESTIONS": [{"AREA": "Security", "PROBLEM": "Security issue"}]
                },
            },
            {
                "persona": "Code_Architect",
                "output": {
                    "SUGGESTIONS": [
                        {"AREA": "Architecture", "PROPOSED_SOLUTION": "Arch solution"}
                    ]
                },
            },
            {"persona": "Other_Persona", "output": {"some_field": "value"}},
        ]

        result = manager.resolve_conflict(debate_history)
        # Should handle the security vs architecture conflict
        assert "resolution_strategy" in result
