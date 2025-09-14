import pytest
import json
from unittest.mock import MagicMock, patch

# Assuming ConflictResolutionManager is in src/conflict_resolution.py
from src.conflict_resolution import ConflictResolutionManager
from src.models import CritiqueOutput  # NEW: Import CritiqueOutput for schema mocking
from src.utils.output_parser import LLMOutputParser  # NEW: Import LLMOutputParser


@pytest.fixture
def conflict_manager():
    """Provides an instance of ConflictResolutionManager for testing."""
    # Mock llm_provider and persona_manager as they are now required for ConflictResolutionManager
    mock_llm_provider = MagicMock()
    mock_llm_provider.tokenizer = MagicMock()
    mock_llm_provider.tokenizer.count_tokens.side_effect = lambda text: len(text) // 4
    mock_llm_provider.tokenizer.max_output_tokens = 8192
    mock_llm_provider.generate.return_value = (
        "{}",
        10,
        10,
        False,
    )  # Default for self-correction

    mock_persona_manager = MagicMock()
    mock_persona_manager.get_adjusted_persona_config.return_value = MagicMock(
        system_prompt="Mock system prompt", temperature=0.1, max_tokens=4096
    )
    mock_persona_manager.PERSONA_OUTPUT_SCHEMAS = {
        "PersonaB": CritiqueOutput,  # Mock a schema for PersonaB
        "Constructive_Critic": CritiqueOutput,  # Mock a schema for Constructive_Critic
        "Devils_Advocate": MagicMock(),  # Mock a schema for Devils_Advocate
        "Self_Improvement_Analyst": MagicMock(),  # Mock a schema for Self_Improvement_Analyst
        "GeneralOutput": MagicMock(),  # Mock GeneralOutput
    }

    # Create a real instance of LLMOutputParser and then mock its method
    real_output_parser = LLMOutputParser()
    with patch.object(
        real_output_parser, "parse_and_validate"
    ) as mock_parse_and_validate:
        manager = ConflictResolutionManager(
            llm_provider=mock_llm_provider, persona_manager=mock_persona_manager
        )
        manager.output_parser = (
            real_output_parser  # Ensure the manager uses this mocked parser
        )
        yield manager


def test_resolve_conflict_with_malformed_string(conflict_manager):
    """Tests resolution when the latest output is a non-JSON string."""
    history = [
        {
            "persona": "PersonaA",
            "output": {
                "CRITIQUE_SUMMARY": "Valid output 1",
                "CRITIQUE_POINTS": [],
                "SUGGESTIONS": [],
                "malformed_blocks": [],
            },
        },
        {"persona": "PersonaB", "output": "This is not JSON, but some text."},
    ]
    result = conflict_manager.resolve_conflict(history)
    assert result is not None
    # Expecting manual intervention as the string cannot be parsed
    assert result["resolution_strategy"] == "manual_intervention"
    assert "Automated resolution failed" in result["resolution_summary"]
    assert "PARSED_STRING_OUTPUT" not in str(
        result["malformed_blocks"]
    )  # Ensure it wasn't parsed as valid JSON


def test_resolve_conflict_with_string_json(conflict_manager):
    """Tests resolution when the latest output is a valid JSON string."""
    history = [
        {
            "persona": "PersonaA",
            "output": {
                "CRITIQUE_SUMMARY": "Valid output 1",
                "CRITIQUE_POINTS": [],
                "SUGGESTIONS": [],
                "malformed_blocks": [],
            },
        },
        {
            "persona": "PersonaB",
            "output": '{"CRITIQUE_SUMMARY": "This is valid JSON as a string", "CRITIQUE_POINTS": [], "SUGGESTIONS": [], "malformed_blocks": []}',
        },
    ]
    # Mock the schema for PersonaB to be CritiqueOutput for successful parsing
    conflict_manager.persona_manager.PERSONA_OUTPUT_SCHEMAS["PersonaB"] = CritiqueOutput

    result = conflict_manager.resolve_conflict(history)
    assert result is not None
    # Expecting successful parsing
    assert result["resolution_strategy"] == "parsed_malformed_string"
    assert (
        result["resolved_output"]["CRITIQUE_SUMMARY"]
        == "This is valid JSON as a string"
    )
    assert any(
        block["type"] == "PARSED_STRING_OUTPUT" for block in result["malformed_blocks"]
    )


def test_resolve_conflict_with_malformed_dict_no_summary(conflict_manager):
    """Tests resolution when the latest output is a dict but lacks expected keys."""
    history = [
        {
            "persona": "PersonaA",
            "output": {
                "CRITIQUE_SUMMARY": "Valid output 1",
                "CRITIQUE_POINTS": [],
                "SUGGESTIONS": [],
                "malformed_blocks": [],
            },
        },
        {
            "persona": "PersonaB",
            "output": {
                "error": "Something failed",
                "details": "Missing CRITIQUE_SUMMARY",
                "malformed_blocks": [{"type": "SCHEMA_VALIDATION_ERROR"}],
            },
        },
    ]
    # Mock the persona manager to return a valid config for PersonaB
    conflict_manager.persona_manager.get_adjusted_persona_config.return_value = (
        MagicMock(system_prompt="Mock system prompt", temperature=0.1, max_tokens=4096)
    )
    # FIX: Mock the generate method to return a valid output for self-correction
    conflict_manager.llm_provider.generate.side_effect = [
        (
            '{"CRITIQUE_SUMMARY": "Self-corrected output", "CRITIQUE_POINTS": [], "SUGGESTIONS": [], "malformed_blocks": []}',
            100,
            50,
            False,
        )
    ]
    # FIX: Mock the parser for the self-correction attempt correctly
    conflict_manager.output_parser.parse_and_validate.side_effect = [  # FIX: Access the mocked method directly
        {
            "CRITIQUE_SUMMARY": "Self-corrected output",
            "CRITIQUE_POINTS": [],
            "SUGGESTIONS": [],
            "malformed_blocks": [],
        }
    ]

    result = conflict_manager.resolve_conflict(history)
    assert result is not None
    # Expecting self-correction to succeed
    assert result["resolution_strategy"] == "self_correction_retry"
    assert (
        "Persona 'PersonaB' self-corrected its output" in result["resolution_summary"]
    )
    assert result["resolved_output"]["CRITIQUE_SUMMARY"] == "Self-corrected output"


def test_resolve_conflict_with_insufficient_history(conflict_manager):
    """Tests resolution when there's not enough valid history for synthesis."""
    history = [
        {
            "persona": "PersonaA",
            "output": {
                "error": "Failed",
                "malformed_blocks": [{"type": "SCHEMA_ERROR"}],
            },
        }
    ]
    # Mock the persona manager to return a valid config for PersonaA
    conflict_manager.persona_manager.get_adjusted_persona_config.return_value = (
        MagicMock(system_prompt="Mock system prompt", temperature=0.1, max_tokens=4096)
    )
    # FIX: Mock the generate method to return an invalid output for self-correction
    conflict_manager.llm_provider.generate.side_effect = [
        ('{"invalid": "output"}', 10, 10, True)
    ]
    # FIX: Mock the parser for the self-correction attempt correctly
    conflict_manager.output_parser.parse_and_validate.side_effect = [  # FIX: Access the mocked method directly
        {"invalid": "output", "malformed_blocks": [{"type": "SCHEMA_VALIDATION_ERROR"}]}
    ]

    result = conflict_manager.resolve_conflict(history)
    assert result is not None
    # Expecting manual intervention due to lack of valid history for synthesis AND self-correction failure
    assert result["resolution_strategy"] == "manual_intervention"
    assert "Automated resolution failed" in result["resolution_summary"]


def test_resolve_conflict_with_valid_history_and_problematic_output(conflict_manager):
    """Tests resolution when there are valid turns and a problematic latest output."""
    history = [
        {
            "persona": "PersonaA",
            "output": {
                "CRITIQUE_SUMMARY": "Valid output 1",
                "CRITIQUE_POINTS": [],
                "SUGGESTIONS": [],
                "malformed_blocks": [],
            },
        },
        {
            "persona": "PersonaB",
            "output": {
                "CRITIQUE_SUMMARY": "Valid output 2",
                "CRITIQUE_POINTS": [],
                "SUGGESTIONS": [],
                "malformed_blocks": [],
            },
        },
        {
            "persona": "PersonaC",
            "output": {"malformed_blocks": [{"type": "CONTENT_MISALIGNMENT"}]},
        },
    ]
    # Mock the persona manager to return a valid config for PersonaC
    conflict_manager.persona_manager.get_adjusted_persona_config.return_value = (
        MagicMock(system_prompt="Mock system prompt", temperature=0.1, max_tokens=4096)
    )
    # FIX: Mock the generate method to return an invalid output for self-correction
    conflict_manager.llm_provider.generate.side_effect = [
        ('{"invalid": "output"}', 10, 10, True)
    ]
    # FIX: Mock the parser for the self-correction attempt correctly
    conflict_manager.output_parser.parse_and_validate.side_effect = [  # FIX: Access the mocked method directly
        {"invalid": "output", "malformed_blocks": [{"type": "SCHEMA_VALIDATION_ERROR"}]}
    ]

    result = conflict_manager.resolve_conflict(history)
    assert result is not None
    # Expecting synthesis from the last valid turn (PersonaB)
    assert result["resolution_strategy"] == "synthesis_from_history"
    assert (
        "Automated synthesis from previous valid debate turns."
        in result["resolution_summary"]
    )
    assert "Valid output 2" in result["resolved_output"]["CONFLICT_RESOLUTION_ATTEMPT"]


def test_resolve_conflict_no_problem_in_latest_output(conflict_manager):
    """Tests the scenario where the latest output is valid."""
    history = [
        {
            "persona": "PersonaA",
            "output": {
                "CRITIQUE_SUMMARY": "Valid output 1",
                "CRITIQUE_POINTS": [],
                "SUGGESTIONS": [],
                "malformed_blocks": [],
            },
        },
        {
            "persona": "PersonaB",
            "output": {
                "CRITIQUE_SUMMARY": "Valid output 2",
                "CRITIQUE_POINTS": [],
                "SUGGESTIONS": [],
                "malformed_blocks": [],
            },
        },
    ]
    # If the latest output is valid, the manager should still attempt synthesis based on the history.
    # The current implementation prioritizes synthesis if called.
    result = conflict_manager.resolve_conflict(history)
    assert result is not None
    assert (
        result["resolution_strategy"] == "synthesis_from_history"
    )  # FIX: Expect synthesis from history
    assert (
        "Valid output 2" in result["resolved_output"]["resolution_summary"]
    )  # FIX: Check resolution_summary


def test_resolve_conflict_empty_history(conflict_manager):
    """Tests resolution when the debate history is empty."""
    history = []
    result = conflict_manager.resolve_conflict(history)
    assert result is not None
    assert result["resolution_strategy"] == "manual_intervention"
    assert "Empty debate history" in result["resolution_summary"]


def test_retry_persona_with_feedback_success(conflict_manager):
    """Tests successful self-correction with feedback."""
    persona_name = "Constructive_Critic"
    history = [
        {
            "persona": persona_name,
            "output": {"malformed_blocks": [{"type": "SCHEMA_VALIDATION_ERROR"}]},
        }
    ]
    # Mock the LLM to return a valid output on the first retry
    conflict_manager.llm_provider.generate.side_effect = [
        (
            '{"CRITIQUE_SUMMARY": "Corrected output", "CRITIQUE_POINTS": [], "SUGGESTIONS": [], "malformed_blocks": []}',
            100,
            50,
            False,
        )
    ]
    conflict_manager.output_parser.parse_and_validate.side_effect = [  # FIX: Access the mocked method directly
        {
            "CRITIQUE_SUMMARY": "Corrected output",
            "CRITIQUE_POINTS": [],
            "SUGGESTIONS": [],
            "malformed_blocks": [],
        }
    ]

    resolved_output = conflict_manager._retry_persona_with_feedback(
        persona_name, history
    )
    assert resolved_output is not None
    assert resolved_output["CRITIQUE_SUMMARY"] == "Corrected output"
    assert (
        conflict_manager.llm_provider.generate.call_count == 1
    )  # Only one call for successful retry


def test_retry_persona_with_feedback_failure(conflict_manager):
    """Tests self-correction failure after max retries."""
    persona_name = "Constructive_Critic"
    history = [
        {
            "persona": persona_name,
            "output": {"malformed_blocks": [{"type": "SCHEMA_VALIDATION_ERROR"}]},
        }
    ]
    # Mock the LLM to always return invalid output
    conflict_manager.llm_provider.generate.side_effect = [
        ('{"invalid": "output"}', 10, 10, True),
        ('{"invalid": "output2"}', 10, 10, True),
        ('{"invalid": "output3"}', 10, 10, True),  # Max retries is 2, so 3 calls total
    ]
    conflict_manager.output_parser.parse_and_validate.side_effect = [  # FIX: Access the mocked method directly
        {
            "invalid": "output",
            "malformed_blocks": [{"type": "SCHEMA_VALIDATION_ERROR"}],
        },
        {
            "invalid": "output2",
            "malformed_blocks": [{"type": "SCHEMA_VALIDATION_ERROR"}],
        },
        {
            "invalid": "output3",
            "malformed_blocks": [{"type": "SCHEMA_VALIDATION_ERROR"}],
        },
    ]

    resolved_output = conflict_manager._retry_persona_with_feedback(
        persona_name, history
    )
    assert resolved_output is None
    assert (
        conflict_manager.llm_provider.generate.call_count
        == conflict_manager.max_self_correction_retries + 1
    )
