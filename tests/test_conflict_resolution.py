# tests/test_conflict_resolution.py
import pytest
import json
from unittest.mock import MagicMock # Import MagicMock for potential future use if needed

# Assuming ConflictResolutionManager is in src/conflict_resolution.py
from src.conflict_resolution import ConflictResolutionManager

@pytest.fixture
def conflict_manager():
    """Provides an instance of ConflictResolutionManager for testing."""
    return ConflictResolutionManager()

def test_resolve_conflict_with_malformed_string(conflict_manager):
    """Tests resolution when the latest output is a non-JSON string."""
    history = [
        {"persona": "PersonaA", "output": {"CRITIQUE_SUMMARY": "Valid output 1"}},
        {"persona": "PersonaB", "output": "This is not JSON, but some text."}
    ]
    result = conflict_manager.resolve_conflict(history)
    assert result is not None
    # Expecting manual intervention as the string cannot be parsed
    assert result["resolution_strategy"] == "manual_intervention"
    assert "Automated resolution failed" in result["resolution_summary"]
    assert "PARSED_STRING_OUTPUT" not in result["malformed_blocks"][0]["type"] # Ensure it wasn't parsed as valid JSON

def test_resolve_conflict_with_string_json(conflict_manager):
    """Tests resolution when the latest output is a valid JSON string."""
    history = [
        {"persona": "PersonaA", "output": {"CRITIQUE_SUMMARY": "Valid output 1"}},
        {"persona": "PersonaB", "output": '{"CRITIQUE_SUMMARY": "This is valid JSON as a string"}'}
    ]
    result = conflict_manager.resolve_conflict(history)
    assert result is not None
    # Expecting successful parsing
    assert result["resolution_strategy"] == "parsed_malformed_string"
    assert result["resolved_output"]["CRITIQUE_SUMMARY"] == "This is valid JSON as a string"
    assert result["malformed_blocks"][0]["type"] == "PARSED_STRING_OUTPUT"

def test_resolve_conflict_with_malformed_dict_no_summary(conflict_manager):
    """Tests resolution when the latest output is a dict but lacks expected keys."""
    history = [
        {"persona": "PersonaA", "output": {"CRITIQUE_SUMMARY": "Valid output 1"}},
        {"persona": "PersonaB", "output": {"error": "Something failed", "details": "Missing CRITIQUE_SUMMARY"}}
    ]
    result = conflict_manager.resolve_conflict(history)
    assert result is not None
    # Expecting synthesis from history as the latest output is problematic but not a parsable string
    assert result["resolution_strategy"] == "synthesis_from_history"
    assert "Automated synthesis from previous valid debate turns." in result["resolution_summary"]
    # Check if the synthesis incorporated information from the previous valid turn
    assert "Valid output 1" in result["resolved_output"]["CONFLICT_RESOLUTION_ATTEMPT"]

def test_resolve_conflict_with_insufficient_history(conflict_manager):
    """Tests resolution when there's not enough valid history for synthesis."""
    history = [
        {"persona": "PersonaA", "output": {"error": "Failed", "malformed_blocks": [{"type": "SCHEMA_ERROR"}]}}
    ]
    result = conflict_manager.resolve_conflict(history)
    assert result is not None
    # Expecting manual intervention due to lack of valid history for synthesis
    assert result["resolution_strategy"] == "manual_intervention"
    assert "Automated resolution failed" in result["resolution_summary"]

def test_resolve_conflict_with_valid_history_and_problematic_output(conflict_manager):
    """Tests resolution when there are valid turns and a problematic latest output."""
    history = [
        {"persona": "PersonaA", "output": {"CRITIQUE_SUMMARY": "Valid output 1"}},
        {"persona": "PersonaB", "output": {"CRITIQUE_SUMMARY": "Valid output 2"}},
        {"persona": "PersonaC", "output": {"malformed_blocks": [{"type": "CONTENT_MISALIGNMENT"}]}}
    ]
    result = conflict_manager.resolve_conflict(history)
    assert result is not None
    # Expecting synthesis from the last valid turn (PersonaB)
    assert result["resolution_strategy"] == "synthesis_from_history"
    assert "Automated synthesis from previous valid debate turns." in result["resolution_summary"]
    assert "Valid output 2" in result["resolved_output"]["CONFLICT_RESOLUTION_ATTEMPT"]

def test_resolve_conflict_no_problem_in_latest_output(conflict_manager):
    """Tests the scenario where the latest output is valid."""
    history = [
        {"persona": "PersonaA", "output": {"CRITIQUE_SUMMARY": "Valid output 1"}},
        {"persona": "PersonaB", "output": {"CRITIQUE_SUMMARY": "Valid output 2"}}
    ]
    # If the latest output is valid, the manager should still attempt synthesis based on the history.
    # The current implementation prioritizes synthesis if called.
    result = conflict_manager.resolve_conflict(history)
    assert result is not None
    assert result["resolution_strategy"] == "synthesis_from_history"
    assert "Valid output 2" in result["resolved_output"]["CONFLICT_RESOLUTION_ATTEMPT"]

def test_resolve_conflict_empty_history(conflict_manager):
    """Tests resolution when the debate history is empty."""
    history = []
    result = conflict_manager.resolve_conflict(history)
    assert result is not None
    assert result["resolution_strategy"] == "manual_intervention"
    assert "Empty debate history" in result["resolution_summary"]
