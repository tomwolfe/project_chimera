from pathlib import Path

import pytest

from src.utils.core_helpers.path_utils import (
    PROJECT_ROOT,
    _map_incorrect_file_path,
    sanitize_and_validate_file_path,
)


def test_sanitize_and_validate_file_path_safe():
    """Test that a valid, in-project path is sanitized and validated correctly."""
    # Assuming 'src/utils/path_utils.py' is a valid path within PROJECT_ROOT
    assert (
        sanitize_and_validate_file_path("src/utils/path_utils.py")
        == "src/utils/path_utils.py"
    )


def test_sanitize_and_validate_file_path_traversal():
    """Test that path traversal attempts are blocked."""
    with pytest.raises(ValueError, match="outside the allowed project directory"):
        sanitize_and_validate_file_path("../../etc/passwd")


def test_sanitize_and_validate_file_path_invalid_chars():
    """Test that invalid characters are removed from the path."""
    # The sanitize_and_validate_file_path function removes invalid characters
    # but still checks against project root.
    # If the path becomes valid after sanitization and is within project root, it passes.
    # If it's still outside or invalid, it fails.
    # For this test, we'll assume a path that becomes valid and in-project.
    # Example: "src/my<file>.py" -> "src/myfile.py"
    # Since the original codebase's sanitize_and_validate_file_path removes '/' from invalid chars,
    # this test needs to be careful. Let's use a simpler invalid char.
    # Re-checking the original codebase's sanitize_and_validate_file_path:
    # sanitized_path_str = re.sub(r'[<>:"\\|?*\x00-\x1f\x7f]', "", raw_path)
    # It removes <, >, :, ", \, |, ?, *, and control characters.
    # It does NOT remove / or ..
    # The `_map_incorrect_file_path` handles `../`
    # Let's test with a simple invalid char.
    assert (
        sanitize_and_validate_file_path("src/my:file.py") == "src/myfile.py"
    )  # Assuming src/myfile.py is a valid path


def test_map_incorrect_file_path_common_hallucinations():
    """Test that common LLM hallucinated paths are mapped to correct ones."""
    assert _map_incorrect_file_path("reasoning_engine.py") == "core.py"
    assert _map_incorrect_file_path("src/main.py") == "app.py"
    assert _map_incorrect_file_path("services/llm_service.py") == "src/llm/client.py"
    assert (
        _map_incorrect_file_path("src/prompt_manager.py")
        == "src/utils/prompt_optimizer.py"
    )
    assert _map_incorrect_file_path("src/llm_interface.py") == "src/llm_provider.py"


def test_map_incorrect_file_path_core_prefix():
    """Test mapping for 'core/' prefix that should be at root."""
    assert _map_incorrect_file_path("core/core.py") == "core.py"
    assert _map_incorrect_file_path("core/config.py") == "config.py"
    assert (
        _map_incorrect_file_path("core/some_module.py") == "src/some_module.py"
    )  # Should map to src/ if not a root file


def test_map_incorrect_file_path_services_prefix():
    """Test mapping for 'services/' prefix."""
    assert (
        _map_incorrect_file_path("services/data_processing_service.py")
        == "src/data_processing_service.py"
    )


def test_map_incorrect_file_path_utils_prefix():
    """Test mapping for 'utils/' prefix."""
    assert (
        _map_incorrect_file_path("utils/file_operations.py")
        == "src/utils/file_operations.py"
    )


def test_map_incorrect_file_path_no_change():
    """Test that correct paths are not altered."""
    assert _map_incorrect_file_path("app.py") == "app.py"
    assert (
        _map_incorrect_file_path("src/utils/path_utils.py") == "src/utils/path_utils.py"
    )


def test_can_create_file_in_existing_dir():
    """Test can_create_file for a file in an existing directory."""
    # Create a dummy directory for testing
    test_dir = Path(PROJECT_ROOT) / "temp_test_dir"
    test_dir.mkdir(exist_ok=True)
    try:
        test_file_path = str(test_dir / "new_file.py")
        from src.utils.path_utils import can_create_file

        assert can_create_file(test_file_path) is True
    finally:
        test_dir.rmdir()


def test_can_create_file_in_non_existent_dir():
    """Test can_create_file for a file in a non-existent directory."""
    test_file_path = str(Path(PROJECT_ROOT) / "non_existent_dir" / "new_file.py")
    from src.utils.path_utils import can_create_file

    assert can_create_file(test_file_path) is False  # Parent dir doesn't exist


def test_can_create_file_root_level():
    """Test can_create_file for a file at the project root."""
    test_file_path = str(Path(PROJECT_ROOT) / "new_root_file.py")
    from src.utils.path_utils import can_create_file

    assert can_create_file(test_file_path) is True


def test_can_create_directory_in_existing_dir():
    """Test can_create_file for a new directory in an existing parent."""
    test_parent_dir = Path(PROJECT_ROOT) / "existing_parent"
    test_parent_dir.mkdir(exist_ok=True)
    try:
        test_new_dir_path = str(test_parent_dir / "new_subdir")
        from src.utils.path_utils import can_create_file

        assert can_create_file(test_new_dir_path) is True
    finally:
        test_parent_dir.rmdir()


def test_can_create_directory_nested_non_existent():
    """Test can_create_file for a nested non-existent directory."""
    test_path = str(Path(PROJECT_ROOT) / "non_existent_parent" / "new_nested_dir")
    from src.utils.path_utils import can_create_file

    assert can_create_file(test_path) is False  # Intermediate parent doesn't exist
