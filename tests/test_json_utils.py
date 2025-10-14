"""Tests for the JSON utilities module."""

import json
from unittest.mock import patch

import numpy as np
from pydantic import BaseModel, Field

from src.utils.core_helpers.json_utils import (
    convert_to_json_friendly,
    safe_json_dumps,
    safe_json_loads,
)


class TestModel(BaseModel):
    """Test Pydantic model for testing."""

    name: str
    value: int = Field(default=0)


class TestConvertToJsonFriendly:
    """Test suite for convert_to_json_friendly function."""

    def test_convert_regular_dict(self):
        """Test conversion of regular Python dict."""
        data = {"name": "test", "value": 42}
        result = convert_to_json_friendly(data)
        assert result == data

    def test_convert_regular_list(self):
        """Test conversion of regular Python list."""
        data = [1, 2, 3, {"nested": "value"}]
        result = convert_to_json_friendly(data)
        assert result == data

    def test_convert_pydantic_model(self):
        """Test conversion of Pydantic model."""
        model = TestModel(name="test", value=42)
        result = convert_to_json_friendly(model)
        expected = {"name": "test", "value": 42}
        assert result == expected

    def test_convert_nested_dict_with_pydantic_model(self):
        """Test conversion of nested dict containing Pydantic model."""
        model = TestModel(name="inner", value=123)
        data = {"outer": "value", "inner_model": model}
        result = convert_to_json_friendly(data)
        expected = {"outer": "value", "inner_model": {"name": "inner", "value": 123}}
        assert result == expected

    def test_convert_nested_list_with_pydantic_model(self):
        """Test conversion of nested list containing Pydantic model."""
        model = TestModel(name="item", value=99)
        data = [1, model, 3]
        result = convert_to_json_friendly(data)
        expected = [1, {"name": "item", "value": 99}, 3]
        assert result == expected

    def test_convert_numpy_scalar(self):
        """Test conversion of NumPy scalar."""
        numpy_int = np.int32(42)
        result = convert_to_json_friendly(numpy_int)
        assert result == 42
        assert isinstance(result, int)

    def test_convert_numpy_float(self):
        """Test conversion of NumPy float."""
        numpy_float = np.float64(3.14)
        result = convert_to_json_friendly(numpy_float)
        assert result == 3.14
        assert isinstance(result, float)

    def test_convert_numpy_array(self):
        """Test conversion of NumPy array."""
        numpy_array = np.array([1, 2, 3])
        result = convert_to_json_friendly(numpy_array)
        assert result == [1, 2, 3]
        assert isinstance(result, list)

    def test_convert_numpy_nested_in_dict(self):
        """Test conversion of NumPy types nested in dict."""
        data = {
            "int_val": np.int64(123),
            "float_val": np.float32(45.6),
            "array_val": np.array([1, 2, 3]),
        }
        result = convert_to_json_friendly(data)
        expected = {
            "int_val": 123,
            "float_val": 45.6,  # May have precision differences
            "array_val": [1, 2, 3],
        }
        # Check each value individually due to floating point precision
        assert result["int_val"] == expected["int_val"]
        assert result["array_val"] == expected["array_val"]
        # For float, check with tolerance
        assert abs(result["float_val"] - expected["float_val"]) < 0.001

    def test_convert_unconvertible_type(self):
        """Test conversion of a type that should remain unchanged."""

        class CustomClass:
            def __init__(self):
                self.value = "test"

        obj = CustomClass()
        result = convert_to_json_friendly(obj)
        assert result is obj  # Should return the same object


class TestSafeJsonLoads:
    """Test suite for safe_json_loads function."""

    def test_safe_json_loads_valid_json(self):
        """Test loading valid JSON string."""
        json_str = '{"name": "test", "value": 42}'
        result = safe_json_loads(json_str)
        expected = {"name": "test", "value": 42}
        assert result == expected

    def test_safe_json_loads_valid_json_list(self):
        """Test loading valid JSON list string."""
        json_str = '[1, 2, 3, {"nested": "value"}]'
        result = safe_json_loads(json_str)
        expected = [1, 2, 3, {"nested": "value"}]
        assert result == expected

    def test_safe_json_loads_invalid_json(self):
        """Test loading invalid JSON string."""
        json_str = '{"name": "test", "value":}'  # Invalid JSON
        result = safe_json_loads(json_str)
        assert result is None

    def test_safe_json_loads_invalid_json_with_default(self):
        """Test loading invalid JSON with default value."""
        json_str = '{"name": "test", "value":}'  # Invalid JSON
        default_value = {"default": "value"}
        result = safe_json_loads(json_str, default_value=default_value)
        assert result == default_value

    def test_safe_json_loads_non_string_input(self):
        """Test loading non-string input."""
        result = safe_json_loads(123)  # Not a string
        assert result is None

    def test_safe_json_loads_non_string_input_with_default(self):
        """Test loading non-string input with default value."""
        default_value = {"default": "value"}
        result = safe_json_loads(123, default_value=default_value)
        assert result == default_value

    def test_safe_json_loads_json_decode_error(self):
        """Test handling of JSONDecodeError specifically."""
        invalid_json = '{"unclosed": "string"'
        result = safe_json_loads(invalid_json)
        assert result is None

    def test_safe_json_loads_type_error(self):
        """Test handling of TypeError during JSON parsing."""
        invalid_json = '["unclosed array"'
        result = safe_json_loads(invalid_json)
        assert result is None


class TestSafeJsonDumps:
    """Test suite for safe_json_dumps function."""

    def test_safe_json_dumps_simple_data(self):
        """Test dumping simple JSON-serializable data."""
        data = {"name": "test", "value": [1, 2, 3]}
        result = safe_json_dumps(data)
        expected = json.dumps(data)
        assert result == expected

    def test_safe_json_dumps_with_indent(self):
        """Test dumping with indentation."""
        data = {"name": "test", "value": 42}
        result = safe_json_dumps(data, indent=2)
        parsed_result = json.loads(result)
        assert parsed_result == data

    def test_safe_json_dumps_non_serializable_with_default(self):
        """Test dumping non-serializable data with default handler."""

        class CustomClass:
            def __str__(self):
                return "custom_object"

        data = {"name": "test", "custom": CustomClass()}
        result = safe_json_dumps(data, default=str)
        parsed_result = json.loads(result)
        assert parsed_result["name"] == "test"
        assert parsed_result["custom"] == "custom_object"

    def test_safe_json_dumps_non_serializable_with_fallback(self):
        """Test dumping non-serializable data with fallback to str."""

        class CustomClass:
            def __str__(self):
                return "custom_object"

        data = {"name": "test", "custom": CustomClass()}
        result = safe_json_dumps(data)
        parsed_result = json.loads(result)
        assert parsed_result["name"] == "test"
        assert parsed_result["custom"] == "custom_object"

    def test_safe_json_dumps_completely_unserializable(self):
        """Test dumping completely unserializable data with error fallback."""

        class UnserializableClass:
            def __str__(self):
                raise Exception("Cannot convert to string")

        data = {"name": "test", "unserializable": UnserializableClass()}
        result = safe_json_dumps(data)
        assert result == "{}"  # Default fallback

    def test_safe_json_dumps_with_custom_error_fallback(self):
        """Test dumping with custom error fallback string."""

        class UnserializableClass:
            def __str__(self):
                raise Exception("Cannot convert to string")

        data = {"unserializable": UnserializableClass()}
        result = safe_json_dumps(
            data, on_error_return_str='{"error": "serialization_failed"}'
        )
        assert result == '{"error": "serialization_failed"}'

    def test_safe_json_dumps_with_numpy_types(self):
        """Test dumping data containing NumPy types."""
        data = {
            "numpy_int": np.int32(42),
            "numpy_float": np.float64(3.14),
            "numpy_array": np.array([1, 2, 3]),
        }
        # First convert to JSON-friendly, then dump
        converted_data = convert_to_json_friendly(data)
        result = safe_json_dumps(converted_data)
        parsed_result = json.loads(result)
        assert parsed_result["numpy_int"] == 42
        assert parsed_result["numpy_float"] == 3.14
        assert parsed_result["numpy_array"] == [1, 2, 3]

    def test_safe_json_dumps_logs_error(self):
        """Test that errors are logged when JSON dump fails."""

        class UnserializableClass:
            def __str__(self):
                raise Exception("Cannot convert to string")

        data = {"unserializable": UnserializableClass()}

        with patch("src.utils.core_helpers.json_utils.logger") as mock_logger:
            result = safe_json_dumps(data)
            assert result == "{}"
            # Check that error was logged
            assert mock_logger.error.called
