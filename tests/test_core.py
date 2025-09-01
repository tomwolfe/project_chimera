# tests/test_core.py
import pytest

# Assuming core.py contains process_complex_logic and analyze_data
# from src.core import process_complex_logic, analyze_data

# Mocking functions from core.py for demonstration
def process_complex_logic(item):
    # Simplified mock implementation for testing
    result = 0
    if item.get('condition1') and item.get('value', 0) > 10:
        result += 5
    if item.get('condition2') or item.get('flag'):
        result += 10
    if 'nested' in item and item['nested'].get('deep_condition'):
        result *= 2
    return result

def analyze_data(data):
    # Simplified mock implementation for testing
    processed_results = []
    for item in data:
        processed_results.append(process_complex_logic(item))
    return processed_results


# Test cases for process_complex_logic
def test_process_complex_logic_base_case():
    """Tests the base case with no conditions met."""
    item = {'value': 5}
    assert process_complex_logic(item) == 0

def test_process_complex_logic_condition1():
    """Tests when only condition1 is met."""
    item = {'condition1': True, 'value': 15}
    assert process_complex_logic(item) == 5

def test_process_complex_logic_condition2():
    """Tests when only condition2 is met."""
    item = {'condition2': True, 'value': 5}
    assert process_complex_logic(item) == 10

def test_process_complex_logic_nested_condition():
    """Tests when the nested condition is met."""
    item = {
        'condition1': True,
        'value': 15,
        'nested': {'deep_condition': True, 'deep_value': 100}
    }
    expected_output = "Processed item with nested condition met."
    assert process_complex_logic(item) == expected_output

def test_process_complex_logic_no_condition():
    """Tests when no conditions are met."""