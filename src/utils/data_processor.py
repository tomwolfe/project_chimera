# src/utils/data_processor.py
from typing import List, Union

def process_numbers(numbers: List[Union[float, int]]) -> Union[float, int]:
    """
    Processes a list of numbers by calculating their sum.
    """
    return sum(numbers)

def format_string(text: str) -> str:
    """Formats a given string by stripping whitespace and converting to uppercase."""
    return text.strip().upper()
