"""
Output Validator Module for Output Parser
Separated from output_parser.py to reduce complexity
"""
from typing import Any


class OutputValidator:
    """
    Handles output validation logic that was previously in output_parser.py
    """
    def __init__(self):
        pass

    def validate_output(self, output_data: dict[str, Any]) -> dict[str, Any]:
        """
        Validate the output data
        """
        # Placeholder for complex validation logic that would be moved from output_parser.py
        # This is a simplified version to demonstrate the separation of concerns
        validation_result = {
            "is_valid": True,
            "errors": [],
            "warnings": []
        }

        # Add basic validation checks that were part of the original output_parser.py
        if not output_data:
            validation_result["is_valid"] = False
            validation_result["errors"].append("Output data is empty")

        return validation_result
