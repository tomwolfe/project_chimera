# scripts/generate_token_usage_report.py
import json
import os
import sys
from pathlib import Path

# Add src to Python path for imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.token_tracker import TokenUsageTracker


def generate_report():
    # This script is meant to be run after a pytest run that might have used TokenUsageTracker.
    # However, the TokenUsageTracker is usually an in-memory object tied to a running application.
    # For a CI/CD context, we would need to persist the token usage data from the pytest run.
    # For simplicity, this mock will just output a dummy report.

    # In a real scenario, pytest would need to be configured to save token usage data
    # (e.g., to a JSON file) and this script would read that file.

    # For now, let's simulate a report.
    report_data = {
        "report_date": "2024-01-01T12:00:00Z",
        "total_tokens_simulated": 12345,
        "estimated_cost_usd_simulated": 0.0123,
        "details": "This is a simulated token usage report. Actual data would be collected from test runs.",
        "warnings": [],
    }

    # Check for a dummy file that pytest might have created
    dummy_token_log_path = Path("token_usage_log.json")
    if dummy_token_log_path.exists():
        try:
            with open(dummy_token_log_path, "r") as f:
                logged_data = json.load(f)
            report_data["total_tokens_simulated"] = logged_data.get(
                "total_tokens", report_data["total_tokens_simulated"]
            )
            report_data["estimated_cost_usd_simulated"] = logged_data.get(
                "total_cost_usd", report_data["estimated_cost_usd_simulated"]
            )
            report_data["details"] = "Token usage data loaded from dummy log file."
            os.remove(dummy_token_log_path)  # Clean up
        except Exception as e:
            report_data["warnings"].append(f"Failed to read dummy token log: {e}")

    print(json.dumps(report_data, indent=2))


if __name__ == "__main__":
    generate_report()
