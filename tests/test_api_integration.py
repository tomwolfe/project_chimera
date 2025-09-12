=== tests/test_api_integration.py ===
# tests/test_api_integration.py
import unittest
import requests
import os
import time
import pytest # NEW: Import pytest for skipping tests

# This test requires the FastAPI server (part of app.py) to be running.
# You can run it separately, for example, using `uvicorn app:app --host 0.0.0.0 --port 8080`
# or by ensuring the Streamlit app is running and its internal FastAPI is active.
# For a dedicated integration test, it's best to run the FastAPI component directly.


class TestAPIIntegration(unittest.TestCase):
    # Use an environment variable for the base URL for flexibility
    BASE_URL = os.environ.get("API_BASE_URL", "http://localhost:8080/api/v1")

    @classmethod
    def setUpClass(cls):
        """
        Optional: Add logic here to start the FastAPI server if it's not expected to be running.
        For simplicity, this example assumes the server is already running.
        If the server is not running, these tests will fail with connection errors.
        """
        print(f"\n--- Starting API Integration Tests ---")
        print(f"Attempting to connect to API at: {cls.BASE_URL}")
        # Basic check to see if the server is reachable
        try:
            response = requests.get(f"{cls.BASE_URL}/status", timeout=5)
            if response.status_code == 200:
                print("API server is reachable.")
            else:
                print(
                    f"API server returned status {response.status_code} on /status endpoint."
                )
        except requests.exceptions.ConnectionError:
            pytest.skip(
                f"API server not reachable at {cls.BASE_URL}. Skipping integration tests."
            )
        except Exception as e:
            pytest.fail(
                f"An unexpected error occurred during API reachability check: {e}"
            )

    def test_status_endpoint(self):
        """Test the /api/v1/status endpoint."""
        try:
            response = requests.get(f"{self.BASE_URL}/status", timeout=5)  # Add timeout
            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.json(), {"status": "operational"})
        except requests.exceptions.ConnectionError as e:
            self.fail(
                f"Could not connect to API server at {self.BASE_URL}. Is it running? Error: {e}"
            )
        except Exception as e:
            self.fail(
                f"An unexpected error occurred while testing /status endpoint: {e}"
            )

    def test_analyze_code_endpoint_success(self):
        """Test the /api/v1/analyze_code endpoint with valid input."""
        # This endpoint requires a valid API key and a prompt.
        # For a real integration test, you'd need to mock or provide a real API key.
        # For now, we'll use a placeholder and expect a 400 or 500 if the key is missing/invalid.
        # A successful 200 would require a full LLM call.
        payload = {
            "prompt": "Analyze the following Python code for potential improvements: def func(): pass",
            "api_key": os.environ.get(
                "TEST_LLM_API_KEY", "sk-test-api-key-placeholder"
            ),
            "model_name": "gemini-2.5-flash-lite",
            "domain": "Software Engineering",
            "codebase_context": {"test_file.py": "def example_func():\n    pass"},
            "max_tokens_budget": 10000,
        }
        try:
            response = requests.post(
                f"{self.BASE_URL}/analyze_code", json=payload, timeout=10
            )  # Add timeout
            # Expect 200 for success, or 400/401/429 if API key is missing/invalid or rate limited
            self.assertIn(
                response.status_code,
                [200, 400, 401, 429],
                f"Unexpected status code: {response.status_code}, Response: {response.text}",
            )
            if response.status_code == 200:
                data = response.json()
                self.assertIn("message", data)
                self.assertEqual(data["message"], "Analysis complete")
            else:
                print(
                    f"Skipping full analysis check due to API error: {response.status_code} - {response.text}"
                )
        except requests.exceptions.ConnectionError as e:
            self.fail(
                f"Could not connect to API server at {self.BASE_URL}. Is it running? Error: {e}"
            )
        except Exception as e:
            self.fail(
                f"An unexpected error occurred while testing /analyze_code endpoint: {e}"
            )

    def test_analyze_code_endpoint_invalid_input(self):
        """Test the /api/v1/analyze_code endpoint with invalid input (missing prompt)."""
        payload = {
            "api_key": "sk-test-api-key-placeholder",
            "model_name": "gemini-2.5-flash-lite",
            "domain": "Software Engineering",
            "codebase_context": {},
            "max_tokens_budget": 10000,
        }
        try:
            response = requests.post(
                f"{self.BASE_URL}/analyze_code", json=payload, timeout=10
            )  # Add timeout
            self.assertEqual(response.status_code, 422)  # FastAPI validation error
            data = response.json()
            self.assertIn("detail", data)
            self.assertIn("field required", str(data["detail"]))
        except requests.exceptions.ConnectionError as e:
            self.fail(
                f"Could not connect to API server at {self.BASE_URL}. Is it running? Error: {e}"
            )
        except Exception as e:
            self.fail(
                f"An unexpected error occurred while testing /analyze_code endpoint: {e}"
            )


if __name__ == "__main__":
    unittest.main()