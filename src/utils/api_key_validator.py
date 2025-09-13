# src/utils/api_key_validator.py
import re
import logging
import os
from typing import Tuple, Optional
import google.genai as genai
from google.genai.errors import APIError

logger = logging.getLogger(__name__)


# Placeholder for a secrets manager client (e.g., AWS Secrets Manager, Google Secret Manager)
# In a real application, this would be configured to interact with a specific service.
class MockSecretsManager:
    def get_secret(self, secret_name: str) -> Optional[str]:
        # Simulate fetching a secret. In production, this would call a real secrets manager API.
        if secret_name == "GEMINI_API_KEY_SECRET":
            return os.getenv(
                "GEMINI_API_KEY_FROM_SECRETS_MANAGER"
            )  # For testing, can use another env var
        return None


_secrets_manager_client = MockSecretsManager()  # Initialize a mock/real client


def fetch_api_key() -> Optional[str]:
    """
    Fetches the Gemini API key, prioritizing a secrets manager if configured,
    then falling back to environment variables.
    """
    # 1. First check environment variables
    env_api_key = os.environ.get("GEMINI_API_KEY")
    if env_api_key:
        logger.info("API key fetched from environment variable.")
        return env_api_key

    # 2. Check Streamlit secrets if running in Streamlit
    try:
        import streamlit as st
        if "GEMINI_API_KEY" in st.secrets:
            logger.info("API key fetched from Streamlit secrets.")
            return st.secrets["GEMINI_API_KEY"]
    except ImportError:
        pass # Not running in Streamlit or streamlit not installed

    # 3. Fallback to a secrets manager (if configured)
    secrets_manager_key = _secrets_manager_client.get_secret("GEMINI_API_KEY_SECRET")
    if secrets_manager_key:
        logger.info("API key fetched from secrets manager.")
        return secrets_manager_key

    logger.warning("No Gemini API key found in environment variables, Streamlit secrets, or secrets manager.")
    return None


def validate_gemini_api_key_format(api_key: str) -> Tuple[bool, str]:
    """Validate Gemini API key format with multiple security layers."""
    if not api_key or not isinstance(api_key, str):
        return False, "API key is empty or invalid type"

    # Check length requirement
    if len(api_key) < 35:
        return False, "API key must be at least 35 characters long"

    # Check character set
    if not all(c.isalnum() or c in "-_" for c in api_key):
        return (
            False,
            "API key must contain only alphanumeric characters, hyphens, or underscores",
        )

    # The original regex check `if not re.match(r"^[A-Za-z0-9_-]{35,}$", api_key):`
    # is now redundant with the explicit length and character set checks above, so it is removed.

    # Check for common secret patterns that indicate exposure (heuristic)
    if (
        "AIza" not in api_key and "AIza" not in api_key[:10]
    ):  # Common prefix for Google API keys
        return False, "API key appears to be missing standard prefix (e.g., 'AIza')"

    # Check for embedded in code patterns (heuristic)
    if (
        "github" in api_key.lower()
        or "gitlab" in api_key.lower()
        or "repo" in api_key.lower()
    ):
        return (
            False,
            "API key appears to contain repository information, indicating potential exposure",
        )

    # Check for standard Google API key prefix
    if not api_key.startswith("AIza"):
        return (
            False,
            "API key is missing the standard 'AIza' prefix for Google API keys.",
        )

    return True, "API key format validated"


def test_gemini_api_key_functional(api_key: str) -> Tuple[bool, str]:
    """Test if the Gemini API key is functional by making a minimal API call."""
    try:
        test_client = genai.Client(api_key=api_key)
        test_client.models.list()  # A simple call to verify authentication with a timeout
        return True, "API key is valid and functional"
    except APIError as e:
        logger.error(f"API key functional test failed: {e}")
        if e.code == 401:
            return False, "Invalid API key - access denied"
        elif e.code == 403:
            return False, "API key valid but lacks required permissions"
        else:
            return False, f"API error: {e.message}"
    except Exception as e:
        logger.error(f"Unexpected error during API key functional test: {e}")
        if "connection" in str(e).lower() or "timeout" in str(e).lower():
            return False, "Network connection issue - check your internet connection"
        else:
            return False, f"Unexpected error during API validation: {e}"