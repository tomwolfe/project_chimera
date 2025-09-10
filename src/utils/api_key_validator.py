import re
import logging
from typing import Tuple
import google.genai as genai
from google.genai.errors import APIError

logger = logging.getLogger(__name__)

def validate_gemini_api_key_format(api_key: str) -> Tuple[bool, str]:
    """Validate Gemini API key format with multiple security layers."""
    if not api_key or not isinstance(api_key, str):
        return False, "API key is empty or invalid type"
    
    # Format validation (basic regex)
    if not re.match(r"^[A-Za-z0-9_-]{35,}$", api_key):
        return False, "API key format is invalid (expected 35+ alphanumeric/hyphen/underscore characters)"
    
    # Check for common secret patterns that indicate exposure (heuristic)
    if 'AIza' not in api_key and 'AIza' not in api_key[:10]: # Common prefix for Google API keys
        return False, "API key appears to be missing standard prefix (e.g., 'AIza')"
    
    # Check for embedded in code patterns (heuristic)
    if 'github' in api_key.lower() or 'gitlab' in api_key.lower() or 'repo' in api_key.lower():
        return False, "API key appears to contain repository information, indicating potential exposure"
    
    # Check for standard Google API key prefix
    if not api_key.startswith('AIza'):
        return False, "API key is missing the standard 'AIza' prefix for Google API keys."

    return True, "API key format validated"


def test_gemini_api_key_functional(api_key: str) -> Tuple[bool, str]:
    """Test if the Gemini API key is functional by making a minimal API call."""
    try:
        test_client = genai.Client(api_key=api_key)
        test_client.models.list() # A simple call to verify authentication
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