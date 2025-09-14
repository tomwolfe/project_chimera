import os
import logging

logger = logging.getLogger(__name__)

API_KEY = os.environ.get("CHIMERA_API_KEY")
if not API_KEY:
    logger.warning("CHIMERA_API_KEY not set in environment variables")
    API_KEY = "dev_api_key_12345"  # Only for development use
