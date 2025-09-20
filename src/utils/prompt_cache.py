import logging
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


class PromptCache:
    """A simple in-memory cache for storing and retrieving generated prompts.
    Aims to reduce redundant computation and token usage for frequently
    accessed or similar prompts.
    """

    def __init__(self):
        self.cache: Dict[str, Any] = {}
        logger.info("PromptCache initialized.")

    def get(self, key: str) -> Optional[Any]:
        """Retrieves an item from the cache.

        Args:
            key: The unique identifier for the cached item.

        Returns:
            The cached item if found, otherwise None.

        """
        item = self.cache.get(key)
        if item:
            logger.debug(f"Cache hit for key: {key}")
        else:
            logger.debug(f"Cache miss for key: {key}")
        return item

    def set(self, key: str, value: Any):
        """Adds or updates an item in the cache.

        Args:
            key: The unique identifier for the item.
            value: The item to be cached.

        """
        self.cache[key] = value
        logger.debug(f"Cached item for key: {key}")

    def has(self, key: str) -> bool:
        """Checks if a key exists in the cache.

        Args:
            key: The unique identifier to check.

        Returns:
            True if the key exists, False otherwise.

        """
        return key in self.cache

    def clear(self):
        """Clears all items from the cache."""
        self.cache.clear()
        logger.info("PromptCache cleared.")


# Instantiate a global cache object for easy access
# In a larger application, consider dependency injection frameworks
prompt_cache = PromptCache()
