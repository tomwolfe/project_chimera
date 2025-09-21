import logging
from typing import Any

from src.utils.prompt_cache import prompt_cache  # Import the global cache instance

logger = logging.getLogger(__name__)


class PromptGenerator:
    """Generates prompts based on templates and caches frequently used prompts.
    This class is intended to be a more direct interface for generating prompts
    than the PromptOptimizer, which focuses on optimization strategies.
    """

    def __init__(self):
        # In a real scenario, this might load templates or models
        self.templates: dict[str, str] = {
            "problem_analysis": "Analyze the following problem: {problem_description}",
            "solution_refinement": "Refine the proposed solution: {solution} for problem: {problem_description}",
            # Add other common prompt templates here
        }
        self.cache = prompt_cache  # Use the global prompt_cache instance

    def _generate_cache_key(self, prompt_key: str, kwargs: dict[str, Any]) -> str:
        """Generates a consistent cache key for a given prompt and its arguments."""
        # Sort kwargs items to ensure consistent key generation regardless of dict order
        sorted_kwargs = tuple(sorted(kwargs.items()))
        return f"{prompt_key}:{hash(sorted_kwargs)}"

    def generate_prompt(self, prompt_key: str, **kwargs: Any) -> str:
        """Generates a prompt based on a prompt template and given arguments, utilizing a cache.

        Args:
            prompt_key: The key identifying the prompt template to use.
            **kwargs: Arguments to format the template with.

        Returns:
            The generated prompt string.

        """
        cache_key = self._generate_cache_key(prompt_key, kwargs)

        cached_prompt = self.cache.get(cache_key)
        if cached_prompt:
            logger.debug(f"Cache hit for prompt_key: {prompt_key}")
            return cached_prompt

        template = self.templates.get(prompt_key)
        if not template:
            logger.error(f"Error: Prompt template '{prompt_key}' not found.")
            return f"Error: Prompt template '{prompt_key}' not found."

        try:
            generated_prompt = template.format(**kwargs)
            self.cache.set(cache_key, generated_prompt)
            logger.debug(f"Generated and cached prompt for prompt_key: {prompt_key}")
            return generated_prompt
        except KeyError as e:
            logger.error(
                f"Missing key for prompt formatting in template '{prompt_key}': {e}. Template: '{template}'"
            )
            return f"Error: Missing data for prompt template '{prompt_key}': {e}"
        except Exception as e:
            logger.error(
                f"Unexpected error during prompt generation for '{prompt_key}': {e}",
                exc_info=True,
            )
            return f"Error: Failed to generate prompt for '{prompt_key}': {e}"

    def generate_response(self, prompt_key: str, **kwargs: Any) -> str:
        """Simulates generating a response from an LLM based on a prompt template.
        This is a placeholder for actual LLM interaction.
        """
        # In a real system, this would involve calling an LLM client.
        # For now, it just returns the generated prompt.
        return self.generate_prompt(prompt_key, **kwargs)
