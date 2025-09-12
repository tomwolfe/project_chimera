from src.utils.prompt_generator import generate_analysis_prompt  # Corrected import path
from src.llm.client import LLMClient  # Corrected import path
import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)


class AnalysisService:
    """
    A service responsible for orchestrating LLM-based analysis queries.
    It uses an LLMClient to generate responses and a prompt generator to format inputs.
    """

    def __init__(self, llm_client: LLMClient):
        self.llm_client = llm_client

    def analyze_query(self, user_query: str, context_data: str) -> str:
        """
        Analyzes a user query using the LLM.
        Orchestrates prompt generation, LLM call, and basic response parsing.
        """
        try:
            prompt = generate_analysis_prompt(user_query, context_data)

            # Use the LLM client to generate response
            response = self.llm_client.generate_response(prompt)

            # Basic response parsing (needs to be more robust in a real application)
            if response and "choices" in response and response["choices"]:
                analysis_result = response["choices"][0]["message"]["content"]
                # Log token usage for this interaction
                prompt_tokens = self.llm_client.count_tokens(prompt)
                response_tokens = self.llm_client.count_tokens(analysis_result)
                logger.info(
                    "LLM analysis completed.",
                    extra={
                        "extra": {
                            "prompt_tokens": prompt_tokens,
                            "response_tokens": response_tokens,
                            "total_tokens": prompt_tokens + response_tokens,
                        }
                    },
                )
                return analysis_result
            else:
                logger.error(
                    "LLM returned an unexpected response format.",
                    extra={"extra": {"llm_response": response}},
                )
                return "Error: Could not process LLM response."

        except Exception as e:
            logger.error(f"Error during LLM analysis: {e}", exc_info=True)
            # Depending on requirements, re-raise or return a specific error message
            return "An error occurred during analysis."
