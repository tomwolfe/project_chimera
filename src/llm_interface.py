import openai
from src.utils.prompt_optimizer import PromptOptimizer

class LLMInterface:
    def __init__(self, api_key):
        openai.api_key = api_key
        self.model_name = 'gpt-3.5-turbo' # Default model

        # Initialize prompt optimizer (will be set by SocraticDebate)
        self.prompt_optimizer: Optional[PromptOptimizer] = None

    def generate_response(self, messages: List[Dict[str, str]], system_prompt: str = "") -> str:
        """Generates a response from the LLM based on the provided messages.

        Args:
            messages (list): A list of message dictionaries representing the conversation history.
            system_prompt (str): The system prompt to prepend to the conversation.

        Returns:
            str: The generated response content.
        """
        # Prepare messages for the LLM API call
        # The system_prompt is now handled by the PromptOptimizer if it's part of the conversation history.
        # If system_prompt is a separate argument, it needs to be prepended to the optimized messages.
        
        # If prompt_optimizer is available, use it. Otherwise, use raw messages.
        if self.prompt_optimizer:
            # The optimize_prompt method now returns a list of message dicts.
            optimized_messages = self.prompt_optimizer.optimize_prompt(
                conversation_history=messages,
                persona_name="LLMInterface_Caller", # A generic name for this context
                max_output_tokens_for_turn=self.model_name # Pass model_name for context
            )
        else:
            # Fallback if optimizer is not initialized (should not happen in normal flow)
            optimized_messages = []
            if system_prompt:
                optimized_messages.append({"role": "system", "content": system_prompt})
            optimized_messages.extend(messages)

        kwargs = {
            "model": self.model_name,
            "messages": optimized_messages,
            "temperature": 0.7, # Default temperature, can be overridden
            "max_tokens": 100 # Default max_tokens, can be overridden
        }

        try:
            response = openai.ChatCompletion.create(**kwargs)
        except openai.error.RateLimitError as e:
            print(f"Rate limit exceeded: {e}")
            # Implement retry logic or backoff strategy here
            raise e
        except openai.error.APIConnectionError as e:
            print(f"API connection error: {e}")
            # Implement retry logic or backoff strategy here
            raise e
        except Exception as e:
            print(f"An unexpected error occurred during LLM API call: {e}")
            raise e

        return response.choices[0].message['content']
