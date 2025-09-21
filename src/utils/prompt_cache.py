import hashlib
from typing import Dict, Any, Optional


class PromptCache:
    def __init__(self):
        self.cache: Dict[str, Any] = {}

    def _get_key(self, prompt: str, model_name: str, temperature: float) -> str:
        key_str = f"{model_name}:{temperature}:{prompt}"
        return hashlib.sha256(key_str.encode()).hexdigest()

    def get(self, prompt: str, model_name: str, temperature: float) -> Optional[Any]:
        return self.cache.get(self._get_key(prompt, model_name, temperature))

    def set(self, prompt: str, model_name: str, temperature: float, response: Any):
        self.cache[self._get_key(prompt, model_name, temperature)] = response


prompt_cache = PromptCache()
