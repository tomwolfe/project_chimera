from typing import Dict, Optional
from pydantic import BaseModel


class ModelSpecification(BaseModel):
    name: str
    max_input_tokens: int
    max_output_tokens: int
    cost_per_1k_input: float
    cost_per_1k_output: float
    capabilities: list[str]
    is_available: bool = True


class ModelRegistry:
    def __init__(self):
        self.models: Dict[str, ModelSpecification] = {}

        # Default model configurations
        self.register_model(
            ModelSpecification(
                name="gemini-2.5-flash-lite",
                max_input_tokens=1048576,
                max_output_tokens=8192,  # Adjusted to match common usage, was 65538 in tokenizer
                cost_per_1k_input=0.075,
                cost_per_1k_output=0.30,
                capabilities=["reasoning", "coding"],
            )
        )

        self.register_model(
            ModelSpecification(
                name="gemini-2.5-pro",
                max_input_tokens=1048576,
                max_output_tokens=8192,  # Adjusted to match common usage, was 65536 in tokenizer
                cost_per_1k_input=0.25,
                cost_per_1k_output=1.00,
                capabilities=["reasoning", "coding", "complex_analysis"],
            )
        )

    def register_model(self, model: ModelSpecification):
        self.models[model.name] = model

    def get_model(
        self,
        requirements: list[str] = None,
        budget: float = None,
        preferred_model_name: Optional[str] = None,
    ) -> Optional[ModelSpecification]:
        """Get the best available model meeting requirements within budget, or a preferred model."""
        if (
            preferred_model_name
            and preferred_model_name in self.models
            and self.models[preferred_model_name].is_available
        ):
            return self.models[preferred_model_name]

        candidates = []

        for model in self.models.values():
            if not model.is_available:
                continue

            # Check capability requirements
            if requirements and not all(
                req in model.capabilities for req in requirements
            ):
                continue

            # Check budget constraints (simplified for now, actual cost depends on tokens)
            # This budget check is more about cost-per-1k-tokens, not total budget
            if budget and (
                model.cost_per_1k_input > budget or model.cost_per_1k_output > budget
            ):
                continue

            candidates.append(model)

        # Sort by cost-effectiveness (cheapest first)
        candidates.sort(key=lambda m: m.cost_per_1k_input)
        return candidates[0] if candidates else None

    def mark_model_unavailable(self, model_name: str):
        if model_name in self.models:
            self.models[model_name].is_available = False
