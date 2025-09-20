# src/config/settings.py
"""Configuration settings for Project Chimera, including token budgeting and retry parameters."""

import logging
from pathlib import Path
from typing import Dict, List, Self

import yaml
from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

logger = logging.getLogger(__name__)


class ChimeraSettings(BaseSettings):
    """Configuration settings for Project Chimera, including token budgeting and retry parameters."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",  # Ignore extra fields in .env or config.yaml not defined in the model
    )

    # NEW: API Key setting, loaded from env_file or environment variables
    # MODIFIED: Make GEMINI_API_KEY optional with an empty string default
    GEMINI_API_KEY: str = Field(
        "", description="Google Gemini API Key"
    )  # <--- CHANGE THIS LINE

    max_retries: int = Field(
        default=5,
        ge=1,
        le=10,
        description="Maximum number of retries for LLM API calls.",
    )
    max_backoff_seconds: int = Field(
        default=30,
        ge=5,
        le=120,
        description="Maximum backoff time in seconds for retries.",
    )

    # Ratios for allocating the total token budget across different phases.
    # These are normalized to sum to 1.0 in the model_validator.
    context_token_budget_ratio: float = Field(
        default=0.2,
        ge=0.05,
        le=0.5,
        description="Proportion of total budget for context analysis.",
    )
    debate_token_budget_ratio: float = Field(
        default=0.7,
        ge=0.5,
        le=0.95,
        description="Proportion of total budget for debate turns.",
    )
    synthesis_token_budget_ratio: float = Field(
        default=0.1,
        ge=0.05,
        le=0.2,
        description="Proportion of total budget for final synthesis.",
    )

    # Specific ratios for self-analysis prompts, also normalized.
    self_analysis_context_ratio: float = Field(
        default=0.45,
        ge=0.1,
        le=0.6,
        description="Proportion of total budget for context analysis during self-analysis.",
    )
    self_analysis_debate_ratio: float = Field(
        default=0.30,
        ge=0.1,
        le=0.9,
        description="Proportion of total budget for debate turns during self-analysis.",
    )
    self_analysis_synthesis_ratio: float = Field(
        default=0.25,
        ge=0.05,
        le=0.3,
        description="Proportion of total budget for synthesis during self-analysis.",
    )

    # Total token budget for a single Socratic debate run.
    total_budget: int = Field(
        default=1000000,
        ge=1000,
        le=2000000,
        description="Maximum total tokens allowed for a single Socratic debate run.",
    )

    # NEW: Persona-specific maximum input token limits for prompt optimization
    # These values are heuristics and can be tuned based on observed persona verbosity
    default_max_input_tokens_per_persona: int = Field(
        default=4000,
        ge=500,
        le=64000,
        description="Default maximum input tokens for a persona's prompt if not specified.",
    )
    max_tokens_per_persona: Dict[str, int] = Field(
        default_factory=dict,
        description="Specific maximum input tokens for individual personas.",
    )

    # NEW: Domain keywords for prompt analysis, loaded from config.yaml
    domain_keywords: Dict[str, List[str]] = Field(
        default_factory=dict,
        description="Keywords for classifying prompts into domains.",
    )

    # NEW: Sentence Transformer cache directory
    sentence_transformer_cache_dir: str = Field(
        default=str(Path.home() / ".cache" / "huggingface" / "transformers"),
        description="Directory for caching SentenceTransformer models.",
    )

    # ADDED: Global token consumption and efficiency thresholds for aggressive optimization
    GLOBAL_TOKEN_CONSUMPTION_THRESHOLD: float = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        description="Threshold for global token consumption rate to trigger aggressive persona optimization.",
    )
    TOKEN_EFFICIENCY_SCORE_THRESHOLD: float = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        description="Threshold for a persona's token efficiency score to trigger aggressive optimization.",
    )

    @model_validator(mode="after")
    def normalize_token_budget_ratios(self) -> Self:
        """Normalizes the token budget ratios to ensure they sum to 1.0,
        maintaining their relative proportions. This is crucial for
        correctly allocating the total budget across phases.
        """
        # Normalize general ratios (context, debate, synthesis)
        total_general_ratio = (
            self.context_token_budget_ratio
            + self.debate_token_budget_ratio
            + self.synthesis_token_budget_ratio
        )

        # Apply normalization factors
        if total_general_ratio > 0:
            ratio_factor = 1.0 / total_general_ratio
            self.context_token_budget_ratio *= ratio_factor
            self.debate_token_budget_ratio *= ratio_factor
            self.synthesis_token_budget_ratio *= ratio_factor
        else:
            # Fallback if ratios are zero or invalid
            self.context_token_budget_ratio = 0.2
            self.debate_token_budget_ratio = 0.7
            self.synthesis_token_budget_ratio = 0.1

        # Normalize self-analysis ratios (context, debate, synthesis)
        self_analysis_total = (
            self.self_analysis_context_ratio
            + self.self_analysis_debate_ratio
            + self.self_analysis_synthesis_ratio
        )

        if self_analysis_total > 0:
            self_analysis_ratio_factor = 1.0 / self_analysis_total
            self.self_analysis_context_ratio *= self_analysis_ratio_factor
            self.self_analysis_debate_ratio *= self_analysis_ratio_factor
            self.self_analysis_synthesis_ratio *= self_analysis_ratio_factor
        else:
            # Fallback for self-analysis ratios (ensure they sum to 1.0)
            self.self_analysis_context_ratio = 0.45
            self.self_analysis_debate_ratio = 0.30
            self.self_analysis_synthesis_ratio = 0.25

        return self

    @classmethod
    def from_yaml(cls, file_path: str) -> "ChimeraSettings":
        """Loads settings from a YAML file."""
        try:
            config_path = Path(file_path)
            with open(config_path) as f:
                config_data = yaml.safe_load(f)
            # BaseSettings will automatically load from env_file and env vars,
            # then merge with provided config_data.
            return cls(**config_data)
        except (FileNotFoundError, TypeError, yaml.YAMLError) as e:
            logger.error(
                f"Failed to load settings from {file_path}: {e}. Returning default settings.",
                exc_info=True,
            )
            return cls()
