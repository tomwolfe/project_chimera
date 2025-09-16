# src/persona_manager.py
import os
import json
import yaml
import datetime
import re
import logging
from typing import Dict, Any, List, Optional, Tuple, Union, Type
from pydantic import ValidationError, BaseModel
import copy
import time

from src.persona.routing import PersonaRouter
from src.models import (
    PersonaConfig,
    ReasoningFrameworkConfig,
    LLMOutput,
    CritiqueOutput,
    GeneralOutput,
    ConflictReport,
    SelfImprovementAnalysisOutputV1,
    ContextAnalysisOutput,
    ConfigurationAnalysisOutput,
    DeploymentAnalysisOutput,
)
from src.config.persistence import ConfigPersistence
from src.utils.prompt_analyzer import PromptAnalyzer
from src.token_tracker import TokenUsageTracker
from src.exceptions import SchemaValidationError
from src.config.settings import ChimeraSettings  # NEW: Import ChimeraSettings
from src.utils.prompt_optimizer import PromptOptimizer  # NEW: Import PromptOptimizer

logger = logging.getLogger(__name__)

DEFAULT_PERSONAS_FILE = "personas.yaml"


class PersonaManager:
    # Define thresholds for triggering truncation as class constants
    TRUNCATION_FAILURE_RATE_THRESHOLD = (
        0.2  # If a persona fails truncation in >20% of its turns
    )
    GLOBAL_TOKEN_CONSUMPTION_THRESHOLD = (
        0.7  # If overall token usage exceeds 70% of budget
    )

    # NEW: Moved PERSONA_OUTPUT_SCHEMAS here from core.py
    PERSONA_OUTPUT_SCHEMAS: Dict[str, Type[BaseModel]] = {
        "Impartial_Arbitrator": LLMOutput,
        "Context_Aware_Assistant": ContextAnalysisOutput,
        "Constructive_Critic": CritiqueOutput,
        "General_Synthesizer": GeneralOutput,
        "Devils_Advocate": ConflictReport,
        "Self_Improvement_Analyst": SelfImprovementAnalysisOutputV1,
        "Code_Architect": CritiqueOutput,
        "Security_Auditor": CritiqueOutput,
        "DevOps_Engineer": CritiqueOutput,
        "Test_Engineer": CritiqueOutput,
        # Add _TRUNCATED versions to map to their base schemas
        "Code_Architect_TRUNCATED": CritiqueOutput,
        "Security_Auditor_TRUNCATED": CritiqueOutput,
        "DevOps_Engineer_TRUNCATED": CritiqueOutput,
        "Test_Engineer_TRUNCATED": CritiqueOutput,
        "Constructive_Critic_TRUNCATED": CritiqueOutput,
        "Impartial_Arbitrator_TRUNCATED": LLMOutput,
        "Devils_Advocate_TRUNCATED": ConflictReport,
        "General_Synthesizer_TRUNCATED": GeneralOutput,
        "Self_Improvement_Analyst_TRUNCATED": SelfImprovementAnalysisOutputV1,
    }

    def __init__(
        self,
        domain_keywords: Dict[str, List[str]],
        token_tracker: Optional[TokenUsageTracker] = None,
        settings: Optional[ChimeraSettings] = None,  # NEW: Add settings parameter
        prompt_optimizer: Optional[
            PromptOptimizer
        ] = None,  # NEW: Add prompt_optimizer parameter
    ):
        self.all_personas: Dict[str, PersonaConfig] = {}
        self.persona_sets: Dict[str, List[str]] = {}
        self.available_domains: List[str] = []
        self.all_custom_frameworks_data: Dict[str, Any] = {}
        self.default_persona_set_name: str = "General"
        self._original_personas: Dict[str, PersonaConfig] = {}  # Store original configs

        # For Adaptive LLM Parameter Adjustment
        self.persona_performance_metrics: Dict[str, Dict[str, Any]] = {}
        self.adjustment_cooldown_seconds = 300  # 5 minutes cooldown
        self.min_turns_for_adjustment = 5  # Minimum turns before considering adjustment

        self.config_persistence = ConfigPersistence()
        self.settings = settings or ChimeraSettings()  # NEW: Store settings

        # Initialize PromptAnalyzer first
        self.prompt_analyzer = PromptAnalyzer(domain_keywords)
        self.prompt_optimizer = prompt_optimizer  # NEW: Store prompt_optimizer

        # Load initial data and custom frameworks, handle errors internally
        load_success, load_msg = self._load_initial_data()
        if not load_success and load_msg:
            logger.error(f"Failed to load initial personas: {load_msg}")
            # In a real app, you might want to raise an exception here or have a more robust fallback.
            # For now, we'll proceed with potentially empty or minimal data, logging the error.

        self._load_custom_frameworks_on_init()
        self._load_original_personas()  # Load original personas after all others are loaded

        # Initialize PersonaRouter with all loaded personas and persona_sets, and the prompt_analyzer
        # This ensures the router always has the correct prompt_analyzer instance.
        self.persona_router: Optional[PersonaRouter] = PersonaRouter(
            self.all_personas,
            self.persona_sets,
            self.prompt_analyzer,
            persona_manager=self,  # Pass self
        )

        # Initialize performance metrics after all personas are loaded
        self._initialize_performance_metrics()

        self.token_tracker = token_tracker  # Store token_tracker

    def _load_initial_data(
        self, file_path: str = DEFAULT_PERSONAS_FILE
    ) -> Tuple[bool, Optional[str]]:
        """Loads the default personas and persona sets from a YAML file.
        Returns:
            Tuple[bool, Optional[str]]: (success_status, error_message_or_None)
        """
        try:
            with open(file_path, "r") as f:
                data = yaml.safe_load(f)

            if not data:  # Handle empty file case
                raise ValueError("Persona configuration file is empty.")

            all_personas_list = [
                PersonaConfig(**p_data) for p_data in data.get("personas", [])
            ]
            self.all_personas = {p.name: p for p in all_personas_list}
            self.persona_sets = data.get("persona_sets", {"General": []})

            # Validate persona sets references
            for set_name, persona_names_in_set in self.persona_sets.items():
                if not isinstance(persona_names_in_set, list):
                    raise ValueError(
                        f"Persona set '{set_name}' must be a list of persona names."
                    )
                for p_name in persona_names_in_set:
                    if p_name not in self.all_personas:
                        raise ValueError(
                            f"Persona '{p_name}' referenced in set '{set_name}' not found in 'personas' list."
                        )

            self.default_persona_set_name = (
                "General"
                if "General" in self.persona_sets
                else next(iter(self.persona_sets.keys()), "General")
            )
            self.available_domains = list(self.persona_sets.keys())
            logger.info(f"Initial personas loaded successfully from {file_path}.")
            return True, None

        except (FileNotFoundError, ValidationError, yaml.YAMLError, ValueError) as e:
            logger.error(f"Error loading initial personas from {file_path}: {e}")
            # Fallback to minimal personas if loading fails
            self.all_personas = {
                "Visionary_Generator": PersonaConfig(
                    name="Visionary_Generator",
                    system_prompt="You are a visionary.",
                    temperature=0.7,
                    max_tokens=1024,
                    description="Generates innovative solutions.",
                ),
                "Skeptical_Generator": PersonaConfig(
                    name="Skeptical_Generator",
                    system_prompt="You are a skeptic.",
                    temperature=0.3,
                    max_tokens=1024,
                    description="Identifies flaws.",
                ),
                "Impartial_Arbitrator": PersonaConfig(
                    name="Impartial_Arbitrator",
                    system_prompt="You are an arbitrator.",
                    temperature=0.2,  # MODIFIED: Changed temperature to 0.2
                    max_tokens=1024,
                    description="Synthesizes outcomes.",
                ),
            }
            self.persona_sets = {
                "General": [
                    "Visionary_Generator",
                    "Skeptical_Generator",
                    "Impartial_Arbitrator",
                ]
            }
            self.default_persona_set_name = "General"
            self.available_domains = ["General"]
            logger.warning(
                "Loaded minimal fallback personas due to initial loading error."
            )
            return False, f"Failed to load default personas from {file_path}: {e}"

    def _load_custom_frameworks_on_init(self):
        """Loads custom frameworks available at startup using ConfigPersistence."""
        saved_names = self.config_persistence._get_saved_custom_framework_names()
        for name in saved_names:
            config = self.config_persistence._load_custom_framework_config_from_file(
                name
            )
            if config:
                # Add personas from custom framework to all_personas
                for p_name, p_data in config.get("personas", {}).items():
                    try:
                        self.all_personas[p_name] = PersonaConfig(**p_data)
                    except ValidationError as e:
                        logger.error(
                            f"Validation error for persona '{p_name}' in custom framework '{name}': {e}"
                        )
                        continue
                # Add persona sets from custom framework to persona_sets
                self.persona_sets.update(config.get("persona_sets", {}))

                # Store the full config data for later reference (e.g., versioning)
                self.all_custom_frameworks_data[name] = config

                # Add framework name to available domains
                if name not in self.available_domains:
                    self.available_domains.append(name)
                self.available_domains = sorted(list(set(self.available_domains)))
        logger.info(f"Loaded {len(saved_names)} custom frameworks.")

    def _load_original_personas(self) -> Tuple[bool, Optional[str]]:
        """Loads the original default personas from the YAML file and stores them."""
        try:
            # Load base configuration (assuming personas.yaml is the source of original personas)
            with open(DEFAULT_PERSONAS_FILE, "r") as f:
                data = yaml.safe_load(f)

            if not data:
                raise ValueError("Original personas file is empty.")

            original_personas_list = [
                PersonaConfig(**p_data) for p_data in data.get("personas", [])
            ]
            self._original_personas = {p.name: p for p in original_personas_list}
            logger.info("Original personas loaded and stored.")
            return True, None

        except (FileNotFoundError, ValidationError, yaml.YAMLError, ValueError) as e:
            logger.error(
                f"Error loading original personas from {DEFAULT_PERSONAS_FILE}: {e}"
            )
            return False, f"Failed to load original personas: {e}"

    # NEW: For Adaptive LLM Parameter Adjustment
    def _initialize_performance_metrics(self):
        """Initializes performance metrics for all loaded personas."""
        for p_name in self.all_personas.keys():
            self.persona_performance_metrics[p_name] = {
                "total_turns": 0,
                "schema_failures": 0,
                "truncation_failures": 0,  # NEW: Track truncation failures
                "last_adjusted_temp": self.all_personas[p_name].temperature,
                "last_adjusted_max_tokens": self.all_personas[p_name].max_tokens,
                "last_adjustment_timestamp": 0.0,
            }

    def save_framework(
        self,
        name: str,
        current_persona_set_name: str,
        current_active_personas: Dict[str, PersonaConfig],
        description: str = "",
    ) -> Tuple[bool, str]:
        """Saves the current framework configuration (including persona edits) as a custom framework.
        Returns:
            Tuple[bool, str]: (success_status, message)
        """
        if not name:
            return False, "Please enter a name for the framework before saving."

        # Use the original name for internal dict key and available domains
        # Sanitization is handled by ConfigPersistence for filename

        current_personas_dict = {
            p_name: p_data.model_dump()
            for p_name, p_data in current_active_personas.items()
        }

        version = 1
        if (
            name in self.all_custom_frameworks_data
        ):  # Check original name for versioning
            version = self.all_custom_frameworks_data[name].get("version", 0) + 1

        try:
            config_to_save = {
                "framework_name": name,
                "description": description,  # ADDED DESCRIPTION
                "personas": current_personas_dict,
                "persona_sets": {
                    name: list(current_active_personas.keys())
                },  # Use original name for persona set key
                "version": version,
            }
            # Validate the structure before saving
            ReasoningFrameworkConfig(
                framework_name=config_to_save["framework_name"],
                personas={
                    p_name: PersonaConfig(**p_data)
                    for p_name, p_data in config_to_save["personas"].items()
                },
                persona_sets=config_to_save["persona_sets"],
                version=config_to_save["version"],
            )
        except Exception as e:
            logger.error(f"Validation error for framework '{name}' before saving: {e}")
            return False, f"Cannot save framework: Invalid data structure. {e}"

        success, message = self.config_persistence.save_user_framework(
            name, config_to_save
        )
        if success:
            self.all_custom_frameworks_data[name] = (
                config_to_save  # Store with original name
            )
            if name not in self.available_domains:
                self.available_domains.append(name)
            self.available_domains = sorted(list(set(self.available_domains)))
            return True, message
        else:
            return False, message

    def load_framework_into_session(
        self, framework_name: str
    ) -> Tuple[bool, str, Dict[str, PersonaConfig], Dict[str, List[str]], str]:
        """Loads a framework's personas and sets, returning them for session state update.
        Returns:
            Tuple[bool, str, Dict[str, PersonaConfig], Dict[str, List[str]], str]:
            (success_status, message, loaded_personas, loaded_persona_sets, new_framework_name)
        """
        loaded_config_data = (
            self.config_persistence._load_custom_framework_config_from_file(
                framework_name
            )
        )

        if loaded_config_data:  # It's a custom framework
            # Update all_personas with personas from the custom framework
            for name, data in loaded_config_data.get("personas", {}).items():
                try:
                    self.all_personas[name] = PersonaConfig(**data)
                except ValidationError as e:
                    logger.error(
                        f"Validation error for persona '{name}' in custom framework '{framework_name}' during load: {e}"
                    )
                    continue

            # Update persona_sets with the custom framework's sets
            custom_sets = loaded_config_data.get("persona_sets", {})
            self.persona_sets.update(custom_sets)

            # Get the specific persona sequence for this framework
            current_domain_persona_names = self.persona_sets.get(framework_name, [])
            personas_for_session = {
                name: self.all_personas[name]
                for name in current_domain_persona_names
                if name in self.all_personas
            }

            self._initialize_performance_metrics()  # Re-initialize performance metrics for potentially new/updated personas

            return (
                True,
                f"Loaded custom framework: '{framework_name}'",
                personas_for_session,
                {framework_name: current_domain_persona_names},
                framework_name,
            )

        elif framework_name in self.persona_sets:  # It's a default framework
            current_domain_persona_names = self.persona_sets.get(framework_name, [])
            personas_for_session = {
                name: self.all_personas[name]
                for name in current_domain_persona_names
                if name in self.all_personas
            }

            self._initialize_performance_metrics()  # Re-initialize performance metrics for potentially new/updated personas

            return (
                True,
                f"Loaded default framework: '{framework_name}'",
                personas_for_session,
                {framework_name: current_domain_persona_names},
                framework_name,
            )
        else:
            return False, f"Framework '{framework_name}' not found.", {}, {}, ""

    def get_persona_sequence_for_framework(self, framework_name: str) -> List[str]:
        """
        Retrieves the persona sequence for a given framework name from persona_sets.
        """
        if framework_name in self.persona_sets:
            return self.persona_sets[framework_name]

        logger.warning(
            f"Persona sequence not found for framework '{framework_name}' in persona_sets. Falling back to 'General' sequence."
        )
        # Fallback to the 'General' sequence if the requested framework is not found.
        return self.persona_sets.get("General", [])

    def update_persona_config(
        self, persona_name: str, parameter: str, new_value: Any
    ) -> bool:
        """Updates a specific parameter of a persona. This is a direct modification."""
        if persona_name not in self.all_personas:
            logger.warning(f"Persona '{persona_name}' not found for update.")
            return False

        if not hasattr(self.all_personas[persona_name], parameter):
            logger.warning(f"Persona '{persona_name}' has no attribute '{parameter}'.")
            return False

        try:
            setattr(self.all_personas[persona_name], parameter, new_value)
            logger.info(
                f"Updated persona '{persona_name}' parameter '{parameter}' to '{new_value}'."
            )
            # Note: Changes made here are not persisted to disk unless saved via save_framework.
            return True
        except Exception as e:
            logger.error(
                f"Failed to update persona '{persona_name}' parameter '{parameter}' with value '{new_value}': {e}"
            )
            return False

    def reset_persona_to_default(self, persona_name: str) -> bool:
        """Resets a persona to its original default configuration."""
        if persona_name not in self._original_personas:
            logger.warning(
                f"Original configuration for persona '{persona_name}' not found. Cannot reset."
            )
            return False

        if persona_name not in self.all_personas:
            logger.warning(
                f"Persona '{persona_name}' not found in current active personas. Cannot reset."
            )
            return False

        original_config = self._original_personas[persona_name]
        current_persona = self.all_personas[persona_name]

        current_persona.system_prompt = original_config.system_prompt
        current_persona.temperature = original_config.temperature
        current_persona.max_tokens = original_config.max_tokens

        logger.info(f"Persona '{persona_name}' reset to default configuration.")

        # Reset performance metrics for this persona too
        if persona_name in self.persona_performance_metrics:
            self.persona_performance_metrics[persona_name] = {
                "total_turns": 0,
                "schema_failures": 0,
                "truncation_failures": 0,
                "last_adjusted_temp": original_config.temperature,
                "last_adjusted_max_tokens": original_config.max_tokens,
                "last_adjustment_timestamp": 0.0,
            }
        return True

    def reset_all_personas_for_current_framework(self, framework_name: str) -> bool:
        """
        Resets all personas belonging to the specified framework to their original default configurations.
        Returns:
            bool: True if the reset operation was successful (or no personas needed resetting), False otherwise.
        """
        if framework_name not in self.persona_sets:
            logger.warning(
                f"Framework '{framework_name}' not found in persona sets. Cannot reset its personas."
            )
            return False

        persona_names_in_framework = self.persona_sets[framework_name]
        success = True
        for p_name in persona_names_in_framework:
            if not self.reset_persona_to_default(p_name):
                logger.error(
                    f"Failed to reset persona '{p_name}' during bulk reset for framework '{framework_name}'."
                )
                success = False

        if success:
            logger.info(
                f"Successfully reset all personas for framework '{framework_name}'."
            )
        else:
            logger.warning(
                f"Partial success or failure during reset of personas for framework '{framework_name}'."
            )
        return success

    def export_framework_for_sharing(
        self, framework_name: str
    ) -> Tuple[bool, str, Optional[str]]:
        """Exports a framework configuration as YAML for sharing using ConfigPersistence."""
        exported_content = self.config_persistence.export_framework_for_sharing(
            framework_name
        )
        if exported_content:
            return (
                True,
                f"Framework '{framework_name}' exported successfully.",
                exported_content,
            )
        return (
            False,
            f"Framework '{framework_name}' not found or could not be exported.",
            None,
        )

    def import_framework(self, file_content: str, filename: str) -> Tuple[bool, str]:
        """Imports a framework from file content using ConfigPersistence."""
        success, message, loaded_config_data = (
            self.config_persistence.import_framework_from_file(file_content, filename)
        )
        if success and loaded_config_data:
            framework_name = loaded_config_data.get(
                "framework_name"
            ) or loaded_config_data.get("name")
            if framework_name:
                # Integrate the loaded framework's personas and sets into the manager's state
                for p_name, p_data in loaded_config_data.get("personas", {}).items():
                    try:
                        self.all_personas[p_name] = PersonaConfig(**p_data)
                    except ValidationError as e:
                        logger.error(
                            f"Validation error for persona '{p_name}' in imported framework '{framework_name}': {e}"
                        )
                        continue
                self.persona_sets.update(loaded_config_data.get("persona_sets", {}))

                if framework_name not in self.available_domains:
                    self.available_domains.append(framework_name)
                self.available_domains = sorted(list(set(self.available_domains)))
                self._initialize_performance_metrics()  # Re-initialize performance metrics
        return success, message

    def get_adjusted_persona_config(self, persona_name: str) -> PersonaConfig:
        """
        Returns a PersonaConfig with dynamically adjusted parameters based on performance.
        Also handles `_TRUNCATED` persona names by adjusting max_tokens and system_prompt.
        Returns a deep copy to prevent direct modification of cached objects.
        """
        base_persona_name = persona_name.replace("_TRUNCATED", "")
        base_config = self.all_personas.get(base_persona_name)
        if not base_config:
            logger.warning(f"Persona '{base_persona_name}' not found for adjustment.")
            return PersonaConfig(
                name=base_persona_name,  # Use the requested name for fallback
                system_prompt="Error",
                temperature=0.7,
                max_tokens=1024,
                description="Fallback persona.",  # Added description
            )  # Fallback

        adjusted_config = copy.deepcopy(base_config)
        metrics = self.persona_performance_metrics.get(base_persona_name)

        # Apply truncation if the persona name indicates it
        if "_TRUNCATED" in persona_name:
            # Reduce max_tokens for truncated versions
            original_max_tokens = adjusted_config.max_tokens
            adjusted_config.max_tokens = max(
                512, int(original_max_tokens * 0.75)
            )  # Reduce by 25%, min 512
            adjusted_config.system_prompt += "\n\nCRITICAL: Be extremely concise and focus only on the most essential information due to token constraints. Prioritize brevity. Your output MUST be shorter than usual."
            logger.info(
                f"Applied truncation to '{persona_name}': max_tokens reduced from {original_max_tokens} to {adjusted_config.max_tokens}."
            )
            # Note: We don't record truncation_failures here, but when the LLM call is made and returns `is_truncated`.
            # This ensures we only count actual truncations, not just requests for truncated personas.
            return adjusted_config

        # NEW: Apply system prompt optimization for high-token personas
        if self.prompt_optimizer:
            # Convert PersonaConfig to dict, optimize, then convert back
            optimized_system_prompt_data = (
                self.prompt_optimizer.optimize_persona_system_prompt(
                    adjusted_config.model_dump()
                )
            )
            adjusted_config = PersonaConfig.model_validate(optimized_system_prompt_data)
            # Log the change if it occurred (check if system_prompt actually changed)
            if adjusted_config.system_prompt != base_config.system_prompt:
                logger.debug(
                    f"System prompt for {persona_name} optimized by PromptOptimizer."
                )

        if (
            not metrics
            or metrics["total_turns"] < self.min_turns_for_adjustment
            or (time.time() - metrics["last_adjustment_timestamp"])
            < self.adjustment_cooldown_seconds
        ):
            return adjusted_config  # Not enough data or still in cooldown

        # Calculate failure rates
        schema_failure_rate = metrics["schema_failures"] / metrics["total_turns"]
        truncation_failure_rate = (
            metrics["truncation_failures"] / metrics["total_turns"]
        )

        # Adaptive Temperature Adjustment (for schema failures/malformed output)
        if schema_failure_rate > 0.2:  # More than 20% schema failures
            adjusted_config.temperature = max(0.1, adjusted_config.temperature - 0.15)
            logger.info(
                f"Adjusted {base_persona_name} temperature to {adjusted_config.temperature:.2f} due to high schema failure rate ({schema_failure_rate:.2f})."
            )
        elif (
            schema_failure_rate < 0.05
            and adjusted_config.temperature < base_config.temperature
        ):
            # Gently revert temperature if performance improves and it was previously lowered
            adjusted_config.temperature = min(
                base_config.temperature, adjusted_config.temperature + 0.05
            )
            logger.info(
                f"Reverted {base_persona_name} temperature to {adjusted_config.temperature:.2f} due to improved schema adherence."
            )

        # Adaptive Max Tokens Adjustment (for truncation)
        if truncation_failure_rate > 0.15:  # More than 15% truncation failures
            adjusted_config.max_tokens = min(
                8192, adjusted_config.max_tokens + 512
            )  # Increase by 512 tokens, max 8192
            logger.info(
                f"Adjusted {base_persona_name} max_tokens to {adjusted_config.max_tokens} due to high truncation rate ({truncation_failure_rate:.2f})."
            )
        elif (
            truncation_failure_rate < 0.05
            and adjusted_config.max_tokens > base_config.max_tokens
        ):
            # Gently revert max_tokens if truncation improves and it was previously increased
            adjusted_config.max_tokens = max(
                base_config.max_tokens, adjusted_config.max_tokens - 256
            )
            logger.info(
                f"Reverted {base_persona_name} max_tokens to {adjusted_config.max_tokens} due to improved truncation."
            )

        # Update last adjusted values and timestamp
        metrics["last_adjusted_temp"] = adjusted_config.temperature
        metrics["last_adjusted_max_tokens"] = adjusted_config.max_tokens
        metrics["last_adjustment_timestamp"] = time.time()

        # Reset counts after adjustment to focus on new performance
        metrics["schema_failures"] = 0
        metrics["truncation_failures"] = 0
        metrics["total_turns"] = 0

        return adjusted_config

    def record_persona_performance(
        self,
        persona_name: str,
        turn_number: int,
        output: Any,
        is_aligned: bool,  # Renamed from is_valid to be more precise
        validation_message: str,
        is_truncated: bool = False,
        schema_validation_failed: bool = False,  # NEW: Explicitly track schema failures
        token_budget_exceeded: bool = False,  # NEW: Explicitly track token budget exceedances
    ):
        """Record performance metrics for a persona's turn."""
        base_persona_name = persona_name.replace("_TRUNCATED", "")
        metrics = self.persona_performance_metrics.get(base_persona_name)
        if metrics:
            metrics["total_turns"] += 1
            if schema_validation_failed:  # Use the new explicit flag
                metrics["schema_failures"] += 1
            if is_truncated:
                metrics["truncation_failures"] += 1
            # Optionally, track token_budget_exceeded if needed for future adjustments
            # metrics["token_budget_exceeded"] += 1 if token_budget_exceeded else 0
            logger.debug(
                f"Recorded performance for {persona_name}: Turn={turn_number}, IsAligned={is_aligned}, IsTruncated={is_truncated}, SchemaFailed={schema_validation_failed}, TokenBudgetExceeded={token_budget_exceeded}, ValidationMessage='{validation_message}'"
            )

    def get_token_optimized_persona_sequence(
        self, persona_sequence: List[str]
    ) -> List[str]:
        """
        Optimizes the persona sequence by potentially replacing personas with their
        '_TRUNCATED' versions if historical performance indicates high truncation rates
        or if the overall token budget is becoming constrained.
        """
        optimized_sequence = []

        global_token_consumption_high = False
        truncation_rate = 0.0  # Initialize truncation_rate
        if self.token_tracker:
            global_token_consumption_high = (
                self.token_tracker.get_consumption_rate()
                > self.GLOBAL_TOKEN_CONSUMPTION_THRESHOLD
            )

        for p_name in persona_sequence:
            base_p_name = p_name.replace("_TRUNCATED", "")

            persona_truncation_prone = False
            metrics = self.persona_performance_metrics.get(base_p_name)
            if metrics and metrics["total_turns"] >= self.min_turns_for_adjustment:
                truncation_rate = (
                    metrics["truncation_failures"] / metrics["total_turns"]
                )
                if truncation_rate > self.TRUNCATION_FAILURE_RATE_THRESHOLD:
                    persona_truncation_prone = True

            if persona_truncation_prone or global_token_consumption_high:
                if "_TRUNCATED" not in p_name:
                    optimized_sequence.append(f"{base_p_name}_TRUNCATED")
                    logger.info(
                        f"Optimizing persona sequence: '{base_p_name}' replaced with '{base_p_name}_TRUNCATED' due to high truncation rate ({truncation_rate:.2f}) or high global token consumption."
                    )
                else:
                    optimized_sequence.append(p_name)
            else:
                optimized_sequence.append(p_name)

        return optimized_sequence

    def _analyze_prompt_complexity(self, prompt: str) -> Dict[str, Any]:
        """Analyze prompt complexity with domain-specific weighting."""
        return self.prompt_analyzer.analyze_complexity(prompt)

    def parse_raw_llm_output(
        self,
        raw_llm_output: str,
        persona_name: str,
        schema: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Parses raw LLM output, attempts JSON decoding, and validates against a schema.
        This method is intended for internal use by the SocraticDebate class.
        """
        try:
            # Attempt to parse as JSON first
            raw_output = json.loads(raw_llm_output)

            # Use Pydantic for validation where appropriate
            if persona_name == "Self_Improvement_Analyst":
                try:
                    # Use the actual model defined in the codebase
                    from src.models import SelfImprovementAnalysisOutput

                    validated = SelfImprovementAnalysisOutput(**raw_output)
                    return validated.model_dump(
                        by_alias=True
                    )  # Use model_dump for Pydantic v2
                except ValidationError as ve:
                    logger.error(f"Pydantic validation failed for {persona_name}: {ve}")
                    # Extract specific field errors for better diagnostics
                    error_details = [
                        {"field": err["loc"][0], "error": err["msg"]}
                        for err in ve.errors()
                    ]
                    raise SchemaValidationError(
                        f"Validation failed: {str(ve)}", error_details
                    ) from None

            return raw_output
        except json.JSONDecodeError as e:
            logger.error(f"JSON decoding failed for {persona_name}: {e}")
            raise SchemaValidationError(f"Invalid JSON: {str(e)}") from None
        except Exception as e:  # Catch any other parsing/validation errors
            logger.error(
                f"An unexpected error occurred during parsing/validation for {persona_name}: {e}"
            )
            raise SchemaValidationError(
                f"Parsing/validation failed: {str(e)}"
            ) from None
