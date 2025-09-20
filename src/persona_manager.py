# src/persona_manager.py
import os
import json
import yaml
import copy
import time

from typing import Dict, Any, List, Optional, Tuple
from pydantic import ValidationError

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
from src.utils.prompting.prompt_analyzer import PromptAnalyzer  # Updated import
from src.token_tracker import TokenUsageTracker
from src.exceptions import SchemaValidationError
from src.config.settings import ChimeraSettings
from src.utils.prompting.prompt_optimizer import PromptOptimizer  # Updated import

import logging

logger = logging.getLogger(__name__)

DEFAULT_PERSONAS_FILE = "personas.yaml"


class PersonaManager:
    TRUNCATION_FAILURE_RATE_THRESHOLD = 0.2
    GLOBAL_TOKEN_CONSUMPTION_THRESHOLD = 0.7

    # REMOVED: The hardcoded PERSONA_OUTPUT_SCHEMAS dictionary.
    # Schemas will now be dynamically retrieved from PersonaConfig.output_schema.

    def __init__(
        self,
        domain_keywords: Dict[str, List[str]],
        token_tracker: Optional[TokenUsageTracker] = None,
        settings: Optional[ChimeraSettings] = None,
        prompt_optimizer: Optional[PromptOptimizer] = None,
    ):
        self.all_personas: Dict[str, PersonaConfig] = {}
        self.persona_sets: Dict[str, List[str]] = {}
        self.available_domains: List[str] = []
        self.all_custom_frameworks_data: Dict[str, Any] = {}
        self.default_persona_set_name: str = "General"
        self._original_personas: Dict[str, PersonaConfig] = {}

        self.persona_performance_metrics: Dict[str, Dict[str, Any]] = {}
        self.adjustment_cooldown_seconds = 300
        self.min_turns_for_adjustment = 5

        self.config_persistence = ConfigPersistence()
        self.settings = settings or ChimeraSettings()

        self.prompt_analyzer = PromptAnalyzer(domain_keywords)
        self.prompt_optimizer = prompt_optimizer  # Store prompt_optimizer

        load_success, load_msg = self._load_initial_data()
        if not load_success and load_msg:
            logger.error(f"Failed to load initial personas: {load_msg}")

        self._load_custom_frameworks_on_init()
        self._load_original_personas()

        self.persona_router: Optional[PersonaRouter] = PersonaRouter(
            self.all_personas,
            self.persona_sets,
            self.prompt_analyzer,
            persona_manager=self,
        )

        self._initialize_performance_metrics()

        self.token_tracker = token_tracker

    def _load_initial_data(
        self, file_path: str = DEFAULT_PERSONAS_FILE
    ) -> Tuple[bool, Optional[str]]:
        """Loads the default personas and persona sets from a YAML file."""
        try:
            with open(file_path, "r") as f:
                data = yaml.safe_load(f)

            if not data:
                raise ValueError("Persona configuration file is empty.")

            # MODIFIED: Read system_prompt_template instead of system_prompt
            all_personas_list = []
            for p_data in data.get("personas", []):
                if "system_prompt" in p_data:  # Handle old format if it exists
                    p_data["system_prompt_template"] = p_data.pop("system_prompt")
                # PersonaConfig(**p_data) will now automatically read the 'output_schema' field
                all_personas_list.append(PersonaConfig(**p_data))

            self.all_personas = {p.name: p for p in all_personas_list}
            self.persona_sets = data.get("persona_sets", {"General": []})

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
            # Fallback to minimal personas with system_prompt_template
            self.all_personas = {
                "Visionary_Generator": PersonaConfig(
                    name="Visionary_Generator",
                    system_prompt_template="You are a visionary.",
                    output_schema="GeneralOutput",  # Added fallback schema
                    temperature=0.7,
                    max_tokens=1024,
                    description="Generates innovative solutions.",
                ),
                "Skeptical_Generator": PersonaConfig(
                    name="Skeptical_Generator",
                    system_prompt_template="You are a skeptic.",
                    output_schema="GeneralOutput",  # Added fallback schema
                    temperature=0.3,
                    max_tokens=1024,
                    description="Identifies flaws.",
                ),
                "Impartial_Arbitrator": PersonaConfig(
                    name="Impartial_Arbitrator",
                    system_prompt_template="You are an arbitrator.",
                    output_schema="LLMOutput",  # Added fallback schema
                    temperature=0.2,
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
                for p_name, p_data in config.get("personas", {}).items():
                    try:
                        if "system_prompt" in p_data:  # Handle old format if it exists
                            p_data["system_prompt_template"] = p_data.pop(
                                "system_prompt"
                            )
                        # PersonaConfig(**p_data) will now automatically read the 'output_schema' field
                        self.all_personas[p_name] = PersonaConfig(**p_data)
                    except ValidationError as e:
                        logger.error(
                            f"Validation error for persona '{p_name}' in custom framework '{name}': {e}"
                        )
                        continue
                self.persona_sets.update(config.get("persona_sets", {}))

                self.all_custom_frameworks_data[name] = config

                if name not in self.available_domains:
                    self.available_domains.append(name)
                self.available_domains = sorted(list(set(self.available_domains)))
        logger.info(f"Loaded {len(saved_names)} custom frameworks.")

    def _load_original_personas(self) -> Tuple[bool, Optional[str]]:
        """Loads the original default personas from the YAML file and stores them."""
        try:
            with open(DEFAULT_PERSONAS_FILE, "r") as f:
                data = yaml.safe_load(f)

            if not data:
                raise ValueError("Original personas file is empty.")

            original_personas_list = []
            for p_data in data.get("personas", []):
                if "system_prompt" in p_data:  # Handle old format if it exists
                    p_data["system_prompt_template"] = p_data.pop("system_prompt")
                # PersonaConfig(**p_data) will now automatically read the 'output_schema' field
                original_personas_list.append(PersonaConfig(**p_data))

            self._original_personas = {p.name: p for p in original_personas_list}
            logger.info("Original personas loaded and stored.")
            return True, None

        except (FileNotFoundError, ValidationError, yaml.YAMLError, ValueError) as e:
            logger.error(
                f"Error loading original personas from {DEFAULT_PERSONAS_FILE}: {e}"
            )
            return False, f"Failed to load original personas: {e}"

    def _initialize_performance_metrics(self):
        """Initializes performance metrics for all loaded personas."""
        for p_name in self.all_personas.keys():
            self.persona_performance_metrics[p_name] = {
                "total_turns": 0,
                "schema_failures": 0,
                "truncation_failures": 0,
                "last_adjusted_temp": self.all_personas[p_name].temperature,
                "last_adjusted_max_tokens": self.all_personas[p_name].max_tokens,
                "last_adjustment_timestamp": 0.0,
                "total_tokens_used": 0,  # ADDED: For token efficiency score calculation
                "successful_outputs": 0,  # ADDED: For token efficiency score calculation
            }

    def save_framework(
        self,
        name: str,
        current_persona_set_name: str,
        current_active_personas: Dict[str, PersonaConfig],
        description: str = "",
    ) -> Tuple[bool, str]:
        """Saves the current framework configuration (including persona edits) as a custom framework."""
        if not name:
            return False, "Please enter a name for the framework before saving."

        current_personas_dict = {
            p_name: p_data.model_dump()
            for p_name, p_data in current_active_personas.items()
        }

        version = 1
        if name in self.all_custom_frameworks_data:
            version = self.all_custom_frameworks_data[name].get("version", 0) + 1

        try:
            config_to_save = {
                "framework_name": name,
                "description": description,
                "personas": current_personas_dict,
                "persona_sets": {name: list(current_active_personas.keys())},
                "version": version,
            }
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
            self.all_custom_frameworks_data[name] = config_to_save
            if name not in self.available_domains:
                self.available_domains.append(name)
            self.available_domains = sorted(list(set(self.available_domains)))
            return True, message
        else:
            return False, message

    def load_framework_into_session(
        self, framework_name: str
    ) -> Tuple[bool, str, Dict[str, PersonaConfig], Dict[str, List[str]], str]:
        """Loads a framework's personas and sets, returning them for session state update."""
        loaded_config_data = (
            self.config_persistence._load_custom_framework_config_from_file(
                framework_name
            )
        )

        if loaded_config_data:
            for name, data in loaded_config_data.get("personas", {}).items():
                try:
                    if "system_prompt" in data:  # Handle old format if it exists
                        data["system_prompt_template"] = data.pop("system_prompt")
                    # PersonaConfig(**data) will now automatically read the 'output_schema' field
                    self.all_personas[name] = PersonaConfig(**data)
                except ValidationError as e:
                    logger.error(
                        f"Validation error for persona '{name}' in custom framework '{framework_name}' during load: {e}"
                    )
                    continue

            custom_sets = loaded_config_data.get("persona_sets", {})
            self.persona_sets.update(custom_sets)

            current_domain_persona_names = self.persona_sets.get(framework_name, [])
            personas_for_session = {
                name: self.all_personas[name]
                for name in current_domain_persona_names
                if name in self.all_personas
            }

            self._initialize_performance_metrics()

            return (
                True,
                f"Loaded custom framework: '{framework_name}'",
                personas_for_session,
                {framework_name: current_domain_persona_names},
                framework_name,
            )

        elif framework_name in self.persona_sets:
            current_domain_persona_names = self.persona_sets.get(framework_name, [])
            personas_for_session = {
                name: self.all_personas[name]
                for name in current_domain_persona_names
                if name in self.all_personas
            }

            self._initialize_performance_metrics()

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
        """Retrieves the persona sequence for a given framework name from persona_sets."""
        if framework_name in self.persona_sets:
            return self.persona_sets[framework_name]

        logger.warning(
            f"Persona sequence not found for framework '{framework_name}' in persona_sets. Falling back to 'General' sequence."
        )
        return self.persona_sets.get("General", [])

    def update_persona_config(
        self, persona_name: str, parameter: str, new_value: Any
    ) -> bool:
        """Updates a specific parameter of a persona. This is a direct modification."""
        if persona_name not in self.all_personas:
            logger.warning(f"Persona '{persona_name}' not found for update.")
            return False

        # MODIFIED: Handle system_prompt_template update
        if (
            parameter == "system_prompt"
        ):  # If UI sends system_prompt, map to system_prompt_template
            parameter = "system_prompt_template"

        if not hasattr(self.all_personas[persona_name], parameter):
            logger.warning(f"Persona '{persona_name}' has no attribute '{parameter}'.")
            return False

        try:
            setattr(self.all_personas[persona_name], parameter, new_value)
            logger.info(
                f"Updated persona '{persona_name}' parameter '{parameter}' to '{new_value}'."
            )
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

        # MODIFIED: Reset system_prompt_template
        current_persona.system_prompt_template = original_config.system_prompt_template
        current_persona.temperature = original_config.temperature
        current_persona.max_tokens = original_config.max_tokens
        current_persona.token_efficiency_score = (
            original_config.token_efficiency_score
        )  # ADDED
        current_persona.output_schema = (
            original_config.output_schema
        )  # ADDED: Reset output_schema

        logger.info(f"Persona '{persona_name}' reset to default configuration.")

        if persona_name in self.persona_performance_metrics:
            self.persona_performance_metrics[persona_name] = {
                "total_turns": 0,
                "schema_failures": 0,
                "truncation_failures": 0,
                "last_adjusted_temp": original_config.temperature,
                "last_adjusted_max_tokens": original_config.max_tokens,
                "last_adjustment_timestamp": 0.0,
                "total_tokens_used": 0,  # ADDED
                "successful_outputs": 0,  # ADDED
            }
        return True

    def reset_all_personas_for_current_framework(self, framework_name: str) -> bool:
        """Resets all personas belonging to the specified framework to their original default configurations."""
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
                for p_name, p_data in loaded_config_data.get("personas", {}).items():
                    try:
                        if "system_prompt" in p_data:  # Handle old format if it exists
                            p_data["system_prompt_template"] = p_data.pop(
                                "system_prompt"
                            )
                        # PersonaConfig(**p_data) will now automatically read the 'output_schema' field
                        self.all_personas[p_name] = PersonaConfig(**p_data)
                    except ValidationError as e:
                        logger.error(
                            f"Validation error for persona '{p_name}' in imported framework '{framework_name}' : {e}"
                        )
                        continue
                self.persona_sets.update(loaded_config_data.get("persona_sets", {}))

                if framework_name not in self.available_domains:
                    self.available_domains.append(framework_name)
                self.available_domains = sorted(list(set(self.available_domains)))
                self._initialize_performance_metrics()
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
                name=base_persona_name,
                system_prompt_template="You are a helpful AI assistant.",
                output_schema="GeneralOutput",  # Added fallback schema
                temperature=0.7,
                max_tokens=1024,
                description="Fallback persona.",
            )

        # Initialize rendered_prompt to a default value to ensure it's always bound
        rendered_prompt: str = ""

        adjusted_config = copy.deepcopy(base_config)
        metrics = self.persona_performance_metrics.get(base_persona_name)

        # Render the system prompt from template first
        if self.prompt_optimizer and adjusted_config.system_prompt_template:
            try:
                # BUG FIX: Pass the persona's base name (converted to snake_case) as the template key.
                # The system_prompt_template field in PersonaConfig now holds the template content,
                # but the prompt_optimizer expects the template's file name (key) to retrieve it.
                rendered_prompt = self.prompt_optimizer.generate_prompt(
                    base_persona_name.lower(),  # Use the persona name (snake_case) as the template key
                    context={"persona_name": base_persona_name},
                )
                # Temporarily store the rendered prompt in a new attribute for optimization
                # This is a workaround as PersonaConfig doesn't have a 'system_prompt' field anymore
                adjusted_config._rendered_system_prompt = rendered_prompt
            except ValueError as e:
                logger.error(
                    f"Failed to render system prompt template for {base_persona_name}: {e}"
                )
                adjusted_config._rendered_system_prompt = (
                    "Error: Could not render system prompt."
                )
                rendered_prompt = (
                    adjusted_config._rendered_system_prompt
                )  # Ensure rendered_prompt is also updated

        # Apply truncation if the persona name indicates it
        if "_TRUNCATED" in persona_name:
            original_max_tokens = adjusted_config.max_tokens
            adjusted_config.max_tokens = max(512, int(original_max_tokens * 0.75))
            # MODIFIED: Append truncation directive to the *rendered* prompt
            if hasattr(adjusted_config, "_rendered_system_prompt"):
                adjusted_config._rendered_system_prompt += "\n\nCRITICAL: Be extremely concise and focus only on the most essential information due to token constraints. Prioritize brevity. Your output MUST be shorter than usual."
            logger.info(
                f"Applied truncation to '{persona_name}': max_tokens reduced from {original_max_tokens} to {adjusted_config.max_tokens}."
            )
            return adjusted_config

        # Apply system prompt optimization for high-token personas (on the rendered prompt)
        if self.prompt_optimizer and hasattr(
            adjusted_config, "_rendered_system_prompt"
        ):
            optimized_system_prompt_data = {
                "name": adjusted_config.name,
                "system_prompt": adjusted_config._rendered_system_prompt,  # Pass the rendered prompt
            }
            optimized_system_prompt_data = (
                self.prompt_optimizer.optimize_persona_system_prompt(
                    optimized_system_prompt_data
                )
            )
            adjusted_config._rendered_system_prompt = optimized_system_prompt_data[
                "system_prompt"
            ]
            if optimized_system_prompt_data["system_prompt"] != rendered_prompt:
                logger.debug(
                    f"System prompt for {persona_name} optimized by PromptOptimizer."
                )

        if (
            not metrics
            or metrics["total_turns"] < self.min_turns_for_adjustment
            or (time.time() - metrics["last_adjustment_timestamp"])
            < self.adjustment_cooldown_seconds
        ):
            return adjusted_config

        schema_failure_rate = metrics["schema_failures"] / metrics["total_turns"]
        truncation_failure_rate = (
            metrics["truncation_failures"] / metrics["total_turns"]
        )

        if schema_failure_rate > 0.2:
            adjusted_config.temperature = max(0.1, adjusted_config.temperature - 0.15)
            logger.info(
                f"Adjusted {base_persona_name} temperature to {adjusted_config.temperature:.2f} due to high schema failure rate ({schema_failure_rate:.2f})."
            )
        elif (
            schema_failure_rate < 0.05
            and adjusted_config.temperature < base_config.temperature
        ):
            adjusted_config.temperature = min(
                base_config.temperature, adjusted_config.temperature + 0.05
            )
            logger.info(
                f"Reverted {base_persona_name} temperature to {adjusted_config.temperature:.2f} due to improved schema adherence."
            )

        if truncation_failure_rate > 0.15:
            adjusted_config.max_tokens = min(8192, adjusted_config.max_tokens + 512)
            logger.info(
                f"Adjusted {base_persona_name} max_tokens to {adjusted_config.max_tokens} due to high truncation rate ({truncation_failure_rate:.2f})."
            )
        elif (
            truncation_failure_rate < 0.05
            and adjusted_config.max_tokens > base_config.max_tokens
        ):
            adjusted_config.max_tokens = max(
                base_config.max_tokens, adjusted_config.max_tokens - 256
            )
            logger.info(
                f"Reverted {base_persona_name} max_tokens to {adjusted_config.max_tokens} due to improved truncation."
            )

        # Calculate token_efficiency_score
        if metrics["total_tokens_used"] > 0 and metrics["successful_outputs"] > 0:
            # Heuristic: Lower tokens per successful output is better
            # Normalize to a 0-1 scale. This is a simple example.
            # A more complex model would use a baseline and inverse relationship.
            avg_tokens_per_success = (
                metrics["total_tokens_used"] / metrics["successful_outputs"]
            )
            # Assuming a target of 1000 tokens/success is ideal, scale inversely
            adjusted_config.token_efficiency_score = min(
                1.0, 1000 / avg_tokens_per_success
            )
        else:
            adjusted_config.token_efficiency_score = 0.5  # Default if no data

        metrics["last_adjusted_temp"] = adjusted_config.temperature
        metrics["last_adjusted_max_tokens"] = adjusted_config.max_tokens
        metrics["last_adjustment_timestamp"] = time.time()
        metrics["schema_failures"] = 0
        metrics["truncation_failures"] = 0
        metrics["total_turns"] = 0
        metrics["total_tokens_used"] = 0  # ADDED: Reset
        metrics["successful_outputs"] = 0  # ADDED: Reset

        return adjusted_config

    def record_persona_performance(
        self,
        persona_name: str,
        turn_number: int,
        output: Any,
        is_aligned: bool,
        validation_message: str,
        is_truncated: bool = False,
        schema_validation_failed: bool = False,
        token_budget_exceeded: bool = False,
        tokens_used_in_turn: int = 0,  # ADDED: For token efficiency score calculation
    ):
        """Record performance metrics for a persona's turn."""
        base_persona_name = persona_name.replace("_TRUNCATED", "")
        metrics = self.persona_performance_metrics.get(base_persona_name)
        if metrics:
            metrics["total_turns"] += 1
            if schema_validation_failed:
                metrics["schema_failures"] += 1
            if is_truncated:
                metrics["truncation_failures"] += 1

            metrics["total_tokens_used"] += tokens_used_in_turn  # ADDED
            if (
                is_aligned
                and not schema_validation_failed
                and not token_budget_exceeded
            ):
                metrics["successful_outputs"] += 1  # ADDED

            logger.debug(
                f"Recorded performance for {persona_name}: Turn={turn_number}, IsAligned={is_aligned}, IsTruncated={is_truncated}, SchemaFailed={schema_validation_failed}, TokenBudgetExceeded={token_budget_exceeded}, TokensUsed={tokens_used_in_turn}, ValidationMessage='{validation_message}'"
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
        truncation_rate = 0.0
        if self.token_tracker:
            global_token_consumption_high = (
                self.token_tracker.get_consumption_rate()
                > self.settings.GLOBAL_TOKEN_CONSUMPTION_THRESHOLD
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

    # REMOVED: The parse_raw_llm_output method as it is unused and redundant with LLMOutputParser.
