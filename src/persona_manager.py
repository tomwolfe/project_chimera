# src/persona_manager.py
import os
import json
import yaml
import datetime
import re
import logging
from typing import Dict, Any, List, Optional, Tuple, Union
from pydantic import ValidationError
import streamlit as st
import copy # Added for deepcopy
import time # Added for time.time()

from src.persona.routing import PersonaRouter
from src.models import PersonaConfig, ReasoningFrameworkConfig

logger = logging.getLogger(__name__)

CUSTOM_FRAMEWORKS_DIR = "custom_frameworks"
DEFAULT_PERSONAS_FILE = "personas.yaml"

@st.cache_resource
class PersonaManager:
    def __init__(self):
        self.all_personas: Dict[str, PersonaConfig] = {}
        self.persona_sets: Dict[str, List[str]] = {}
        self.available_domains: List[str] = []
        self.all_custom_frameworks_data: Dict[str, Any] = {}
        self.default_persona_set_name: str = "General"
        self._original_personas: Dict[str, PersonaConfig] = {}
        self.persona_router: Optional[PersonaRouter] = None

        # NEW: For Adaptive LLM Parameter Adjustment
        self.persona_performance_metrics: Dict[str, Dict[str, Any]] = {}
        self.adjustment_cooldown_seconds = 300 # 5 minutes cooldown
        self.min_turns_for_adjustment = 5 # Minimum turns before considering adjustment

        # Load initial data and custom frameworks, handle errors internally
        load_success, load_msg = self._load_initial_data()
        if not load_success and load_msg:
            logger.error(f"Failed to load initial personas: {load_msg}")
            # In a real app, you might want to raise an exception here or have a more robust fallback.
            # For now, we'll proceed with potentially empty or minimal data, logging the error.

        self._load_custom_frameworks_on_init()
        self._load_original_personas()
        
        # Initialize PersonaRouter with all loaded personas and persona_sets
        self.persona_router = PersonaRouter(self.all_personas, self.persona_sets)

        # NEW: Initialize performance metrics after all personas are loaded
        self._initialize_performance_metrics()

    def _ensure_custom_frameworks_dir(self) -> Tuple[bool, Optional[str]]:
        """Ensures the custom frameworks directory exists, creating it if necessary.
        Returns:
            Tuple[bool, Optional[str]]: (success_status, message_or_None)
        """
        if not os.path.exists(CUSTOM_FRAMEWORKS_DIR):
            try:
                os.makedirs(CUSTOM_FRAMEWORKS_DIR)
                return True, f"Created directory for custom frameworks: '{CUSTOM_FRAMEWORKS_DIR}'"
            except OSError as e:
                return False, f"Error creating custom frameworks directory: {e}"
        return True, None # Directory already exists or was created successfully

    def _sanitize_framework_filename(self, name: str) -> str:
        """Sanitizes a string to be used as a filename for custom frameworks."""
        sanitized = re.sub(r'[<>:"/\\|?*\s]+', '_', name)
        sanitized = re.sub(r'^[^a-zA-Z0-9_]+|[^a-zA-Z0-9_]+$', '', sanitized)
        if not sanitized:
            sanitized = "unnamed_framework"
        return sanitized

    def _load_initial_data(self, file_path: str = DEFAULT_PERSONAS_FILE) -> Tuple[bool, Optional[str]]:
        """Loads the default personas and persona sets from a YAML file.
        Returns:
            Tuple[bool, Optional[str]]: (success_status, error_message_or_None)
        """
        try:
            with open(file_path, 'r') as f:
                data = yaml.safe_load(f)
            
            if not data: # Handle empty file case
                raise ValueError("Persona configuration file is empty.")

            all_personas_list = [PersonaConfig(**p_data) for p_data in data.get('personas', [])]
            self.all_personas = {p.name: p for p in all_personas_list}
            self.persona_sets = data.get('persona_sets', {"General": []})
            
            # Validate persona sets references
            for set_name, persona_names_in_set in self.persona_sets.items():
                if not isinstance(persona_names_in_set, list):
                    raise ValueError(f"Persona set '{set_name}' must be a list of persona names.")
                for p_name in persona_names_in_set:
                    if p_name not in self.all_personas:
                        raise ValueError(f"Persona '{p_name}' referenced in set '{set_name}' not found in 'personas' list.")
            
            self.default_persona_set_name = "General" if "General" in self.persona_sets else next(iter(self.persona_sets.keys()), "General")
            self.available_domains = list(self.persona_sets.keys())
            logger.info(f"Initial personas loaded successfully from {file_path}.")
            return True, None
            
        except (FileNotFoundError, ValidationError, yaml.YAMLError, ValueError) as e:
            logger.error(f"Error loading initial personas from {file_path}: {e}")
            # Fallback to minimal personas if loading fails
            self.all_personas = {
                "Visionary_Generator": PersonaConfig(name="Visionary_Generator", system_prompt="You are a visionary.", temperature=0.7, max_tokens=1024),
                "Skeptical_Generator": PersonaConfig(name="Skeptical_Generator", system_prompt="You are a skeptic.", temperature=0.3, max_tokens=1024),
                "Impartial_Arbitrator": PersonaConfig(name="Impartial_Arbitrator", system_prompt="You are an arbitrator.", temperature=0.2, max_tokens=1024)
            }
            self.persona_sets = {"General": ["Visionary_Generator", "Skeptical_Generator", "Impartial_Arbitrator"]}
            self.default_persona_set_name = "General"
            self.available_domains = ["General"]
            logger.warning("Loaded minimal fallback personas due to initial loading error.")
            return False, f"Failed to load default personas from {file_path}: {e}"

    def _load_custom_frameworks_on_init(self):
        """Loads custom frameworks available at startup."""
        dir_success, dir_msg = self._ensure_custom_frameworks_dir()
        if not dir_success and dir_msg:
            logger.error(dir_msg) # Log the error if directory creation failed

        saved_names = self._get_saved_custom_framework_names()
        for name in saved_names:
            config = self._load_custom_framework_config_from_file(name)
            if config:
                self.all_custom_frameworks_data[name] = config
                if name not in self.available_domains:
                    self.available_domains.append(name)
        self.available_domains = sorted(list(set(self.available_domains)))

    def _get_saved_custom_framework_names(self) -> List[str]:
        """Returns a list of names of custom frameworks found on disk."""
        dir_success, dir_msg = self._ensure_custom_frameworks_dir()
        if not dir_success and dir_msg:
            logger.error(dir_msg)
            return [] # Return empty list if directory is inaccessible

        framework_names = []
        try:
            for filename in os.listdir(CUSTOM_FRAMEWORKS_DIR):
                if filename.endswith(".json"):
                    framework_names.append(os.path.splitext(filename)[0])
            return sorted(framework_names)
        except OSError as e:
            logger.error(f"Error listing custom frameworks: {e}")
            return []

    def _load_custom_framework_config_from_file(self, name: str) -> Optional[Dict[str, Any]]:
        """Loads a specific custom framework configuration from a JSON file.
        Returns:
            Optional[Dict[str, Any]]: The loaded configuration data, or None if an error occurs.
        """
        filename = f"{self._sanitize_framework_filename(name)}.json"
        filepath = os.path.join(CUSTOM_FRAMEWORKS_DIR, filename)
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                config_data = json.load(f)
            return config_data
        except (FileNotFoundError, json.JSONDecodeError, OSError) as e:
            if isinstance(e, FileNotFoundError):
                logger.error(f"Custom framework '{name}' not found at '{filepath}'.")
            elif isinstance(e, json.JSONDecodeError):
                logger.error(f"Error decoding JSON from '{filepath}'. Please check its format: {e}")
            else: # Catch other OSError
                logger.error(f"Error reading custom framework file '{filepath}': {e}")
            return None

    def _load_original_personas(self) -> Tuple[bool, Optional[str]]:
        """Loads the original default personas from the YAML file and stores them."""
        try:
            # Load base configuration (assuming personas.yaml is the source of original personas)
            with open(DEFAULT_PERSONAS_FILE, 'r') as f:
                data = yaml.safe_load(f)
            
            if not data:
                raise ValueError("Original personas file is empty.")

            original_personas_list = [PersonaConfig(**p_data) for p_data in data.get('personas', [])]
            self._original_personas = {p.name: p for p in original_personas_list}
            logger.info("Original personas loaded and stored.")
            return True, None
            
        except (FileNotFoundError, ValidationError, yaml.YAMLError, ValueError) as e:
            logger.error(f"Error loading original personas from {DEFAULT_PERSONAS_FILE}: {e}")
            return False, f"Failed to load original personas: {e}"

    # NEW: For Adaptive LLM Parameter Adjustment
    def _initialize_performance_metrics(self):
        """Initializes performance metrics for all loaded personas."""
        for p_name in self.all_personas.keys():
            self.persona_performance_metrics[p_name] = {
                'total_turns': 0,
                'schema_failures': 0,
                'truncation_failures': 0,
                'last_adjusted_temp': self.all_personas[p_name].temperature,
                'last_adjusted_max_tokens': self.all_personas[p_name].max_tokens,
                'last_adjustment_timestamp': 0.0
            }

    def save_framework(self, name: str, current_persona_set_name: str, current_active_personas: Dict[str, PersonaConfig]) -> Tuple[bool, str]:
        """Saves the current framework configuration (including persona edits) as a custom framework.
        Returns:
            Tuple[bool, str]: (success_status, message)
        """
        if not name:
            return False, "Please enter a name for the framework before saving."
        
        framework_name_sanitized = self._sanitize_framework_filename(name)
        if not framework_name_sanitized:
            return False, "Invalid framework name provided after sanitization."
        
        current_personas_dict = {
            p_name: p_data.model_dump()
            for p_name, p_data in current_active_personas.items()
        }

        version = 1
        if framework_name_sanitized in self.all_custom_frameworks_data:
            version = self.all_custom_frameworks_data[framework_name_sanitized].get('version', 0) + 1

        try:
            # Create a ReasoningFrameworkConfig for validation before saving
            temp_config_validation = ReasoningFrameworkConfig(
                framework_name=name, # Use the original name for the framework key
                personas={p_name: PersonaConfig(**p_data) for p_name, p_data in current_personas_dict.items()},
                persona_sets={name: list(current_active_personas.keys())}, # Use the framework name as the key for its persona set
                version=version
            )
            config_to_save = temp_config_validation.model_dump()
        except Exception as e:
            return False, f"Cannot save framework: Invalid data structure. {e}"
        
        filename = f"{framework_name_sanitized}.json"
        filepath = os.path.join(CUSTOM_FRAMEWORKS_DIR, filename)
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(config_to_save, f, indent=2)
            
            self.all_custom_frameworks_data[framework_name_sanitized] = config_to_save
            if framework_name_sanitized not in self.available_domains:
                 self.available_domains.append(framework_name_sanitized)
            self.available_domains = sorted(list(set(self.available_domains)))
            return True, f"Framework '{name}' saved successfully to '{filepath}'!"
        except OSError as e:
            return False, f"Error saving framework '{name}' to '{filepath}': {e}"
        except Exception as e:
            return False, f"An unexpected error occurred while saving framework '{name}': {e}"

    def load_framework_into_session(self, framework_name: str) -> Tuple[bool, str, Dict[str, PersonaConfig], Dict[str, List[str]], str]:
        """Loads a framework's personas and sets, returning them for session state update.
        Returns:
            Tuple[bool, str, Dict[str, PersonaConfig], Dict[str, List[str]], str]:
            (success_status, message, loaded_personas, loaded_persona_sets, new_framework_name)
        """
        if framework_name in self.all_custom_frameworks_data:
            loaded_config_data = self.all_custom_frameworks_data[framework_name]
            
            # Update all_personas with personas from the custom framework
            for name, data in loaded_config_data.get('personas', {}).items():
                self.all_personas[name] = PersonaConfig(**data)
            
            # Update persona_sets with the custom framework's sets
            custom_sets = loaded_config_data.get('persona_sets', {})
            self.persona_sets.update(custom_sets)
            
            # Get the specific persona sequence for this framework
            current_domain_persona_names = self.persona_sets.get(framework_name, [])
            personas_for_session = {name: self.all_personas[name] for name in current_domain_persona_names if name in self.all_personas}
            
            # Re-initialize performance metrics for potentially new/updated personas
            self._initialize_performance_metrics()

            return True, f"Loaded custom framework: '{framework_name}'", personas_for_session, {framework_name: current_domain_persona_names}, framework_name
            
        elif framework_name in self.persona_sets: # It's a default framework
            current_domain_persona_names = self.persona_sets.get(framework_name, [])
            personas_for_session = {name: self.all_personas[name] for name in current_domain_persona_names if name in self.all_personas}
            
            # Re-initialize performance metrics for potentially new/updated personas
            self._initialize_performance_metrics()

            return True, f"Loaded default framework: '{framework_name}'", personas_for_session, {framework_name: current_domain_persona_names}, framework_name
        else:
            return False, f"Framework '{framework_name}' not found.", {}, {}, ""

    def get_persona_sequence_for_framework(self, framework_name: str) -> List[str]:
        """
        Retrieves the persona sequence for a given framework name from persona_sets.
        """
        if framework_name in self.persona_sets:
            return self.persona_sets[framework_name]
        
        logger.warning(f"Persona sequence not found for framework '{framework_name}' in persona_sets. Falling back to 'General' sequence.")
        # Fallback to the 'General' sequence if the requested framework is not found.
        return self.persona_sets.get("General", [])

    def update_persona_config(self, persona_name: str, parameter: str, new_value: Any) -> bool:
        """Updates a specific parameter of a persona. This is a direct modification."""
        if persona_name not in self.all_personas:
            logger.warning(f"Persona '{persona_name}' not found for update.")
            return False
        
        if not hasattr(self.all_personas[persona_name], parameter):
            logger.warning(f"Persona '{persona_name}' has no attribute '{parameter}'.")
            return False
        
        try:
            setattr(self.all_personas[persona_name], parameter, new_value)
            logger.info(f"Updated persona '{persona_name}' parameter '{parameter}' to '{new_value}'.")
            # Note: Changes made here are not persisted to disk unless saved via save_framework.
            return True
        except Exception as e:
            logger.error(f"Failed to update persona '{persona_name}' parameter '{parameter}' with value '{new_value}': {e}")
            return False

    def reset_persona_to_default(self, persona_name: str) -> bool:
        """Resets a persona to its original default configuration."""
        if persona_name not in self._original_personas:
            logger.warning(f"Original configuration for persona '{persona_name}' not found. Cannot reset.")
            return False
        
        if persona_name not in self.all_personas:
            logger.warning(f"Persona '{persona_name}' not found in current active personas. Cannot reset.")
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
                'total_turns': 0,
                'schema_failures': 0,
                'truncation_failures': 0,
                'last_adjusted_temp': original_config.temperature,
                'last_adjusted_max_tokens': original_config.max_tokens,
                'last_adjustment_timestamp': 0.0
            }
        return True

    def reset_all_personas_for_current_framework(self, framework_name: str) -> bool:
        """
        Resets all personas belonging to the specified framework to their original default configurations.
        Returns:
            bool: True if the reset operation was successful (or no personas needed resetting), False otherwise.
        """
        if framework_name not in self.persona_sets:
            logger.warning(f"Framework '{framework_name}' not found in persona sets. Cannot reset its personas.")
            return False

        persona_names_in_framework = self.persona_sets[framework_name]
        success = True
        for p_name in persona_names_in_framework:
            if not self.reset_persona_to_default(p_name):
                logger.error(f"Failed to reset persona '{p_name}' during bulk reset for framework '{framework_name}'.")
                success = False
        
        if success:
            logger.info(f"Successfully reset all personas for framework '{framework_name}'.")
        else:
            logger.warning(f"Partial success or failure during reset of personas for framework '{framework_name}'.")
        return success

    # NEW: For Adaptive LLM Parameter Adjustment
    def get_adjusted_persona_config(self, persona_name: str) -> PersonaConfig:
        """
        Returns a PersonaConfig with dynamically adjusted parameters based on performance.
        Returns a deep copy to prevent direct modification of cached objects.
        """
        base_config = self.all_personas.get(persona_name)
        if not base_config:
            logger.warning(f"Persona '{persona_name}' not found for adjustment.")
            return PersonaConfig(name="Fallback", system_prompt="Error", temperature=0.7, max_tokens=1024) # Fallback

        adjusted_config = copy.deepcopy(base_config)
        metrics = self.persona_performance_metrics.get(persona_name)

        if not metrics or metrics['total_turns'] < self.min_turns_for_adjustment or \
           (time.time() - metrics['last_adjustment_timestamp']) < self.adjustment_cooldown_seconds:
            return adjusted_config # Not enough data or still in cooldown

        # Calculate failure rates
        schema_failure_rate = metrics['schema_failures'] / metrics['total_turns']
        truncation_failure_rate = metrics['truncation_failures'] / metrics['total_turns']

        # Adaptive Temperature Adjustment (for schema failures/malformed output)
        if schema_failure_rate > 0.2: # More than 20% schema failures
            adjusted_config.temperature = max(0.1, adjusted_config.temperature - 0.15)
            logger.info(f"Adjusted {persona_name} temperature to {adjusted_config.temperature:.2f} due to high schema failure rate ({schema_failure_rate:.2f}).")
        elif schema_failure_rate < 0.05 and adjusted_config.temperature < base_config.temperature:
            # Gently revert temperature if performance improves and it was previously lowered
            adjusted_config.temperature = min(base_config.temperature, adjusted_config.temperature + 0.05)
            logger.info(f"Reverted {persona_name} temperature to {adjusted_config.temperature:.2f} due to improved schema adherence.")

        # Adaptive Max Tokens Adjustment (for truncation)
        if truncation_failure_rate > 0.15: # More than 15% truncation failures
            adjusted_config.max_tokens = min(8192, adjusted_config.max_tokens + 512) # Increase by 512 tokens, max 8192
            logger.info(f"Adjusted {persona_name} max_tokens to {adjusted_config.max_tokens} due to high truncation rate ({truncation_failure_rate:.2f}).")
        elif truncation_failure_rate < 0.05 and adjusted_config.max_tokens > base_config.max_tokens:
            # Gently revert max_tokens if truncation improves and it was previously increased
            adjusted_config.max_tokens = max(base_config.max_tokens, adjusted_config.max_tokens - 256)
            logger.info(f"Reverted {persona_name} max_tokens to {adjusted_config.max_tokens} due to improved truncation.")

        # Update last adjusted values and timestamp
        metrics['last_adjusted_temp'] = adjusted_config.temperature
        metrics['last_adjusted_max_tokens'] = adjusted_config.max_tokens
        metrics['last_adjustment_timestamp'] = time.time()
        
        # Reset counts after adjustment to focus on new performance
        metrics['schema_failures'] = 0
        metrics['truncation_failures'] = 0
        metrics['total_turns'] = 0

        return adjusted_config

    # NEW: For Adaptive LLM Parameter Adjustment
    def record_persona_performance(self, persona_name: str, success: bool, is_truncated: bool, has_schema_error: bool):
        """Records performance metrics for a persona after a turn."""
        metrics = self.persona_performance_metrics.get(persona_name)
        if metrics:
            metrics['total_turns'] += 1
            if has_schema_error:
                metrics['schema_failures'] += 1
            if is_truncated:
                metrics['truncation_failures'] += 1
            logger.debug(f"Recorded performance for {persona_name}: Success={success}, Truncated={is_truncated}, SchemaError={has_schema_error}")