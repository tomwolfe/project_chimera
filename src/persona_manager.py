# src/persona_manager.py
import os
import json
import yaml
import datetime
import re
import logging
from typing import Dict, Any, List, Optional, Tuple
from pydantic import ValidationError
import streamlit as st # For st.cache_resource, st.toast, st.error

from src.models import PersonaConfig, ReasoningFrameworkConfig

logger = logging.getLogger(__name__)

CUSTOM_FRAMEWORKS_DIR = "custom_frameworks"
DEFAULT_PERSONAS_FILE = "personas.yaml"

@st.cache_resource
class PersonaManager:
    def __init__(self):
        self.all_personas: Dict[str, PersonaConfig] = {}
        self.persona_sets: Dict[str, List[str]] = {}
        self.persona_sequence: List[str] = []
        self.available_domains: List[str] = []
        self.all_custom_frameworks_data: Dict[str, Any] = {}
        self.default_persona_set_name: str = "General"
        self._load_initial_data()
        self._load_custom_frameworks_on_init()

    def _ensure_custom_frameworks_dir(self):
        if not os.path.exists(CUSTOM_FRAMEWORKS_DIR):
            try:
                os.makedirs(CUSTOM_FRAMEWORKS_DIR)
                st.toast(f"Created directory for custom frameworks: '{CUSTOM_FRAMEWORKS_DIR}'")
            except OSError as e:
                st.error(f"Error creating custom frameworks directory: {e}")

    def _sanitize_framework_filename(self, name: str) -> str:
        sanitized = re.sub(r'[<>:"/\\|?*\s]+', '_', name)
        sanitized = re.sub(r'^[^a-zA-Z0-9_]+|[^a-zA-Z0-9_]+$', '', sanitized)
        if not sanitized:
            sanitized = "unnamed_framework"
        return sanitized

    def _load_initial_data(self, file_path: str = DEFAULT_PERSONAS_FILE):
        try:
            with open(file_path, 'r') as f:
                data = yaml.safe_load(f)
            all_personas_list = [PersonaConfig(**p_data) for p_data in data.get('personas', [])]
            self.all_personas = {p.name: p for p in all_personas_list}
            self.persona_sets = data.get('persona_sets', {"General": []})
            self.persona_sequence = data.get('persona_sequence', [
                "Visionary_Generator", "Skeptical_Generator", "Constructive_Critic",
                "Impartial_Arbitrator", "Devils_Advocate"
            ])
            for set_name, persona_names_in_set in self.persona_sets.items():
                if not isinstance(persona_names_in_set, list):
                    raise ValueError(f"Persona set '{set_name}' must be a list of persona names.")
                for p_name in persona_names_in_set:
                    if p_name not in self.all_personas:
                        raise ValueError(f"Persona '{p_name}' referenced in set '{set_name}' not found in 'personas' list.")
            for p_name in self.persona_sequence:
                if p_name not in self.all_personas:
                    raise ValueError(f"Persona '{p_name}' in persona_sequence not found in 'personas' list.")
            self.default_persona_set_name = "General" if "General" in self.persona_sets else next(iter(self.persona_sets.keys()))
            self.available_domains = list(self.persona_sets.keys())
            logger.info(f"Initial personas loaded successfully from {file_path}.")
        except (FileNotFoundError, ValidationError, yaml.YAMLError) as e:
            logger.error(f"Error loading initial personas from {file_path}: {e}")
            st.error(f"Failed to load default personas from {file_path}: {e}")
            self.all_personas = {}
            self.persona_sets = {}
            self.persona_sequence = []
            self.available_domains = ["General"]
            self.default_persona_set_name = "General"

    def _load_custom_frameworks_on_init(self):
        self._ensure_custom_frameworks_dir()
        saved_names = self._get_saved_custom_framework_names()
        for name in saved_names:
            try:
                config = self._load_custom_framework_config_from_file(name)
                if config:
                    self.all_custom_frameworks_data[name] = config
                    if name not in self.available_domains:
                        self.available_domains.append(name)
            except Exception as e:
                st.error(f"Failed to load custom framework '{name}': {e}")
        self.available_domains = sorted(list(set(self.available_domains))) # Ensure unique and sorted

    def _get_saved_custom_framework_names(self) -> List[str]:
        self._ensure_custom_frameworks_dir()
        framework_names = []
        try:
            for filename in os.listdir(CUSTOM_FRAMEWORKS_DIR):
                if filename.endswith(".json"):
                    framework_names.append(os.path.splitext(filename)[0])
            return sorted(framework_names)
        except OSError as e:
            st.error(f"Error listing custom frameworks: {e}")
            return []

    def _load_custom_framework_config_from_file(self, name: str) -> Dict[str, Any]:
        filename = f"{self._sanitize_framework_filename(name)}.json"
        filepath = os.path.join(CUSTOM_FRAMEWORKS_DIR, filename)
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                config_data = json.load(f)
            return config_data
        except (FileNotFoundError, json.JSONDecodeError, OSError) as e:
            if isinstance(e, FileNotFoundError):
                st.error(f"Custom framework '{name}' not found at '{filepath}'.")
            elif isinstance(e, json.JSONDecodeError):
                st.error(f"Error decoding JSON from '{filepath}'. Please check its format: {e}")
            st.error(f"Error reading custom framework file '{filepath}': {e}")
            return {}

    def save_framework(self, name: str, current_persona_set_name: str, current_active_personas: Dict[str, PersonaConfig]):
        if not name:
            st.warning("Please enter a name for the framework before saving.")
            return False
        framework_name_sanitized = self._sanitize_framework_filename(name)
        if not framework_name_sanitized:
            st.error("Invalid framework name provided after sanitization.")
            return False
        
        current_personas_dict = {
            p_name: p_data.model_dump()
            for p_name, p_data in current_active_personas.items()
        }

        version = 1
        if framework_name_sanitized in self.all_custom_frameworks_data:
            version = self.all_custom_frameworks_data[framework_name_sanitized].get('version', 0) + 1

        try:
            temp_config_validation = ReasoningFrameworkConfig(
                framework_name=name,
                personas={p_name: PersonaConfig(**p_data) for p_name, p_data in current_personas_dict.items()},
                persona_sets={current_persona_set_name: list(current_active_personas.keys())},
                version=version
            )
            config_to_save = temp_config_validation.model_dump()
        except Exception as e:
            st.error(f"Cannot save framework: Invalid data structure. {e}")
            return False
        
        filename = f"{framework_name_sanitized}.json"
        filepath = os.path.join(CUSTOM_FRAMEWORKS_DIR, filename)
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(config_to_save, f, indent=2)
            st.toast(f"Framework '{name}' saved successfully to '{filepath}'!")
            
            self.all_custom_frameworks_data[framework_name_sanitized] = config_to_save
            if framework_name_sanitized not in self.available_domains:
                 self.available_domains.append(framework_name_sanitized)
            self.available_domains = sorted(list(set(self.available_domains))) # Re-sort after adding
            return True
        except OSError as e:
            st.error(f"Error saving framework '{name}' to '{filepath}': {e}")
        except Exception as e:
            st.error(f"An unexpected error occurred while saving framework '{name}': {e}")
        return False

    def load_framework_into_session(self, framework_name: str) -> Tuple[Dict[str, PersonaConfig], Dict[str, List[str]], str]:
        """Loads a framework's personas and sets, returning them for session state update."""
        if framework_name in self.all_custom_frameworks_data:
            loaded_config_data = self.all_custom_frameworks_data[framework_name]
            
            # Update internal all_personas with custom framework's personas
            for name, data in loaded_config_data.get('personas', {}).items():
                self.all_personas[name] = PersonaConfig(**data)
            
            # Update internal persona_sets with custom framework's persona_sets
            self.persona_sets.update(loaded_config_data.get('persona_sets', {}))
            
            # Return the specific personas and sets for the selected framework
            current_domain_persona_names = self.persona_sets.get(framework_name, [])
            personas_for_session = {name: self.all_personas[name] for name in current_domain_persona_names if name in self.all_personas}
            
            st.success(f"Loaded custom framework: '{framework_name}'")
            return personas_for_session, {framework_name: current_domain_persona_names}, framework_name
        elif framework_name in self.persona_sets: # It's a default framework
            current_domain_persona_names = self.persona_sets.get(framework_name, [])
            personas_for_session = {name: self.all_personas[name] for name in current_domain_persona_names if name in self.all_personas}
            st.success(f"Loaded default framework: '{framework_name}'")
            return personas_for_session, {framework_name: current_domain_persona_names}, framework_name
        else:
            st.error(f"Framework '{framework_name}' not found.")
            return {}, {}, ""
