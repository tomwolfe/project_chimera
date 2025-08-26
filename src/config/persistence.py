# src/config/persistence.py
"""
Configuration persistence system that synchronizes UI changes with YAML configuration.
Handles loading and saving of default and custom persona/framework configurations.
"""

import os
import yaml
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
from src.models import ReasoningFrameworkConfig, PersonaConfig # Assuming these models exist
import json # Added for JSON operations
import re # Added for sanitization
import logging

logger = logging.getLogger(__name__)

# Define custom frameworks directory as a Path object
CUSTOM_FRAMEWORKS_PATH = Path("custom_frameworks")

class ConfigPersistence:
    """Handles loading and saving of configuration data with proper synchronization."""
    
    def __init__(self, config_dir: str = "config"):
        self.config_dir = Path(config_dir)
        self.personas_file = self.config_dir / "personas.yaml"
        self.user_overrides_file = self.config_dir / "user_overrides.yaml"
        self.custom_frameworks_dir = self.config_dir / CUSTOM_FRAMEWORKS_PATH # Use the Path object relative to config_dir
        
        # Ensure config directory and custom frameworks directory exist
        self.config_dir.mkdir(exist_ok=True)
        self.custom_frameworks_dir.mkdir(parents=True, exist_ok=True) # Use parents=True for nested creation
        
        # Initialize user overrides if they don't exist
        if not self.user_overrides_file.exists():
            with open(self.user_overrides_file, 'w') as f:
                yaml.dump({"frameworks": {}, "personas": {}}, f)
    
    def load_personas_config(self) -> Dict[str, Any]:
        """Load the complete personas configuration including user overrides."""
        # Load base configuration
        # Ensure personas.yaml exists, create a default if not
        if not self.personas_file.exists():
            logger.warning(f"Default personas file not found at '{self.personas_file}'. Creating a minimal default.")
            with open(self.personas_file, 'w') as f:
                yaml.dump({"personas": [], "persona_sets": {"General": []}}, f)
        with open(self.personas_file, 'r') as f:
            base_config = yaml.safe_load(f) or {}
        
        # Load user overrides
        with open(self.user_overrides_file, 'r') as f:
            user_overrides = yaml.safe_load(f) or {"frameworks": {}, "personas": {}}
        
        # Merge configurations (user overrides take precedence)
        return self._merge_configs(base_config, user_overrides)
    
    def _merge_configs(self, base: Dict[str, Any], overrides: Dict[str, Any]) -> Dict[str, Any]:
        """Merge base configuration with user overrides."""
        # Deep copy base config
        result = {**base}
        
        # Merge frameworks
        if "frameworks" in overrides:
            if "frameworks" not in result:
                result["frameworks"] = {}
            for framework_name, framework_data in overrides["frameworks"].items():
                if framework_name in result["frameworks"]:
                    # Update existing framework
                    result["frameworks"][framework_name] = {
                        **result["frameworks"][framework_name],
                        **framework_data
                    }
                else:
                    # Add new framework
                    result["frameworks"][framework_name] = framework_data
        
        # Merge personas
        if "personas" in overrides:
            if "personas" not in result:
                result["personas"] = []
            # Convert to dict for easier merging
            persona_dict = {p["name"]: p for p in result.get("personas", [])}
            for persona_data in overrides["personas"]:
                persona_dict[persona_data["name"]] = persona_data
            result["personas"] = list(persona_dict.values())
        
        return result
    
    def _sanitize_framework_filename(self, name: str) -> str:
        """Sanitizes a string to be used as a filename for custom frameworks."""
        sanitized = re.sub(r'[<>:"/\\|?*\s]+', '_', name)
        sanitized = re.sub(r'^[^a-zA-Z0-9_]+|[^a-zA-Z0-9_]+$', '', sanitized)
        if not sanitized:
            sanitized = "unnamed_framework"
        return sanitized

    def _get_filepath_for_framework(self, framework_name: str) -> Path:
        """Returns the full file path for a given custom framework name."""
        sanitized_name = self._sanitize_framework_filename(framework_name)
        return self.custom_frameworks_dir / f"{sanitized_name}.json"

    def _get_saved_custom_framework_names(self) -> List[str]:
        """Returns a list of actual framework names found on disk by reading their files."""
        framework_names = []
        try:
            for filename in os.listdir(self.custom_frameworks_dir):
                if filename.endswith(".json"):
                    filepath = self.custom_frameworks_dir / filename
                    try:
                        with open(filepath, 'r', encoding='utf-8') as f:
                            config_data = json.load(f)
                            # Prefer 'framework_name' key, fallback to 'name', then filename stem
                            actual_name = config_data.get("framework_name") or config_data.get("name") or os.path.splitext(filename)[0]
                            framework_names.append(actual_name)
                    except (json.JSONDecodeError, OSError) as e:
                        logger.warning(f"Could not read or parse custom framework file {filename}: {e}")
            return sorted(list(set(framework_names))) # Ensure uniqueness and sort
        except OSError as e:
            logger.error(f"Error listing custom frameworks in {self.custom_frameworks_dir}: {e}")
            return []

    def _load_custom_framework_config_from_file(self, framework_name: str) -> Optional[Dict[str, Any]]:
        """Loads a specific custom framework configuration from a JSON file."""
        filepath = self._get_filepath_for_framework(framework_name)
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                config_data = json.load(f)
            return config_data
        except (FileNotFoundError, json.JSONDecodeError, OSError) as e:
            logger.error(f"Error loading custom framework '{framework_name}' from '{filepath}': {e}")
            return None

    def save_user_framework(self, framework_name: str, framework_data: Dict[str, Any]) -> Tuple[bool, str]:
        """Save a user-defined framework to a dedicated JSON file."""
        filepath = self._get_filepath_for_framework(framework_name)
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(framework_data, f, indent=2)
            return True, f"Framework '{framework_name}' saved successfully to '{filepath}'!"
        except OSError as e:
            logger.error(f"Error saving framework '{framework_name}' to '{filepath}': {e}")
            return False, f"Error saving framework '{framework_name}' to '{filepath}': {e}"
        except Exception as e:
            logger.exception(f"An unexpected error occurred while saving framework '{framework_name}': {e}")
            return False, f"An unexpected error occurred while saving framework '{framework_name}': {e}"

    def export_framework_for_sharing(self, framework_name: str) -> Optional[str]:
        """Export a framework configuration as YAML for sharing."""
        framework_data = self._load_custom_framework_config_from_file(framework_name)
        if framework_data:
            # For sharing, YAML is often preferred for readability.
            return yaml.dump(framework_data, sort_keys=False)
        return None

    def import_framework_from_file(self, file_content: str, original_filename: str) -> Tuple[bool, str, Optional[Dict[str, Any]]]:
        """
        Imports a framework configuration from a YAML/JSON string.
        Returns (success, message, loaded_config_data).
        """
        try:
            # Attempt to load as YAML first, then JSON
            try:
                config_data = yaml.safe_load(file_content)
            except yaml.YAMLError:
                config_data = json.loads(file_content)

            if not isinstance(config_data, dict):
                return False, "Invalid framework file format: Expected a dictionary.", None
            
            framework_name = config_data.get("framework_name") or config_data.get("name")
            if not framework_name:
                return False, "Framework name not found in the file. Please ensure 'framework_name' or 'name' is present.", None

            # Validate basic structure (optional, but good practice)
            if "personas" not in config_data or "persona_sets" not in config_data:
                return False, "Invalid framework structure: Missing 'personas' or 'persona_sets' section.", None

            # Save to custom frameworks directory using the framework_name from the file
            success, message = self.save_user_framework(framework_name, config_data)
            if success:
                return True, f"Framework '{framework_name}' imported and saved successfully.", config_data
            else:
                return False, message, None
        except (yaml.YAMLError, json.JSONDecodeError) as e:
            logger.error(f"Error parsing framework file '{original_filename}': {e}")
            return False, f"Error parsing framework file '{original_filename}': {e}. Please ensure it's valid YAML or JSON.", None
        except Exception as e:
            logger.exception(f"An unexpected error occurred during import of '{original_filename}': {e}")
            return False, f"An unexpected error occurred during import: {e}", None