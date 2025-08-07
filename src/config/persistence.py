"""
Configuration persistence system that synchronizes UI changes with YAML configuration.
"""

import os
import yaml
from pathlib import Path
from typing import Dict, Any, Optional
from src.models import ReasoningFrameworkConfig, PersonaConfig # Assuming these models exist

class ConfigPersistence:
    """Handles loading and saving of configuration data with proper synchronization."""
    
    def __init__(self, config_dir: str = "config"):
        self.config_dir = Path(config_dir)
        self.personas_file = self.config_dir / "personas.yaml"
        self.user_overrides_file = self.config_dir / "user_overrides.yaml"
        
        # Ensure config directory exists
        self.config_dir.mkdir(exist_ok=True)
        
        # Initialize user overrides if they don't exist
        if not self.user_overrides_file.exists():
            with open(self.user_overrides_file, 'w') as f:
                yaml.dump({"frameworks": {}, "personas": {}}, f)
    
    def load_personas_config(self) -> Dict[str, Any]:
        """Load the complete personas configuration including user overrides."""
        # Load base configuration
        # Ensure personas.yaml exists, create a default if not
        if not self.personas_file.exists():
            with open(self.personas_file, 'w') as f:
                yaml.dump({"frameworks": {}, "personas": []}, f)
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
    
    def save_user_framework(self, framework_name: str, framework_config: Dict[str, Any]):
        """Save a user-defined framework to the overrides file."""
        with open(self.user_overrides_file, 'r') as f:
            user_overrides = yaml.safe_load(f) or {"frameworks": {}, "personas": {}}
        
        # Add or update the framework
        if "frameworks" not in user_overrides:
            user_overrides["frameworks"] = {}
        user_overrides["frameworks"][framework_name] = framework_config
        
        # Save back to file
        with open(self.user_overrides_file, 'w') as f:
            yaml.dump(user_overrides, f, sort_keys=False)
    
    def export_framework_for_sharing(self, framework_name: str) -> str:
        """Export a framework configuration as YAML for sharing."""
        config = self.load_personas_config()
        if framework_name in config.get("frameworks", {}):
            framework_data = config["frameworks"][framework_name]
            # Create a clean export without internal metadata
            export_data = {
                "name": framework_name,
                "description": framework_data.get("description", ""),
                "personas": framework_data.get("personas", [])
            }
            return yaml.dump(export_data, sort_keys=False)
        return ""
