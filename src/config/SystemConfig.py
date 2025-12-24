"""SystemConfig.py

This module contains the SystemConfig class, which represents a system configuration object.
"""

import logging
from typing import Any, Dict


class SystemConfig:
    """
    Represents a system configuration object.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initializes the SystemConfig with a configuration dictionary.
        """
        # Validate the configuration
        if not isinstance(config, dict):
            raise ValueError("Configuration must be a dictionary.")
        self.config = config
        self.logger = logging.getLogger("SystemConfig")
        self._load_config()

    def _load_config(self):
        """
        Loads the system configuration.
        """
        # Validate the configuration
        if not isinstance(self.config, dict):
            raise ValueError("Configuration must be a dictionary.")

        self.logger.info("Loading system configuration.")
        self.logger.debug(f"System configuration: {self.config}")
        self.logger.info("System configuration loaded.")

    def get_config(self) -> Dict[str, Any]:
        """
        Returns the system configuration.
        """
        return self.config

    def update_config(self, config: Dict[str, Any]):
        """
        Updates the system configuration.
        """
        # Validate the configuration
        if not isinstance(config, dict):
            raise ValueError("Configuration must be a dictionary.")
        self.config = config
        self.logger.info("System configuration updated.")
        self.logger.debug(f"System configuration: {self.config}")
        self._load_config()

    def __str__(self) -> str:
        """
        Returns the string representation of the system configuration.
        """
        # Validate the configuration
        if not isinstance(self.config, dict):
            raise ValueError("Configuration must be a dictionary.")
        return str(self.config)

    def __repr__(self) -> str:
        """
        Returns the string representation of the system configuration.
        """
        # Validate the configuration
        if not isinstance(self.config, dict):
            raise ValueError("Configuration must be a dictionary.")
        return str(self.config)

    def __eq__(self, other: Any) -> bool:
        """
        Returns true if the other object is equal to this object.
        """
        # Validate the configuration
        if not isinstance(other, SystemConfig):
            return False
        return self.config == other.config

    def __ne__(self, other: Any) -> bool:
        """
        Returns true if the other object is not equal to this object.
        """
        # Validate the configuration
        return not self == other if isinstance(other, SystemConfig) else True

    def __hash__(self) -> int:
        """
        Returns the hash of the system configuration.
        """
        # Validate the configuration
        if not isinstance(self.config, dict):
            raise ValueError("Configuration must be a dictionary.")
        return hash(self.config)

    def __copy__(self) -> "SystemConfig":
        """
        Returns a copy of the system configuration.
        """
        # Validate the configuration
        if not isinstance(self.config, dict):
            raise ValueError("Configuration must be a dictionary.")
        return SystemConfig(self.config)

    def __deepcopy__(self, memo: Dict) -> "SystemConfig":
        """
        Returns a deep copy of the system configuration.
        """
        # Validate the configuration
        if not isinstance(self.config, dict):
            raise ValueError("Configuration must be a dictionary.")
        return SystemConfig(self.config)

    def __reduce__(self) -> tuple:
        """
        Returns a tuple that can be used to recreate the object.
        """
        # Validate the configuration
        if not isinstance(self.config, dict):
            raise ValueError("Configuration must be a dictionary.")
        return (SystemConfig, (self.config,))

    def __getstate__(self) -> Dict[str, Any]:
        """
        Returns the state of the object.
        """
        # Validate the configuration
        if not isinstance(self.config, dict):
            raise ValueError("Configuration must be a dictionary.")
        return {"config": self.config}

    def __setstate__(self, state: Dict[str, Any]):
        """
        Sets the state of the object.
        """
        # Validate the configuration
        if not isinstance(state, dict):
            raise ValueError("State must be a dictionary.")
        self.config = state["config"]
        self.logger = logging.getLogger("SystemConfig")
        self._load_config()
