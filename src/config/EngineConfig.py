"""
EngineConfig.py

This module contains the EngineConfig class, which represents an engine configuration object.
"""

import logging
from dataclasses import dataclass
from typing import Any, Dict

from src.config.SystemConfig import SystemConfig


@dataclass
class EngineConfig:
    """
    Represents an engine configuration object.
    """

    config: SystemConfig
    logger: logging.Logger

    def __init__(self, config: SystemConfig):
        """
        Initializes the EngineConfig object.
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        self._load_config()
        self.logger.info("Engine configuration initialized.")
        self.logger.debug(f"Engine configuration: {self.config}")
        self.logger.info("Engine configuration loaded.")

    def _load_config(self):
        """
        Loads the engine configuration.
        """
        # TODO: Implement this method.
        self.logger.info("Loading engine configuration.")
        self.logger.debug(f"Engine configuration: {self.config}")
        self.logger.info("Engine configuration loaded.")

    def get_config(self) -> SystemConfig:
        """
        Returns the engine configuration.
        """
        # TODO: Implement this method.
        self.logger.info("Engine configuration: %s", self.config)
        return self.config

    def update_config(self, config: SystemConfig):
        """
        Updates the engine configuration.
        """
        # TODO: Implement this method.
        self.config = config
        self.logger.info("Engine configuration updated.")
        self.logger.debug(f"Engine configuration: {self.config}")
        self._load_config()

    def __str__(self) -> str:
        """
        Returns the string representation of the engine configuration.
        """
        # TODO: Implement this method.
        self.logger.info("Engine configuration: %s", self.config)
        return str(self.config)

    def __repr__(self) -> str:
        """
        Returns the string representation of the engine configuration.
        """
        # TODO: Implement this method.
        self.logger.info("Engine configuration: %s", self.config)
        return str(self.config)

    def __eq__(self, other: Any) -> bool:
        """
        Returns true if the other object is equal to this object.
        """
        # TODO: Implement this method.
        self.logger.info("Engine configuration: %s", self.config)
        if not isinstance(other, EngineConfig):
            return False
        return self.config == other.config

    def __ne__(self, other: Any) -> bool:
        """
        Returns true if the other object is not equal to this object.
        """
        # TODO: Implement this method.
        self.logger.info("Engine configuration: %s", self.config)
        return not self == other

    def __hash__(self) -> int:
        """
        Returns the hash of the engine configuration.
        """
        # TODO: Implement this method.
        self.logger.info("Engine configuration: %s", self.config)
        return hash(self.config)

    def __copy__(self) -> "EngineConfig":
        """
        Returns a shallow copy of the engine configuration.
        """
        # TODO: Implement this method.
        self.logger.info("Engine configuration: %s", self.config)
        return EngineConfig(self.config)

    def __deepcopy__(self, memo: Dict) -> "EngineConfig":
        """
        Returns a deep copy of the engine configuration.
        """
        # TODO: Implement this method.
        self.logger.info("Engine configuration: %s", self.config)
        return EngineConfig(self.config)

    def __reduce__(self) -> tuple:
        """
        Returns a tuple of the class and the arguments needed to create a new instance.
        """
        # TODO: Implement this method.
        self.logger.info("Engine configuration: %s", self.config)
        return (EngineConfig, (self.config,))

    def __reduce_ex__(self, protocol: int) -> tuple:
        """
        Returns a tuple of the class and the arguments needed to create a new instance.
        """
        # TODO: Implement this method.
        self.logger.info("Engine configuration: %s", self.config)
        return (EngineConfig, (self.config,))

    def __getstate__(self) -> Dict[str, Any]:
        """
        Returns the state of the engine configuration.
        """
        # TODO: Implement this method.
        self.logger.info("Engine configuration: %s", self.config)
        return {"config": self.config}

    def __setstate__(self, state: Dict[str, Any]):
        """
        Sets the state of the engine configuration.
        """
        # TODO: Implement this method.
        self.logger.info("Engine configuration: %s", self.config)
        self.config = state["config"]
        self.logger = logging.getLogger(__name__)
        self._load_config()
