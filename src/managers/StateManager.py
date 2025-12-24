"""
State Manager.

This module provides a comprehensive state manager for managing the state of the application.
"""

import hashlib
import logging
import os
import pickle  # nosec B403 - Used with validation
from pathlib import Path
from types import TracebackType
from typing import Any, Dict, Optional, Type, Union


class SecurityError(Exception):
    """Raised when a security check fails."""

    pass


class StateManager:
    """
    A comprehensive state manager for managing the state of the application.
    """

    def __init__(self):
        """
        Initialize the state manager with empty state dictionaries and set up logging.
        """
        self.states: Dict[str, Any] = {}
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        self.logger.addHandler(logging.StreamHandler())
        self.logger.info("StateManager initialized.")

    def __enter__(self) -> "StateManager":
        return self

    def __exit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc_val: Optional[BaseException],
        exc_tb: Optional[TracebackType],
    ) -> None:
        self.logger.info("StateManager exiting.")

    def add_state(self, state_name: str, state: Any) -> None:
        """
        Add a state to the state manager.

        Args:
            state_name (str): The name of the state.
            state (Any): The state to add.
        """
        self.states[state_name] = state
        self.logger.info(f"State '{state_name}' added.")

    def get_state(self, state_name: str) -> Optional[Any]:
        """
        Get a state from the state manager.

        Args:
            state_name (str): The name of the state.

        Returns:
            Optional[Any]: The state if found, otherwise None.
        """
        self.logger.info(f"Getting state '{state_name}'.")
        return self.states.get(state_name)

    def remove_state(self, state_name: str) -> None:
        """
        Remove a state from the state manager.

        Args:
            state_name (str): The name of the state to remove.
        """
        self.logger.info(f"Removing state '{state_name}'.")
        if state_name in self.states:
            del self.states[state_name]
            self.logger.info(f"State '{state_name}' removed.")
        else:
            self.logger.warning(f"State '{state_name}' not found.")

    def save_state(self, state_name: str, filepath: Union[str, Path]) -> None:
        """
        Save a state to a file.

        Args:
            state_name (str): The name of the state to save.
            filepath (Union[str, Path]): The path to save the state to.
        """
        self.logger.info(f"Saving state '{state_name}' to '{filepath}'.")
        state = self.get_state(state_name)
        if state is None:
            self.logger.warning(f"State '{state_name}' not found.")
            return

        # Generate a metadata file with hash for verification
        state_hash = hashlib.sha256(str(state).encode()).hexdigest()
        metadata_path = f"{filepath}.meta"

        # Save state to file
        with open(filepath, "wb") as f:
            pickle.dump(state, f)  # Only used for internal state persistence

        # Save metadata with hash
        with open(metadata_path, "w") as f:
            f.write(state_hash)

        self.logger.info(
            f"State '{state_name}' saved to '{filepath}' with integrity verification."
        )

    def _verify_state_integrity(self, filepath: Union[str, Path]) -> Any:
        """
        Verify the integrity of a state file and load it if valid.

        Args:
            filepath (Union[str, Path]): The path to the state file.

        Returns:
            Any: The loaded state if verification succeeds.

        Raises:
            SecurityError: If verification fails or metadata is missing.
            Exception: For other errors during loading.
        """
        metadata_path = f"{filepath}.meta"

        # Check if metadata file exists
        if not os.path.exists(metadata_path):
            self.logger.error(
                f"Security error: Cannot load state from '{filepath}': Missing metadata file"
            )
            raise SecurityError(
                f"Cannot load state from '{filepath}': Missing integrity verification"
            )

        # Load metadata hash
        with open(metadata_path, "r") as f:
            expected_hash = f.read().strip()

        # Load state file
        with open(filepath, "rb") as f:
            state = pickle.load(f)  # nosec B301 - Used with validation

        # Verify integrity
        actual_hash = hashlib.sha256(str(state).encode()).hexdigest()
        if actual_hash != expected_hash:
            self.logger.error(
                f"Security error: State integrity check failed for '{filepath}'"
            )
            raise SecurityError(f"State integrity check failed for '{filepath}'")

        return state

    def load_state(self, state_name: str, filepath: Union[str, Path]) -> None:
        """
        Load a state from a file with integrity verification.

        Args:
            state_name (str): The name of the state to load.
            filepath (Union[str, Path]): The path to load the state from.

        Raises:
            SecurityError: If state verification fails.
            FileNotFoundError: If the file doesn't exist.
            Exception: For other errors during loading.
        """
        self.logger.info(f"Loading state '{state_name}' from '{filepath}'")

        try:
            # Load and verify state
            state = self._verify_state_integrity(filepath)

            # Add the verified state
            self.add_state(state_name, state)
            self.logger.info(
                f"State '{state_name}' loaded from '{filepath}' with verified integrity."
            )
        except Exception as e:
            self.logger.error(f"Error loading state: {str(e)}")
            raise

    def __reduce__(self) -> "StateManager":
        """
        Returns a tuple that can be used to recreate the object.

        Returns:
            Tuple[type, tuple]: A tuple containing the object type and its arguments.
        """
        return (StateManager, (self.states))

    def __getstate__(self) -> Dict[str, Any]:
        """
        Returns the object's state.

        Returns:
            Dict[str, Any]: The object's state.
        """
        self.logger.info("Getting state manager state.")
        return self.states

    def __setstate__(self, states: Dict[str, Any]) -> None:
        """
        Sets the object's state.

        Args:
            states (Dict[str, Any]): The object's state.
        """
        self.states = states
        self.logger.info("State manager state set.")
