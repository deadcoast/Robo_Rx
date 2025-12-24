"""StateValidationError class."""

from dataclasses import dataclass, field
from typing import Any, Dict


@dataclass
class StateValidationError(Exception):
    """StateValidationError class."""

    message: str = ""
    details: Dict[str, Any] = field(default_factory=dict)

    def __init__(self, message: str, details: Dict[str, Any] = None):
        self.message = message
        self.details = details or {}
        super().__init__(self.message)

    def __str__(self) -> str:
        """
        Returns a string representation of the error.

        Returns:
            str: A string representation of the error.
        """
        return f"{self.message} ({self.details})"

    def __repr__(self) -> str:
        """
        Returns a string representation of the error.

        Returns:
            str: A string representation of the error.
        """
        return f"StateValidationError(message={self.message}, details={self.details})"

    def __reduce__(self) -> "StateValidationError":
        """
        Returns a tuple that can be used to recreate the error.

        Returns:
            tuple: A tuple that can be used to recreate the error.
        """
        return (StateValidationError, (self.message, self.details))

    def __getstate__(self) -> Dict[str, Any]:
        """
        Returns the state of the error.

        Returns:
            dict: The state of the error.
        """
        return {"message": self.message, "details": self.details}

    def __setstate__(self, state: Dict[str, Any]) -> None:
        """
        Sets the state of the error.

        Args:
            state (dict): The state of the error.
        """
        self.message = state["message"]
        self.details = state["details"]
        super().__init__(self.message)

    def __eq__(self, other: object) -> bool:
        """
        Returns a boolean indicating whether two errors are equal.

        Args:
            other (object): The other object to compare.

        Returns:
            bool: True if the errors are equal, False otherwise.
        """
        if not isinstance(other, StateValidationError):
            return False
        return self.message == other.message and self.details == other.details

    def __hash__(self) -> int:
        """
        Returns a hash value for the error.

        Returns:
            int: A hash value for the error.
        """
        return hash((self.message, self.details))
