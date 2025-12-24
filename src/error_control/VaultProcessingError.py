"""
A custom exception class for vault processing errors.
"""

from dataclasses import dataclass, field
from typing import Any, Dict


@dataclass
class VaultProcessingError(Exception):
    """VaultProcessingError class."""

    message: str = ""
    details: Dict[str, Any] = field(default_factory=dict)

    def __init__(self, message: str, details: Dict[str, Any] = None):
        """
        Initializes a new instance of the VaultProcessingError class.

        Args:
            message (str): The error message.
            details (Dict[str, Any], optional): Additional details about the error.
        """
        super().__init__(message)
        self.message = message
        self.details = details

    def __str__(self) -> str:
        """
        Returns a string representation of the error.

        Returns:
            str: A string representation of the error.
        """
        return f"{self.message} - {self.details}"

    def __repr__(self) -> str:
        """
        Returns a string representation of the error.

        Returns:
            str: A string representation of the error.
        """
        return f"{self.message} - {self.details}"

    def __eq__(self, other: Any) -> bool:
        """
        Returns a boolean indicating whether two errors are equal.

        Args:
            other (Any): The other object to compare.

        Returns:
            bool: True if the errors are equal, False otherwise.
        """
        if isinstance(other, VaultProcessingError):
            return self.message == other.message and self.details == other.details
        return False

    def __hash__(self) -> int:
        """
        Returns a hash value for the error.

        Returns:
            int: A hash value for the error.
        """
        return hash((self.message, self.details))
