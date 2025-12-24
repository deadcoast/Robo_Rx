"""
A custom exception class for enhanced processing errors.
"""

from dataclasses import dataclass, field
from typing import Any, Dict


@dataclass
class EnhancedProcessingError(Exception):
    """
    A custom exception class for enhanced processing errors.
    """

    message: str = ""
    details: Dict[str, Any] = field(default_factory=dict)

    def __init__(self, message: str, details: Dict[str, Any] = None):
        """
        Initialize the EnhancedProcessingError with a message and optional details.
        """
        super().__init__(message)
        self.message = message
        self.details = details

    def __str__(self) -> str:
        """
        Return a string representation of the error.
        """
        return f"{self.message} - {self.details}"

    def __repr__(self) -> str:
        """
        Return a string representation of the error.
        """
        return f"{self.message} - {self.details}"

    def __eq__(self, other: Any) -> bool:
        """
        Compare two EnhancedProcessingError objects for equality.
        """
        if isinstance(other, EnhancedProcessingError):
            return self.message == other.message and self.details == other.details
        return False

    def __hash__(self) -> int:
        """
        Return a hash of the error.
        """
        return hash((self.message, self.details))

    def __lt__(self, other: Any) -> bool:
        """
        Compare two EnhancedProcessingError objects for less than.
        """
        if isinstance(other, EnhancedProcessingError):
            return self.message < other.message
        return False

    def __le__(self, other: Any) -> bool:
        """
        Compare two EnhancedProcessingError objects for less than or equal to.
        """
        if isinstance(other, EnhancedProcessingError):
            return self.message <= other.message
        return False

    def __gt__(self, other: Any) -> bool:
        """
        Compare two EnhancedProcessingError objects for greater than.
        """
        if isinstance(other, EnhancedProcessingError):
            return self.message > other.message
        return False

    def __ge__(self, other: Any) -> bool:
        """
        Compare two EnhancedProcessingError objects for greater than or equal to.
        """
        if isinstance(other, EnhancedProcessingError):
            return self.message >= other.message
        return False

    def __ne__(self, other: Any) -> bool:
        """
        Compare two EnhancedProcessingError objects for inequality.
        """
        return not self == other
