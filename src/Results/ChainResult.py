"""
A class for representing the result of a task chain.
"""

import copy
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict


@dataclass
class ChainResult:
    """
    A class for representing the result of a task chain.
    """

    success: bool = False
    metrics: Any = None
    error: str = ""
    timestamp: datetime = field(default_factory=datetime.now)

    def __str__(self) -> str:
        return f"ChainResult(success={self.success}, metrics={self.metrics}, error={self.error}, timestamp={self.timestamp})"

    def __repr__(self) -> str:
        """
        Returns a string representation of the object.
        """
        return self.__str__()

    def __reduce__(self) -> tuple:
        """
        Returns a tuple that can be used to recreate the object.
        """
        return (ChainResult, (self.success, self.metrics, self.error, self.timestamp))

    def __getstate__(self) -> Dict[str, Any]:
        """
        Returns a dictionary that can be used to recreate the object.
        """
        return {
            "success": self.success,
            "metrics": self.metrics,
            "error": self.error,
            "timestamp": self.timestamp,
        }

    def __setstate__(self, state: Dict[str, Any]):
        """
        Sets the state of the object from a dictionary.
        """
        self.success = state["success"]
        self.metrics = state["metrics"]
        self.error = state["error"]
        self.timestamp = state["timestamp"]

    def __eq__(self, other: Any) -> bool:
        """
        Returns True if the other object is equal to this object, False otherwise.
        """
        if not isinstance(other, ChainResult):
            return False
        return (
            self.success == other.success
            and self.metrics == other.metrics
            and self.error == other.error
            and self.timestamp == other.timestamp
        )

    def __hash__(self) -> int:
        """
        Returns a hash value for the object.
        """
        return hash((self.success, self.metrics, self.error, self.timestamp))

    def __copy__(self) -> "ChainResult":
        """
        Returns a shallow copy of the object.
        """
        return ChainResult(
            success=self.success,
            metrics=copy.deepcopy(self.metrics),
            error=self.error,
            timestamp=self.timestamp,
        )

    def __deepcopy__(self, memo: Dict[int, Any]) -> "ChainResult":
        """
        Returns a deep copy of the object.
        """
        return ChainResult(
            success=self.success,
            metrics=copy.deepcopy(self.metrics, memo),
            error=self.error,
            timestamp=self.timestamp,
        )
