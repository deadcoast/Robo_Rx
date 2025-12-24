"""
A class for representing the result of a continuation.
"""

import copy
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict


@dataclass
class ContinuationResult:
    """
    A class for representing the result of a continuation.
    """

    success: bool
    metrics: Dict[str, Any] = field(default_factory=dict)
    error: Any = None
    timestamp: datetime = field(default_factory=datetime.now)

    def __str__(self) -> str:
        return f"ContinuationResult(success={self.success}, metrics={self.metrics}, error={self.error}, timestamp={self.timestamp})"

    def __repr__(self) -> str:
        return f"ContinuationResult(success={self.success}, metrics={self.metrics}, error={self.error}, timestamp={self.timestamp})"

    def __copy__(self) -> "ContinuationResult":
        """
        Returns a shallow copy of the object.
        """
        return ContinuationResult(
            success=self.success,
            metrics=copy.deepcopy(self.metrics),
            error=self.error,
            timestamp=self.timestamp,
        )

    def __deepcopy__(self, memo: Dict[int, Any]) -> "ContinuationResult":
        """
        Returns a deep copy of the object.
        """
        return ContinuationResult(
            success=self.success,
            metrics=copy.deepcopy(self.metrics, memo),
            error=self.error,
            timestamp=self.timestamp,
        )

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
        if not isinstance(other, ContinuationResult):
            return False
        return (
            self.success == other.success
            and self.metrics == other.metrics
            and self.error == other.error
            and self.timestamp == other.timestamp
        )

    def __hash__(self) -> int:
        return hash((self.success, self.metrics, self.error, self.timestamp))

    def __reduce__(self) -> tuple:
        return (
            ContinuationResult,
            (self.success, self.metrics, self.error, self.timestamp),
        )

    def __reduce_ex__(self, protocol: int) -> tuple:
        return self.__reduce__()
