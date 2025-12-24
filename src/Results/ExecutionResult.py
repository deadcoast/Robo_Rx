"""
A class for representing the result of an execution.
"""

import copy
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, Iterator


@dataclass
class ExecutionResult:
    """
    A class for representing the result of an execution.
    """

    success: bool = False
    metrics: Any = None
    error: str = ""
    timestamp: datetime = field(default_factory=datetime.now)

    def __str__(self) -> str:
        """
        Returns a string representation of the object.
        """
        if self.error:
            return f"ExecutionResult(success={self.success}, metrics={self.metrics}, error={self.error}, timestamp={self.timestamp})"
        return f"ExecutionResult(success={self.success}, metrics={self.metrics}, timestamp={self.timestamp})"

    def __repr__(self) -> str:
        """
        Returns a string representation of the object.
        """
        return self.__str__()

    def __reduce__(self) -> tuple:
        """
        Returns a tuple that can be used to recreate the object.
        """
        return (
            ExecutionResult,
            (self.success, self.metrics, self.error, self.timestamp),
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
        Sets the state of the object.
        """
        self.success = state["success"]
        self.metrics = state["metrics"]
        self.error = state["error"]
        self.timestamp = state["timestamp"]

    def __copy__(self) -> "ExecutionResult":
        """
        Returns a shallow copy of the object.
        """
        return ExecutionResult(
            success=self.success,
            metrics=copy.deepcopy(self.metrics),
            error=self.error,
            timestamp=self.timestamp,
        )

    def __deepcopy__(self, memo: Dict[int, Any]) -> "ExecutionResult":
        """
        Returns a deep copy of the object.
        """
        return ExecutionResult(
            success=self.success,
            metrics=copy.deepcopy(self.metrics, memo),
            error=self.error,
            timestamp=self.timestamp,
        )

    def __eq__(self, other: Any) -> bool:
        """
        Returns True if the other object is equal to this object, False otherwise.
        """
        if not isinstance(other, ExecutionResult):
            return False
        return (
            self.success == other.success
            and self.metrics == other.metrics
            and self.error == other.error
            and self.timestamp == other.timestamp
        )

    def __ne__(self, other: Any) -> bool:
        """
        Returns True if the other object is not equal to this object, False otherwise.
        """
        return not self == other

    def __hash__(self) -> int:
        """
        Returns a hash of the object.
        """
        return hash((self.success, self.metrics, self.error, self.timestamp))

    def __bool__(self) -> bool:
        """
        Returns True if the object is successful, False otherwise.
        """
        return self.success

    def __len__(self) -> int:
        """
        Returns the length of the metrics.
        """
        return len(self.metrics)

    def __iter__(self) -> Iterator[Any]:
        """
        Returns an iterator over the metrics.
        """
        return iter(self.metrics)

    def __contains__(self, item: Any) -> bool:
        """
        Returns True if the item is in the metrics, False otherwise.
        """
        return item in self.metrics

    def __getitem__(self, key: Any) -> Any:
        """
        Returns the value at the given key.
        """
        return self.metrics[key]

    def __setitem__(self, key: Any, value: Any) -> None:
        """
        Sets the value at the given key.
        """
        self.metrics[key] = value

    def __delitem__(self, key: Any) -> None:
        """
        Deletes the value at the given key.
        """
        del self.metrics[key]
