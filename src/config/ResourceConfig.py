"""
A configuration class for resource management.
"""

from dataclasses import dataclass, field
from typing import Any, Dict


@dataclass
class ResourceConfig:
    """
    A configuration class for resource management.
    """

    resource_id: str
    amount: float
    priority: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __str__(self) -> str:
        """
        Returns a string representation of the ResourceConfig object.

        Returns:
            str: A string representation of the ResourceConfig object.

        Raises:
            None
        """

        return f"ResourceConfig(resource_id={self.resource_id}, amount={self.amount}, priority={self.priority}, metadata={self.metadata})"

    def __reduce__(self) -> tuple:
        """
        Returns a tuple that can be used to reconstruct the ResourceConfig object.

        Returns:
            tuple: A tuple containing the class constructor and the arguments needed to reconstruct the ResourceConfig object.

        Raises:
            None
        """

        return (
            ResourceConfig,
            (self.resource_id, self.amount, self.priority, self.metadata),
        )

    def __getstate__(self) -> Dict[str, Any]:
        """
        Returns a dictionary representing the state of the ResourceConfig object.

        Returns:
            Dict[str, Any]: A dictionary containing the following key-value pairs:
                - "resource_id" (Any): The resource ID.
                - "amount" (Any): The amount of the resource.
                - "priority" (Any): The priority of the resource.
                - "metadata" (Any): The metadata associated with the resource.

        Raises:
            None
        """

        return {
            "resource_id": self.resource_id,
            "amount": self.amount,
            "priority": self.priority,
            "metadata": self.metadata,
        }

    def __setstate__(self, state: Dict[str, Any]):
        """
        Sets the state of the ResourceConfig object based on the provided dictionary.

        Args:
            state (Dict[str, Any]): A dictionary containing the state information of the ResourceConfig object.

        Returns:
            None

        Raises:
            None
        """

        self.resource_id = state["resource_id"]
        self.amount = state["amount"]
        self.priority = state["priority"]
        self.metadata = state["metadata"]

    def __eq__(self, other: Any) -> bool:
        """
        Compares the ResourceConfig object with another object for equality.

        Args:
            other (Any): The object to compare with.

        Returns:
            bool: True if the objects are equal, False otherwise.

        Raises:
            None
        """

        if not isinstance(other, ResourceConfig):
            return False
        return (
            self.resource_id == other.resource_id
            and self.amount == other.amount
            and self.priority == other.priority
            and self.metadata == other.metadata
        )

    def __ne__(self, other: Any) -> bool:
        """
        Compares the ResourceConfig object with another object for inequality.

        Args:
            other (Any): The object to compare with.

        Returns:
            bool: True if the objects are not equal, False otherwise.

        Raises:
            None
        """

        return not self == other

    def __hash__(self) -> int:
        """
        Returns the hash value of the ResourceConfig object.

        Returns:
            int: The hash value of the ResourceConfig object.

        Raises:
            None
        """

        return hash((self.resource_id, self.amount, self.priority, self.metadata))

    def __copy__(self) -> "ResourceConfig":
        """
        Creates and returns a copy of the ResourceConfig object.

        Returns:
            ResourceConfig: A copy of the ResourceConfig object.

        Raises:
            None
        """

        return ResourceConfig(
            resource_id=self.resource_id,
            amount=self.amount,
            priority=self.priority,
            metadata=self.metadata.copy(),
        )

    def __deepcopy__(self, memo: Dict[int, Any]) -> "ResourceConfig":
        """
        Creates and returns a deep copy of the ResourceConfig object.

        Args:
            memo (Dict[int, Any]): A dictionary used by the deepcopy implementation.

        Returns:
            ResourceConfig: A deep copy of the ResourceConfig object.

        Raises:
            None
        """

        return ResourceConfig(
            resource_id=self.resource_id,
            amount=self.amount,
            priority=self.priority,
            metadata=self.metadata.copy(),
        )


@dataclass
class ResourceAllocation:
    """
    Represents a resource allocation.

    Args:
        resource_id: The ID of the resource being allocated
        amount: The amount of the resource being allocated
        priority: The priority of this allocation
        metadata: Additional metadata for this allocation
    """

    resource_id: str
    amount: float
    priority: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)
