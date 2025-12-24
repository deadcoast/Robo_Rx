"""TaskChainConfig class."""

from dataclasses import dataclass, field
from typing import Any, Dict


@dataclass
class TaskChainConfig:
    """A configuration class for task chains."""

    task_chain_id: str
    task_chain_name: str
    task_chain_description: str
    task_chain_priority: int
    task_chain_metadata: Dict[str, Any] = field(default_factory=dict)

    def __str__(self) -> str:
        """
        Returns a string representation of the TaskChainConfig object.

        Returns:
            str: A string representation of the TaskChainConfig object.

        Raises:
            None
        """
        return f"TaskChainConfig(task_chain_id={self.task_chain_id}, task_chain_name={self.task_chain_name}, task_chain_description={self.task_chain_description}, task_chain_priority={self.task_chain_priority}, task_chain_metadata={self.task_chain_metadata})"

    def __reduce__(self) -> tuple:
        """
        Returns a tuple that can be used to reconstruct the TaskChainConfig object.

        Returns:
            tuple: A tuple containing the class constructor and the arguments needed to reconstruct the TaskChainConfig object.

        Raises:
            None
        """
        return (
            TaskChainConfig,
            (
                self.task_chain_id,
                self.task_chain_name,
                self.task_chain_description,
                self.task_chain_priority,
                self.task_chain_metadata,
            ),
        )

    def __getstate__(self) -> Dict[str, Any]:
        """
        Returns a dictionary representing the state of the TaskChainConfig object.

        Returns:
            Dict[str, Any]: A dictionary containing the following key-value pairs:
                - "task_chain_id" (Any): The task chain ID.
                - "task_chain_name" (Any): The task chain name.
                - "task_chain_description" (Any): The task chain description.
                - "task_chain_priority" (Any): The task chain priority.
                - "task_chain_metadata" (Any): The metadata associated with the task chain.

        Raises:
            None
        """
        return {
            "task_chain_id": self.task_chain_id,
            "task_chain_name": self.task_chain_name,
            "task_chain_description": self.task_chain_description,
            "task_chain_priority": self.task_chain_priority,
            "task_chain_metadata": self.task_chain_metadata,
        }

    def __setstate__(self, state: Dict[str, Any]):
        """
        Sets the state of the TaskChainConfig object based on the provided dictionary.

        Args:
            state (Dict[str, Any]): A dictionary containing the state information of the TaskChainConfig object.

        Returns:
            None

        Raises:
            None
        """
        self.task_chain_id = state["task_chain_id"]
        self.task_chain_name = state["task_chain_name"]
        self.task_chain_description = state["task_chain_description"]
        self.task_chain_priority = state["task_chain_priority"]
        self.task_chain_metadata = state["task_chain_metadata"]

    def __eq__(self, other: Any) -> bool:
        """
        Compares the TaskChainConfig object with another object for equality.

        Args:
            other (Any): The object to compare with.

        Returns:
            bool: True if the objects are equal, False otherwise.

        Raises:
            None
        """
        if not isinstance(other, TaskChainConfig):
            return False
        return (
            self.task_chain_id == other.task_chain_id
            and self.task_chain_name == other.task_chain_name
            and self.task_chain_description == other.task_chain_description
            and self.task_chain_priority == other.task_chain_priority
            and self.task_chain_metadata == other.task_chain_metadata
        )

    def __ne__(self, other: Any) -> bool:
        """
        Compares the TaskChainConfig object with another object for inequality.

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
        Returns the hash value of the TaskChainConfig object.

        Returns:
            int: The hash value of the TaskChainConfig object.

        Raises:
            None
        """
        return hash(
            (
                self.task_chain_id,
                self.task_chain_name,
                self.task_chain_description,
                self.task_chain_priority,
                self.task_chain_metadata,
            )
        )

    def __copy__(self) -> "TaskChainConfig":
        """
        Creates and returns a copy of the TaskChainConfig object.

        Returns:
            TaskChainConfig: A copy of the TaskChainConfig object.

        Raises:
            None
        """
        return TaskChainConfig(
            task_chain_id=self.task_chain_id,
            task_chain_name=self.task_chain_name,
            task_chain_description=self.task_chain_description,
            task_chain_priority=self.task_chain_priority,
            task_chain_metadata=self.task_chain_metadata.copy(),
        )

    def __deepcopy__(self, memo: Dict[int, Any]) -> "TaskChainConfig":
        """
        Creates and returns a deep copy of the TaskChainConfig object.

        Args:
            memo (Dict[int, Any]): A dictionary used by the deepcopy implementation.

        Returns:
            TaskChainConfig: A deep copy of the TaskChainConfig object.

        Raises:
            None
        """
        return TaskChainConfig(
            task_chain_id=self.task_chain_id,
            task_chain_name=self.task_chain_name,
            task_chain_description=self.task_chain_description,
            task_chain_priority=self.task_chain_priority,
            task_chain_metadata=self.task_chain_metadata.copy(),
        )
