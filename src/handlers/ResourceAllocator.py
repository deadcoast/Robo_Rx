"""
A class for managing resource allocation.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List

from src.config.ResourceConfig import ResourceAllocation
from src.config.SystemConfig import SystemConfig
from src.config.TaskChainConfig import TaskChainConfig


@dataclass
class ResourceAllocator:
    """
    Manages allocation of resources to different tasks or processes.

    Attributes:
        available_resources: Dictionary mapping resource IDs to their available amounts
        allocated_resources: Dictionary mapping resource IDs to their allocated amounts
        allocation_history: List of past allocations for auditing purposes
    """

    resource_id: str
    amount: float
    priority: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)
    available_resources: Dict[str, float] = field(default_factory=dict)
    allocated_resources: Dict[str, float] = field(default_factory=dict)
    allocation_history: List[ResourceAllocation] = field(default_factory=list)

    def allocate(self, resource_id: str, amount: float, priority: int = 0) -> bool:
        """
        Allocate a resource if available.

        Args:
            resource_id: ID of the resource to allocate
            amount: Amount of resource to allocate
            priority: Priority of this allocation

        Returns:
            True if allocation was successful, False otherwise
        """
        if resource_id not in self.available_resources:
            return False

        if self.available_resources[resource_id] < amount:
            return False

        self.available_resources[resource_id] -= amount

        if resource_id in self.allocated_resources:
            self.allocated_resources[resource_id] += amount
        else:
            self.allocated_resources[resource_id] = amount

        # Record this allocation
        allocation = ResourceAllocation(
            resource_id=resource_id, amount=amount, priority=priority
        )
        self.allocation_history.append(allocation)

        return True

    def deallocate(self, resource_id: str, amount: float) -> bool:
        """
        Deallocate a previously allocated resource.

        Args:
            resource_id: ID of the resource to deallocate
            amount: Amount of resource to deallocate

        Returns:
            True if deallocation was successful, False otherwise
        """
        if resource_id not in self.allocated_resources:
            return False

        if self.allocated_resources[resource_id] < amount:
            return False

        self.allocated_resources[resource_id] -= amount

        if resource_id in self.available_resources:
            self.available_resources[resource_id] += amount
        else:
            self.available_resources[resource_id] = amount

        return True

    def get_available(self, resource_id: str) -> float:
        """
        Get the amount of available resource.

        Args:
            resource_id: ID of the resource to check

        Returns:
            Amount of resource available
        """
        self._logger.info(f"Getting available resource {resource_id}")
        self._logger.info(
            f"Available resource {resource_id}: {self.available_resources.get(resource_id, 0.0)}"
        )
        self._logger.info("Getting available resource complete")
        # TODO: Generate the Logic for get_available
        return self.available_resources.get(resource_id, 0.0)

    def get_allocated(self, resource_id: str) -> float:
        """
        Get the amount of allocated resource.

        Args:
            resource_id: ID of the resource to check

        Returns:
            Amount of resource allocated
        """
        self._logger.info(f"Getting allocated resource {resource_id}")
        self._logger.info(
            f"Allocated resource {resource_id}: {self.allocated_resources.get(resource_id, 0.0)}"
        )
        self._logger.info("Getting allocated resource complete")
        # TODO: Generate the Logic for get_allocated
        return self.allocated_resources.get(resource_id, 0.0)

    def add_resource(self, resource_id: str, amount: float) -> None:
        """
        Add a new resource or increase an existing one.

        Args:
            resource_id: ID of the resource to add
            amount: Amount of resource to add
        """
        self._logger.info(f"Adding resource {resource_id} with amount {amount}")
        self._logger.info(
            f"Adding resource {resource_id} with amount {amount} complete"
        )
        # TODO: Generate the Logic for add_resource
        if resource_id in self.available_resources:
            self.available_resources[resource_id] += amount
        else:
            self.available_resources[resource_id] = amount

    def __init__(self, config: SystemConfig):
        self.config = config
        self.resource_registry = {}

    def allocate_resources(self, task_chain: TaskChainConfig) -> Dict[str, float]:
        """
        Allocate resources for a task chain.

        Args:
            task_chain: The task chain configuration to allocate resources for

        Returns:
            A dictionary of resource allocations
        """
        self._logger.info(f"Allocating resources for task chain {task_chain}")
        self._logger.info(f"Allocating resources for task chain {task_chain} complete")
        # TODO: Generation the Logic for allocate_resources
        return {}

    def release_resources(self, task_chain: TaskChainConfig) -> None:
        """
        Release resources for a task chain.

        Args:
            task_chain: The task chain configuration to release resources for
        """
        self._logger.info(f"Releasing resources for task chain {task_chain}")
        self._logger.info(f"Releasing resources for task chain {task_chain} complete")
        # TODO: Generation the Logic for release_resources

    def __str__(self) -> str:
        return f"ResourceAllocator(config={self.config}, resource_registry={self.resource_registry})"

    def __repr__(self) -> str:
        """
        Get the string representation of the resource allocator.
        """
        self._logger.info("Getting resource allocator representation")
        self._logger.info(f"Resource allocator: {self}")
        self._logger.info("Resource allocator representation")
        self._logger.info("Getting resource allocator representation complete")
        # TODO: Generate the Logic for __repr__
        return self.__str__()

    def __reduce__(self) -> tuple:
        """
        Get the reduced representation of the resource allocator.
        """
        self._logger.info("Getting resource allocator reduced representation")
        self._logger.info("Getting resource allocator reduced representation complete")
        # TODO: Generate the Logic for __reduce__
        return (ResourceAllocator, (self.config, self.resource_registry))

    def __getstate__(self) -> Dict[str, Any]:
        """
        Get the state of the resource allocator.
        """
        self._logger.info("Getting resource allocator state")
        self._logger.info("Getting resource allocator state complete")
        # TODO: Generate the Logic for __getstate__
        return {"config": self.config, "resource_registry": self.resource_registry}

    def __setstate__(self, state: Dict[str, Any]):
        """
        Set the state of the resource allocator.
        """
        self._logger.info("Setting resource allocator state")
        self._logger.info("Setting resource allocator state complete")
        self.config = state["config"]
        self.resource_registry = state["resource_registry"]
        self.available_resources = state["available_resources"]
        self.allocated_resources = state["allocated_resources"]
        # TODO: Generate the Logic for __setstate__
