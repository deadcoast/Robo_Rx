"""
A class for managing resource allocation.
"""

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple

from src.config.ResourceConfig import ResourceAllocation
from src.config.SystemConfig import SystemConfig
from src.config.TaskChainConfig import TaskChainConfig


@dataclass
class ResourceAllocator:
    """
    Manages allocation of resources to different tasks or processes.

    Attributes:
        config: System configuration
        resource_registry: Dictionary mapping resource IDs to their registered amounts
        available_resources: Dictionary mapping resource IDs to their available amounts
        allocated_resources: Dictionary mapping resource IDs to their allocated amounts
        allocation_history: List of past allocations for auditing purposes
    """

    config: SystemConfig
    resource_registry: Dict[str, float] = field(default_factory=dict)
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
        allocation = ResourceAllocation(resource_id, amount, priority)
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
        return self.available_resources.get(resource_id, 0.0)

    def get_allocated(self, resource_id: str) -> float:
        """
        Get the amount of allocated resource.

        Args:
            resource_id: ID of the resource to check

        Returns:
            Amount of resource allocated
        """
        return self.allocated_resources.get(resource_id, 0.0)

    def add_resource(self, resource_id: str, amount: float) -> None:
        """
        Add a new resource or increase an existing one.

        Args:
            resource_id: ID of the resource to add
            amount: Amount of resource to add
        """
        if resource_id in self.available_resources:
            self.available_resources[resource_id] += amount
        else:
            self.available_resources[resource_id] = amount

    def __post_init__(self):
        """Set up logger after dataclass initialization"""
        self.logger = logging.getLogger(__name__)

    def allocate_resources(self, task_chain: TaskChainConfig) -> Dict[str, float]:
        """
        Allocate resources for a task chain.

        Args:
            task_chain: The task chain configuration to allocate resources for

        Returns:
            TaskChainConfig
        """
        self.logger.info(f"Allocating resources for task chain {task_chain}")
        self.logger.info(f"Available resources: {self.available_resources}")
        self.logger.info(f"Allocated resources: {self.allocated_resources}")
        self.logger.info(f"Allocation history: {self.allocation_history}")
        # TODO: Implementation would go here

        return {}

    def release_resources(self, task_chain: TaskChainConfig) -> None:
        """
        Release resources for a task chain.

        Args:
            task_chain: The task chain configuration to release resources for
        """
        self.logger.info(f"Releasing resources for task chain {task_chain}")
        self.logger.info(f"Available resources: {self.available_resources}")
        self.logger.info(f"Allocated resources: {self.allocated_resources}")
        self.logger.info(f"Allocation history: {self.allocation_history}")
        # TODO: Implementation would go here

    def __str__(self) -> str:
        """
        Returns a string representation of the object.
        """
        return f"ResourceAllocator(config={self.config}, resource_registry={self.resource_registry})"

    def __reduce__(self) -> Tuple:
        """
        Returns a tuple that can be used to recreate the object.
        """
        return (ResourceAllocator, (self.config, self.resource_registry))

    def __setstate__(self, state: Dict[str, Any]):
        """
        Sets the state of the object from a dictionary.
        """
        self.config = state["config"]
        self.resource_registry = state["resource_registry"]

    def __getstate__(self) -> Dict[str, Any]:
        """
        Returns a dictionary that can be used to recreate the object.
        """
        return {"config": self.config, "resource_registry": self.resource_registry}
