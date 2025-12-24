"""ChainCore class.

This module contains the ChainCore class, which is a class for managing task chains.

"""

from dataclasses import dataclass, field
from datetime import datetime
from queue import PriorityQueue
from typing import Any, Dict, List

# Import the necessary components from the correct locations
from src.ChainResult import ChainResult  # type: ignore

# Import with type ignores for modules lacking py.typed markers
from src.config.SystemConfig import SystemConfig  # type: ignore
from src.ExecutionMonitor import ExecutionMonitor  # type: ignore
from src.ExecutionValidator import ChainValidator  # type: ignore
from src.ResourceManager import ResourceManager  # type: ignore
from src.Results.ExecutionResult import ExecutionResult  # type: ignore


@dataclass
class TaskChainConfig:
    """
    A class for configuring task chains.

    Args:
        chain_id (str): The ID of the task chain.
        priority (int): The priority of the task chain.
        dependencies (List[str]): A list of dependencies for the task chain.
        resource_requirements (Dict[str, float]): A dictionary of resource requirements for the task chain.

    Attributes:
        chain_id (str): The ID of the task chain.
        priority (int): The priority of the task chain.
        dependencies (List[str]): A list of dependencies for the task chain.
        resource_requirements (Dict[str, float]): A dictionary of resource requirements for the task chain.
    """

    chain_id: str
    priority: int
    dependencies: List[str]
    resource_requirements: Dict[str, float]


class ChainManager:
    """
    A class for managing task chains.

    Args:
        self: The instance of the ChainManager.
        config (SystemConfig): The system configuration.

    Attributes:
        active_chains (List[TaskChainConfig]): A list of active task chains.
        execution_queue (PriorityQueue): A priority queue of task chains.
        completion_registry (Dict[str, TaskChainConfig]): A dictionary of completed task chains.

    Methods:
        process_next_chain: Process the next task chain in the queue.
    """

    def __init__(self, config: SystemConfig):
        self.active_chains: List[TaskChainConfig] = []
        self.execution_queue: PriorityQueue[TaskChainConfig] = PriorityQueue()
        self.completion_registry: Dict[str, TaskChainConfig] = {}

    async def process_next_chain(self) -> ChainResult:
        # Use get() without await since PriorityQueue.get() is not a coroutine
        chain = self.execution_queue.get()
        return await self._execute_chain(chain)

    async def _execute_chain(self, chain: TaskChainConfig) -> ChainResult:
        """Execute a chain with the given configuration.

        Args:
            chain: The task chain configuration to execute

        Returns:
            The result of the chain execution
        """
        # Implementation would go here
        executor = ChainExecutor()
        result = await executor.execute_chain(chain)
        return ChainResult(success=result.success, metrics=result.metrics)


@dataclass
class ChainMetrics:
    """
    A class for tracking metrics of a task chain.

    Args:
        self: The instance of the ChainMetrics.
        chain_id (str): The ID of the task chain.
        execution_time (float): The execution time of the task chain.
        task_completion (Dict[str, float]): A dictionary of task completion metrics.
        resource_allocation (Dict[str, float]): A dictionary of resource allocation metrics.
        error_registry (List[str]): A list of error registry.

    Attributes:
        chain_id (str): The ID of the task chain.
        execution_time (float): The execution time of the task chain.
        task_completion (Dict[str, float]): A dictionary of task completion metrics.
        resource_allocation (Dict[str, float]): A dictionary of resource allocation metrics.
        error_registry (List[str]): A list of error registry.
    """

    chain_id: str
    execution_time: float
    task_completion: Dict[str, float]
    resource_allocation: Dict[str, float]
    error_registry: List[str] = field(default_factory=list)
    metrics: Dict[str, Any] = field(default_factory=dict)
    status: str = ""
    timestamp: datetime = field(default_factory=datetime.now)

    def __post_init__(self) -> None:
        self.error_registry = []
        self.task_completion = {}
        self.resource_allocation = {}
        self.metrics = {}
        self.execution_time = 0.0
        self.status = ""
        self.timestamp = datetime.now()
        self.chain_id = ""

    def __str__(self) -> str:
        """
        Returns a string representation of the object.

        Returns:
            str: The string representation of the object.
        """
        # Sort the error_registry list - this is safe as error_registry is a list
        sorted_errors = sorted(
            self.error_registry,
            key=lambda error: error.get("timestamp", ""),
            reverse=False,
        )
        self.error_registry = sorted_errors

        # For dictionaries, we should use sorted items for display purposes
        # but we don't modify the dictionaries themselves with sort methods
        sorted_task_completion = dict(sorted(self.task_completion.items()))
        sorted_resource_allocation = dict(sorted(self.resource_allocation.items()))

        self.status = ""
        self.timestamp = datetime.now()
        return f"ChainMetrics(chain_id={self.chain_id}, execution_time={self.execution_time}, task_completion={sorted_task_completion}, resource_allocation={sorted_resource_allocation}, error_registry={self.error_registry})"

    def __repr__(self) -> str:
        """
        Returns a string representation of the object.

        Returns:
            str: The string representation of the object.
        """
        # Sort error_registry if it's a list
        sorted_errors = sorted(
            self.error_registry,
            key=lambda error: error.get("timestamp", ""),
            reverse=False,
        )
        self.error_registry = sorted_errors
        return str(self)

    def __eq__(self, other: object) -> bool:
        """
            Checks if the object is equal to another ChainMetrics object.

        Returns:
            bool: True if the objects are equal, False otherwise.

        Args:
            self: The object to compare.
            other (ChainMetrics): The other object to compare against.
        """

        if not isinstance(other, ChainMetrics):
            return False
        return (
            self.chain_id == other.chain_id
            and self.execution_time == other.execution_time
            and self.task_completion == other.task_completion
            and self.resource_allocation == other.resource_allocation
            and self.error_registry == other.error_registry
        )

    def __ne__(self, other: object) -> bool:
        """
        Checks if the object is not equal to another ChainMetrics object.

        Returns:
            bool: True if the objects are not equal, False otherwise.
        """
        return not self.__eq__(other)


class ChainExecutor:
    """
    A class for executing task chains.

    Args:
        self: The instance of the ChainExecutor.

    Attributes:
        validator (ChainValidator): The chain validator.
        resource_manager (ResourceManager): The resource manager.
        monitor (ExecutionMonitor): The execution monitor.

    Methods:
        execute_chain: Execute a task chain.
    """

    def __init__(self):
        self.validator = ChainValidator()
        self.resource_manager = ResourceManager()
        self.monitor = ExecutionMonitor()

    async def execute_chain(self, chain_config: TaskChainConfig) -> ExecutionResult:
        """
        Executes the chain with the given chain configuration and returns the execution result.

        Returns:
            ExecutionResult: The result of the chain execution.

        Args:
            self: The current instance of the ChainCore class.
            chain_config (TaskChainConfig): The configuration for the chain execution.

        Raises:
            Exception: If an error occurs during the chain execution.
        """

        try:
            # Initialize chain context
            context = await self._prepare_chain_context(chain_config)

            # Execute with monitoring
            with self.monitor.track_chain():
                result = await self._process_chain(context)

            return ChainResult(
                success=True, metrics=self._compute_chain_metrics(result)
            )

        except Exception as e:
            return ChainResult(success=False, error=str(e))

    async def _prepare_chain_context(
        self, chain_config: TaskChainConfig
    ) -> Dict[str, Any]:
        """Prepare the execution context for a chain.

        Args:
            chain_config: The configuration for the chain

        Returns:
            A dictionary containing the execution context
        """
        # Implementation would go here
        return {"chain_id": chain_config.chain_id}

    async def _process_chain(self, context: Dict[str, Any]) -> Any:
        """Process a chain with the given context.

        Args:
            context: The execution context

        Returns:
            The result of processing the chain
        """
        # Implementation would go here
        return {"processed": True}

    def _compute_chain_metrics(self, result: Any) -> "ChainMetrics":
        """Compute metrics for a chain execution.

        Args:
            result: The result of the chain execution

        Returns:
            Metrics for the chain execution
        """
        # Implementation would go here
        return ChainMetrics(
            chain_id="sample",
            execution_time=0.0,
            task_completion={},
            resource_allocation={},
        )
