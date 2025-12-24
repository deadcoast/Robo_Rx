from datetime import datetime
from unittest.mock import MagicMock, Mock, patch

import pytest

from src.ChainResult import ChainResult
from src.core.ChainCore import (
    ChainExecutor,
    ChainManager,
    ChainMetrics,
    TaskChainConfig,
)
from src.Results.ExecutionResult import ExecutionResult


class TestTaskChainConfig:
    """Test suite for the TaskChainConfig class."""

    @pytest.fixture
    def task_chain_config(self):
        """Create a TaskChainConfig instance for testing."""
        return TaskChainConfig(
            chain_id="test-chain",
            priority=5,
            dependencies=["dep1", "dep2"],
            resource_requirements={"cpu": 2.0, "memory": 4.0},
        )

    def test_init(self, task_chain_config):
        """Test initialization of TaskChainConfig."""
        assert task_chain_config.chain_id == "test-chain"
        assert task_chain_config.priority == 5
        assert task_chain_config.dependencies == ["dep1", "dep2"]
        assert task_chain_config.resource_requirements == {"cpu": 2.0, "memory": 4.0}

    def test_equality(self):
        """Test equality of TaskChainConfig instances."""
        config1 = TaskChainConfig(
            chain_id="test-chain",
            priority=5,
            dependencies=["dep1", "dep2"],
            resource_requirements={"cpu": 2.0, "memory": 4.0},
        )

        config2 = TaskChainConfig(
            chain_id="test-chain",
            priority=5,
            dependencies=["dep1", "dep2"],
            resource_requirements={"cpu": 2.0, "memory": 4.0},
        )

        config3 = TaskChainConfig(
            chain_id="different-chain",
            priority=5,
            dependencies=["dep1", "dep2"],
            resource_requirements={"cpu": 2.0, "memory": 4.0},
        )

        assert config1 == config2
        assert config1 != config3
        assert config1 != "not a config"

    def test_repr_and_str(self, task_chain_config):
        """Test string representation of TaskChainConfig."""
        repr_string = repr(task_chain_config)
        str_string = str(task_chain_config)

        assert "test-chain" in repr_string
        assert "test-chain" in str_string
        assert "priority=5" in repr_string or "priority=5" in str_string


class TestChainManager:
    """Test suite for the ChainManager class."""

    @pytest.fixture
    def chain_manager(self):
        """Create a ChainManager instance for testing."""
        with patch("src.core.ChainCore.SystemConfig") as _:
            return ChainManager()

    @pytest.fixture
    def task_chain_configs(self):
        """Create sample TaskChainConfig instances for testing."""
        return [
            TaskChainConfig(
                chain_id="chain1",
                priority=1,
                dependencies=[],
                resource_requirements={"cpu": 1.0},
            ),
            TaskChainConfig(
                chain_id="chain2",
                priority=2,
                dependencies=["chain1"],
                resource_requirements={"cpu": 2.0},
            ),
            TaskChainConfig(
                chain_id="chain3",
                priority=3,
                dependencies=["chain1", "chain2"],
                resource_requirements={"cpu": 3.0},
            ),
        ]

    def test_init(self, chain_manager):
        """Test initialization of ChainManager."""
        assert hasattr(chain_manager, "chains")
        assert isinstance(chain_manager.chains, dict)
        assert len(chain_manager.chains) == 0

    def test_add_chain(self, chain_manager, task_chain_configs):
        """Test adding a chain to the manager."""
        # Add the first chain
        chain_manager.add_chain(task_chain_configs[0])

        assert len(chain_manager.chains) == 1
        assert "chain1" in chain_manager.chains
        assert chain_manager.chains["chain1"] == task_chain_configs[0]

        # Add the second chain
        chain_manager.add_chain(task_chain_configs[1])

        assert len(chain_manager.chains) == 2
        assert "chain2" in chain_manager.chains
        assert chain_manager.chains["chain2"] == task_chain_configs[1]

    def test_add_duplicate_chain(self, chain_manager, task_chain_configs):
        """Test adding a duplicate chain."""
        # Add a chain
        chain_manager.add_chain(task_chain_configs[0])

        # Add the same chain again
        chain_manager.add_chain(task_chain_configs[0])

        # Should still only have one entry
        assert len(chain_manager.chains) == 1

        # Create a different chain with the same ID
        duplicate_chain = TaskChainConfig(
            chain_id="chain1",
            priority=5,  # Different priority
            dependencies=[],
            resource_requirements={"cpu": 1.0},
        )

        # Add the duplicate chain (should replace the original)
        chain_manager.add_chain(duplicate_chain)

        # Should still only have one entry
        assert len(chain_manager.chains) == 1
        assert chain_manager.chains["chain1"].priority == 5  # The new priority

    def test_remove_chain(self, chain_manager, task_chain_configs):
        """Test removing a chain from the manager."""
        # Add chains
        for config in task_chain_configs:
            chain_manager.add_chain(config)

        self._extracted_from_test_remove_nonexistent_chain_7(
            chain_manager, 3, "chain2", 2
        )
        assert "chain1" in chain_manager.chains
        assert "chain2" not in chain_manager.chains
        assert "chain3" in chain_manager.chains

    def test_remove_nonexistent_chain(self, chain_manager):
        """Test removing a chain that doesn't exist."""
        self._extracted_from_test_remove_nonexistent_chain_7(
            chain_manager, 0, "nonexistent-chain", 0
        )

    # TODO Rename this here and in `test_remove_chain` and `test_remove_nonexistent_chain`
    def _extracted_from_test_remove_nonexistent_chain_7(
        self, chain_manager, arg1, arg2, arg3
    ):
        assert len(chain_manager.chains) == arg1
        chain_manager.remove_chain(arg2)
        assert len(chain_manager.chains) == arg3

    def test_get_chain(self, chain_manager, task_chain_configs):
        """Test getting a chain from the manager."""
        # Add chains
        for config in task_chain_configs:
            chain_manager.add_chain(config)

        # Get a chain
        chain = chain_manager.get_chain("chain2")

        assert chain == task_chain_configs[1]
        assert chain.chain_id == "chain2"
        assert chain.priority == 2

    def test_get_nonexistent_chain(self, chain_manager):
        """Test getting a chain that doesn't exist."""
        # Try to get a nonexistent chain
        chain = chain_manager.get_chain("nonexistent-chain")

        assert chain is None

    def test_get_all_chains(self, chain_manager, task_chain_configs):
        """Test getting all chains from the manager."""
        # Add chains
        for config in task_chain_configs:
            chain_manager.add_chain(config)

        # Get all chains
        chains = chain_manager.get_all_chains()

        assert len(chains) == 3
        assert all(chain in chains for chain in task_chain_configs)

    def test_get_all_chains_empty(self, chain_manager):
        """Test getting all chains when there are none."""
        # Get all chains
        chains = chain_manager.get_all_chains()

        assert len(chains) == 0
        assert isinstance(chains, list)


class TestChainMetrics:
    """Test suite for the ChainMetrics class."""

    @pytest.fixture
    def chain_metrics(self):
        """Create a ChainMetrics instance for testing."""
        return ChainMetrics(
            chain_id="test-chain",
            execution_time=10.5,
            task_completion={"task1": 100.0, "task2": 75.0},
            resource_allocation={"cpu": 2.0, "memory": 4.0},
        )

    def test_init(self, chain_metrics):
        """Test initialization of ChainMetrics."""
        assert chain_metrics.chain_id == "test-chain"
        assert chain_metrics.execution_time == 10.5
        assert chain_metrics.task_completion == {"task1": 100.0, "task2": 75.0}
        assert chain_metrics.resource_allocation == {"cpu": 2.0, "memory": 4.0}
        assert chain_metrics.error_registry == []
        assert isinstance(chain_metrics.metrics, dict)
        assert chain_metrics.status == ""
        assert isinstance(chain_metrics.timestamp, datetime)

    def test_post_init(self):
        """Test __post_init__ method."""
        metrics = ChainMetrics(
            chain_id="test-chain",
            execution_time=10.5,
            task_completion={"task1": 100.0, "task2": 75.0},
            resource_allocation={"cpu": 2.0, "memory": 4.0},
        )

        # Check metrics are initialized
        assert "chain_id" in metrics.metrics
        assert metrics.metrics["chain_id"] == "test-chain"
        assert "execution_time" in metrics.metrics
        assert metrics.metrics["execution_time"] == 10.5
        assert "task_completion" in metrics.metrics
        assert metrics.metrics["task_completion"] == {"task1": 100.0, "task2": 75.0}
        assert "resource_allocation" in metrics.metrics
        assert metrics.metrics["resource_allocation"] == {"cpu": 2.0, "memory": 4.0}
        assert "error_registry" in metrics.metrics
        assert metrics.metrics["error_registry"] == []
        assert "timestamp" in metrics.metrics
        assert isinstance(metrics.metrics["timestamp"], datetime)

    def test_str_and_repr(self, chain_metrics):
        """Test string representation of ChainMetrics."""
        str_value = str(chain_metrics)
        repr_value = repr(chain_metrics)

        assert "test-chain" in str_value
        assert "test-chain" in repr_value
        assert "10.5" in str_value or "10.5" in repr_value
        assert "task_completion" in str_value or "task_completion" in repr_value
        assert "resource_allocation" in str_value or "resource_allocation" in repr_value

    def test_equality(self, chain_metrics):
        """Test equality of ChainMetrics instances."""
        # Create an identical metrics object
        metrics2 = ChainMetrics(
            chain_id="test-chain",
            execution_time=10.5,
            task_completion={"task1": 100.0, "task2": 75.0},
            resource_allocation={"cpu": 2.0, "memory": 4.0},
        )

        # Create a different metrics object
        metrics3 = ChainMetrics(
            chain_id="different-chain",
            execution_time=10.5,
            task_completion={"task1": 100.0, "task2": 75.0},
            resource_allocation={"cpu": 2.0, "memory": 4.0},
        )

        # Fix timestamps for comparison
        metrics2.timestamp = chain_metrics.timestamp
        metrics3.timestamp = chain_metrics.timestamp

        # Check equality
        assert chain_metrics == metrics2
        assert chain_metrics != metrics3
        assert chain_metrics != "not a metrics object"

        # Check inequality
        assert chain_metrics == metrics2
        assert chain_metrics != metrics3

    def test_with_error_registry(self):
        """Test creating ChainMetrics with error registry."""
        errors = ["Error 1", "Error 2"]
        metrics = ChainMetrics(
            chain_id="test-chain",
            execution_time=10.5,
            task_completion={"task1": 100.0, "task2": 75.0},
            resource_allocation={"cpu": 2.0, "memory": 4.0},
            error_registry=errors,
        )

        assert metrics.error_registry == errors
        assert metrics.metrics["error_registry"] == errors


class TestChainExecutor:
    """Test suite for the ChainExecutor class."""

    @pytest.fixture
    def chain_executor(self):
        """Create a ChainExecutor instance for testing."""
        with patch("src.core.ChainCore.ChainValidator") as mock_validator, patch(
            "src.core.ChainCore.ResourceManager"
        ) as mock_resource_manager, patch(
            "src.core.ChainCore.ExecutionMonitor"
        ) as mock_monitor:

            executor = ChainExecutor()

            # Replace with mocks for testing
            executor.validator = mock_validator.return_value
            executor.resource_manager = mock_resource_manager.return_value
            executor.monitor = mock_monitor.return_value

            return executor

    @pytest.fixture
    def task_chain_config(self):
        """Create a TaskChainConfig instance for testing."""
        return TaskChainConfig(
            chain_id="test-chain",
            priority=5,
            dependencies=["dep1", "dep2"],
            resource_requirements={"cpu": 2.0, "memory": 4.0},
        )

    def test_init(self, chain_executor):
        """Test initialization of ChainExecutor."""
        assert chain_executor.validator is not None
        assert chain_executor.resource_manager is not None
        assert chain_executor.monitor is not None

    def test_execute_chain_success(self, chain_executor, task_chain_config):
        """Test successful execution of a chain."""
        self._extracted_from_test_execute_chain_processing_exception_4(chain_executor)
        chain_executor._process_chain = Mock(return_value={"result": "success"})
        chain_executor._compute_chain_metrics = Mock(return_value=MagicMock())

        _ = self._extracted_from_test_execute_chain_processing_exception_13(
            chain_executor, task_chain_config, True
        )
        # Verify method calls
        chain_executor.validator.validate_chain.assert_called_once_with(
            task_chain_config
        )
        self._extracted_from_test_execute_chain_processing_exception_23(
            chain_executor, task_chain_config
        )
        chain_executor._compute_chain_metrics.assert_called_once()
        chain_executor.resource_manager.release_resources.assert_called_once()

    def test_execute_chain_validation_failure(self, chain_executor, task_chain_config):
        """Test chain execution when validation fails."""
        # Setup mocks - validation fails
        chain_executor.validator.validate_chain.return_value = False

        self._extracted_from_test_execute_chain_processing_exception_7(
            chain_executor, task_chain_config, "failed validation"
        )
        chain_executor.resource_manager.allocate_resources.assert_not_called()
        chain_executor._prepare_chain_context.assert_not_called()
        chain_executor._process_chain.assert_not_called()

    def test_execute_chain_resource_allocation_failure(
        self, chain_executor, task_chain_config
    ):
        """Test chain execution when resource allocation fails."""
        # Setup mocks - validation passes but resource allocation fails
        chain_executor.validator.validate_chain.return_value = True
        chain_executor.resource_manager.allocate_resources.return_value = False

        self._extracted_from_test_execute_chain_processing_exception_7(
            chain_executor, task_chain_config, "resource allocation"
        )
        chain_executor.resource_manager.allocate_resources.assert_called_once()
        chain_executor._prepare_chain_context.assert_not_called()
        chain_executor._process_chain.assert_not_called()

    def test_execute_chain_processing_exception(
        self, chain_executor, task_chain_config
    ):
        """Test chain execution when processing throws an exception."""
        self._extracted_from_test_execute_chain_processing_exception_4(chain_executor)
        chain_executor._process_chain = Mock(side_effect=Exception("Processing error"))

        self._extracted_from_test_execute_chain_processing_exception_7(
            chain_executor, task_chain_config, "processing error"
        )
        self._extracted_from_test_execute_chain_processing_exception_23(
            chain_executor, task_chain_config
        )
        chain_executor.resource_manager.release_resources.assert_called_once()

    # TODO Rename this here and in `test_execute_chain_success`, `test_execute_chain_validation_failure`, `test_execute_chain_resource_allocation_failure` and `test_execute_chain_processing_exception`
    def _extracted_from_test_execute_chain_processing_exception_23(
        self, chain_executor, task_chain_config
    ):
        chain_executor.resource_manager.allocate_resources.assert_called_once()
        chain_executor._prepare_chain_context.assert_called_once_with(task_chain_config)
        chain_executor._process_chain.assert_called_once()

    # TODO Rename this here and in `test_execute_chain_success`, `test_execute_chain_validation_failure`, `test_execute_chain_resource_allocation_failure` and `test_execute_chain_processing_exception`
    def _extracted_from_test_execute_chain_processing_exception_4(self, chain_executor):
        chain_executor.validator.validate_chain.return_value = True
        chain_executor.resource_manager.allocate_resources.return_value = True
        chain_executor._prepare_chain_context = Mock(
            return_value={"chain_id": "test-chain"}
        )

    # TODO Rename this here and in `test_execute_chain_success`, `test_execute_chain_validation_failure`, `test_execute_chain_resource_allocation_failure` and `test_execute_chain_processing_exception`
    def _extracted_from_test_execute_chain_processing_exception_7(
        self, chain_executor, task_chain_config, arg2
    ):
        result = self._extracted_from_test_execute_chain_processing_exception_13(
            chain_executor, task_chain_config, False
        )
        assert arg2 in result.error.lower()
        chain_executor.validator.validate_chain.assert_called_once_with(
            task_chain_config
        )

    # TODO Rename this here and in `test_execute_chain_success`, `test_execute_chain_validation_failure`, `test_execute_chain_resource_allocation_failure` and `test_execute_chain_processing_exception`
    def _extracted_from_test_execute_chain_processing_exception_13(
        self, chain_executor, task_chain_config, arg2
    ):
        result = chain_executor.execute_chain(task_chain_config)
        assert isinstance(result, ExecutionResult)
        assert result.success is arg2
        return result

    def test_prepare_chain_context(self, chain_executor, task_chain_config):
        """Test preparing chain context."""
        # Call the method
        context = chain_executor._prepare_chain_context(task_chain_config)

        # Verify the context
        assert isinstance(context, dict)
        assert "chain_id" in context
        assert context["chain_id"] == task_chain_config.chain_id
        assert "dependencies" in context
        assert context["dependencies"] == task_chain_config.dependencies
        assert "priority" in context
        assert context["priority"] == task_chain_config.priority
        assert "resources" in context
        assert context["resources"] == task_chain_config.resource_requirements
        assert "status" in context
        assert context["status"] == "prepared"

    def test_process_chain(self, chain_executor):
        """Test processing a chain."""
        # Setup
        context = {
            "chain_id": "test-chain",
            "dependencies": ["dep1", "dep2"],
            "priority": 5,
            "resources": {"cpu": 2.0, "memory": 4.0},
            "status": "prepared",
        }

        # Call the method
        result = chain_executor._process_chain(context)

        # Verify the result
        assert isinstance(result, dict)
        assert "chain_id" in result
        assert result["chain_id"] == context["chain_id"]
        assert "status" in result
        assert result["status"] == "completed"
        assert "result" in result
        assert isinstance(result["result"], ChainResult)

    def test_compute_chain_metrics(self, chain_executor):
        """Test computing chain metrics."""
        # Setup
        result = {
            "chain_id": "test-chain",
            "status": "completed",
            "result": ChainResult(success=True, data={"key": "value"}, error=None),
            "execution_time": 10.5,
            "task_completion": {"task1": 100.0, "task2": 75.0},
            "resource_usage": {"cpu": 1.8, "memory": 3.5},
        }

        # Call the method
        metrics = chain_executor._compute_chain_metrics(result)

        # Verify the metrics
        assert isinstance(metrics, ChainMetrics)
        assert metrics.chain_id == result["chain_id"]
        assert metrics.execution_time == result["execution_time"]
        assert metrics.task_completion == result["task_completion"]
        # Resource allocation should be based on resource_usage
        assert metrics.resource_allocation == result["resource_usage"]
        assert metrics.status == "completed"
        assert metrics.error_registry == []
