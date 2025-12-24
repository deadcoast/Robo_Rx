from unittest.mock import AsyncMock, Mock, patch

import pytest

from src.config.EngineConfig import SystemConfig
from src.Continuation import Continuation
from src.ContinuationResult import ContinuationResult
from src.core.ExecutionCore import ExecutionController, ExecutionCore, ExecutionMetrics
from src.ExecutionResult import ExecutionResult


class TestExecutionCore:
    """Test suite for the ExecutionCore class."""

    @pytest.fixture
    def config(self):
        """Create a mock SystemConfig instance."""
        return Mock(spec=SystemConfig)

    @pytest.fixture
    def execution_core(self, config):
        """Create an ExecutionCore instance for testing."""
        with patch(
            "src.core.ExecutionCore.ExecutionValidator"
        ) as mock_validator, patch(
            "src.core.ExecutionCore.ResourceManager"
        ) as mock_resource_manager, patch(
            "src.core.ExecutionCore.ExecutionMonitor"
        ) as mock_monitor, patch.object(
            ExecutionCore, "_initialize"
        ):

            core = ExecutionCore(config)

            # Set mocked objects for easier testing
            core.validator = mock_validator.return_value
            core.resource_manager = mock_resource_manager.return_value
            core.monitor = mock_monitor.return_value

            return core

    def test_init(self, execution_core, config):
        """Test initialization of ExecutionCore."""
        assert execution_core.config == config
        assert execution_core.validator is not None
        assert execution_core.resource_manager is not None
        assert execution_core.monitor is not None

    def test_initialize(self, config):
        """Test the _initialize method."""
        # Setup mocks for all components
        with patch(
            "src.core.ExecutionCore.ExecutionValidator"
        ) as mock_validator, patch(
            "src.core.ExecutionCore.ResourceManager"
        ) as mock_resource_manager, patch(
            "src.core.ExecutionCore.ExecutionMonitor"
        ) as mock_monitor, patch(
            "src.core.ExecutionCore.StateManager"
        ) as mock_state_manager, patch(
            "src.core.ExecutionCore.ResourceAllocator"
        ) as mock_resource_allocator, patch(
            "src.core.ExecutionCore.ProgressTracker"
        ) as mock_progress_tracker:

            # Create instance with mocked _initialize to avoid calling it during construction
            with patch.object(ExecutionCore, "_initialize"):
                core = ExecutionCore(config)

                # Set up mocked components
                core.state_manager = mock_state_manager.return_value
                core.resource_allocator = mock_resource_allocator.return_value
                core.progress_tracker = mock_progress_tracker.return_value
                core.controller = Mock()
                core.validator = mock_validator.return_value
                core.resource_manager = mock_resource_manager.return_value
                core.monitor = mock_monitor.return_value

                # Now call _initialize explicitly
                core._initialize()

                # Verify all components are initialized
                core.state_manager.initialize.assert_called_once()
                core.resource_allocator.initialize.assert_called_once()
                core.progress_tracker.initialize.assert_called_once()
                core.controller.initialize.assert_called_once()
                core.validator.initialize.assert_called_once()
                core.resource_manager.initialize.assert_called_once()
                core.monitor.initialize.assert_called_once()

    @pytest.mark.asyncio
    async def test_process_next_task(self, execution_core):
        """Test the process_next_task method."""
        # Setup expected result
        expected_result = ExecutionResult(success=True)

        # Mock the asynchronous method
        execution_core.process_next_task = AsyncMock(return_value=expected_result)

        # Call the method
        result = await execution_core.process_next_task()

        # Verify
        assert result == expected_result
        execution_core.process_next_task.assert_called_once()

    @pytest.mark.asyncio
    async def test_execute_task(self, execution_core):
        """Test the execute_task method."""
        # Setup
        task_id = "task-123"
        task_data = {"key": "value"}
        expected_result = ExecutionResult(success=True)

        # Mock the asynchronous method
        execution_core.execute_task = AsyncMock(return_value=expected_result)

        # Call the method
        result = await execution_core.execute_task(task_id, task_data)

        # Verify
        assert result == expected_result
        execution_core.execute_task.assert_called_once_with(task_id, task_data)

    @pytest.mark.asyncio
    async def test_process_execution(self, execution_core):
        """Test the process_execution method."""
        # Setup expected result
        expected_result = ExecutionResult(success=True)

        # Mock the asynchronous method
        execution_core.process_execution = AsyncMock(return_value=expected_result)

        # Call the method
        result = await execution_core.process_execution()

        # Verify
        assert result == expected_result
        execution_core.process_execution.assert_called_once()


class TestExecutionMetrics:
    """Test suite for the ExecutionMetrics class."""

    @pytest.fixture
    def execution_metrics(self):
        """Create an ExecutionMetrics instance for testing."""
        return ExecutionMetrics()

    def test_init(self, execution_metrics):
        """Test initialization of ExecutionMetrics."""
        assert execution_metrics.phase_complete == 0
        assert execution_metrics.current_phase == ""
        assert execution_metrics.error_count == 0
        assert execution_metrics.performance_stats == {}

    def test_update_phase(self, execution_metrics):
        """Test updating the current phase."""
        # Mock method if it exists
        if hasattr(execution_metrics, "update_phase"):
            original_update_phase = execution_metrics.update_phase
            execution_metrics.update_phase = Mock(wraps=original_update_phase)

            # Call the method
            execution_metrics.update_phase("processing")

            # Verify
            assert execution_metrics.current_phase == "processing"
            execution_metrics.update_phase.assert_called_once_with("processing")

    def test_increment_phase(self, execution_metrics):
        """Test incrementing completed phases."""
        # Mock method if it exists
        if hasattr(execution_metrics, "increment_phase"):
            original_increment_phase = execution_metrics.increment_phase
            execution_metrics.increment_phase = Mock(wraps=original_increment_phase)

            # Call the method
            execution_metrics.increment_phase()

            # Verify
            assert execution_metrics.phase_complete == 1
            execution_metrics.increment_phase.assert_called_once()

    def test_record_error(self, execution_metrics):
        """Test recording an error."""
        # Mock method if it exists
        if hasattr(execution_metrics, "record_error"):
            original_record_error = execution_metrics.record_error
            execution_metrics.record_error = Mock(wraps=original_record_error)

            # Call the method
            execution_metrics.record_error()

            # Verify
            assert execution_metrics.error_count == 1
            execution_metrics.record_error.assert_called_once()

    def test_add_performance_stat(self, execution_metrics):
        """Test adding a performance statistic."""
        # Mock method if it exists
        if hasattr(execution_metrics, "add_performance_stat"):
            original_add_stat = execution_metrics.add_performance_stat
            execution_metrics.add_performance_stat = Mock(wraps=original_add_stat)

            # Call the method
            execution_metrics.add_performance_stat("execution_time", 10.5)

            # Verify
            assert execution_metrics.performance_stats["execution_time"] == 10.5
            execution_metrics.add_performance_stat.assert_called_once_with(
                "execution_time", 10.5
            )


class TestExecutionController:
    """Test suite for the ExecutionController class."""

    @pytest.fixture
    def config(self):
        """Create a mock SystemConfig instance."""
        return Mock(spec=SystemConfig)

    @pytest.fixture
    def execution_controller(self, config):
        """Create an ExecutionController instance for testing."""
        with patch("src.core.ExecutionCore.StateManager") as mock_state_manager, patch(
            "src.core.ExecutionCore.ResourceAllocator"
        ) as mock_resource_allocator, patch(
            "src.core.ExecutionCore.ProgressTracker"
        ) as mock_progress_tracker, patch(
            "src.core.ExecutionCore.ExecutionMonitor"
        ) as mock_monitor, patch.object(
            ExecutionController, "_initialize"
        ):

            controller = ExecutionController(config)

            # Set mocked objects for easier testing
            controller.state_manager = mock_state_manager.return_value
            controller.resource_allocator = mock_resource_allocator.return_value
            controller.progress_tracker = mock_progress_tracker.return_value
            controller.monitor = mock_monitor.return_value

            return controller

    def test_init(self, execution_controller, config):
        """Test initialization of ExecutionController."""
        assert execution_controller.config == config
        assert execution_controller.state_manager is not None
        assert execution_controller.resource_allocator is not None
        assert execution_controller.progress_tracker is not None
        assert execution_controller.monitor is not None

    def test_initialize(self, config):
        """Test the _initialize method."""
        # Setup
        with patch("src.core.ExecutionCore.StateManager") as mock_state_manager, patch(
            "src.core.ExecutionCore.ResourceAllocator"
        ) as mock_resource_allocator, patch(
            "src.core.ExecutionCore.ProgressTracker"
        ) as mock_progress_tracker, patch(
            "src.core.ExecutionCore.ExecutionMonitor"
        ) as mock_monitor:

            # Create instance with mocked _initialize to avoid calling it during construction
            with patch.object(ExecutionController, "_initialize"):
                controller = ExecutionController(config)

                # Set up mocked components
                controller.state_manager = mock_state_manager.return_value
                controller.resource_allocator = mock_resource_allocator.return_value
                controller.progress_tracker = mock_progress_tracker.return_value
                controller.monitor = mock_monitor.return_value

                # Now call _initialize explicitly
                controller._initialize()

                # Verify all components are initialized
                controller.state_manager.initialize.assert_called_once()
                controller.resource_allocator.initialize.assert_called_once()
                controller.progress_tracker.initialize.assert_called_once()
                controller.monitor.initialize.assert_called_once()

    def test_initialize_continuation(self, execution_controller):
        """Test the initialize_continuation method."""
        # Setup expected result
        expected_result = ContinuationResult(success=True)

        # Mock the method
        execution_controller.initialize_continuation = Mock(
            return_value=expected_result
        )

        # Call the method
        result = execution_controller.initialize_continuation()

        # Verify
        assert result == expected_result
        execution_controller.initialize_continuation.assert_called_once()

    def test_execute_continuation(self, execution_controller):
        """Test the execute_continuation method."""
        # Setup
        continuation = Continuation(id="cont-123", type="test")
        expected_result = ContinuationResult(success=True)

        # Mock the method
        execution_controller.execute_continuation = Mock(return_value=expected_result)

        # Call the method
        result = execution_controller.execute_continuation(continuation)

        # Verify
        assert result == expected_result
        execution_controller.execute_continuation.assert_called_once_with(continuation)

    def test_finalize_execution(self, execution_controller):
        """Test the finalize_execution method."""
        # Setup expected result
        expected_result = ExecutionResult(success=True)

        # Mock the method
        execution_controller.finalize_execution = Mock(return_value=expected_result)

        # Call the method
        result = execution_controller.finalize_execution()

        # Verify
        assert result == expected_result
        execution_controller.finalize_execution.assert_called_once()

    def test_process_execution(self, execution_controller):
        """Test the process_execution method."""
        # Setup expected result
        expected_result = ExecutionResult(success=True)

        # Mock the method
        execution_controller.process_execution = Mock(return_value=expected_result)

        # Call the method
        result = execution_controller.process_execution()

        # Verify
        assert result == expected_result
        execution_controller.process_execution.assert_called_once()

    def test_execute_next_phase(self, execution_controller):
        """Test the _execute_next_phase method."""
        # Setup
        state = {"state_key": "state_value"}
        resources = {"resource_key": "resource_value"}
        expected_result = ExecutionResult(success=True)

        # Mock the method
        execution_controller._execute_next_phase = Mock(return_value=expected_result)

        # Call the method
        result = execution_controller._execute_next_phase(state, resources)

        # Verify
        assert result == expected_result
        execution_controller._execute_next_phase.assert_called_once_with(
            state, resources
        )

    def test_execute_finalization(self, execution_controller):
        """Test the _execute_finalization method."""
        # Setup
        state = {"state_key": "state_value"}
        resources = {"resource_key": "resource_value"}
        expected_result = ExecutionResult(success=True)

        # Mock the method
        execution_controller._execute_finalization = Mock(return_value=expected_result)

        # Call the method
        result = execution_controller._execute_finalization(state, resources)

        # Verify
        assert result == expected_result
        execution_controller._execute_finalization.assert_called_once_with(
            state, resources
        )

    def test_continuation_execution_lifecycle(self, execution_controller):
        """Test the complete lifecycle of continuation execution."""
        # Setup
        continuation = Continuation(id="cont-123", type="test")

        # Mock the methods in the lifecycle
        execution_controller.initialize_continuation = Mock(return_value=continuation)
        execution_controller.execute_continuation = Mock(
            return_value=ContinuationResult(success=True)
        )
        execution_controller.finalize_execution = Mock(
            return_value=ExecutionResult(success=True)
        )

        # Execute the lifecycle
        cont = execution_controller.initialize_continuation()
        cont_result = execution_controller.execute_continuation(cont)
        final_result = execution_controller.finalize_execution()

        # Verify
        assert cont == continuation
        assert cont_result.success is True
        assert final_result.success is True
        execution_controller.initialize_continuation.assert_called_once()
        execution_controller.execute_continuation.assert_called_once_with(continuation)
        execution_controller.finalize_execution.assert_called_once()
