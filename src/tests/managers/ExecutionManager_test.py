from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest

from src.ContinuationResult import ContinuationResult
from src.core.ExecutionCore import ExecutionController
from src.ExecutionMonitor import ExecutionMonitor
from src.managers.ExecutionManager import ExecutionManager
from src.results.ExecutionResult import ExecutionResult
from src.validators.ExecutionValidator import ExecutionValidator


class TestExecutionManager:
    """Test suite for the ExecutionManager class."""

    @pytest.fixture
    def mock_controller(self):
        """Create a mock ExecutionController."""
        controller = AsyncMock(spec=ExecutionController)
        controller.initialize_continuation = AsyncMock(
            return_value={"id": "continuation-123"}
        )
        controller.execute_continuation = AsyncMock(
            return_value=ExecutionResult(success=True)
        )
        controller.finalize_execution = AsyncMock(return_value=True)
        return controller

    @pytest.fixture
    def mock_validator(self):
        """Create a mock ExecutionValidator."""
        validator = AsyncMock(spec=ExecutionValidator)
        validator.check_state = AsyncMock(return_value=True)
        validator.validate_execution = AsyncMock(return_value=True)
        validator.validate_result = AsyncMock(return_value=True)
        return validator

    @pytest.fixture
    def mock_monitor(self):
        """Create a mock ExecutionMonitor."""
        monitor = Mock(spec=ExecutionMonitor)
        # Create a context manager mock
        context_manager_mock = MagicMock()
        monitor.track_execution.return_value = context_manager_mock
        context_manager_mock.__enter__.return_value = monitor
        context_manager_mock.__exit__.return_value = None
        return monitor

    @pytest.fixture
    def execution_manager(self, mock_controller, mock_validator, mock_monitor):
        """Create an ExecutionManager with mocked dependencies."""
        with patch(
            "src.managers.ExecutionManager.ExecutionController",
            return_value=mock_controller,
        ), patch(
            "src.managers.ExecutionManager.ExecutionValidator",
            return_value=mock_validator,
        ), patch(
            "src.managers.ExecutionManager.ExecutionMonitor", return_value=mock_monitor
        ):
            manager = ExecutionManager()
            # Set mocked objects for easier testing
            manager.controller = mock_controller
            manager.validator = mock_validator
            manager.monitor = mock_monitor
            return manager

    @pytest.mark.asyncio
    async def test_process_continuation_success(self, execution_manager):
        """Test the process_continuation method when successful."""
        # Call the method
        result = await execution_manager.process_continuation()

        # Verify all expected methods were called
        execution_manager.validator.check_state.assert_called_once()
        execution_manager.controller.initialize_continuation.assert_called_once()
        execution_manager.controller.execute_continuation.assert_called_once()
        execution_manager.controller.finalize_execution.assert_called_once()

        # Verify the result
        assert isinstance(result, ContinuationResult)
        assert result.success is True

    @pytest.mark.asyncio
    async def test_process_continuation_failed_state_check(self, execution_manager):
        """Test process_continuation when state check fails."""
        # Setup
        execution_manager.validator.check_state.return_value = False

        # Call the method and verify exception
        result = await execution_manager.process_continuation()

        # Verify
        assert isinstance(result, ContinuationResult)
        assert result.success is False
        assert "Invalid system state" in result.error

        # Verify controller methods were not called
        execution_manager.controller.initialize_continuation.assert_not_called()
        execution_manager.controller.execute_continuation.assert_not_called()
        execution_manager.controller.finalize_execution.assert_not_called()

    @pytest.mark.asyncio
    async def test_process_continuation_execution_error(self, execution_manager):
        """Test process_continuation when execution fails."""
        # Setup
        execution_manager.controller.execute_continuation.return_value = (
            ExecutionResult(success=False, error="Execution failed")
        )

        # Call the method
        result = await execution_manager.process_continuation()

        # Verify
        assert isinstance(result, ContinuationResult)
        assert result.success is False
        assert "Execution failed" in result.error

        # Verify methods were called appropriately
        execution_manager.validator.check_state.assert_called_once()
        execution_manager.controller.initialize_continuation.assert_called_once()
        execution_manager.controller.execute_continuation.assert_called_once()
        execution_manager.controller.finalize_execution.assert_called_once()

    @pytest.mark.asyncio
    async def test_process_continuation_exception(self, execution_manager):
        """Test process_continuation when an exception occurs."""
        # Setup
        execution_manager.controller.execute_continuation.side_effect = Exception(
            "Unexpected error"
        )

        # Call the method
        result = await execution_manager.process_continuation()

        # Verify
        assert isinstance(result, ContinuationResult)
        assert result.success is False
        assert "Unexpected error" in result.error

        # Verify methods were called appropriately
        execution_manager.validator.check_state.assert_called_once()
        execution_manager.controller.initialize_continuation.assert_called_once()
        execution_manager.controller.execute_continuation.assert_called_once()
        execution_manager.controller.finalize_execution.assert_not_called()  # Should not be called after exception

    @pytest.mark.asyncio
    async def test_execute_task_success(self, execution_manager):
        """Test the execute_task method when successful."""
        # Setup
        task_id = "task-123"
        task_data = {"key": "value"}
        execution_manager.controller.execute_task = AsyncMock(
            return_value=ExecutionResult(success=True)
        )

        # Call the method
        result = await execution_manager.execute_task(task_id, task_data)

        # Verify
        assert result.success is True
        execution_manager.validator.validate_execution.assert_called_once_with(
            task_id, task_data
        )
        execution_manager.controller.execute_task.assert_called_once_with(
            task_id, task_data
        )

    @pytest.mark.asyncio
    async def test_execute_task_validation_failure(self, execution_manager):
        """Test execute_task when validation fails."""
        # Setup
        task_id = "task-123"
        task_data = {"key": "value"}
        execution_manager.validator.validate_execution.return_value = False

        # Call the method
        result = await execution_manager.execute_task(task_id, task_data)

        # Verify
        assert result.success is False
        assert "Invalid task parameters" in result.error
        execution_manager.validator.validate_execution.assert_called_once_with(
            task_id, task_data
        )
        execution_manager.controller.execute_task.assert_not_called()

    @pytest.mark.asyncio
    async def test_execute_task_execution_failure(self, execution_manager):
        """Test execute_task when execution fails."""
        # Setup
        task_id = "task-123"
        task_data = {"key": "value"}
        execution_manager.controller.execute_task = AsyncMock(
            return_value=ExecutionResult(success=False, error="Task execution failed")
        )

        # Call the method
        result = await execution_manager.execute_task(task_id, task_data)

        # Verify
        assert result.success is False
        assert "Task execution failed" in result.error
        execution_manager.validator.validate_execution.assert_called_once_with(
            task_id, task_data
        )
        execution_manager.controller.execute_task.assert_called_once_with(
            task_id, task_data
        )

    @pytest.mark.asyncio
    async def test_execute_task_exception(self, execution_manager):
        """Test execute_task when an exception occurs."""
        # Setup
        task_id = "task-123"
        task_data = {"key": "value"}
        execution_manager.controller.execute_task = AsyncMock(
            side_effect=Exception("Unexpected task error")
        )

        # Call the method
        result = await execution_manager.execute_task(task_id, task_data)

        # Verify
        assert result.success is False
        assert "Unexpected task error" in result.error
        execution_manager.validator.validate_execution.assert_called_once_with(
            task_id, task_data
        )
        execution_manager.controller.execute_task.assert_called_once_with(
            task_id, task_data
        )

    @pytest.mark.asyncio
    async def test_cancel_execution_success(self, execution_manager):
        """Test the cancel_execution method when successful."""
        # Setup
        execution_id = "exec-123"
        execution_manager.controller.cancel_execution = AsyncMock(return_value=True)

        # Call the method
        result = await execution_manager.cancel_execution(execution_id)

        # Verify
        assert result.success is True
        execution_manager.controller.cancel_execution.assert_called_once_with(
            execution_id
        )

    @pytest.mark.asyncio
    async def test_cancel_execution_failure(self, execution_manager):
        """Test cancel_execution when cancellation fails."""
        # Setup
        execution_id = "exec-123"
        execution_manager.controller.cancel_execution = AsyncMock(return_value=False)

        # Call the method
        result = await execution_manager.cancel_execution(execution_id)

        # Verify
        assert result.success is False
        assert "Failed to cancel execution" in result.error
        execution_manager.controller.cancel_execution.assert_called_once_with(
            execution_id
        )

    @pytest.mark.asyncio
    async def test_cancel_execution_exception(self, execution_manager):
        """Test cancel_execution when an exception occurs."""
        # Setup
        execution_id = "exec-123"
        execution_manager.controller.cancel_execution = AsyncMock(
            side_effect=Exception("Cancellation error")
        )

        # Call the method
        result = await execution_manager.cancel_execution(execution_id)

        # Verify
        assert result.success is False
        assert "Cancellation error" in result.error
        execution_manager.controller.cancel_execution.assert_called_once_with(
            execution_id
        )
