"""Execution Manager Class."""

from src.ContinuationResult import ContinuationResult
from src.core.ExecutionCore import ExecutionController
from src.ExecutionMonitor import ExecutionMonitor
from src.results.ExecutionResult import ExecutionResult
from src.validators.ExecutionValidator import ExecutionValidator


class ExecutionManager:
    """
    _summary_

    _extended_summary_

    :ivar controller: The controller for the execution manager.
    :type controller: ExecutionController
    :ivar validator: The validator for the execution manager.
    :type validator: ExecutionValidator
    :ivar monitor: The monitor for the execution manager.
    :type monitor: ExecutionMonitor
    """

    def __init__(self):
        self.controller = ExecutionController()
        self.validator = ExecutionValidator()
        self.monitor = ExecutionMonitor()

    async def process_continuation(self) -> ContinuationResult:
        """
        _summary_

        _extended_summary_

        Returns:
            ContinuationResult: _description_
        """
        # TODO: Implement the process_continuation Logic
        try:
            # Validate current state
            state_valid = await self.validator.check_state()
            if not state_valid:
                raise ExecutionValidator("Invalid system state")

            # Initialize continuation
            continuation = await self.controller.initialize_continuation()

            # Monitor execution
            with self.monitor.track_execution():
                result = await self.controller.execute_continuation(continuation)

            return ContinuationResult(
                success=True, state=result.state, metrics=result.metrics
            )

        except Exception as e:
            return ContinuationResult(success=False, error=str(e))

    async def process_execution(self) -> ExecutionResult:
        """
        _summary_

        _extended_summary_

        Returns:
            ExecutionResult: _description_
        """
        # TODO: Implement the process_execution logic
        try:
            # Validate current state
            state_valid = await self.validator.check_state()
            if not state_valid:
                raise ExecutionValidator("Invalid system state")

            # Initialize execution
            execution = await self.controller.initialize_execution()

            # Monitor execution
            with self.monitor.track_execution():
                result = await self.controller.execute_execution(execution)

            return ExecutionResult(
                success=True, state=result.state, metrics=result.metrics
            )

        except Exception as e:
            return ExecutionResult(success=False, error=str(e))
