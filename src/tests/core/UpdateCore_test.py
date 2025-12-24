import copy
from unittest.mock import AsyncMock

import pytest

from src.core.UpdateCore import UpdateProgress, UpdateTracker


class TestUpdateProgress:
    """Test suite for the UpdateProgress class."""

    def test_init(self):
        """Test initialization of UpdateProgress."""
        progress = UpdateProgress(
            component_id="test-component",
            update_status="IN_PROGRESS",
            completion_percentage=50.0,
            error_count=0,
        )

        # Verify all attributes are set correctly
        assert progress.component_id == "test-component"
        assert progress.update_status == "IN_PROGRESS"
        assert progress.completion_percentage == 50.0
        assert progress.error_count == 0

    def test_default_error_count(self):
        """Test default value for error_count."""
        progress = UpdateProgress(
            component_id="test-component",
            update_status="IN_PROGRESS",
            completion_percentage=50.0,
        )

        # Verify error_count has default value of 0
        assert progress.error_count == 0


class TestUpdateTracker:
    """Test suite for the UpdateTracker class."""

    @pytest.fixture
    def update_tracker(self):
        """Create an UpdateTracker instance for testing."""
        return UpdateTracker()

    def test_init(self, update_tracker):
        """Test initialization of UpdateTracker."""
        assert update_tracker.progress_map == {}

    @pytest.mark.asyncio
    async def test_track_update(self, update_tracker):
        """Test tracking an update."""
        # Call track_update
        await update_tracker.track_update("test-component", 75.0)

        # Verify progress is recorded
        assert "test-component" in update_tracker.progress_map
        progress = update_tracker.progress_map["test-component"]
        assert progress.component_id == "test-component"
        assert progress.update_status == "IN_PROGRESS"
        assert progress.completion_percentage == 75.0
        assert progress.error_count == 0

    @pytest.mark.asyncio
    async def test_track_multiple_updates(self, update_tracker):
        """Test tracking multiple updates."""
        # Track multiple components
        await update_tracker.track_update("component1", 25.0)
        await update_tracker.track_update("component2", 50.0)
        await update_tracker.track_update("component3", 75.0)

        # Verify all components are tracked
        assert len(update_tracker.progress_map) == 3
        assert update_tracker.progress_map["component1"].completion_percentage == 25.0
        assert update_tracker.progress_map["component2"].completion_percentage == 50.0
        assert update_tracker.progress_map["component3"].completion_percentage == 75.0

    @pytest.mark.asyncio
    async def test_update_existing_component(self, update_tracker):
        """Test updating an existing component."""
        # Track initial progress
        await update_tracker.track_update("test-component", 25.0)

        # Update the same component
        await update_tracker.track_update("test-component", 50.0)

        # Verify the component is updated, not duplicated
        assert len(update_tracker.progress_map) == 1
        assert (
            update_tracker.progress_map["test-component"].completion_percentage == 50.0
        )


class TestUpdateVerification:
    """Test suite for the UpdateVerification nested class."""

    @pytest.fixture
    def update_verification(self):
        """Create an UpdateVerification instance for testing."""
        return UpdateTracker.UpdateVerification()

    @pytest.fixture
    def update_verification_with_steps(self):
        """Create an UpdateVerification instance with verification steps."""
        verification_steps = [
            AsyncMock(return_value=True),
            AsyncMock(return_value=True),
        ]
        return UpdateTracker.UpdateVerification(verification_steps)

    def test_init_default(self, update_verification):
        """Test initialization with default parameters."""
        assert update_verification.verification_steps == []

    def test_init_with_steps(self):
        """Test initialization with verification steps."""
        step1 = AsyncMock(return_value=True)
        step2 = AsyncMock(return_value=False)
        verification = UpdateTracker.UpdateVerification([step1, step2])

        assert len(verification.verification_steps) == 2
        assert verification.verification_steps[0] is step1
        assert verification.verification_steps[1] is step2

    def test_add_verification_step(self, update_verification):
        """Test adding a verification step."""
        # Create a mock step
        step = AsyncMock(return_value=True)

        # Add the step
        update_verification.add_verification_step(step)

        # Verify the step was added
        assert len(update_verification.verification_steps) == 1
        assert update_verification.verification_steps[0] is step

    def test_add_duplicate_verification_step(self, update_verification):
        """Test adding the same verification step twice."""
        # Create a mock step
        step = AsyncMock(return_value=True)

        # Add the step twice
        update_verification.add_verification_step(step)
        update_verification.add_verification_step(step)

        # Verify the step was added only once
        assert len(update_verification.verification_steps) == 1
        assert update_verification.verification_steps[0] is step

    def test_remove_verification_step(self, update_verification_with_steps):
        """Test removing a verification step."""
        # Get the step to remove
        step = update_verification_with_steps.verification_steps[0]

        # Remove the step
        update_verification_with_steps.remove_verification_step(step)

        # Verify the step was removed
        assert len(update_verification_with_steps.verification_steps) == 1
        assert step not in update_verification_with_steps.verification_steps

    def test_remove_nonexistent_step(self, update_verification):
        """Test removing a step that doesn't exist."""
        # Create a mock step
        step = AsyncMock()

        # Remove the step that wasn't added
        update_verification.remove_verification_step(step)

        # Verify no errors and no change
        assert update_verification.verification_steps == []

    @pytest.mark.asyncio
    async def test_verify_update_success(self, update_verification_with_steps):
        """Test successful verification."""
        # Call verify_update
        result = await update_verification_with_steps.verify_update(
            "test-component", "test-result"
        )

        # Verify all steps were called and result is True
        assert result is True
        for step in update_verification_with_steps.verification_steps:
            step.assert_called_once_with("test-component", "test-result")

    @pytest.mark.asyncio
    async def test_verify_update_failure(self):
        """Test verification with a failing step."""
        # Create steps with one that fails
        steps = [
            AsyncMock(return_value=True),
            AsyncMock(return_value=False),
            AsyncMock(return_value=True),
        ]
        verification = UpdateTracker.UpdateVerification(steps)

        # Call verify_update
        result = await verification.verify_update("test-component", "test-result")

        # Verify result is False and all steps were called up to the failing one
        assert result is False
        steps[0].assert_called_once()
        steps[1].assert_called_once()
        steps[2].assert_not_called()  # Should not be called after a failure

    @pytest.mark.asyncio
    async def test_run_verification_empty_steps(self, update_verification):
        """Test running verification with no steps."""
        # Call _run_verification directly
        result = await update_verification._run_verification(
            "test-component", "test-result"
        )

        # Verify result is True when no steps to run
        assert result is True

    def test_copy(self, update_verification_with_steps):
        """Test copy method."""
        # Create a copy
        copied = copy.copy(update_verification_with_steps)

        # Verify it's a different instance with the same steps
        assert copied is not update_verification_with_steps
        assert (
            copied.verification_steps
            == update_verification_with_steps.verification_steps
        )

    def test_deepcopy(self, update_verification_with_steps):
        """Test deepcopy method."""
        # Create a deep copy
        deep_copied = copy.deepcopy(update_verification_with_steps)

        # Verify it's a different instance with copied steps
        assert deep_copied is not update_verification_with_steps
        assert (
            deep_copied.verification_steps
            is not update_verification_with_steps.verification_steps
        )
        assert len(deep_copied.verification_steps) == len(
            update_verification_with_steps.verification_steps
        )

    def test_reduce(self, update_verification_with_steps):
        """Test reduce method."""
        # Get reduce result
        reduce_result = update_verification_with_steps.__reduce__()

        # Verify it's the correct class and arguments
        assert reduce_result[0] == UpdateTracker.UpdateVerification
        assert isinstance(reduce_result[1], tuple)
        assert len(reduce_result[1]) == 1  # Should contain the verification_steps

    def test_getstate(self, update_verification_with_steps):
        """Test getstate method."""
        # Get state
        state = update_verification_with_steps.__getstate__()

        # Verify state contains the verification steps
        assert "verification_steps" in state
        assert (
            state["verification_steps"]
            == update_verification_with_steps.verification_steps
        )

    def test_setstate(self, update_verification):
        """Test setstate method."""
        # Create a state
        steps = [AsyncMock(), AsyncMock()]
        state = {"verification_steps": steps}

        # Set state
        update_verification.__setstate__(state)

        # Verify state was set
        assert update_verification.verification_steps == steps

    def test_setstate_missing_key(self, update_verification):
        """Test setstate method with missing verification_steps key."""
        # Create an empty state
        state = {}

        # Set state
        update_verification.__setstate__(state)

        # Verify default empty list is used
        assert update_verification.verification_steps == []
