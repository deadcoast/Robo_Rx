import copy
import pickle
from datetime import datetime

import pytest

from src.Results.ChainResult import ChainResult


class TestChainResult:
    """Test suite for the ChainResult class."""

    @pytest.fixture
    def sample_metrics(self):
        """Create sample metrics for testing."""
        return {
            "duration": 123.45,
            "steps_completed": 5,
            "memory_usage": "256MB",
            "tasks": ["task1", "task2", "task3"],
        }

    @pytest.fixture
    def sample_timestamp(self):
        """Create a fixed timestamp for testing."""
        return datetime(2025, 3, 31, 4, 1, 47)

    @pytest.fixture
    def success_result(self, sample_metrics, sample_timestamp):
        """Create a successful ChainResult for testing."""
        return ChainResult(
            success=True, metrics=sample_metrics, error="", timestamp=sample_timestamp
        )

    @pytest.fixture
    def error_result(self, sample_timestamp):
        """Create a ChainResult with error for testing."""
        return ChainResult(
            success=False,
            metrics=None,
            error="Task chain execution failed: timeout",
            timestamp=sample_timestamp,
        )

    def test_init_default(self):
        """Test initialization with default values."""
        result = ChainResult()

        assert result.success is False
        assert result.metrics is None
        assert result.error == ""
        assert isinstance(result.timestamp, datetime)

    def test_init_with_values(self, sample_metrics, sample_timestamp):
        """Test initialization with provided values."""
        result = ChainResult(
            success=True, metrics=sample_metrics, error="", timestamp=sample_timestamp
        )

        assert result.success is True
        assert result.metrics == sample_metrics
        assert result.error == ""
        assert result.timestamp == sample_timestamp

    def test_str(self, success_result, error_result):
        """Test __str__ method for string representation."""
        success_str = str(success_result)
        assert "ChainResult(success=True" in success_str
        assert "metrics={'duration': 123.45" in success_str
        assert "error=" in success_str
        assert "timestamp=2025-03-31" in success_str

        error_str = str(error_result)
        assert "ChainResult(success=False" in error_str
        assert "metrics=None" in error_str
        assert "error=Task chain execution failed: timeout" in error_str

    def test_repr(self, success_result):
        """Test __repr__ method for string representation."""
        repr_str = repr(success_result)
        str_value = str(success_result)

        # __repr__ should return the same as __str__
        assert repr_str == str_value

    def test_reduce(self, success_result):
        """Test __reduce__ method for pickling support."""
        # Get the reduction
        cls, args = success_result.__reduce__()

        # Verify
        assert cls is ChainResult
        assert len(args) == 4
        assert args[0] is success_result.success
        assert args[1] is success_result.metrics
        assert args[2] is success_result.error
        assert args[3] is success_result.timestamp

        # Test reconstruction
        reconstructed = cls(*args)
        assert reconstructed == success_result

    def test_getstate(self, success_result):
        """Test __getstate__ method for serialization."""
        state = success_result.__getstate__()

        assert isinstance(state, dict)
        assert "success" in state and state["success"] is True
        assert "metrics" in state and state["metrics"] == success_result.metrics
        assert "error" in state and state["error"] == ""
        assert "timestamp" in state and state["timestamp"] == success_result.timestamp

    def test_setstate(self, sample_metrics, sample_timestamp):
        """Test __setstate__ method for deserialization."""
        # Create an empty result
        result = ChainResult()

        # Create a state to set
        state = {
            "success": True,
            "metrics": sample_metrics,
            "error": "",
            "timestamp": sample_timestamp,
        }

        # Set the state
        result.__setstate__(state)

        # Verify
        assert result.success is True
        assert result.metrics == sample_metrics
        assert result.error == ""
        assert result.timestamp == sample_timestamp

    def test_equality(
        self, success_result, error_result, sample_metrics, sample_timestamp
    ):
        """Test equality operator."""
        # Create an identical result
        identical_result = ChainResult(
            success=True, metrics=sample_metrics, error="", timestamp=sample_timestamp
        )

        # Test equality
        assert success_result == identical_result
        assert success_result != error_result
        assert success_result != "not a result"

        # Test with slightly different values
        slightly_different = ChainResult(
            success=True,
            metrics=sample_metrics,
            error="Small difference",  # Different error message
            timestamp=sample_timestamp,
        )
        assert success_result != slightly_different

    def test_hash(self, success_result, sample_metrics, sample_timestamp):
        """Test __hash__ method."""
        # Create an identical result
        identical_result = ChainResult(
            success=True, metrics=sample_metrics, error="", timestamp=sample_timestamp
        )

        # Test hash equality
        assert hash(success_result) == hash(identical_result)

        # Create a different result
        different_result = ChainResult(
            success=False, metrics=None, error="Error", timestamp=sample_timestamp
        )

        # Test hash inequality
        assert hash(success_result) != hash(different_result)

    def test_copy(self, success_result):
        """Test __copy__ method for shallow copying."""
        # Create a copy
        copied_result = copy.copy(success_result)

        # Verify it's a different object but with same values
        assert copied_result is not success_result
        assert copied_result == success_result

        # Verify metrics is deeply copied (not the same object)
        assert copied_result.metrics is not success_result.metrics

    def test_deepcopy(self, success_result):
        """Test __deepcopy__ method for deep copying."""
        # Create a deep copy
        deep_copied = copy.deepcopy(success_result)

        # Verify it's a different object but with same values
        assert deep_copied is not success_result
        assert deep_copied == success_result

        # Verify metrics is deeply copied (not the same object)
        assert deep_copied.metrics is not success_result.metrics

    def test_pickle_serialization(self, success_result):
        """Test that ChainResult can be pickled and unpickled."""
        # Pickle the result
        pickled = pickle.dumps(success_result)

        # Unpickle
        unpickled = pickle.loads(pickled)

        # Verify
        assert unpickled == success_result
        assert unpickled.success == success_result.success
        assert unpickled.metrics == success_result.metrics
        assert unpickled.error == success_result.error
        assert unpickled.timestamp == success_result.timestamp

    def test_with_complex_metrics(self):
        """Test with complex nested metrics structure."""
        complex_metrics = {
            "performance": {
                "cpu": {"usage": 75.5, "cores": 8},
                "memory": {"used": "2.5GB", "available": "8GB"},
            },
            "tasks": [
                {"id": "task1", "status": "completed", "duration": 10.5},
                {"id": "task2", "status": "completed", "duration": 15.2},
                {"id": "task3", "status": "failed", "error": "timeout"},
            ],
            "summary": {"total_tasks": 3, "completed": 2, "failed": 1},
        }

        result = ChainResult(success=True, metrics=complex_metrics)

        # Verify metrics are stored correctly
        assert result.metrics == complex_metrics

        # Test deep copying with complex metrics
        copied = copy.deepcopy(result)
        assert copied.metrics == complex_metrics
        assert copied.metrics is not complex_metrics  # Different object

    def test_mutability(self, success_result):
        """Test that ChainResult attributes can be modified."""
        # Initial state
        assert success_result.success is True

        # Modify attributes
        success_result.success = False
        success_result.error = "New error"

        # Verify changes
        assert not success_result.success
        assert success_result.error == "New error"
