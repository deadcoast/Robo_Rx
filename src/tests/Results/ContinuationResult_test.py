import copy
import pickle
from datetime import datetime

import pytest

from src.Results.ContinuationResult import ContinuationResult


class TestContinuationResult:
    """Test suite for the ContinuationResult class."""

    @pytest.fixture
    def sample_metrics(self):
        """Create sample metrics for testing."""
        return {
            "processing_time": 234.56,
            "resources_used": {"cpu": "45%", "memory": "512MB"},
            "continuation_count": 3,
        }

    @pytest.fixture
    def sample_timestamp(self):
        """Create a fixed timestamp for testing."""
        return datetime(2025, 3, 31, 4, 1, 47)

    @pytest.fixture
    def success_result(self, sample_metrics, sample_timestamp):
        """Create a successful ContinuationResult for testing."""
        return ContinuationResult(
            success=True, metrics=sample_metrics, error=None, timestamp=sample_timestamp
        )

    @pytest.fixture
    def error_result(self, sample_timestamp):
        """Create a ContinuationResult with error for testing."""
        return ContinuationResult(
            success=False,
            metrics={},
            error="Continuation failed: network error",
            timestamp=sample_timestamp,
        )

    def test_init_default_metrics(self):
        """Test initialization with default metrics value."""
        result = ContinuationResult(success=True)

        assert result.success is True
        assert result.metrics == {}  # Default empty dictionary
        assert result.error is None
        assert isinstance(result.timestamp, datetime)

    def test_init_with_values(self, sample_metrics, sample_timestamp):
        """Test initialization with provided values."""
        result = ContinuationResult(
            success=True, metrics=sample_metrics, error=None, timestamp=sample_timestamp
        )

        assert result.success is True
        assert result.metrics == sample_metrics
        assert result.error is None
        assert result.timestamp == sample_timestamp

    def test_str(self, success_result, error_result):
        """Test __str__ method for string representation."""
        success_str = str(success_result)
        assert "ContinuationResult(success=True" in success_str
        assert "metrics={'processing_time': 234.56" in success_str
        assert "error=None" in success_str
        assert "timestamp=2025-03-31" in success_str

        error_str = str(error_result)
        assert "ContinuationResult(success=False" in error_str
        assert "metrics={}" in error_str
        assert "error=Continuation failed: network error" in error_str

    def test_repr(self, success_result):
        """Test __repr__ method for string representation."""
        repr_str = repr(success_result)
        str_value = str(success_result)

        # __repr__ should return the same as __str__
        assert repr_str == str_value

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

    def test_getstate(self, success_result):
        """Test __getstate__ method for serialization."""
        state = success_result.__getstate__()

        assert isinstance(state, dict)
        assert "success" in state and state["success"] is True
        assert "metrics" in state and state["metrics"] == success_result.metrics
        assert "error" in state and state["error"] is None
        assert "timestamp" in state and state["timestamp"] == success_result.timestamp

    def test_setstate(self, sample_metrics, sample_timestamp):
        """Test __setstate__ method for deserialization."""
        # Create a result with different values
        result = ContinuationResult(success=False)

        # Create a state to set
        state = {
            "success": True,
            "metrics": sample_metrics,
            "error": None,
            "timestamp": sample_timestamp,
        }

        # Set the state
        result.__setstate__(state)

        # Verify
        assert result.success is True
        assert result.metrics == sample_metrics
        assert result.error is None
        assert result.timestamp == sample_timestamp

    def test_equality(
        self, success_result, error_result, sample_metrics, sample_timestamp
    ):
        """Test equality operator."""
        # Create an identical result
        identical_result = ContinuationResult(
            success=True, metrics=sample_metrics, error=None, timestamp=sample_timestamp
        )

        # Test equality
        assert success_result == identical_result
        assert success_result != error_result
        assert success_result != "not a result"

        # Test with slightly different values
        slightly_different = ContinuationResult(
            success=True,
            metrics=sample_metrics.copy(),
            error="Minor error",  # Different error message
            timestamp=sample_timestamp,
        )
        assert success_result != slightly_different

    def test_hash(self, success_result, sample_metrics, sample_timestamp):
        """Test __hash__ method."""
        # Create an identical result
        identical_result = ContinuationResult(
            success=True, metrics=sample_metrics, error=None, timestamp=sample_timestamp
        )

        # Test hash equality
        assert hash(success_result) == hash(identical_result)

        # Create a different result
        different_result = ContinuationResult(
            success=False, metrics={}, error="Error", timestamp=sample_timestamp
        )

        # Test hash inequality
        assert hash(success_result) != hash(different_result)

    def test_reduce(self, success_result):
        """Test __reduce__ method for pickling support."""
        # Get the reduction
        cls, args = success_result.__reduce__()

        # Verify
        assert cls is ContinuationResult
        assert len(args) == 4
        assert args[0] is success_result.success
        assert args[1] is success_result.metrics
        assert args[2] is success_result.error
        assert args[3] is success_result.timestamp

        # Test reconstruction
        reconstructed = cls(*args)
        assert reconstructed == success_result

    def test_reduce_ex(self, success_result):
        """Test __reduce_ex__ method for pickling support."""
        # Get the reduction with protocol 4
        result1 = success_result.__reduce__()
        result2 = success_result.__reduce_ex__(4)

        # Verify both methods return the same result
        assert result1 == result2

    def test_pickle_serialization(self, success_result):
        """Test that ContinuationResult can be pickled and unpickled."""
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
            "execution": {
                "stages": [
                    {"name": "initialization", "duration": 1.2, "status": "completed"},
                    {"name": "processing", "duration": 5.7, "status": "completed"},
                    {"name": "finalization", "duration": 0.8, "status": "completed"},
                ],
                "total_duration": 7.7,
            },
            "resources": {
                "network": {"bytes_sent": 1024, "bytes_received": 2048},
                "storage": {"read_operations": 15, "write_operations": 8},
            },
            "flags": {"retry_needed": False, "cache_used": True},
        }

        result = ContinuationResult(success=True, metrics=complex_metrics)

        # Verify metrics are stored correctly
        assert result.metrics == complex_metrics

        # Test deep copying with complex metrics
        copied = copy.deepcopy(result)
        assert copied.metrics == complex_metrics
        assert copied.metrics is not complex_metrics  # Different object

    def test_with_different_error_types(self):
        """Test with different types of error values."""
        # String error
        string_error_result = ContinuationResult(
            success=False, error="String error message"
        )
        assert string_error_result.error == "String error message"

        # Exception error
        exception = ValueError("Value error in continuation")
        exception_error_result = ContinuationResult(success=False, error=exception)
        assert exception_error_result.error == exception

        # Dictionary error
        dict_error = {"code": 500, "message": "Internal error", "retry": True}
        dict_error_result = ContinuationResult(success=False, error=dict_error)
        assert dict_error_result.error == dict_error

    def test_mutability(self, success_result):
        """Test that ContinuationResult attributes can be modified."""
        # Initial state
        assert success_result.success is True
        assert success_result.error is None

        # Modify attributes
        success_result.success = False
        success_result.error = "Modified error"
        success_result.metrics["new_key"] = "new_value"

        # Verify changes
        assert not success_result.success
        assert success_result.error == "Modified error"
        assert success_result.metrics["new_key"] == "new_value"

    def test_default_timestamp_changes(self):
        """Test that default timestamp factory creates different timestamps."""
        result1 = ContinuationResult(success=True)

        # Small delay
        import time

        time.sleep(0.001)

        result2 = ContinuationResult(success=True)

        # Timestamps should be different
        assert result1.timestamp != result2.timestamp
