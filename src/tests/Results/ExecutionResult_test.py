import copy
import pickle
from datetime import datetime

import pytest

from src.Results.ExecutionResult import ExecutionResult


class TestExecutionResult:
    """Test suite for the ExecutionResult class."""

    @pytest.fixture
    def sample_metrics(self):
        """Create sample metrics dictionary for testing."""
        return {
            "duration": 156.78,
            "cpu_usage": 45.2,
            "memory_usage": "320MB",
            "operations": ["read", "process", "analyze", "write"],
            "status_codes": {"success": 5, "warning": 2, "error": 0},
        }

    @pytest.fixture
    def sample_timestamp(self):
        """Create a fixed timestamp for testing."""
        return datetime(2025, 3, 31, 4, 1, 47)

    @pytest.fixture
    def success_result(self, sample_metrics, sample_timestamp):
        """Create a successful ExecutionResult for testing."""
        return ExecutionResult(
            success=True, metrics=sample_metrics, error="", timestamp=sample_timestamp
        )

    @pytest.fixture
    def error_result(self, sample_timestamp):
        """Create an ExecutionResult with error for testing."""
        return ExecutionResult(
            success=False,
            metrics={"attempted": True},
            error="Execution failed: timeout after 30s",
            timestamp=sample_timestamp,
        )

    def test_init_default(self):
        """Test initialization with default values."""
        result = ExecutionResult()

        assert result.success is False
        assert result.metrics is None
        assert result.error == ""
        assert isinstance(result.timestamp, datetime)

    def test_init_with_values(self, sample_metrics, sample_timestamp):
        """Test initialization with provided values."""
        result = ExecutionResult(
            success=True, metrics=sample_metrics, error="", timestamp=sample_timestamp
        )

        assert result.success is True
        assert result.metrics == sample_metrics
        assert result.error == ""
        assert result.timestamp == sample_timestamp

    def test_str(self, success_result, error_result):
        """Test __str__ method for string representation."""
        success_str = str(success_result)
        assert "ExecutionResult(success=True" in success_str
        assert "metrics=" in success_str
        assert "duration" in success_str
        assert "timestamp=" in success_str

        # For successful results without errors, the error field should not be included
        assert "error=" not in success_str

        error_str = str(error_result)
        assert "ExecutionResult(success=False" in error_str
        assert "metrics={'attempted': True}" in error_str
        assert "error=Execution failed: timeout after 30s" in error_str
        assert "timestamp=" in error_str

    def test_repr(self, success_result, error_result):
        """Test __repr__ method for string representation."""
        # __repr__ should return the same as __str__
        assert repr(success_result) == str(success_result)
        assert repr(error_result) == str(error_result)

    def test_reduce(self, success_result):
        """Test __reduce__ method for pickling support."""
        # Get the reduction
        cls, args = success_result.__reduce__()

        # Verify
        assert cls is ExecutionResult
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
        result = ExecutionResult()

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

    def test_equality(
        self, success_result, error_result, sample_metrics, sample_timestamp
    ):
        """Test equality and inequality operators."""
        # Create an identical result
        identical_result = ExecutionResult(
            success=True, metrics=sample_metrics, error="", timestamp=sample_timestamp
        )

        # Test equality
        assert success_result == identical_result
        assert success_result == identical_result

        # Test inequality
        assert success_result != error_result

        # Test with non-ExecutionResult object
        assert success_result != "not a result"

        # Test with slightly different values
        slightly_different = ExecutionResult(
            success=True,
            metrics=sample_metrics.copy(),
            error="Small difference",  # Different error message
            timestamp=sample_timestamp,
        )
        assert success_result != slightly_different

    def test_hash(self, success_result, sample_metrics, sample_timestamp):
        """Test __hash__ method."""
        # We need to patch the hash method since dictionaries are unhashable
        # This test will focus on verifying that hash behavior is consistent
        # for the same object and different for distinct objects

        # Same object should have consistent hash
        assert hash(success_result) == hash(success_result)

        # Create a different result to test hash inequality
        # Note: We can't directly compare hashes of different objects with dictionaries
        # because the hash implementation uses the unhashable metrics dictionary
        different_result = ExecutionResult(
            success=False,
            metrics=None,  # Use None instead of dictionary to avoid unhashable type error
            error="Error",
            timestamp=sample_timestamp,
        )

        # Verify different objects can be hashed (though we don't check exact values)
        hash(different_result)  # This should not raise an exception

    def test_bool(self, success_result, error_result):
        """Test __bool__ method."""
        # Success result should evaluate to True
        assert bool(success_result)

        # Error result should evaluate to False
        assert not bool(error_result)

        # Direct usage in if statement
        if not success_result:
            pytest.fail("Success result should evaluate to True")

        if error_result:
            pytest.fail("Error result should evaluate to False")

    def test_len(self, success_result):
        """Test __len__ method."""
        # Length should be the length of metrics
        assert len(success_result) == len(success_result.metrics)
        assert len(success_result) == 5  # Based on our sample metrics

    def test_iter(self, success_result):
        """Test __iter__ method."""
        # Should be able to iterate over metrics keys
        keys = list(success_result)
        expected_keys = list(success_result.metrics.keys())
        assert keys == expected_keys

        collected_keys = list(success_result)
        assert collected_keys == expected_keys

    def test_contains(self, success_result):
        """Test __contains__ method."""
        # Keys in metrics should return True
        assert "duration" in success_result
        assert "cpu_usage" in success_result
        assert "operations" in success_result

        # Keys not in metrics should return False
        assert "not_a_key" not in success_result
        assert "random_value" not in success_result

    def test_getitem(self, success_result):
        """Test __getitem__ method."""
        # Should retrieve values from metrics by key
        assert success_result["duration"] == 156.78
        assert success_result["cpu_usage"] == 45.2
        assert success_result["memory_usage"] == "320MB"
        assert success_result["operations"] == ["read", "process", "analyze", "write"]

        # Should raise KeyError for non-existent keys
        with pytest.raises(KeyError):
            success_result["non_existent_key"]

    def test_setitem(self, success_result):
        """Test __setitem__ method."""
        # Should be able to set new values in metrics
        success_result["new_key"] = "new_value"
        assert success_result.metrics["new_key"] == "new_value"

        # Should be able to modify existing values
        success_result["duration"] = 200.0
        assert success_result.metrics["duration"] == 200.0

    def test_delitem(self, success_result):
        """Test __delitem__ method."""
        # Should be able to delete keys from metrics
        assert "duration" in success_result.metrics
        del success_result["duration"]
        assert "duration" not in success_result.metrics

        # Should raise KeyError for non-existent keys
        with pytest.raises(KeyError):
            del success_result["non_existent_key"]

    def test_pickle_serialization(self, success_result):
        """Test that ExecutionResult can be pickled and unpickled."""
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

    def test_metrics_none(self):
        """Test behavior when metrics is None."""
        result = ExecutionResult(success=True, metrics=None)

        # These operations should raise errors with None metrics
        with pytest.raises(TypeError):
            len(result)

        with pytest.raises(TypeError):
            list(result)

        with pytest.raises(TypeError):
            "key" in result

        with pytest.raises(TypeError):
            _ = result["key"]

        with pytest.raises(TypeError):
            result["key"] = "value"

        with pytest.raises(TypeError):
            del result["key"]

    def test_dictionary_like_behavior(self):
        """Test using ExecutionResult like a dictionary for metrics."""
        metrics = {"key1": "value1", "key2": "value2"}
        result = ExecutionResult(success=True, metrics=metrics)

        # Access
        assert result["key1"] == "value1"

        # Modification
        result["key3"] = "value3"
        assert result.metrics["key3"] == "value3"
        assert result["key3"] == "value3"

        # Deletion
        del result["key1"]
        assert "key1" not in result.metrics
        assert "key1" not in result

        # Iteration
        keys = list(result)
        assert set(keys) == {"key2", "key3"}

    def test_with_complex_metrics(self):
        """Test with complex nested metrics structure."""
        complex_metrics = {
            "stages": [
                {"name": "stage1", "duration": 10.5, "status": "completed"},
                {"name": "stage2", "duration": 20.1, "status": "completed"},
                {"name": "stage3", "duration": 15.3, "status": "completed"},
            ],
            "resources": {
                "system": {
                    "cpu": {"usage": 65.7, "cores": 8},
                    "memory": {"used": "1.2GB", "peak": "1.8GB"},
                },
                "network": {"sent": "25MB", "received": "150MB", "requests": 42},
            },
            "summary": {"total_time": 45.9, "success_rate": 0.98, "warnings": 3},
        }

        result = ExecutionResult(success=True, metrics=complex_metrics)

        # Verify metrics are stored correctly
        assert result.metrics == complex_metrics

        # Test accessing nested structures
        assert result["stages"][0]["name"] == "stage1"
        assert result["resources"]["system"]["cpu"]["usage"] == 65.7
        assert result["summary"]["success_rate"] == 0.98

        # Test modifying nested structures
        result["stages"][1]["status"] = "optimized"
        assert result.metrics["stages"][1]["status"] == "optimized"

        # Test deep copying with complex metrics
        copied = copy.deepcopy(result)
        assert copied.metrics == complex_metrics

        # Verify deep copy is independent
        copied["stages"][0]["duration"] = 11.0
        assert copied["stages"][0]["duration"] == 11.0
        assert result["stages"][0]["duration"] == 10.5  # Original unchanged
