import pytest

from src.Continuation import Continuation


class TestContinuation:
    """Test suite for the Continuation class."""

    @pytest.fixture
    def sample_continuation(self):
        """Create a sample Continuation instance for testing."""
        return Continuation(
            status="pending",
            details="Awaiting processing completion",
            timestamp="2025-03-31T04:01:47-07:00",
        )

    @pytest.fixture
    def another_continuation(self):
        """Create another Continuation instance for comparison testing."""
        return Continuation(
            status="completed",
            details="Processing finished successfully",
            timestamp="2025-03-31T04:10:00-07:00",
        )

    def test_init(self, sample_continuation):
        """Test initialization of Continuation."""
        assert sample_continuation.status == "pending"
        assert sample_continuation.details == "Awaiting processing completion"
        assert sample_continuation.timestamp == "2025-03-31T04:01:47-07:00"

    def test_str(self, sample_continuation):
        """Test __str__ method."""
        expected_str = "Status: pending, Details: Awaiting processing completion, Timestamp: 2025-03-31T04:01:47-07:00"
        assert str(sample_continuation) == expected_str

    def test_repr(self, sample_continuation):
        """Test __repr__ method."""
        expected_repr = "Status: pending, Details: Awaiting processing completion, Timestamp: 2025-03-31T04:01:47-07:00"
        assert repr(sample_continuation) == expected_repr

    def test_eq(self, sample_continuation, another_continuation):
        """Test equality operator."""
        # Same content should be equal
        identical_continuation = Continuation(
            status="pending",
            details="Awaiting processing completion",
            timestamp="2025-03-31T04:01:47-07:00",
        )
        assert sample_continuation == identical_continuation

        # Different continuations should not be equal
        assert sample_continuation != another_continuation

        # Comparison with non-Continuation object
        assert sample_continuation != "Not a continuation"
        assert sample_continuation != 42
        assert sample_continuation is not None

    def test_ne(self, sample_continuation, another_continuation):
        """Test inequality operator."""
        # Different continuations should be not equal
        assert sample_continuation != another_continuation

        # Same content should not be not equal
        identical_continuation = Continuation(
            status="pending",
            details="Awaiting processing completion",
            timestamp="2025-03-31T04:01:47-07:00",
        )
        assert sample_continuation == identical_continuation

    def test_hash(self, sample_continuation):
        """Test hash function."""
        # Same content should have same hash
        identical_continuation = Continuation(
            status="pending",
            details="Awaiting processing completion",
            timestamp="2025-03-31T04:01:47-07:00",
        )
        assert hash(sample_continuation) == hash(identical_continuation)

        # Different content should have different hash
        different_continuation = Continuation(
            status="failed",
            details="Awaiting processing completion",
            timestamp="2025-03-31T04:01:47-07:00",
        )
        assert hash(sample_continuation) != hash(different_continuation)

    def test_hash_consistency(self, sample_continuation):
        """Test hash consistency with dictionary usage."""
        # Create a dictionary with Continuation as key
        continuation_dict = {sample_continuation: "test_value"}

        # Create an identical continuation
        identical_continuation = Continuation(
            status="pending",
            details="Awaiting processing completion",
            timestamp="2025-03-31T04:01:47-07:00",
        )

        # Should be able to retrieve with identical object
        assert continuation_dict[identical_continuation] == "test_value"

    def test_with_various_statuses(self):
        """Test creating Continuation with different statuses."""
        statuses = ["pending", "processing", "completed", "failed", "cancelled"]

        for status in statuses:
            continuation = Continuation(
                status=status,
                details="Test details",
                timestamp="2025-03-31T04:01:47-07:00",
            )
            assert continuation.status == status

    def test_with_empty_values(self):
        """Test Continuation with empty values."""
        continuation = Continuation("", "", "")
        assert continuation.status == ""
        assert continuation.details == ""
        assert continuation.timestamp == ""

        # String representation should still work
        assert str(continuation) == "Status: , Details: , Timestamp: "

    def test_immutability_between_instances(self, sample_continuation):
        """Test that modifying one instance doesn't affect others."""
        # Create a copy by initializing with same values
        copy_continuation = Continuation(
            sample_continuation.status,
            sample_continuation.details,
            sample_continuation.timestamp,
        )

        # Modify the copy
        copy_continuation.status = "modified"

        # Original should remain unchanged
        assert sample_continuation.status == "pending"
        assert copy_continuation.status == "modified"

    def test_attribute_mutation(self, sample_continuation):
        """Test that Continuation attributes can be modified."""
        # Initial state
        assert sample_continuation.status == "pending"

        # Modify attributes
        sample_continuation.status = "completed"
        sample_continuation.details = "Process completed"
        sample_continuation.timestamp = "2025-04-01T00:00:00-07:00"

        # Verify changes
        assert sample_continuation.status == "completed"
        assert sample_continuation.details == "Process completed"
        assert sample_continuation.timestamp == "2025-04-01T00:00:00-07:00"
