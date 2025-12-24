import pickle

import pytest

from src.error_control.StateValidationError import StateValidationError


class TestStateValidationError:
    """Test suite for the StateValidationError class."""

    @pytest.fixture
    def basic_error(self):
        """Create a basic StateValidationError instance for testing."""
        return StateValidationError(message="Test error message")

    @pytest.fixture
    def detailed_error(self):
        """Create a StateValidationError instance with details for testing."""
        return StateValidationError(
            message="Detailed error message",
            details={"code": 123, "field": "username", "reason": "Invalid format"},
        )

    def test_init_default(self, basic_error):
        """Test initialization with default values."""
        assert basic_error.message == "Test error message"
        assert basic_error.details == {}
        # Verify it inherits from Exception
        assert isinstance(basic_error, Exception)

    def test_init_with_details(self, detailed_error):
        """Test initialization with custom details."""
        assert detailed_error.message == "Detailed error message"
        assert detailed_error.details == {
            "code": 123,
            "field": "username",
            "reason": "Invalid format",
        }

    def test_str(self, basic_error, detailed_error):
        """Test __str__ method."""
        assert str(basic_error) == "Test error message ({})"
        assert (
            str(detailed_error)
            == "Detailed error message ({'code': 123, 'field': 'username', 'reason': 'Invalid format'})"
        )

    def test_repr(self, basic_error, detailed_error):
        """Test __repr__ method."""
        assert (
            repr(basic_error)
            == "StateValidationError(message=Test error message, details={})"
        )
        assert (
            repr(detailed_error)
            == "StateValidationError(message=Detailed error message, details={'code': 123, 'field': 'username', 'reason': 'Invalid format'})"
        )

    def test_reduce(self, detailed_error):
        """Test __reduce__ method."""
        # Get the reduction
        cls, args = detailed_error.__reduce__()

        # Verify
        assert cls is StateValidationError
        assert len(args) == 2
        assert args[0] == "Detailed error message"
        assert args[1] == {"code": 123, "field": "username", "reason": "Invalid format"}

        # Test reconstruction
        reconstructed = cls(*args)
        assert reconstructed == detailed_error
        assert reconstructed is not detailed_error  # Not the same object

    def test_getstate(self, detailed_error):
        """Test __getstate__ method."""
        state = detailed_error.__getstate__()

        assert isinstance(state, dict)
        assert state["message"] == "Detailed error message"
        assert state["details"] == {
            "code": 123,
            "field": "username",
            "reason": "Invalid format",
        }

    def test_setstate(self):
        """Test __setstate__ method."""
        # Create an empty error
        error = StateValidationError("")

        # Create a state to set
        state = {"message": "New message", "details": {"new": "details"}}

        # Set the state
        error.__setstate__(state)

        # Verify
        assert error.message == "New message"
        assert error.details == {"new": "details"}

    def test_equality(self, basic_error, detailed_error):
        """Test equality methods."""
        # Same values
        identical_error = StateValidationError(message="Test error message")

        # Different values
        different_error = StateValidationError(message="Different message")

        # Test equality
        assert basic_error == identical_error
        assert basic_error != different_error
        assert basic_error != detailed_error
        assert basic_error != "not an error"

        # Test with same message but different details
        error_with_details = StateValidationError(
            message="Test error message", details={"some": "details"}
        )
        assert basic_error != error_with_details

    def test_hash(self, basic_error, detailed_error):
        """Test __hash__ method."""
        # Same values should have same hash
        identical_error = StateValidationError(message="Test error message")

        assert hash(basic_error) == hash(identical_error)

        # Different values should have different hash
        assert hash(basic_error) != hash(detailed_error)

    def test_pickle_serialization(self, detailed_error):
        """Test pickle serialization and deserialization."""
        # Pickle the error
        pickled = pickle.dumps(detailed_error)

        # Unpickle
        unpickled = pickle.loads(pickled)

        # Verify
        assert unpickled == detailed_error
        assert unpickled.message == detailed_error.message
        assert unpickled.details == detailed_error.details

    def test_as_exception(self):
        """Test using StateValidationError as an exception."""
        try:
            raise StateValidationError("Test exception")
        except StateValidationError as e:
            assert str(e) == "Test exception ({})"
            assert isinstance(e, Exception)
        except Exception:
            pytest.fail("StateValidationError was not caught as Exception")

    def test_with_empty_message(self):
        """Test creating an error with an empty message."""
        error = StateValidationError("")
        assert error.message == ""
        assert (
            str(error) == " ({})"
        )  # The string representation includes the empty message

    def test_with_none_details(self):
        """Test explicitly passing None for details."""
        error = StateValidationError("Message", None)
        assert error.details == {}  # Should convert None to empty dict
