import pickle

import pytest

from src.error_control.ExecutionError import ExecutionError


class TestExecutionError:
    """Test suite for the ExecutionError class."""

    @pytest.fixture
    def basic_error(self):
        """Create a basic ExecutionError instance for testing."""
        return ExecutionError(message="Test error message")

    @pytest.fixture
    def detailed_error(self):
        """Create an ExecutionError instance with details for testing."""
        return ExecutionError(
            message="Detailed error message",
            details={
                "task_id": "task-123",
                "component": "processor",
                "status_code": 500,
            },
        )

    def test_init_default(self, basic_error):
        """Test initialization with default values."""
        assert basic_error.message == "Test error message"
        assert basic_error.details == {}

    def test_init_with_details(self, detailed_error):
        """Test initialization with custom details."""
        assert detailed_error.message == "Detailed error message"
        assert detailed_error.details == {
            "task_id": "task-123",
            "component": "processor",
            "status_code": 500,
        }

    def test_str(self, basic_error, detailed_error):
        """Test __str__ method."""
        assert str(basic_error) == "Test error message ({})"
        assert str(detailed_error) == (
            "Detailed error message ({'task_id': 'task-123', 'component': 'processor', 'status_code': 500})"
        )

    def test_repr(self, basic_error, detailed_error):
        """Test __repr__ method."""
        assert (
            repr(basic_error)
            == "ExecutionError(message=Test error message, details={})"
        )
        assert repr(detailed_error) == (
            "ExecutionError(message=Detailed error message, "
            "details={'task_id': 'task-123', 'component': 'processor', 'status_code': 500})"
        )

    def test_equality(self, basic_error, detailed_error):
        """Test equality and inequality methods."""
        # Create identical errors
        identical_basic = ExecutionError(message="Test error message")
        identical_detailed = ExecutionError(
            message="Detailed error message",
            details={
                "task_id": "task-123",
                "component": "processor",
                "status_code": 500,
            },
        )

        # Create different errors
        different_message = ExecutionError(message="Different message")
        different_details = ExecutionError(
            message="Detailed error message",
            details={
                "task_id": "task-999",
                "component": "executor",
                "status_code": 400,
            },
        )

        # Test equality
        assert basic_error == identical_basic
        assert detailed_error == identical_detailed
        assert basic_error != detailed_error
        assert basic_error != different_message
        assert detailed_error != different_details
        assert basic_error != "not an error"

        # Test inequality
        assert basic_error == identical_basic
        assert detailed_error == identical_detailed
        assert basic_error != detailed_error
        assert basic_error != different_message

    def test_hash(self, basic_error, detailed_error):
        """Test __hash__ method."""
        # Create identical errors
        identical_basic = ExecutionError(message="Test error message")
        identical_detailed = ExecutionError(
            message="Detailed error message",
            details={
                "task_id": "task-123",
                "component": "processor",
                "status_code": 500,
            },
        )

        # Create different error
        different_error = ExecutionError(message="Different message")

        # Test hash equality
        assert hash(basic_error) == hash(identical_basic)
        assert hash(detailed_error) == hash(identical_detailed)
        assert hash(basic_error) != hash(detailed_error)
        assert hash(basic_error) != hash(different_error)

    def test_reduce(self, detailed_error):
        """Test __reduce__ method."""
        # Get the reduction
        cls, args = detailed_error.__reduce__()

        # Verify
        assert cls is ExecutionError
        assert len(args) == 2
        assert args[0] == "Detailed error message"
        assert args[1] == {
            "task_id": "task-123",
            "component": "processor",
            "status_code": 500,
        }

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
            "task_id": "task-123",
            "component": "processor",
            "status_code": 500,
        }

    def test_setstate(self):
        """Test __setstate__ method."""
        # Create an empty error
        error = ExecutionError("")

        # Create a state to set
        state = {"message": "New message", "details": {"new": "details", "code": 123}}

        # Set the state
        error.__setstate__(state)

        # Verify
        assert error.message == "New message"
        assert error.details == {"new": "details", "code": 123}

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

    def test_init_with_none_details(self):
        """Test explicitly passing None for details."""
        error = ExecutionError("Message", None)
        assert error.details == {}  # Should convert None to empty dict

    def test_super_init_call(self, monkeypatch):
        """Test that the parent class's __init__ is called."""
        # Since we're not actually deriving from an exception, we need
        # to verify the super().__init__() call differently
        init_called = False

        original_init = object.__init__

        def mock_init(self, *args, **kwargs):
            nonlocal init_called
            init_called = True
            original_init(self)

        # Patch object.__init__
        monkeypatch.setattr(object, "__init__", mock_init)

        # Create a new instance
        ExecutionError("Test error")

        # Verify super().__init__ was called
        assert init_called

    def test_empty_message(self):
        """Test error with empty message."""
        error = ExecutionError("")
        assert error.message == ""
        assert str(error) == " ({})"

    def test_nested_details(self):
        """Test error with nested details."""
        nested_details = {"outer": {"inner": {"value": 42}}, "list": [1, 2, 3]}

        error = ExecutionError("Nested error", nested_details)

        # Verify details are stored correctly
        assert error.details == nested_details

        # Verify string representation includes nested details
        assert str(error) == f"Nested error ({nested_details})"
