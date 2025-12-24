import pickle

import pytest

from src.processors.EnhancedProcessingError import EnhancedProcessingError


class TestEnhancedProcessingError:
    """Test suite for the EnhancedProcessingError class."""

    @pytest.fixture
    def simple_error(self):
        """Create a simple EnhancedProcessingError instance for testing."""
        return EnhancedProcessingError(message="Simple error message")

    @pytest.fixture
    def detailed_error(self):
        """Create an EnhancedProcessingError instance with details for testing."""
        return EnhancedProcessingError(
            message="Detailed processing error",
            details={
                "file_id": "doc-123",
                "process_type": "markdown",
                "stage": "parsing",
            },
        )

    def test_init_simple(self, simple_error):
        """Test initialization with just a message."""
        assert simple_error.message == "Simple error message"
        assert simple_error.details is None  # Note that None is not converted to {}
        assert isinstance(simple_error, Exception)  # Verify inheritance from Exception

    def test_init_with_details(self, detailed_error):
        """Test initialization with message and details."""
        assert detailed_error.message == "Detailed processing error"
        assert detailed_error.details == {
            "file_id": "doc-123",
            "process_type": "markdown",
            "stage": "parsing",
        }
        assert isinstance(detailed_error, Exception)

    def test_str(self, simple_error, detailed_error):
        """Test __str__ method for proper string representation."""
        assert str(simple_error) == "Simple error message - None"
        assert str(detailed_error) == (
            "Detailed processing error - {'file_id': 'doc-123', 'process_type': 'markdown', 'stage': 'parsing'}"
        )

    def test_repr(self, simple_error, detailed_error):
        """Test __repr__ method for proper string representation."""
        assert repr(simple_error) == "Simple error message - None"
        assert repr(detailed_error) == (
            "Detailed processing error - {'file_id': 'doc-123', 'process_type': 'markdown', 'stage': 'parsing'}"
        )

    def test_equality(self, simple_error, detailed_error):
        """Test equality operator for proper comparison."""
        # Create identical errors
        identical_simple = EnhancedProcessingError(message="Simple error message")
        identical_detailed = EnhancedProcessingError(
            message="Detailed processing error",
            details={
                "file_id": "doc-123",
                "process_type": "markdown",
                "stage": "parsing",
            },
        )

        # Create different errors
        different_message = EnhancedProcessingError(message="Different message")
        different_details = EnhancedProcessingError(
            message="Detailed processing error",
            details={
                "file_id": "doc-456",
                "process_type": "html",
                "stage": "rendering",
            },
        )

        # Test equality
        assert simple_error == identical_simple
        assert detailed_error == identical_detailed
        assert simple_error != detailed_error
        assert simple_error != different_message
        assert detailed_error != different_details
        assert simple_error != "not an error"  # Compare to non-error object

        # Test inequality explicitly
        assert simple_error != detailed_error
        assert simple_error != detailed_error

    def test_hash(self, simple_error, detailed_error):
        """Test hash function for consistent hash values."""
        # Create identical errors
        identical_simple = EnhancedProcessingError(message="Simple error message")
        identical_detailed = EnhancedProcessingError(
            message="Detailed processing error",
            details={
                "file_id": "doc-123",
                "process_type": "markdown",
                "stage": "parsing",
            },
        )

        # Test hash equality for identical objects
        assert hash(simple_error) == hash(identical_simple)
        assert hash(detailed_error) == hash(identical_detailed)

        # Test hash inequality for different objects
        assert hash(simple_error) != hash(detailed_error)

    def test_comparison_operators(self):
        """Test all comparison operators."""
        # Create errors with different messages for comparison testing
        error_a = EnhancedProcessingError("A message")
        error_b = EnhancedProcessingError("B message")
        error_a_copy = EnhancedProcessingError("A message")

        # Test less than
        assert error_a < error_b
        assert not error_b < error_a
        assert not error_a < error_a_copy
        assert error_a >= "not an error"

        # Test less than or equal
        assert error_a <= error_b
        assert not error_b <= error_a
        assert error_a <= error_a_copy
        assert error_a > "not an error"

        # Test greater than
        assert error_b > error_a
        assert not error_a > error_b
        assert not error_a > error_a_copy
        assert error_a <= "not an error"

        # Test greater than or equal
        assert error_b >= error_a
        assert not error_a >= error_b
        assert error_a >= error_a_copy
        assert error_a < "not an error"

    def test_as_exception(self):
        """Test using EnhancedProcessingError as an actual exception."""
        try:
            raise EnhancedProcessingError("Test exception")
        except EnhancedProcessingError as e:
            assert str(e) == "Test exception - None"
            assert isinstance(e, Exception)
        except Exception:
            pytest.fail("EnhancedProcessingError was not caught as Exception")

    def test_pickle_serialization(self, detailed_error):
        """Test that the error can be properly serialized with pickle."""
        # Pickle the error
        pickled = pickle.dumps(detailed_error)

        # Unpickle
        unpickled = pickle.loads(pickled)

        # Verify
        assert unpickled == detailed_error
        assert unpickled.message == detailed_error.message
        assert unpickled.details == detailed_error.details

    def test_empty_message(self):
        """Test with empty message."""
        error = EnhancedProcessingError("")
        assert error.message == ""
        assert str(error) == " - None"

    def test_empty_details_dict(self):
        """Test with empty details dictionary."""
        error = EnhancedProcessingError("Message", {})
        assert error.details == {}
        assert str(error) == "Message - {}"

    def test_details_with_complex_structure(self):
        """Test with complex nested details structure."""
        complex_details = {
            "file": {
                "path": "/path/to/file.md",
                "size": 1024,
                "metadata": {
                    "author": "User",
                    "created_at": "2025-03-31T03:38:56-07:00",
                },
            },
            "processing": {
                "steps": ["parse", "analyze", "transform"],
                "status": "failed",
                "error_codes": [101, 203],
            },
        }

        error = EnhancedProcessingError("Complex processing error", complex_details)

        # Verify details are stored correctly
        assert error.details == complex_details

        # Verify string representation includes complex details
        assert str(error) == f"Complex processing error - {complex_details}"

    def test_exception_chaining(self):
        """Test that EnhancedProcessingError works with exception chaining."""
        try:
            try:
                # Raise an initial error
                raise ValueError("Initial processing error")
            except ValueError as initial_error:
                # Chain with EnhancedProcessingError
                raise EnhancedProcessingError(
                    "Enhanced processing failed",
                    {"original_error_type": type(initial_error).__name__},
                ) from initial_error
        except EnhancedProcessingError as e:
            # Verify the error
            assert (
                str(e)
                == "Enhanced processing failed - {'original_error_type': 'ValueError'}"
            )
            # Verify the chain
            assert isinstance(e.__cause__, ValueError)
            assert str(e.__cause__) == "Initial processing error"
