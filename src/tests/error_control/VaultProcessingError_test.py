import pickle

import pytest

from src.error_control.VaultProcessingError import VaultProcessingError


class TestVaultProcessingError:
    """Test suite for the VaultProcessingError class."""

    @pytest.fixture
    def simple_error(self):
        """Create a simple VaultProcessingError instance for testing."""
        return VaultProcessingError(message="Test vault error")

    @pytest.fixture
    def detailed_error(self):
        """Create a VaultProcessingError instance with details for testing."""
        return VaultProcessingError(
            message="Vault access failed",
            details={
                "vault_id": "vault-456",
                "operation": "read",
                "status": "permission_denied",
            },
        )

    def test_init_simple(self, simple_error):
        """Test initialization with just a message."""
        assert simple_error.message == "Test vault error"
        assert (
            simple_error.details is None
        )  # Note that unlike other error classes, None is not converted to {}
        assert isinstance(simple_error, Exception)  # Verify inheritance from Exception

    def test_init_with_details(self, detailed_error):
        """Test initialization with message and details."""
        assert detailed_error.message == "Vault access failed"
        assert detailed_error.details == {
            "vault_id": "vault-456",
            "operation": "read",
            "status": "permission_denied",
        }
        assert isinstance(detailed_error, Exception)

    def test_str(self, simple_error, detailed_error):
        """Test __str__ method for proper string representation."""
        assert str(simple_error) == "Test vault error - None"
        assert (
            str(detailed_error)
            == "Vault access failed - {'vault_id': 'vault-456', 'operation': 'read', 'status': 'permission_denied'}"
        )

    def test_repr(self, simple_error, detailed_error):
        """Test __repr__ method for proper string representation."""
        assert repr(simple_error) == "Test vault error - None"
        assert (
            repr(detailed_error)
            == "Vault access failed - {'vault_id': 'vault-456', 'operation': 'read', 'status': 'permission_denied'}"
        )

    def test_equality(self, simple_error, detailed_error):
        """Test equality operator for proper comparison."""
        # Create identical errors
        identical_simple = VaultProcessingError(message="Test vault error")
        identical_detailed = VaultProcessingError(
            message="Vault access failed",
            details={
                "vault_id": "vault-456",
                "operation": "read",
                "status": "permission_denied",
            },
        )

        # Create different errors
        different_message = VaultProcessingError(message="Different message")
        different_details = VaultProcessingError(
            message="Vault access failed",
            details={
                "vault_id": "vault-789",
                "operation": "write",
                "status": "timeout",
            },
        )

        # Test equality
        assert simple_error == identical_simple
        assert detailed_error == identical_detailed
        assert simple_error != detailed_error
        assert simple_error != different_message
        assert detailed_error != different_details
        assert simple_error != "not an error"  # Compare to non-error object

    def test_hash(self, simple_error, detailed_error):
        """Test hash function for consistent hash values."""
        # Create identical errors
        identical_simple = VaultProcessingError(message="Test vault error")
        identical_detailed = VaultProcessingError(
            message="Vault access failed",
            details={
                "vault_id": "vault-456",
                "operation": "read",
                "status": "permission_denied",
            },
        )

        # Test hash equality for identical objects
        assert hash(simple_error) == hash(identical_simple)
        assert hash(detailed_error) == hash(identical_detailed)

        # Test hash inequality for different objects
        assert hash(simple_error) != hash(detailed_error)

    def test_as_exception(self):
        """Test using VaultProcessingError as an actual exception."""
        try:
            raise VaultProcessingError("Test exception")
        except VaultProcessingError as e:
            assert str(e) == "Test exception - None"
            assert isinstance(e, Exception)
        except Exception:
            pytest.fail("VaultProcessingError was not caught as Exception")

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
        error = VaultProcessingError("")
        assert error.message == ""
        assert str(error) == " - None"

    def test_empty_details_dict(self):
        """Test with empty details dictionary."""
        error = VaultProcessingError("Message", {})
        assert error.details == {}
        assert str(error) == "Message - {}"

    def test_details_with_complex_structure(self):
        """Test with complex nested details structure."""
        complex_details = {
            "metadata": {"vault": {"id": "v123", "access_levels": ["read", "write"]}},
            "errors": [
                {"code": 401, "message": "Unauthorized"},
                {"code": 404, "message": "Resource not found"},
            ],
            "timestamp": "2025-03-31T10:15:30Z",
        }

        error = VaultProcessingError("Complex error", complex_details)

        # Verify details are stored correctly
        assert error.details == complex_details

        # Verify string representation includes complex details
        assert str(error) == f"Complex error - {complex_details}"

    def test_exception_chaining(self):
        """Test that VaultProcessingError works with exception chaining."""
        try:
            try:
                # Raise an initial error
                raise ValueError("Initial error")
            except ValueError as initial_error:
                # Chain with VaultProcessingError
                raise VaultProcessingError(
                    "Vault error occurred",
                    {"initial_error_type": type(initial_error).__name__},
                ) from initial_error
        except VaultProcessingError as e:
            # Verify the error
            assert (
                str(e) == "Vault error occurred - {'initial_error_type': 'ValueError'}"
            )
            # Verify the chain
            assert isinstance(e.__cause__, ValueError)
            assert str(e.__cause__) == "Initial error"
