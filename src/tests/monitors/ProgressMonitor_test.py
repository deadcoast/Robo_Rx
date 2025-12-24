import logging
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from src.monitors.ProgressMonitor import ProgressMonitor


class TestProgressMonitor:
    """Test suite for the ProgressMonitor class."""

    @pytest.fixture
    def progress_monitor(self):
        """Create a ProgressMonitor instance for testing."""
        with patch("logging.getLogger") as mock_get_logger:
            # Create a mock logger to avoid actual logging during tests
            mock_logger = Mock(spec=logging.Logger)
            mock_get_logger.return_value = mock_logger

            # Create and return ProgressMonitor instance
            monitor = ProgressMonitor()

            # Set the mocked logger so tests can access it
            monitor._logger = mock_logger
            return monitor

    def test_init(self, progress_monitor):
        """Test initialization of ProgressMonitor."""
        assert progress_monitor._logger is not None
        progress_monitor._logger.info.assert_called_once_with(
            "ProgressMonitor initialized"
        )

    def test_monitor(self, progress_monitor):
        """Test the monitor method."""
        # Setup
        test_path = Path("/test/vault")

        # Call the method
        result = progress_monitor.monitor(test_path)

        # Verify
        progress_monitor._logger.info.assert_called_with(
            "Monitoring progress for %s", test_path
        )
        assert result == {"progress": 0}

    def test_get_progress(self, progress_monitor):
        """Test the get_progress method."""
        # Setup
        test_path = Path("/test/vault")

        # Call the method
        result = progress_monitor.get_progress(test_path)

        # Verify
        progress_monitor._logger.info.assert_called_with(
            "Getting progress for %s", test_path
        )
        assert result == {"progress": 0}

    def test_set_progress(self, progress_monitor):
        """Test the set_progress method."""
        # Setup
        test_path = Path("/test/vault")
        test_progress = 50

        # Call the method
        progress_monitor.set_progress(test_path, test_progress)

        # Verify
        progress_monitor._logger.info.assert_called_with(
            "Setting progress for %s to %s", test_path, test_progress
        )

    def test_reset_progress(self, progress_monitor):
        """Test the reset_progress method."""
        # Setup
        test_path = Path("/test/vault")

        # Call the method
        progress_monitor.reset_progress(test_path)

        # Verify
        progress_monitor._logger.info.assert_called_with(
            "Resetting progress for %s", test_path
        )

    def test_context_manager(self, progress_monitor):
        """Test the context manager interface (__enter__ and __exit__ methods)."""
        # Use the context manager
        with progress_monitor as monitor:
            assert monitor is progress_monitor
            progress_monitor._logger.info.assert_called_with("Entering ProgressMonitor")

        # Verify __exit__ was called
        progress_monitor._logger.info.assert_called_with("Exiting ProgressMonitor")

    def test_str_and_repr(self, progress_monitor):
        """Test the __str__ and __repr__ methods."""
        # Setup
        progress_monitor._logger.name = "test_logger"

        # Test __str__
        str_result = str(progress_monitor)
        assert str_result == "ProgressMonitor: test_logger"

        # Test __repr__
        repr_result = repr(progress_monitor)
        assert repr_result == "ProgressMonitor: test_logger"

    def test_del(self, progress_monitor):
        """Test the __del__ method."""
        # Call __del__ directly
        progress_monitor.__del__()

        # Verify
        progress_monitor._logger.info.assert_called_with("ProgressMonitor deleted")

    def test_delattr(self, progress_monitor):
        """Test the __delattr__ method."""
        # Setup - add an attribute to delete
        progress_monitor.test_attr = "test_value"

        # Use __delattr__
        with patch.object(progress_monitor, "_logger") as mock_logger:
            delattr(progress_monitor, "test_attr")

            # Verify
            mock_logger.info.assert_called_with("Deleting attribute %s", "test_attr")
            assert not hasattr(progress_monitor, "test_attr")

    def test_getattr(self, progress_monitor):
        """Test the __getattribute__ method."""
        # Setup - add an attribute to get
        progress_monitor.test_attr = "test_value"

        # Use __getattribute__ through attribute access
        with patch.object(progress_monitor, "_logger") as mock_logger:
            value = progress_monitor.test_attr

            # Verify
            mock_logger.info.assert_called_with("Getting attribute %s", "test_attr")
            assert value == "test_value"

    def test_setattr(self, progress_monitor):
        """Test the __setattr__ method."""
        # Use __setattr__ through attribute assignment
        with patch.object(progress_monitor, "_logger") as mock_logger:
            progress_monitor.test_attr = "new_value"

            # Verify
            mock_logger.info.assert_called_with(
                "Setting attribute %s to %s", "test_attr", "new_value"
            )
            assert progress_monitor.test_attr == "new_value"

    def test_reduce(self, progress_monitor):
        """Test the __reduce__ method."""
        # Setup
        progress_monitor._logger.name = "test_logger"

        # Call __reduce__
        reduce_result = progress_monitor.__reduce__()

        # Verify
        assert reduce_result[0] == progress_monitor.__class__
        assert reduce_result[1] == ("test_logger")

    def test_reduce_ex(self, progress_monitor):
        """Test the __reduce_ex__ method."""
        # Setup
        progress_monitor._logger.name = "test_logger"

        # Call __reduce_ex__
        reduce_result = progress_monitor.__reduce_ex__(
            4
        )  # Protocol version doesn't matter for our test

        # Verify
        assert reduce_result[0] == progress_monitor.__class__
        assert reduce_result[1] == ("test_logger")

    def test_copy(self, progress_monitor):
        """Test the __copy__ method."""
        # Setup
        progress_monitor._logger.name = "test_logger"

        # Call __copy__
        copy_result = progress_monitor.__copy__()

        # Verify
        assert isinstance(copy_result, ProgressMonitor)
        assert copy_result is not progress_monitor

    def test_deepcopy(self, progress_monitor):
        """Test the __deepcopy__ method."""
        # Setup
        progress_monitor._logger.name = "test_logger"

        # Call __deepcopy__
        deepcopy_result = progress_monitor.__deepcopy__({})

        # Verify
        assert isinstance(deepcopy_result, ProgressMonitor)
        assert deepcopy_result is not progress_monitor

    def test_getstate(self, progress_monitor):
        """Test the __getstate__ method."""
        # Call __getstate__
        state = progress_monitor.__getstate__()

        # Verify
        assert "_logger" in state
        assert state["_logger"] == progress_monitor._logger

    def test_setstate(self, progress_monitor):
        """Test the __setstate__ method."""
        # Setup
        mock_logger = Mock(spec=logging.Logger)
        state = {"_logger": mock_logger}

        # Call __setstate__
        progress_monitor.__setstate__(state)

        # Verify
        assert progress_monitor._logger == mock_logger
        mock_logger.info.assert_called_with("ProgressMonitor state set")
