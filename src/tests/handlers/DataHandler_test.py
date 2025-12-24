import logging
from unittest.mock import Mock, patch

import pytest

from src.handlers.DataHandler import DataHandler
from src.managers.FileManager import FileManager


@pytest.fixture
def mock_file_manager():
    """Create a mock FileManager for testing."""
    mock = Mock(spec=FileManager)
    mock.save_data = Mock()
    mock.load_data = Mock(return_value={"key": "value"})
    mock.delete_data = Mock()
    mock.update_data = Mock()
    return mock


@pytest.fixture
def data_handler(mock_file_manager):
    """Create a DataHandler with a mocked FileManager."""
    with patch("src.handlers.DataHandler.FileManager", return_value=mock_file_manager):
        with patch("src.handlers.DataHandler.logging"):
            handler = DataHandler()
            # Directly set the mocked file manager
            handler._file_manager = mock_file_manager
            return handler


@pytest.fixture
def sample_data():
    """Sample data for testing."""
    return {
        "id": "test-123",
        "name": "Test Data",
        "values": [1, 2, 3],
        "metadata": {"created": "2025-03-31", "version": 1.0},
    }


class TestDataHandler:
    """Test suite for the DataHandler class."""

    def test_init(self):
        """Test initialization of DataHandler."""
        with patch("src.handlers.DataHandler.FileManager") as mock_file_manager:
            with patch("src.handlers.DataHandler.logging") as mock_logging:
                # Setup logger mock
                mock_logger = Mock(spec=logging.Logger)
                mock_logging.getLogger.return_value = mock_logger

                # Create handler
                handler = DataHandler()

                # Verify initializations
                assert handler._data == {}
                assert handler._path == ""
                assert mock_file_manager.called

                # Verify logger setup
                mock_logging.getLogger.assert_called_once_with(
                    "src.handlers.DataHandler"
                )
                mock_logger.setLevel.assert_called_once_with(logging.INFO)
                mock_logger.addHandler.assert_called_once()
                mock_logger.info.assert_called_once_with("DataHandler initialized")

    @pytest.mark.asyncio
    async def test_process_data(self, data_handler, sample_data):
        """Test process_data method."""
        result = await data_handler.process_data(sample_data)

        # Verify data is stored and returned
        assert data_handler._data == sample_data
        assert result == sample_data

        # Verify logging
        data_handler._logger.info.assert_called_with("Data processed")

    @pytest.mark.asyncio
    async def test_save_data(self, data_handler, sample_data):
        """Test save_data method."""
        test_path = "/path/to/data.json"

        await data_handler.save_data(sample_data, test_path)

        # Verify data and path are stored
        assert data_handler._data == sample_data
        assert data_handler._path == test_path

        # Verify file manager is called correctly
        data_handler._file_manager.save_data.assert_called_once_with(
            sample_data, test_path
        )

        # Verify logging
        data_handler._logger.info.assert_called_with(f"Data saved to {test_path}")

    @pytest.mark.asyncio
    async def test_load_data(self, data_handler):
        """Test load_data method."""
        test_path = "/path/to/data.json"
        expected_data = {"key": "value"}
        data_handler._file_manager.load_data.return_value = expected_data

        result = await data_handler.load_data(test_path)

        # Verify data and path are stored
        assert data_handler._data == expected_data
        assert data_handler._path == test_path
        assert result == expected_data

        # Verify file manager is called correctly
        data_handler._file_manager.load_data.assert_called_once_with(test_path)

        # Verify logging
        data_handler._logger.info.assert_called_with(f"Data loaded from {test_path}")

    @pytest.mark.asyncio
    async def test_delete_data(self, data_handler):
        """Test delete_data method."""
        test_path = "/path/to/data.json"

        # Set some initial data
        data_handler._data = {"existing": "data"}

        await data_handler.delete_data(test_path)

        # Verify data is cleared
        assert not data_handler._data

        # Verify file manager is called correctly
        data_handler._file_manager.delete_data.assert_called_once_with(test_path)

        # Verify logging
        data_handler._logger.info.assert_called_with(f"Data deleted from {test_path}")

    @pytest.mark.asyncio
    async def test_update_data(self, data_handler):
        """Test update_data method."""
        test_path = "/path/to/data.json"
        initial_data = {"id": "test-123", "name": "Test Data"}
        update_data = {"name": "Updated Test", "version": 2.0}

        # Set initial data
        data_handler._data = initial_data.copy()

        await data_handler.update_data(update_data, test_path)

        # Verify data is updated
        expected_updated_data = {
            "id": "test-123",
            "name": "Updated Test",
            "version": 2.0,
        }
        assert data_handler._data == expected_updated_data

        # Verify file manager is called correctly
        data_handler._file_manager.update_data.assert_called_once_with(
            update_data, test_path
        )

        # Verify logging
        data_handler._logger.info.assert_called_with(f"Data updated in {test_path}")

    @pytest.mark.asyncio
    async def test_list_data(self, data_handler):
        """Test list_data method."""
        test_path = "/path/to/directory"
        test_data = {
            "item1": {"id": "1", "name": "First"},
            "item2": {"id": "2", "name": "Second"},
            "item3": {"id": "3", "name": "Third"},
        }

        # Set test data
        data_handler._data = test_data

        result = await data_handler.list_data(test_path)

        # Verify path is stored
        assert data_handler._path == test_path

        # Verify result is a list of the data values
        assert isinstance(result, list)
        assert len(result) == 3
        assert {"id": "1", "name": "First"} in result
        assert {"id": "2", "name": "Second"} in result
        assert {"id": "3", "name": "Third"} in result

    @pytest.mark.asyncio
    async def test_search_data(self, data_handler):
        """Test search_data method."""
        test_data = {
            "item1": {"id": "1", "name": "Apple"},
            "item2": {"id": "2", "name": "Banana"},
            "item3": {"id": "3", "name": "Cherry"},
        }

        # Set test data
        data_handler._data = test_data

        # Search for items containing "an"
        result = await data_handler.search_data("an")

        # Verify only items with "an" in values are returned
        assert len(result) == 1
        assert {"id": "2", "name": "Banana"} in result

        # Empty search should find nothing
        empty_result = await data_handler.search_data("xyz")
        assert empty_result == []

    @pytest.mark.asyncio
    async def test_get_data(self, data_handler):
        """Test get_data method."""
        test_path = "/path/to/data.json"
        test_data = {"key": "value"}

        # Set test data
        data_handler._data = test_data

        result = await data_handler.get_data(test_path)

        # Verify path is stored
        assert data_handler._path == test_path

        # Verify data is returned correctly
        assert result == test_data
        assert result is data_handler._data  # Should be the same object
