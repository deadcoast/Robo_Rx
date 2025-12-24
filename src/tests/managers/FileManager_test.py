import logging
from unittest.mock import Mock, mock_open, patch

import pytest

from src.managers.FileManager import FileManager


class TestFileManager:
    """Test suite for the FileManager class."""

    @pytest.fixture
    def file_manager(self):
        """Create a FileManager instance for testing."""
        with patch("logging.getLogger") as mock_get_logger:
            # Create a mock logger to avoid actual logging during tests
            mock_logger = Mock(spec=logging.Logger)
            mock_get_logger.return_value = mock_logger

            # Create and return FileManager instance
            manager = FileManager()

            # Set the mocked logger so tests can access it
            manager._logger = mock_logger
            return manager

    def test_init(self, file_manager):
        """Test initialization of FileManager."""
        # Verify logger is initialized
        assert file_manager._logger is not None
        file_manager._logger.info.assert_called_with("FileManager initialized")

    @patch("builtins.open", new_callable=mock_open)
    @patch("json.dump")
    def test_save_data(self, mock_json_dump, mock_file_open, file_manager):
        """Test save_data method."""
        # Setup
        test_data = {"key": "value"}
        test_path = "test_file.json"

        # Call the method
        file_manager.save_data(test_data, test_path)

        # Verify
        file_manager._logger.info.assert_called_with(f"Saving data to {test_path}")
        mock_file_open.assert_called_once_with(test_path, "w")
        mock_json_dump.assert_called_once()
        assert mock_json_dump.call_args[0][0] == test_data

    @patch("builtins.open", new_callable=mock_open, read_data='{"key": "value"}')
    @patch("json.load")
    def test_load_data(self, mock_json_load, mock_file_open, file_manager):
        """Test load_data method."""
        # Setup
        test_path = "test_file.json"
        expected_data = {"key": "value"}
        mock_json_load.return_value = expected_data

        # Call the method
        result = file_manager.load_data(test_path)

        # Verify
        file_manager._logger.info.assert_called_with(f"Loading data from {test_path}")
        mock_file_open.assert_called_once_with(test_path, "r")
        mock_json_load.assert_called_once()
        assert result == expected_data

    @patch("os.remove")
    def test_delete_data(self, mock_remove, file_manager):
        """Test delete_data method."""
        # Setup
        test_path = "test_file.json"

        # Call the method
        file_manager.delete_data(test_path)

        # Verify
        file_manager._logger.info.assert_called_with(f"Deleting data from {test_path}")
        mock_remove.assert_called_once_with(test_path)

    @patch("builtins.open", new_callable=mock_open, read_data='{"key": "old_value"}')
    @patch("json.load")
    @patch("json.dump")
    def test_update_data(
        self, mock_json_dump, mock_json_load, mock_file_open, file_manager
    ):
        """Test update_data method."""
        # Setup
        test_path = "test_file.json"
        test_data = {"key": "new_value"}
        existing_data = {"key": "old_value"}
        mock_json_load.return_value = existing_data

        # Call the method
        file_manager.update_data(test_data, test_path)

        # Verify
        file_manager._logger.info.assert_called_with(f"Updating data in {test_path}")
        mock_file_open.assert_called()
        mock_json_load.assert_called_once()
        mock_json_dump.assert_called_once()
        # Check that the updated data was written
        updated_data = mock_json_dump.call_args[0][0]
        assert updated_data["key"] == "new_value"

    @patch("os.listdir")
    @patch("os.path.isfile")
    @patch("builtins.open", new_callable=mock_open, read_data='{"key": "value"}')
    @patch("json.load")
    def test_list_data(
        self, mock_json_load, mock_file_open, mock_is_file, mock_listdir, file_manager
    ):
        """Test list_data method."""
        # Setup
        test_path = "test_directory"
        file_list = ["file1.json", "file2.json", "file3.txt"]
        mock_listdir.return_value = file_list
        mock_is_file.return_value = True
        mock_json_load.return_value = {"key": "value"}

        # Call the method
        result = file_manager.list_data(test_path)

        # Verify
        file_manager._logger.info.assert_called_with(f"Listing data in {test_path}")
        mock_listdir.assert_called_once_with(test_path)
        # Should be called only for json files
        assert mock_file_open.call_count == 2  # Only the .json files
        assert len(result) == 2  # Only the .json files

    @patch("os.walk")
    @patch(
        "builtins.open",
        new_callable=mock_open,
        read_data='{"key": "searchterm", "other": "value"}',
    )
    @patch("json.load")
    def test_search_data(self, mock_json_load, mock_file_open, mock_walk, file_manager):
        """Test search_data method."""
        # Setup
        test_query = "searchterm"
        mock_walk.return_value = [
            ("root", ["dir1"], ["file1.json", "file2.json"]),
            ("root/dir1", [], ["file3.json"]),
        ]
        # Set up mock_json_load to return different values for different files
        mock_json_load.side_effect = [
            {"key": "searchterm", "other": "value"},  # file1.json - should match
            {"key": "nomatch", "other": "value"},  # file2.json - should not match
            {"other": "searchterm"},  # file3.json - should match
        ]

        # Call the method
        result = file_manager.search_data(test_query)

        # Verify
        file_manager._logger.info.assert_called_with(
            f"Searching for data: {test_query}"
        )
        mock_walk.assert_called_once()
        assert mock_file_open.call_count == 3  # All three json files
        assert len(result) == 2  # Only files with searchterm should be included

    @patch("builtins.open", new_callable=mock_open, read_data='{"key": "value"}')
    @patch("json.load")
    def test_get_data(self, mock_json_load, mock_file_open, file_manager):
        """Test get_data method."""
        # Setup
        test_path = "test_file.json"
        expected_data = {"key": "value"}
        mock_json_load.return_value = expected_data

        # Call the method
        result = file_manager.get_data(test_path)

        # Verify
        file_manager._logger.info.assert_called_with(f"Getting data from {test_path}")
        mock_file_open.assert_called_once_with(test_path, "r")
        mock_json_load.assert_called_once()
        assert result == expected_data

    @patch("builtins.open", new_callable=mock_open)
    @patch("json.dump")
    def test_set_data(self, mock_json_dump, mock_file_open, file_manager):
        """Test set_data method."""
        # Setup
        test_data = {"key": "value"}
        test_path = "test_file.json"

        # Call the method
        file_manager.set_data(test_data, test_path)

        # Verify
        file_manager._logger.info.assert_called_with(f"Setting data in {test_path}")
        mock_file_open.assert_called_once_with(test_path, "w")
        mock_json_dump.assert_called_once()
        assert mock_json_dump.call_args[0][0] == test_data

    @patch("os.path.exists")
    def test_exists_data_true(self, mock_exists, file_manager):
        """Test exists_data method when file exists."""
        # Setup
        test_path = "test_file.json"
        mock_exists.return_value = True

        # Call the method
        result = file_manager.exists_data(test_path)

        # Verify
        file_manager._logger.info.assert_called_with(
            f"Checking if data exists at {test_path}"
        )
        mock_exists.assert_called_once_with(test_path)
        assert result is True

    @patch("os.path.exists")
    def test_exists_data_false(self, mock_exists, file_manager):
        """Test exists_data method when file does not exist."""
        # Setup
        test_path = "test_file.json"
        mock_exists.return_value = False

        # Call the method
        result = file_manager.exists_data(test_path)

        # Verify
        file_manager._logger.info.assert_called_with(
            f"Checking if data exists at {test_path}"
        )
        mock_exists.assert_called_once_with(test_path)
        assert result is False

    @patch("os.path.getsize")
    def test_get_data_size(self, mock_getsize, file_manager):
        """Test get_data_size method."""
        # Setup
        test_path = "test_file.json"
        expected_size = 1024  # 1KB
        mock_getsize.return_value = expected_size

        # Call the method
        result = file_manager.get_data_size(test_path)

        # Verify
        file_manager._logger.info.assert_called_with(
            f"Getting data size for {test_path}"
        )
        mock_getsize.assert_called_once_with(test_path)
        assert result == expected_size

    @patch("os.path.getmtime")
    def test_get_data_modified_time(self, mock_getmtime, file_manager):
        """Test get_data_modified_time method."""
        # Setup
        test_path = "test_file.json"
        expected_time = 1609459200.0  # 2021-01-01 00:00:00
        mock_getmtime.return_value = expected_time

        # Call the method
        result = file_manager.get_data_modified_time(test_path)

        # Verify
        file_manager._logger.info.assert_called_with(
            f"Getting data modified time for {test_path}"
        )
        mock_getmtime.assert_called_once_with(test_path)
        assert result == expected_time

    @patch("os.path.getctime")
    def test_get_data_created_time(self, mock_getctime, file_manager):
        """Test get_data_created_time method."""
        # Setup
        test_path = "test_file.json"
        expected_time = 1609459200.0  # 2021-01-01 00:00:00
        mock_getctime.return_value = expected_time

        # Call the method
        result = file_manager.get_data_created_time(test_path)

        # Verify
        file_manager._logger.info.assert_called_with(
            f"Getting data created time for {test_path}"
        )
        mock_getctime.assert_called_once_with(test_path)
        assert result == expected_time

    @patch("os.path.basename")
    def test_get_data_name(self, mock_basename, file_manager):
        """Test get_data_name method."""
        # Setup
        test_path = "/path/to/test_file.json"
        expected_name = "test_file.json"
        mock_basename.return_value = expected_name

        # Call the method
        result = file_manager.get_data_name(test_path)

        # Verify
        file_manager._logger.info.assert_called_with(
            f"Getting data name for {test_path}"
        )
        mock_basename.assert_called_once_with(test_path)
        assert result == expected_name

    @patch("os.path.dirname")
    def test_get_data_directory(self, mock_dirname, file_manager):
        """Test get_data_directory method."""
        # Setup
        test_path = "/path/to/test_file.json"
        expected_dir = "/path/to"
        mock_dirname.return_value = expected_dir

        # Call the method
        result = file_manager.get_data_directory(test_path)

        # Verify
        file_manager._logger.info.assert_called_with(
            f"Getting data directory for {test_path}"
        )
        mock_dirname.assert_called_once_with(test_path)
        assert result == expected_dir
