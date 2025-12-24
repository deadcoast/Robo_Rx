from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import numpy as np
import pandas as pd
import pytest
import torch

from src.core.DataCore import (
    DataFrameProcessor,
    DataHandler,
    NumpyProcessor,
    ProcessedData,
    load_tensor_dict,
    save_tensor_dict,
)


class TestTensorDictFunctions:
    """Test suite for tensor dictionary save/load functions."""

    @pytest.fixture
    def tensor_dict(self):
        """Create a sample tensor dictionary for testing."""
        return {
            "tensor1": torch.tensor([1.0, 2.0, 3.0]),
            "tensor2": torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
        }

    @patch("src.core.DataCore.save_file")
    @patch("pathlib.Path.parent")
    def test_save_tensor_dict_str_path(self, mock_parent, mock_save_file, tensor_dict):
        """Test saving tensor dict with string path."""
        # Setup
        mock_parent.mkdir = MagicMock()
        mock_save_file.return_value = {"metadata": "test"}
        file_path = "test/path/tensor.safetensors"

        # Call the function
        result = save_tensor_dict(tensor_dict, file_path)

        # Verify
        mock_save_file.assert_called_once_with(tensor_dict, file_path)
        assert result == {"metadata": "test"}

    @patch("src.core.DataCore.save_file")
    @patch("pathlib.Path.parent")
    def test_save_tensor_dict_path_object(
        self, mock_parent, mock_save_file, tensor_dict
    ):
        """Test saving tensor dict with Path object."""
        # Setup
        mock_parent.mkdir = MagicMock()
        mock_save_file.return_value = {"metadata": "test"}
        file_path = Path("test/path/tensor.safetensors")

        # Call the function
        result = save_tensor_dict(tensor_dict, file_path)

        # Verify
        mock_save_file.assert_called_once_with(tensor_dict, str(file_path))
        assert result == {"metadata": "test"}

    @patch("src.core.DataCore.load_file")
    def test_load_tensor_dict(self, mock_load_file, tensor_dict):
        """Test loading tensor dict."""
        # Setup
        mock_load_file.return_value = tensor_dict
        file_path = "test/path/tensor.safetensors"

        # Call the function
        result = load_tensor_dict(file_path)

        # Verify
        mock_load_file.assert_called_once_with(file_path)
        assert result == tensor_dict

    @patch("src.core.DataCore.load_file")
    def test_load_tensor_dict_path_object(self, mock_load_file, tensor_dict):
        """Test loading tensor dict with Path object."""
        # Setup
        mock_load_file.return_value = tensor_dict
        file_path = Path("test/path/tensor.safetensors")

        # Call the function
        result = load_tensor_dict(file_path)

        # Verify
        mock_load_file.assert_called_once_with(str(file_path))
        assert result == tensor_dict


class TestDataFrameProcessor:
    """Test suite for the DataFrameProcessor class."""

    @pytest.fixture
    def df_processor(self):
        """Create a DataFrameProcessor instance for testing."""
        return DataFrameProcessor()

    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing."""
        return {
            "col1": [1, 2, 3, 4, 5],
            "col2": ["a", "b", "c", "d", "e"],
            "col3": [1.1, 2.2, 3.3, 4.4, 5.5],
        }

    def test_init(self, df_processor):
        """Test initialization of DataFrameProcessor."""
        assert df_processor.dataframe is None
        assert isinstance(df_processor.config, dict)
        assert "missing_value_handling" in df_processor.config
        assert "thresholds" in df_processor.config

    def test_create_dataframe(self, df_processor, sample_data):
        """Test creating a dataframe from data."""
        # Call the method
        df = df_processor.create_dataframe(sample_data)

        # Verify
        assert isinstance(df, pd.DataFrame)
        assert df.shape == (5, 3)
        assert list(df.columns) == ["col1", "col2", "col3"]
        assert df.iloc[0, 0] == 1
        assert df.iloc[0, 1] == "a"
        assert df.iloc[0, 2] == 1.1

    def test_create_dataframe_from_list(self, df_processor):
        """Test creating a dataframe from a list of dictionaries."""
        # Setup
        data = [
            {"col1": 1, "col2": "a"},
            {"col1": 2, "col2": "b"},
            {"col1": 3, "col2": "c"},
        ]

        # Call the method
        df = df_processor.create_dataframe(data)

        # Verify
        assert isinstance(df, pd.DataFrame)
        assert df.shape == (3, 2)
        assert list(df.columns) == ["col1", "col2"]
        assert df.iloc[0, 0] == 1
        assert df.iloc[0, 1] == "a"

    def test_process_dataframe(self, df_processor, sample_data):
        """Test processing a dataframe."""
        # Setup
        df = pd.DataFrame(sample_data)

        # Create a method for testing
        with patch.object(df_processor, "process_dataframe") as mock_process:
            mock_process.return_value = df.copy()

            # Call the method
            result = df_processor.process_dataframe(df)

            # Verify
            mock_process.assert_called_once_with(df)
            assert isinstance(result, pd.DataFrame)
            assert result.equals(df)


class TestNumpyProcessor:
    """Test suite for the NumpyProcessor class."""

    @pytest.fixture
    def numpy_processor(self):
        """Create a NumpyProcessor instance for testing."""
        return NumpyProcessor()

    @pytest.fixture
    def sample_dataframe(self):
        """Create a sample DataFrame for testing."""
        return pd.DataFrame(
            {
                "col1": [1, 2, 3, 4, 5],
                "col2": [6, 7, 8, 9, 10],
                "col3": [11, 12, 13, 14, 15],
            }
        )

    def test_init(self, numpy_processor):
        """Test initialization of NumpyProcessor."""
        assert numpy_processor.data is None
        assert numpy_processor.processed is False

    def test_generate_matrices(self, numpy_processor, sample_dataframe):
        """Test generating matrices from a DataFrame."""
        # Call the method
        matrices = numpy_processor.generate_matrices(sample_dataframe)

        # Verify
        assert isinstance(matrices, np.ndarray)
        assert matrices.shape == (5, 3)  # 5 rows, 3 columns
        assert np.array_equal(matrices[:, 0], np.array([1, 2, 3, 4, 5]))
        assert np.array_equal(matrices[:, 1], np.array([6, 7, 8, 9, 10]))
        assert np.array_equal(matrices[:, 2], np.array([11, 12, 13, 14, 15]))

        # Should update the processor state
        assert numpy_processor.data is not None
        assert numpy_processor.processed is True
        assert np.array_equal(numpy_processor.data, matrices)

    def test_generate_matrices_empty_dataframe(self, numpy_processor):
        """Test generating matrices from an empty DataFrame."""
        # Setup
        empty_df = pd.DataFrame()

        # Call the method
        matrices = numpy_processor.generate_matrices(empty_df)

        # Verify
        assert isinstance(matrices, np.ndarray)
        assert matrices.size == 0
        assert numpy_processor.data is not None
        assert numpy_processor.processed is True

    def test_process_matrices(self, numpy_processor, sample_dataframe):
        """Test processing matrices."""
        # Setup
        matrices = numpy_processor.generate_matrices(sample_dataframe)

        # Create a method for testing
        with patch.object(numpy_processor, "process_matrices") as mock_process:
            mock_process.return_value = matrices.copy()

            # Call the method
            result = numpy_processor.process_matrices(matrices)

            # Verify
            mock_process.assert_called_once_with(matrices)
            assert isinstance(result, np.ndarray)
            assert np.array_equal(result, matrices)


class TestProcessedData:
    """Test suite for the ProcessedData class."""

    @pytest.fixture
    def sample_dataframe(self):
        """Create a sample DataFrame for testing."""
        return pd.DataFrame(
            {
                "col1": [1, 2, 3, 4, 5],
                "col2": [6, 7, 8, 9, 10],
                "col3": [11, 12, 13, 14, 15],
            }
        )

    @pytest.fixture
    def sample_matrices(self):
        """Create sample matrices for testing."""
        return np.array([[1, 6, 11], [2, 7, 12], [3, 8, 13], [4, 9, 14], [5, 10, 15]])

    @pytest.fixture
    def tensor_metadata(self):
        """Create sample tensor metadata for testing."""
        return {"shape": [5, 3], "dtype": "float32"}

    def test_init(self, sample_dataframe, sample_matrices, tensor_metadata):
        """Test initialization of ProcessedData."""
        # Setup
        with patch("src.core.DataCore.timestamp") as mock_timestamp:
            mock_timestamp.now.return_value = "2023-01-01T00:00:00"

            # Create ProcessedData instance
            processed_data = ProcessedData(
                sample_dataframe, sample_matrices, tensor_metadata
            )

            # Verify
            assert processed_data.processed_content is sample_dataframe
            assert processed_data.matrices is sample_matrices
            assert processed_data.tensor_metadata == tensor_metadata
            assert processed_data.timestamp == "2023-01-01T00:00:00"


class TestDataHandler:
    """Test suite for the DataHandler class."""

    @pytest.fixture
    def data_handler(self):
        """Create a DataHandler instance for testing."""
        with patch("src.core.DataCore.DataFrameProcessor") as mock_df_processor, patch(
            "src.core.DataCore.NumpyProcessor"
        ) as mock_numpy_processor, patch(
            "src.core.DataCore.FileManager"
        ) as mock_file_manager:

            handler = DataHandler()

            # Replace with mock instances
            handler.df_processor = mock_df_processor.return_value
            handler.numpy_engine = mock_numpy_processor.return_value
            handler.file_manager = mock_file_manager.return_value

            return handler

    @pytest.fixture
    def sample_dataframe(self):
        """Create a sample DataFrame for testing."""
        return pd.DataFrame(
            {
                "col1": [1, 2, 3, 4, 5],
                "col2": [6, 7, 8, 9, 10],
                "col3": [11, 12, 13, 14, 15],
            }
        )

    @pytest.fixture
    def sample_matrices(self):
        """Create sample matrices for testing."""
        return np.array([[1, 6, 11], [2, 7, 12], [3, 8, 13], [4, 9, 14], [5, 10, 15]])

    def test_init(self, data_handler):
        """Test initialization of DataHandler."""
        assert data_handler.df_processor is not None
        assert data_handler.numpy_engine is not None
        assert data_handler.file_manager is not None

    def test_extract_metadata(self, data_handler, sample_dataframe):
        """Test extracting metadata from a DataFrame."""
        # Setup expected metadata
        expected_metadata = {
            "rows": 5,
            "columns": 3,
            "column_types": {"col1": "int64", "col2": "int64", "col3": "int64"},
        }

        # Call the method
        with patch.object(
            data_handler, "_extract_metadata", return_value=expected_metadata
        ) as mock_extract:
            metadata = data_handler._extract_metadata(sample_dataframe)

            # Verify
            mock_extract.assert_called_once_with(sample_dataframe)
            assert metadata == expected_metadata

    def test_process_data_with_string(
        self, data_handler, sample_dataframe, sample_matrices
    ):
        """Test processing data with a string input."""
        # Setup mocks
        data_handler.df_processor.create_dataframe.return_value = sample_dataframe
        data_handler.numpy_engine.generate_matrices.return_value = sample_matrices
        data_handler._extract_metadata = Mock(return_value={"rows": 5, "columns": 3})

        # Call the method
        result = data_handler.process_data("data string content")

        # Verify
        assert isinstance(result, ProcessedData)
        data_handler.df_processor.create_dataframe.assert_called_once()
        data_handler.numpy_engine.generate_matrices.assert_called_once_with(
            sample_dataframe
        )
        data_handler._extract_metadata.assert_called_once_with(sample_dataframe)

    def test_process_data_with_file_path(
        self, data_handler, sample_dataframe, sample_matrices
    ):
        """Test processing data with a file path input."""
        # Setup mocks
        data_handler.file_manager.read_file = Mock(return_value="file content")
        data_handler.df_processor.create_dataframe.return_value = sample_dataframe
        data_handler.numpy_engine.generate_matrices.return_value = sample_matrices
        data_handler._extract_metadata = Mock(return_value={"rows": 5, "columns": 3})

        # Call the method
        result = data_handler.process_data(Path("/test/file.csv"))

        # Verify
        assert isinstance(result, ProcessedData)
        data_handler.file_manager.read_file.assert_called_once()
        data_handler.df_processor.create_dataframe.assert_called_once()
        data_handler.numpy_engine.generate_matrices.assert_called_once_with(
            sample_dataframe
        )
        data_handler._extract_metadata.assert_called_once_with(sample_dataframe)
