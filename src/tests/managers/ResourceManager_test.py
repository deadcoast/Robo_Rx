from pathlib import Path
from unittest.mock import patch

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
import torch
import torch.nn as nn

from src.managers.ResourceManager import (
    ResourceManager,
    ResourceNotFoundError,
    ResourceType,
)


class TestResourceManager:
    """Test suite for the ResourceManager class."""

    @pytest.fixture
    def resource_manager(self):
        """Create a ResourceManager instance for testing."""
        return ResourceManager()

    def test_init(self, resource_manager):
        """Test initialization of ResourceManager."""
        assert resource_manager.resources == {}
        assert resource_manager.resource_types == {}
        assert resource_manager.logger is not None

    def test_add_get_general_resource(self, resource_manager):
        """Test adding and retrieving a general resource."""
        # Setup
        resource_name = "test_resource"
        resource = {"key": "value"}

        # Add resource
        resource_manager.add_resource(resource_name, resource)

        # Verify
        assert resource_name in resource_manager.resources
        assert resource_manager.resources[resource_name] == resource
        assert resource_manager.resource_types[resource_name] == ResourceType.GENERAL

        # Test retrieval
        retrieved = resource_manager.get_resource(resource_name)
        assert retrieved == resource

    def test_get_nonexistent_resource(self, resource_manager):
        """Test retrieving a nonexistent resource."""
        with pytest.raises(ResourceNotFoundError):
            resource_manager.get_resource("nonexistent")

    def test_remove_resource(self, resource_manager):
        """Test removing a resource."""
        # Setup
        resource_manager.add_resource("test", "value")
        assert "test" in resource_manager.resources

        # Remove
        resource_manager.remove_resource("test")

        # Verify
        assert "test" not in resource_manager.resources
        assert "test" not in resource_manager.resource_types

    def test_clear_resources(self, resource_manager):
        """Test clearing all resources."""
        # Setup
        resource_manager.add_resource("test1", "value1")
        resource_manager.add_resource("test2", "value2")
        assert len(resource_manager.resources) == 2

        # Clear
        resource_manager.clear_resources()

        # Verify
        assert len(resource_manager.resources) == 0
        assert len(resource_manager.resource_types) == 0

    def test_add_get_numpy_array(self, resource_manager):
        """Test adding and retrieving a NumPy array."""
        # Setup
        resource_name = "test_array"
        array = np.array([1, 2, 3])

        # Add
        resource_manager.add_numpy_array(resource_name, array)

        # Verify type
        assert (
            resource_manager.resource_types[resource_name] == ResourceType.NUMPY_ARRAY
        )

        # Test retrieval
        retrieved = resource_manager.get_numpy_array(resource_name)
        assert np.array_equal(retrieved, array)

    def test_get_wrong_type(self, resource_manager):
        """Test retrieving a resource with wrong type expectation."""
        # Setup
        resource_manager.add_resource("test", "string_value", ResourceType.GENERAL)

        # Try to get as wrong type
        with pytest.raises(TypeError):
            resource_manager.get_numpy_array("test")

    def test_add_get_torch_tensor(self, resource_manager):
        """Test adding and retrieving a PyTorch tensor."""
        # Setup
        resource_name = "test_tensor"
        tensor = torch.tensor([1.0, 2.0, 3.0])

        # Add
        resource_manager.add_torch_tensor(resource_name, tensor)

        # Verify type
        assert (
            resource_manager.resource_types[resource_name] == ResourceType.TORCH_TENSOR
        )

        # Test retrieval
        retrieved = resource_manager.get_torch_tensor(resource_name)
        assert torch.equal(retrieved, tensor)

    def test_add_get_pandas_dataframe(self, resource_manager):
        """Test adding and retrieving a pandas DataFrame."""
        # Setup
        resource_name = "test_df"
        df = pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]})

        # Add
        resource_manager.add_pandas_dataframe(resource_name, df)

        # Verify type
        assert (
            resource_manager.resource_types[resource_name]
            == ResourceType.PANDAS_DATAFRAME
        )

        # Test retrieval
        retrieved = resource_manager.get_pandas_dataframe(resource_name)
        assert retrieved.equals(df)

    def test_add_get_matplotlib_figure(self, resource_manager):
        """Test adding and retrieving a matplotlib figure."""
        # Setup
        resource_name = "test_figure"
        fig = plt.figure()

        # Add
        resource_manager.add_matplotlib_figure(resource_name, fig)

        # Verify type
        assert (
            resource_manager.resource_types[resource_name]
            == ResourceType.MATPLOTLIB_FIGURE
        )

        # Test retrieval
        retrieved = resource_manager.get_matplotlib_figure(resource_name)
        assert retrieved is fig

    def test_add_get_torch_model(self, resource_manager):
        """Test adding and retrieving a PyTorch model."""
        # Setup
        resource_name = "test_model"
        model = nn.Linear(10, 5)

        # Add
        resource_manager.add_torch_model(resource_name, model)

        # Verify type
        assert (
            resource_manager.resource_types[resource_name] == ResourceType.TORCH_MODEL
        )

        # Test retrieval
        retrieved = resource_manager.get_torch_model(resource_name)
        assert retrieved is model

    def test_contains(self, resource_manager):
        """Test the __contains__ method."""
        # Setup
        resource_manager.add_resource("test", "value")

        # Test
        assert "test" in resource_manager
        assert "nonexistent" not in resource_manager

    def test_len(self, resource_manager):
        """Test the __len__ method."""
        # Setup
        assert len(resource_manager) == 0

        resource_manager.add_resource("test1", "value1")
        assert len(resource_manager) == 1

        resource_manager.add_resource("test2", "value2")
        assert len(resource_manager) == 2

    def test_iter(self, resource_manager):
        """Test the __iter__ method."""
        # Setup
        resource_manager.add_resource("test1", "value1")
        resource_manager.add_resource("test2", "value2")

        # Convert iterator to list for comparison
        resources = list(resource_manager)

        # Verify
        assert len(resources) == 2
        assert "test1" in resources
        assert "test2" in resources

    def test_str_repr(self, resource_manager):
        """Test __str__ and __repr__ methods."""
        # Setup
        resource_manager.add_resource("test", "value")

        # Test
        str_result = str(resource_manager)
        repr_result = repr(resource_manager)

        assert "ResourceManager" in str_result
        assert "test" in str_result
        assert "ResourceManager" in repr_result

    @patch("torch.save")
    def test_save_torch_tensor(self, mock_save, resource_manager):
        """Test saving a PyTorch tensor."""
        # Setup
        resource_name = "test_tensor"
        tensor = torch.tensor([1.0, 2.0, 3.0])
        resource_manager.add_torch_tensor(resource_name, tensor)

        # Save
        resource_manager.save_resource(resource_name, "test.pt")

        # Verify
        mock_save.assert_called_once()
        args, _ = mock_save.call_args
        assert torch.equal(args[0], tensor)
        assert args[1] == "test.pt"

    @patch("numpy.save")
    def test_save_numpy_array(self, mock_save, resource_manager):
        """Test saving a NumPy array."""
        # Setup
        resource_name = "test_array"
        array = np.array([1, 2, 3])
        resource_manager.add_numpy_array(resource_name, array)

        # Save
        resource_manager.save_resource(resource_name, "test.npy")

        # Verify
        mock_save.assert_called_once()
        args, _ = mock_save.call_args
        assert args[1] == "test.npy"
        assert np.array_equal(args[0], array)

    @patch("os.path.exists")
    @patch("torch.load")
    def test_load_torch_tensor(self, mock_load, mock_exists, resource_manager):
        """Test loading a PyTorch tensor."""
        # Setup
        mock_exists.return_value = True
        tensor = torch.tensor([1.0, 2.0, 3.0])
        mock_load.return_value = tensor

        # Load
        resource_manager.load_resource(
            "test_tensor", "test.pt", ResourceType.TORCH_TENSOR
        )

        # Verify
        mock_load.assert_called_once()
        assert "test_tensor" in resource_manager.resources
        assert (
            resource_manager.resource_types["test_tensor"] == ResourceType.TORCH_TENSOR
        )

    def test_is_safe_file_source(self, resource_manager):
        """Test the _is_safe_file_source method."""
        # Mock Path.absolute and Path.home
        with patch("pathlib.Path.absolute") as mock_absolute, patch(
            "pathlib.Path.home"
        ) as mock_home, patch("pathlib.Path.cwd") as mock_cwd:

            # Setup
            mock_home.return_value = Path("/home/user")
            mock_cwd.return_value = Path("/home/user/project")

            # Test trusted path
            mock_absolute.return_value = Path("/home/user/trusted_data/file.pkl")
            assert resource_manager._is_safe_file_source("file.pkl")

            # Test untrusted path
            mock_absolute.return_value = Path("/tmp/malicious.pkl")
            assert not resource_manager._is_safe_file_source("malicious.pkl")

    def test_context_manager(self, resource_manager):
        """Test the context manager interface."""
        # Use as context manager
        with resource_manager as rm:
            rm.add_resource("test", "value")
            assert "test" in rm

        # Verify resource is still there after context exit
        assert "test" in resource_manager

    def test_copy(self, resource_manager):
        """Test the copy method."""
        # Setup
        resource_manager.add_resource("test", "value")

        # Copy
        copied = resource_manager.__copy__()

        # Verify
        assert copied is not resource_manager
        assert "test" in copied
        assert copied.resources["test"] == "value"

        # Modify copy and verify original is unchanged
        copied.add_resource("new", "new_value")
        assert "new" in copied
        assert "new" not in resource_manager

    def test_get_resource_info(self, resource_manager):
        """Test the get_resource_info method."""
        # Setup
        resource_manager.add_resource("test1", "string", ResourceType.GENERAL)
        resource_manager.add_numpy_array("test2", np.array([1, 2, 3]))

        # Get info
        info_df = resource_manager.get_resource_info()

        # Verify
        assert isinstance(info_df, pd.DataFrame)
        assert len(info_df) == 2
        assert "test1" in info_df["Name"].values
        assert "test2" in info_df["Name"].values
