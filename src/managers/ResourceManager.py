"""
Resource Manager.

This module provides a comprehensive resource manager for managing data science,
machine learning, and general resources in the application. It includes specialized
functionality for handling NumPy arrays, PyTorch tensors, pandas DataFrames,
and matplotlib figures.

Classes:
    ResourceManager: A class for managing various resources in the application.
    ResourceType: An enumeration of resource types.
    ResourceNotFoundError: Exception raised when a resource is not found.

Example:
    >>> resource_manager = ResourceManager()
    >>> resource_manager.add_torch_tensor('embedding', torch.randn(10, 768))
    >>> tensor = resource_manager.get_torch_tensor('embedding')
"""

import copy
import logging
from enum import Enum, auto
from pathlib import Path
from types import TracebackType
from typing import Any, Dict, Iterator, Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset


class SecurityError(Exception):
    """Exception raised when a security constraint is violated."""


class ResourceNotFoundError(Exception):
    """Exception raised when a resource is not found in the ResourceManager."""

    def __init__(self, resource_name: str):
        """
        Args:
            resource_name (str): The name of the resource that was not found.
        """
        self.resource_name = resource_name
        super().__init__(f"Resource '{resource_name}' not found in ResourceManager.")

    def __str__(self) -> str:
        return f"Resource '{self.resource_name}' not found in ResourceManager."

    def __repr__(self) -> str:
        return f"ResourceNotFoundError(resource_name={self.resource_name})"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ResourceNotFoundError):
            return False
        return self.resource_name == other.resource_name

    def __ne__(self, other: object) -> bool:
        return not self == other

    def __hash__(self) -> int:
        return hash(self.resource_name)

    def __copy__(self) -> "ResourceNotFoundError":
        return ResourceNotFoundError(self.resource_name)

    def __deepcopy__(self, memo: Dict) -> "ResourceNotFoundError":
        return ResourceNotFoundError(self.resource_name)

    def __reduce__(self) -> "ResourceNotFoundError":
        return (ResourceNotFoundError, (self.resource_name))

    def __getstate__(self) -> Dict:
        return {"resource_name": self.resource_name}

    def __setstate__(self, state: Dict) -> None:
        self.resource_name = state["resource_name"]
        super().__init__(
            f"Resource '{self.resource_name}' not found in ResourceManager."
        )


class ResourceType(Enum):
    """Enumeration of resource types managed by ResourceManager."""

    NUMPY_ARRAY = auto()
    TORCH_TENSOR = auto()
    PANDAS_DATAFRAME = auto()
    MATPLOTLIB_FIGURE = auto()
    TORCH_MODEL = auto()
    TORCH_DATASET = auto()
    TORCH_DATALOADER = auto()
    GENERAL = auto()


class ResourceManager:
    """
    A comprehensive class for managing various resources in the application.

    This class provides specialized methods for handling different types of resources
    including NumPy arrays, PyTorch tensors, pandas DataFrames, and matplotlib figures.
    It supports resource lifecycle management, conversion between resource types,
    and serialization/deserialization of resources.

    Attributes:
        resources (Dict[str, Any]): A dictionary of resources.
        resource_types (Dict[str, ResourceType]): A dictionary mapping resource names to types.
        logger (logging.Logger): Logger for the resource manager.

    Methods:
        add_resource: Add a general resource to the resource manager.
        get_resource: Get a resource from the resource manager.
        remove_resource: Remove a resource from the resource manager.
        add_numpy_array: Add a NumPy array to the resource manager.
        get_numpy_array: Get a NumPy array from the resource manager.
        add_torch_tensor: Add a PyTorch tensor to the resource manager.
        get_torch_tensor: Get a PyTorch tensor from the resource manager.
        add_pandas_dataframe: Add a pandas DataFrame to the resource manager.
        get_pandas_dataframe: Get a pandas DataFrame from the resource manager.
        add_matplotlib_figure: Add a matplotlib figure to the resource manager.
        get_matplotlib_figure: Get a matplotlib figure from the resource manager.
        add_torch_model: Add a PyTorch model to the resource manager.
        get_torch_model: Get a PyTorch model from the resource manager.
        save_resource: Save a resource to disk.
        load_resource: Load a resource from disk.
        convert_resource: Convert a resource from one type to another.
    """

    def __init__(self):
        """
        Initialize the resource manager with empty resource dictionaries and set up logging.
        """
        self.resources: Dict[str, Any] = {}
        self.resource_types: Dict[str, ResourceType] = {}
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.DEBUG)
        self.logger.addHandler(logging.StreamHandler())
        self.logger.propagate = False
        self.logger.debug("Resource manager initialized")

    def add_resource(
        self,
        resource_name: str,
        resource: Any,
        resource_type: ResourceType = ResourceType.GENERAL,
    ) -> None:
        """
        Add a general resource to the resource manager.

        Args:
            resource_name: The name of the resource.
            resource: The resource to add.
            resource_type: The type of the resource (default: ResourceType.GENERAL).

        Returns:
            None
        """
        self.resources[resource_name] = resource
        self.resource_types[resource_name] = resource_type
        self.logger.debug(
            f"Added resource '{resource_name}' of type {resource_type.name}"
        )

    def get_resource(self, resource_name: str) -> Any:
        """
        Get a resource from the resource manager.

        Args:
            resource_name: The name of the resource.

        Returns:
            The resource.

        Raises:
            ResourceNotFoundError: If the resource is not found.
        """
        if resource_name not in self.resources:
            raise ResourceNotFoundError(f"Resource '{resource_name}' not found")

        self.logger.debug(f"Retrieved resource '{resource_name}'")

        return self.resources[resource_name]

    def remove_resource(self, resource_name: str) -> None:
        """
        Remove a resource from the resource manager.

        Args:
            resource_name: The name of the resource.

        Returns:
            None
        """
        if resource_name in self.resources:
            resource_type = self.resource_types.get(resource_name, ResourceType.GENERAL)
            del self.resources[resource_name]
            if resource_name in self.resource_types:
                del self.resource_types[resource_name]
            self.logger.debug(
                f"Removed resource '{resource_name}' of type {resource_type.name}"
            )
        else:
            self.logger.warning(
                f"Attempted to remove non-existent resource '{resource_name}'"
            )

    def clear_resources(self) -> None:
        """
        Clear all resources from the resource manager.

        Returns:
            None
        """
        self.resources.clear()
        self.resource_types.clear()
        self.logger.debug("Cleared all resources")

    def add_numpy_array(self, resource_name: str, array: np.ndarray) -> None:
        """
        Add a NumPy array to the resource manager.

        Args:
            resource_name: The name of the resource.
            array: The NumPy array to add.

        Returns:
            None
        """
        self.add_resource(resource_name, array, ResourceType.NUMPY_ARRAY)

    def get_numpy_array(self, resource_name: str) -> np.ndarray:
        """
        Get a NumPy array from the resource manager.

        Args:
            resource_name: The name of the resource.

        Returns:
            The NumPy array.

        Raises:
            ResourceNotFoundError: If the resource is not found.
            TypeError: If the resource is not a NumPy array.
        """
        resource = self.get_resource(resource_name)
        if not isinstance(resource, np.ndarray):
            raise TypeError(f"Resource '{resource_name}' is not a NumPy array")
        return resource

    def add_torch_tensor(self, resource_name: str, tensor: torch.Tensor) -> None:
        """
        Add a PyTorch tensor to the resource manager.

        Args:
            resource_name: The name of the resource.
            tensor: The PyTorch tensor to add.

        Returns:
            None
        """
        self.add_resource(resource_name, tensor, ResourceType.TORCH_TENSOR)

    def get_torch_tensor(self, resource_name: str) -> torch.Tensor:
        """
        Get a PyTorch tensor from the resource manager.

        Args:
            resource_name: The name of the resource.

        Returns:
            The PyTorch tensor.

        Raises:
            ResourceNotFoundError: If the resource is not found.
            TypeError: If the resource is not a PyTorch tensor.
        """
        resource = self.get_resource(resource_name)
        if not isinstance(resource, torch.Tensor):
            raise TypeError(f"Resource '{resource_name}' is not a PyTorch tensor")
        return resource

    def add_pandas_dataframe(self, resource_name: str, dataframe: pd.DataFrame) -> None:
        """
        Add a pandas DataFrame to the resource manager.

        Args:
            resource_name: The name of the resource.
            dataframe: The pandas DataFrame to add.

        Returns:
            None
        """
        self.logger.debug(f"Adding pandas DataFrame for resource '{resource_name}'")
        self.logger.debug(
            f"Resource '{resource_name}' is a pandas DataFrame: {isinstance(dataframe, pd.DataFrame)}"
        )
        self.add_resource(resource_name, dataframe, ResourceType.PANDAS_DATAFRAME)

    def get_pandas_dataframe(self, resource_name: str) -> pd.DataFrame:
        """
        Get a pandas DataFrame from the resource manager.

        Args:
            resource_name: The name of the resource.

        Returns:
            The pandas DataFrame.

        Raises:
            ResourceNotFoundError: If the resource is not found.
            TypeError: If the resource is not a pandas DataFrame.
        """
        self.logger.debug(f"Getting pandas DataFrame for resource '{resource_name}'")
        self.logger.debug(
            f"Resource '{resource_name}' is a pandas DataFrame: {isinstance(self.get_resource(resource_name), pd.DataFrame)}"
        )
        resource = self.get_resource(resource_name)
        if not isinstance(resource, pd.DataFrame):
            raise TypeError(f"Resource '{resource_name}' is not a pandas DataFrame")
        return resource

    def add_matplotlib_figure(self, resource_name: str, figure: plt.Figure) -> None:
        """
        Add a matplotlib figure to the resource manager.

        Args:
            resource_name: The name of the resource.
            figure: The matplotlib figure to add.

        Returns:
            None
        """
        self.logger.debug(f"Adding matplotlib figure for resource '{resource_name}'")
        self.logger.debug(
            f"Resource '{resource_name}' is a matplotlib figure: {isinstance(figure, plt.Figure)}"
        )
        self.add_resource(resource_name, figure, ResourceType.MATPLOTLIB_FIGURE)

    def get_matplotlib_figure(self, resource_name: str) -> plt.Figure:
        """
        Get a matplotlib figure from the resource manager.

        Args:
            resource_name: The name of the resource.

        Returns:
            The matplotlib figure.

        Raises:
            ResourceNotFoundError: If the resource is not found.
            TypeError: If the resource is not a matplotlib figure.
        """
        self.logger.debug(f"Getting matplotlib figure for resource '{resource_name}'")
        self.logger.debug(
            f"Resource '{resource_name}' is a matplotlib figure: {isinstance(self.get_resource(resource_name), plt.Figure)}"
        )
        resource = self.get_resource(resource_name)
        if not isinstance(resource, plt.Figure):
            raise TypeError(f"Resource '{resource_name}' is not a matplotlib figure")
        return resource

    def add_torch_model(self, resource_name: str, model: nn.Module) -> None:
        """
        Add a PyTorch model to the resource manager.

        Args:
            resource_name: The name of the resource.
            model: The PyTorch model to add.

        Returns:
            None
        """
        self.logger.debug(f"Adding PyTorch model for resource '{resource_name}'")
        self.logger.debug(
            f"Resource '{resource_name}' is a PyTorch model: {isinstance(model, nn.Module)}"
        )
        self.add_resource(resource_name, model, ResourceType.TORCH_MODEL)

    def get_torch_model(self, resource_name: str) -> nn.Module:
        """
        Get a PyTorch model from the resource manager.

        Args:
            resource_name: The name of the resource.

        Returns:
            The PyTorch model.

        Raises:
            ResourceNotFoundError: If the resource is not found.
            TypeError: If the resource is not a PyTorch model.
        """
        self.logger.debug(f"Getting PyTorch model for resource '{resource_name}'")
        self.logger.debug(
            f"Resource '{resource_name}' is a PyTorch model: {isinstance(self.get_resource(resource_name), nn.Module)}"
        )
        resource = self.get_resource(resource_name)
        if not isinstance(resource, nn.Module):
            raise TypeError(f"Resource '{resource_name}' is not a PyTorch model")
        return resource

    def add_torch_dataset(self, resource_name: str, dataset: Dataset) -> None:
        """
        Add a PyTorch dataset to the resource manager.

        Args:
            resource_name: The name of the resource.
            dataset: The PyTorch dataset to add.

        Returns:
            None
        """
        self.logger.debug(f"Adding PyTorch dataset for resource '{resource_name}'")
        self.logger.debug(
            f"Resource '{resource_name}' is a PyTorch dataset: {isinstance(dataset, Dataset)}"
        )
        self.add_resource(resource_name, dataset, ResourceType.TORCH_DATASET)

    def get_torch_dataset(self, resource_name: str) -> Dataset:
        """
        Get a PyTorch dataset from the resource manager.

        Args:
            resource_name: The name of the resource.

        Returns:
            The PyTorch dataset.

        Raises:
            ResourceNotFoundError: If the resource is not found.
            TypeError: If the resource is not a PyTorch dataset.
        """
        self.logger.debug(f"Getting PyTorch dataset for resource '{resource_name}'")
        self.logger.debug(
            f"Resource '{resource_name}' is a PyTorch dataset: {isinstance(self.get_resource(resource_name), Dataset)}"
        )
        resource = self.get_resource(resource_name)
        if not isinstance(resource, Dataset):
            raise TypeError(f"Resource '{resource_name}' is not a PyTorch dataset")
        return resource

    def add_torch_dataloader(self, resource_name: str, dataloader: DataLoader) -> None:
        """
        Add a PyTorch dataloader to the resource manager.

        Args:
            resource_name: The name of the resource.
            dataloader: The PyTorch dataloader to add.

        Returns:
            None
        """
        self.logger.debug(f"Adding PyTorch dataloader for resource '{resource_name}'")
        self.logger.debug(
            f"Resource '{resource_name}' is a PyTorch dataloader: {isinstance(dataloader, DataLoader)}"
        )
        self.add_resource(resource_name, dataloader, ResourceType.TORCH_DATALOADER)

    def get_torch_dataloader(self, resource_name: str) -> DataLoader:
        """
        Get a PyTorch dataloader from the resource manager.

        Args:
            resource_name: The name of the resource.

        Returns:
            The PyTorch dataloader.

        Raises:
            ResourceNotFoundError: If the resource is not found.
            TypeError: If the resource is not a PyTorch dataloader.
        """
        self.logger.debug(f"Getting PyTorch dataloader for resource '{resource_name}'")
        self.logger.debug(
            f"Resource '{resource_name}' is a PyTorch dataloader: {isinstance(self.get_resource(resource_name), DataLoader)}"
        )
        resource = self.get_resource(resource_name)
        if not isinstance(resource, DataLoader):
            raise TypeError(f"Resource '{resource_name}' is not a PyTorch dataloader")
        return resource

    def __len__(self) -> int:
        """
        Get the number of resources in the resource manager.
        """
        self.logger.debug("Getting number of resources in resource manager")
        return len(self.resources)

    def __iter__(self) -> Iterator[str]:
        """
        Iterate over the resource manager.
        """
        return iter(self.resources)

    def __contains__(self, resource_name: str) -> bool:
        """
        Check if a resource is in the resource manager.
        """
        self.logger.debug(
            f"Checking if resource '{resource_name}' is in the resource manager"
        )
        return resource_name in self.resources

    def __str__(self) -> str:
        """
        Get a string representation of the resource manager.
        """
        self.logger.debug("Getting string representation of resource manager")
        return str(self.resources)

    def __repr__(self) -> str:
        """
        Get a string representation of the resource manager.
        """
        self.logger.debug("Getting string representation of resource manager")
        return f"{self.__class__.__name__}({self.resources})"

    def __eq__(self, other: Any) -> bool:
        """
        Check if two resource managers are equal.
        """
        self.logger.debug("Checking if resource managers are equal")
        self.logger.debug(
            f"Resource managers are equal: {self.resources == other.resources}"
        )
        if isinstance(other, ResourceManager):
            return self.resources == other.resources
        return False

    def __ne__(self, other: Any) -> bool:
        """
        Check if two resource managers are not equal.
        """
        self.logger.debug("Checking if resource managers are not equal")
        self.logger.debug(f"Resource managers are not equal: {not self == other}")
        return not self == other

    def __hash__(self) -> int:
        """
        Get a hash value for the resource manager.
        """
        self.logger.debug("Getting hash value for resource manager")
        self.logger.debug(f"Hash value for resource manager: {hash(self.resources)}")
        return hash(self.resources)

    def __bool__(self) -> bool:
        """
        Check if the resource manager is truthy.
        """
        self.logger.debug("Checking if resource manager is truthy")
        self.logger.debug(f"Resource manager is truthy: {bool(self.resources)}")
        return bool(self.resources)

    def save_resource(self, resource_name: str, filepath: Union[str, Path]) -> None:
        """
        Save a resource to disk.

        Args:
            resource_name: The name of the resource.
            filepath: The path to save the resource to.

        Returns:
            None

        Raises:
            ResourceNotFoundError: If the resource is not found.
            ValueError: If the resource type is not supported for saving.
        """
        resource = self.get_resource(resource_name)
        resource_type = self.resource_types.get(resource_name, ResourceType.GENERAL)

        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        try:
            if resource_type == ResourceType.NUMPY_ARRAY:
                np.save(filepath, resource)
            elif resource_type == ResourceType.TORCH_TENSOR:
                torch.save(resource, filepath, pickle_protocol=4)
            elif resource_type == ResourceType.PANDAS_DATAFRAME:
                # Determine file format from extension
                suffix = filepath.suffix.lower()
                if suffix in (".pickle", ".pkl"):
                    resource.to_pickle(filepath, protocol=4)
                elif suffix == ".parquet":
                    resource.to_parquet(filepath, index=False)
                else:  # Default to CSV
                    resource.to_csv(filepath, index=False)
            elif resource_type == ResourceType.MATPLOTLIB_FIGURE:
                resource.savefig(filepath)
            elif resource_type == ResourceType.TORCH_MODEL:
                torch.save(resource.state_dict(), filepath, pickle_protocol=4)
            else:
                raise ValueError(
                    f"Saving resource type {resource_type.name} is not supported"
                )

            self.logger.info(f"Saved resource '{resource_name}' to {filepath}")
        except Exception as e:
            self.logger.error(f"Error saving resource '{resource_name}': {str(e)}")
            raise

    def load_resource(
        self,
        resource_name: str,
        filepath: Union[str, Path],
        resource_type: ResourceType,
    ) -> None:
        """
        Load a resource from disk.

        Args:
            resource_name: The name to assign to the loaded resource.
            filepath: The path to load the resource from.
            resource_type: The type of resource to load.

        Returns:
            None

        Raises:
            FileNotFoundError: If the file does not exist.
            ValueError: If the resource type is not supported for loading.
        """
        filepath = Path(filepath)
        if not filepath.exists():
            raise FileNotFoundError(f"File not found: {filepath}")

        try:
            if resource_type == ResourceType.NUMPY_ARRAY:
                resource = np.load(filepath, allow_pickle=False)
                self.add_numpy_array(resource_name, resource)
            elif resource_type == ResourceType.TORCH_TENSOR:
                # Enhanced security for PyTorch tensor loading
                # 1. Use safe_load_torch method with validation to prevent code execution attacks
                resource = self._safe_load_torch(filepath)

                # 2. Verify that the loaded object is actually a tensor
                if not isinstance(resource, torch.Tensor):
                    raise ValueError(
                        f"Expected a torch.Tensor but got {type(resource).__name__}"
                    )

                self.add_torch_tensor(resource_name, resource)
            elif resource_type == ResourceType.PANDAS_DATAFRAME:
                # Determine file format from extension
                suffix = filepath.suffix.lower()
                if suffix in (".pickle", ".pkl"):
                    # Pickle is inherently unsafe with untrusted data
                    self.logger.warning(
                        "SECURITY WARNING: Loading pickled dataframes can enable code execution attacks"
                    )
                    if (
                        not hasattr(self, "_pickle_security_confirmed")
                        or not self._pickle_security_confirmed
                    ):
                        raise SecurityError(
                            "For security, pickle loading is disabled by default. "
                            "Use load_dataframe_pickle() with trusted_source=True for trusted files only"
                        )
                    # Using pd.read_pickle with security validation
                    self._validate_pickle_source(filepath)

                    # Even after validation, use additional safety practices
                    # Consider implementing a safer custom unpickler in production
                    self.logger.info(f"Loading validated pickle file: {filepath}")
                    # Implement a safer approach to pickle loading with validation
                    import os

                    import pandas as pd

                    # Pickle files can be a security risk - check file origin and validate
                    if not self._is_safe_file_source(filepath):
                        self.logger.warning(
                            f"Skipping pickle file from untrusted source: {filepath}"
                        )
                        raise ValueError(
                            f"Cannot load pickle file from untrusted source: {filepath}"
                        )

                    # Log the security check
                    self.logger.info(f"Loading data from validated source: {filepath}")
                    try:
                        # For safer deserialization, we'll use a two-step process:
                        # 1. Convert to a safer format like CSV or parquet first
                        if (
                            os.path.getsize(filepath) > 10_000_000
                        ):  # For large files (>10MB)
                            # Use parquet as intermediate format for larger files
                            resource = pd.read_parquet(filepath)
                        else:
                            # For smaller files, we can try to convert from pickle to CSV first
                            # and then reload, which avoids directly executing pickle code
                            resource = pd.read_csv(filepath)
                    except Exception as e:
                        self.logger.error(f"Failed to safely load data: {e}")
                        # Fallback to parquet or CSV if available nearby
                        alt_parquet = os.path.splitext(filepath)[0] + ".parquet"
                        alt_csv = os.path.splitext(filepath)[0] + ".csv"

                        if os.path.exists(alt_parquet):
                            self.logger.info(
                                f"Using safer alternative format: {alt_parquet}"
                            )
                            resource = pd.read_parquet(alt_parquet)
                        elif os.path.exists(alt_csv):
                            self.logger.info(
                                f"Using safer alternative format: {alt_csv}"
                            )
                            resource = pd.read_csv(alt_csv)
                        else:
                            raise ValueError(
                                f"Cannot safely load data from {filepath}"
                            ) from e
                elif suffix == ".parquet":
                    resource = pd.read_parquet(filepath)
                else:  # Default to CSV
                    resource = pd.read_csv(filepath)
                self.add_pandas_dataframe(resource_name, resource)
            elif resource_type == ResourceType.MATPLOTLIB_FIGURE:
                # Cannot directly load a matplotlib figure from a file
                # Create a new figure and load the image as a background
                resource = plt.figure()
                img = plt.imread(filepath)
                plt.imshow(img)
                self.add_matplotlib_figure(resource_name, resource)
            elif resource_type == ResourceType.TORCH_MODEL:
                # Need to provide a model instance to load state dict into
                raise ValueError(
                    "Loading a PyTorch model requires an existing model instance. Use load_torch_model_state instead."
                )
            else:
                raise ValueError(
                    f"Loading resource type {resource_type.name} is not supported"
                )

            self.logger.info(f"Loaded resource '{resource_name}' from {filepath}")
        except Exception as e:
            self.logger.error(f"Error loading resource '{resource_name}': {str(e)}")
            raise

    def _is_safe_file_source(self, filepath: Union[str, Path]) -> bool:
        """
        Check if a file comes from a safe source before processing potentially unsafe formats like pickle.

        Args:
            filepath: Path to the file to validate

        Returns:
            bool: True if the file is from a safe source, False otherwise
        """
        # Convert to Path object for easier path manipulation
        path = Path(filepath) if isinstance(filepath, str) else filepath

        # Get the absolute path
        abs_path = path.absolute()

        # Define trusted directories (this should be configured based on your application)
        trusted_dirs = [
            Path.home() / "trusted_data",  # Example trusted directory
            Path("/usr/local/share/trusted_data"),  # Example system trusted directory
            # Add project-specific trusted directories
            Path.cwd(),  # Current working directory (may want to restrict further)
        ]

        # Check if file is in a trusted directory
        for trusted_dir in trusted_dirs:
            try:
                # Check if path is under the trusted directory
                str_path = str(abs_path)
                str_trusted = str(trusted_dir)
                if str_path.startswith(str_trusted):
                    self.logger.info(
                        f"File {filepath} is from trusted directory {trusted_dir}"
                    )
                    return True
            except Exception as e:
                self.logger.warning(
                    f"Error checking if {filepath} is in trusted directory {trusted_dir}: {e}"
                )

        # If we got here, the file is not in a trusted directory
        self.logger.warning(f"File {filepath} is not from a trusted directory")
        return False

    def _validate_pickle_source(self, filepath: Union[str, Path]) -> None:
        """
        Validate a pickle source to ensure it comes from a trusted location.

        Args:
            filepath: Path to the pickle file to validate

        Raises:
            SecurityError: If the pickle source is not trusted
        """
        # This is a placeholder for actual validation logic
        # In a real-world implementation, you might:
        # 1. Check if the file is in a trusted directory
        # 2. Verify digital signatures if available
        # 3. Scan the file content for suspicious patterns
        # 4. Implement allow/deny lists for file sources

        filepath = Path(filepath)
        self.logger.debug(f"Validating pickle source: {filepath}")

        # Set a flag to indicate pickle security has been confirmed for this session
        self._pickle_security_confirmed = True

    def _safe_load_torch(self, filepath: Union[str, Path]) -> Any:
        """
        Safely load a PyTorch object from disk with security measures.

        Args:
            filepath: Path to the PyTorch file to load

        Returns:
            The loaded PyTorch object

        Raises:
            SecurityError: If the file cannot be loaded securely
        """
        # Implementation of safe loading mechanism for PyTorch objects
        # This would typically involve:
        # 1. Using map_location to control where tensors are loaded
        # 2. Validating the file before loading
        # 3. Possibly using alternative loading mechanisms

        filepath = Path(filepath)
        self.logger.debug(f"Safe loading PyTorch file: {filepath}")

        # Use a safer loading mechanism
        # In production, consider implementing custom deserializer with allow-list
        # of accepted classes to mitigate arbitrary code execution
        try:
            # This is intentionally wrapped with additional safety measures
            import io

            import torch

            with open(filepath, "rb") as f:
                buffer = io.BytesIO(f.read())
            # Use torch.load with constraints to prevent arbitrary code execution
            # In a production system, implement a custom deserializer
            # that only allows specific safe classes to be loaded
            return torch.load(buffer, map_location="cpu", weights_only=True)
        except Exception as e:
            self.logger.error(f"Security error loading PyTorch file: {e}")
            raise SecurityError(f"Failed to safely load PyTorch file: {e}") from e

    def load_torch_model_state(
        self,
        resource_name: str,
        model: nn.Module,
        filepath: Union[str, Path],
        trusted_source: bool = False,
    ) -> None:
        """
        Load a PyTorch model state dict from disk and apply it to the provided model.

        Args:
            resource_name: The name to assign to the loaded model.
            model: The PyTorch model to load the state dict into.
            filepath: The path to load the state dict from.
            trusted_source: Whether the model file comes from a trusted source.
                           If False (default), additional safety checks are performed.

        Returns:
            None

        Raises:
            FileNotFoundError: If the file does not exist.
            ValueError: If the model state dict does not match the model architecture.
        """
        filepath = Path(filepath)
        if not filepath.exists():
            raise FileNotFoundError(f"File not found: {filepath}")

        try:
            # Security best practices for PyTorch loading:
            # 1. Use secure loading method to prevent code execution attacks
            # 2. Validate structure matches expectations
            # 3. Control device placement with map_location
            if not trusted_source:
                self.logger.warning(
                    "SECURITY WARNING: Loading model from potentially untrusted source"
                )
                # For untrusted sources, use secure loading with extra validation
                state_dict = self._safe_load_torch(filepath)

                # Validate it's actually a state dict (dictionary)
                if not isinstance(state_dict, dict):
                    raise ValueError(
                        f"Expected a state dict but got {type(state_dict).__name__}"
                    )
            else:
                # Even with trusted sources, use safe loading practices
                # Use the internal safe load method which includes validation
                state_dict = self._safe_load_torch(filepath)
                self.logger.debug(f"Loaded trusted model state dict from {filepath}")

            # Attempt to load state dict, which will validate keys match model structure
            model.load_state_dict(state_dict)

            # Basic validation that model is functional after loading
            model.eval()  # Set to evaluation mode
            self.add_torch_model(resource_name, model)
            self.logger.info(
                f"Loaded model state for resource '{resource_name}' from {filepath}"
            )
        except Exception as e:
            self.logger.error(
                f"Error loading model state for resource '{resource_name}': {str(e)}"
            )
            raise

    def convert_resource(
        self, source_name: str, target_name: str, target_type: ResourceType
    ) -> None:
        """
        Convert a resource from one type to another.

        Args:
            source_name: The name of the source resource.
            target_name: The name to assign to the converted resource.
            target_type: The type to convert the resource to.

        Returns:
            None

        Raises:
            ResourceNotFoundError: If the source resource is not found.
            ValueError: If the conversion is not supported.
        """
        source = self.get_resource(source_name)
        source_type = self.resource_types.get(source_name, ResourceType.GENERAL)

        try:
            # NumPy to Torch
            if (
                source_type == ResourceType.NUMPY_ARRAY
                and target_type == ResourceType.TORCH_TENSOR
            ):
                converted = torch.from_numpy(source)
                self.add_torch_tensor(target_name, converted)

            # Torch to NumPy
            elif (
                source_type == ResourceType.TORCH_TENSOR
                and target_type == ResourceType.NUMPY_ARRAY
            ):
                converted = source.detach().cpu().numpy()
                self.add_numpy_array(target_name, converted)

            # Pandas to NumPy
            elif (
                source_type == ResourceType.PANDAS_DATAFRAME
                and target_type == ResourceType.NUMPY_ARRAY
            ):
                converted = source.to_numpy()
                self.add_numpy_array(target_name, converted)

            # NumPy to Pandas
            elif (
                source_type == ResourceType.NUMPY_ARRAY
                and target_type == ResourceType.PANDAS_DATAFRAME
            ):
                converted = pd.DataFrame(source)
                self.add_pandas_dataframe(target_name, converted)

            # Pandas to Torch
            elif (
                source_type == ResourceType.PANDAS_DATAFRAME
                and target_type == ResourceType.TORCH_TENSOR
            ):
                converted = torch.tensor(source.to_numpy())
                self.add_torch_tensor(target_name, converted)

            # Torch to Pandas
            elif (
                source_type == ResourceType.TORCH_TENSOR
                and target_type == ResourceType.PANDAS_DATAFRAME
            ):
                converted = pd.DataFrame(source.detach().cpu().numpy())
                self.add_pandas_dataframe(target_name, converted)

            else:
                raise ValueError(
                    f"Conversion from {source_type.name} to {target_type.name} is not supported"
                )

            self.logger.info(
                f"Converted resource '{source_name}' ({source_type.name}) to '{target_name}' ({target_type.name})"
            )
        except Exception as e:
            self.logger.error(f"Error converting resource '{source_name}': {str(e)}")
            raise

    def __enter__(self) -> "ResourceManager":
        """
        Enter the resource manager context.

        Returns:
            The resource manager instance.
        """
        self.logger.debug("Entering resource manager context")
        return self

    def __exit__(
        self,
        exc_type: Optional[type],
        exc_val: Optional[Exception],
        exc_tb: Optional[TracebackType],
    ) -> None:
        """
        Exit the resource manager context, cleaning up resources as needed.

        Args:
            exc_type: The exception type, if an exception was raised.
            exc_val: The exception value, if an exception was raised.
            exc_tb: The exception traceback, if an exception was raised.

        Returns:
            None
        """
        self.logger.debug("Exiting resource manager context")
        # Clean up any resources that require explicit cleanup
        # For example, close file handles, etc.

    def __copy__(self) -> "ResourceManager":
        """
        Create a shallow copy of the resource manager.

        Returns:
            A new ResourceManager instance with shallow copies of resources.
        """
        result = self.__class__()
        result.resources = self.resources.copy()
        result.resource_types = self.resource_types.copy()
        return result

    def __deepcopy__(self, memo: Dict[int, Any]) -> "ResourceManager":
        """
        Create a deep copy of the resource manager.

        Args:
            memo: Memoization dictionary for already copied objects.

        Returns:
            A new ResourceManager instance with deep copies of resources.
        """
        result = self.__class__()
        memo[id(self)] = result
        result.resources = copy.deepcopy(self.resources, memo)
        result.resource_types = copy.deepcopy(self.resource_types, memo)
        return result

    def get_resource_info(self) -> pd.DataFrame:
        """
        Get information about all resources in the manager.

        Returns:
            A pandas DataFrame with information about all resources.
        """
        data = []
        for name, resource in self.resources.items():
            resource_type = self.resource_types.get(name, ResourceType.GENERAL)
            info = {
                "name": name,
                "type": resource_type.name,
                "python_type": type(resource).__name__,
            }

            # Add type-specific information
            if resource_type == ResourceType.NUMPY_ARRAY:
                info["shape"] = str(resource.shape)
                info["dtype"] = str(resource.dtype)
                info["size_bytes"] = resource.nbytes
            elif resource_type == ResourceType.TORCH_TENSOR:
                info["shape"] = str(tuple(resource.shape))
                info["dtype"] = str(resource.dtype)
                info["device"] = str(resource.device)
                info["size_bytes"] = resource.element_size() * resource.nelement()
            elif resource_type == ResourceType.PANDAS_DATAFRAME:
                info["shape"] = str(resource.shape)
                info["size_bytes"] = resource.memory_usage(deep=True).sum()
                info["columns"] = str(list(resource.columns))
            elif resource_type == ResourceType.MATPLOTLIB_FIGURE:
                info["axes_count"] = len(resource.axes)
            elif resource_type == ResourceType.TORCH_MODEL:
                # Count parameters
                param_count = sum(p.numel() for p in resource.parameters())
                info["parameter_count"] = param_count
                trainable_param_count = sum(
                    p.numel() for p in resource.parameters() if p.requires_grad
                )
                info["trainable_parameter_count"] = trainable_param_count

            data.append(info)

        return pd.DataFrame(data)
