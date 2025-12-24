"""
DataHandler class
"""

import logging
from typing import Any, Dict, List

from src.managers.FileManager import FileManager


class DataHandler:
    """Data handler for processing and managing data."""

    def __init__(self):
        """Initialize the DataHandler."""
        self._data: Dict[str, Any] = {}
        self._path: str = ""
        self._file_manager: FileManager = FileManager()
        self._logger: logging.Logger = logging.getLogger(__name__)
        self._logger.setLevel(logging.INFO)
        self._logger.addHandler(logging.StreamHandler())
        self._logger.info("DataHandler initialized")

    async def process_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Process data."""
        self._data = data
        self._logger.info("Data processed")
        return data

    async def save_data(self, data: Dict[str, Any], path: str) -> None:
        """Save data to a file."""
        self._data = data
        self._path = path
        self._file_manager.save_data(data, path)
        self._logger.info(f"Data saved to {path}")

    async def load_data(self, path: str) -> Dict[str, Any]:
        """Load data from a file."""
        self._path = path
        self._data = self._file_manager.load_data(path)
        self._logger.info(f"Data loaded from {path}")
        return self._data

    async def delete_data(self, path: str) -> None:
        """Delete data from a file."""
        self._data = {}
        self._file_manager.delete_data(path)
        self._logger.info(f"Data deleted from {path}")

    async def update_data(self, data: Dict[str, Any], path: str) -> None:
        """Update data in a file."""
        self._data.update(data)
        self._file_manager.update_data(data, path)
        self._logger.info(f"Data updated in {path}")

    async def list_data(self, path: str) -> List[Dict[str, Any]]:
        """List data in a directory."""
        self._path = path
        return list(self._data.values())

    async def search_data(self, query: str) -> List[Dict[str, Any]]:
        """Search for data."""
        return [data for data in self._data.values() if query in data.values()]

    async def get_data(self, path: str) -> Dict[str, Any]:
        """Get data from a file."""
        self._path = path
        return self._data
