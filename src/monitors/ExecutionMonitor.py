"""
ExecutionMonitor
"""

import logging
from pathlib import Path
from typing import Any, Dict

from src.processors.MarkdownProcessor import MarkdownProcessingError


class ExecutionMonitor:
    """
    A class representing an execution monitor.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize an ExecutionMonitor object.

        Args:
            config (Dict[str, Any]): The configuration for the execution monitor.
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        self.logger.addHandler(logging.StreamHandler())
        self.logger.propagate = False
        self.logger.info("ExecutionMonitor initialized.")
        self.logger.info("Config: %s", self.config)

    async def monitor(self, vault_path: Path) -> Dict[str, Any]:
        """
        Monitor a vault of Markdown files and return the aggregated results.

        Args:
            vault_path (Path): The path to the vault directory.

        Returns:
            Dict[str, Any]: The aggregated results of monitoring the vault.

        Raises:
            MarkdownProcessingError: If there is an error monitoring the vault.
        """
        self.logger.info("Monitoring vault: %s", vault_path)

        files = []

        try:
            files = list(vault_path.rglob("*.md"))

        except Exception as e:
            self.logger.error(f"Monitoring error: {str(e)}")
            raise MarkdownProcessingError(str(e)) from e

        return {"files": files}

    async def _read_file(self, file_path: Path) -> str:
        """
        Read a Markdown file and return its content.

        Args:
            file_path (Path): The path to the Markdown file.

        Returns:
            str: The content of the Markdown file.

        Raises:
            MarkdownProcessingError: If there is an error reading the file.
        """
        try:
            pass
        except Exception as e:
            self.logger.error(f"File reading error: {str(e)}")
            raise MarkdownProcessingError(str(e)) from e

        return ""
