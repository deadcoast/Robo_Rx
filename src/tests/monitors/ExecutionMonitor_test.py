import asyncio
import logging
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from src.monitors.ExecutionMonitor import ExecutionMonitor
from src.processors.MarkdownProcessor import MarkdownProcessingError


class TestExecutionMonitor:
    """Test suite for the ExecutionMonitor class."""

    @pytest.fixture
    def mock_config(self):
        """Create a mock configuration for testing."""
        return {
            "log_level": "INFO",
            "timeout": 30,
            "max_files": 100,
            "file_extensions": [".md"],
            "ignore_patterns": ["_template", ".obsidian"],
        }

    @pytest.fixture
    def execution_monitor(self, mock_config):
        """Create an ExecutionMonitor instance for testing."""
        with patch("src.monitors.ExecutionMonitor.logging"):
            monitor = ExecutionMonitor(mock_config)
            # Replace the logger with a mock for testing
            monitor.logger = Mock(spec=logging.Logger)
            return monitor

    @pytest.fixture
    def mock_vault_path(self, tmp_path):
        """Create a temporary vault path with some Markdown files for testing."""
        vault_path = tmp_path / "test_vault"
        vault_path.mkdir()

        # Create some test markdown files
        (vault_path / "file1.md").write_text("# Test File 1\nContent 1")
        (vault_path / "file2.md").write_text("# Test File 2\nContent 2")

        # Create a subdirectory with files
        subdir = vault_path / "subdir"
        subdir.mkdir()
        (subdir / "file3.md").write_text("# Test File 3\nContent 3")

        # Create a non-markdown file
        (vault_path / "not_markdown.txt").write_text("Not a markdown file")

        return vault_path

    def test_init(self, mock_config):
        """Test initialization of ExecutionMonitor."""
        with patch("src.monitors.ExecutionMonitor.logging") as mock_logging:
            # Setup logger mock
            mock_logger = Mock(spec=logging.Logger)
            mock_logging.getLogger.return_value = mock_logger

            # Create monitor
            monitor = ExecutionMonitor(mock_config)

            # Verify initializations
            assert monitor.config == mock_config

            # Verify logger setup
            mock_logging.getLogger.assert_called_once_with(
                "src.monitors.ExecutionMonitor"
            )
            mock_logger.setLevel.assert_called_once_with(logging.INFO)
            mock_logger.addHandler.assert_called_once()
            mock_logger.info.assert_any_call("ExecutionMonitor initialized.")
            mock_logger.info.assert_any_call(f"Config: {mock_config}")

    @pytest.mark.asyncio
    async def test_monitor_success(self, execution_monitor, mock_vault_path):
        """Test the monitor method with a valid vault path."""
        result = await execution_monitor.monitor(mock_vault_path)

        # Verify that the method found the markdown files
        assert isinstance(result, dict)
        assert "files" in result
        assert len(result["files"]) == 3  # We expect 3 .md files

        # All files should be Path objects with .md extension
        for file in result["files"]:
            assert isinstance(file, Path)
            assert file.suffix == ".md"

        # Verify logging
        execution_monitor.logger.info.assert_any_call(
            f"Monitoring vault: {mock_vault_path}"
        )

    @pytest.mark.asyncio
    async def test_monitor_empty_vault(self, execution_monitor, tmp_path):
        """Test the monitor method with an empty vault."""
        empty_vault = tmp_path / "empty_vault"
        empty_vault.mkdir()

        result = await execution_monitor.monitor(empty_vault)

        # Verify that the method returns an empty file list
        assert isinstance(result, dict)
        assert "files" in result
        assert len(result["files"]) == 0

        # Verify logging
        execution_monitor.logger.info.assert_any_call(
            f"Monitoring vault: {empty_vault}"
        )

    @pytest.mark.asyncio
    async def test_monitor_nonexistent_vault(self, execution_monitor, tmp_path):
        """Test the monitor method with a nonexistent vault path."""
        nonexistent_vault = (
            tmp_path / "nonexistent_vault"
        )  # This directory doesn't exist

        # Should raise MarkdownProcessingError
        with pytest.raises(MarkdownProcessingError):
            await execution_monitor.monitor(nonexistent_vault)

        # Verify error logging
        execution_monitor.logger.error.assert_called_once()

    @pytest.mark.asyncio
    async def test_monitor_with_exception(self, execution_monitor, mock_vault_path):
        """Test the monitor method when an exception occurs during processing."""
        # Mock Path.rglob to raise an exception
        with patch("pathlib.Path.rglob", side_effect=PermissionError("Access denied")):
            # Should raise MarkdownProcessingError
            with pytest.raises(MarkdownProcessingError) as excinfo:
                await execution_monitor.monitor(mock_vault_path)

            # Verify the original exception is propagated
            assert "Access denied" in str(excinfo.value)

            # Verify error logging
            execution_monitor.logger.error.assert_called_once()

    @pytest.mark.asyncio
    async def test_read_file_not_implemented(self, execution_monitor, mock_vault_path):
        """Test the _read_file method which is not fully implemented."""
        test_file = mock_vault_path / "file1.md"

        # Method should return empty string (based on current implementation)
        content = await execution_monitor._read_file(test_file)
        assert content == ""

    def test_multiple_instances(self, mock_config):
        """Test that multiple instances can be created with different configs."""
        with patch("src.monitors.ExecutionMonitor.logging"):
            # Create two monitors with different configs
            config1 = mock_config.copy()
            config2 = mock_config.copy()
            config2["timeout"] = 60

            monitor1 = ExecutionMonitor(config1)
            monitor2 = ExecutionMonitor(config2)

            # Verify they have different configs
            assert monitor1.config != monitor2.config
            assert monitor1.config["timeout"] == 30
            assert monitor2.config["timeout"] == 60

    @pytest.mark.asyncio
    async def test_monitor_with_concurrency(self, execution_monitor, mock_vault_path):
        """Test that the monitor method can be called concurrently."""
        # Create tasks for concurrent execution
        tasks = [
            asyncio.create_task(execution_monitor.monitor(mock_vault_path)),
            asyncio.create_task(execution_monitor.monitor(mock_vault_path)),
        ]

        # Wait for all tasks to complete
        results = await asyncio.gather(*tasks)

        # Verify both calls succeeded
        assert len(results) == 2
        assert all(isinstance(result, dict) for result in results)
        assert all("files" in result for result in results)
        assert all(len(result["files"]) == 3 for result in results)  # 3 .md files each

    @pytest.mark.asyncio
    async def test_custom_exception_handling(self, execution_monitor):
        """Test that exceptions are properly wrapped in MarkdownProcessingError."""
        # Create a test file path
        test_file = Path("/nonexistent/path/file.md")

        # Should raise MarkdownProcessingError
        with pytest.raises(MarkdownProcessingError) as excinfo:
            # Test the _read_file method with a path that will cause an exception
            with patch.object(
                execution_monitor,
                "_read_file",
                side_effect=FileNotFoundError("File not found"),
            ):
                await execution_monitor._read_file(test_file)

        # Verify the original exception is wrapped properly
        assert "File not found" in str(excinfo.value)

        # Verify the original exception is set as the cause
        assert isinstance(excinfo.value.__cause__, FileNotFoundError)
