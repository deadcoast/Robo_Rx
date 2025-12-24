import logging
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import AsyncMock, Mock, mock_open, patch

import pytest

from src.processors.MarkdownProcessor import MarkdownProcessingError, MarkdownProcessor


class TestMarkdownProcessingError:
    """Test suite for the MarkdownProcessingError class."""

    def test_init_with_file_path(self):
        """Test initialization with file path."""
        message = "Error processing markdown"
        file_path = Path("/test/file.md")
        error = MarkdownProcessingError(message=message, file_path=file_path)

        assert error.message == message
        assert error.file_path == file_path

    def test_init_without_file_path(self):
        """Test initialization without file path."""
        message = "Error processing markdown"
        error = MarkdownProcessingError(message=message)

        assert error.message == message
        assert error.file_path is None

    def test_str_with_file_path(self):
        """Test string representation with file path."""
        message = "Error processing markdown"
        file_path = Path("/test/file.md")
        error = MarkdownProcessingError(message=message, file_path=file_path)

        expected = f"{message} - File: {file_path}"
        assert str(error) == expected

    def test_str_without_file_path(self):
        """Test string representation without file path."""
        message = "Error processing markdown"
        error = MarkdownProcessingError(message=message)

        assert str(error) == message

    def test_repr(self):
        """Test repr representation."""
        message = "Error processing markdown"
        file_path = Path("/test/file.md")
        error = MarkdownProcessingError(message=message, file_path=file_path)

        expected = f"MarkdownProcessingError(message={message}, file_path={file_path})"
        assert repr(error) == expected

    def test_eq_equal(self):
        """Test equality operator with equal objects."""
        message = "Error processing markdown"
        error1 = MarkdownProcessingError(message=message)
        error2 = MarkdownProcessingError(message=message)

        assert error1 == error2

    def test_eq_not_equal(self):
        """Test equality operator with unequal objects."""
        error1 = MarkdownProcessingError(message="Error 1")
        error2 = MarkdownProcessingError(message="Error 2")

        assert error1 != error2

    def test_eq_different_type(self):
        """Test equality operator with different type."""
        error = MarkdownProcessingError(message="Error")

        assert error != "Error"


class TestMarkdownProcessor:
    """Test suite for the MarkdownProcessor class."""

    @pytest.fixture
    def config(self):
        """Create a configuration dictionary for testing."""
        return {"max_threads": 2, "batch_size": 5}

    @pytest.fixture
    def markdown_processor(self, config):
        """Create a MarkdownProcessor instance for testing."""
        with patch("logging.getLogger") as mock_get_logger:
            # Create a mock logger to avoid actual logging during tests
            mock_logger = Mock(spec=logging.Logger)
            mock_get_logger.return_value = mock_logger

            # Create and return MarkdownProcessor instance
            processor = MarkdownProcessor(config)

            # Set the mocked logger so tests can access it
            processor.logger = mock_logger

            # Mock the ThreadPoolExecutor
            processor.executor = Mock(spec=ThreadPoolExecutor)

            return processor

    def test_init(self, markdown_processor, config):
        """Test initialization of MarkdownProcessor."""
        assert markdown_processor.config == config
        assert markdown_processor.max_threads == config["max_threads"]
        assert markdown_processor.batch_size == config["batch_size"]
        assert markdown_processor.logger is not None
        assert markdown_processor.executor is not None
        markdown_processor.logger.info.assert_any_call("MarkdownProcessor initialized.")

    def test_init_default_values(self):
        """Test initialization with default values."""
        empty_config = {}

        with patch("logging.getLogger") as mock_get_logger:
            mock_logger = Mock(spec=logging.Logger)
            mock_get_logger.return_value = mock_logger

            processor = MarkdownProcessor(empty_config)

            assert processor.max_threads == 4  # Default value
            assert processor.batch_size == 10  # Default value

    @pytest.mark.asyncio
    async def test_process_vault_success(self, markdown_processor):
        """Test processing a vault successfully."""
        # Setup
        vault_path = Path("/test/vault")
        mock_files = [
            Path("/test/vault/file1.md"),
            Path("/test/vault/file2.md"),
            Path("/test/vault/subdir/file3.md"),
        ]

        # Mock file discovery
        with patch("pathlib.Path.rglob") as mock_rglob:
            mock_rglob.return_value = mock_files

            # Mock processing of individual files
            markdown_processor._process_file = AsyncMock(
                side_effect=[
                    {"id": "file1", "content": "content1"},
                    {"id": "file2", "content": "content2"},
                    {"id": "file3", "content": "content3"},
                ]
            )

            # Mock result aggregation
            expected_result = {
                "files": 3,
                "data": {"content": ["content1", "content2", "content3"]},
            }
            markdown_processor._aggregate_results = Mock(return_value=expected_result)

            # Call the method
            result = await markdown_processor.process_vault(vault_path)

            # Verify
            assert result == expected_result
            mock_rglob.assert_called_once_with("*.md")
            assert markdown_processor._process_file.call_count == 3
            markdown_processor._aggregate_results.assert_called_once()
            markdown_processor.logger.info.assert_any_call(
                f"Processing vault: {vault_path}"
            )
            markdown_processor.logger.info.assert_any_call(
                f"Found {len(mock_files)} markdown files"
            )

    @pytest.mark.asyncio
    async def test_process_vault_empty(self, markdown_processor):
        """Test processing an empty vault."""
        # Setup
        vault_path = Path("/test/empty_vault")

        # Mock file discovery - empty list
        with patch("pathlib.Path.rglob") as mock_rglob:
            mock_rglob.return_value = []

            # Mock result aggregation
            expected_result = {"files": 0, "data": {}}
            markdown_processor._aggregate_results = Mock(return_value=expected_result)

            # Call the method
            result = await markdown_processor.process_vault(vault_path)

            # Verify
            assert result == expected_result
            mock_rglob.assert_called_once_with("*.md")
            markdown_processor._process_file.assert_not_called()
            markdown_processor._aggregate_results.assert_called_once_with([])

    @pytest.mark.asyncio
    async def test_process_vault_batching(self, markdown_processor):
        """Test processing files in batches."""
        # Setup
        vault_path = Path("/test/vault")
        # Create 12 mock files (should be processed in 3 batches with batch_size=5)
        mock_files = [Path(f"/test/vault/file{i}.md") for i in range(1, 13)]

        # Mock file discovery
        with patch("pathlib.Path.rglob") as mock_rglob:
            mock_rglob.return_value = mock_files

            # Mock processing of individual files
            file_results = [
                {"id": f"file{i}", "content": f"content{i}"} for i in range(1, 13)
            ]
            markdown_processor._process_file = AsyncMock(side_effect=file_results)

            # Mock result aggregation
            expected_result = {"files": 12, "data": {}}
            markdown_processor._aggregate_results = Mock(return_value=expected_result)

            # Call the method
            result = await markdown_processor.process_vault(vault_path)

            # Verify
            assert result == expected_result
            assert markdown_processor._process_file.call_count == 12
            markdown_processor.logger.info.assert_any_call(
                f"Found {len(mock_files)} markdown files"
            )

    @pytest.mark.asyncio
    async def test_process_vault_error(self, markdown_processor):
        """Test error handling during vault processing."""
        # Setup
        vault_path = Path("/test/vault")

        # Mock file discovery to raise an exception
        with patch("pathlib.Path.rglob") as mock_rglob:
            mock_rglob.side_effect = Exception("Test error")

            # Call the method and verify exception
            with pytest.raises(MarkdownProcessingError) as exc_info:
                await markdown_processor.process_vault(vault_path)

            # Verify
            assert "Error processing vault" in str(exc_info.value)
            assert "Test error" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_process_file_success(self, markdown_processor):
        """Test processing a single file successfully."""
        # Setup
        file_path = Path("/test/vault/file1.md")
        file_content = "# Heading\n\nContent paragraph."

        # Mock file reading
        with patch("builtins.open", mock_open(read_data=file_content)):
            # Mock parsing and metadata extraction
            markdown_processor._parse_markdown = Mock(
                return_value={"content": "parsed content"}
            )
            markdown_processor._extract_metadata = Mock(
                return_value={"title": "Heading"}
            )

            # Call the method
            result = await markdown_processor._process_file(file_path)

            # Verify
            assert result is not None
            assert "file_path" in result
            assert result["file_path"] == str(file_path)
            assert "content" in result
            assert "metadata" in result
            markdown_processor._parse_markdown.assert_called_once_with(file_content)
            markdown_processor._extract_metadata.assert_called_once_with(file_content)
            markdown_processor.logger.info.assert_any_call(
                f"Processing file: {file_path}"
            )

    @pytest.mark.asyncio
    async def test_process_file_read_error(self, markdown_processor):
        """Test error handling when file cannot be read."""
        # Setup
        file_path = Path("/test/vault/file1.md")

        # Mock file reading to raise an exception
        with patch("builtins.open", side_effect=IOError("File not found")):
            # Call the method
            result = await markdown_processor._process_file(file_path)

            # Verify
            assert result is None
            markdown_processor.logger.error.assert_called()

    @pytest.mark.asyncio
    async def test_process_file_processing_error(self, markdown_processor):
        """Test error handling during file processing."""
        # Setup
        file_path = Path("/test/vault/file1.md")
        file_content = "# Heading\n\nContent paragraph."

        # Mock file reading
        with patch("builtins.open", mock_open(read_data=file_content)):
            # Mock parsing to raise an exception
            markdown_processor._parse_markdown = Mock(
                side_effect=Exception("Parsing error")
            )

            # Call the method
            result = await markdown_processor._process_file(file_path)

            # Verify
            assert result is None
            markdown_processor.logger.error.assert_called()

    def test_parse_markdown(self, markdown_processor):
        """Test parsing markdown content."""
        # Setup
        markdown_content = "# Heading\n\nContent paragraph."

        # Mock the HTML conversion and content extraction
        with patch("src.processors.MarkdownProcessor.markdown") as mock_markdown:
            mock_markdown.return_value = "<h1>Heading</h1><p>Content paragraph.</p>"

            # Call the method
            result = markdown_processor._parse_markdown(markdown_content)

            # Verify
            assert result is not None
            assert "html" in result
            assert result["html"] == "<h1>Heading</h1><p>Content paragraph.</p>"
            assert "plain_text" in result
            mock_markdown.assert_called_once_with(markdown_content)

    def test_extract_metadata(self, markdown_processor):
        """Test extracting metadata from markdown content."""
        # Setup
        markdown_content = "---\ntitle: Test Title\ntags: [test, markdown]\n---\n\n# Heading\n\nContent."

        # Call the method
        result = markdown_processor._extract_metadata(markdown_content)

        # Verify
        assert result is not None
        assert "title" in result
        assert result["title"] == "Test Title"
        assert "tags" in result
        assert result["tags"] == ["test", "markdown"]

    def test_extract_metadata_no_frontmatter(self, markdown_processor):
        """Test extracting metadata when no frontmatter is present."""
        # Setup
        markdown_content = "# Heading\n\nContent paragraph."

        # Call the method
        result = markdown_processor._extract_metadata(markdown_content)

        # Verify
        assert result == {}

    def test_extract_metadata_invalid_frontmatter(self, markdown_processor):
        """Test extracting metadata with invalid frontmatter."""
        # Setup
        markdown_content = "---\ntitle: Test Title\ntags: [test, markdown\n---\n\n# Heading\n\nContent."

        # Call the method
        result = markdown_processor._extract_metadata(markdown_content)

        # Verify
        assert result == {}
        markdown_processor.logger.warning.assert_called()

    def test_aggregate_results(self, markdown_processor):
        """Test aggregating results from multiple files."""
        # Setup
        results = [
            {
                "file_path": "/test/vault/file1.md",
                "content": {"html": "<h1>Title 1</h1>", "plain_text": "Title 1"},
                "metadata": {"title": "Title 1", "tags": ["tag1", "tag2"]},
            },
            {
                "file_path": "/test/vault/file2.md",
                "content": {"html": "<h1>Title 2</h1>", "plain_text": "Title 2"},
                "metadata": {"title": "Title 2", "tags": ["tag2", "tag3"]},
            },
        ]

        # Call the method
        aggregated = markdown_processor._aggregate_results(results)

        # Verify
        assert aggregated is not None
        assert "files" in aggregated
        assert aggregated["files"] == 2
        assert "files_data" in aggregated
        assert len(aggregated["files_data"]) == 2
        assert "metadata" in aggregated
        assert "tags" in aggregated["metadata"]
        # Check that tags are properly aggregated
        assert set(aggregated["metadata"]["tags"]) == {"tag1", "tag2", "tag3"}

    def test_aggregate_results_empty(self, markdown_processor):
        """Test aggregating empty results."""
        # Call the method with empty list
        aggregated = markdown_processor._aggregate_results([])

        # Verify
        assert aggregated is not None
        assert "files" in aggregated
        assert aggregated["files"] == 0
        assert "files_data" in aggregated
        assert len(aggregated["files_data"]) == 0
        assert "metadata" in aggregated
        assert "tags" in aggregated["metadata"]
        assert len(aggregated["metadata"]["tags"]) == 0
