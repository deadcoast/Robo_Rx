import asyncio
import os
import tempfile
from pathlib import Path

import pytest

from src.main import DocumentProcessor


class TestDocumentProcessor:
    """Test suite for the DocumentProcessor class."""

    @pytest.fixture
    def mock_config(self):
        """Create a mock configuration for testing."""
        return {
            "document_processor": {
                "supported_formats": ["md", "txt", "pdf", "html"],
                "chunk_size": 1000,
                "chunk_overlap": 200,
                "batch_size": 10,
                "extract_metadata": True,
                "normalize_text": True,
            }
        }

    @pytest.fixture
    def document_processor(self, mock_config):
        """Create a DocumentProcessor instance for testing."""
        return DocumentProcessor(mock_config)

    @pytest.fixture
    def sample_markdown_content(self):
        """Create sample markdown content for testing."""
        return """# Sample Document

## Introduction
This is a sample document for testing the DocumentProcessor class.

## Content Section
Here is some content with **bold text** and *italic text*.

### Subsection
- Item 1
- Item 2
- Item 3

## Conclusion
This is the end of the sample document.
"""

    @pytest.fixture
    def sample_text_content(self):
        """Create sample plain text content for testing."""
        return """Sample Document

Introduction
This is a sample document for testing the DocumentProcessor class.

Content Section
Here is some content with bold text and italic text.

Subsection
* Item 1
* Item 2
* Item 3

Conclusion
This is the end of the sample document.
"""

    @pytest.fixture
    def sample_html_content(self):
        """Create sample HTML content for testing."""
        return """<!DOCTYPE html>
<html>
<head>
    <title>Sample Document</title>
</head>
<body>
    <h1>Sample Document</h1>
    
    <h2>Introduction</h2>
    <p>This is a sample document for testing the DocumentProcessor class.</p>
    
    <h2>Content Section</h2>
    <p>Here is some content with <strong>bold text</strong> and <em>italic text</em>.</p>
    
    <h3>Subsection</h3>
    <ul>
        <li>Item 1</li>
        <li>Item 2</li>
        <li>Item 3</li>
    </ul>
    
    <h2>Conclusion</h2>
    <p>This is the end of the sample document.</p>
</body>
</html>
"""

    @pytest.fixture
    def sample_files(
        self, sample_markdown_content, sample_text_content, sample_html_content
    ):
        """Create temporary sample files for testing."""
        temp_dir = tempfile.mkdtemp()

        # Create sample markdown file
        md_path = os.path.join(temp_dir, "sample.md")
        with open(md_path, "w") as f:
            f.write(sample_markdown_content)

        # Create sample text file
        txt_path = os.path.join(temp_dir, "sample.txt")
        with open(txt_path, "w") as f:
            f.write(sample_text_content)

        # Create sample HTML file
        html_path = os.path.join(temp_dir, "sample.html")
        with open(html_path, "w") as f:
            f.write(sample_html_content)

        yield {
            "markdown": Path(md_path),
            "text": Path(txt_path),
            "html": Path(html_path),
            "directory": Path(temp_dir),
        }
        # Cleanup temporary files
        for file_path in [md_path, txt_path, html_path]:
            if os.path.exists(file_path):
                os.remove(file_path)

        if os.path.exists(temp_dir):
            os.rmdir(temp_dir)

    def test_init(self, document_processor, mock_config):
        """Test initialization of DocumentProcessor."""
        assert document_processor.config == mock_config
        assert (
            document_processor.supported_formats
            == mock_config["document_processor"]["supported_formats"]
        )
        assert (
            document_processor.chunk_size
            == mock_config["document_processor"]["chunk_size"]
        )
        assert (
            document_processor.chunk_overlap
            == mock_config["document_processor"]["chunk_overlap"]
        )

    @pytest.mark.asyncio
    async def test_process_markdown_file(self, document_processor, sample_files):
        """Test processing a markdown file."""
        # Process the markdown file
        result = await document_processor.process_file(sample_files["markdown"])

        # Verify the result
        assert isinstance(result, dict)
        assert "content" in result
        assert "metadata" in result
        assert "chunks" in result

        # Verify content is extracted
        assert "Sample Document" in result["content"]
        assert "Introduction" in result["content"]
        assert "Content Section" in result["content"]
        assert "Conclusion" in result["content"]

        # Verify metadata is extracted
        assert "path" in result["metadata"]
        assert "format" in result["metadata"]
        assert "created" in result["metadata"] or "modified" in result["metadata"]
        assert result["metadata"]["format"] == "md"

        # Verify chunks are created
        assert isinstance(result["chunks"], list)
        assert len(result["chunks"]) > 0

    @pytest.mark.asyncio
    async def test_process_text_file(self, document_processor, sample_files):
        """Test processing a plain text file."""
        # Process the text file
        result = await document_processor.process_file(sample_files["text"])

        # Verify the result
        assert isinstance(result, dict)
        assert "content" in result
        assert "metadata" in result
        assert "chunks" in result

        # Verify content is extracted
        assert "Sample Document" in result["content"]
        assert "Introduction" in result["content"]
        assert "Content Section" in result["content"]
        assert "Conclusion" in result["content"]

        # Verify metadata is extracted
        assert "path" in result["metadata"]
        assert "format" in result["metadata"]
        assert result["metadata"]["format"] == "txt"

    @pytest.mark.asyncio
    async def test_process_html_file(self, document_processor, sample_files):
        """Test processing an HTML file."""
        # Process the HTML file
        result = await document_processor.process_file(sample_files["html"])

        # Verify the result
        assert isinstance(result, dict)
        assert "content" in result
        assert "metadata" in result
        assert "chunks" in result

        # Verify content is extracted (HTML tags should be stripped)
        assert "Sample Document" in result["content"]
        assert "Introduction" in result["content"]
        assert "Content Section" in result["content"]
        assert "Conclusion" in result["content"]

        # HTML markup should be removed
        assert "<h1>" not in result["content"]
        assert "<p>" not in result["content"]
        assert "<strong>" not in result["content"]

        # Verify metadata is extracted
        assert "path" in result["metadata"]
        assert "format" in result["metadata"]
        assert result["metadata"]["format"] == "html"

        # HTML might include title metadata
        if "title" in result["metadata"]:
            assert result["metadata"]["title"] == "Sample Document"

    @pytest.mark.asyncio
    async def test_process_directory(self, document_processor, sample_files):
        """Test processing all files in a directory."""
        # Process the directory
        results = await document_processor.process_directory(sample_files["directory"])

        # Verify the results
        assert isinstance(results, list)
        assert len(results) == 3  # Should process all 3 sample files

        # Verify each result has the expected structure
        for result in results:
            assert isinstance(result, dict)
            assert "content" in result
            assert "metadata" in result
            assert "chunks" in result
            assert "path" in result["metadata"]
            assert "format" in result["metadata"]

            # Verify the format is one of the expected formats
            assert result["metadata"]["format"] in ["md", "txt", "html"]

    @pytest.mark.asyncio
    async def test_chunk_text(self, document_processor):
        """Test chunking text into smaller pieces."""
        # Create a long text
        long_text = "This is a test sentence. " * 100  # Should create multiple chunks

        # Chunk the text
        chunks = document_processor.chunk_text(long_text)

        # Verify the chunks
        assert isinstance(chunks, list)
        assert len(chunks) > 1  # Should create multiple chunks

        # Verify each chunk is no larger than the chunk size
        for chunk in chunks:
            assert len(chunk) <= document_processor.chunk_size

        # Verify overlap between chunks
        if len(chunks) >= 2:
            # Get the end of first chunk and start of second chunk
            first_chunk_end = chunks[0][-document_processor.chunk_overlap :]
            second_chunk_start = chunks[1][: document_processor.chunk_overlap]

            # For text chunking, the overlap might not be exact due to
            # chunking at sentence or paragraph boundaries
            # But there should be some common text
            common_text = set(first_chunk_end.split()).intersection(
                set(second_chunk_start.split())
            )
            assert common_text

    @pytest.mark.asyncio
    async def test_extract_metadata_markdown(self, document_processor, sample_files):
        """Test extracting metadata from a markdown file."""
        # Process the markdown file
        result = await document_processor.process_file(sample_files["markdown"])

        # Verify basic metadata
        assert "metadata" in result
        assert "path" in result["metadata"]
        assert "format" in result["metadata"]
        assert result["metadata"]["format"] == "md"

        # Markdown-specific metadata that might be extracted
        if "headings" in result["metadata"]:
            assert isinstance(result["metadata"]["headings"], list)
            assert "Sample Document" in result["metadata"]["headings"]
            assert "Introduction" in result["metadata"]["headings"]
            assert "Content Section" in result["metadata"]["headings"]
            assert "Conclusion" in result["metadata"]["headings"]

        # Check for section count
        if "section_count" in result["metadata"]:
            assert result["metadata"]["section_count"] >= 3  # At least 3 sections

    @pytest.mark.asyncio
    async def test_normalize_text(self, document_processor):
        """Test text normalization."""
        # Text with various formatting issues
        messy_text = """
        This   has  multiple   spaces.
        
        And multiple
        
        line breaks. Also some\ttabs.
        
        UPPERCASE TEXT and MiXeD CaSe.
        
        Special chars: &amp; &lt; &gt; &quot;
        
        URLs: http://example.com and https://test.org
        
        Email: test@example.com
        """

        # Normalize the text
        normalized = document_processor.normalize_text(messy_text)

        # Verify normalization
        assert "  " not in normalized  # Multiple spaces should be condensed
        assert "&amp;" not in normalized  # HTML entities should be decoded
        assert "&lt;" not in normalized
        assert "&gt;" not in normalized
        assert "&quot;" not in normalized

        # The normalized text should be more compact
        assert len(normalized) < len(messy_text)

    @pytest.mark.asyncio
    async def test_process_unsupported_format(self, document_processor):
        """Test processing a file with unsupported format."""
        # Create a temporary unsupported file
        with tempfile.NamedTemporaryFile(suffix=".xyz", delete=False) as temp_file:
            temp_file.write(b"This is an unsupported file format.")
            temp_path = Path(temp_file.name)

        try:
            result = await document_processor.process_file(temp_path)

            # If processing doesn't raise an exception, verify it handled gracefully
            assert isinstance(result, dict)
            assert "error" in result or result["content"] == ""
        except ValueError as e:
            # It's also acceptable to raise an exception
            assert "unsupported" in str(e).lower() or "format" in str(e).lower()
        finally:
            # Clean up
            if os.path.exists(temp_path):
                os.remove(temp_path)

    @pytest.mark.asyncio
    async def test_batch_processing(self, document_processor, sample_files):
        """Test batch processing of multiple files."""
        # Create a list of files
        files = [sample_files["markdown"], sample_files["text"], sample_files["html"]]

        # Process in batch
        results = await document_processor.process_batch(files)

        # Verify the results
        assert isinstance(results, list)
        assert len(results) == 3

        # Verify each result has the expected structure
        for result in results:
            assert isinstance(result, dict)
            assert "content" in result
            assert "metadata" in result
            assert "chunks" in result

    @pytest.mark.asyncio
    async def test_process_empty_file(self, document_processor):
        """Test processing an empty file."""
        # Create a temporary empty file
        with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as temp_file:
            temp_path = Path(temp_file.name)

        try:
            # Process the empty file
            result = await document_processor.process_file(temp_path)

            # Verify the result
            assert isinstance(result, dict)
            assert "content" in result
            assert "metadata" in result
            assert "chunks" in result

            # Content should be empty
            assert result["content"] == ""

            # There should be no chunks
            assert len(result["chunks"]) == 0
        finally:
            # Clean up
            if os.path.exists(temp_path):
                os.remove(temp_path)

    @pytest.mark.asyncio
    async def test_process_nonexistent_file(self, document_processor):
        """Test processing a file that doesn't exist."""
        # Create a path to a nonexistent file
        nonexistent_file = Path("/tmp/nonexistent_file_for_testing.txt")

        # Process the nonexistent file
        try:
            result = await document_processor.process_file(nonexistent_file)

            # If processing doesn't raise an exception, verify it handled gracefully
            assert isinstance(result, dict)
            assert "error" in result
            assert (
                "not found" in result["error"].lower()
                or "not exist" in result["error"].lower()
            )
        except FileNotFoundError:
            # It's also acceptable to raise an exception
            pass

    @pytest.mark.asyncio
    async def test_max_file_size_handling(self, document_processor):
        """Test handling of files exceeding maximum size."""
        # Set a small max file size for testing
        document_processor.max_file_size = 100  # 100 bytes

        # Create a temporary file larger than the max size
        with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as temp_file:
            temp_file.write(b"X" * 1000)  # 1000 bytes, exceeds max
            temp_path = Path(temp_file.name)

        try:
            result = await document_processor.process_file(temp_path)

            # If processing doesn't raise an exception, verify it handled gracefully
            assert isinstance(result, dict)
            assert "error" in result
            assert (
                "size" in result["error"].lower()
                or "too large" in result["error"].lower()
            )
        except ValueError as e:
            # It's also acceptable to raise an exception
            assert "size" in str(e).lower() or "too large" in str(e).lower()
        finally:
            # Clean up
            if os.path.exists(temp_path):
                os.remove(temp_path)

    @pytest.mark.asyncio
    async def test_custom_chunking_strategy(
        self, document_processor, sample_markdown_content
    ):
        """Test using a custom chunking strategy."""

        # Create a custom chunking function
        def custom_chunker(text, chunk_size=500, chunk_overlap=50):
            # Simple chunking by paragraphs
            paragraphs = text.split("\n\n")
            chunks = []
            current_chunk = ""

            for para in paragraphs:
                if len(current_chunk) + len(para) <= chunk_size:
                    current_chunk += para + "\n\n"
                else:
                    chunks.append(current_chunk)
                    current_chunk = para + "\n\n"

            if current_chunk:
                chunks.append(current_chunk)

            return chunks

        # Set the custom chunker
        document_processor.chunk_text = custom_chunker

        # Create a temporary markdown file
        with tempfile.NamedTemporaryFile(suffix=".md", delete=False) as temp_file:
            temp_file.write(sample_markdown_content.encode())
            temp_path = Path(temp_file.name)

        try:
            # Process the file with custom chunking
            result = await document_processor.process_file(temp_path)

            # Verify the chunks
            assert isinstance(result["chunks"], list)

            # Verify paragraphs are preserved in chunks
            for chunk in result["chunks"]:
                paragraphs = chunk.split("\n\n")
                assert len(paragraphs) >= 1
        finally:
            # Clean up
            if os.path.exists(temp_path):
                os.remove(temp_path)

    @pytest.mark.asyncio
    async def test_concurrent_processing(self, document_processor, sample_files):
        """Test that multiple concurrent processing operations work correctly."""
        # Create tasks for concurrent processing
        files = [sample_files["markdown"], sample_files["text"], sample_files["html"]]

        tasks = [
            asyncio.create_task(document_processor.process_file(file)) for file in files
        ]

        # Wait for all tasks to complete
        results = await asyncio.gather(*tasks)

        # Verify all operations succeeded
        assert len(results) == 3
        assert all(isinstance(result, dict) for result in results)
        assert all("content" in result for result in results)
        assert all("metadata" in result for result in results)
        assert all("chunks" in result for result in results)
