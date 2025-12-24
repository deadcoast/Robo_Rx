import pytest

from src.main import MetaExtractor


class TestMetaExtractor:
    """Test suite for the MetaExtractor class."""

    @pytest.fixture
    def extractor(self):
        """Create a MetaExtractor instance for testing."""
        return MetaExtractor()

    def test_extract_basic(self, extractor):
        """Test basic metadata extraction from content."""
        content = "# Sample Document\nThis is sample content."
        result = extractor.extract(content)

        assert isinstance(result, dict)
        assert "tags" in result
        assert "created" in result
        assert "modified" in result

        assert isinstance(result["tags"], list)
        assert result["created"] is None
        assert result["modified"] is None

    def test_extract_empty_content(self, extractor):
        """Test metadata extraction from empty content."""
        result = extractor.extract("")

        assert isinstance(result, dict)
        assert "tags" in result
        assert "created" in result
        assert "modified" in result

        assert isinstance(result["tags"], list)
        assert len(result["tags"]) == 0
        assert result["created"] is None
        assert result["modified"] is None

    def test_extract_null_content(self, extractor):
        """Test metadata extraction with None content."""
        # Should handle None gracefully or raise a specific error
        try:
            result = extractor.extract(None)
            # If it doesn't raise an exception, it should return default values
            assert isinstance(result, dict)
            assert "tags" in result
        except (TypeError, AttributeError):
            # This is also acceptable behavior for None input
            pass

    def test_extract_with_whitespace(self, extractor):
        """Test metadata extraction from content with excess whitespace."""
        content = "  \n\n  # Document with Whitespace  \n\n  Content here.  \n\n  "
        result = extractor.extract(content)

        assert isinstance(result, dict)
        assert "tags" in result
        assert isinstance(result["tags"], list)

    def test_extract_with_special_characters(self, extractor):
        """Test metadata extraction from content with special characters."""
        content = "# Special Characters: !@#$%^&*()_+{}|:<>?~"
        result = extractor.extract(content)

        assert isinstance(result, dict)
        assert "tags" in result
        assert isinstance(result["tags"], list)

    def test_extract_consistency(self, extractor):
        """Test consistency of metadata extraction."""
        content = "# Test Document\nThis is a test."

        result1 = extractor.extract(content)
        result2 = extractor.extract(content)

        # Extracting metadata from the same content twice should produce the same result
        assert result1 == result2
