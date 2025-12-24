from unittest.mock import Mock

import pytest

from src.main import MDParser


class TestMDParser:
    """Test suite for the MDParser class."""

    @pytest.fixture
    def mock_config(self):
        """Create a mock configuration for testing."""
        return Mock()

    @pytest.fixture
    def parser(self, mock_config):
        """Create an MDParser instance with mocked config for testing."""
        return MDParser(mock_config)

    def test_init(self, parser, mock_config):
        """Test initialization of MDParser."""
        assert parser.config is mock_config

    def test_parse_basic_content(self, parser):
        """Test parsing basic markdown content."""
        content = "# Heading\nThis is some text."
        result = parser.parse(content)

        assert isinstance(result, dict)
        assert "content" in result
        assert "metadata" in result
        assert result["content"] == content
        assert isinstance(result["metadata"], dict)

    def test_parse_empty_content(self, parser):
        """Test parsing empty content."""
        result = parser.parse("")

        assert isinstance(result, dict)
        assert "content" in result
        assert "metadata" in result
        assert result["content"] == ""
        assert isinstance(result["metadata"], dict)

    def test_parse_complex_markdown(self, parser):
        """Test parsing complex markdown content."""
        content = """# Complex Document
        
## Introduction
This is a *complex* document with **bold text** and `code`.

### Lists
- Item 1
- Item 2
  - Subitem 2.1
  - Subitem 2.2
        
```python
def example():
    return "Code block"
```

> This is a blockquote.
"""
        result = parser.parse(content)

        assert isinstance(result, dict)
        assert "content" in result
        assert "metadata" in result
        assert result["content"] == content

    def test_parse_consistency(self, parser):
        """Test parsing consistency with the same content."""
        content = "# Test Document\nContent here."

        result1 = parser.parse(content)
        result2 = parser.parse(content)

        # Parsing the same content twice should produce the same result
        assert result1["content"] == result2["content"]
        assert result1["metadata"] == result2["metadata"]

    def test_config_usage(self, parser):
        """Test that the parser is properly connected to config."""
        # This is a minimal test since the actual implementation doesn't use config yet
        assert hasattr(parser, "config")

        # Real implementation would test config parameter effects on parsing behavior
