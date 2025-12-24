import pytest

from src.main import Normalizer


class TestNormalizer:
    """Test suite for the Normalizer class."""

    @pytest.fixture
    def normalizer(self):
        """Create a Normalizer instance for testing."""
        return Normalizer()

    def test_normalize_removes_whitespace(self, normalizer):
        """Test that normalize removes extra whitespace."""
        text = "  This has extra   spaces   "
        result = normalizer.normalize(text)
        assert result == "This has extra   spaces"

    def test_normalize_with_empty_string(self, normalizer):
        """Test normalize with empty string."""
        result = normalizer.normalize("")
        assert result == ""

    def test_normalize_with_only_whitespace(self, normalizer):
        """Test normalize with string containing only whitespace."""
        result = normalizer.normalize("   ")
        assert result == ""

    def test_normalize_with_newlines(self, normalizer):
        """Test normalize with text containing newlines."""
        text = "Line 1\n  Line 2  \nLine 3"
        result = normalizer.normalize(text)
        assert result == "Line 1\n  Line 2  \nLine 3"

    def test_normalize_with_tabs(self, normalizer):
        """Test normalize with text containing tabs."""
        text = "\tIndented\t\tText\t"
        result = normalizer.normalize(text)
        assert result == "Indented\t\tText"

    def test_normalize_preserves_internal_whitespace(self, normalizer):
        """Test that normalize preserves internal whitespace."""
        text = "  Preserve   internal   spaces   "
        result = normalizer.normalize(text)
        assert result == "Preserve   internal   spaces"
        assert "   " in result  # Internal multiple spaces are preserved
