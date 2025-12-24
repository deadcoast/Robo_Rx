import pytest

from src.main import SummaryEngine


class TestSummaryEngine:
    """Test suite for the SummaryEngine class."""

    @pytest.fixture
    def summary_engine(self):
        """Create a SummaryEngine instance for testing."""
        return SummaryEngine()

    def test_summarize_short_content(self, summary_engine):
        """Test summarizing content that is already short (less than 100 chars)."""
        short_content = "This is a short content that doesn't need summarization."
        summary = summary_engine.summarize(short_content)

        # For short content, should return the content as is
        assert summary == short_content

    def test_summarize_long_content(self, summary_engine):
        """Test summarizing content that is longer than 100 chars."""
        # Create a string longer than 100 characters
        long_content = "This is a very long content that should be summarized. " * 5
        assert len(long_content) > 100

        summary = summary_engine.summarize(long_content)

        # Summary should be truncated with ellipsis
        assert summary.endswith("...")

        # Summary should be the first 100 chars + "..."
        expected_summary = f"{long_content[:100]}..."
        assert summary == expected_summary

        # Summary should be shorter than the original content
        assert len(summary) < len(long_content)

    def test_summarize_empty_content(self, summary_engine):
        """Test summarizing empty content."""
        summary = summary_engine.summarize("")

        # Empty content should result in empty summary
        assert summary == ""

    def test_summarize_exactly_100_chars(self, summary_engine):
        """Test summarizing content that is exactly 100 chars."""
        # Create a string of exactly 100 characters
        content_100_chars = "x" * 100
        assert len(content_100_chars) == 100

        summary = summary_engine.summarize(content_100_chars)

        # Content of exactly 100 chars should be returned as is
        assert summary == content_100_chars
        assert not summary.endswith("...")

    def test_summarize_101_chars(self, summary_engine):
        """Test summarizing content that is 101 chars."""
        # Create a string of exactly 101 characters
        content_101_chars = "x" * 101
        assert len(content_101_chars) == 101

        summary = summary_engine.summarize(content_101_chars)

        # Content of 101 chars should be truncated to 100 + "..."
        expected_summary = f"{content_101_chars[:100]}..."
        assert summary == expected_summary

    def test_summarize_with_special_characters(self, summary_engine):
        """Test summarizing content with special characters."""
        # Create content with special characters
        content_with_special = "!@#$%^&*() " * 15
        assert len(content_with_special) > 100

        summary = summary_engine.summarize(content_with_special)

        # Summary should be truncated with ellipsis
        assert summary.endswith("...")

        # Summary should be the first 100 chars + "..."
        expected_summary = f"{content_with_special[:100]}..."
        assert summary == expected_summary

    def test_summarize_with_newlines(self, summary_engine):
        """Test summarizing content with newlines."""
        # Create content with newlines
        content_with_newlines = "Line 1\nLine 2\nLine 3\n" * 10
        assert len(content_with_newlines) > 100

        summary = summary_engine.summarize(content_with_newlines)

        # Summary should be truncated with ellipsis
        assert summary.endswith("...")

        # Summary should be the first 100 chars + "..."
        expected_summary = f"{content_with_newlines[:100]}..."
        assert summary == expected_summary

        # Newlines should be preserved in the summary
        assert "\n" in summary
