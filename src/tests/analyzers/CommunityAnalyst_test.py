from graphlib import Graph
from unittest.mock import Mock, patch

import pytest

from src.analyzers.AnalyzerCore import ResultAnalyzer
from src.analyzers.CommunityAnalyst import CommunityAnalyst


class TestCommunityAnalyst:
    """Test suite for the CommunityAnalyst class."""

    @pytest.fixture
    def mock_graph(self):
        """Create a mock graph for testing."""
        return Mock(spec=Graph)

    @pytest.fixture
    def mock_analyzer(self):
        """Create a mock ResultAnalyzer for testing."""
        return Mock(spec=ResultAnalyzer)

    @pytest.fixture
    def community_analyst(self, mock_graph):
        """Create a CommunityAnalyst instance for testing."""
        return CommunityAnalyst(mock_graph)

    def test_init(self, community_analyst, mock_graph):
        """Test initialization of CommunityAnalyst."""
        assert community_analyst.graph == mock_graph
        assert community_analyst.analyzer is None
        assert community_analyst.report is None
        assert community_analyst.metrics is None
        assert community_analyst.status is None
        assert community_analyst.error_log is None
        assert community_analyst.warning_log is None
        assert community_analyst.error_registry is None

    def test_analyze(self, community_analyst, mock_analyzer):
        """Test the analyze method."""
        # Setup mock
        expected_result = {"communities": ["community1", "community2"]}
        mock_analyzer.analyze.return_value = expected_result

        # Call method
        result = community_analyst.analyze(mock_analyzer)

        # Assertions
        assert result == expected_result
        mock_analyzer.analyze.assert_called_once()

    def test_analyze_empty_graph(self, mock_analyzer):
        """Test analyze method with an empty graph."""
        # Create an empty graph
        empty_graph = Mock(spec=Graph)
        community_analyst = CommunityAnalyst(empty_graph)

        # Setup mock
        mock_analyzer.analyze.return_value = None

        # Call method
        result = community_analyst.analyze(mock_analyzer)

        # Assertions
        assert result is None
        mock_analyzer.analyze.assert_called_once()

    def test_analyze_error_handling(self, community_analyst, mock_analyzer):
        """Test error handling in analyze method."""
        # Setup mock to raise an exception
        mock_analyzer.analyze.side_effect = ValueError("Analysis error")

        # Call method and assert it handles the exception
        with pytest.raises(ValueError):
            community_analyst.analyze(mock_analyzer)

    @patch("src.analyzers.CommunityAnalyst.CommunityAnalyst.analyze")
    def test_detect_communities(self, mock_analyze, community_analyst, mock_analyzer):
        """Test the detect_communities method."""
        # Setup
        mock_analyze.return_value = {"communities": [1, 2, 3]}

        # Call
        result = community_analyst.detect_communities(mock_analyzer)

        # Assert
        assert result == [1, 2, 3]
        mock_analyze.assert_called_once_with(mock_analyzer)

    def test_get_community_metrics(self, community_analyst):
        """Test the get_community_metrics method."""
        # Setup
        community_id = "community1"
        expected_metrics = {"size": 10, "density": 0.75}
        community_analyst.metrics = {community_id: expected_metrics}

        # Call
        result = community_analyst.get_community_metrics(community_id)

        # Assert
        assert result == expected_metrics

    def test_get_community_metrics_nonexistent(self, community_analyst):
        """Test get_community_metrics with a nonexistent community."""
        # Setup
        community_analyst.metrics = {"community1": {"size": 10}}

        # Call & Assert
        with pytest.raises(KeyError):
            community_analyst.get_community_metrics("nonexistent")

    def test_get_all_communities(self, community_analyst):
        """Test the get_all_communities method."""
        # Setup
        expected_communities = ["community1", "community2"]
        community_analyst.communities = expected_communities

        # Call
        result = community_analyst.get_all_communities()

        # Assert
        assert result == expected_communities
