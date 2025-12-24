"""
A class for analyzing communities in a graph.

"""

from graphlib import Graph
from typing import Any, Dict, Optional

from src.analyzers.AnalyzerCore import ResultAnalyzer


class CommunityAnalyst:
    """
    A class for analyzing communities in a graph.

    """

    def __init__(self, graph: Graph):
        """
        Initialize the CommunityAnalyst with a graph.

        Args:
            graph (Graph): The graph to analyze.
        """
        self.graph = graph
        self.analyzer = None
        self.report = None
        self.metrics = None
        self.status = None
        self.error_log = None
        self.warning_log = None
        self.error_registry = None
        self.report_metadata = None
        self.report_status = None
        self.report_errors = None
        self.report_warnings = None
        self.report_metrics = None
        self.logger = None
        self.error_log = None
        self.warning_log = None
        self.error_registry = None

    def analyze(self, analyzer: ResultAnalyzer) -> Optional[Dict[str, Any]]:
        """
        Analyze the graph for communities.

        Args:
            analyzer (ResultAnalyzer): The analyzer to use for analysis.

        Returns:
            Optional[Dict[str, Any]]: A dictionary containing the analysis results.
        """
        self.analyzer = analyzer
        self.report = self.analyzer.analyze(self.graph)
        return self._extract_metrics()

    def detect(self, analyzer: ResultAnalyzer) -> Optional[Dict[str, Any]]:
        """
        Detect communities in the graph.

        Args:
            analyzer (ResultAnalyzer): The analyzer to use for detection.

        Returns:
            Optional[Dict[str, Any]]: A dictionary containing the detected communities.
        """
        self.analyzer = analyzer
        self.report = self.analyzer.detect(self.graph)
        return self._extract_metrics()

    def rank(self, analyzer: ResultAnalyzer) -> Optional[Dict[str, Any]]:
        """
        Rank the communities in the graph.

        Args:
            analyzer (ResultAnalyzer): The analyzer to use for ranking.

        Returns:
            Optional[Dict[str, Any]]: A dictionary containing the ranked communities.
        """
        self.analyzer = analyzer
        self.report = self.analyzer.rank(self.graph)
        self.metrics = self.analyzer.metrics(self.graph)
        return self._extract_metrics()

    def visualize(self, analyzer: ResultAnalyzer) -> Optional[Dict[str, Any]]:
        """
        Visualize the communities in the graph.

        Args:
            analyzer (ResultAnalyzer): The analyzer to use for visualization.

        Returns:
            Optional[Dict[str, Any]]: A dictionary containing the visualization results.
        """
        self.analyzer = analyzer
        self.report = self.analyzer.visualize(self.graph)
        self.metrics = self.analyzer.metrics(self.graph)
        return self._extract_metrics()

    def export(self, analyzer: ResultAnalyzer) -> Optional[Dict[str, Any]]:
        """
        Export the communities in the graph.

        Args:
            analyzer (ResultAnalyzer): The analyzer to use for export.

        Returns:
            Optional[Dict[str, Any]]: A dictionary containing the exported communities.
        """
        self.analyzer = analyzer
        self.report = self.analyzer.export(self.graph)
        self.metrics = self.analyzer.metrics(self.graph)
        return self._extract_metrics()

    def save(self, analyzer: ResultAnalyzer) -> Optional[Dict[str, Any]]:
        """
        Save the communities in the graph.

        Args:
            analyzer (ResultAnalyzer): The analyzer to use for saving.

        Returns:
            Optional[Dict[str, Any]]: A dictionary containing the saved communities.
        """
        self.analyzer = analyzer
        self.report = self.analyzer.save(self.graph)
        self.metrics = self.analyzer.metrics(self.graph)
        return self._extract_metrics()

    def load(self, analyzer: ResultAnalyzer) -> Optional[Dict[str, Any]]:
        """
        Load the communities in the graph.

        Args:
            analyzer (ResultAnalyzer): The analyzer to use for loading.

        Returns:
            Optional[Dict[str, Any]]: A dictionary containing the loaded communities.
        """
        self.analyzer = analyzer
        self.report = self.analyzer.load(self.graph)
        self.metrics = self.analyzer.metrics(self.graph)
        return self._extract_metrics()

    def delete(self, analyzer: ResultAnalyzer) -> Optional[Dict[str, Any]]:
        """
        Delete the communities in the graph.

        Args:
            analyzer (ResultAnalyzer): The analyzer to use for deletion.

        Returns:
            Optional[Dict[str, Any]]: A dictionary containing the deleted communities.
        """
        self.analyzer = analyzer
        self.report = self.analyzer.delete(self.graph)
        self.metrics = self.analyzer.metrics(self.graph)
        return self._extract_metrics()

    def update(self, analyzer: ResultAnalyzer) -> Optional[Dict[str, Any]]:
        """
        Update the communities in the graph.

        Args:
            analyzer (ResultAnalyzer): The analyzer to use for updating.

        Returns:
            Optional[Dict[str, Any]]: A dictionary containing the updated communities.
        """
        self.analyzer = analyzer
        self.report = self.analyzer.update(self.graph)
        self.metrics = self.analyzer.metrics(self.graph)
        return self._extract_metrics()

    def get(self, analyzer: ResultAnalyzer) -> Optional[Dict[str, Any]]:
        """
        Get the communities in the graph.

        Args:
            analyzer (ResultAnalyzer): The analyzer to use for getting.

        Returns:
            Optional[Dict[str, Any]]: A dictionary containing the communities.
        """
        self.analyzer = analyzer
        self.report = self.analyzer.get(self.graph)
        self.metrics = self.analyzer.metrics(self.graph)
        return self._extract_metrics()

    def set(self, analyzer: ResultAnalyzer) -> Optional[Dict[str, Any]]:
        """
        Set the communities in the graph.

        Args:
            analyzer (ResultAnalyzer): The analyzer to use for setting.

        Returns:
            Optional[Dict[str, Any]]: A dictionary containing the set communities.
        """
        self.analyzer = analyzer
        self.report = self.analyzer.set(self.graph)
        self.metrics = self.analyzer.metrics(self.graph)
        return self._extract_metrics()

    def _extract_metrics(self):
        """
        Extracts metrics and logs from the analyzer.
        """
        self.metrics = self.analyzer.metrics(self.graph)
        self.status = self.analyzer.status
        self.error_log = self.analyzer.error_log
        self.warning_log = self.analyzer.warning_log
        self.error_registry = self.analyzer.error_registry
        self.report_metadata = self.analyzer.report_metadata
        self.report_status = self.analyzer.report_status
        self.report_errors = self.analyzer.report_errors
        self.report_warnings = self.analyzer.report_warnings
        self.report_metrics = self.analyzer.report_metrics
        return self.report
