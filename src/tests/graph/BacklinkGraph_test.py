import networkx as nx
import pytest

from src.main import BacklinkGraph


class TestBacklinkGraph:
    """Test suite for the BacklinkGraph class."""

    @pytest.fixture
    def mock_config(self):
        """Create a mock configuration for testing."""
        return {
            "graph": {"edge_weight_threshold": 0.5, "max_nodes": 1000, "directed": True}
        }

    @pytest.fixture
    def backlink_graph(self, mock_config):
        """Create a BacklinkGraph instance for testing."""
        return BacklinkGraph(mock_config)

    def test_init(self, backlink_graph, mock_config):
        """Test initialization of BacklinkGraph."""
        # Verify the graph was initialized as a networkx DiGraph
        assert isinstance(backlink_graph.graph, nx.DiGraph)
        assert len(backlink_graph.graph.nodes) == 0
        assert len(backlink_graph.graph.edges) == 0

        # Verify config was stored
        assert backlink_graph.config == mock_config

    def test_graph_is_directed(self, backlink_graph):
        """Test that the graph is directed."""
        assert nx.is_directed(backlink_graph.graph)

    def test_node_addition(self, backlink_graph):
        """Test adding nodes to the graph."""
        # Add nodes
        backlink_graph.graph.add_node("note1", title="Note 1", created="2025-03-31")
        backlink_graph.graph.add_node("note2", title="Note 2", created="2025-03-31")

        # Verify nodes were added with attributes
        assert "note1" in backlink_graph.graph.nodes
        assert "note2" in backlink_graph.graph.nodes
        assert backlink_graph.graph.nodes["note1"]["title"] == "Note 1"
        assert backlink_graph.graph.nodes["note2"]["title"] == "Note 2"

    def test_edge_addition(self, backlink_graph):
        """Test adding edges (backlinks) to the graph."""
        # Add nodes
        backlink_graph.graph.add_node("note1")
        backlink_graph.graph.add_node("note2")

        # Add edge (backlink)
        backlink_graph.graph.add_edge("note1", "note2", weight=0.75, type="reference")

        # Verify edge was added with attributes
        assert backlink_graph.graph.has_edge("note1", "note2")
        assert not backlink_graph.graph.has_edge("note2", "note1")  # Directed graph
        assert backlink_graph.graph.edges["note1", "note2"]["weight"] == 0.75
        assert backlink_graph.graph.edges["note1", "note2"]["type"] == "reference"

    def test_bidirectional_links(self, backlink_graph):
        """Test adding bidirectional links between nodes."""
        # Add nodes
        backlink_graph.graph.add_node("note1")
        backlink_graph.graph.add_node("note2")

        # Add edges in both directions
        backlink_graph.graph.add_edge("note1", "note2", weight=0.6)
        backlink_graph.graph.add_edge("note2", "note1", weight=0.4)

        # Verify both edges exist
        assert backlink_graph.graph.has_edge("note1", "note2")
        assert backlink_graph.graph.has_edge("note2", "note1")

        # Verify weights are different
        assert backlink_graph.graph.edges["note1", "note2"]["weight"] == 0.6
        assert backlink_graph.graph.edges["note2", "note1"]["weight"] == 0.4

    def test_multiple_edge_attributes(self, backlink_graph):
        """Test adding multiple attributes to edges."""
        # Add nodes
        backlink_graph.graph.add_node("note1")
        backlink_graph.graph.add_node("note2")

        # Add edge with multiple attributes
        attributes = {
            "weight": 0.8,
            "type": "citation",
            "count": 3,
            "first_referenced": "2025-03-30",
            "last_referenced": "2025-03-31",
            "context": "In the introduction section",
        }
        backlink_graph.graph.add_edge("note1", "note2", **attributes)

        # Verify all attributes were added
        edge_attrs = backlink_graph.graph.edges["note1", "note2"]
        for key, value in attributes.items():
            assert edge_attrs[key] == value

    def test_graph_algorithms(self, backlink_graph):
        """Test using networkx algorithms with the backlink graph."""
        # Create a more complex graph
        nodes = ["note1", "note2", "note3", "note4", "note5"]
        for node in nodes:
            backlink_graph.graph.add_node(node)

        # Add edges
        edges = [
            ("note1", "note2", {"weight": 0.7}),
            ("note2", "note3", {"weight": 0.6}),
            ("note3", "note4", {"weight": 0.9}),
            ("note4", "note5", {"weight": 0.5}),
            ("note5", "note1", {"weight": 0.8}),  # Creates a cycle
        ]
        backlink_graph.graph.add_edges_from(edges)

        # Test path finding
        path = nx.shortest_path(backlink_graph.graph, source="note1", target="note4")
        assert path == ["note1", "note2", "note3", "note4"]

        # Test path length (considering weights)
        path_length = nx.shortest_path_length(
            backlink_graph.graph, source="note1", target="note4", weight="weight"
        )
        # With the given weights, the path length should be the sum of weights
        expected_length = 0.7 + 0.6 + 0.9
        assert path_length == expected_length

        # Test cycle finding
        cycles = list(nx.simple_cycles(backlink_graph.graph))
        assert len(cycles) == 1
        assert set(cycles[0]) == set(nodes)  # One cycle containing all nodes

        # Test centrality measures
        in_centrality = nx.in_degree_centrality(backlink_graph.graph)
        out_centrality = nx.out_degree_centrality(backlink_graph.graph)

        # Each node should have centrality values
        for node in nodes:
            assert node in in_centrality
            assert node in out_centrality

    def test_subgraph_extraction(self, backlink_graph):
        """Test extracting subgraphs from the main graph."""
        # Create a graph with several nodes
        for i in range(10):
            backlink_graph.graph.add_node(f"note{i}")

        # Add edges to create communities
        # Community 1: notes 0-3
        for i in range(3):
            for j in range(i + 1, 4):
                backlink_graph.graph.add_edge(f"note{i}", f"note{j}")

        # Community 2: notes 4-7
        for i in range(4, 7):
            for j in range(i + 1, 8):
                backlink_graph.graph.add_edge(f"note{i}", f"note{j}")

        # Bridge node connecting both communities
        backlink_graph.graph.add_edge("note3", "note4")

        # Isolated nodes: 8, 9

        # Extract subgraph for community 1
        community1_nodes = [f"note{i}" for i in range(4)]
        subgraph1 = backlink_graph.graph.subgraph(community1_nodes)

        # Verify subgraph properties
        assert len(subgraph1.nodes) == 4
        assert all(node in subgraph1.nodes for node in community1_nodes)

        # The subgraph should have all internal edges
        for i in range(3):
            for j in range(i + 1, 4):
                assert subgraph1.has_edge(f"note{i}", f"note{j}")

        # But not edges to other communities
        assert "note4" not in subgraph1.nodes

    def test_graph_metrics(self, backlink_graph):
        """Test computing various graph metrics."""
        # Create a simple graph
        nodes = ["note1", "note2", "note3", "note4"]
        for node in nodes:
            backlink_graph.graph.add_node(node)

        backlink_graph.graph.add_edge("note1", "note2")
        backlink_graph.graph.add_edge("note2", "note3")
        backlink_graph.graph.add_edge("note3", "note4")
        backlink_graph.graph.add_edge("note4", "note1")  # Creates a cycle

        # Calculate metrics

        # Density
        density = nx.density(backlink_graph.graph)
        expected_density = 4 / (4 * 3)  # n_edges / (n_nodes * (n_nodes - 1))
        assert density == expected_density

        # Average clustering coefficient
        # In a cycle, clustering coefficient is 0 for all nodes
        avg_clustering = nx.average_clustering(backlink_graph.graph)
        assert avg_clustering == 0

        # Add diagonal edges to create triangles
        backlink_graph.graph.add_edge("note1", "note3")
        backlink_graph.graph.add_edge("note2", "note4")

        # Recalculate clustering with triangles
        avg_clustering_with_triangles = nx.average_clustering(backlink_graph.graph)
        assert avg_clustering_with_triangles > 0

        # Calculate strongly connected components
        sccs = list(nx.strongly_connected_components(backlink_graph.graph))
        assert len(sccs) == 1  # One strongly connected component
        assert set(nodes) == set.union(*sccs)  # All nodes in the SCC

    def test_empty_graph(self):
        """Test operations on an empty graph."""
        empty_config = {"graph": {"directed": True}}
        empty_graph = BacklinkGraph(empty_config)

        # Verify empty graph properties
        assert len(empty_graph.graph.nodes) == 0
        assert len(empty_graph.graph.edges) == 0

        # Test operations that should work on empty graphs
        assert nx.density(empty_graph.graph) == 0

        # Test operations that may raise exceptions on empty graphs
        try:
            components = list(nx.strongly_connected_components(empty_graph.graph))
            assert not components
        except nx.NetworkXError:
            # Some algorithms might raise exceptions on empty graphs
            pass
