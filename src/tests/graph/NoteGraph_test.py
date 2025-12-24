import networkx as nx
import pytest

from src.main import NoteGraph


class TestNoteGraph:
    """Test suite for the NoteGraph class."""

    @pytest.fixture
    def note_graph(self):
        """Create a NoteGraph instance for testing."""
        return NoteGraph()

    def test_init(self, note_graph):
        """Test initialization of NoteGraph."""
        # Verify the graph was initialized as a networkx Graph
        assert isinstance(note_graph.graph, nx.Graph)
        assert len(note_graph.graph.nodes) == 0
        assert len(note_graph.graph.edges) == 0

        # Verify node_metadata was initialized as an empty dict
        assert isinstance(note_graph.node_metadata, dict)
        assert len(note_graph.node_metadata) == 0

    def test_graph_manipulation(self, note_graph):
        """Test basic graph manipulation using the underlying networkx Graph."""
        # Add nodes
        note_graph.graph.add_node("note1")
        note_graph.graph.add_node("note2")
        note_graph.graph.add_node("note3")

        # Verify nodes were added
        assert "note1" in note_graph.graph.nodes
        assert "note2" in note_graph.graph.nodes
        assert "note3" in note_graph.graph.nodes
        assert len(note_graph.graph.nodes) == 3

        # Add edges
        note_graph.graph.add_edge("note1", "note2")
        note_graph.graph.add_edge("note2", "note3")

        # Verify edges were added
        assert note_graph.graph.has_edge("note1", "note2")
        assert note_graph.graph.has_edge("note2", "note3")
        assert not note_graph.graph.has_edge("note1", "note3")
        assert len(note_graph.graph.edges) == 2

    def test_node_metadata(self, note_graph):
        """Test adding and accessing node metadata."""
        # Add nodes to the graph
        note_graph.graph.add_node("note1")
        note_graph.graph.add_node("note2")

        # Add metadata for the nodes
        note_graph.node_metadata["note1"] = {
            "title": "Note 1",
            "created": "2025-03-31T06:04:56-07:00",
            "tags": ["tag1", "tag2"],
        }

        note_graph.node_metadata["note2"] = {
            "title": "Note 2",
            "created": "2025-03-31T06:10:00-07:00",
            "tags": ["tag2", "tag3"],
        }

        # Verify metadata was stored correctly
        assert "note1" in note_graph.node_metadata
        assert note_graph.node_metadata["note1"]["title"] == "Note 1"
        assert note_graph.node_metadata["note1"]["tags"] == ["tag1", "tag2"]

        assert "note2" in note_graph.node_metadata
        assert note_graph.node_metadata["note2"]["title"] == "Note 2"
        assert note_graph.node_metadata["note2"]["tags"] == ["tag2", "tag3"]

    def test_metadata_without_node(self, note_graph):
        """Test adding metadata for nodes that don't exist in the graph."""
        # Add metadata for a node that doesn't exist in the graph
        note_graph.node_metadata["nonexistent_note"] = {"title": "Nonexistent Note"}

        # Verify metadata was stored despite node not existing in graph
        assert "nonexistent_note" in note_graph.node_metadata
        assert (
            note_graph.node_metadata["nonexistent_note"]["title"] == "Nonexistent Note"
        )

        # Verify node wasn't automatically added to the graph
        assert "nonexistent_note" not in note_graph.graph.nodes

    def test_graph_algorithms(self, note_graph):
        """Test using networkx algorithms with the graph."""
        # Create a simple graph
        note_graph.graph.add_nodes_from(["note1", "note2", "note3", "note4"])
        note_graph.graph.add_edges_from(
            [
                ("note1", "note2"),
                ("note2", "note3"),
                ("note3", "note4"),
                ("note4", "note1"),
            ]
        )

        # Test path finding
        path = nx.shortest_path(note_graph.graph, source="note1", target="note3")
        assert path == ["note1", "note2", "note3"]

        # Test cycle detection
        cycles = list(nx.simple_cycles(note_graph.graph.to_directed()))
        assert cycles

        # Test connected components
        components = list(nx.connected_components(note_graph.graph))
        assert len(components) == 1  # There should be one connected component

        # Test centrality
        centrality = nx.degree_centrality(note_graph.graph)
        assert set(centrality.keys()) == {"note1", "note2", "note3", "note4"}
        assert all(0 <= value <= 1 for value in centrality.values())

    def test_node_removal(self, note_graph):
        """Test removing nodes and associated metadata."""
        # Setup graph with nodes and metadata
        note_graph.graph.add_nodes_from(["note1", "note2", "note3"])
        note_graph.graph.add_edge("note1", "note2")

        note_graph.node_metadata["note1"] = {"title": "Note 1"}
        note_graph.node_metadata["note2"] = {"title": "Note 2"}
        note_graph.node_metadata["note3"] = {"title": "Note 3"}

        # Remove a node from the graph
        note_graph.graph.remove_node("note2")

        # Verify node was removed from graph
        assert "note2" not in note_graph.graph.nodes
        assert not note_graph.graph.has_edge("note1", "note2")

        # Verify metadata still exists (not automatically removed)
        assert "note2" in note_graph.node_metadata

    def test_add_node_with_attributes(self, note_graph):
        """Test adding nodes with attributes to the graph."""
        # Add nodes with attributes
        note_graph.graph.add_node("note1", weight=5, category="important")
        note_graph.graph.add_node("note2", weight=3, category="normal")

        # Verify nodes were added with attributes
        assert note_graph.graph.nodes["note1"]["weight"] == 5
        assert note_graph.graph.nodes["note1"]["category"] == "important"
        assert note_graph.graph.nodes["note2"]["weight"] == 3
        assert note_graph.graph.nodes["note2"]["category"] == "normal"

    def test_add_edge_with_attributes(self, note_graph):
        """Test adding edges with attributes to the graph."""
        # Add nodes
        note_graph.graph.add_node("note1")
        note_graph.graph.add_node("note2")

        # Add edge with attributes
        note_graph.graph.add_edge("note1", "note2", weight=0.8, type="reference")

        # Verify edge was added with attributes
        assert note_graph.graph.edges["note1", "note2"]["weight"] == 0.8
        assert note_graph.graph.edges["note1", "note2"]["type"] == "reference"

    def test_graph_properties(self, note_graph):
        """Test computing graph properties."""
        # Create a graph
        note_graph.graph.add_nodes_from(["note1", "note2", "note3", "note4", "note5"])
        note_graph.graph.add_edges_from(
            [
                ("note1", "note2"),
                ("note2", "note3"),
                ("note3", "note4"),
                ("note1", "note5"),
            ]
        )

        # Compute various properties
        assert note_graph.graph.number_of_nodes() == 5
        assert note_graph.graph.number_of_edges() == 4
        assert nx.is_connected(note_graph.graph)
        assert not nx.is_directed(note_graph.graph)

        # Density
        density = nx.density(note_graph.graph)
        assert 0 <= density <= 1

        # Diameter (longest shortest path)
        diameter = nx.diameter(note_graph.graph)
        assert diameter == 3  # Longest path: note5 -> note1 -> note2 -> note3 -> note4
