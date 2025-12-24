import json
from unittest.mock import MagicMock, patch

import networkx as nx
import pytest

from src.main import KnowledgeGraph


class TestKnowledgeGraph:
    """Test suite for the KnowledgeGraph class."""

    @pytest.fixture
    def mock_config(self):
        """Create a mock configuration for testing."""
        return {
            "knowledge_graph": {
                "storage_path": "/tmp/test_knowledge_graph",
                "max_nodes": 10000,
                "similarity_threshold": 0.75,
                "enable_caching": True,
            }
        }

    @pytest.fixture
    def knowledge_graph(self, mock_config):
        """Create a KnowledgeGraph instance for testing."""
        return KnowledgeGraph(mock_config)

    @pytest.fixture
    def sample_entities(self):
        """Create sample entities for testing."""
        return [
            {
                "id": "entity1",
                "name": "Entity 1",
                "type": "concept",
                "attributes": {"key1": "value1"},
            },
            {
                "id": "entity2",
                "name": "Entity 2",
                "type": "person",
                "attributes": {"key2": "value2"},
            },
            {
                "id": "entity3",
                "name": "Entity 3",
                "type": "location",
                "attributes": {"key3": "value3"},
            },
            {
                "id": "entity4",
                "name": "Entity 4",
                "type": "concept",
                "attributes": {"key4": "value4"},
            },
            {
                "id": "entity5",
                "name": "Entity 5",
                "type": "organization",
                "attributes": {"key5": "value5"},
            },
        ]

    @pytest.fixture
    def sample_relationships(self):
        """Create sample relationships for testing."""
        return [
            {
                "source": "entity1",
                "target": "entity2",
                "type": "related_to",
                "weight": 0.8,
                "attributes": {"since": "2023"},
            },
            {
                "source": "entity2",
                "target": "entity3",
                "type": "located_in",
                "weight": 0.9,
                "attributes": {"verified": True},
            },
            {
                "source": "entity3",
                "target": "entity4",
                "type": "contains",
                "weight": 0.7,
                "attributes": {"count": 5},
            },
            {
                "source": "entity4",
                "target": "entity5",
                "type": "part_of",
                "weight": 0.6,
                "attributes": {"role": "member"},
            },
            {
                "source": "entity5",
                "target": "entity1",
                "type": "created",
                "weight": 0.75,
                "attributes": {"date": "2024-01-01"},
            },
        ]

    def test_init(self, knowledge_graph, mock_config):
        """Test initialization of KnowledgeGraph."""
        assert knowledge_graph.config == mock_config
        assert hasattr(knowledge_graph, "graph")
        assert isinstance(knowledge_graph.graph, nx.DiGraph)
        assert len(knowledge_graph.graph.nodes) == 0
        assert len(knowledge_graph.graph.edges) == 0

    def test_add_entity(self, knowledge_graph, sample_entities):
        """Test adding entities to the graph."""
        # Add a single entity
        entity = sample_entities[0]
        result = knowledge_graph.add_entity(entity)

        # Verify the entity was added successfully
        assert result is True
        assert entity["id"] in knowledge_graph.graph.nodes
        assert knowledge_graph.graph.nodes[entity["id"]]["name"] == entity["name"]
        assert knowledge_graph.graph.nodes[entity["id"]]["type"] == entity["type"]
        assert (
            knowledge_graph.graph.nodes[entity["id"]]["attributes"]
            == entity["attributes"]
        )

        results = [
            knowledge_graph.add_entity(entity) for entity in sample_entities[1:]
        ]
        # Verify all entities were added successfully
        assert all(results)
        assert len(knowledge_graph.graph.nodes) == len(sample_entities)

        # Verify all entity IDs are in the graph
        for entity in sample_entities:
            assert entity["id"] in knowledge_graph.graph.nodes

    def test_add_duplicate_entity(self, knowledge_graph, sample_entities):
        """Test adding an entity that already exists."""
        # Add an entity
        entity = sample_entities[0]
        knowledge_graph.add_entity(entity)

        # Try to add the same entity again
        result = knowledge_graph.add_entity(entity)

        # Implementation could either return False or update the entity
        # Both are valid, so check both possibilities
        if result is not False:
            assert knowledge_graph.graph.nodes[entity["id"]]["name"] == entity["name"]
        # If False, verify no changes were made
        assert len(knowledge_graph.graph.nodes) == 1

    def test_add_relationship(
        self, knowledge_graph, sample_entities, sample_relationships
    ):
        """Test adding relationships between entities."""
        # First, add all entities
        for entity in sample_entities:
            knowledge_graph.add_entity(entity)

        # Add a single relationship
        relationship = sample_relationships[0]
        result = knowledge_graph.add_relationship(relationship)

        # Verify the relationship was added successfully
        assert result is True
        assert knowledge_graph.graph.has_edge(
            relationship["source"], relationship["target"]
        )
        edge_data = knowledge_graph.graph.edges[
            relationship["source"], relationship["target"]
        ]
        assert edge_data["type"] == relationship["type"]
        assert edge_data["weight"] == relationship["weight"]
        assert edge_data["attributes"] == relationship["attributes"]

        results = [
            knowledge_graph.add_relationship(rel)
            for rel in sample_relationships[1:]
        ]
        # Verify all relationships were added successfully
        assert all(results)
        assert len(knowledge_graph.graph.edges) == len(sample_relationships)

    def test_add_relationship_missing_entities(
        self, knowledge_graph, sample_relationships
    ):
        """Test adding a relationship when entities don't exist."""
        # Try to add a relationship without adding entities first
        relationship = sample_relationships[0]
        result = knowledge_graph.add_relationship(relationship)

        # Should return False or add the entities automatically
        if result is False:
            # If False, verify no relationship was added
            assert len(knowledge_graph.graph.edges) == 0
        else:
            # If True, verify both entities and the relationship were added
            assert relationship["source"] in knowledge_graph.graph.nodes
            assert relationship["target"] in knowledge_graph.graph.nodes
            assert knowledge_graph.graph.has_edge(
                relationship["source"], relationship["target"]
            )

    def test_get_entity(self, knowledge_graph, sample_entities):
        """Test retrieving an entity from the graph."""
        # Add an entity
        entity = sample_entities[0]
        knowledge_graph.add_entity(entity)

        # Get the entity
        retrieved_entity = knowledge_graph.get_entity(entity["id"])

        # Verify the retrieved entity matches the original
        assert retrieved_entity is not None
        assert retrieved_entity["id"] == entity["id"]
        assert retrieved_entity["name"] == entity["name"]
        assert retrieved_entity["type"] == entity["type"]
        assert retrieved_entity["attributes"] == entity["attributes"]

    def test_get_nonexistent_entity(self, knowledge_graph):
        """Test retrieving an entity that doesn't exist."""
        # Try to get a nonexistent entity
        retrieved_entity = knowledge_graph.get_entity("nonexistent-id")

        # Should return None
        assert retrieved_entity is None

    def test_update_entity(self, knowledge_graph, sample_entities):
        """Test updating an entity."""
        # Add an entity
        entity = sample_entities[0]
        knowledge_graph.add_entity(entity)

        # Update the entity
        updated_entity = entity.copy()
        updated_entity["name"] = "Updated Name"
        updated_entity["attributes"]["new_key"] = "new_value"

        result = knowledge_graph.update_entity(updated_entity)

        # Verify the update was successful
        assert result is True

        # Verify the entity was actually updated
        retrieved_entity = knowledge_graph.get_entity(entity["id"])
        assert retrieved_entity["name"] == "Updated Name"
        assert retrieved_entity["attributes"]["new_key"] == "new_value"
        # Original attributes should still be there
        assert retrieved_entity["attributes"]["key1"] == "value1"

    def test_update_nonexistent_entity(self, knowledge_graph, sample_entities):
        """Test updating an entity that doesn't exist."""
        # Try to update a nonexistent entity
        entity = sample_entities[0].copy()
        entity["id"] = "nonexistent-id"

        result = knowledge_graph.update_entity(entity)

        # Should return False
        assert result is False

    def test_delete_entity(
        self, knowledge_graph, sample_entities, sample_relationships
    ):
        """Test deleting an entity and its relationships."""
        # Add all entities and relationships
        for entity in sample_entities:
            knowledge_graph.add_entity(entity)
        for rel in sample_relationships:
            knowledge_graph.add_relationship(rel)

        # Delete an entity
        entity_id = "entity1"
        result = knowledge_graph.delete_entity(entity_id)

        # Verify the deletion was successful
        assert result is True

        # Verify the entity was actually deleted
        assert entity_id not in knowledge_graph.graph.nodes

        # Verify relationships involving the entity were also deleted
        for rel in sample_relationships:
            if rel["source"] == entity_id or rel["target"] == entity_id:
                assert not knowledge_graph.graph.has_edge(rel["source"], rel["target"])

    def test_delete_nonexistent_entity(self, knowledge_graph):
        """Test deleting an entity that doesn't exist."""
        # Try to delete a nonexistent entity
        result = knowledge_graph.delete_entity("nonexistent-id")

        # Should return False
        assert result is False

    def test_get_relationship(
        self, knowledge_graph, sample_entities, sample_relationships
    ):
        """Test retrieving a relationship from the graph."""
        # Add all entities and relationships
        for entity in sample_entities:
            knowledge_graph.add_entity(entity)
        for rel in sample_relationships:
            knowledge_graph.add_relationship(rel)

        # Get a relationship
        rel = sample_relationships[0]
        retrieved_rel = knowledge_graph.get_relationship(rel["source"], rel["target"])

        # Verify the retrieved relationship matches the original
        assert retrieved_rel is not None
        assert retrieved_rel["source"] == rel["source"]
        assert retrieved_rel["target"] == rel["target"]
        assert retrieved_rel["type"] == rel["type"]
        assert retrieved_rel["weight"] == rel["weight"]
        assert retrieved_rel["attributes"] == rel["attributes"]

    def test_get_nonexistent_relationship(self, knowledge_graph, sample_entities):
        """Test retrieving a relationship that doesn't exist."""
        # Add entities but no relationships
        for entity in sample_entities:
            knowledge_graph.add_entity(entity)

        # Try to get a nonexistent relationship
        retrieved_rel = knowledge_graph.get_relationship("entity1", "entity2")

        # Should return None
        assert retrieved_rel is None

    def test_update_relationship(
        self, knowledge_graph, sample_entities, sample_relationships
    ):
        """Test updating a relationship."""
        # Add all entities and relationships
        for entity in sample_entities:
            knowledge_graph.add_entity(entity)
        for rel in sample_relationships:
            knowledge_graph.add_relationship(rel)

        # Update a relationship
        rel = sample_relationships[0]
        updated_rel = rel.copy()
        updated_rel["type"] = "updated_type"
        updated_rel["weight"] = 0.95
        updated_rel["attributes"]["updated_key"] = "updated_value"

        result = knowledge_graph.update_relationship(updated_rel)

        # Verify the update was successful
        assert result is True

        # Verify the relationship was actually updated
        retrieved_rel = knowledge_graph.get_relationship(rel["source"], rel["target"])
        assert retrieved_rel["type"] == "updated_type"
        assert retrieved_rel["weight"] == 0.95
        assert retrieved_rel["attributes"]["updated_key"] == "updated_value"
        # Original attributes should still be there
        assert retrieved_rel["attributes"]["since"] == "2023"

    def test_update_nonexistent_relationship(
        self, knowledge_graph, sample_entities, sample_relationships
    ):
        """Test updating a relationship that doesn't exist."""
        # Add entities but no relationships
        for entity in sample_entities:
            knowledge_graph.add_entity(entity)

        # Try to update a nonexistent relationship
        rel = sample_relationships[0]

        result = knowledge_graph.update_relationship(rel)

        # Should return False or add the relationship
        if result is False:
            # If False, verify no relationship was added
            assert not knowledge_graph.graph.has_edge(rel["source"], rel["target"])
        else:
            # If True, verify the relationship was added
            assert knowledge_graph.graph.has_edge(rel["source"], rel["target"])

    def test_delete_relationship(
        self, knowledge_graph, sample_entities, sample_relationships
    ):
        """Test deleting a relationship."""
        # Add all entities and relationships
        for entity in sample_entities:
            knowledge_graph.add_entity(entity)
        for rel in sample_relationships:
            knowledge_graph.add_relationship(rel)

        # Delete a relationship
        rel = sample_relationships[0]
        result = knowledge_graph.delete_relationship(rel["source"], rel["target"])

        # Verify the deletion was successful
        assert result is True

        # Verify the relationship was actually deleted
        assert not knowledge_graph.graph.has_edge(rel["source"], rel["target"])

        # Verify entities still exist
        assert rel["source"] in knowledge_graph.graph.nodes
        assert rel["target"] in knowledge_graph.graph.nodes

    def test_delete_nonexistent_relationship(self, knowledge_graph, sample_entities):
        """Test deleting a relationship that doesn't exist."""
        # Add entities but no relationships
        for entity in sample_entities:
            knowledge_graph.add_entity(entity)

        # Try to delete a nonexistent relationship
        result = knowledge_graph.delete_relationship("entity1", "entity2")

        # Should return False
        assert result is False

    def test_search_entities(self, knowledge_graph, sample_entities):
        """Test searching for entities with specific criteria."""
        # Add all entities
        for entity in sample_entities:
            knowledge_graph.add_entity(entity)

        # Search for entities by type
        matching_entities = knowledge_graph.search_entities({"type": "concept"})

        # Verify the search returned the correct entities
        assert len(matching_entities) == 2  # There are 2 concept entities
        assert all(entity["type"] == "concept" for entity in matching_entities)

        # Search with multiple criteria
        matching_entities = knowledge_graph.search_entities(
            {"type": "concept", "attributes.key1": "value1"}
        )

        # Verify the search returned the correct entities
        assert len(matching_entities) == 1  # Only one entity matches both criteria
        assert matching_entities[0]["id"] == "entity1"

    def test_search_relationships(
        self, knowledge_graph, sample_entities, sample_relationships
    ):
        """Test searching for relationships with specific criteria."""
        # Add all entities and relationships
        for entity in sample_entities:
            knowledge_graph.add_entity(entity)
        for rel in sample_relationships:
            knowledge_graph.add_relationship(rel)

        # Search for relationships by type
        matching_rels = knowledge_graph.search_relationships({"type": "related_to"})

        # Verify the search returned the correct relationships
        assert len(matching_rels) == 1  # There is 1 related_to relationship
        assert matching_rels[0]["type"] == "related_to"

        # Search with weight threshold
        matching_rels = knowledge_graph.search_relationships({"min_weight": 0.8})

        # Verify the search returned the correct relationships
        assert len(matching_rels) == 2  # Two relationships have weight >= 0.8
        assert all(rel["weight"] >= 0.8 for rel in matching_rels)

    def test_get_entity_relationships(
        self, knowledge_graph, sample_entities, sample_relationships
    ):
        """Test getting all relationships for a specific entity."""
        # Add all entities and relationships
        for entity in sample_entities:
            knowledge_graph.add_entity(entity)
        for rel in sample_relationships:
            knowledge_graph.add_relationship(rel)

        # Get relationships for entity1
        entity_rels = knowledge_graph.get_entity_relationships("entity1")

        # Verify the function returned the correct relationships
        # entity1 is involved in 2 relationships:
        # - source of a relationship to entity2
        # - target of a relationship from entity5
        assert len(entity_rels) == 2

        # Verify specific relationship direction
        outgoing_rels = [rel for rel in entity_rels if rel["source"] == "entity1"]
        incoming_rels = [rel for rel in entity_rels if rel["target"] == "entity1"]

        assert len(outgoing_rels) == 1
        assert len(incoming_rels) == 1
        assert outgoing_rels[0]["target"] == "entity2"
        assert incoming_rels[0]["source"] == "entity5"

    def test_get_entity_neighborhood(
        self, knowledge_graph, sample_entities, sample_relationships
    ):
        """Test getting the neighborhood (related entities) for a specific entity."""
        # Add all entities and relationships
        for entity in sample_entities:
            knowledge_graph.add_entity(entity)
        for rel in sample_relationships:
            knowledge_graph.add_relationship(rel)

        # Get neighborhood for entity1 with depth 1
        neighborhood = knowledge_graph.get_entity_neighborhood("entity1", depth=1)

        # Verify the function returned the correct neighborhood
        # entity1 is directly connected to entity2 and entity5
        assert len(neighborhood["entities"]) == 3  # entity1 + 2 neighbors
        entity_ids = [entity["id"] for entity in neighborhood["entities"]]
        assert "entity1" in entity_ids
        assert "entity2" in entity_ids
        assert "entity5" in entity_ids

        # Verify the relationships are included
        assert len(neighborhood["relationships"]) == 2

        # Get neighborhood with depth 2
        neighborhood_depth2 = knowledge_graph.get_entity_neighborhood(
            "entity1", depth=2
        )

        # Verify depth 2 neighborhood includes additional entities
        # Should include entity3 (connected to entity2)
        assert len(neighborhood_depth2["entities"]) > len(neighborhood["entities"])
        entity_ids_depth2 = [entity["id"] for entity in neighborhood_depth2["entities"]]
        assert "entity3" in entity_ids_depth2

    def test_compute_shortest_path(
        self, knowledge_graph, sample_entities, sample_relationships
    ):
        """Test computing the shortest path between two entities."""
        # Add all entities and relationships
        for entity in sample_entities:
            knowledge_graph.add_entity(entity)
        for rel in sample_relationships:
            knowledge_graph.add_relationship(rel)

        # Compute shortest path from entity1 to entity4
        # Expected path: entity1 -> entity2 -> entity3 -> entity4
        path = knowledge_graph.compute_shortest_path("entity1", "entity4")

        # Verify the correct path was found
        assert path is not None
        assert len(path["entities"]) == 4
        assert path["entities"][0]["id"] == "entity1"
        assert path["entities"][1]["id"] == "entity2"
        assert path["entities"][2]["id"] == "entity3"
        assert path["entities"][3]["id"] == "entity4"

        # Verify the relationships in the path
        assert len(path["relationships"]) == 3

    def test_compute_shortest_path_nonexistent(
        self, knowledge_graph, sample_entities, sample_relationships
    ):
        """Test computing a path when one of the entities doesn't exist."""
        # Add all entities and relationships
        for entity in sample_entities:
            knowledge_graph.add_entity(entity)
        for rel in sample_relationships:
            knowledge_graph.add_relationship(rel)

        # Try to compute path with nonexistent entity
        path = knowledge_graph.compute_shortest_path("entity1", "nonexistent-id")

        # Should return None or empty path
        assert path is None or (
            len(path["entities"]) == 0 and len(path["relationships"]) == 0
        )

    @patch("builtins.open", new_callable=MagicMock)
    def test_save_to_file(
        self, mock_open, knowledge_graph, sample_entities, sample_relationships
    ):
        """Test saving the graph to a file."""
        # Add all entities and relationships
        for entity in sample_entities:
            knowledge_graph.add_entity(entity)
        for rel in sample_relationships:
            knowledge_graph.add_relationship(rel)

        # Configure the mock to capture the written data
        mock_file = MagicMock()
        mock_open.return_value.__enter__.return_value = mock_file

        # Save the graph
        knowledge_graph.save_to_file("test_graph.json")

        # Verify the file was opened for writing
        mock_open.assert_called_once_with("test_graph.json", "w")

        # Verify write was called
        assert mock_file.write.called

        # Verify the written content contains all entities and relationships
        written_data = mock_file.write.call_args[0][0]
        data = json.loads(written_data)

        assert "entities" in data
        assert "relationships" in data
        assert len(data["entities"]) == len(sample_entities)
        assert len(data["relationships"]) == len(sample_relationships)

    @patch("builtins.open", new_callable=MagicMock)
    def test_load_from_file(self, mock_open, knowledge_graph):
        """Test loading the graph from a file."""
        # Create sample data to load
        data = {
            "entities": [
                {
                    "id": "entity1",
                    "name": "Entity 1",
                    "type": "concept",
                    "attributes": {"key1": "value1"},
                }
            ],
            "relationships": [
                {
                    "source": "entity1",
                    "target": "entity2",
                    "type": "related_to",
                    "weight": 0.8,
                    "attributes": {},
                }
            ],
        }

        # Configure the mock to return the sample data
        mock_file = MagicMock()
        mock_file.read.return_value = json.dumps(data)
        mock_open.return_value.__enter__.return_value = mock_file

        # Load the graph
        result = knowledge_graph.load_from_file("test_graph.json")

        # Verify the file was opened for reading
        mock_open.assert_called_once_with("test_graph.json", "r")

        # Verify the load was successful
        assert result is True

        # Verify the graph contains the loaded data
        assert "entity1" in knowledge_graph.graph.nodes

        # Note: The relationship might not be loaded if entity2 doesn't exist
        # and the implementation doesn't automatically create missing entities
