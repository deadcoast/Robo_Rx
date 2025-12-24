import asyncio

import pytest

from src.main import RelationExtractor


class TestRelationExtractor:
    """Test suite for the RelationExtractor class."""

    @pytest.fixture
    def mock_config(self):
        """Create a mock configuration for testing."""
        return {
            "relation_extractor": {
                "model": "base",
                "confidence_threshold": 0.7,
                "max_relations_per_entity": 20,
                "enable_caching": True,
                "relation_types": [
                    "contains",
                    "references",
                    "is_related_to",
                    "depends_on",
                    "follows",
                ],
            }
        }

    @pytest.fixture
    def relation_extractor(self, mock_config):
        """Create a RelationExtractor instance for testing."""
        return RelationExtractor(mock_config)

    @pytest.fixture
    def sample_text(self):
        """Create sample text with potential relations for testing."""
        return """
        The project depends on Python 3.8 and requires TensorFlow for machine learning functionality.
        John Smith works at Acme Corporation, which is located in New York City.
        The report was written by Sarah Johnson and reviewed by Michael Brown.
        Chapter 2 follows Chapter 1 and contains information about data structures.
        The API references the database schema, which contains user information.
        """

    @pytest.fixture
    def sample_entities(self):
        """Create sample entities for testing."""
        return [
            {
                "id": "e1",
                "text": "Python 3.8",
                "type": "technology",
                "start": 22,
                "end": 32,
            },
            {
                "id": "e2",
                "text": "TensorFlow",
                "type": "technology",
                "start": 50,
                "end": 60,
            },
            {
                "id": "e3",
                "text": "machine learning",
                "type": "concept",
                "start": 65,
                "end": 81,
            },
            {
                "id": "e4",
                "text": "John Smith",
                "type": "person",
                "start": 91,
                "end": 102,
            },
            {
                "id": "e5",
                "text": "Acme Corporation",
                "type": "organization",
                "start": 113,
                "end": 130,
            },
            {
                "id": "e6",
                "text": "New York City",
                "type": "location",
                "start": 152,
                "end": 165,
            },
            {
                "id": "e7",
                "text": "report",
                "type": "document",
                "start": 175,
                "end": 181,
            },
            {
                "id": "e8",
                "text": "Sarah Johnson",
                "type": "person",
                "start": 197,
                "end": 210,
            },
            {
                "id": "e9",
                "text": "Michael Brown",
                "type": "person",
                "start": 225,
                "end": 238,
            },
            {
                "id": "e10",
                "text": "Chapter 2",
                "type": "section",
                "start": 248,
                "end": 257,
            },
            {
                "id": "e11",
                "text": "Chapter 1",
                "type": "section",
                "start": 266,
                "end": 275,
            },
            {
                "id": "e12",
                "text": "data structures",
                "type": "concept",
                "start": 298,
                "end": 313,
            },
            {
                "id": "e13",
                "text": "API",
                "type": "technology",
                "start": 323,
                "end": 326,
            },
            {
                "id": "e14",
                "text": "database schema",
                "type": "technology",
                "start": 342,
                "end": 357,
            },
            {
                "id": "e15",
                "text": "user information",
                "type": "concept",
                "start": 376,
                "end": 392,
            },
        ]

    def test_init(self, relation_extractor, mock_config):
        """Test initialization of RelationExtractor."""
        assert relation_extractor.config == mock_config
        assert (
            relation_extractor.confidence_threshold
            == mock_config["relation_extractor"]["confidence_threshold"]
        )
        assert (
            relation_extractor.relation_types
            == mock_config["relation_extractor"]["relation_types"]
        )
        assert hasattr(relation_extractor, "model") or hasattr(
            relation_extractor, "pipeline"
        )

    @pytest.mark.asyncio
    async def test_extract_relations_from_text(self, relation_extractor, sample_text):
        """Test extracting relations directly from text."""
        # Extract relations from text
        relations = await relation_extractor.extract_relations_from_text(sample_text)

        # Verify the result structure
        assert isinstance(relations, list)

        # Each relation should have the expected structure
        for relation in relations:
            assert "source" in relation
            assert "target" in relation
            assert "relation_type" in relation
            assert "confidence" in relation

            # Verify source and target are dictionaries with expected fields
            assert "text" in relation["source"]
            assert "type" in relation["source"]
            assert "text" in relation["target"]
            assert "type" in relation["target"]

            # Verify relation_type is one of the supported types
            assert relation["relation_type"] in relation_extractor.relation_types

            # Verify confidence is within expected range and above threshold
            assert 0 <= relation["confidence"] <= 1.0
            assert relation["confidence"] >= relation_extractor.confidence_threshold

    @pytest.mark.asyncio
    async def test_extract_relations_from_entities(
        self, relation_extractor, sample_text, sample_entities
    ):
        """Test extracting relations from pre-extracted entities."""
        # Extract relations from entities
        relations = await relation_extractor.extract_relations_from_entities(
            sample_text, sample_entities
        )

        # Verify the result structure
        assert isinstance(relations, list)

        # Should have found some relations
        assert len(relations) > 0

        # Each relation should have the expected structure
        for relation in relations:
            assert "source_id" in relation
            assert "target_id" in relation
            assert "relation_type" in relation
            assert "confidence" in relation

            # Verify source_id and target_id are valid entity IDs
            entity_ids = [entity["id"] for entity in sample_entities]
            assert relation["source_id"] in entity_ids
            assert relation["target_id"] in entity_ids

            # Verify relation_type is one of the supported types
            assert relation["relation_type"] in relation_extractor.relation_types

            # Verify confidence is within expected range and above threshold
            assert 0 <= relation["confidence"] <= 1.0
            assert relation["confidence"] >= relation_extractor.confidence_threshold

    @pytest.mark.asyncio
    async def test_extract_relations_between_specific_entities(
        self, relation_extractor, sample_text, sample_entities
    ):
        """Test extracting relations between specific entity pairs."""
        # Select specific entities
        entity_pairs = [
            (sample_entities[0], sample_entities[2]),  # Python 3.8 -> machine learning
            (sample_entities[3], sample_entities[4]),  # John Smith -> Acme Corporation
            (sample_entities[9], sample_entities[10]),  # Chapter 2 -> Chapter 1
        ]

        # Extract relations for specific entity pairs
        relations = (
            await relation_extractor.extract_relations_between_specific_entities(
                sample_text, entity_pairs
            )
        )

        # Verify the result structure
        assert isinstance(relations, list)

        # Should have found some relations (at least one per pair)
        assert len(relations) > 0

        # Each relation should have the expected structure
        for relation in relations:
            assert "source_id" in relation
            assert "target_id" in relation
            assert "relation_type" in relation
            assert "confidence" in relation

            # Verify relation involves one of the specified pairs
            pair_ids = [(pair[0]["id"], pair[1]["id"]) for pair in entity_pairs]
            relation_pair = (relation["source_id"], relation["target_id"])
            assert relation_pair in pair_ids

    @pytest.mark.asyncio
    async def test_filter_relations_by_confidence(
        self, relation_extractor, sample_text, sample_entities
    ):
        """Test filtering relations by confidence threshold."""
        # Set a low initial threshold to get more relations
        original_threshold = relation_extractor.confidence_threshold
        relation_extractor.confidence_threshold = 0.1

        # Extract relations with low threshold
        all_relations = await relation_extractor.extract_relations_from_entities(
            sample_text, sample_entities
        )

        # Apply a higher filter threshold
        high_threshold = 0.8
        filtered_relations = relation_extractor.filter_relations_by_confidence(
            all_relations, high_threshold
        )

        # Verify filtering
        assert len(filtered_relations) <= len(all_relations)
        assert all(
            relation["confidence"] >= high_threshold for relation in filtered_relations
        )

        # Restore original threshold
        relation_extractor.confidence_threshold = original_threshold

    @pytest.mark.asyncio
    async def test_filter_relations_by_type(
        self, relation_extractor, sample_text, sample_entities
    ):
        """Test filtering relations by relation type."""
        # Extract all relations
        all_relations = await relation_extractor.extract_relations_from_entities(
            sample_text, sample_entities
        )

        # Apply type filter for a specific relation type
        target_type = "depends_on"
        filtered_relations = relation_extractor.filter_relations_by_type(
            all_relations, [target_type]
        )

        # Verify filtering
        assert all(
            relation["relation_type"] == target_type for relation in filtered_relations
        )

        # Filter for multiple types
        target_types = ["depends_on", "follows"]
        filtered_relations = relation_extractor.filter_relations_by_type(
            all_relations, target_types
        )

        # Verify filtering
        assert all(
            relation["relation_type"] in target_types for relation in filtered_relations
        )

    @pytest.mark.asyncio
    async def test_group_relations_by_entity(
        self, relation_extractor, sample_text, sample_entities
    ):
        """Test grouping relations by entity."""
        # Extract all relations
        all_relations = await relation_extractor.extract_relations_from_entities(
            sample_text, sample_entities
        )

        # Group relations by source entity
        grouped_relations = relation_extractor.group_relations_by_entity(
            all_relations, "source"
        )

        # Verify the grouping
        assert isinstance(grouped_relations, dict)

        # Each key should be an entity ID, and each value a list of relations
        for entity_id, relations in grouped_relations.items():
            assert entity_id in [entity["id"] for entity in sample_entities]
            assert isinstance(relations, list)
            assert all(relation["source_id"] == entity_id for relation in relations)

    @pytest.mark.asyncio
    async def test_extract_from_large_text(self, relation_extractor):
        """Test extraction from a larger text to ensure scalability."""
        # Create a larger text by repeating the sample text
        large_text = (
            """
        The project depends on Python 3.8 and requires TensorFlow for machine learning functionality.
        John Smith works at Acme Corporation, which is located in New York City.
        The report was written by Sarah Johnson and reviewed by Michael Brown.
        Chapter 2 follows Chapter 1 and contains information about data structures.
        The API references the database schema, which contains user information.
        """
            * 10
        )  # Repeat 10 times

        # Extract relations from the large text
        relations = await relation_extractor.extract_relations_from_text(large_text)

        # Verify the extraction completed and returned relations
        assert isinstance(relations, list)
        assert len(relations) > 0

    @pytest.mark.asyncio
    async def test_extract_with_custom_relation_types(
        self, relation_extractor, sample_text
    ):
        """Test extraction with custom relation types."""
        # Set custom relation types
        custom_types = ["employs", "located_in", "authored_by", "reviewed_by"]
        relation_extractor.relation_types = custom_types

        # Extract relations with custom types
        relations = await relation_extractor.extract_relations_from_text(sample_text)

        # Verify the result structure
        assert isinstance(relations, list)

        # If any relations were found, they should only use the custom types
        for relation in relations:
            if "relation_type" in relation:
                assert relation["relation_type"] in custom_types

    @pytest.mark.asyncio
    async def test_extract_with_no_entities(self, relation_extractor, sample_text):
        """Test extraction when no entities are present."""
        # Extract relations with empty entities list
        relations = await relation_extractor.extract_relations_from_entities(
            sample_text, []
        )

        # Should return an empty list
        assert isinstance(relations, list)
        assert len(relations) == 0

    @pytest.mark.asyncio
    async def test_extraction_with_entity_context(
        self, relation_extractor, sample_text, sample_entities
    ):
        """Test extraction with entity context information."""
        # Add context information to entities
        entities_with_context = []
        for entity in sample_entities:
            entity_copy = entity.copy()

            # Calculate context as 50 characters before and after the entity
            start_ctx = max(0, entity["start"] - 50)
            end_ctx = min(len(sample_text), entity["end"] + 50)
            context = sample_text[start_ctx:end_ctx]

            entity_copy["context"] = context
            entities_with_context.append(entity_copy)

        # Extract relations with context-enhanced entities
        relations = await relation_extractor.extract_relations_from_entities(
            sample_text, entities_with_context, use_entity_context=True
        )

        # Verify the extraction completed and returned relations
        assert isinstance(relations, list)
        assert len(relations) > 0

    @pytest.mark.asyncio
    async def test_caching_behavior(self, relation_extractor, sample_text):
        """Test caching behavior of relation extraction."""
        # Enable caching
        relation_extractor.enable_caching = True

        # First extraction
        first_result = await relation_extractor.extract_relations_from_text(sample_text)

        # Mock the internal extraction method to verify it's not called twice
        original_extract = relation_extractor._extract_relations
        extract_called = False

        async def mock_extract(*args, **kwargs):
            nonlocal extract_called
            extract_called = True
            return await original_extract(*args, **kwargs)

        relation_extractor._extract_relations = mock_extract

        # Second extraction with the same text
        second_result = await relation_extractor.extract_relations_from_text(
            sample_text
        )

        # Verify extraction was not called again (using cache)
        assert not extract_called

        # Verify results are the same
        assert second_result == first_result

        # Clear cache and verify extraction is called again
        relation_extractor.clear_cache()
        await relation_extractor.extract_relations_from_text(sample_text)
        assert extract_called

    @pytest.mark.asyncio
    async def test_relation_extraction_with_batching(
        self, relation_extractor, sample_entities
    ):
        """Test relation extraction with batching for efficiency."""
        # Create a larger list of entities by duplicating the sample
        large_entity_list = sample_entities * 10  # 150 entities

        # Set a small batch size
        relation_extractor.batch_size = 10

        # Extract relations with batching
        batch_start_time = asyncio.get_event_loop().time()
        batch_relations = (
            await relation_extractor.extract_relations_from_entities_batch(
                "sample text", large_entity_list
            )
        )
        asyncio.get_event_loop().time() - batch_start_time

        # Verify the extraction completed and returned relations
        assert isinstance(batch_relations, list)

        # Should have processed all potential entity pairs across batches
        max_potential_relations = (
            len(large_entity_list) * (len(large_entity_list) - 1)
        ) // 2
        assert len(batch_relations) <= max_potential_relations

    @pytest.mark.asyncio
    async def test_concurrent_extraction(
        self, relation_extractor, sample_text, sample_entities
    ):
        """Test that multiple concurrent extraction operations work correctly."""
        # Split entities into three groups
        entity_groups = [
            sample_entities[:5],
            sample_entities[5:10],
            sample_entities[10:],
        ]

        # Create tasks for concurrent extraction
        tasks = [
            asyncio.create_task(
                relation_extractor.extract_relations_from_entities(
                    sample_text, entity_group
                )
            )
            for entity_group in entity_groups
        ]

        # Wait for all tasks to complete
        results = await asyncio.gather(*tasks)

        # Verify all operations succeeded
        assert len(results) == 3
        assert all(isinstance(result, list) for result in results)

        # Merge results
        all_relations = []
        for result in results:
            all_relations.extend(result)

        # Verify we found some relations
        assert all_relations
