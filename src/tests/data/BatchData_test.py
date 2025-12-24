import pytest

from src.main import BatchData


class TestBatchData:
    """Test suite for the BatchData class."""

    @pytest.fixture
    def sample_items(self):
        """Create sample items for testing."""
        return [
            {"id": "item1", "content": "Content 1", "tags": ["tag1", "tag2"]},
            {"id": "item2", "content": "Content 2", "tags": ["tag2", "tag3"]},
            {"id": "item3", "content": "Content 3", "tags": ["tag1", "tag3"]},
        ]

    @pytest.fixture
    def sample_metadata(self):
        """Create sample metadata for testing."""
        return {
            "batch_created": "2025-03-31T04:10:29-07:00",
            "source": "test_source",
            "priority": "high",
        }

    def test_init_with_required_fields(self, sample_items):
        """Test initialization with only required fields."""
        batch_id = "batch-123"
        batch_data = BatchData(items=sample_items, batch_id=batch_id)

        # Verify required fields
        assert batch_data.items == sample_items
        assert batch_data.batch_id == batch_id

        # Verify default values
        assert isinstance(batch_data.metadata, dict)
        assert len(batch_data.metadata) == 0

    def test_init_with_all_fields(self, sample_items, sample_metadata):
        """Test initialization with all fields."""
        batch_id = "batch-123"
        batch_data = BatchData(
            items=sample_items, batch_id=batch_id, metadata=sample_metadata
        )

        # Verify all fields
        assert batch_data.items == sample_items
        assert batch_data.batch_id == batch_id
        assert batch_data.metadata == sample_metadata

    def test_dataclass_behavior(self):
        """Test that BatchData behaves as expected for a dataclass."""
        # Create two identical BatchData instances
        items = [{"id": "test"}]
        batch_id = "batch-123"
        metadata = {"key": "value"}

        batch1 = BatchData(items, batch_id, metadata)
        batch2 = BatchData(items, batch_id, metadata)

        # Dataclasses should implement __eq__ by default
        assert batch1 == batch2

        # Different batch_id should result in inequality
        batch3 = BatchData(items, "different-id", metadata)
        assert batch1 != batch3

    def test_metadata_default_factory(self):
        """Test that metadata default factory creates a new dict each time."""
        batch1 = BatchData(items=[{"id": "item1"}], batch_id="batch1")
        batch2 = BatchData(items=[{"id": "item2"}], batch_id="batch2")

        # Each instance should have its own empty dict
        assert batch1.metadata == {}
        assert batch2.metadata == {}

        # Modifying one metadata dict should not affect the other
        batch1.metadata["key"] = "value"
        assert "key" in batch1.metadata
        assert "key" not in batch2.metadata

    def test_mutability(self, sample_items, sample_metadata):
        """Test that BatchData attributes can be modified."""
        batch_data = BatchData(
            items=sample_items.copy(),
            batch_id="batch-123",
            metadata=sample_metadata.copy(),
        )

        # Modify attributes
        new_item = {"id": "item4", "content": "Content 4", "tags": ["tag4"]}
        batch_data.items.append(new_item)
        batch_data.batch_id = "modified-batch-id"
        batch_data.metadata["new_key"] = "new_value"

        # Verify modifications
        assert len(batch_data.items) == len(sample_items) + 1
        assert batch_data.items[-1] == new_item
        assert batch_data.batch_id == "modified-batch-id"
        assert batch_data.metadata["new_key"] == "new_value"

    def test_with_empty_items(self):
        """Test BatchData with empty items list."""
        batch_data = BatchData(items=[], batch_id="empty-batch")

        # Verify items is an empty list
        assert batch_data.items == []
        assert len(batch_data.items) == 0

    def test_with_complex_nested_items(self):
        """Test BatchData with complex nested item structure."""
        complex_items = [
            {
                "id": "complex1",
                "content": {
                    "title": "Complex Item 1",
                    "sections": [
                        {"heading": "Section 1", "text": "Text 1"},
                        {"heading": "Section 2", "text": "Text 2"},
                    ],
                },
                "metadata": {
                    "created": "2025-03-31T04:10:29-07:00",
                    "stats": {"word_count": 150, "read_time": 5},
                },
            }
        ]

        batch_data = BatchData(items=complex_items, batch_id="complex-batch")

        # Verify the complex items structure is retained
        assert batch_data.items == complex_items
        assert batch_data.items[0]["content"]["sections"][1]["heading"] == "Section 2"
