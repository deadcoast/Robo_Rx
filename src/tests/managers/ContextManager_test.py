import asyncio
from contextlib import suppress

import pytest

from src.main import ContextManager


class TestContextManager:
    """Test suite for the ContextManager class."""

    @pytest.fixture
    def mock_config(self):
        """Create a mock configuration for testing."""
        return {
            "context": {
                "max_context_size": 10,
                "context_window": 1000,
                "preserve_history": True,
                "default_ttl": 3600,  # 1 hour in seconds
            }
        }

    @pytest.fixture
    def context_manager(self, mock_config):
        """Create a ContextManager instance for testing."""
        return ContextManager(mock_config)

    @pytest.fixture
    def sample_context_data(self):
        """Create sample context data for testing."""
        return {
            "user_id": "user123",
            "session_id": "session456",
            "query": "test query",
            "timestamp": 1648704000,  # Example timestamp
            "metadata": {"source": "web", "language": "en"},
        }

    def test_init(self, context_manager, mock_config):
        """Test initialization of ContextManager."""
        assert context_manager.config == mock_config
        assert hasattr(context_manager, "contexts")
        assert isinstance(context_manager.contexts, dict)
        assert len(context_manager.contexts) == 0

    @pytest.mark.asyncio
    async def test_create_context(self, context_manager, sample_context_data, mock_config):
        """Test creating a new context."""
        # Create a new context
        context_id = await context_manager.create_context(sample_context_data)

        # Verify the context was created
        assert context_id is not None
        assert isinstance(context_id, str)
        assert context_id in context_manager.contexts

        # Verify the context contains the provided data
        context = context_manager.contexts[context_id]
        assert "data" in context
        assert context["data"] == sample_context_data

        # Verify the context contains creation time
        assert "created_at" in context
        assert isinstance(context["created_at"], int)

        # Verify the context contains TTL (time-to-live)
        assert "ttl" in context
        assert context["ttl"] == mock_config["context"]["default_ttl"]

    @pytest.mark.asyncio
    async def test_create_context_with_custom_ttl(
        self, context_manager, sample_context_data
    ):
        """Test creating a context with a custom TTL."""
        # Create a context with custom TTL
        custom_ttl = 7200  # 2 hours
        context_id = await context_manager.create_context(
            sample_context_data, ttl=custom_ttl
        )

        # Verify the context uses the custom TTL
        assert context_manager.contexts[context_id]["ttl"] == custom_ttl

    @pytest.mark.asyncio
    async def test_get_context(self, context_manager, sample_context_data):
        """Test retrieving a context."""
        # Create a context
        context_id = await context_manager.create_context(sample_context_data)

        # Retrieve the context
        context = await context_manager.get_context(context_id)

        # Verify the context data
        assert context is not None
        assert context == sample_context_data

    @pytest.mark.asyncio
    async def test_get_nonexistent_context(self, context_manager):
        """Test retrieving a context that doesn't exist."""
        # Try to retrieve a nonexistent context
        context = await context_manager.get_context("nonexistent-id")

        # Should return None
        assert context is None

    @pytest.mark.asyncio
    async def test_update_context(self, context_manager, sample_context_data):
        """Test updating a context."""
        # Create a context
        context_id = await context_manager.create_context(sample_context_data)

        # Update the context
        updated_data = sample_context_data.copy()
        updated_data["query"] = "updated query"
        updated_data["metadata"]["additional"] = "new value"

        success = await context_manager.update_context(context_id, updated_data)

        # Verify the update was successful
        assert success is True

        # Verify the context was updated
        updated_context = await context_manager.get_context(context_id)
        assert updated_context == updated_data
        assert updated_context["query"] == "updated query"
        assert updated_context["metadata"]["additional"] == "new value"

    @pytest.mark.asyncio
    async def test_update_nonexistent_context(
        self, context_manager, sample_context_data
    ):
        """Test updating a context that doesn't exist."""
        # Try to update a nonexistent context
        success = await context_manager.update_context(
            "nonexistent-id", sample_context_data
        )

        # Should return False
        assert success is False

    @pytest.mark.asyncio
    async def test_delete_context(self, context_manager, sample_context_data):
        """Test deleting a context."""
        # Create a context
        context_id = await context_manager.create_context(sample_context_data)

        # Delete the context
        success = await context_manager.delete_context(context_id)

        # Verify the deletion was successful
        assert success is True

        # Verify the context was actually deleted
        assert context_id not in context_manager.contexts

        # Try to retrieve the deleted context
        deleted_context = await context_manager.get_context(context_id)
        assert deleted_context is None

    @pytest.mark.asyncio
    async def test_delete_nonexistent_context(self, context_manager):
        """Test deleting a context that doesn't exist."""
        # Try to delete a nonexistent context
        success = await context_manager.delete_context("nonexistent-id")

        # Should return False
        assert success is False

    @pytest.mark.asyncio
    async def test_context_expiration(self, context_manager, sample_context_data):
        """Test that contexts expire after their TTL."""
        # Create a context with a short TTL
        short_ttl = 1  # 1 second
        context_id = await context_manager.create_context(
            sample_context_data, ttl=short_ttl
        )

        # Verify the context exists
        assert context_id in context_manager.contexts

        # Wait for the context to expire
        await asyncio.sleep(short_ttl + 1)

        # Run the expiration check
        await context_manager.clean_expired_contexts()

        # Verify the context was expired and removed
        assert context_id not in context_manager.contexts

    @pytest.mark.asyncio
    async def test_list_contexts(self, context_manager, sample_context_data):
        """Test listing all contexts."""
        # Create multiple contexts
        context_id1 = await context_manager.create_context(sample_context_data)

        context_data2 = sample_context_data.copy()
        context_data2["user_id"] = "user456"
        context_id2 = await context_manager.create_context(context_data2)

        context_data3 = sample_context_data.copy()
        context_data3["session_id"] = "session789"
        context_id3 = await context_manager.create_context(context_data3)

        # List all contexts
        contexts = await context_manager.list_contexts()

        # Verify we got all contexts
        assert len(contexts) == 3
        assert all(isinstance(ctx, dict) for ctx in contexts)
        assert all("id" in ctx for ctx in contexts)
        assert all("data" in ctx for ctx in contexts)

        # Verify context IDs
        context_ids = [ctx["id"] for ctx in contexts]
        assert context_id1 in context_ids
        assert context_id2 in context_ids
        assert context_id3 in context_ids

    @pytest.mark.asyncio
    async def test_list_contexts_with_filter(
        self, context_manager, sample_context_data
    ):
        """Test listing contexts with filters."""
        # Create multiple contexts
        await context_manager.create_context(sample_context_data)

        context_data2 = sample_context_data.copy()
        context_data2["user_id"] = "user456"
        await context_manager.create_context(context_data2)

        context_data3 = sample_context_data.copy()
        context_data3["session_id"] = "session789"
        await context_manager.create_context(context_data3)

        # Filter by user_id
        def filter_fn(ctx):
            return ctx["data"]["user_id"] == "user123"

        filtered_contexts = await context_manager.list_contexts(filter_fn)

        # Should return contexts for user123 only
        assert len(filtered_contexts) == 2
        assert all(ctx["data"]["user_id"] == "user123" for ctx in filtered_contexts)

    @pytest.mark.asyncio
    async def test_merge_contexts(self, context_manager, sample_context_data):
        """Test merging multiple contexts."""
        # Create contexts with different data
        context_data1 = sample_context_data.copy()
        context_data1["metadata"]["source"] = "web"
        context_id1 = await context_manager.create_context(context_data1)

        context_data2 = sample_context_data.copy()
        context_data2["metadata"]["language"] = "fr"
        context_data2["additional_field"] = "value"
        context_id2 = await context_manager.create_context(context_data2)

        # Merge the contexts
        merged_context_id = await context_manager.merge_contexts(
            [context_id1, context_id2]
        )

        # Verify the merged context
        merged_context = await context_manager.get_context(merged_context_id)

        # Should contain values from both contexts
        assert merged_context["metadata"]["source"] == "web"
        assert merged_context["metadata"]["language"] == "fr"
        assert "additional_field" in merged_context
        assert merged_context["additional_field"] == "value"

        # Original contexts should still exist
        assert await context_manager.get_context(context_id1) is not None
        assert await context_manager.get_context(context_id2) is not None

    @pytest.mark.asyncio
    async def test_merge_contexts_with_conflict_resolution(
        self, context_manager, sample_context_data
    ):
        """Test merging contexts with conflict resolution."""
        # Create contexts with conflicting data
        context_data1 = sample_context_data.copy()
        context_data1["priority"] = 10
        context_id1 = await context_manager.create_context(context_data1)

        context_data2 = sample_context_data.copy()
        context_data2["priority"] = 20
        context_id2 = await context_manager.create_context(context_data2)

        # Define a conflict resolution function (take highest priority)
        def resolve_conflict(v1, v2):
            return v1 if v1 > v2 else v2

        # Merge with conflict resolution
        merged_context_id = await context_manager.merge_contexts(
            [context_id1, context_id2],
            conflict_resolution={"priority": resolve_conflict},
        )

        # Verify the merged context resolved the conflict correctly
        merged_context = await context_manager.get_context(merged_context_id)
        assert merged_context["priority"] == 20  # Higher value wins

    @pytest.mark.asyncio
    async def test_append_to_context(self, context_manager, sample_context_data):
        """Test appending to an existing context."""
        # Create a context with a list field
        context_data = sample_context_data.copy()
        context_data["history"] = ["item1", "item2"]
        context_id = await context_manager.create_context(context_data)

        # Append to the list
        success = await context_manager.append_to_context(
            context_id, "history", "item3"
        )

        # Verify the append was successful
        assert success is True

        # Verify the context was updated
        updated_context = await context_manager.get_context(context_id)
        assert "history" in updated_context
        assert len(updated_context["history"]) == 3
        assert updated_context["history"] == ["item1", "item2", "item3"]

    @pytest.mark.asyncio
    async def test_append_to_nonexistent_field(
        self, context_manager, sample_context_data
    ):
        """Test appending to a nonexistent field."""
        # Create a context without the target field
        context_id = await context_manager.create_context(sample_context_data)

        # Append to a nonexistent field
        success = await context_manager.append_to_context(
            context_id, "history", "item1"
        )

        # Should still succeed and create the field
        assert success is True

        # Verify the context was updated
        updated_context = await context_manager.get_context(context_id)
        assert "history" in updated_context
        assert len(updated_context["history"]) == 1
        assert updated_context["history"] == ["item1"]

    @pytest.mark.asyncio
    async def test_append_to_non_list_field(self, context_manager, sample_context_data):
        """Test appending to a field that's not a list."""
        # Create a context with a non-list field
        context_data = sample_context_data.copy()
        context_data["non_list"] = "string value"
        context_id = await context_manager.create_context(context_data)

        # Append to a non-list field - using suppress to handle TypeErrors
        with suppress(TypeError):
            success = await context_manager.append_to_context(
                context_id, "non_list", "item1"
            )

            # Should either return False
            assert success is False
            # Or convert the field to a list containing both values
            if success:
                updated_context = await context_manager.get_context(context_id)
                assert isinstance(updated_context["non_list"], list)
                assert "string value" in updated_context["non_list"]
                assert "item1" in updated_context["non_list"]

    @pytest.mark.asyncio
    async def test_concurrent_context_operations(
        self, context_manager, sample_context_data
    ):
        """Test that multiple concurrent context operations work correctly."""
        # Create tasks for concurrent execution
        tasks = [
            asyncio.create_task(context_manager.create_context(sample_context_data)),
            asyncio.create_task(context_manager.create_context(sample_context_data)),
            asyncio.create_task(context_manager.create_context(sample_context_data)),
        ]

        # Wait for all tasks to complete
        context_ids = await asyncio.gather(*tasks)

        # Verify all calls succeeded
        assert len(context_ids) == 3
        assert all(isinstance(ctx_id, str) for ctx_id in context_ids)
        assert all(ctx_id in context_manager.contexts for ctx_id in context_ids)
