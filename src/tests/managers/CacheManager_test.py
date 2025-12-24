import asyncio
import time
from contextlib import suppress
from unittest.mock import AsyncMock, MagicMock, Mock, patch
import tempfile

import pytest

from src.managers.CacheManager import CacheManager


class TestCacheManager:
    """Test suite for the CacheManager class."""

    @pytest.fixture
    def mock_config(self):
        """Create a mock configuration for testing."""
        return {
            "cache": {
                "max_size": 1000,
                "default_ttl": 3600,  # 1 hour in seconds
                "enable_persistence": True,
                "persistence_path": tempfile.mkdtemp(prefix="test_cache_"),
                "cleanup_interval": 300,  # 5 minutes in seconds
            }
        }

    @pytest.fixture
    def cache_manager(self, mock_config):
        """Create a CacheManager instance for testing."""
        return CacheManager(mock_config)

    def test_init(self, cache_manager, mock_config):
        """Test initialization of CacheManager."""
        assert cache_manager.config == mock_config
        assert hasattr(cache_manager, "cache")
        assert isinstance(cache_manager.cache, dict)
        assert len(cache_manager.cache) == 0

    @pytest.mark.asyncio
    async def test_set_and_get(self, cache_manager):
        """Test setting and getting cache items."""
        # Set a cache item
        key = "test_key"
        value = {"data": "test_value"}
        ttl = 3600  # 1 hour

        await cache_manager.set(key, value, ttl)

        # Verify the item was cached
        assert key in cache_manager.cache
        assert "value" in cache_manager.cache[key]
        assert cache_manager.cache[key]["value"] == value

        # Verify TTL was set
        assert "expires_at" in cache_manager.cache[key]
        expires_at = cache_manager.cache[key]["expires_at"]
        now = time.time()
        assert expires_at > now
        assert expires_at <= now + ttl + 1  # Allow for slight timing differences

        # Get the cached item
        retrieved_value = await cache_manager.get(key)

        # Verify the retrieved value matches the original
        assert retrieved_value == value

    @pytest.mark.asyncio
    async def test_get_nonexistent_key(self, cache_manager):
        """Test getting a key that doesn't exist in the cache."""
        # Get a nonexistent key
        value = await cache_manager.get("nonexistent_key")

        # Should return None
        assert value is None

    @pytest.mark.asyncio
    async def test_get_expired_key(self, cache_manager):
        """Test getting a key that has expired."""
        # Set a cache item with a very short TTL
        key = "expired_key"
        value = {"data": "will_expire"}
        ttl = 0.1  # 100 milliseconds

        await cache_manager.set(key, value, ttl)

        # Verify the item was cached
        assert key in cache_manager.cache

        # Wait for the TTL to expire
        await asyncio.sleep(0.2)  # 200 milliseconds

        # Get the expired key
        retrieved_value = await cache_manager.get(key)

        # Should return None
        assert retrieved_value is None

        # The key should be removed from the cache
        assert key not in cache_manager.cache

    @pytest.mark.asyncio
    async def test_set_existing_key(self, cache_manager):
        """Test setting a key that already exists."""
        # Set a cache item
        key = "test_key"
        value1 = {"data": "original_value"}
        await cache_manager.set(key, value1)

        # Set the same key with a different value
        value2 = {"data": "updated_value"}
        await cache_manager.set(key, value2)

        # Get the cached item
        retrieved_value = await cache_manager.get(key)

        # Should return the updated value
        assert retrieved_value == value2

    @pytest.mark.asyncio
    async def test_delete(self, cache_manager):
        """Test deleting a cache item."""
        # Set a cache item
        key = "test_key"
        value = {"data": "test_value"}
        await cache_manager.set(key, value)

        # Verify the item was cached
        assert key in cache_manager.cache

        # Delete the item
        await cache_manager.delete(key)

        # Verify the item was removed
        assert key not in cache_manager.cache

        # Try to get the deleted item
        retrieved_value = await cache_manager.get(key)
        assert retrieved_value is None

    @pytest.mark.asyncio
    async def test_delete_nonexistent_key(self, cache_manager):
        """Test deleting a key that doesn't exist."""
        # Try to delete a nonexistent key
        result = await cache_manager.delete("nonexistent_key")

        # Should return False or None
        assert not result

    @pytest.mark.asyncio
    async def test_clear(self, cache_manager):
        """Test clearing the entire cache."""
        # Set multiple cache items
        await cache_manager.set("key1", "value1")
        await cache_manager.set("key2", "value2")
        await cache_manager.set("key3", "value3")

        # Verify items were cached
        assert len(cache_manager.cache) == 3

        # Clear the cache
        await cache_manager.clear()

        # Verify all items were removed
        assert len(cache_manager.cache) == 0

    @pytest.mark.asyncio
    async def test_exists(self, cache_manager):
        """Test checking if a key exists in the cache."""
        # Set a cache item
        key = "test_key"
        value = {"data": "test_value"}
        await cache_manager.set(key, value)

        # Check if the key exists
        exists = await cache_manager.exists(key)

        # Should return True
        assert exists is True

        # Check if a nonexistent key exists
        exists = await cache_manager.exists("nonexistent_key")

        # Should return False
        assert exists is False

    @pytest.mark.asyncio
    async def test_exists_with_expired_key(self, cache_manager):
        """Test checking if an expired key exists."""
        # Set a cache item with a short TTL
        key = "expired_key"
        value = {"data": "will_expire"}
        ttl = 0.1  # 100 milliseconds

        await cache_manager.set(key, value, ttl)

        # Wait for the TTL to expire
        await asyncio.sleep(0.2)  # 200 milliseconds

        # Check if the key exists
        exists = await cache_manager.exists(key)

        # Should return False
        assert exists is False

    @pytest.mark.asyncio
    async def test_clean_expired(self, cache_manager):
        """Test cleaning expired cache items."""
        # Set some items with different TTLs
        await cache_manager.set("key1", "value1", 0.1)  # Will expire quickly
        await cache_manager.set("key2", "value2", 10)  # Won't expire during test
        await cache_manager.set("key3", "value3", 0.2)  # Will expire quickly

        # Wait for some items to expire
        await asyncio.sleep(0.3)  # 300 milliseconds

        # Clean expired items
        cleaned_count = await cache_manager.clean_expired()

        # Should have cleaned 2 items
        assert cleaned_count == 2

        # Verify expired items were removed
        assert "key1" not in cache_manager.cache
        assert "key2" in cache_manager.cache
        assert "key3" not in cache_manager.cache

    @pytest.mark.asyncio
    async def test_get_with_default(self, cache_manager):
        """Test getting a cache item with a default value."""
        # Get a nonexistent key with a default value
        default_value = {"default": "value"}
        retrieved_value = await cache_manager.get(
            "nonexistent_key", default=default_value
        )

        # Should return the default value
        assert retrieved_value == default_value

    @pytest.mark.asyncio
    async def test_get_many(self, cache_manager):
        """Test getting multiple cache items at once."""
        # Set multiple cache items
        await cache_manager.set("key1", "value1")
        await cache_manager.set("key2", "value2")
        await cache_manager.set("key3", "value3")

        # Get multiple items
        keys = ["key1", "key2", "nonexistent_key"]
        values = await cache_manager.get_many(keys)

        # Should return values for existing keys and None for nonexistent keys
        assert len(values) == 3
        assert values["key1"] == "value1"
        assert values["key2"] == "value2"
        assert values["nonexistent_key"] is None

    @pytest.mark.asyncio
    async def test_set_many(self, cache_manager):
        """Test setting multiple cache items at once."""
        # Set multiple items
        items = {"key1": "value1", "key2": "value2", "key3": "value3"}
        ttl = 3600

        await cache_manager.set_many(items, ttl)

        # Verify all items were cached
        assert "key1" in cache_manager.cache
        assert "key2" in cache_manager.cache
        assert "key3" in cache_manager.cache

        # Verify values
        assert (await cache_manager.get("key1")) == "value1"
        assert (await cache_manager.get("key2")) == "value2"
        assert (await cache_manager.get("key3")) == "value3"

    @pytest.mark.asyncio
    async def test_delete_many(self, cache_manager):
        """Test deleting multiple cache items at once."""
        # Set multiple cache items
        await cache_manager.set("key1", "value1")
        await cache_manager.set("key2", "value2")
        await cache_manager.set("key3", "value3")

        # Delete multiple items
        keys = ["key1", "key2", "nonexistent_key"]
        deleted_count = await cache_manager.delete_many(keys)

        # Should return the number of keys actually deleted
        assert deleted_count == 2

        # Verify items were removed
        assert "key1" not in cache_manager.cache
        assert "key2" not in cache_manager.cache
        assert "key3" in cache_manager.cache

    @pytest.mark.asyncio
    async def test_increment(self, cache_manager):
        """Test incrementing a numeric cache value."""
        # Set a numeric cache value
        key = "counter"
        await cache_manager.set(key, 5)

        # Increment the value
        new_value = await cache_manager.increment(key)

        # Should be incremented by 1
        assert new_value == 6

        # Verify the cached value was updated
        assert (await cache_manager.get(key)) == 6

        # Increment with a custom amount
        new_value = await cache_manager.increment(key, 3)

        # Should be incremented by the specified amount
        assert new_value == 9
        assert (await cache_manager.get(key)) == 9

    @pytest.mark.asyncio
    async def test_increment_nonexistent_key(self, cache_manager):
        """Test incrementing a nonexistent key."""
        # Increment a nonexistent key
        new_value = await cache_manager.increment("nonexistent_key")

        # Should initialize to 1
        assert new_value == 1
        assert (await cache_manager.get("nonexistent_key")) == 1

    @pytest.mark.asyncio
    async def test_increment_non_numeric(self, cache_manager):
        """Test incrementing a non-numeric value."""
        # Set a non-numeric cache value
        key = "non_numeric"
        await cache_manager.set(key, "string")

        # Try to increment the value
        # Using suppress to silence TypeError/ValueError which are acceptable here
        with suppress(TypeError, ValueError):
            await cache_manager.increment(key)
            # If it doesn't raise an exception, it should have replaced the value
            assert isinstance(await cache_manager.get(key), int)
            pass

    @pytest.mark.asyncio
    async def test_decrement(self, cache_manager):
        """Test decrementing a numeric cache value."""
        # Set a numeric cache value
        key = "counter"
        await cache_manager.set(key, 10)

        # Decrement the value
        new_value = await cache_manager.decrement(key)

        # Should be decremented by 1
        assert new_value == 9

        # Verify the cached value was updated
        assert (await cache_manager.get(key)) == 9

        # Decrement with a custom amount
        new_value = await cache_manager.decrement(key, 3)

        # Should be decremented by the specified amount
        assert new_value == 6
        assert (await cache_manager.get(key)) == 6

    @pytest.mark.asyncio
    async def test_get_or_set(self, cache_manager):
        """Test get_or_set functionality."""

        # Define a function to compute the value if not in cache
        def compute_value():
            return "computed_value"

        # Get a nonexistent key
        value = await cache_manager.get_or_set("new_key", compute_value)

        # Should call compute_value and cache the result
        assert value == "computed_value"
        assert (await cache_manager.get("new_key")) == "computed_value"

        # Mock compute_value to verify it's not called for cached keys
        mock_compute = Mock(return_value="should_not_be_used")

        # Get an existing key
        value = await cache_manager.get_or_set("new_key", mock_compute)

        # Should return the cached value without calling compute_value
        assert value == "computed_value"
        mock_compute.assert_not_called()

    @pytest.mark.asyncio
    async def test_keys(self, cache_manager):
        """Test getting all keys in the cache."""
        # Set multiple cache items
        await cache_manager.set("key1", "value1")
        await cache_manager.set("key2", "value2")
        await cache_manager.set("key3", "value3")

        # Get all keys
        keys = await cache_manager.keys()

        # Should return all keys
        assert len(keys) == 3
        assert "key1" in keys
        assert "key2" in keys
        assert "key3" in keys

    @pytest.mark.asyncio
    async def test_values(self, cache_manager):
        """Test getting all values in the cache."""
        # Set multiple cache items
        await cache_manager.set("key1", "value1")
        await cache_manager.set("key2", "value2")
        await cache_manager.set("key3", "value3")

        # Get all values
        values = await cache_manager.values()

        # Should return all values
        assert len(values) == 3
        assert "value1" in values
        assert "value2" in values
        assert "value3" in values

    @pytest.mark.asyncio
    async def test_items(self, cache_manager):
        """Test getting all items in the cache."""
        # Set multiple cache items
        await cache_manager.set("key1", "value1")
        await cache_manager.set("key2", "value2")
        await cache_manager.set("key3", "value3")

        # Get all items
        items = await cache_manager.items()

        # Should return all items as (key, value) pairs
        assert len(items) == 3
        assert ("key1", "value1") in items
        assert ("key2", "value2") in items
        assert ("key3", "value3") in items

    @pytest.mark.asyncio
    async def test_contains(self, cache_manager):
        """Test checking if the cache contains a key using the 'in' operator."""
        # Set a cache item
        await cache_manager.set("test_key", "test_value")

        # Check if the key exists
        result = await cache_manager.contains("test_key")

        # Should return True
        assert result is True

        # Check if a nonexistent key exists
        result = await cache_manager.contains("nonexistent_key")

        # Should return False
        assert result is False

    @pytest.mark.asyncio
    async def test_ttl(self, cache_manager):
        """Test getting the remaining TTL for a key."""
        # Set a cache item with TTL
        key = "test_key"
        ttl = 60  # 60 seconds
        await cache_manager.set(key, "value", ttl)

        # Get the remaining TTL
        remaining = await cache_manager.ttl(key)

        # Should be close to the original TTL
        assert remaining <= ttl
        assert remaining > ttl - 2  # Allow for slight timing differences

        # Get TTL for a nonexistent key
        remaining = await cache_manager.ttl("nonexistent_key")

        # Should return None or -1
        assert remaining is None or remaining == -1

    @patch("builtins.open", new_callable=MagicMock)
    @pytest.mark.asyncio
    async def test_save_to_disk(self, mock_open, cache_manager):
        """Test saving the cache to disk."""
        # Set some cache items
        await cache_manager.set("key1", "value1")
        await cache_manager.set("key2", "value2")

        # Configure the mock to capture the written data
        mock_file = MagicMock()
        mock_open.return_value.__enter__.return_value = mock_file

        # Save the cache
        result = await cache_manager.save_to_disk()

        # Verify the result
        assert result is True

        # Verify the file was opened for writing
        mock_open.assert_called_once()

        # Verify write was called
        assert mock_file.write.called

    @patch("builtins.open", new_callable=MagicMock)
    @patch("os.path.exists")
    @pytest.mark.asyncio
    async def test_load_from_disk(self, mock_exists, mock_open, cache_manager):
        """Test loading the cache from disk."""
        # Mock file existence
        mock_exists.return_value = True

        # Create sample cache data
        import json

        cache_data = {
            "key1": {"value": "value1", "expires_at": time.time() + 3600},
            "key2": {"value": "value2", "expires_at": time.time() + 3600},
        }

        # Configure the mock to return the sample data
        mock_file = MagicMock()
        mock_file.read.return_value = json.dumps(cache_data)
        mock_open.return_value.__enter__.return_value = mock_file

        # Load the cache
        result = await cache_manager.load_from_disk()

        # Verify the result
        assert result is True

        # Verify the file was opened for reading
        mock_open.assert_called_once()

        # Verify the cache was loaded
        assert "key1" in cache_manager.cache
        assert "key2" in cache_manager.cache
        assert cache_manager.cache["key1"]["value"] == "value1"
        assert cache_manager.cache["key2"]["value"] == "value2"

    @pytest.mark.asyncio
    async def test_background_cleanup(self, cache_manager):
        """Test background cleanup of expired items."""
        # Mock the clean_expired method
        cache_manager.clean_expired = AsyncMock(return_value=2)

        # Start background cleanup
        task = cache_manager.start_background_cleanup()

        # Wait a bit to allow the task to run
        await asyncio.sleep(0.1)

        # Cancel the task to stop the cleanup
        task.cancel()

        # Using suppress to silence CancelledError which we expect when cancelling the task
        with suppress(asyncio.CancelledError):
            await task

        # Verify clean_expired was called at least once
        assert cache_manager.clean_expired.called
