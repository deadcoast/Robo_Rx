import asyncio
import os

import numpy as np
import pytest

from src.main import VectorStore


class TestVectorStore:
    """Test suite for the VectorStore class."""

    @pytest.fixture
    def mock_config(self):
        """Create a mock configuration for testing."""
        return {
            "vector_store": {
                "storage_path": "/tmp/test_vector_store",
                "index_type": "flat",  # flat, hnsw, ivf, etc.
                "dimensions": 768,
                "metric": "cosine",  # cosine, euclidean, dot
                "auto_save": True,
                "save_interval": 60,  # seconds
            }
        }

    @pytest.fixture
    def vector_store(self, mock_config):
        """Create a VectorStore instance for testing."""
        store = VectorStore(mock_config)
        # Clean up any existing index
        if os.path.exists(mock_config["vector_store"]["storage_path"]):
            store.reset()
        return store

    @pytest.fixture
    def sample_vectors(self):
        """Create sample vectors for testing."""
        # Create 5 random vectors with 768 dimensions
        rng = np.random.RandomState(42)  # For reproducibility
        vectors = rng.rand(5, 768).astype(np.float32)
        # Normalize the vectors for cosine similarity
        vectors = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
        return vectors

    @pytest.fixture
    def sample_metadata(self):
        """Create sample metadata for testing."""
        return [
            {"id": "doc1", "text": "Sample document 1", "tags": ["tag1", "tag2"]},
            {"id": "doc2", "text": "Sample document 2", "tags": ["tag2", "tag3"]},
            {"id": "doc3", "text": "Sample document 3", "tags": ["tag3", "tag4"]},
            {"id": "doc4", "text": "Sample document 4", "tags": ["tag4", "tag5"]},
            {"id": "doc5", "text": "Sample document 5", "tags": ["tag5", "tag1"]},
        ]

    def test_init(self, vector_store, mock_config):
        """Test initialization of VectorStore."""
        assert vector_store.config == mock_config
        assert vector_store.dimensions == mock_config["vector_store"]["dimensions"]
        assert vector_store.metric == mock_config["vector_store"]["metric"]
        assert hasattr(vector_store, "index")

    @pytest.mark.asyncio
    async def test_add_vectors(self, vector_store, sample_vectors, sample_metadata):
        """Test adding vectors to the store."""
        # Add vectors with metadata
        ids = [f"id{i}" for i in range(len(sample_vectors))]
        await vector_store.add_vectors(sample_vectors, ids, sample_metadata)

        # Verify the vectors were added
        assert vector_store.count() == len(sample_vectors)

        # Internal structure can vary by implementation, so just verify count is correct

    @pytest.mark.asyncio
    async def test_add_single_vector(self, vector_store):
        """Test adding a single vector to the store."""
        # Create a single vector
        vector = np.random.rand(768).astype(np.float32)
        vector = vector / np.linalg.norm(vector)

        # Add the vector
        vector_id = "single_test"
        metadata = {"text": "Single test vector"}
        await vector_store.add_vector(vector, vector_id, metadata)

        # Verify the vector was added
        assert vector_store.count() == 1

        # Try to retrieve it
        result = await vector_store.get_vector(vector_id)

        # Verify the retrieval
        assert result is not None
        assert "vector" in result
        assert "metadata" in result
        assert np.array_equal(result["vector"], vector)
        assert result["metadata"] == metadata

    @pytest.mark.asyncio
    async def test_search_vectors(self, vector_store, sample_vectors, sample_metadata):
        """Test searching for similar vectors."""
        # Add vectors
        ids = [f"id{i}" for i in range(len(sample_vectors))]
        await vector_store.add_vectors(sample_vectors, ids, sample_metadata)

        # Search using the first vector as query
        query_vector = sample_vectors[0]
        results = await vector_store.search(query_vector, k=3)

        # Verify the results
        assert isinstance(results, list)
        assert len(results) == 3  # Requested top 3 results

        # First result should be the query vector itself
        assert results[0]["id"] == "id0"
        assert results[0]["score"] >= 0.99  # Should be almost 1.0 (self-similarity)

        # All results should have correct structure
        for result in results:
            assert "id" in result
            assert "score" in result
            assert "metadata" in result
            assert 0 <= result["score"] <= 1.0  # Similarity score between 0 and 1

    @pytest.mark.asyncio
    async def test_search_with_filter(
        self, vector_store, sample_vectors, sample_metadata
    ):
        """Test searching with metadata filters."""
        # Add vectors
        ids = [f"id{i}" for i in range(len(sample_vectors))]
        await vector_store.add_vectors(sample_vectors, ids, sample_metadata)

        # Search with a filter (only documents with tag "tag1")
        query_vector = sample_vectors[0]

        def filter_func(meta):
            return "tag1" in meta["tags"]

        results = await vector_store.search(query_vector, k=5, filter_func=filter_func)

        # Verify the results are filtered correctly
        assert all("tag1" in result["metadata"]["tags"] for result in results)

        # With this filter, only 2 documents should match (doc1 and doc5)
        assert len(results) == 2

    @pytest.mark.asyncio
    async def test_delete_vectors(self, vector_store, sample_vectors, sample_metadata):
        """Test deleting vectors from the store."""
        # Add vectors
        ids = [f"id{i}" for i in range(len(sample_vectors))]
        await vector_store.add_vectors(sample_vectors, ids, sample_metadata)

        # Verify initial count
        assert vector_store.count() == len(sample_vectors)

        # Delete some vectors
        await vector_store.delete_vectors(["id0", "id2"])

        # Verify the vectors were deleted
        assert vector_store.count() == len(sample_vectors) - 2

        # Search should not return deleted vectors
        query_vector = sample_vectors[0]
        results = await vector_store.search(query_vector, k=5)
        result_ids = [result["id"] for result in results]

        assert "id0" not in result_ids
        assert "id2" not in result_ids

    @pytest.mark.asyncio
    async def test_update_vector(self, vector_store, sample_vectors, sample_metadata):
        """Test updating a vector's metadata or value."""
        # Add vectors
        ids = [f"id{i}" for i in range(len(sample_vectors))]
        await vector_store.add_vectors(sample_vectors, ids, sample_metadata)

        # Update metadata for id1
        updated_metadata = sample_metadata[1].copy()
        updated_metadata["text"] = "Updated document text"
        updated_metadata["new_field"] = "new value"

        await vector_store.update_metadata("id1", updated_metadata)

        # Verify the metadata was updated
        result = await vector_store.get_vector("id1")
        assert result["metadata"]["text"] == "Updated document text"
        assert result["metadata"]["new_field"] == "new value"

        # Update the vector value itself
        new_vector = np.random.rand(768).astype(np.float32)
        new_vector = new_vector / np.linalg.norm(new_vector)

        await vector_store.update_vector("id1", new_vector)

        # Verify the vector was updated
        result = await vector_store.get_vector("id1")
        assert np.array_equal(result["vector"], new_vector)

    @pytest.mark.asyncio
    async def test_save_and_load(self, vector_store, sample_vectors, sample_metadata):
        """Test saving and loading the vector store."""
        # Add vectors
        ids = [f"id{i}" for i in range(len(sample_vectors))]
        await vector_store.add_vectors(sample_vectors, ids, sample_metadata)

        # Save the index
        save_path = "/tmp/test_vector_store_save"
        await vector_store.save(save_path)

        # Create a new store and load the saved index
        new_store = VectorStore(vector_store.config)
        await new_store.load(save_path)

        # Verify the loaded store has the same vectors
        assert new_store.count() == vector_store.count()

        # Verify search works on the loaded store
        query_vector = sample_vectors[0]
        results = await new_store.search(query_vector, k=1)

        assert results[0]["id"] == "id0"
        assert results[0]["score"] >= 0.99  # Self-similarity should be high

        # Clean up
        import shutil

        if os.path.exists(save_path):
            shutil.rmtree(save_path)

    @pytest.mark.asyncio
    async def test_reset(self, vector_store, sample_vectors, sample_metadata):
        """Test resetting the vector store."""
        # Add vectors
        ids = [f"id{i}" for i in range(len(sample_vectors))]
        await vector_store.add_vectors(sample_vectors, ids, sample_metadata)

        # Verify initial count
        assert vector_store.count() == len(sample_vectors)

        # Reset the store
        await vector_store.reset()

        # Verify the store is empty
        assert vector_store.count() == 0

    @pytest.mark.asyncio
    async def test_batch_search(self, vector_store, sample_vectors, sample_metadata):
        """Test searching with multiple query vectors."""
        # Add vectors
        ids = [f"id{i}" for i in range(len(sample_vectors))]
        await vector_store.add_vectors(sample_vectors, ids, sample_metadata)

        # Create query vectors (first 2 from sample)
        query_vectors = sample_vectors[:2]

        # Batch search
        results = await vector_store.batch_search(query_vectors, k=2)

        # Verify the results
        assert isinstance(results, list)
        assert len(results) == len(query_vectors)

        # Each entry should contain results for one query
        for i, query_results in enumerate(results):
            assert isinstance(query_results, list)
            assert len(query_results) == 2  # top k=2 results
            # First result should match the query vector
            assert query_results[0]["id"] == f"id{i}"
            assert query_results[0]["score"] >= 0.99  # Self-similarity

    @pytest.mark.asyncio
    async def test_get_all_vectors(self, vector_store, sample_vectors, sample_metadata):
        """Test retrieving all vectors."""
        # Add vectors
        ids = [f"id{i}" for i in range(len(sample_vectors))]
        await vector_store.add_vectors(sample_vectors, ids, sample_metadata)

        # Get all vectors
        all_vectors = await vector_store.get_all_vectors()

        # Verify all vectors were retrieved
        assert len(all_vectors) == len(sample_vectors)

        # Each entry should have the correct structure
        for entry in all_vectors:
            assert "id" in entry
            assert "vector" in entry
            assert "metadata" in entry
            assert isinstance(entry["vector"], np.ndarray)
            assert entry["vector"].shape == (vector_store.dimensions,)

    @pytest.mark.asyncio
    async def test_get_metadata(self, vector_store, sample_vectors, sample_metadata):
        """Test retrieving metadata only."""
        # Add vectors
        ids = [f"id{i}" for i in range(len(sample_vectors))]
        await vector_store.add_vectors(sample_vectors, ids, sample_metadata)

        # Get metadata for a specific ID
        metadata = await vector_store.get_metadata("id2")

        # Verify the metadata
        assert metadata == sample_metadata[2]

        # Try with a non-existent ID
        nonexistent_metadata = await vector_store.get_metadata("nonexistent")
        assert nonexistent_metadata is None

    @pytest.mark.asyncio
    async def test_bulk_operations(self, vector_store):
        """Test bulk operations for efficiency."""
        # Create a larger set of vectors
        large_vectors = np.random.rand(100, 768).astype(np.float32)
        large_vectors = large_vectors / np.linalg.norm(
            large_vectors, axis=1, keepdims=True
        )

        # Create IDs and metadata
        ids = [f"bulk{i}" for i in range(100)]
        metadata = [{"id": f"bulk{i}", "index": i} for i in range(100)]

        # Test bulk add
        start_time = asyncio.get_event_loop().time()
        await vector_store.add_vectors(large_vectors, ids, metadata)
        asyncio.get_event_loop().time() - start_time

        # Verify all vectors were added
        assert vector_store.count() == 100

        # Test bulk search
        query_vectors = large_vectors[:10]  # First 10 vectors

        start_time = asyncio.get_event_loop().time()
        batch_results = await vector_store.batch_search(query_vectors, k=5)
        batch_time = asyncio.get_event_loop().time() - start_time

        # Verify batch search results
        assert len(batch_results) == 10
        assert all(len(results) == 5 for results in batch_results)

        # Compare with individual searches
        individual_times = []
        for query_vector in query_vectors:
            start_time = asyncio.get_event_loop().time()
            await vector_store.search(query_vector, k=5)
            individual_times.append(asyncio.get_event_loop().time() - start_time)

        total_individual_time = sum(individual_times)

        # Batch operations should generally be more efficient than individual ones
        # This is not a strict requirement but a performance guideline
        print(
            f"Batch search time: {batch_time}, Total individual search time: {total_individual_time}"
        )

        # Clean up
        await vector_store.reset()

    @pytest.mark.asyncio
    async def test_concurrent_operations(self, vector_store, sample_vectors):
        """Test concurrent operations on the vector store."""
        # Add initial vectors
        ids = [f"id{i}" for i in range(len(sample_vectors))]
        metadata = [{"id": f"id{i}"} for i in range(len(sample_vectors))]
        await vector_store.add_vectors(sample_vectors, ids, metadata)

        # Create tasks for concurrent operations
        search_task = asyncio.create_task(vector_store.search(sample_vectors[0], k=3))

        # Create a new vector to add
        new_vector = np.random.rand(768).astype(np.float32)
        new_vector = new_vector / np.linalg.norm(new_vector)

        add_task = asyncio.create_task(
            vector_store.add_vector(new_vector, "concurrent_add", {"test": True})
        )

        delete_task = asyncio.create_task(vector_store.delete_vectors(["id1"]))

        # Wait for all tasks to complete
        search_result, _, _ = await asyncio.gather(search_task, add_task, delete_task)

        # Verify final state after concurrent operations
        assert vector_store.count() == len(sample_vectors)  # +1 added, -1 deleted

        # Verify the search completed successfully
        assert len(search_result) == 3

        # Verify the added vector exists
        added_result = await vector_store.get_vector("concurrent_add")
        assert added_result is not None
        assert added_result["metadata"]["test"] is True

        # Verify the deleted vector is gone
        deleted_result = await vector_store.get_vector("id1")
        assert deleted_result is None

    @pytest.mark.asyncio
    async def test_different_metrics(self):
        """Test vector store with different distance metrics."""
        # Test with different metrics
        for metric in ["cosine", "euclidean", "dot"]:
            config = {
                "vector_store": {
                    "storage_path": f"/tmp/test_vector_store_{metric}",
                    "index_type": "flat",
                    "dimensions": 768,
                    "metric": metric,
                    "auto_save": False,
                }
            }

            # Create a store with this metric
            store = VectorStore(config)

            # Create some vectors
            vectors = np.random.rand(5, 768).astype(np.float32)
            if metric in ["cosine", "dot"]:
                # Normalize for cosine and dot product
                vectors = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)

            # Add vectors
            ids = [f"{metric}_{i}" for i in range(5)]
            metadata = [{"metric": metric, "id": ids[i]} for i in range(5)]

            await store.add_vectors(vectors, ids, metadata)

            # Search with the first vector
            query_vector = vectors[0]
            results = await store.search(query_vector, k=5)

            # Verify results
            assert len(results) == 5

            # For cosine and dot, first result should be the query vector
            # For euclidean, smallest distance is to self
            assert results[0]["id"] == f"{metric}_0"

            # Clean up
            await store.reset()

    @pytest.mark.asyncio
    async def test_custom_serialization(
        self, vector_store, sample_vectors, sample_metadata
    ):
        """Test custom serialization and deserialization of metadata."""
        # Create metadata with custom Python objects that need special serialization
        from datetime import datetime

        custom_metadata = sample_metadata.copy()
        for meta in custom_metadata:
            meta["timestamp"] = datetime.now()
            meta["complex"] = {"nested": {"value": 42}}

        # Add vectors with custom metadata
        ids = [f"id{i}" for i in range(len(sample_vectors))]
        await vector_store.add_vectors(sample_vectors, ids, custom_metadata)

        # Save the index
        save_path = "/tmp/test_custom_serialization"
        await vector_store.save(save_path)

        # Load the index in a new store
        new_store = VectorStore(vector_store.config)
        await new_store.load(save_path)

        # Retrieve and check metadata
        retrieved_meta = await new_store.get_metadata("id0")

        # DateTime objects might be serialized as strings
        assert "timestamp" in retrieved_meta
        assert "complex" in retrieved_meta
        assert retrieved_meta["complex"]["nested"]["value"] == 42

        # Clean up
        import shutil

        if os.path.exists(save_path):
            shutil.rmtree(save_path)

    @pytest.mark.asyncio
    async def test_large_dimension_vectors(self):
        """Test with very large dimension vectors."""
        # Create a store for large vectors
        config = {
            "vector_store": {
                "storage_path": "/tmp/test_large_vectors",
                "index_type": "flat",
                "dimensions": 1536,  # OpenAI's text-embedding-ada-002 size
                "metric": "cosine",
                "auto_save": False,
            }
        }

        store = VectorStore(config)

        # Create large dimension vectors
        large_vectors = np.random.rand(3, 1536).astype(np.float32)
        large_vectors = large_vectors / np.linalg.norm(
            large_vectors, axis=1, keepdims=True
        )

        # Add vectors
        ids = ["large1", "large2", "large3"]
        metadata = [{"id": id} for id in ids]

        await store.add_vectors(large_vectors, ids, metadata)

        # Search
        results = await store.search(large_vectors[0], k=3)

        # Verify
        assert len(results) == 3
        assert results[0]["id"] == "large1"

        # Clean up
        await store.reset()
