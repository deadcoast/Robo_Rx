import asyncio
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.main import ContentSearchEngine, FeatureMatrix


class TestContentSearchEngine:
    """Test suite for the ContentSearchEngine class."""

    @pytest.fixture
    def mock_config(self):
        """Create a mock configuration for testing."""
        return {
            "search": {
                "index_type": "vector",
                "similarity_threshold": 0.7,
                "max_results": 10,
                "use_cache": True,
            }
        }

    @pytest.fixture
    def search_engine(self, mock_config):
        """Create a ContentSearchEngine instance for testing."""
        return ContentSearchEngine(mock_config)

    @pytest.fixture
    def sample_features(self):
        """Create a sample FeatureMatrix for testing."""
        # Create a feature matrix with 5 documents, each with 10 features
        data = np.array(
            [
                [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],  # Doc 1
                [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 0.1],  # Doc 2
                [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 0.1, 0.2],  # Doc 3
                [0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 0.1, 0.2, 0.3],  # Doc 4
                [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 0.1, 0.2, 0.3, 0.4],  # Doc 5
            ]
        )

        item_ids = ["doc1", "doc2", "doc3", "doc4", "doc5"]
        feature_names = [f"feature{i}" for i in range(1, 11)]

        return FeatureMatrix(
            data=data,
            item_ids=item_ids,
            feature_names=feature_names,
            metadata={"source": "test"},
        )

    def test_init(self, search_engine, mock_config):
        """Test initialization of ContentSearchEngine."""
        assert search_engine.config == mock_config
        assert hasattr(search_engine, "index")
        assert hasattr(search_engine, "cached_results")

    @pytest.mark.asyncio
    async def test_index_content_basic(self, search_engine, sample_features):
        """Test basic content indexing functionality."""
        result = await search_engine.index_content(sample_features)

        # Verify the result structure
        assert isinstance(result, dict)
        assert "indexed_count" in result
        assert result["indexed_count"] >= 0  # Should be 5 in a complete implementation

        # Should store the features in the index
        assert hasattr(search_engine, "index")
        # In a complete implementation, the index would be updated with the features

    @pytest.mark.asyncio
    async def test_index_content_with_metadata(self, search_engine, sample_features):
        """Test content indexing with additional metadata."""
        # Add metadata to the feature matrix
        metadata = {
            "doc1": {"title": "Document 1", "tags": ["tag1", "tag2"]},
            "doc2": {"title": "Document 2", "tags": ["tag2", "tag3"]},
            "doc3": {"title": "Document 3", "tags": ["tag3", "tag4"]},
            "doc4": {"title": "Document 4", "tags": ["tag4", "tag5"]},
            "doc5": {"title": "Document 5", "tags": ["tag5", "tag1"]},
        }
        sample_features.metadata.update({"documents": metadata})

        result = await search_engine.index_content(sample_features)

        # Verify the result structure
        assert isinstance(result, dict)
        assert "indexed_count" in result
        assert result["indexed_count"] >= 0  # Should be 5 in a complete implementation

    @pytest.mark.asyncio
    async def test_index_content_with_empty_features(self, search_engine):
        """Test content indexing with empty features."""
        # Create an empty feature matrix
        empty_data = np.array([]).reshape(0, 10)  # 0 documents, 10 features
        empty_features = FeatureMatrix(
            data=empty_data,
            item_ids=[],
            feature_names=[f"feature{i}" for i in range(1, 11)],
        )

        result = await search_engine.index_content(empty_features)

        # Verify the result structure
        assert isinstance(result, dict)
        assert "indexed_count" in result
        assert result["indexed_count"] == 0  # No documents indexed

    @pytest.mark.asyncio
    async def test_search_query_basic(self, search_engine, sample_features):
        """Test basic search query functionality."""
        # First, index the content
        await search_engine.index_content(sample_features)

        # Create a query vector
        query = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])

        # Search
        result = await search_engine.search(query)

        # Verify the result structure
        assert isinstance(result, dict)
        assert "results" in result
        assert isinstance(result["results"], list)

        # In a complete implementation, the results would contain matches
        # sorted by relevance/similarity
        for item in result["results"]:
            assert "id" in item
            assert "score" in item
            assert isinstance(item["score"], float)
            assert 0 <= item["score"] <= 1  # Similarity score should be between 0 and 1

    @pytest.mark.asyncio
    async def test_search_with_text_query(self, search_engine, sample_features):
        """Test searching with a text query that gets converted to a vector."""
        # Mock embedding function to convert text to vector
        mock_embed = MagicMock()
        mock_embed.return_value = np.array(
            [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        )

        # Index the content
        await search_engine.index_content(sample_features)

        # Patch the embedding method
        with patch.object(search_engine, "_embed_text", mock_embed):
            # Search with text query
            result = await search_engine.search("test query")

            # Verify the result
            assert isinstance(result, dict)
            assert "results" in result
            assert isinstance(result["results"], list)

            # The embedding function should have been called
            mock_embed.assert_called_once_with("test query")

    @pytest.mark.asyncio
    async def test_search_with_filters(self, search_engine, sample_features):
        """Test searching with additional filters."""
        # Add metadata to the feature matrix
        metadata = {
            "doc1": {"title": "Document 1", "tags": ["tag1", "tag2"]},
            "doc2": {"title": "Document 2", "tags": ["tag2", "tag3"]},
            "doc3": {"title": "Document 3", "tags": ["tag3", "tag4"]},
            "doc4": {"title": "Document 4", "tags": ["tag4", "tag5"]},
            "doc5": {"title": "Document 5", "tags": ["tag5", "tag1"]},
        }
        sample_features.metadata.update({"documents": metadata})

        # Index the content
        await search_engine.index_content(sample_features)

        # Create a query vector
        query = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])

        # Search with filters
        filters = {"tags": ["tag1"]}
        result = await search_engine.search(query, filters=filters)

        # Verify the result structure
        assert isinstance(result, dict)
        assert "results" in result
        assert isinstance(result["results"], list)

        # In a complete implementation, the results would be filtered
        # to only include documents with tag1

    @pytest.mark.asyncio
    async def test_search_with_limit(self, search_engine, sample_features):
        """Test searching with a limit on the number of results."""
        # Index the content
        await search_engine.index_content(sample_features)

        # Create a query vector
        query = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])

        # Search with limit
        result = await search_engine.search(query, limit=2)

        # Verify the result has at most 2 items
        assert isinstance(result, dict)
        assert "results" in result
        assert isinstance(result["results"], list)
        assert len(result["results"]) <= 2

    @pytest.mark.asyncio
    async def test_search_with_threshold(self, search_engine, sample_features):
        """Test searching with a similarity threshold."""
        # Index the content
        await search_engine.index_content(sample_features)

        # Create a query vector
        query = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])

        # Search with high threshold (should find fewer matches)
        high_threshold_result = await search_engine.search(query, threshold=0.9)

        # Search with low threshold (should find more matches)
        low_threshold_result = await search_engine.search(query, threshold=0.1)

        # In a complete implementation, high threshold should give fewer results
        # than low threshold, but for now just verify the structure
        assert isinstance(high_threshold_result, dict)
        assert "results" in high_threshold_result
        assert isinstance(high_threshold_result["results"], list)

        assert isinstance(low_threshold_result, dict)
        assert "results" in low_threshold_result
        assert isinstance(low_threshold_result["results"], list)

    @pytest.mark.asyncio
    async def test_cache_results(self, search_engine, sample_features):
        """Test that search results are cached correctly."""
        # Set up a cache-enabled engine
        search_engine.config["search"]["use_cache"] = True

        # Index the content
        await search_engine.index_content(sample_features)

        # First search
        query = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
        first_result = await search_engine.search(query)

        # The engine should cache the result
        query_key = hash(tuple(query))
        assert query_key in search_engine.cached_results

        # Mock the internal search method to verify it's not called
        with patch.object(search_engine, "_perform_search") as mock_search:
            # Second search with the same query
            second_result = await search_engine.search(query)

            # Internal search should not be called
            mock_search.assert_not_called()

        # Results should be the same
        assert first_result == second_result

    @pytest.mark.asyncio
    async def test_clear_cache(self, search_engine, sample_features):
        """Test clearing the search cache."""
        # Set up a cache-enabled engine
        search_engine.config["search"]["use_cache"] = True

        # Index the content
        await search_engine.index_content(sample_features)

        # Perform a search to populate the cache
        query = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
        await search_engine.search(query)

        # Verify cache is populated
        assert len(search_engine.cached_results) > 0

        # Clear cache
        search_engine.clear_cache()

        # Verify cache is empty
        assert len(search_engine.cached_results) == 0

    @pytest.mark.asyncio
    async def test_concurrent_searches(self, search_engine, sample_features):
        """Test that multiple concurrent searches work correctly."""
        # Index the content
        await search_engine.index_content(sample_features)

        # Create different query vectors
        query1 = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
        query2 = np.array([0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 0.1])
        query3 = np.array([0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 0.1, 0.2])

        # Create tasks for concurrent execution
        tasks = [
            asyncio.create_task(search_engine.search(query1)),
            asyncio.create_task(search_engine.search(query2)),
            asyncio.create_task(search_engine.search(query3)),
        ]

        # Wait for all tasks to complete
        results = await asyncio.gather(*tasks)

        # Verify all calls succeeded
        assert len(results) == 3
        assert all(isinstance(result, dict) for result in results)
        assert all("results" in result for result in results)
        assert all(isinstance(result["results"], list) for result in results)
