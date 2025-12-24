import asyncio

import numpy as np
import pytest

from src.main import EmbeddingEngine


class TestEmbeddingEngine:
    """Test suite for the EmbeddingEngine class."""

    @pytest.fixture
    def mock_config(self):
        """Create a mock configuration for testing."""
        return {
            "embedding": {
                "model": "bert-base-uncased",
                "dimensions": 768,
                "pooling_method": "mean",
                "enable_batching": True,
                "batch_size": 16,
                "device": "cpu",
            }
        }

    @pytest.fixture
    def embedding_engine(self, mock_config):
        """Create an EmbeddingEngine instance for testing."""
        return EmbeddingEngine(mock_config)

    @pytest.fixture
    def sample_texts(self):
        """Create sample texts for testing."""
        return [
            "This is the first sample text for embedding generation.",
            "A second sample with different content and length.",
            "Third sample text that contains some technical terms like BERT and embeddings.",
            "Short text.",
            "This is a longer piece of text that should contain more semantic information for the embedding model to work with. It includes multiple sentences and a rich vocabulary to ensure comprehensive vector representation.",
        ]

    def test_init(self, embedding_engine, mock_config):
        """Test initialization of EmbeddingEngine."""
        assert embedding_engine.config == mock_config
        assert hasattr(embedding_engine, "model")
        assert hasattr(embedding_engine, "tokenizer")
        assert embedding_engine.dimensions == mock_config["embedding"]["dimensions"]
        assert (
            embedding_engine.pooling_method
            == mock_config["embedding"]["pooling_method"]
        )

    @pytest.mark.asyncio
    async def test_embed_text(self, embedding_engine, sample_texts):
        """Test embedding generation for a single text."""
        # Get embedding for a single text
        text = sample_texts[0]
        embedding = await embedding_engine.embed_text(text)

        # Verify the embedding has the correct shape
        assert isinstance(embedding, np.ndarray)
        assert embedding.shape == (embedding_engine.dimensions,)

        # Verify the embedding is normalized (if that's part of the implementation)
        # This assumes L2 normalization is used
        norm = np.linalg.norm(embedding)
        assert np.isclose(norm, 1.0, atol=1e-5) or norm <= 1.0

    @pytest.mark.asyncio
    async def test_embed_texts(self, embedding_engine, sample_texts):
        """Test embedding generation for multiple texts."""
        # Get embeddings for multiple texts
        embeddings = await embedding_engine.embed_texts(sample_texts)

        # Verify the embeddings have the correct shape
        assert isinstance(embeddings, list)
        assert len(embeddings) == len(sample_texts)

        for embedding in embeddings:
            assert isinstance(embedding, np.ndarray)
            assert embedding.shape == (embedding_engine.dimensions,)

            # Verify normalization
            norm = np.linalg.norm(embedding)
            assert np.isclose(norm, 1.0, atol=1e-5) or norm <= 1.0

    @pytest.mark.asyncio
    async def test_embed_empty_text(self, embedding_engine):
        """Test embedding generation for empty text."""
        # Get embedding for empty text
        empty_text = ""

        try:
            embedding = await embedding_engine.embed_text(empty_text)

            # If it doesn't raise an exception, verify the embedding has the correct shape
            assert isinstance(embedding, np.ndarray)
            assert embedding.shape == (embedding_engine.dimensions,)
        except ValueError as e:
            # It's also acceptable to raise a ValueError for empty input
            assert "empty text" in str(e).lower() or "no text" in str(e).lower()

    @pytest.mark.asyncio
    async def test_embed_long_text(self, embedding_engine):
        """Test embedding generation for very long text."""
        # Create a long text (likely to exceed model token limits)
        long_text = "word " * 2000  # About 2000 words

        # Get embedding for long text
        embedding = await embedding_engine.embed_text(long_text)

        # Verify the embedding has the correct shape, regardless of input length
        assert isinstance(embedding, np.ndarray)
        assert embedding.shape == (embedding_engine.dimensions,)

    @pytest.mark.asyncio
    async def test_embed_texts_with_batch_processing(
        self, embedding_engine, sample_texts
    ):
        """Test batch processing of multiple texts."""
        # Set a small batch size to force multiple batches
        embedding_engine.batch_size = 2

        # Mock the internal batch processing method
        original_process_batch = embedding_engine._process_batch
        batch_calls = []

        async def mock_process_batch(batch):
            batch_calls.append(batch)
            return await original_process_batch(batch)

        embedding_engine._process_batch = mock_process_batch

        # Get embeddings for multiple texts
        embeddings = await embedding_engine.embed_texts(sample_texts)

        # Verify batch processing was used
        assert len(batch_calls) == 3  # 5 texts with batch size 2 should need 3 batches
        assert len(batch_calls[0]) == 2
        assert len(batch_calls[1]) == 2
        assert len(batch_calls[2]) == 1

        # Verify correct results
        assert len(embeddings) == len(sample_texts)

    @pytest.mark.asyncio
    async def test_compute_similarity(self, embedding_engine, sample_texts):
        """Test computing similarity between embeddings."""
        # Generate embeddings
        embeddings = await embedding_engine.embed_texts(sample_texts)

        # Compute similarity between the first and second embeddings
        similarity = embedding_engine.compute_similarity(embeddings[0], embeddings[1])

        # Verify the similarity is a float between -1 and 1
        assert isinstance(similarity, float)
        assert -1.0 <= similarity <= 1.0

        # Similarity with itself should be 1.0 (or very close)
        self_similarity = embedding_engine.compute_similarity(
            embeddings[0], embeddings[0]
        )
        assert np.isclose(self_similarity, 1.0, atol=1e-5)

    @pytest.mark.asyncio
    async def test_compute_similarity_batch(self, embedding_engine, sample_texts):
        """Test computing similarity between a query and multiple embeddings."""
        # Generate embeddings
        embeddings = await embedding_engine.embed_texts(sample_texts)

        # Select the first embedding as the query
        query_embedding = embeddings[0]

        # Compute similarities between the query and all embeddings
        similarities = embedding_engine.compute_similarity_batch(
            query_embedding, embeddings
        )

        # Verify the similarities have the correct shape
        assert isinstance(similarities, list)
        assert len(similarities) == len(embeddings)

        # Verify similarity values are within expected range
        for similarity in similarities:
            assert isinstance(similarity, float)
            assert -1.0 <= similarity <= 1.0

        # Similarity with itself should be 1.0 (or very close)
        assert np.isclose(similarities[0], 1.0, atol=1e-5)

    @pytest.mark.asyncio
    async def test_nearest_neighbors(self, embedding_engine, sample_texts):
        """Test finding nearest neighbors for a query embedding."""
        # Generate embeddings
        embeddings = await embedding_engine.embed_texts(sample_texts)

        # Select the first text as the query
        sample_texts[0]
        query_embedding = embeddings[0]

        # Find nearest neighbors
        neighbors = await embedding_engine.nearest_neighbors(
            query_embedding, embeddings, k=3
        )

        # Verify the neighbors have the correct structure
        assert isinstance(neighbors, list)
        assert len(neighbors) <= 3  # Should return at most k neighbors

        # Each neighbor should be a tuple of (index, similarity)
        for neighbor in neighbors:
            assert isinstance(neighbor, tuple)
            assert len(neighbor) == 2
            assert isinstance(neighbor[0], int)
            assert isinstance(neighbor[1], float)
            assert -1.0 <= neighbor[1] <= 1.0

        # The first neighbor should be the query itself
        assert neighbors[0][0] == 0
        assert np.isclose(neighbors[0][1], 1.0, atol=1e-5)

    @pytest.mark.asyncio
    async def test_semantic_search(self, embedding_engine, sample_texts):
        """Test semantic search functionality."""
        # Index the sample texts
        await embedding_engine.index_texts(sample_texts)

        # Perform a semantic search
        query = "embedding technical terms"
        results = await embedding_engine.semantic_search(query, top_k=3)

        # Verify the results have the correct structure
        assert isinstance(results, list)
        assert len(results) <= 3  # Should return at most top_k results

        # Each result should have the expected fields
        for result in results:
            assert isinstance(result, dict)
            assert "index" in result
            assert "text" in result
            assert "score" in result
            assert isinstance(result["index"], int)
            assert isinstance(result["text"], str)
            assert isinstance(result["score"], float)
            assert 0.0 <= result["score"] <= 1.0  # Scores should be normalized

    @pytest.mark.asyncio
    async def test_embed_with_custom_pooling(self, embedding_engine, sample_texts):
        """Test embedding generation with different pooling methods."""
        # Test with different pooling methods
        pooling_methods = ["mean", "max", "cls"]

        for method in pooling_methods:
            # Set the pooling method
            embedding_engine.pooling_method = method

            # Generate embedding
            text = sample_texts[0]
            embedding = await embedding_engine.embed_text(text)

            # Verify the embedding has the correct shape
            assert isinstance(embedding, np.ndarray)
            assert embedding.shape == (embedding_engine.dimensions,)

    @pytest.mark.asyncio
    async def test_model_loading(self):
        """Test model loading with different configurations."""
        # Test loading with different model names
        model_configs = [
            {"embedding": {"model": "bert-base-uncased", "dimensions": 768}},
            {"embedding": {"model": "distilbert-base-uncased", "dimensions": 768}},
        ]

        for config in model_configs:
            # Initialize with the config
            engine = EmbeddingEngine(config)

            # Verify the model was loaded
            assert engine.model is not None
            assert engine.tokenizer is not None

            # Verify the model has the expected output dimension
            assert engine.dimensions == config["embedding"]["dimensions"]

    @pytest.mark.asyncio
    async def test_model_to_device(self, embedding_engine):
        """Test moving the model to different devices."""
        # Skip this test if CUDA is not available
        try:
            import torch

            if not torch.cuda.is_available():
                pytest.skip("CUDA not available")

            # Move to CUDA
            embedding_engine.to_device("cuda")

            # Verify the model is on CUDA
            assert next(embedding_engine.model.parameters()).device.type == "cuda"

            # Move back to CPU
            embedding_engine.to_device("cpu")

            # Verify the model is on CPU
            assert next(embedding_engine.model.parameters()).device.type == "cpu"
        except ImportError:
            pytest.skip("PyTorch not available")

    @pytest.mark.asyncio
    async def test_cache_management(self, embedding_engine, sample_texts):
        """Test caching of embeddings."""
        # Enable caching
        embedding_engine.enable_cache = True

        # Generate embedding for the first time
        text = sample_texts[0]
        first_embedding = await embedding_engine.embed_text(text)

        # Mock the internal embedding method
        original_embed = embedding_engine._generate_embedding
        embed_called = False

        async def mock_embed(*args, **kwargs):
            nonlocal embed_called
            embed_called = True
            return await original_embed(*args, **kwargs)

        embedding_engine._generate_embedding = mock_embed

        # Generate embedding for the same text again
        second_embedding = await embedding_engine.embed_text(text)

        # Verify the internal embed method was not called (using cache)
        assert not embed_called

        # Verify the embeddings are the same
        assert np.array_equal(first_embedding, second_embedding)

        # Clear the cache
        embedding_engine.clear_cache()

        # Generate embedding again
        third_embedding = await embedding_engine.embed_text(text)

        # Verify the internal embed method was called (cache was cleared)
        assert embed_called

        # Verify the embeddings are still the same (deterministic)
        assert np.array_equal(first_embedding, third_embedding)

    @pytest.mark.asyncio
    async def test_concurrent_embedding(self, embedding_engine, sample_texts):
        """Test that multiple concurrent embedding operations work correctly."""
        # Create tasks for concurrent execution
        tasks = [
            asyncio.create_task(embedding_engine.embed_text(text))
            for text in sample_texts
        ]

        # Wait for all tasks to complete
        embeddings = await asyncio.gather(*tasks)

        # Verify all operations succeeded
        assert len(embeddings) == len(sample_texts)
        assert all(isinstance(embedding, np.ndarray) for embedding in embeddings)
        assert all(
            embedding.shape == (embedding_engine.dimensions,)
            for embedding in embeddings
        )
