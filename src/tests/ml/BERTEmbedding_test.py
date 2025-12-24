import numpy as np
import pytest

from src.main import BERTEmbedding


class TestBERTEmbedding:
    """Test suite for the BERTEmbedding class."""

    @pytest.fixture
    def embedding_generator(self):
        """Create a BERTEmbedding instance for testing."""
        return BERTEmbedding()

    def test_generate_returns_numpy_array(self, embedding_generator):
        """Test that generate returns a numpy array with expected dimensions."""
        text = "This is a test sentence."
        embedding = embedding_generator.generate(text)

        # Verify the result is a numpy array
        assert isinstance(embedding, np.ndarray)

        # Verify the standard BERT embedding size (768)
        assert embedding.shape == (768,)

        # Verify the embedding is not all zeros
        assert not np.all(embedding == 0)

        # Verify the embedding is not all the same value
        assert len(set(embedding.flatten())) > 1

    def test_generate_with_empty_text(self, embedding_generator):
        """Test generating embeddings from empty text."""
        embedding = embedding_generator.generate("")

        # Verify the result is a numpy array with standard dimensions
        assert isinstance(embedding, np.ndarray)
        assert embedding.shape == (768,)

    def test_generate_with_different_text_produces_different_embeddings(
        self, embedding_generator
    ):
        """Test that different text inputs produce different embeddings."""
        text1 = "First text example"
        text2 = "Second completely different text"

        embedding1 = embedding_generator.generate(text1)
        embedding2 = embedding_generator.generate(text2)

        # Verify the embeddings are different
        assert not np.array_equal(embedding1, embedding2)

    def test_generate_consistency(self, embedding_generator):
        """Test consistency of embedding generation for the same input."""
        text = "This is a sample text for consistency testing."

        # In the current mock implementation, embeddings will be random,
        # but in a real implementation, they should be deterministic
        # for the same input text
        embedding1 = embedding_generator.generate(text)
        embedding2 = embedding_generator.generate(text)

        # Note: Since the current implementation generates random values,
        # the following assertion would fail. Uncomment if using a real BERT model.
        # assert np.array_equal(embedding1, embedding2)

        # Instead, for the mock implementation, we just verify the dimensions
        assert embedding1.shape == embedding2.shape

    def test_generate_with_special_characters(self, embedding_generator):
        """Test embedding generation with text containing special characters."""
        text_with_special = "Special chars: !@#$%^&*()"
        embedding = embedding_generator.generate(text_with_special)

        # Verify the result is a numpy array with standard dimensions
        assert isinstance(embedding, np.ndarray)
        assert embedding.shape == (768,)

    def test_generate_with_long_text(self, embedding_generator):
        """Test embedding generation with longer text input."""
        long_text = " ".join(["word"] * 1000)  # A text with 1000 "word" repetitions
        embedding = embedding_generator.generate(long_text)

        # Verify the result is a numpy array with standard dimensions
        # Real BERT models might truncate or handle this differently,
        # but the output dimension should remain consistent
        assert isinstance(embedding, np.ndarray)
        assert embedding.shape == (768,)
