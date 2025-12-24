import numpy as np
import pytest

from src.main import MetaFeatureExtractor


class TestMetaFeatureExtractor:
    """Test suite for the MetaFeatureExtractor class."""

    @pytest.fixture
    def feature_extractor(self):
        """Create a MetaFeatureExtractor instance for testing."""
        return MetaFeatureExtractor()

    @pytest.fixture
    def sample_metadata(self):
        """Create sample metadata for testing."""
        return {
            "tags": ["test", "sample", "metadata"],
            "created": "2025-03-31T04:10:29-07:00",
            "modified": "2025-03-31T04:10:29-07:00",
            "title": "Test Document",
            "authors": ["Test Author"],
            "category": "Testing",
        }

    def test_extract_returns_dict(self, feature_extractor, sample_metadata):
        """Test that extract returns a dictionary with expected structure."""
        result = feature_extractor.extract(sample_metadata)

        # Verify the result is a dictionary
        assert isinstance(result, dict)

        # Verify it contains the feature_vector key
        assert "feature_vector" in result

        # Verify feature_vector is a numpy array
        assert isinstance(result["feature_vector"], np.ndarray)

        # Verify the feature vector has non-zero elements
        assert len(result["feature_vector"]) > 0

        # Verify the feature vector is not all zeros
        assert not np.all(result["feature_vector"] == 0)

    def test_extract_with_empty_metadata(self, feature_extractor):
        """Test feature extraction with empty metadata."""
        empty_metadata = {}
        result = feature_extractor.extract(empty_metadata)

        # Verify the result is a dictionary
        assert isinstance(result, dict)

        # Verify it contains the feature_vector key
        assert "feature_vector" in result

        # Verify feature_vector is a numpy array
        assert isinstance(result["feature_vector"], np.ndarray)

    def test_extract_with_none_metadata(self, feature_extractor):
        """Test feature extraction with None metadata."""
        # Should either handle None gracefully or raise a specific error
        try:
            result = feature_extractor.extract(None)

            # If it doesn't raise an exception, check the result
            assert isinstance(result, dict)
            assert "feature_vector" in result
            assert isinstance(result["feature_vector"], np.ndarray)
        except (TypeError, AttributeError):
            # This is also acceptable behavior for None input
            pass

    def test_extract_with_different_metadata_produces_different_features(
        self, feature_extractor
    ):
        """Test that different metadata produces different feature vectors."""
        metadata1 = {"tags": ["first", "document"], "title": "First Document"}

        metadata2 = {
            "tags": ["second", "completely", "different"],
            "title": "Second Document",
        }

        result1 = feature_extractor.extract(metadata1)
        result2 = feature_extractor.extract(metadata2)

        # Verify the feature vectors are different
        # Note: This test depends on the actual implementation
        # The mock implementation returns random values, so they'll likely be different
        # For a deterministic implementation, specific metadata should produce specific features
        assert not np.array_equal(result1["feature_vector"], result2["feature_vector"])

    def test_extract_preserves_metadata(self, feature_extractor, sample_metadata):
        """Test that extract doesn't modify the input metadata."""
        # Create a copy of the metadata for comparison
        metadata_copy = sample_metadata.copy()

        # Extract features
        feature_extractor.extract(sample_metadata)

        # Verify the metadata wasn't modified
        assert sample_metadata == metadata_copy

    def test_feature_vector_dimensions(self, feature_extractor):
        """Test that the feature vector has consistent dimensions."""
        metadata1 = {"tags": ["one", "tag"]}
        metadata2 = {"tags": ["multiple", "different", "tags", "here"]}

        result1 = feature_extractor.extract(metadata1)
        result2 = feature_extractor.extract(metadata2)

        # Verify that feature vectors have the same dimension regardless of input
        assert result1["feature_vector"].shape == result2["feature_vector"].shape
