import logging
from unittest.mock import Mock, patch

import numpy as np
import pytest
import torch

from src.core.FeatureCore import (
    AnalysisError,
    AnalysisResult,
    AnalyticsEngine,
    FeatureGenerationError,
    FeatureProcessor,
    SystemConfig,
)


class TestFeatureGenerationError:
    """Test suite for the FeatureGenerationError class."""

    def test_init(self):
        """Test initialization of FeatureGenerationError."""
        error_message = "Feature generation failed"
        error = FeatureGenerationError(error_message)

        assert str(error) == error_message
        assert isinstance(error, Exception)


class TestAnalysisError:
    """Test suite for the AnalysisError class."""

    def test_init(self):
        """Test initialization of AnalysisError."""
        error_message = "Analysis failed"
        error = AnalysisError(error_message)

        assert str(error) == error_message
        assert isinstance(error, Exception)


class TestAnalysisResult:
    """Test suite for the AnalysisResult class."""

    @pytest.fixture
    def analysis_result(self):
        """Create an AnalysisResult instance for testing."""
        clusters = np.array([0, 1, 0, 2, 1])
        topics = np.array([[0.1, 0.2, 0.7], [0.3, 0.5, 0.2]])
        metadata = {"num_docs": 10, "processing_time": 5.2}

        return AnalysisResult(clusters=clusters, topics=topics, metadata=metadata)

    def test_init(self, analysis_result):
        """Test initialization of AnalysisResult."""
        assert isinstance(analysis_result.clusters, np.ndarray)
        assert isinstance(analysis_result.topics, np.ndarray)
        assert isinstance(analysis_result.metadata, dict)

        assert np.array_equal(analysis_result.clusters, np.array([0, 1, 0, 2, 1]))
        assert np.array_equal(
            analysis_result.topics, np.array([[0.1, 0.2, 0.7], [0.3, 0.5, 0.2]])
        )
        assert analysis_result.metadata == {"num_docs": 10, "processing_time": 5.2}

    def test_init_default_metadata(self):
        """Test initialization with default metadata."""
        clusters = np.array([0, 1, 0, 2, 1])
        topics = np.array([[0.1, 0.2, 0.7], [0.3, 0.5, 0.2]])

        result = AnalysisResult(clusters=clusters, topics=topics)

        assert result.metadata == {}


class TestSystemConfig:
    """Test suite for the SystemConfig class."""

    @pytest.fixture
    def system_config(self):
        """Create a default SystemConfig instance for testing."""
        return SystemConfig()

    def test_init_default(self, system_config):
        """Test initialization with default values."""
        assert system_config.max_threads == 8
        assert system_config.batch_size == 1000
        assert system_config.buffer_size == 2048 * 1024  # 2MB
        assert system_config.processing_mode == "CUDA_ENABLED"
        assert hasattr(system_config, "error_tolerance")

    def test_init_custom(self):
        """Test initialization with custom values."""
        config = SystemConfig(
            max_threads=4,
            batch_size=500,
            buffer_size=1024 * 1024,
            processing_mode="CPU_ONLY",
            error_tolerance=0.95,
        )

        assert config.max_threads == 4
        assert config.batch_size == 500
        assert config.buffer_size == 1024 * 1024
        assert config.processing_mode == "CPU_ONLY"
        assert config.error_tolerance == 0.95


class TestFeatureProcessor:
    """Test suite for the FeatureProcessor class."""

    @pytest.fixture
    def config(self):
        """Create a SystemConfig instance for testing."""
        return SystemConfig(max_threads=2, batch_size=10, processing_mode="CPU_ONLY")

    @pytest.fixture
    def feature_processor(self, config):
        """Create a FeatureProcessor instance with mocked dependencies for testing."""
        with patch(
            "src.core.FeatureCore.logging.getLogger"
        ) as mock_get_logger, patch.object(
            FeatureProcessor, "_initialize_bert"
        ) as mock_initialize_bert, patch(
            "src.core.FeatureCore.spacy.load"
        ) as mock_spacy_load:

            # Create mock logger
            mock_logger = Mock(spec=logging.Logger)
            mock_get_logger.return_value = mock_logger

            # Create mock BERT model
            mock_bert_model = Mock()
            mock_initialize_bert.return_value = mock_bert_model

            # Create mock spaCy model
            mock_nlp = Mock()
            mock_spacy_load.return_value = mock_nlp

            # Create processor
            processor = FeatureProcessor(config)

            # Set mocked components for easier testing
            processor.logger = mock_logger
            processor.model = mock_bert_model
            processor.nlp = mock_nlp

            return processor

    def test_init(self, feature_processor, config):
        """Test initialization of FeatureProcessor."""
        assert feature_processor.config == config
        assert feature_processor.logger is not None
        assert feature_processor.model is not None
        assert feature_processor.nlp is not None

    @patch.object(FeatureProcessor, "_generate_embedding")
    def test_generate_features(self, mock_generate_embedding, feature_processor):
        """Test generating features from processed documents."""
        # Setup
        processed_docs = [
            {"id": "doc1", "content": "This is document 1"},
            {"id": "doc2", "content": "This is document 2"},
            {"id": "doc3", "content": "This is document 3"},
        ]

        # Mock the _generate_embedding method to return constant embeddings
        mock_embeddings = np.array([0.1, 0.2, 0.3])
        mock_generate_embedding.return_value = mock_embeddings

        # Call the method
        features = feature_processor.generate_features(processed_docs)

        # Verify
        assert isinstance(features, np.ndarray)
        assert features.shape == (3, 3)  # 3 docs, 3 embedding dimensions
        mock_generate_embedding.assert_called()
        assert mock_generate_embedding.call_count == 3

    @patch.object(FeatureProcessor, "_generate_embedding")
    def test_generate_features_empty_docs(
        self, mock_generate_embedding, feature_processor
    ):
        """Test generating features with empty document list."""
        # Setup
        processed_docs = []

        # Call the method
        features = feature_processor.generate_features(processed_docs)

        # Verify
        assert isinstance(features, np.ndarray)
        assert features.size == 0
        mock_generate_embedding.assert_not_called()

    @patch.object(
        FeatureProcessor, "_generate_embedding", side_effect=Exception("Test error")
    )
    def test_generate_features_error(self, mock_generate_embedding, feature_processor):
        """Test error handling when generating features."""
        # Setup
        processed_docs = [{"id": "doc1", "content": "This is document 1"}]

        # Call the method and verify exception
        with pytest.raises(FeatureGenerationError):
            feature_processor.generate_features(processed_docs)

        # Verify
        mock_generate_embedding.assert_called_once()

    def test_generate_embedding(self, feature_processor):
        """Test generating embeddings for content."""
        # Setup
        content = "This is test content"

        # Mock BERT model to return a tensor
        feature_processor.model.encode.return_value = torch.tensor([[0.1, 0.2, 0.3]])

        # Call the method
        embedding = feature_processor._generate_embedding(content)

        # Verify
        assert isinstance(embedding, np.ndarray)
        feature_processor.model.encode.assert_called_once_with([content])


class TestAnalyticsEngine:
    """Test suite for the AnalyticsEngine class."""

    @pytest.fixture
    def config(self):
        """Create a SystemConfig instance for testing."""
        return SystemConfig(max_threads=2, batch_size=10, processing_mode="CPU_ONLY")

    @pytest.fixture
    def analytics_engine(self, config):
        """Create an AnalyticsEngine instance with mocked dependencies for testing."""
        with patch("src.core.FeatureCore.HDBSCAN") as mock_hdbscan_class, patch(
            "src.core.FeatureCore.LatentDirichletAllocation"
        ) as mock_lda_class:

            # Create mock HDBSCAN instance
            mock_hdbscan = Mock()
            mock_hdbscan_class.return_value = mock_hdbscan

            # Create mock LDA instance
            mock_lda = Mock()
            mock_lda_class.return_value = mock_lda

            # Create engine
            engine = AnalyticsEngine(config)

            # Set mocked components for easier testing
            engine.hdbscan = mock_hdbscan
            engine.lda = mock_lda

            return engine

    def test_init(self, analytics_engine, config):
        """Test initialization of AnalyticsEngine."""
        assert analytics_engine.config == config
        assert analytics_engine.hdbscan is not None
        assert analytics_engine.lda is not None

    def test_parallel_process(self, analytics_engine):
        """Test parallel processing of items."""
        # Setup
        items = [1, 2, 3, 4, 5]

        def process_func(x):
            return x * 2

        with patch("src.core.FeatureCore.ThreadPoolExecutor") as mock_executor_class:
            # Mock the executor's map method
            Mock()
            mock_executor_class.return_value.__enter__.return_value.map.return_value = [
                2,
                4,
                6,
                8,
                10,
            ]
            mock_executor_class.return_value.__exit__.return_value = None

            # Call the method
            results = analytics_engine.parallel_process(items, process_func)

            # Verify
            assert results == [2, 4, 6, 8, 10]
            mock_executor_class.assert_called_once()

    def test_analyze_features(self, analytics_engine):
        """Test analyzing feature matrix."""
        # Setup
        feature_matrix = np.random.rand(10, 5)  # 10 samples, 5 features

        # Mock HDBSCAN fit_predict to return cluster labels
        analytics_engine.hdbscan.fit_predict.return_value = np.array(
            [0, 1, 0, 2, 1, 0, 1, 2, 0, 1]
        )

        # Mock LDA fit_transform to return topic distributions
        topic_matrix = np.random.rand(10, 20)  # 10 samples, 20 topics
        analytics_engine.lda.fit_transform.return_value = topic_matrix

        # Call the method
        result = analytics_engine.analyze_features(feature_matrix)

        # Verify
        assert isinstance(result, AnalysisResult)
        assert np.array_equal(
            result.clusters, analytics_engine.hdbscan.fit_predict.return_value
        )
        assert np.array_equal(result.topics, topic_matrix)
        assert "num_samples" in result.metadata
        assert result.metadata["num_samples"] == 10

        analytics_engine.hdbscan.fit_predict.assert_called_once_with(feature_matrix)
        analytics_engine.lda.fit_transform.assert_called_once_with(feature_matrix)

    def test_analyze_features_error(self, analytics_engine):
        """Test error handling when analyzing features."""
        # Setup
        feature_matrix = np.random.rand(10, 5)

        # Mock HDBSCAN to raise an exception
        analytics_engine.hdbscan.fit_predict.side_effect = Exception("Clustering error")

        # Call the method and verify exception
        with pytest.raises(AnalysisError):
            analytics_engine.analyze_features(feature_matrix)

        # Verify
        analytics_engine.hdbscan.fit_predict.assert_called_once_with(feature_matrix)

    def test_analyze_features_empty_matrix(self, analytics_engine):
        """Test analyzing an empty feature matrix."""
        # Setup
        feature_matrix = np.array([])

        # Call the method and verify exception
        with pytest.raises(AnalysisError):
            analytics_engine.analyze_features(feature_matrix)
