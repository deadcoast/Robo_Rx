import logging
from datetime import datetime
from unittest.mock import Mock, patch

import numpy as np
import pytest

from src.core.EarlyCore import (
    AnalyticsProgress,
    BatchData,
    BatchMetrics,
    BERTEmbedding,
    FeatureGenerationStats,
    FeatureMatrix,
    FeatureProcessor,
    MarkdownProcessor,
    MDParser,
    MetaExtractor,
    MetaFeatureExtractor,
    ProcessingStats,
    ProgressTracker,
)


class TestProgressTracker:
    """Test suite for the ProgressTracker class."""

    @pytest.fixture
    def progress_tracker(self):
        """Create a ProgressTracker instance for testing."""
        return ProgressTracker()

    def test_init(self, progress_tracker):
        """Test initialization of ProgressTracker."""
        assert progress_tracker.task_status == {}
        assert progress_tracker.error_log == []

    def test_update_status(self, progress_tracker):
        """Test update_status method."""
        # Call the method
        progress_tracker.update_status("task1", "running")

        # Verify
        assert progress_tracker.task_status["task1"] == "running"

        # Update same task with different status
        progress_tracker.update_status("task1", "completed")

        # Verify update
        assert progress_tracker.task_status["task1"] == "completed"

    def test_log_error(self, progress_tracker):
        """Test log_error method."""
        # Mock get_timestamp
        with patch.object(progress_tracker, "get_timestamp") as mock_timestamp:
            mock_timestamp.return_value = "2023-01-01T12:00:00"

            # Call the method
            progress_tracker.log_error("ERR001", "task1")

            # Verify
            assert len(progress_tracker.error_log) == 1
            error_entry = progress_tracker.error_log[0]
            assert error_entry["code"] == "ERR001"
            assert error_entry["task"] == "task1"
            assert error_entry["timestamp"] == "2023-01-01T12:00:00"

    def test_get_timestamp(self, progress_tracker):
        """Test get_timestamp method."""
        # Call the method
        timestamp = progress_tracker.get_timestamp()

        # Verify it's in ISO format
        try:
            datetime.fromisoformat(timestamp)
            is_valid = True
        except ValueError:
            is_valid = False

        assert is_valid


class TestMDParser:
    """Test suite for the MDParser class."""

    @pytest.fixture
    def config(self):
        """Create a config dictionary for testing."""
        return {"parse_mode": "html", "max_depth": 3, "enable_extensions": False}

    @pytest.fixture
    def md_parser(self, config):
        """Create an MDParser instance with mocked logger for testing."""
        with patch("src.core.EarlyCore.logging.getLogger") as mock_get_logger:
            # Create mock logger
            mock_logger = Mock(spec=logging.Logger)
            mock_get_logger.return_value = mock_logger

            # Create parser
            parser = MDParser(config)
            parser.logger = mock_logger

            return parser

    def test_init(self, md_parser, config):
        """Test initialization of MDParser."""
        assert md_parser.parse_mode == config["parse_mode"]
        assert md_parser.max_depth == config["max_depth"]
        assert md_parser.enable_extensions == config["enable_extensions"]
        assert md_parser.logger is not None


class TestMetaExtractor:
    """Test suite for the MetaExtractor class."""

    @pytest.fixture
    def meta_extractor(self):
        """Create a MetaExtractor instance with mocked logger for testing."""
        with patch("src.core.EarlyCore.logging.getLogger") as mock_get_logger:
            # Create mock logger
            mock_logger = Mock(spec=logging.Logger)
            mock_get_logger.return_value = mock_logger

            # Create extractor
            extractor = MetaExtractor()
            extractor.logger = mock_logger

            return extractor

    def test_init(self, meta_extractor):
        """Test initialization of MetaExtractor."""
        assert meta_extractor.source is None
        assert meta_extractor.format is None
        assert meta_extractor.metadata == {}
        assert meta_extractor.logger is not None

    def test_extract_empty_parsed(self, meta_extractor):
        """Test extract method with empty parsed input."""
        result = meta_extractor.extract({})
        assert result == {}

    def test_extract_non_dict_parsed(self, meta_extractor):
        """Test extract method with non-dict parsed input."""
        with pytest.raises(ValueError) as excinfo:
            meta_extractor.extract("not a dict")

        assert "Parsed input must be a dictionary" in str(excinfo.value)

    def test_extract_missing_source(self, meta_extractor):
        """Test extract method with missing source."""
        meta_extractor.source = None
        meta_extractor.format = "json"
        meta_extractor.metadata = {}

        with pytest.raises(ValueError) as excinfo:
            meta_extractor.extract({"key": "value"})

        assert "Source must be specified" in str(excinfo.value)

    def test_extract_missing_format(self, meta_extractor):
        """Test extract method with missing format."""
        meta_extractor.source = "file"
        meta_extractor.format = None
        meta_extractor.metadata = {}

        with pytest.raises(ValueError) as excinfo:
            meta_extractor.extract({"key": "value"})

        assert "Format must be specified" in str(excinfo.value)

    def test_extract_missing_metadata(self, meta_extractor):
        """Test extract method with missing metadata."""
        meta_extractor.source = "file"
        meta_extractor.format = "json"
        # Don't initialize metadata

        with pytest.raises(ValueError) as excinfo:
            meta_extractor.extract({"key": "value"})

        assert "Metadata must be initialized" in str(excinfo.value)

    def test_extract_non_dict_metadata(self, meta_extractor):
        """Test extract method with non-dict metadata."""
        meta_extractor.source = "file"
        meta_extractor.format = "json"
        meta_extractor.metadata = "not a dict"

        with pytest.raises(ValueError) as excinfo:
            meta_extractor.extract({"key": "value"})

        assert "Metadata must be a dictionary" in str(excinfo.value)


class TestMarkdownProcessor:
    """Test suite for the MarkdownProcessor class."""

    @pytest.fixture
    def config(self):
        """Create a config dictionary for testing."""
        return {"parse_mode": "html", "max_depth": 3}

    @pytest.fixture
    def markdown_processor(self, config):
        """Create a MarkdownProcessor instance with mocked dependencies for testing."""
        with patch("src.core.EarlyCore.MDParser") as mock_parser_class, patch(
            "src.core.EarlyCore.MetaExtractor"
        ) as mock_extractor_class, patch(
            "src.core.EarlyCore.Normalizer"
        ) as mock_normalizer_class, patch(
            "src.core.EarlyCore.logging.getLogger"
        ) as mock_get_logger:

            # Create mock objects
            mock_parser = Mock()
            mock_extractor = Mock()
            mock_normalizer = Mock()
            mock_logger = Mock(spec=logging.Logger)

            # Configure mocks
            mock_parser_class.return_value = mock_parser
            mock_extractor_class.return_value = mock_extractor
            mock_normalizer_class.return_value = mock_normalizer
            mock_get_logger.return_value = mock_logger

            # Create processor
            processor = MarkdownProcessor(config)

            # Set mocked components for easier testing
            processor.parser = mock_parser
            processor.metadata_extractor = mock_extractor
            processor.content_normalizer = mock_normalizer
            processor.logger = mock_logger

            return processor

    def test_init(self, markdown_processor, config):
        """Test initialization of MarkdownProcessor."""
        assert markdown_processor.parser is not None
        assert markdown_processor.metadata_extractor is not None
        assert markdown_processor.content_normalizer is not None
        assert markdown_processor.logger is not None

    def test_process_batch_empty(self, markdown_processor):
        """Test process_batch method with empty batch."""
        result = markdown_processor.process_batch([])
        assert result == []

    def test_process_batch_non_list(self, markdown_processor):
        """Test process_batch method with non-list input."""
        with pytest.raises(ValueError) as excinfo:
            markdown_processor.process_batch("not a list")

        assert "File batch must be a list" in str(excinfo.value)


class TestProcessingStats:
    """Test suite for the ProcessingStats class."""

    @pytest.fixture
    def processing_stats(self):
        """Create a ProcessingStats instance for testing."""
        stats = ProcessingStats()
        stats.logger = Mock(spec=logging.Logger)  # Replace with mock logger
        return stats

    def test_init(self, processing_stats):
        """Test initialization of ProcessingStats."""
        assert processing_stats.files_processed == 0
        assert processing_stats.current_batch == 0
        assert processing_stats.errors_encountered == []

    def test_update(self, processing_stats):
        """Test update method."""
        # Create a mock batch result
        batch_result = Mock()
        batch_result.success = ["file1.md", "file2.md"]
        batch_result.errors = ["Error in file3.md", "Error in file4.md"]

        # Call the method
        processing_stats.update(batch_result)

        # Verify
        assert processing_stats.files_processed == 2
        assert processing_stats.current_batch == 1
        assert len(processing_stats.errors_encountered) == 2
        assert "Error in file3.md" in processing_stats.errors_encountered
        assert "Error in file4.md" in processing_stats.errors_encountered


class TestBERTEmbedding:
    """Test suite for the BERTEmbedding class."""

    @pytest.fixture
    def bert_embedding(self):
        """Create a BERTEmbedding instance with mocked dependencies for testing."""
        with patch("src.core.EarlyCore.BertTokenizer") as mock_tokenizer_class, patch(
            "src.core.EarlyCore.torch.cuda.is_available"
        ) as mock_cuda_available, patch(
            "src.core.EarlyCore.logging.getLogger"
        ) as mock_get_logger:

            # Create mock objects
            mock_tokenizer = Mock()
            mock_logger = Mock(spec=logging.Logger)

            # Configure mocks
            mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer
            mock_cuda_available.return_value = False  # Use CPU for tests
            mock_get_logger.return_value = mock_logger

            # Create embedding
            embedding = BERTEmbedding(model_name="bert-base-uncased")

            # Set mocked components for easier testing
            embedding.tokenizer = mock_tokenizer
            embedding.logger = mock_logger

            return embedding

    def test_init(self, bert_embedding):
        """Test initialization of BERTEmbedding."""
        assert bert_embedding.model_name == "bert-base-uncased"
        assert bert_embedding.tokenizer is not None
        assert bert_embedding.embedding_dim == 768
        assert bert_embedding.max_seq_length == 512
        assert bert_embedding.device == "cpu"
        assert bert_embedding.logger is not None


class TestMetaFeatureExtractor:
    """Test suite for the MetaFeatureExtractor class."""

    @pytest.fixture
    def meta_feature_extractor(self):
        """Create a MetaFeatureExtractor instance with mocked logger for testing."""
        with patch("src.core.EarlyCore.logging.getLogger") as mock_get_logger:
            # Create mock logger
            mock_logger = Mock(spec=logging.Logger)
            mock_get_logger.return_value = mock_logger

            # Create extractor
            extractor = MetaFeatureExtractor()
            extractor.logger = mock_logger

            return extractor

    def test_init_default(self, meta_feature_extractor):
        """Test initialization with default values."""
        assert meta_feature_extractor.dataset is None
        assert meta_feature_extractor.extracted_features == {}
        assert meta_feature_extractor.logger is not None

    def test_init_with_dataset(self):
        """Test initialization with custom dataset."""
        with patch("src.core.EarlyCore.logging.getLogger") as mock_get_logger:
            mock_logger = Mock(spec=logging.Logger)
            mock_get_logger.return_value = mock_logger

            dataset = [1, 2, 3]
            extractor = MetaFeatureExtractor(dataset)

            assert extractor.dataset == dataset


class TestBatchData:
    """Test suite for the BatchData class."""

    @pytest.fixture
    def batch_data(self):
        """Create a BatchData instance with mocked logger for testing."""
        with patch("src.core.EarlyCore.logging.getLogger") as mock_get_logger:
            # Create mock logger
            mock_logger = Mock(spec=logging.Logger)
            mock_get_logger.return_value = mock_logger

            # Create batch data
            data = ["item1", "item2", "item3"]
            batch = BatchData(data=data, batch_size=3, metadata={"source": "test"})
            batch.logger = mock_logger

            return batch

    def test_init(self, batch_data):
        """Test initialization of BatchData."""
        assert batch_data.data == ["item1", "item2", "item3"]
        assert batch_data.batch_size == 3
        assert batch_data.metadata == {"source": "test"}
        assert batch_data.logger is not None

    def test_init_default_metadata(self):
        """Test initialization with default metadata."""
        with patch("src.core.EarlyCore.logging.getLogger") as mock_get_logger:
            mock_logger = Mock(spec=logging.Logger)
            mock_get_logger.return_value = mock_logger

            batch = BatchData(data=["item1"])
            batch.logger = mock_logger

            assert batch.metadata == {}


class TestFeatureMatrix:
    """Test suite for the FeatureMatrix class."""

    @pytest.fixture
    def feature_matrix(self):
        """Create a FeatureMatrix instance with mocked logger for testing."""
        with patch("src.core.EarlyCore.logging.getLogger") as mock_get_logger:
            # Create mock logger
            mock_logger = Mock(spec=logging.Logger)
            mock_get_logger.return_value = mock_logger

            # Create feature matrix
            data = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
            feature_names = ["feature1", "feature2", "feature3"]
            matrix = FeatureMatrix(data=data, feature_names=feature_names)
            matrix.logger = mock_logger

            return matrix

    def test_init(self, feature_matrix):
        """Test initialization of FeatureMatrix."""
        assert feature_matrix.data == [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
        assert feature_matrix.feature_names == ["feature1", "feature2", "feature3"]
        assert feature_matrix.num_features == 3
        assert feature_matrix.num_samples == 2
        assert feature_matrix.logger is not None

    def test_init_without_feature_names(self):
        """Test initialization without feature names."""
        with patch("src.core.EarlyCore.logging.getLogger") as mock_get_logger:
            mock_logger = Mock(spec=logging.Logger)
            mock_get_logger.return_value = mock_logger

            data = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
            matrix = FeatureMatrix(data=data)
            matrix.logger = mock_logger

            assert matrix.feature_names is None
            assert matrix.num_features == 0


class TestFeatureProcessor:
    """Test suite for the FeatureProcessor class."""

    @pytest.fixture
    def nlp_core(self):
        """Create a mock NLPCore for testing."""
        return Mock()

    @pytest.fixture
    def embedding_generator(self):
        """Create a mock BERTEmbedding for testing."""
        return Mock()

    @pytest.fixture
    def meta_feature_extractor(self):
        """Create a mock MetaFeatureExtractor for testing."""
        return Mock()

    @pytest.fixture
    def feature_processor(self, nlp_core, embedding_generator, meta_feature_extractor):
        """Create a FeatureProcessor instance with mocked dependencies for testing."""
        with patch("src.core.EarlyCore.logging.getLogger") as mock_get_logger:
            # Create mock logger
            mock_logger = Mock(spec=logging.Logger)
            mock_get_logger.return_value = mock_logger

            # Create processor
            processor = FeatureProcessor(
                nlp_core=nlp_core,
                embedding_generator=embedding_generator,
                meta_feature_extractor=meta_feature_extractor,
            )
            processor.logger = mock_logger

            return processor

    def test_init(
        self, feature_processor, nlp_core, embedding_generator, meta_feature_extractor
    ):
        """Test initialization of FeatureProcessor."""
        assert feature_processor.nlp_core is nlp_core
        assert feature_processor.embedding_generator is embedding_generator
        assert feature_processor.meta_feature_extractor is meta_feature_extractor
        assert feature_processor.logger is not None

    def test_generate_features(self, feature_processor):
        """Test generate_features method."""
        # Setup mock batch data
        mock_batch = Mock(spec=BatchData)
        mock_batch.data = ["content1", "content2"]

        # Mock component behaviors
        feature_processor.nlp_core.process_text = Mock(return_value={"nlp": "features"})
        feature_processor.embedding_generator.generate = Mock(
            return_value=np.array([[0.1, 0.2], [0.3, 0.4]])
        )
        feature_processor.meta_feature_extractor.extract = Mock(
            return_value={"meta": "features"}
        )

        # Mock merge_features
        mock_merged = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
        feature_processor.merge_features = Mock(return_value=mock_merged)

        # Call the method
        with patch("src.core.EarlyCore.FeatureMatrix") as mock_feature_matrix_class:
            mock_feature_matrix = Mock(spec=FeatureMatrix)
            mock_feature_matrix_class.return_value = mock_feature_matrix

            result = feature_processor.generate_features(mock_batch)

            # Verify
            assert result is mock_feature_matrix
            feature_processor.embedding_generator.generate.assert_called_once()
            feature_processor.meta_feature_extractor.extract.assert_called_once()
            feature_processor.merge_features.assert_called_once()
            mock_feature_matrix_class.assert_called_once_with(mock_merged)

    def test_merge_features(self, feature_processor):
        """Test merge_features method."""
        # Setup
        embeddings = np.array([[0.1, 0.2], [0.3, 0.4]])
        meta_features = {"feature1": [0.5, 0.6], "feature2": [0.7, 0.8]}

        # Call the method
        with patch("src.core.EarlyCore.np.hstack") as mock_hstack:
            mock_result = np.array([[0.1, 0.2, 0.5, 0.7], [0.3, 0.4, 0.6, 0.8]])
            mock_hstack.return_value = mock_result

            result = feature_processor.merge_features(embeddings, meta_features)

            # Verify
            assert result is mock_result
            # Verify np.hstack was called (specific args depend on implementation)
            mock_hstack.assert_called_once()


class TestBatchMetrics:
    """Test suite for the BatchMetrics class."""

    @pytest.fixture
    def batch_metrics(self):
        """Create a BatchMetrics instance for testing."""
        return BatchMetrics(
            token_count=100,
            progress_percentage=50.0,
            batch_completion=0.5,
            batch_start_time=1000.0,
            batch_end_time=1100.0,
            batch_id="batch1",
        )

    def test_init(self, batch_metrics):
        """Test initialization of BatchMetrics."""
        assert batch_metrics.token_count == 100
        assert batch_metrics.progress_percentage == 50.0
        assert batch_metrics.batch_completion == 0.5
        assert batch_metrics.batch_start_time == 1000.0
        assert batch_metrics.batch_end_time == 1100.0
        assert batch_metrics.batch_id == "batch1"
        assert batch_metrics.metadata == {}
        assert batch_metrics.processed_count == 0
        assert batch_metrics.success_rate == 0.0
        assert batch_metrics.error_count == 0
        assert batch_metrics.execution_time_ms == 0
        assert batch_metrics.errors == []

    def test_update(self, batch_metrics):
        """Test update method."""
        # Setup
        new_metrics = {
            "processed_count": 10,
            "success_rate": 0.9,
            "error_count": 1,
            "execution_time_ms": 100,
            "errors": ["Error 1"],
        }

        # Call the method
        batch_metrics.update(new_metrics)

        # Verify
        assert batch_metrics.processed_count == 10
        assert batch_metrics.success_rate == 0.9
        assert batch_metrics.error_count == 1
        assert batch_metrics.execution_time_ms == 100
        assert batch_metrics.errors == ["Error 1"]

    def test_str_repr(self, batch_metrics):
        """Test __str__ and __repr__ methods."""
        # Verify str method produces a string
        assert isinstance(str(batch_metrics), str)

        # Verify repr method produces a string
        assert isinstance(repr(batch_metrics), str)

    def test_equality(self, batch_metrics):
        """Test equality methods."""
        # Create an identical metrics object
        identical_metrics = BatchMetrics(
            token_count=100,
            progress_percentage=50.0,
            batch_completion=0.5,
            batch_start_time=1000.0,
            batch_end_time=1100.0,
            batch_id="batch1",
        )

        # Create a different metrics object
        different_metrics = BatchMetrics(
            token_count=200,
            progress_percentage=60.0,
            batch_completion=0.6,
            batch_start_time=2000.0,
            batch_end_time=2100.0,
            batch_id="batch2",
        )

        # Test equality
        assert batch_metrics == identical_metrics
        assert batch_metrics != different_metrics
        assert batch_metrics != "not_a_batch_metrics"

        # Test hash
        assert hash(batch_metrics) == hash(identical_metrics)
        assert hash(batch_metrics) != hash(different_metrics)


class TestFeatureGenerationStats:
    """Test suite for the FeatureGenerationStats class."""

    @pytest.fixture
    def feature_generation_stats(self):
        """Create a FeatureGenerationStats instance for testing."""
        return FeatureGenerationStats()

    def test_init(self, feature_generation_stats):
        """Test initialization of FeatureGenerationStats."""
        assert feature_generation_stats.processed_tokens == 0
        assert feature_generation_stats.embedding_dimensions == []
        assert feature_generation_stats.batch_completion == 0.0

    def test_update_progress(self, feature_generation_stats):
        """Test update_progress method."""
        # Setup
        batch_metrics = BatchMetrics(
            token_count=100,
            progress_percentage=50.0,
            batch_completion=0.5,
            batch_start_time=1000.0,
            batch_end_time=1100.0,
        )

        # Call the method
        feature_generation_stats.update_progress(batch_metrics)

        # Verify
        assert feature_generation_stats.processed_tokens == 100
        assert feature_generation_stats.batch_completion == 0.5


class TestAnalyticsProgress:
    """Test suite for the AnalyticsProgress class."""

    @pytest.fixture
    def analytics_progress(self):
        """Create an AnalyticsProgress instance for testing."""
        return AnalyticsProgress()

    def test_init(self, analytics_progress):
        """Test initialization of AnalyticsProgress."""
        assert analytics_progress.cluster_count == 0
        assert analytics_progress.topic_coherence == 0.0
        assert analytics_progress.classification_depth == 0

    def test_track_metrics(self, analytics_progress):
        """Test track_metrics method."""
        # Setup
        mock_metrics = Mock()
        mock_metrics.clusters = 5
        mock_metrics.coherence = 0.75

        # Call the method
        analytics_progress.track_metrics(mock_metrics)

        # Verify
        assert analytics_progress.cluster_count == 5
        assert analytics_progress.topic_coherence == 0.75
