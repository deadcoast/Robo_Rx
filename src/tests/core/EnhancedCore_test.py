from pathlib import Path
from unittest.mock import AsyncMock, Mock, patch

import numpy as np
import pytest

from src.config.EngineConfig import SystemConfig
from src.core.EnhancedCore import EnhancedFeatureSet, EnhancedMarkdownProcessor
from src.processors.EnhancedProcessingError import EnhancedProcessingError


class TestEnhancedFeatureSet:
    """Test suite for the EnhancedFeatureSet dataclass."""

    def test_init(self):
        """Test initialization of EnhancedFeatureSet."""
        # Create sample data
        embeddings = np.array([[0.1, 0.2], [0.3, 0.4]])
        topic_features = np.array([[0.5, 0.6], [0.7, 0.8]])
        graph_features = np.array([[0.9, 1.0], [1.1, 1.2]])
        metadata_features = np.array([[1.3, 1.4], [1.5, 1.6]])

        # Create an instance
        feature_set = EnhancedFeatureSet(
            embeddings=embeddings,
            topic_features=topic_features,
            graph_features=graph_features,
            metadata_features=metadata_features,
        )

        # Verify fields
        assert np.array_equal(feature_set.embeddings, embeddings)
        assert np.array_equal(feature_set.topic_features, topic_features)
        assert np.array_equal(feature_set.graph_features, graph_features)
        assert np.array_equal(feature_set.metadata_features, metadata_features)


class TestEnhancedMarkdownProcessor:
    """Test suite for the EnhancedMarkdownProcessor class."""

    @pytest.fixture
    def config(self):
        """Create a SystemConfig instance for testing."""
        return Mock(spec=SystemConfig)

    @pytest.fixture
    def processor(self, config):
        """Create an EnhancedMarkdownProcessor instance with mocked dependencies for testing."""
        with patch(
            "src.core.EnhancedCore.MarkdownProcessor.__init__"
        ) as mock_super_init, patch(
            "src.core.EnhancedCore.NLPCore"
        ) as mock_nlp_core_class, patch(
            "src.core.EnhancedCore.DataHandler"
        ) as mock_data_handler_class:

            # Create mocks
            mock_nlp_core = Mock()
            mock_data_handler = Mock()

            # Configure mocks
            mock_super_init.return_value = None
            mock_nlp_core_class.return_value = mock_nlp_core
            mock_data_handler_class.return_value = mock_data_handler

            # Create processor
            processor = EnhancedMarkdownProcessor(config)

            # Assign mocks directly for easier testing
            processor.nlp_core = mock_nlp_core
            processor.data_handler = mock_data_handler
            processor.logger = Mock()

            return processor

    def test_init(self, processor, config):
        """Test initialization of EnhancedMarkdownProcessor."""
        assert processor.nlp_core is not None
        assert processor.data_handler is not None

    @pytest.mark.asyncio
    async def test_process_vault_success(self, processor):
        """Test process_vault method with successful processing."""
        # Setup
        mock_vault_path = Mock(spec=Path)
        mock_files = [Mock(spec=Path), Mock(spec=Path)]

        # Mock Path.rglob to return our mock files
        mock_vault_path.rglob.return_value = mock_files

        # Mock ProcessingPool and its async context manager
        mock_pool = AsyncMock()
        mock_pool.map = AsyncMock()
        mock_pool.map.return_value = [
            {"path": "file1.md", "data": "processed1"},
            {"path": "file2.md", "data": "processed2"},
        ]

        # Mock the _aggregate_enhanced_results method
        mock_aggregated_result = {"combined": "result"}
        processor._aggregate_enhanced_results = Mock(
            return_value=mock_aggregated_result
        )

        # Mock the async context manager
        with patch("src.core.EnhancedCore.ProcessingPool", return_value=mock_pool):
            # Call the method
            result = await processor.process_vault(mock_vault_path)

            # Verify
            assert result == mock_aggregated_result
            mock_vault_path.rglob.assert_called_once_with("*.md")
            mock_pool.map.assert_called_once()
            processor._aggregate_enhanced_results.assert_called_once_with(
                mock_pool.map.return_value
            )

    @pytest.mark.asyncio
    async def test_process_vault_error(self, processor):
        """Test process_vault method with error handling."""
        # Setup
        mock_vault_path = Mock(spec=Path)

        # Mock Path.rglob to raise an exception
        mock_error = Exception("Test error")
        mock_vault_path.rglob.side_effect = mock_error

        # Call the method and verify exception handling
        with pytest.raises(EnhancedProcessingError) as excinfo, patch(
            "src.core.EnhancedCore.ProcessingPool"
        ):
            await processor.process_vault(mock_vault_path)

        # Verify
        assert str(mock_error) in str(excinfo.value)
        processor.logger.error.assert_called_once()

    @pytest.mark.asyncio
    async def test_enhanced_file_processing(self, processor):
        """Test _enhanced_file_processing method."""
        # Setup
        mock_file_path = Mock(spec=Path)
        mock_content = "# Test Markdown Content"

        # Mock the methods called by _enhanced_file_processing
        processor._read_file = AsyncMock(return_value=mock_content)
        processor.nlp_core.process_content = AsyncMock(return_value={"nlp": "features"})
        processor.data_handler.process_data = AsyncMock(
            return_value={"structured": "data"}
        )
        processor._extract_enhanced_metadata = Mock(return_value={"meta": "data"})

        # Call the method
        result = await processor._enhanced_file_processing(mock_file_path)

        # Verify
        assert result["path"] == str(mock_file_path)
        assert result["nlp_features"] == {"nlp": "features"}
        assert result["structured_data"] == {"structured": "data"}
        assert result["metadata"] == {"meta": "data"}

        processor._read_file.assert_called_once_with(mock_file_path)
        processor.nlp_core.process_content.assert_called_once_with(mock_content)
        processor.data_handler.process_data.assert_called_once_with({"nlp": "features"})
        processor._extract_enhanced_metadata.assert_called_once_with(mock_content)

    def test_extract_enhanced_metadata(self, processor):
        """Test _extract_enhanced_metadata method."""
        # Since this method is a placeholder (pass) in the implementation,
        # we can only verify that it can be called without errors.
        # We can't test its functionality until it's implemented.
        content = "# Test Markdown Content"

        # This should not raise an exception if calling the method
        # Note: If this is a strict test, it might fail since method just has 'pass'
        try:
            processor._extract_enhanced_metadata(content)
        except NotImplementedError:
            # This is acceptable since the method is not implemented yet
            pass
