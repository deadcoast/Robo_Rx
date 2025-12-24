from unittest.mock import Mock, patch

import pytest
import torch

from src.main import EngineConfig, ModelPipelineConfig, ModelPipelineManager


class TestModelPipelineManager:
    """Test suite for the ModelPipelineManager class."""

    @pytest.fixture
    def mock_pipeline_config(self):
        """Create a mock ModelPipelineConfig for testing."""
        return ModelPipelineConfig()

    @pytest.fixture
    def mock_engine_config(self, mock_pipeline_config):
        """Create a mock EngineConfig with a ModelPipelineConfig for testing."""
        config = Mock(spec=EngineConfig)
        config.pipeline_config = mock_pipeline_config
        return config

    @pytest.fixture
    def pipeline_manager(self, mock_engine_config):
        """Create a ModelPipelineManager instance for testing."""
        with patch("torch.distributed.pipelining.pipeline", return_value=Mock()):
            with patch(
                "src.main.analyze_model_graph", return_value={"mock": "analysis"}
            ):
                return ModelPipelineManager(mock_engine_config)

    @pytest.fixture
    def mock_model_parts(self):
        """Create mock model parts for testing."""
        return [
            Mock(spec=torch.nn.Module),
            Mock(spec=torch.nn.Module),
            Mock(spec=torch.nn.Module),
        ]

    def test_init(self, pipeline_manager, mock_engine_config):
        """Test initialization of ModelPipelineManager."""
        assert pipeline_manager.config is mock_engine_config
        assert pipeline_manager.pipeline_config is mock_engine_config.pipeline_config
        assert pipeline_manager.current_pipeline is None
        assert pipeline_manager.analysis_results == {}

    @patch("src.main.setup_model_pipeline")
    def test_initialize_pipeline(
        self, mock_setup_pipeline, pipeline_manager, mock_model_parts
    ):
        """Test the initialize_pipeline method."""
        # Configure the mock return value
        mock_pipeline = Mock()
        mock_setup_pipeline.return_value = mock_pipeline

        # Call the method
        result = pipeline_manager.initialize_pipeline(mock_model_parts)

        # Verify setup_model_pipeline was called with correct args
        mock_setup_pipeline.assert_called_once_with(
            mock_model_parts, pipeline_manager.pipeline_config
        )

        # Verify the current_pipeline was set
        assert pipeline_manager.current_pipeline is mock_pipeline

        # Verify a success result was returned
        assert result is True

    @patch("src.main.setup_model_pipeline")
    def test_initialize_pipeline_handles_errors(
        self, mock_setup_pipeline, pipeline_manager, mock_model_parts
    ):
        """Test that initialize_pipeline handles exceptions gracefully."""
        # Configure the mock to raise an exception
        mock_setup_pipeline.side_effect = RuntimeError("Test error")

        # Call the method - it should handle the error and return False
        result = pipeline_manager.initialize_pipeline(mock_model_parts)

        # Verify a failure result was returned
        assert result is False

        # Verify the current_pipeline is still None
        assert pipeline_manager.current_pipeline is None

    def test_process_batch_without_pipeline(self, pipeline_manager):
        """Test that process_batch handles the case when no pipeline is initialized."""
        # Create a mock input tensor
        mock_input = torch.tensor([1.0, 2.0, 3.0])

        # Call the method - since no pipeline is initialized, it should return None
        result = pipeline_manager.process_batch(mock_input)

        # Verify the result is a tuple with None and metrics
        assert isinstance(result, tuple)
        assert result[0] is None
        assert isinstance(result[1], dict)
        assert "error" in result[1]

    def test_process_batch_with_pipeline(self, pipeline_manager, mock_model_parts):
        """Test processing a batch through the pipeline."""
        # Set up a mock pipeline
        mock_output = torch.tensor([4.0, 5.0, 6.0])
        mock_pipeline = Mock(return_value=mock_output)
        pipeline_manager.current_pipeline = mock_pipeline

        # Create a mock input tensor
        mock_input = torch.tensor([1.0, 2.0, 3.0])

        # Call the method
        output, metrics = pipeline_manager.process_batch(mock_input)

        # Verify the pipeline was called with the input
        mock_pipeline.assert_called_once_with(mock_input)

        # Verify the result is correct
        assert torch.equal(output, mock_output)
        assert isinstance(metrics, dict)
        assert "execution_time_ms" in metrics

    @patch("src.main.analyze_model_graph")
    def test_analyze_performance(self, mock_analyze, pipeline_manager):
        """Test the analyze_performance method."""
        # Create a mock model
        mock_model = Mock(spec=torch.nn.Module)

        # Configure the mock return value
        mock_analysis = {
            "layers": 3,
            "params": 1000000,
            "recommendations": ["Test recommendation"],
        }
        mock_analyze.return_value = mock_analysis

        # Call the method
        result = pipeline_manager.analyze_performance(mock_model)

        # Verify analyze_model_graph was called with the model
        mock_analyze.assert_called_once_with(mock_model)

        # Verify the analysis results were stored
        assert pipeline_manager.analysis_results == mock_analysis

        # Verify the result is the analysis
        assert result is mock_analysis

    def test_get_optimization_recommendations_without_analysis(self, pipeline_manager):
        """Test getting optimization recommendations without prior analysis."""
        # Call the method when no analysis has been done
        recommendations = pipeline_manager.get_optimization_recommendations()

        # Verify an empty list is returned
        assert recommendations == []

    def test_get_optimization_recommendations_with_analysis(self, pipeline_manager):
        """Test getting optimization recommendations with prior analysis."""
        # Set up mock analysis results with recommendations
        pipeline_manager.analysis_results = {
            "recommendations": [
                "Optimize layer 1",
                "Reduce parameter count in layer 2",
                "Consider quantization",
            ]
        }

        # Call the method
        recommendations = pipeline_manager.get_optimization_recommendations()

        # Verify the recommendations from the analysis are returned
        assert recommendations == pipeline_manager.analysis_results["recommendations"]
        assert len(recommendations) == 3
        assert "Optimize layer 1" in recommendations
        assert "Reduce parameter count in layer 2" in recommendations
        assert "Consider quantization" in recommendations

    def test_get_optimization_recommendations_without_recommendations_key(
        self, pipeline_manager
    ):
        """Test getting optimization recommendations when analysis doesn't have recommendations."""
        # Set up mock analysis results without recommendations
        pipeline_manager.analysis_results = {"layers": 3, "params": 1000000}

        # Call the method
        recommendations = pipeline_manager.get_optimization_recommendations()

        # Verify an empty list is returned
        assert recommendations == []
