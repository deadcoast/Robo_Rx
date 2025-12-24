from unittest.mock import Mock, patch

import pytest

from src.core.ProccessCore import (
    DictIterator,
    ExecutionResult,
    ProcessingConfig,
    ProcessingContext,
    ProcessingEngine,
    ProcessingError,
    ProcessingResult,
    ProcessingStatus,
    ResultProcessor,
    ValidationStatus,
)
from src.ValidationData import ValidationData


class TestProcessingConfig:
    """Test suite for the ProcessingConfig class."""

    def test_init(self):
        """Test initialization of ProcessingConfig."""
        config_data = {"key": "value"}
        config = ProcessingConfig(config_data)

        assert config.config == config_data


class TestProcessingContext:
    """Test suite for the ProcessingContext class."""

    @pytest.fixture
    def config(self):
        """Create a ProcessingConfig instance for testing."""
        return ProcessingConfig({"key": "value"})

    @pytest.fixture
    def context(self, config):
        """Create a ProcessingContext instance for testing."""
        return ProcessingContext(config)

    def test_init(self, context, config):
        """Test initialization of ProcessingContext."""
        assert context.config == config
        assert context.context == {}

    def test_context_manager(self, context):
        """Test context manager functionality."""
        with context as ctx:
            assert ctx is context

    def test_get_set(self, context):
        """Test get and set methods."""
        # Test get on non-existent key
        assert context.get("non_existent") is None

        # Test set and get
        context.set("test_key", "test_value")
        assert context.get("test_key") == "test_value"

        # Test overwrite
        context.set("test_key", "new_value")
        assert context.get("test_key") == "new_value"


class TestProcessingError:
    """Test suite for the ProcessingError class."""

    @pytest.fixture
    def error(self):
        """Create a ProcessingError instance for testing."""
        return ProcessingError("Test error message")

    def test_init(self, error):
        """Test initialization of ProcessingError."""
        assert error.message == "Test error message"
        assert str(error) == "Test error message"
        assert repr(error) == "Test error message"

    def test_equality(self, error):
        """Test equality methods."""
        same_error = ProcessingError("Test error message")
        different_error = ProcessingError("Different message")

        assert error == same_error
        assert error != different_error
        assert error != "Not an error object"

    def test_hash(self, error):
        """Test hash method."""
        assert hash(error) == hash("Test error message")

    def test_len(self, error):
        """Test len method."""
        assert len(error) == len("Test error message")

    def test_iter(self, error):
        """Test iter method."""
        assert list(error) == list("Test error message")

    def test_contains(self, error):
        """Test contains method."""
        assert "error" in error
        assert "xyz" not in error

    def test_getitem(self, error):
        """Test getitem method."""
        assert error[0] == "T"
        assert error[:4] == "Test"


class TestValidationStatus:
    """Test suite for the ValidationStatus class."""

    @pytest.fixture
    def valid_status(self):
        """Create a valid ValidationStatus instance for testing."""
        return ValidationStatus(is_valid=True, validation_score=0.95)

    @pytest.fixture
    def invalid_status(self):
        """Create an invalid ValidationStatus instance for testing."""
        return ValidationStatus(
            is_valid=False, validation_score=0.45, issues=["Issue 1", "Issue 2"]
        )

    def test_init_valid(self, valid_status):
        """Test initialization with valid status."""
        assert valid_status.is_valid is True
        assert valid_status.validation_score == 0.95
        assert valid_status.issues == []

    def test_init_invalid(self, invalid_status):
        """Test initialization with invalid status and issues."""
        assert invalid_status.is_valid is False
        assert invalid_status.validation_score == 0.45
        assert len(invalid_status.issues) == 2
        assert "Issue 1" in invalid_status.issues
        assert "Issue 2" in invalid_status.issues


class TestExecutionResult:
    """Test suite for the ExecutionResult class."""

    @pytest.fixture
    def success_result(self):
        """Create a successful ExecutionResult instance for testing."""
        return ExecutionResult(
            success=True, metrics={"time": 1.5, "memory": 256}, status="completed"
        )

    @pytest.fixture
    def failure_result(self):
        """Create a failed ExecutionResult instance for testing."""
        return ExecutionResult(
            success=False, status="failed", error="Operation timed out"
        )

    def test_init_success(self, success_result):
        """Test initialization with success=True."""
        assert success_result.success is True
        assert success_result.metrics == {"time": 1.5, "memory": 256}
        assert success_result.status == "completed"
        assert success_result.error is None

    def test_init_failure(self, failure_result):
        """Test initialization with success=False."""
        assert failure_result.success is False
        assert failure_result.metrics is None
        assert failure_result.status == "failed"
        assert failure_result.error == "Operation timed out"


class TestProcessingStatus:
    """Test suite for the ProcessingStatus class."""

    @pytest.fixture
    def validation_status(self):
        """Create a ValidationStatus instance for testing."""
        return ValidationStatus(is_valid=True, validation_score=0.95)

    @pytest.fixture
    def processing_status(self, validation_status):
        """Create a ProcessingStatus instance for testing."""
        return ProcessingStatus(
            sequence_id="seq123",
            processing_phase="phase1",
            completion_metrics={"progress": 0.75},
            validation_status=validation_status,
        )

    def test_init(self, processing_status, validation_status):
        """Test initialization of ProcessingStatus."""
        assert processing_status.sequence_id == "seq123"
        assert processing_status.processing_phase == "phase1"
        assert processing_status.completion_metrics == {"progress": 0.75}
        assert processing_status.validation_status == validation_status
        assert processing_status.error_log == []
        assert processing_status.context == {}

    def test_init_with_error_log_and_context(self, validation_status):
        """Test initialization with error_log and context."""
        error_log = [ProcessingError("Error 1"), ProcessingError("Error 2")]
        context = {"key1": "value1", "key2": "value2"}

        status = ProcessingStatus(
            sequence_id="seq456",
            processing_phase="phase2",
            completion_metrics={"progress": 0.5},
            validation_status=validation_status,
            error_log=error_log,
            context=context,
        )

        assert status.error_log == error_log
        assert status.context == context


class TestDictIterator:
    """Test suite for the DictIterator class."""

    @pytest.fixture
    def test_dict(self):
        """Create a test dictionary for testing."""
        return {"key1": "value1", "key2": "value2", "key3": "value3"}

    @pytest.fixture
    def dict_iterator(self, test_dict):
        """Create a DictIterator instance for testing."""
        return DictIterator(test_dict)

    def test_init(self, dict_iterator, test_dict):
        """Test initialization of DictIterator."""
        assert dict_iterator.context == test_dict
        assert set(dict_iterator.keys) == set(test_dict.keys())
        assert dict_iterator.index == 0

    def test_iteration(self, dict_iterator, test_dict):
        """Test iteration functionality."""
        # Convert iterator to list for comparison
        keys = list(dict_iterator)

        # Verify all keys from the dictionary are present
        assert set(keys) == set(test_dict.keys())

        # Verify the iterator is exhausted
        assert dict_iterator.index == len(dict_iterator.keys)

        # Verify StopIteration is raised when iterator is exhausted
        with pytest.raises(StopIteration):
            next(dict_iterator)

    def test_len(self, dict_iterator, test_dict):
        """Test len method."""
        assert len(dict_iterator) == len(test_dict)


class TestProcessingResult:
    """Test suite for the ProcessingResult class."""

    @pytest.fixture
    def processing_result(self):
        """Create a ProcessingResult instance for testing."""
        return ProcessingResult({"data": "test_data"})

    def test_init(self, processing_result):
        """Test initialization of ProcessingResult."""
        assert processing_result.result == {"data": "test_data"}
        assert processing_result.error_log == []
        assert processing_result.context == {}

        # Verify status is initialized with default values
        assert processing_result.status.sequence_id == ""
        assert processing_result.status.processing_phase == ""
        assert processing_result.status.completion_metrics == {}
        assert processing_result.status.validation_status.is_valid is False
        assert processing_result.status.validation_status.validation_score == 0.0

    def test_str_repr(self, processing_result):
        """Test __str__ and __repr__ methods."""
        # Verify str and repr methods produce strings
        assert isinstance(str(processing_result), str)
        assert isinstance(repr(processing_result), str)

    def test_equality(self, processing_result):
        """Test equality methods."""
        # Create an identical result
        identical_result = ProcessingResult({"data": "test_data"})

        # Create a different result
        different_result = ProcessingResult({"data": "other_data"})

        # Add same structures to make them actually equal for testing
        identical_result.status = processing_result.status

        # Test equality
        assert processing_result == identical_result
        assert processing_result != different_result
        assert processing_result != "not a processing result"

    def test_hash(self, processing_result):
        """Test hash method."""
        # Just verify it doesn't raise an exception
        hash_value = hash(processing_result)
        assert isinstance(hash_value, int)

    def test_len(self, processing_result):
        """Test len method."""
        assert len(processing_result) == 0

        # Add items to context
        processing_result.context["key1"] = "value1"
        processing_result.context["key2"] = "value2"

        assert len(processing_result) == 2

    def test_iter(self, processing_result):
        """Test iter method."""
        # Add items to context
        processing_result.context["key1"] = "value1"
        processing_result.context["key2"] = "value2"

        # Convert to list for comparison
        keys = list(processing_result)

        assert set(keys) == {"key1", "key2"}

    def test_contains(self, processing_result):
        """Test contains method."""
        # Add item to context
        processing_result.context["test_key"] = "test_value"

        assert "test_key" in processing_result
        assert "non_existent" not in processing_result


class TestResultProcessor:
    """Test suite for the ResultProcessor class."""

    @pytest.fixture
    def config(self):
        """Create a ProcessingConfig instance for testing."""
        return ProcessingConfig({"key": "value"})

    @pytest.fixture
    def result_processor(self, config):
        """Create a ResultProcessor instance with mocked dependencies for testing."""
        with patch(
            "src.core.ProccessCore.ResultAnalyzer"
        ) as mock_analyzer_class, patch(
            "src.core.ProccessCore.ValidationEngine"
        ) as mock_validator_class, patch(
            "src.core.ProccessCore.MetricsComputer"
        ) as mock_metrics_class:

            # Create mock instances
            mock_analyzer = Mock()
            mock_validator = Mock()
            mock_metrics = Mock()

            # Configure mocks
            mock_analyzer_class.return_value = mock_analyzer
            mock_validator_class.return_value = mock_validator
            mock_metrics_class.return_value = mock_metrics

            # Create processor
            processor = ResultProcessor(config)

            # Set mocked components for easier testing
            processor.analyzer = mock_analyzer
            processor.validator = mock_validator
            processor.metrics = mock_metrics

            return processor

    def test_init(self, result_processor, config):
        """Test initialization of ResultProcessor."""
        assert result_processor.analyzer is not None
        assert result_processor.validator is not None
        assert result_processor.metrics is not None

    def test_process_results(self, result_processor):
        """Test process_results method."""
        # Setup mocks to return expected values
        result_processor.analyzer.analyze = Mock(return_value={"analyzed": True})
        result_processor.validator.validate = Mock(return_value={"validated": True})
        result_processor.metrics.compute = Mock(return_value={"metrics": {"time": 1.5}})

        # Mock _compile_results
        mock_compiled = {"status": "success"}
        result_processor._compile_results = Mock(return_value=mock_compiled)

        # Call the method
        result = result_processor.process_results()

        # Verify
        assert result == mock_compiled
        result_processor.analyzer.analyze.assert_called_once()
        result_processor.validator.validate.assert_called_once()
        result_processor.metrics.compute.assert_called_once()
        result_processor._compile_results.assert_called_once_with(
            {"analyzed": True}, {"validated": True}, {"metrics": {"time": 1.5}}
        )

    def test_compile_results(self, result_processor):
        """Test _compile_results method."""
        # Setup
        analysis = {"analyzed": True}
        validation = {"validated": True}
        metrics = {"time": 1.5}

        # Call the method
        result = result_processor._compile_results(analysis, validation, metrics)

        # Verify basic structure - specifics will depend on implementation
        assert isinstance(result, dict)


class TestProcessingEngine:
    """Test suite for the ProcessingEngine class."""

    @pytest.fixture
    def processing_engine(self):
        """Create a ProcessingEngine instance with mocked dependencies for testing."""
        with patch(
            "src.core.ProccessCore.AnalysisInitializer"
        ) as mock_initializer_class, patch(
            "src.core.ProccessCore.ExecutionCore"
        ) as mock_executor_class, patch(
            "src.core.ProccessCore.ResultCompiler"
        ) as mock_compiler_class:

            # Create mock instances
            mock_initializer = Mock()
            mock_executor = Mock()
            mock_compiler = Mock()

            # Configure mocks
            mock_initializer_class.return_value = mock_initializer
            mock_executor_class.return_value = mock_executor
            mock_compiler_class.return_value = mock_compiler

            # Create engine
            engine = ProcessingEngine()

            # Set mocked components for easier testing
            engine.initializer = mock_initializer
            engine.executor = mock_executor
            engine.compiler = mock_compiler

            return engine

    def test_init(self, processing_engine):
        """Test initialization of ProcessingEngine."""
        assert processing_engine.initializer is not None
        assert processing_engine.executor is not None
        assert processing_engine.compiler is not None

    def test_compute_execution_status(self, processing_engine):
        """Test _compute_execution_status method."""
        # Setup
        result = {"status": "running", "completion": 0.5}

        # Call the method
        status = processing_engine._compute_execution_status(result)

        # Verify result - specific values depend on implementation
        assert isinstance(status, dict)

    def test_execute_processing(self, processing_engine):
        """Test execute_processing method."""
        # Setup
        mock_data = Mock(spec=ValidationData)

        # Configure mock component behaviors
        processing_engine.initializer.initialize = Mock(
            return_value={"initialized": True}
        )
        processing_engine.executor.execute = Mock(return_value={"executed": True})

        mock_status = {"status": "complete"}
        processing_engine._compute_execution_status = Mock(return_value=mock_status)

        processing_engine.compiler.compile = Mock(return_value={"compiled": True})

        # Call the method
        result = processing_engine.execute_processing(mock_data)

        # Verify
        assert result == {"compiled": True}
        processing_engine.initializer.initialize.assert_called_once_with(mock_data)
        processing_engine.executor.execute.assert_called_once_with(
            {"initialized": True}
        )
        processing_engine._compute_execution_status.assert_called_once_with(
            {"executed": True}
        )
        processing_engine.compiler.compile.assert_called_once_with(
            {"executed": True}, mock_status
        )
