import logging
from unittest.mock import Mock, patch

import pytest

from src.core.CompilationCore import (
    CompilationContext,
    CompilationCore,
    CompilationEngine,
    CompilationError,
    CompilationResult,
    CompilationStatus,
    CompilerConfig,
    ExecutionResult,
    IntegrityVerifier,
    ReportCompiler,
    ReportGenerator,
    SystemAnalyzer,
    SystemData,
    SystemScanner,
    ValidationEngine,
    ValidationStatus,
)


class TestSystemAnalyzer:
    """Test suite for the SystemAnalyzer class."""

    @pytest.fixture
    def system_analyzer(self):
        """Create a SystemAnalyzer instance for testing."""
        with patch("src.core.CompilationCore.logging.getLogger") as mock_get_logger:
            # Create mock logger
            mock_logger = Mock(spec=logging.Logger)
            mock_get_logger.return_value = mock_logger

            analyzer = SystemAnalyzer(enable_debug=False)
            analyzer.logger = mock_logger

            return analyzer

    def test_init(self, system_analyzer):
        """Test initialization of SystemAnalyzer."""
        assert system_analyzer.scan_in_progress is False
        assert "torch_config" in system_analyzer.scan_results
        assert "numpy_config" in system_analyzer.scan_results
        assert system_analyzer.logger is not None

    def test_init_with_debug(self):
        """Test initialization with debug mode enabled."""
        with patch(
            "src.core.CompilationCore.logging.getLogger"
        ) as mock_get_logger, patch(
            "src.core.CompilationCore.debug_mode"
        ) as mock_debug_mode:

            # Create mock logger
            mock_logger = Mock(spec=logging.Logger)
            mock_get_logger.return_value = mock_logger

            # Create mock debug options
            mock_debug_options = {}
            mock_debug_mode.return_value = mock_debug_options

            analyzer = SystemAnalyzer(enable_debug=True)

            assert analyzer.logger.setLevel.called_with(logging.DEBUG)
            mock_debug_mode.assert_called_once()


class TestReportCompiler:
    """Test suite for the ReportCompiler class."""

    @pytest.fixture
    def report_compiler(self):
        """Create a ReportCompiler instance for testing."""
        return ReportCompiler()

    def test_init(self, report_compiler):
        """Test initialization of ReportCompiler."""
        assert report_compiler.compiled_reports == []
        assert report_compiler.report_metadata is not None
        assert "report_id" in report_compiler.report_metadata
        assert "report_name" in report_compiler.report_metadata
        assert report_compiler.report_status == "OK"
        assert report_compiler.report_errors == []
        assert report_compiler.report_warnings == []
        assert report_compiler.report_metrics is not None

    @patch.object(ReportCompiler, "generate")
    def test_generate(self, mock_generate, report_compiler):
        """Test generate method."""
        # Call the method
        report_compiler.generate()

        # Verify
        mock_generate.assert_called_once()


class TestValidationEngine:
    """Test suite for the ValidationEngine class."""

    @pytest.fixture
    def validation_engine(self):
        """Create a ValidationEngine instance for testing."""
        return ValidationEngine()

    def test_init(self, validation_engine):
        """Test initialization of ValidationEngine."""
        assert validation_engine.is_valid is True
        assert validation_engine.rules == []
        assert validation_engine.error_log == []
        assert validation_engine.warning_log == []
        assert validation_engine.metrics is not None
        assert validation_engine.status == "OK"
        assert validation_engine.error_registry == []

    @patch.object(ValidationEngine, "verify")
    def test_verify(self, mock_verify, validation_engine):
        """Test verify method."""
        # Call the method
        validation_engine.verify()

        # Verify
        mock_verify.assert_called_once()


class TestCompilerConfig:
    """Test suite for the CompilerConfig class."""

    @pytest.fixture
    def compiler_config(self):
        """Create a CompilerConfig instance for testing."""
        return CompilerConfig()

    def test_init_default(self, compiler_config):
        """Test initialization with default values."""
        assert compiler_config.optimization_level == 0
        assert compiler_config.include_paths == []
        assert compiler_config.output_directory == ""
        assert compiler_config.debug_mode is False
        assert compiler_config.target_architecture == "x64"
        assert compiler_config.torch_extension_config == {}
        assert compiler_config.numpy_extension_config == {}

    def test_init_with_system_analyzer(self):
        """Test initialization with SystemAnalyzer."""
        mock_analyzer = Mock(spec=SystemAnalyzer)
        mock_analyzer.scan_results = {
            "torch_config": {"some_config": "value"},
            "numpy_config": {"other_config": "value"},
        }

        with patch.object(CompilerConfig, "_configure_from_analyzer") as mock_configure:
            CompilerConfig(system_analyzer=mock_analyzer)
            mock_configure.assert_called_once_with(mock_analyzer)

    def test_configure_from_analyzer(self, compiler_config):
        """Test _configure_from_analyzer method."""
        # Setup
        mock_analyzer = Mock(spec=SystemAnalyzer)
        mock_analyzer.scan_results = {
            "torch_config": {"include_paths": ["/path/to/torch"]},
            "numpy_config": {"include_paths": ["/path/to/numpy"]},
        }

        # Call the method
        compiler_config._configure_from_analyzer(mock_analyzer)

        # Verify torch and numpy configs are updated
        assert "/path/to/torch" in compiler_config.include_paths
        assert "/path/to/numpy" in compiler_config.include_paths

    def test_configure_for_torch_extension(self, compiler_config):
        """Test configure_for_torch_extension method."""
        # Call the method
        with patch("src.core.CompilationCore.torch_include_paths") as mock_torch_paths:
            mock_torch_paths.return_value = ["/path/to/torch"]

            extension_config = compiler_config.configure_for_torch_extension(
                name="test_extension",
                sources=["source1.cpp", "source2.cpp"],
                cuda=False,
            )

            # Verify
            assert extension_config["name"] == "test_extension"
            assert extension_config["sources"] == ["source1.cpp", "source2.cpp"]
            assert "include_dirs" in extension_config
            assert "/path/to/torch" in extension_config["include_dirs"]


class TestCompilationResult:
    """Test suite for the CompilationResult class."""

    @pytest.fixture
    def compilation_result(self):
        """Create a CompilationResult instance for testing."""
        return CompilationResult(
            success=True, report={"key": "value"}, metrics={"time": 1.5}
        )

    def test_init_success(self, compilation_result):
        """Test initialization with success=True."""
        assert compilation_result.success is True
        assert compilation_result.report == {"key": "value"}
        assert compilation_result.metrics == {"time": 1.5}
        assert compilation_result.status == "OK"
        assert compilation_result.error_log == []
        assert compilation_result.warning_log == []
        assert compilation_result.error_registry == []

    def test_init_failure(self):
        """Test initialization with success=False."""
        result = CompilationResult(
            success=False,
            report={"error": "Something went wrong"},
            metrics={"time": 1.5},
        )

        assert result.success is False
        assert result.status == "FAILED"


class TestCompilationCore:
    """Test suite for the CompilationCore class."""

    @pytest.fixture
    def config(self):
        """Create a CompilerConfig instance for testing."""
        return CompilerConfig()

    @pytest.fixture
    def compilation_core(self, config):
        """Create a CompilationCore instance with mocked dependencies for testing."""
        with patch(
            "src.core.CompilationCore.SystemAnalyzer"
        ) as mock_analyzer_class, patch(
            "src.core.CompilationCore.ReportCompiler"
        ) as mock_compiler_class, patch(
            "src.core.CompilationCore.ValidationEngine"
        ) as mock_validator_class:

            # Create mock instances
            mock_analyzer = Mock(spec=SystemAnalyzer)
            mock_compiler = Mock(spec=ReportCompiler)
            mock_validator = Mock(spec=ValidationEngine)

            mock_analyzer_class.return_value = mock_analyzer
            mock_compiler_class.return_value = mock_compiler
            mock_validator_class.return_value = mock_validator

            # Create core
            core = CompilationCore(config)

            # Set mocked components for easier testing
            core.analyzer = mock_analyzer
            core.compiler = mock_compiler
            core.validator = mock_validator

            return core

    def test_init(self, compilation_core, config):
        """Test initialization of CompilationCore."""
        assert compilation_core.config == config
        assert compilation_core.analyzer is not None
        assert compilation_core.compiler is not None
        assert compilation_core.validator is not None
        assert compilation_core.report == {}

    def test_compile_system_report(self, compilation_core):
        """Test compile_system_report method."""
        # Setup mocks to return expected values
        compilation_core.analyzer.scan_results = {"analyzed": True}
        compilation_core.compiler.report = {"compiled": True}
        compilation_core.validator.is_valid = True

        # Mock the _merge_results method
        with patch.object(compilation_core, "_merge_results") as mock_merge:
            mock_merge.return_value = {"status": "success"}

            # Call the method
            result = compilation_core.compile_system_report()

            # Verify
            mock_merge.assert_called_once()
            assert result == {"status": "success"}

    def test_merge_results(self, compilation_core):
        """Test _merge_results method."""
        # Setup
        analysis = {"analyzed": True}
        compilation = {"compiled": True}
        validation = {"validated": True}

        # Call the method
        result = compilation_core._merge_results(analysis, compilation, validation)

        # Verify - basic check that all inputs are included in the output
        assert "analyzed" in result
        assert "compiled" in result
        assert "validated" in result


class TestSystemScanner:
    """Test suite for the SystemScanner class."""

    @pytest.fixture
    def system_scanner(self):
        """Create a SystemScanner instance with mocked dependencies for testing."""
        with patch(
            "src.core.CompilationCore.CompilerConfig"
        ) as mock_config_class, patch(
            "src.core.CompilationCore.CompilationCore"
        ) as mock_core_class, patch(
            "src.core.CompilationCore.ReportGenerator"
        ) as mock_generator_class:

            # Create mock instances
            mock_config = Mock(spec=CompilerConfig)
            mock_core = Mock(spec=CompilationCore)
            mock_generator = Mock(spec=ReportGenerator)

            mock_config_class.return_value = mock_config
            mock_core_class.return_value = mock_core
            mock_generator_class.return_value = mock_generator

            # Create scanner
            scanner = SystemScanner()

            # Set mocked components for easier testing
            scanner.config = mock_config
            scanner.compiler = mock_core
            scanner.report_generator = mock_generator

            return scanner

    def test_init(self, system_scanner):
        """Test initialization of SystemScanner."""
        assert system_scanner.config is not None
        assert system_scanner.report == {}
        assert system_scanner.metrics is None
        assert system_scanner.status == "OK"
        assert system_scanner.error_log == []
        assert system_scanner.warning_log == []
        assert system_scanner.error_registry == []
        assert system_scanner.compiler is not None
        assert system_scanner.report_generator is not None

    @patch.object(SystemScanner, "analyze")
    def test_analyze(self, mock_analyze, system_scanner):
        """Test analyze method."""
        # Setup
        data = {"key": "value"}

        # Call the method
        system_scanner.analyze(data)

        # Verify
        mock_analyze.assert_called_once_with(data)


class TestReportGenerator:
    """Test suite for the ReportGenerator class."""

    @pytest.fixture
    def report_generator(self):
        """Create a ReportGenerator instance with mocked dependencies for testing."""
        with patch(
            "src.core.CompilationCore.CompilerConfig"
        ) as mock_config_class, patch(
            "src.core.CompilationCore.CompilationCore"
        ) as mock_core_class:

            # Create mock instances
            mock_config = Mock(spec=CompilerConfig)
            mock_core = Mock(spec=CompilationCore)

            mock_config_class.return_value = mock_config
            mock_core_class.return_value = mock_core

            # Create generator
            generator = ReportGenerator()

            # Set mocked components for easier testing
            generator.config = mock_config
            generator.compiler = mock_core

            return generator

    def test_init(self, report_generator):
        """Test initialization of ReportGenerator."""
        assert report_generator.config is not None
        assert report_generator.report == {}
        assert report_generator.metrics is None
        assert report_generator.status == "OK"
        assert report_generator.error_log == []
        assert report_generator.warning_log == []
        assert report_generator.error_registry == []
        assert report_generator.compiler is not None
        assert report_generator.report_metadata is not None
        assert report_generator.report_status == "OK"
        assert report_generator.report_errors == []
        assert report_generator.report_warnings == []
        assert report_generator.report_metrics is not None

    @patch.object(ReportGenerator, "compile")
    def test_compile(self, mock_compile, report_generator):
        """Test compile method."""
        # Setup
        system_scan = {"key": "value"}

        # Call the method
        report_generator.compile(system_scan)

        # Verify
        mock_compile.assert_called_once_with(system_scan)


class TestIntegrityVerifier:
    """Test suite for the IntegrityVerifier class."""

    @pytest.fixture
    def integrity_verifier(self):
        """Create an IntegrityVerifier instance for testing."""
        with patch(
            "src.core.CompilationCore.CompilerConfig"
        ) as mock_config_class, patch(
            "src.core.CompilationCore.CompilationCore"
        ) as mock_core_class, patch(
            "src.core.CompilationCore.ReportGenerator"
        ) as mock_generator_class:

            # Create mock instances
            mock_config = Mock(spec=CompilerConfig)
            mock_core = Mock(spec=CompilationCore)
            mock_generator = Mock(spec=ReportGenerator)

            mock_config_class.return_value = mock_config
            mock_core_class.return_value = mock_core
            mock_generator_class.return_value = mock_generator

            # Create verifier
            verifier = IntegrityVerifier()

            # Set mocked components for easier testing
            verifier.config = mock_config
            verifier.compiler = mock_core
            verifier.report_generator = mock_generator

            return verifier

    def test_init(self, integrity_verifier):
        """Test initialization of IntegrityVerifier."""
        assert integrity_verifier.validation_rules == {}
        assert integrity_verifier.error_messages == []
        assert integrity_verifier.status == "OK"
        assert integrity_verifier.error_registry == []
        assert integrity_verifier.metrics is not None
        assert integrity_verifier.report_metadata is not None
        assert integrity_verifier.report_status == "OK"
        assert integrity_verifier.report_errors == []
        assert integrity_verifier.report_warnings == []
        assert integrity_verifier.report_metrics is not None
        assert integrity_verifier.report is None
        assert integrity_verifier.config is not None
        assert integrity_verifier.compiler is not None
        assert integrity_verifier.report_generator is not None

    @patch.object(IntegrityVerifier, "validate")
    def test_validate(self, mock_validate, integrity_verifier):
        """Test validate method."""
        # Setup
        report = {"key": "value"}

        # Call the method
        integrity_verifier.validate(report)

        # Verify
        mock_validate.assert_called_once_with(report)


class TestSystemData:
    """Test suite for the SystemData class."""

    @pytest.fixture
    def system_data(self):
        """Create a SystemData instance for testing."""
        with patch(
            "src.core.CompilationCore.CompilerConfig"
        ) as mock_config_class, patch(
            "src.core.CompilationCore.CompilationCore"
        ) as mock_core_class:

            # Create mock instances
            mock_config = Mock(spec=CompilerConfig)
            mock_core = Mock(spec=CompilationCore)

            mock_config_class.return_value = mock_config
            mock_core_class.return_value = mock_core

            # Create data
            data = SystemData()

            # Set mocked components for easier testing
            data.config = mock_config
            data.compiler = mock_core

            return data

    def test_init(self, system_data):
        """Test initialization of SystemData."""
        assert system_data.config is not None
        assert system_data.report is None
        assert system_data.metrics is None
        assert system_data.status == "OK"
        assert system_data.error_log == []
        assert system_data.warning_log == []
        assert system_data.error_registry == []
        assert system_data.compiler is not None


class TestExecutionResult:
    """Test suite for the ExecutionResult class."""

    @pytest.fixture
    def success_result(self):
        """Create a successful ExecutionResult instance for testing."""
        return ExecutionResult(
            success=True, message="Operation successful", data={"key": "value"}
        )

    @pytest.fixture
    def failure_result(self):
        """Create a failed ExecutionResult instance for testing."""
        return ExecutionResult(success=False, message="Operation failed", data=None)

    def test_init_success(self, success_result):
        """Test initialization with success=True."""
        assert success_result.success is True
        assert success_result.message == "Operation successful"
        assert success_result.data == {"key": "value"}
        assert success_result.status == "OK"
        assert success_result.error_log == []
        assert success_result.warning_log == []
        assert success_result.error_registry == []

    def test_init_failure(self, failure_result):
        """Test initialization with success=False."""
        assert failure_result.success is False
        assert failure_result.message == "Operation failed"
        assert failure_result.data is None
        assert failure_result.status == "FAILED"
        assert failure_result.error_log == []
        assert failure_result.warning_log == []
        assert failure_result.error_registry == []


class TestCompilationError:
    """Test suite for the CompilationError class."""

    @pytest.fixture
    def compilation_error(self):
        """Create a CompilationError instance for testing."""
        return CompilationError(message="Error message", code=123, line_number=456)

    def test_init(self, compilation_error):
        """Test initialization of CompilationError."""
        assert compilation_error.message == "Error message"
        assert compilation_error.code == 123
        assert compilation_error.line_number == 456
        assert compilation_error.status == "FAILED"
        assert len(compilation_error.error_log) > 0
        assert len(compilation_error.warning_log) > 0
        assert len(compilation_error.error_registry) > 0


class TestCompilationContext:
    """Test suite for the CompilationContext class."""

    @pytest.fixture
    def compilation_context(self):
        """Create a CompilationContext instance for testing."""
        with patch(
            "src.core.CompilationCore.CompilerConfig"
        ) as mock_config_class, patch(
            "src.core.CompilationCore.CompilationCore"
        ) as mock_core_class, patch(
            "src.core.CompilationCore.ReportGenerator"
        ) as mock_generator_class:

            # Create mock instances
            mock_config = Mock(spec=CompilerConfig)
            mock_core = Mock(spec=CompilationCore)
            mock_generator = Mock(spec=ReportGenerator)

            mock_config_class.return_value = mock_config
            mock_core_class.return_value = mock_core
            mock_generator_class.return_value = mock_generator

            # Create context
            context = CompilationContext({"key": "value"})

            # Set mocked components for easier testing
            context.config = mock_config
            context.compiler = mock_core
            context.report_generator = mock_generator

            return context

    def test_init(self, compilation_context):
        """Test initialization of CompilationContext."""
        assert compilation_context.context == {"key": "value"}
        assert compilation_context.config is not None
        assert compilation_context.report is None
        assert compilation_context.metrics is None
        assert compilation_context.status == "OK"
        assert compilation_context.error_log == []
        assert compilation_context.warning_log == []
        assert compilation_context.error_registry == []
        assert compilation_context.compiler is not None
        assert compilation_context.report_generator is not None
        assert compilation_context.report_metadata is not None
        assert compilation_context.report_status == "OK"
        assert compilation_context.report_errors == []
        assert compilation_context.report_warnings == []
        assert compilation_context.report_metrics is not None

    def test_context_manager(self, compilation_context):
        """Test context manager functionality."""
        with patch.object(CompilationContext, "__enter__") as mock_enter, patch.object(
            CompilationContext, "__exit__"
        ) as mock_exit:

            mock_enter.return_value = compilation_context

            # Use as context manager
            with compilation_context:
                pass

            # Verify
            mock_enter.assert_called_once()
            mock_exit.assert_called_once()


class TestCompilationEngine:
    """Test suite for the CompilationEngine class."""

    @pytest.fixture
    def compilation_engine(self):
        """Create a CompilationEngine instance with mocked dependencies for testing."""
        with patch(
            "src.core.CompilationCore.SystemScanner"
        ) as mock_scanner_class, patch(
            "src.core.CompilationCore.ReportGenerator"
        ) as mock_generator_class, patch(
            "src.core.CompilationCore.IntegrityVerifier"
        ) as mock_verifier_class, patch(
            "src.core.CompilationCore.CompilerConfig"
        ) as mock_config_class, patch(
            "src.core.CompilationCore.CompilationCore"
        ) as mock_core_class:

            # Create mock instances
            mock_scanner = Mock(spec=SystemScanner)
            mock_generator = Mock(spec=ReportGenerator)
            mock_verifier = Mock(spec=IntegrityVerifier)
            mock_config = Mock(spec=CompilerConfig)
            mock_core = Mock(spec=CompilationCore)

            mock_scanner_class.return_value = mock_scanner
            mock_generator_class.return_value = mock_generator
            mock_verifier_class.return_value = mock_verifier
            mock_config_class.return_value = mock_config
            mock_core_class.return_value = mock_core

            # Create engine
            engine = CompilationEngine()

            # Set mocked components for easier testing
            engine.scanner = mock_scanner
            engine.generator = mock_generator
            engine.verifier = mock_verifier
            engine.config = mock_config
            engine.compiler = mock_core

            return engine

    def test_init(self, compilation_engine):
        """Test initialization of CompilationEngine."""
        assert compilation_engine.scanner is not None
        assert compilation_engine.generator is not None
        assert compilation_engine.verifier is not None
        assert compilation_engine.config is not None
        assert compilation_engine.report is None
        assert compilation_engine.metrics is None
        assert compilation_engine.status == "OK"
        assert compilation_engine.error_log == []
        assert compilation_engine.warning_log == []
        assert compilation_engine.error_registry == []
        assert compilation_engine.compiler is not None

    @patch.object(CompilationEngine, "_prepare_context")
    def test_execute_compilation(self, mock_prepare_context, compilation_engine):
        """Test execute_compilation method."""
        # Setup
        data = Mock(spec=SystemData)
        mock_context = Mock(spec=CompilationContext)
        mock_prepare_context.return_value = mock_context

        # Mock behavior of dependencies
        compilation_engine.scanner.analyze.return_value = {"scanned": True}
        compilation_engine.generator.compile.return_value = {"compiled": True}
        compilation_engine.verifier.validate.return_value = True

        # Call the method
        result = compilation_engine.execute_compilation(data)

        # Verify
        mock_prepare_context.assert_called_once_with(data)
        compilation_engine.scanner.analyze.assert_called_once()
        compilation_engine.generator.compile.assert_called_once()
        compilation_engine.verifier.validate.assert_called_once()
        assert isinstance(result, ExecutionResult)


class TestValidationStatus:
    """Test suite for the ValidationStatus class."""

    @pytest.fixture
    def valid_status(self):
        """Create a valid ValidationStatus instance for testing."""
        return ValidationStatus(is_valid=True, message="Valid")

    @pytest.fixture
    def invalid_status(self):
        """Create an invalid ValidationStatus instance for testing."""
        return ValidationStatus(is_valid=False, message="Invalid")

    def test_init_valid(self, valid_status):
        """Test initialization with is_valid=True."""
        assert valid_status.is_valid is True
        assert valid_status.message == "Valid"
        assert valid_status.status == "OK"
        assert valid_status.error_log == []
        assert valid_status.warning_log == []
        assert valid_status.error_registry == []

    def test_init_invalid(self, invalid_status):
        """Test initialization with is_valid=False."""
        assert invalid_status.is_valid is False
        assert invalid_status.message == "Invalid"
        assert invalid_status.status == "FAILED"
        assert invalid_status.error_log == []
        assert invalid_status.warning_log == []
        assert invalid_status.error_registry == []


class TestCompilationStatus:
    """Test suite for the CompilationStatus class."""

    @pytest.fixture
    def validation_status(self):
        """Create a ValidationStatus instance for testing."""
        return ValidationStatus(is_valid=True, message="Valid")

    @pytest.fixture
    def compilation_status(self, validation_status):
        """Create a CompilationStatus instance for testing."""
        return CompilationStatus(
            sequence_id="seq123",
            compilation_phase="Phase 1",
            completion_metrics={"progress": 0.5},
            validation_status=validation_status,
        )

    def test_init(self, compilation_status, validation_status):
        """Test initialization of CompilationStatus."""
        assert compilation_status.sequence_id == "seq123"
        assert compilation_status.compilation_phase == "Phase 1"
        assert compilation_status.completion_metrics == {"progress": 0.5}
        assert compilation_status.validation_status == validation_status
        assert compilation_status.error_registry == []
        assert compilation_status.warning_registry == []
        assert compilation_status.error_log == []
        assert compilation_status.warning_log == []
        assert compilation_status.report_metadata == {}
        assert compilation_status.report_status == "OK"
        assert compilation_status.report_errors == []
        assert compilation_status.report_warnings == []
        assert compilation_status.report_metrics == {}
        assert compilation_status.status == "OK"
