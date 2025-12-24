from unittest.mock import AsyncMock

import pytest
from pytest_cases import case, parametrize_with_cases

from src.analyzers.AnalyzerCore import AnalyzerCore, ResultAnalyzer

# Define modular test cases using pytest-cases
# This approach separates test data from test logic for better organization


class HappyPathCases:
    """Happy path test cases for analyzer methods"""

    @case(id="analyze_normal_data")
    def case_analyze(self):
        return "analyze", "data_to_analyze", "analysis_result"

    @case(id="validate_normal_data")
    def case_validate(self):
        return "validate", "data_to_validate", "validation_result"

    @case(id="compile_normal_data")
    def case_compile(self):
        return "compile", "data_to_compile", "compilation_result"

    @case(id="execute_normal_data")
    def case_execute(self):
        return "execute", "data_to_execute", "execution_result"

    @case(id="finalize_normal_data")
    def case_finalize(self):
        return "finalize", "data_to_finalize", "finalization_result"

    @case(id="compute_normal_data")
    def case_compute(self):
        return "compute", "data_to_compute", "computation_result"

    @case(id="process_normal_data")
    def case_process(self):
        return "process", "data_to_process", "processing_result"

    @case(id="generate_report_normal_data")
    def case_generate_report(self):
        return "generate_report", "data_for_report", "report_generated"

    @case(id="save_report_normal_data")
    def case_save_report(self):
        return "save_report", "report_data", "report_saved"

    @case(id="send_report_normal_data")
    def case_send_report(self):
        return "send_report", "report_to_send", "report_sent"


# Happy path tests with various realistic test values
@pytest.mark.asyncio
@parametrize_with_cases("method, input_value, expected_output", cases=HappyPathCases)
async def test_happy_path(method, input_value, expected_output):
    # Arrange
    analyzer = AsyncMock(spec=ResultAnalyzer)
    getattr(analyzer, method).return_value = expected_output
    analyzer_core = AnalyzerCore(analyzer)

    # Act
    result = await getattr(analyzer_core, method)(input_value)

    # Assert
    getattr(analyzer, method).assert_called_once_with(input_value)
    assert result == expected_output


# Define edge case test cases
class EdgeCases:
    """Edge case test cases for analyzer methods"""

    @case(id="analyze_empty_string")
    def case_analyze_empty(self):
        return "analyze", ""

    @case(id="validate_none")
    def case_validate_none(self):
        return "validate", None

    @case(id="compile_empty_list")
    def case_compile_empty_list(self):
        return "compile", []

    @case(id="execute_empty_dict")
    def case_execute_empty_dict(self):
        return "execute", {}

    @case(id="finalize_zero")
    def case_finalize_zero(self):
        return "finalize", 0

    @case(id="compute_false")
    def case_compute_false(self):
        return "compute", False

    @case(id="process_true")
    def case_process_true(self):
        return "process", True

    @case(id="generate_report_empty_set")
    def case_generate_report_empty_set(self):
        return "generate_report", set()

    @case(id="save_report_empty_tuple")
    def case_save_report_empty_tuple(self):
        return "save_report", ()

    @case(id="send_report_nan")
    def case_send_report_nan(self):
        return "send_report", float("nan")


# Edge cases test
@pytest.mark.asyncio
@parametrize_with_cases("method, input_value", cases=EdgeCases)
async def test_edge_cases(method, input_value):
    # Arrange
    analyzer = AsyncMock(spec=ResultAnalyzer)
    analyzer_core = AnalyzerCore(analyzer)

    # Act
    result = await getattr(analyzer_core, method)(input_value)

    # Assert
    getattr(analyzer, method).assert_called_once_with(input_value)
    # Assuming the analyzer methods handle edge cases gracefully and return a default value or None
    assert result is None or isinstance(result, (str, list, dict, int, bool))


# Define error test cases
class ErrorCases:
    """Error test cases for analyzer methods"""

    @case(id="analyze_runtime_error")
    def case_analyze_error(self):
        return "analyze", "invalid_data", RuntimeError("Analysis failed")

    @case(id="validate_value_error")
    def case_validate_error(self):
        return "validate", "invalid_data", ValueError("Validation failed")

    @case(id="compile_type_error")
    def case_compile_error(self):
        return "compile", "invalid_data", TypeError("Compilation failed")

    @case(id="execute_general_exception")
    def case_execute_error(self):
        return "execute", "invalid_data", Exception("Execution failed")

    @case(id="finalize_buffer_error")
    def case_finalize_error(self):
        return "finalize", "invalid_data", BufferError("Finalization failed")

    @case(id="compute_arithmetic_error")
    def case_compute_error(self):
        return "compute", "invalid_data", ArithmeticError("Computation failed")

    @case(id="process_environment_error")
    def case_process_error(self):
        return "process", "invalid_data", EnvironmentError("Processing failed")

    @case(id="generate_report_io_error")
    def case_generate_report_error(self):
        return "generate_report", "invalid_data", IOError("Report generation failed")

    @case(id="save_report_eof_error")
    def case_save_report_error(self):
        return "save_report", "invalid_data", EOFError("Saving failed")

    @case(id="send_report_connection_error")
    def case_send_report_error(self):
        return "send_report", "invalid_data", ConnectionError("Sending failed")


# Error cases test
@pytest.mark.asyncio
@parametrize_with_cases("method, input_value, exception", cases=ErrorCases)
async def test_error_cases(method, input_value, exception):
    # Arrange
    analyzer = AsyncMock(spec=ResultAnalyzer)
    getattr(analyzer, method).side_effect = exception
    analyzer_core = AnalyzerCore(analyzer)

    # Act & Assert
    with pytest.raises(type(exception)):
        await getattr(analyzer_core, method)(input_value)
    getattr(analyzer, method).assert_called_once_with(input_value)
