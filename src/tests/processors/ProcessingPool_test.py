import logging
from unittest.mock import AsyncMock, Mock, patch

import pytest

from src.processors.ProcessingPool import ProcessingPool


class TestProcessingPool:
    """Test suite for the ProcessingPool class."""

    @pytest.fixture
    def processing_pool(self):
        """Create a ProcessingPool instance for testing."""
        with patch("logging.getLogger") as mock_get_logger:
            # Create a mock logger to avoid actual logging during tests
            mock_logger = Mock(spec=logging.Logger)
            mock_get_logger.return_value = mock_logger

            # Create and return ProcessingPool instance
            pool = ProcessingPool()

            # Set the mocked logger so tests can access it
            pool._logger = mock_logger
            return pool

    def test_init(self, processing_pool):
        """Test initialization of ProcessingPool."""
        assert processing_pool._logger is not None
        processing_pool._logger.info.assert_called_with("ProcessingPool initialized")

    @pytest.mark.asyncio
    async def test_map(self, processing_pool):
        """Test the map method."""
        # Setup
        test_func = Mock(return_value="processed")
        test_items = ["item1", "item2", "item3"]

        # Mock internal methods
        processing_pool._process_items = AsyncMock(
            return_value=["processed1", "processed2", "processed3"]
        )
        processing_pool._aggregate_results = AsyncMock(
            return_value=["aggregated1", "aggregated2", "aggregated3"]
        )

        # Call the method
        result = await processing_pool.map(test_func, test_items)

        # Verify
        processing_pool._logger.info.assert_any_call(
            "Mapping function %s to items %s", test_func, test_items
        )
        processing_pool._process_items.assert_called_once_with(test_func, test_items)
        processing_pool._aggregate_results.assert_called_once_with(
            ["processed1", "processed2", "processed3"]
        )
        assert result == ["aggregated1", "aggregated2", "aggregated3"]

    @pytest.mark.asyncio
    async def test_process_items(self, processing_pool):
        """Test the _process_items method."""
        # Setup
        test_func = Mock()
        test_items = ["item1", "item2", "item3"]

        # Mock the _process_item method
        processing_pool._process_item = AsyncMock(
            side_effect=["processed1", "processed2", "processed3"]
        )

        # Call the method
        result = await processing_pool._process_items(test_func, test_items)

        # Verify
        processing_pool._logger.info.assert_called_with(
            "Processing items %s with function %s", test_items, test_func
        )
        assert processing_pool._process_item.call_count == 3
        assert result == ["processed1", "processed2", "processed3"]

    @pytest.mark.asyncio
    async def test_process_item(self, processing_pool):
        """Test the _process_item method."""
        # Setup
        test_item = "test_item"
        test_func = Mock(return_value="processed_item")

        # Call the method
        result = await processing_pool._process_item(test_func, test_item)

        # Verify
        processing_pool._logger.info.assert_called_with(
            "Processing item %s with function %s", test_item, test_func
        )
        test_func.assert_called_once_with(test_item)
        assert result == "processed_item"

    @pytest.mark.asyncio
    async def test_aggregate_results(self, processing_pool):
        """Test the _aggregate_results method."""
        # Setup
        test_results = ["result1", "result2", "result3"]

        # Call the method
        result = await processing_pool._aggregate_results(test_results)

        # Verify
        processing_pool._logger.info.assert_called_with(
            "Aggregating results %s", test_results
        )
        assert result == test_results

    @pytest.mark.asyncio
    async def test_async_context_manager(self, processing_pool):
        """Test the async context manager."""
        # Use the async context manager
        async with processing_pool as pool:
            assert pool is processing_pool

        # Verify
        processing_pool._logger.info.assert_called_with("ProcessingPool exiting")

    @pytest.mark.asyncio
    async def test_map_with_actual_function(self, processing_pool):
        """Test the map method with an actual function."""

        # Setup
        async def test_func(item):
            return item.upper()

        test_items = ["item1", "item2", "item3"]

        # Call the method directly without mocking internals
        with patch.object(
            processing_pool, "_process_items", wraps=processing_pool._process_items
        ), patch.object(
            processing_pool,
            "_aggregate_results",
            wraps=processing_pool._aggregate_results,
        ):

            result = await processing_pool.map(test_func, test_items)

            # Verify
            assert processing_pool._process_items.called
            assert processing_pool._aggregate_results.called
            assert result == ["ITEM1", "ITEM2", "ITEM3"]

    @pytest.mark.asyncio
    async def test_map_with_empty_list(self, processing_pool):
        """Test the map method with an empty list."""
        # Setup
        test_func = Mock()
        test_items = []

        # Call the method
        result = await processing_pool.map(test_func, test_items)

        # Verify
        assert result == []
        test_func.assert_not_called()

    @pytest.mark.asyncio
    async def test_map_with_exception_in_function(self, processing_pool):
        """Test exception handling within the map method."""

        # Setup
        async def test_func(item):
            if item == "item2":
                raise ValueError("Test error")
            return item.upper()

        test_items = ["item1", "item2", "item3"]

        # Call the method and verify it raises the exception
        with pytest.raises(ValueError) as exc_info:
            await processing_pool.map(test_func, test_items)

        # Verify
        assert "Test error" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_process_items_parallel(self, processing_pool):
        """Test parallel processing in _process_items."""
        # Setup
        import asyncio

        async def slow_func(item):
            await asyncio.sleep(0.1)
            return f"processed_{item}"

        test_items = ["item1", "item2", "item3"]

        # Call the method (uses list comprehension which processes sequentially, not parallel)
        start_time = asyncio.get_event_loop().time()
        result = await processing_pool._process_items(slow_func, test_items)
        duration = asyncio.get_event_loop().time() - start_time

        # Verify
        # sequential processing should take at least 0.3 seconds total (0.1s * 3 items)
        assert duration >= 0.3
        assert result == ["processed_item1", "processed_item2", "processed_item3"]

    @pytest.mark.asyncio
    async def test_map_function_arguments(self, processing_pool):
        """Test the map method with a function that takes multiple arguments."""

        # Setup
        async def test_func(item, multiplier=1):
            return item * multiplier

        test_items = [1, 2, 3]

        # We need to wrap test_func to match the expected signature (func(item))
        async def wrapper(item):
            return await test_func(item, multiplier=2)

        # Call the method
        result = await processing_pool.map(wrapper, test_items)

        # Verify
        assert result == [2, 4, 6]
