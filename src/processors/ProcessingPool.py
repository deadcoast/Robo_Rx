"""
ProcessingPool class
"""

import logging


class ProcessingPool:
    """ProcessingPool class."""

    def __init__(self):
        """Initialize the ProcessingPool."""
        self._logger: logging.Logger = logging.getLogger(__name__)
        self._logger.setLevel(logging.INFO)
        self._logger.addHandler(logging.StreamHandler())
        self._logger.info("ProcessingPool initialized")

    async def map(self, func, items):
        """Map a function to a list of items."""
        self._logger.info("Mapping function %s to items %s", func, items)
        results = await self._process_items(func, items)
        return await self._aggregate_results(results)

    async def _process_items(self, func, items):
        """Process a list of items."""
        self._logger.info("Processing items %s with function %s", items, func)
        return [await self._process_item(func, item) for item in items]

    async def _process_item(self, func, item):
        """Process a single item."""
        self._logger.info("Processing item %s with function %s", item, func)
        return func(item)

    async def _aggregate_results(self, results):
        """Aggregate results from processing."""
        self._logger.info("Aggregating results %s", results)
        return results

    async def __aenter__(self):
        """Enter the async context."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Exit the async context."""
        self._logger.info("ProcessingPool exiting")
