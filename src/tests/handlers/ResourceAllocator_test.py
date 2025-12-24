from unittest.mock import Mock, patch

import pytest

from src.config.SystemConfig import SystemConfig
from src.config.TaskChainConfig import TaskChainConfig
from src.ResourceAllocator import ResourceAllocator


class TestResourceAllocator:
    """Test suite for the ResourceAllocator class."""

    @pytest.fixture
    def mock_system_config(self):
        """Create a mock SystemConfig for testing."""
        return Mock(spec=SystemConfig)

    @pytest.fixture
    def resource_allocator(self, mock_system_config):
        """Create a ResourceAllocator instance for testing."""
        return ResourceAllocator(config=mock_system_config)

    @pytest.fixture
    def populated_allocator(self, mock_system_config):
        """Create a ResourceAllocator with pre-populated resources."""
        allocator = ResourceAllocator(config=mock_system_config)

        # Add test resources
        allocator.resource_registry = {
            "cpu": 100.0,
            "memory": 1024.0,
            "storage": 2048.0,
        }

        # Set available resources (initially equal to registered)
        allocator.available_resources = {
            "cpu": 100.0,
            "memory": 1024.0,
            "storage": 2048.0,
        }

        # No allocations initially
        allocator.allocated_resources = {"cpu": 0.0, "memory": 0.0, "storage": 0.0}

        return allocator

    def test_init(self, resource_allocator, mock_system_config):
        """Test initialization of ResourceAllocator."""
        assert resource_allocator.config == mock_system_config
        assert resource_allocator.resource_registry == {}
        assert resource_allocator.available_resources == {}
        assert resource_allocator.allocated_resources == {}
        assert resource_allocator.allocation_history == []

    def test_allocate_success(self, populated_allocator):
        """Test successful resource allocation."""
        # Allocate CPU resource
        result = populated_allocator.allocate("cpu", 50.0, 1)

        # Verify allocation was successful
        assert result is True
        assert populated_allocator.available_resources["cpu"] == 50.0
        assert populated_allocator.allocated_resources["cpu"] == 50.0

        # Verify allocation history has an entry
        assert len(populated_allocator.allocation_history) == 1
        allocation = populated_allocator.allocation_history[0]
        assert allocation.resource_id == "cpu"
        assert allocation.amount == 50.0
        assert allocation.priority == 1

    def test_allocate_nonexistent_resource(self, resource_allocator):
        """Test allocation of a nonexistent resource."""
        result = resource_allocator.allocate("nonexistent", 10.0)
        assert result is False
        assert len(resource_allocator.allocation_history) == 0

    def test_allocate_insufficient_resource(self, populated_allocator):
        """Test allocation when requested amount exceeds available."""
        result = populated_allocator.allocate("memory", 2000.0)  # More than available
        assert result is False
        assert populated_allocator.available_resources["memory"] == 1024.0  # Unchanged
        assert populated_allocator.allocated_resources["memory"] == 0.0  # Unchanged
        assert len(populated_allocator.allocation_history) == 0

    def test_allocate_multiple_times(self, populated_allocator):
        """Test multiple allocations of the same resource."""
        # First allocation
        result1 = populated_allocator.allocate("cpu", 30.0)
        assert result1 is True
        assert populated_allocator.available_resources["cpu"] == 70.0
        assert populated_allocator.allocated_resources["cpu"] == 30.0

        # Second allocation
        result2 = populated_allocator.allocate("cpu", 40.0)
        assert result2 is True
        assert populated_allocator.available_resources["cpu"] == 30.0
        assert populated_allocator.allocated_resources["cpu"] == 70.0

        # Verify allocation history
        assert len(populated_allocator.allocation_history) == 2

    def test_deallocate_success(self, populated_allocator):
        """Test successful resource deallocation."""
        # First allocate
        populated_allocator.allocate("memory", 512.0)
        assert populated_allocator.available_resources["memory"] == 512.0
        assert populated_allocator.allocated_resources["memory"] == 512.0

        # Then deallocate
        result = populated_allocator.deallocate("memory", 512.0)

        # Verify deallocation was successful
        assert result is True
        assert populated_allocator.available_resources["memory"] == 1024.0
        assert populated_allocator.allocated_resources["memory"] == 0.0

    def test_deallocate_nonexistent_resource(self, resource_allocator):
        """Test deallocation of a nonexistent resource."""
        result = resource_allocator.deallocate("nonexistent", 10.0)
        assert result is False

    def test_deallocate_too_much(self, populated_allocator):
        """Test deallocation when requested amount exceeds allocated."""
        # Allocate memory
        populated_allocator.allocate("memory", 200.0)

        # Try to deallocate more than allocated
        result = populated_allocator.deallocate("memory", 300.0)

        # Verify deallocation failed
        assert result is False
        # Verify resources remain unchanged
        assert populated_allocator.available_resources["memory"] == 824.0
        assert populated_allocator.allocated_resources["memory"] == 200.0

    def test_get_available(self, populated_allocator):
        """Test getting available resource amount."""
        assert populated_allocator.get_available("cpu") == 100.0
        assert populated_allocator.get_available("nonexistent") == 0.0

    def test_get_allocated(self, populated_allocator):
        """Test getting allocated resource amount."""
        # Allocate some resources
        populated_allocator.allocate("storage", 1000.0)

        assert populated_allocator.get_allocated("storage") == 1000.0
        assert populated_allocator.get_allocated("nonexistent") == 0.0

    def test_add_resource(self, resource_allocator):
        """Test adding a new resource."""
        # Add new resource
        resource_allocator.add_resource("gpu", 2.0)

        # Verify resource was added
        assert "gpu" in resource_allocator.available_resources
        assert resource_allocator.available_resources["gpu"] == 2.0

        # Test adding more to existing resource
        resource_allocator.add_resource("gpu", 2.0)
        assert resource_allocator.available_resources["gpu"] == 4.0

    @patch("logging.getLogger")
    def test_post_init(self, mock_get_logger, mock_system_config):
        """Test the __post_init__ method."""
        # Setup
        mock_logger = Mock()
        mock_get_logger.return_value = mock_logger

        # Create ResourceAllocator to trigger __post_init__
        resource_allocator = ResourceAllocator(config=mock_system_config)

        # Verify logger was created
        assert resource_allocator.logger is mock_logger

    def test_allocate_resources_task_chain(self, populated_allocator):
        """Test allocating resources for a task chain."""
        # Create a mock task chain
        mock_task_chain = Mock(spec=TaskChainConfig)

        # Call method
        result = populated_allocator.allocate_resources(mock_task_chain)

        # Method is a stub returning empty dict
        assert result == {}

    def test_release_resources_task_chain(self, populated_allocator):
        """Test releasing resources for a task chain."""
        # Create a mock task chain
        mock_task_chain = Mock(spec=TaskChainConfig)

        # Call method
        populated_allocator.release_resources(mock_task_chain)

        # Mostly verifying it doesn't error since it's a stub
        # No assertions since there's no return or side effects

    def test_string_representation(self, resource_allocator):
        """Test string representation of ResourceAllocator."""
        str_repr = str(resource_allocator)
        assert "ResourceAllocator" in str_repr
        assert str(resource_allocator.config) in str_repr
