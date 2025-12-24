from copy import copy, deepcopy

import pytest

from src.config.ResourceConfig import ResourceAllocation, ResourceConfig


class TestResourceConfig:
    """Test suite for the ResourceConfig class."""

    @pytest.fixture
    def basic_config(self):
        """Create a basic ResourceConfig instance for testing."""
        return ResourceConfig(resource_id="test_resource", amount=100.5, priority=2)

    @pytest.fixture
    def config_with_metadata(self):
        """Create a ResourceConfig instance with metadata for testing."""
        return ResourceConfig(
            resource_id="test_resource",
            amount=100.5,
            priority=2,
            metadata={"owner": "test_user", "created_at": "2023-01-01"},
        )

    def test_init_default(self, basic_config):
        """Test initialization with default metadata."""
        assert basic_config.resource_id == "test_resource"
        assert basic_config.amount == 100.5
        assert basic_config.priority == 2
        assert basic_config.metadata == {}

    def test_init_with_metadata(self, config_with_metadata):
        """Test initialization with custom metadata."""
        assert config_with_metadata.resource_id == "test_resource"
        assert config_with_metadata.amount == 100.5
        assert config_with_metadata.priority == 2
        assert config_with_metadata.metadata == {
            "owner": "test_user",
            "created_at": "2023-01-01",
        }

    def test_str(self, basic_config):
        """Test __str__ method."""
        result = str(basic_config)
        assert "ResourceConfig" in result
        assert "resource_id=test_resource" in result
        assert "amount=100.5" in result
        assert "priority=2" in result

    def test_reduce(self, config_with_metadata):
        """Test __reduce__ method."""
        cls, args = config_with_metadata.__reduce__()

        assert cls is ResourceConfig
        assert len(args) == 4
        assert args[0] == "test_resource"
        assert args[1] == 100.5
        assert args[2] == 2
        assert args[3] == {"owner": "test_user", "created_at": "2023-01-01"}

    def test_getstate(self, config_with_metadata):
        """Test __getstate__ method."""
        state = config_with_metadata.__getstate__()

        assert isinstance(state, dict)
        assert state["resource_id"] == "test_resource"
        assert state["amount"] == 100.5
        assert state["priority"] == 2
        assert state["metadata"] == {"owner": "test_user", "created_at": "2023-01-01"}

    def test_setstate(self):
        """Test __setstate__ method."""
        config = ResourceConfig(resource_id="temp", amount=0)

        state = {
            "resource_id": "test_resource",
            "amount": 100.5,
            "priority": 2,
            "metadata": {"key": "value"},
        }

        config.__setstate__(state)

        assert config.resource_id == "test_resource"
        assert config.amount == 100.5
        assert config.priority == 2
        assert config.metadata == {"key": "value"}

    def test_equality(self, basic_config):
        """Test equality methods."""
        # Same values
        identical_config = ResourceConfig(
            resource_id="test_resource", amount=100.5, priority=2
        )

        # Different values
        different_config = ResourceConfig(
            resource_id="other_resource", amount=200.0, priority=1
        )

        # Test equality
        assert basic_config == identical_config
        assert basic_config != different_config
        assert basic_config != "not a resource config"

        # Test inequality
        assert basic_config == identical_config
        assert basic_config != different_config

    def test_hash(self, basic_config, config_with_metadata):
        """Test __hash__ method."""
        # Same values should have same hash
        identical_config = ResourceConfig(
            resource_id="test_resource", amount=100.5, priority=2
        )

        assert hash(basic_config) == hash(identical_config)

        # Different values should have different hash
        assert hash(basic_config) != hash(config_with_metadata)

    def test_copy(self, config_with_metadata):
        """Test __copy__ method."""
        copied = copy(config_with_metadata)

        # Verify it's a proper copy
        assert copied is not config_with_metadata
        assert copied == config_with_metadata

        # Verify metadata is copied, not referenced
        assert copied.metadata is not config_with_metadata.metadata

        # Modifying metadata in one should not affect the other
        copied.metadata["new_key"] = "new_value"
        assert "new_key" not in config_with_metadata.metadata

    def test_deepcopy(self, config_with_metadata):
        """Test __deepcopy__ method."""
        copied = deepcopy(config_with_metadata)

        # Verify it's a proper copy
        assert copied is not config_with_metadata
        assert copied == config_with_metadata

        # Verify metadata is copied, not referenced
        assert copied.metadata is not config_with_metadata.metadata

        # Modifying metadata in one should not affect the other
        copied.metadata["new_key"] = "new_value"
        assert "new_key" not in config_with_metadata.metadata


class TestResourceAllocation:
    """Test suite for the ResourceAllocation class."""

    @pytest.fixture
    def basic_allocation(self):
        """Create a basic ResourceAllocation instance for testing."""
        return ResourceAllocation(resource_id="test_resource", amount=100.5, priority=2)

    @pytest.fixture
    def allocation_with_metadata(self):
        """Create a ResourceAllocation instance with metadata for testing."""
        return ResourceAllocation(
            resource_id="test_resource",
            amount=100.5,
            priority=2,
            metadata={"requester": "test_process", "timestamp": "2023-01-01"},
        )

    def test_init_default(self, basic_allocation):
        """Test initialization with default metadata."""
        assert basic_allocation.resource_id == "test_resource"
        assert basic_allocation.amount == 100.5
        assert basic_allocation.priority == 2
        assert basic_allocation.metadata == {}

    def test_init_with_metadata(self, allocation_with_metadata):
        """Test initialization with custom metadata."""
        assert allocation_with_metadata.resource_id == "test_resource"
        assert allocation_with_metadata.amount == 100.5
        assert allocation_with_metadata.priority == 2
        assert allocation_with_metadata.metadata == {
            "requester": "test_process",
            "timestamp": "2023-01-01",
        }

    def test_dataclass_properties(self, basic_allocation):
        """Test that ResourceAllocation has dataclass properties."""
        # Check that we can access attributes directly
        assert basic_allocation.resource_id == "test_resource"
        assert basic_allocation.amount == 100.5

        # Check that we can modify attributes
        basic_allocation.amount = 200.0
        assert basic_allocation.amount == 200.0

        # Check equality comparison (auto-generated by dataclass)
        identical = ResourceAllocation(
            resource_id="test_resource", amount=200.0, priority=2  # Updated value
        )
        assert basic_allocation == identical
