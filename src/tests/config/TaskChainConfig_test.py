import copy
import pickle

import pytest

from src.config.TaskChainConfig import TaskChainConfig


class TestTaskChainConfig:
    """Test suite for the TaskChainConfig class."""

    @pytest.fixture
    def task_chain_config(self):
        """Create a TaskChainConfig instance for testing."""
        return TaskChainConfig(
            task_chain_id="test-id",
            task_chain_name="Test Chain",
            task_chain_description="A test task chain",
            task_chain_priority=5,
            task_chain_metadata={"test_key": "test_value"},
        )

    def test_init(self, task_chain_config):
        """Test initialization of TaskChainConfig."""
        assert task_chain_config.task_chain_id == "test-id"
        assert task_chain_config.task_chain_name == "Test Chain"
        assert task_chain_config.task_chain_description == "A test task chain"
        assert task_chain_config.task_chain_priority == 5
        assert task_chain_config.task_chain_metadata == {"test_key": "test_value"}

    def test_str(self, task_chain_config):
        """Test string representation of TaskChainConfig."""
        string_rep = str(task_chain_config)
        assert "test-id" in string_rep
        assert "Test Chain" in string_rep
        assert "A test task chain" in string_rep
        assert "5" in string_rep
        assert "test_key" in string_rep
        assert "test_value" in string_rep

    def test_reduce(self, task_chain_config):
        """Test __reduce__ method of TaskChainConfig."""
        reduce_result = task_chain_config.__reduce__()
        assert reduce_result[0] == TaskChainConfig
        assert reduce_result[1][0] == "test-id"
        assert reduce_result[1][1] == "Test Chain"
        assert reduce_result[1][2] == "A test task chain"
        assert reduce_result[1][3] == 5
        assert reduce_result[1][4] == {"test_key": "test_value"}

    def test_getstate(self, task_chain_config):
        """Test __getstate__ method of TaskChainConfig."""
        state = task_chain_config.__getstate__()
        assert state["task_chain_id"] == "test-id"
        assert state["task_chain_name"] == "Test Chain"
        assert state["task_chain_description"] == "A test task chain"
        assert state["task_chain_priority"] == 5
        assert state["task_chain_metadata"] == {"test_key": "test_value"}

    def test_setstate(self, task_chain_config):
        """Test __setstate__ method of TaskChainConfig."""
        new_state = {
            "task_chain_id": "new-id",
            "task_chain_name": "New Chain",
            "task_chain_description": "New description",
            "task_chain_priority": 10,
            "task_chain_metadata": {"new_key": "new_value"},
        }
        task_chain_config.__setstate__(new_state)
        assert task_chain_config.task_chain_id == "new-id"
        assert task_chain_config.task_chain_name == "New Chain"
        assert task_chain_config.task_chain_description == "New description"
        assert task_chain_config.task_chain_priority == 10
        assert task_chain_config.task_chain_metadata == {"new_key": "new_value"}

    def test_eq(self, task_chain_config):
        """Test equality comparison."""
        # Create an identical config
        identical_config = TaskChainConfig(
            task_chain_id="test-id",
            task_chain_name="Test Chain",
            task_chain_description="A test task chain",
            task_chain_priority=5,
            task_chain_metadata={"test_key": "test_value"},
        )

        # Create a different config
        different_config = TaskChainConfig(
            task_chain_id="different-id",
            task_chain_name="Different Chain",
            task_chain_description="A different task chain",
            task_chain_priority=10,
            task_chain_metadata={"different_key": "different_value"},
        )

        assert task_chain_config == identical_config
        assert task_chain_config != different_config
        assert task_chain_config != "not-a-task-chain"

    def test_hash(self, task_chain_config):
        """Test hash method."""
        # Create an identical config
        identical_config = TaskChainConfig(
            task_chain_id="test-id",
            task_chain_name="Test Chain",
            task_chain_description="A test task chain",
            task_chain_priority=5,
            task_chain_metadata={"test_key": "test_value"},
        )

        assert hash(task_chain_config) == hash(identical_config)

    def test_copy(self, task_chain_config):
        """Test copy method."""
        config_copy = copy.copy(task_chain_config)
        assert config_copy == task_chain_config
        assert config_copy is not task_chain_config

        # Modify the copy and verify the original is unchanged
        config_copy.task_chain_name = "Modified"
        assert task_chain_config.task_chain_name == "Test Chain"

    def test_deepcopy(self, task_chain_config):
        """Test deepcopy method."""
        config_deepcopy = copy.deepcopy(task_chain_config)
        assert config_deepcopy == task_chain_config
        assert config_deepcopy is not task_chain_config

        # Modify the copied metadata and verify the original is unchanged
        config_deepcopy.task_chain_metadata["new_key"] = "new_value"
        assert "new_key" not in task_chain_config.task_chain_metadata

    def test_pickle_unpickle(self, task_chain_config):
        """Test pickling and unpickling."""
        # Pickle the object
        pickled_data = pickle.dumps(task_chain_config)

        # Unpickle the object
        unpickled_config = pickle.loads(pickled_data)

        # Verify it's the same
        assert unpickled_config == task_chain_config
        assert unpickled_config is not task_chain_config
