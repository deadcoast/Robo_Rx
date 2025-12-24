import os
import pickle  # nosec B403 - Safe usage in tests only

import pytest

from managers.StateManager import StateManager

# Constants for test file paths
TEST_STATE_FILE = "test_state.pkl"

# Test IDs for parametrization
HAPPY_PATH_ID = "happy_path"
EDGE_CASE_ID = "edge_case"
ERROR_CASE_ID = "error_case"


@pytest.fixture
def state_manager():
    # Fixture to create a StateManager instance
    return StateManager()


@pytest.fixture
def cleanup_files():
    # Fixture to clean up files after tests
    yield
    if os.path.exists(TEST_STATE_FILE):
        os.remove(TEST_STATE_FILE)


@pytest.mark.parametrize(
    "test_id, state_name, state_value, expected",
    [
        (HAPPY_PATH_ID, "test_state_1", {"key": "value"}, {"key": "value"}),
        (HAPPY_PATH_ID, "test_state_2", [1, 2, 3], [1, 2, 3]),
        (EDGE_CASE_ID, "empty_state", {}, {}),
        (EDGE_CASE_ID, "none_state", None, None),
        (ERROR_CASE_ID, "", "invalid_name", None),  # Invalid state name
    ],
)
def test_add_get_state(state_manager, test_id, state_name, state_value, expected):
    # Arrange
    # Act
    state_manager.add_state(state_name, state_value)
    result = state_manager.get_state(state_name)

    # Assert
    assert (
        result == expected
    ), f"Test failed for {test_id}"  # nosec B101 - pytest requires asserts


@pytest.mark.parametrize(
    "test_id, state_name, state_value, remove_state_name, expected",
    [
        (HAPPY_PATH_ID, "test_state", "value", "test_state", None),
        (
            EDGE_CASE_ID,
            "test_state",
            "value",
            "nonexistent_state",
            "value",
        ),  # Removing non-existent state
    ],
)
def test_remove_state(
    state_manager, test_id, state_name, state_value, remove_state_name, expected
):
    # Arrange
    state_manager.add_state(state_name, state_value)

    # Act
    state_manager.remove_state(remove_state_name)
    result = state_manager.get_state(state_name)

    # Assert
    assert (
        result == expected
    ), f"Test failed for {test_id}"  # nosec B101 - pytest requires asserts


@pytest.mark.parametrize(
    "test_id, state_name, state_value, filepath, expected",
    [
        (HAPPY_PATH_ID, "test_state", {"key": "value"}, TEST_STATE_FILE, True),
        (
            ERROR_CASE_ID,
            "nonexistent_state",
            None,
            TEST_STATE_FILE,
            False,
        ),  # Saving non-existent state
    ],
)
def test_save_state(
    state_manager, cleanup_files, test_id, state_name, state_value, filepath, expected
):
    # Arrange
    if state_value is not None:
        state_manager.add_state(state_name, state_value)

    # Act
    state_manager.save_state(state_name, filepath)
    result = os.path.exists(filepath)

    # Assert
    assert (
        result == expected
    ), f"Test failed for {test_id}"  # nosec B101 - pytest requires asserts


@pytest.mark.parametrize(
    "test_id, state_name, state_value, filepath, expected",
    [
        (
            HAPPY_PATH_ID,
            "test_state",
            {"key": "value"},
            TEST_STATE_FILE,
            {"key": "value"},
        ),
        (
            ERROR_CASE_ID,
            "test_state",
            None,
            "nonexistent_file.pkl",
            None,
        ),  # Loading from non-existent file
    ],
)
def test_load_state(
    state_manager, cleanup_files, test_id, state_name, state_value, filepath, expected
):
    # Arrange
    if state_value is not None:
        with open(filepath, "wb") as f:
            pickle.dump(state_value, f)  # nosec B301 - Safe usage in tests only

    # Act
    if os.path.exists(filepath):
        state_manager.load_state(state_name, filepath)
    result = state_manager.get_state(state_name)

    # Assert
    assert (
        result == expected
    ), f"Test failed for {test_id}"  # nosec B101 - pytest requires asserts


@pytest.mark.parametrize(
    "test_id, initial_states, expected",
    [
        (
            HAPPY_PATH_ID,
            {"state1": "value1", "state2": "value2"},
            {"state1": "value1", "state2": "value2"},
        ),
        (EDGE_CASE_ID, {}, {}),  # No states
    ],
)
def test_getstate_setstate(state_manager, test_id, initial_states, expected):
    # Arrange
    for state_name, state_value in initial_states.items():
        state_manager.add_state(state_name, state_value)

    # Act
    state_manager.__setstate__(initial_states)
    result = state_manager.__getstate__()

    # Assert
    assert (
        result == expected
    ), f"Test failed for {test_id}"  # nosec B101 - pytest requires asserts
