from unittest.mock import Mock, patch

import pytest

from src.config.SystemConfig import SystemConfig


class TestSystemConfig:
    """Tests for the SystemConfig class."""

    @pytest.fixture
    def system_config(self):
        """Create a SystemConfig instance for testing."""
        return SystemConfig()

    def test_init(self, system_config):
        """Test initialization of SystemConfig."""
        assert system_config is not None
        assert hasattr(system_config, "config_data")

    @patch("os.path.exists")
    def test_load_config_file_exists(self, mock_exists, system_config):
        """Test loading configuration when file exists."""
        # Setup
        mock_exists.return_value = True
        mock_open = Mock()
        mock_open.return_value.__enter__.return_value.read.return_value = "{}"

        with patch("builtins.open", mock_open):
            # Call
            result = system_config.load_config("config.json")

            # Assert
            assert result is True
            mock_exists.assert_called_once_with("config.json")
            mock_open.assert_called_once_with("config.json", "r")

    @patch("os.path.exists")
    def test_load_config_file_not_exists(self, mock_exists, system_config):
        """Test loading configuration when file doesn't exist."""
        # Setup
        mock_exists.return_value = False

        # Call
        result = system_config.load_config("nonexistent.json")

        # Assert
        assert result is False
        mock_exists.assert_called_once_with("nonexistent.json")

    def test_get_config_value(self, system_config):
        """Test getting a configuration value."""
        # Setup
        system_config.config_data = {"section": {"key": "value"}}

        # Call & Assert
        assert system_config.get_config_value("section", "key") == "value"
        assert (
            system_config.get_config_value("section", "nonexistent", "default")
            == "default"
        )
        assert (
            system_config.get_config_value("nonexistent", "key", "default") == "default"
        )

    def test_set_config_value(self, system_config):
        """Test setting a configuration value."""
        # Setup
        system_config.config_data = {"section": {}}

        # Call
        system_config.set_config_value("section", "key", "value")

        # Assert
        assert system_config.config_data["section"]["key"] == "value"

        # Test creating a new section
        system_config.set_config_value("new_section", "key", "value")
        assert system_config.config_data["new_section"]["key"] == "value"

    @patch("os.path.exists")
    @patch("os.makedirs")
    def test_save_config(self, mock_makedirs, mock_exists, system_config):
        """Test saving configuration."""
        # Setup
        system_config.config_data = {"section": {"key": "value"}}
        mock_exists.return_value = False
        mock_open = Mock()

        with patch("builtins.open", mock_open):
            # Call
            system_config.save_config("dir/config.json")

            # Assert
            mock_exists.assert_called_once_with("dir")
            mock_makedirs.assert_called_once_with("dir")
            mock_open.assert_called_once_with("dir/config.json", "w")
            mock_open.return_value.__enter__.return_value.write.assert_called_once()

    def test_validate_config(self, system_config):
        """Test configuration validation."""
        # Setup
        valid_config = {"required_section": {"required_key": "value"}}
        invalid_config = {"section": {"key": "value"}}

        # Call & Assert with valid config
        system_config.config_data = valid_config
        assert (
            system_config.validate_config(["required_section"], ["required_key"])
            is True
        )

        # Call & Assert with invalid config
        system_config.config_data = invalid_config
        assert (
            system_config.validate_config(["required_section"], ["required_key"])
            is False
        )
