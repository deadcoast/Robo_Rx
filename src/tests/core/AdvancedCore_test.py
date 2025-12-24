import logging
from datetime import datetime
from queue import Queue
from unittest.mock import AsyncMock, Mock, patch

import numpy as np
import pytest

from src.core.AdvancedCore import (
    AdvancedFeatureProcessor,
    MetadataFeatures,
    SizeStatistics,
    Task,
    TaskRegistration,
    TaskRegistryManager,
    TimestampData,
)
from src.engines.EngineConfig import EngineConfig
from src.SystemConfig import SystemConfig


class TestTimestampData:
    """Test suite for the TimestampData TypedDict."""

    def test_valid_structure(self):
        """Test creating a valid TimestampData object."""
        data: TimestampData = {
            "earliest": datetime(2022, 1, 1),
            "latest": datetime(2022, 12, 31),
            "distribution": {"Jan": 5, "Feb": 10},
        }

        assert "earliest" in data
        assert "latest" in data
        assert "distribution" in data
        assert isinstance(data["distribution"], dict)


class TestSizeStatistics:
    """Test suite for the SizeStatistics TypedDict."""

    def test_valid_structure(self):
        """Test creating a valid SizeStatistics object."""
        data: SizeStatistics = {
            "min_size": 100.5,
            "max_size": 1000,
            "avg_size": 500.75,
            "total_size": 5000,
        }

        assert "min_size" in data
        assert "max_size" in data
        assert "avg_size" in data
        assert "total_size" in data


class TestMetadataFeatures:
    """Test suite for the MetadataFeatures TypedDict."""

    def test_valid_structure(self):
        """Test creating a valid MetadataFeatures object."""
        timestamp_data: TimestampData = {
            "earliest": datetime(2022, 1, 1),
            "latest": datetime(2022, 12, 31),
            "distribution": {"Jan": 5, "Feb": 10},
        }

        size_stats: SizeStatistics = {
            "min_size": 100.5,
            "max_size": 1000,
            "avg_size": 500.75,
            "total_size": 5000,
        }

        data: MetadataFeatures = {
            "document_count": 150,
            "categories": {"tech": 50, "science": 100},
            "authors": {"Alice": 75, "Bob": 75},
            "timestamps": timestamp_data,
            "size_statistics": size_stats,
            "tag_frequency": {"python": 35, "data": 42, "ml": 28},
        }

        assert "document_count" in data
        assert "categories" in data
        assert "authors" in data
        assert "timestamps" in data
        assert "size_statistics" in data
        assert "tag_frequency" in data


class TestTask:
    """Test suite for the Task dataclass."""

    def test_init_default(self):
        """Test initialization with default values."""
        task = Task(id="test123", type="processing")

        assert task.id == "test123"
        assert task.type == "processing"
        assert task.status == "pending"
        assert task.weight == 1.0
        assert task.context == {}
        assert task.registration == {}
        assert isinstance(task.timestamp, str)

    def test_init_custom(self):
        """Test initialization with custom values."""
        context = {"key": "value"}
        registration = {"reg_key": "reg_value"}

        task = Task(
            id="test456",
            type="analysis",
            status="running",
            weight=2.5,
            context=context,
            registration=registration,
            timestamp="2023-01-01 12:00:00",
        )

        assert task.id == "test456"
        assert task.type == "analysis"
        assert task.status == "running"
        assert task.weight == 2.5
        assert task.context == context
        assert task.registration == registration
        assert task.timestamp == "2023-01-01 12:00:00"


class TestTaskRegistration:
    """Test suite for the TaskRegistration dataclass."""

    def test_init(self):
        """Test initialization with required values."""
        task = Task(id="test123", type="processing")
        registration = TaskRegistration(task_id="test123", task=task)

        assert registration.task_id == "test123"
        assert registration.task is task


class TestAdvancedFeatureProcessor:
    """Test suite for the AdvancedFeatureProcessor class."""

    @pytest.fixture
    def config(self):
        """Create a SystemConfig instance for testing."""
        return Mock(spec=SystemConfig)

    @pytest.fixture
    def processor(self, config):
        """Create an AdvancedFeatureProcessor instance with mocked dependencies for testing."""
        with patch(
            "src.core.AdvancedCore.TopicModelingEngine"
        ) as mock_topic_engine, patch(
            "src.core.AdvancedCore.GraphFeatureProcessor"
        ) as mock_graph_processor, patch(
            "src.core.AdvancedCore.logging.getLogger"
        ) as mock_get_logger, patch(
            "src.core.AdvancedCore.FeatureProcessor.__init__"
        ) as mock_super_init:

            # Create mock objects
            mock_topic = Mock()
            mock_graph = Mock()
            mock_logger = Mock(spec=logging.Logger)

            mock_topic_engine.return_value = mock_topic
            mock_graph_processor.return_value = mock_graph
            mock_get_logger.return_value = mock_logger
            mock_super_init.return_value = None

            # Create processor
            processor = AdvancedFeatureProcessor(config)

            # Set mocked components for easier testing
            processor.topic_engine = mock_topic
            processor.graph_processor = mock_graph
            processor.logger = mock_logger

            return processor

    def test_init(self, processor, config):
        """Test initialization of AdvancedFeatureProcessor."""
        assert processor.topic_engine is not None
        assert processor.graph_processor is not None
        assert processor.logger is not None

    @pytest.mark.asyncio
    async def test_generate_enhanced_features(self, processor):
        """Test generate_enhanced_features method."""
        # Setup
        docs = [
            {"id": "doc1", "content": "This is document 1"},
            {"id": "doc2", "content": "This is document 2"},
        ]

        # Mock component methods
        with patch.object(
            processor, "_generate_embeddings", new_callable=AsyncMock
        ) as mock_gen_emb, patch.object(
            processor.topic_engine, "extract_features", new_callable=AsyncMock
        ) as mock_extract, patch.object(
            processor.graph_processor, "compute_features", new_callable=AsyncMock
        ) as mock_compute, patch.object(
            processor, "_process_metadata", new_callable=AsyncMock
        ) as mock_process:

            # Setup mock return values
            mock_gen_emb.return_value = [[0.1, 0.2], [0.3, 0.4]]
            mock_extract.return_value = {"topics": [1, 2]}
            mock_compute.return_value = {"edges": 10}
            mock_process.return_value = {"count": 2}

            # Call the method
            result = await processor.generate_enhanced_features(docs)

            # Verify
            mock_gen_emb.assert_called_once_with(docs)
            mock_extract.assert_called_once_with(docs)
            mock_compute.assert_called_once_with(docs)
            mock_process.assert_called_once_with(docs)

            assert result.embeddings == [[0.1, 0.2], [0.3, 0.4]]
            assert result.topic_features == {"topics": [1, 2]}
            assert result.graph_features == {"edges": 10}
            assert result.metadata_features == {"count": 2}

    @pytest.mark.asyncio
    async def test_generate_embeddings_with_sentence_transformers(self, processor):
        """Test _generate_embeddings method with SentenceTransformer."""
        # Setup
        docs = [
            {"id": "doc1", "content": "This is document 1"},
            {"id": "doc2", "content": "This is document 2"},
        ]

        model_mock = Mock()
        model_mock.encode.return_value = np.array([[0.1, 0.2], [0.3, 0.4]])

        with patch(
            "src.core.AdvancedCore.logging.getLogger"
        ) as mock_get_logger, patch.dict(
            "sys.modules", {"sentence_transformers": Mock()}
        ), patch(
            "src.core.AdvancedCore.SentenceTransformer", return_value=model_mock
        ):

            # Create mock logger
            mock_logger = Mock(spec=logging.Logger)
            mock_get_logger.return_value = mock_logger
            processor.logger = mock_logger

            # Call the method
            result = await processor._generate_embeddings(docs)

            # Verify
            assert len(result) == 2
            model_mock.encode.assert_called()
            processor.logger.info.assert_called()

    @pytest.mark.asyncio
    async def test_generate_embeddings_fallback(self, processor):
        """Test _generate_embeddings method fallback mechanism."""
        # Setup
        docs = [
            {"id": "doc1", "content": "This is document 1"},
            {"id": "doc2", "title": "Document 2 Title"},
            {"id": "doc3", "other": "No content or title"},
        ]

        with patch(
            "src.core.AdvancedCore.logging.getLogger"
        ) as mock_get_logger, patch.dict(
            "sys.modules", {"sentence_transformers": None}
        ), patch(
            "src.core.AdvancedCore.hashlib.md5"
        ) as mock_md5:

            # Create mock logger
            mock_logger = Mock(spec=logging.Logger)
            mock_get_logger.return_value = mock_logger
            processor.logger = mock_logger

            # Setup mock md5
            mock_hash = Mock()
            mock_hash.hexdigest.return_value = "a0b1c2d3e4f5"
            mock_md5.return_value = mock_hash

            # Call the method
            result = await processor._generate_embeddings(docs)

            # Verify
            assert len(result) == 3
            processor.logger.warning.assert_called()
            assert all(isinstance(emb, list) for emb in result)
            assert len(result[0]) == 768  # Default embedding size

    @pytest.mark.asyncio
    async def test_process_metadata(self, processor):
        """Test _process_metadata method."""
        # Setup
        docs = [
            {
                "id": "doc1",
                "content": "This is document 1",
                "category": "tech",
                "author": "Alice",
                "timestamp": "2023-01-01T12:00:00Z",
                "size": 100,
                "tags": ["python", "data"],
            },
            {
                "id": "doc2",
                "content": "This is document 2",
                "category": "science",
                "author": "Bob",
                "timestamp": "2023-02-01T12:00:00Z",
                "size": 200,
                "tags": ["python", "ml"],
            },
        ]

        # Call the method
        with patch("src.core.AdvancedCore.datetime") as mock_datetime:
            # Setup datetime parser mock
            mock_datetime.strptime.side_effect = lambda s, f: (
                datetime(2023, 1, 1) if "01-01" in s else datetime(2023, 2, 1)
            )
            mock_datetime.now.return_value = datetime(2023, 3, 1)

            result = await processor._process_metadata(docs)

            # Verify
            assert result["document_count"] == 2
            assert "tech" in result["categories"]
            assert "science" in result["categories"]
            assert "Alice" in result["authors"]
            assert "Bob" in result["authors"]
            assert "timestamps" in result
            assert "size_statistics" in result
            assert "tag_frequency" in result
            assert result["tag_frequency"]["python"] == 2
            assert result["size_statistics"]["min_size"] == 100
            assert result["size_statistics"]["max_size"] == 200


class TestTaskRegistryManager:
    """Test suite for the TaskRegistryManager class."""

    @pytest.fixture
    def config(self):
        """Create an EngineConfig instance for testing."""
        return Mock(spec=EngineConfig)

    @pytest.fixture
    def manager(self, config):
        """Create a TaskRegistryManager instance with mocked dependencies for testing."""
        with patch("src.core.AdvancedCore.logging.getLogger") as mock_get_logger:
            # Create mock logger
            mock_logger = Mock(spec=logging.Logger)
            mock_get_logger.return_value = mock_logger

            # Create manager
            manager = TaskRegistryManager(config)
            manager.logger = mock_logger

            return manager

    def test_init(self, manager, config):
        """Test initialization of TaskRegistryManager."""
        assert manager.active_tasks == []
        assert isinstance(manager.pending_queue, Queue)
        assert manager.completion_log == []
        assert manager.task_registry == {}
        assert manager.error_log == []
        assert manager.logger is not None

    def test_register_next_task(self, manager):
        """Test register_next_task method."""
        # Setup
        task = Task(id="test123", type="processing")
        manager.pending_queue.put(task)

        with patch.object(manager, "_generate_task_context") as mock_generate:
            mock_generate.return_value = TaskRegistration(task_id="test123", task=task)

            # Call the method
            registration = manager.register_next_task()

            # Verify
            assert registration.task_id == "test123"
            assert registration.task is task
            assert task in manager.active_tasks
            mock_generate.assert_called_once_with(task)

    def test_register_next_task_empty_queue(self, manager):
        """Test register_next_task method with empty queue."""
        # Call the method
        registration = manager.register_next_task()

        # Verify
        assert registration is None

    def test_generate_task_context(self, manager):
        """Test _generate_task_context method."""
        # Setup
        task = Task(id="test123", type="processing")

        # Call the method
        registration = manager._generate_task_context(task)

        # Verify
        assert registration.task_id == task.id
        assert registration.task is task
        assert task.id in manager.task_registry

    def test_log_task_completion(self, manager):
        """Test _log_task_completion method."""
        # Setup
        task = Task(id="test123", type="processing", status="running")
        manager.active_tasks.append(task)

        # Call the method
        manager._log_task_completion(task)

        # Verify
        assert task not in manager.active_tasks
        assert task in manager.completion_log
        assert task.status == "completed"

    def test_log_task_error(self, manager):
        """Test _log_task_error method."""
        # Setup
        task = Task(id="test123", type="processing", status="running")
        error = Exception("Test error")

        # Call the method
        manager._log_task_error(task, error)

        # Verify
        assert len(manager.error_log) == 1
        assert manager.error_log[0]["task_id"] == task.id
        assert manager.error_log[0]["error"] == error

    def test_get_timestamp(self, manager):
        """Test get_timestamp method."""
        # Call the method
        timestamp = manager.get_timestamp()

        # Verify
        assert isinstance(timestamp, str)
        # Check timestamp format
        try:
            datetime.strptime(timestamp, "%Y-%m-%d %H:%M:%S")
            is_valid_format = True
        except ValueError:
            is_valid_format = False
        assert is_valid_format

    def test_ensure_task_list_attribute(self, manager):
        """Test _ensure_task_list_attribute method."""
        # Setup - attribute doesn't exist
        attr_name = "test_attribute"
        assert not hasattr(manager, attr_name)

        # Call the method
        result = manager._ensure_task_list_attribute(attr_name)

        # Verify
        assert hasattr(manager, attr_name)
        assert getattr(manager, attr_name) == []
        assert result == []

        # Setup - attribute exists
        setattr(
            manager,
            attr_name,
            [Task(id="test1", type="t1"), Task(id="test2", type="t2")],
        )

        # Call the method again
        result = manager._ensure_task_list_attribute(attr_name)

        # Verify
        assert len(result) == 2
        assert all(isinstance(task, Task) for task in result)

    def test_get_active_tasks(self, manager):
        """Test get_active_tasks method."""
        # Setup
        task1 = Task(id="test1", type="t1")
        task2 = Task(id="test2", type="t2")
        manager.active_tasks = [task1, task2]

        # Call the method
        result = manager.get_active_tasks()

        # Verify
        assert len(result) == 2
        assert task1 in result
        assert task2 in result

    def test_get_completion_log(self, manager):
        """Test get_completion_log method."""
        # Setup
        task1 = Task(id="test1", type="t1", status="completed")
        task2 = Task(id="test2", type="t2", status="completed")
        manager.completion_log = [task1, task2]

        # Call the method
        result = manager.get_completion_log()

        # Verify
        assert len(result) == 2
        assert task1 in result
        assert task2 in result

    def test_get_error_log(self, manager):
        """Test get_error_log method."""
        # Setup
        manager.error_log = [
            {"task_id": "test1", "error": Exception("Error 1")},
            {"task_id": "test2", "error": Exception("Error 2")},
        ]

        # Call the method
        result = manager.get_error_log()

        # Verify
        assert len(result) == 2
        assert result[0]["task_id"] == "test1"
        assert result[1]["task_id"] == "test2"

    def test_get_task_registry(self, manager):
        """Test get_task_registry method."""
        # Setup
        manager.task_registry = {
            "task1": {"status": "running"},
            "task2": {"status": "pending"},
        }

        # Call the method
        result = manager.get_task_registry()

        # Verify
        assert len(result) == 2
        assert "task1" in result
        assert "task2" in result
        assert result["task1"]["status"] == "running"
        assert result["task2"]["status"] == "pending"

    def test_get_task_status(self, manager):
        """Test get_task_status method."""
        # Setup
        task_id = "test123"
        manager.task_registry = {task_id: {"status": "running"}}

        # Call the method
        result = manager.get_task_status(task_id)

        # Verify
        assert result == "running"

    def test_get_task_status_nonexistent(self, manager):
        """Test get_task_status method with nonexistent task."""
        # Call the method
        result = manager.get_task_status("nonexistent")

        # Verify
        assert result == "unknown"

    def test_get_task_log(self, manager):
        """Test get_task_log method."""
        # Setup
        task_id = "test123"
        manager.task_registry = {
            task_id: {
                "log": [
                    {"timestamp": "2023-01-01 12:00:00", "message": "Log 1"},
                    {"timestamp": "2023-01-01 12:01:00", "message": "Log 2"},
                ]
            }
        }

        # Call the method
        with patch.object(manager, "_get_task_entries") as mock_get_entries:
            mock_get_entries.return_value = manager.task_registry[task_id]["log"]

            result = manager.get_task_log(task_id)

            # Verify
            assert len(result) == 2
            mock_get_entries.assert_called_once_with(
                task_id, "log", "Retrieving log for task"
            )

    def test_get_task_errors(self, manager):
        """Test get_task_errors method."""
        # Setup
        task_id = "test123"
        manager.task_registry = {
            task_id: {
                "errors": [
                    {"timestamp": "2023-01-01 12:00:00", "message": "Error 1"},
                    {"timestamp": "2023-01-01 12:01:00", "message": "Error 2"},
                ]
            }
        }

        # Call the method
        with patch.object(manager, "_get_task_entries") as mock_get_entries:
            mock_get_entries.return_value = manager.task_registry[task_id]["errors"]

            result = manager.get_task_errors(task_id)

            # Verify
            assert len(result) == 2
            mock_get_entries.assert_called_once_with(
                task_id, "errors", "Retrieving errors for task"
            )

    def test_add_task(self, manager):
        """Test add_task method."""
        # Setup
        task = Task(id="test123", type="processing")

        # Call the method
        manager.add_task(task)

        # Verify
        assert manager.pending_queue.qsize() == 1
        assert manager.pending_queue.get() is task

    def test_update_task_progress(self, manager):
        """Test update_task_progress method."""
        # Setup
        task_id = "test123"

        with patch.object(manager, "update_task_progress_data") as mock_update:
            # Call the method
            manager.update_task_progress(
                task_id=task_id,
                percent_complete=50.0,
                current_step="Processing step 2",
                total_steps=4,
            )

            # Verify
            mock_update.assert_called_once_with(
                task_id=task_id,
                percent_complete=50.0,
                current_step="Processing step 2",
                total_steps=4,
            )

    def test_update_task_progress_data(self, manager):
        """Test update_task_progress_data method."""
        # Setup
        task_id = "test123"
        manager.task_registry = {task_id: {}}

        # Call the method
        with patch.object(manager, "get_timestamp") as mock_timestamp:
            mock_timestamp.return_value = "2023-01-01 12:00:00"

            manager.update_task_progress_data(
                task_id=task_id,
                percent_complete=50.0,
                current_step="Processing step 2",
                total_steps=4,
            )

            # Verify
            assert "progress" in manager.task_registry[task_id]
            progress = manager.task_registry[task_id]["progress"]
            assert progress["percent_complete"] == 50.0
            assert progress["current_step"] == "Processing step 2"
            assert progress["total_steps"] == 4
            assert progress["timestamp"] == "2023-01-01 12:00:00"

    def test_pending_tasks(self, manager):
        """Test pending_tasks method."""
        # Setup
        tasks = [
            Task(id="test1", type="t1"),
            Task(id="test2", type="t2"),
            Task(id="test3", type="t3"),
        ]

        for task in tasks:
            manager.pending_queue.put(task)

        # Mock _temporary_que_list to return task IDs
        with patch.object(manager, "_temporary_que_list") as mock_temp_list:
            mock_temp_list.return_value = ["test1", "test2", "test3"]

            # Call the method
            result = manager.pending_tasks()

            # Verify
            assert len(result) == 3
            assert "test1" in result
            assert "test2" in result
            assert "test3" in result

    def test_complete_task(self, manager):
        """Test complete_task method."""
        # Setup
        task_id = "test123"
        task = Task(id=task_id, type="processing", status="running")
        manager.active_tasks = [task]

        # Mock finalize_task_in_registry and _log_task_completion
        with patch.object(
            manager, "finalize_task_in_registry"
        ) as mock_finalize, patch.object(
            manager, "_log_task_completion"
        ) as mock_log_completion:

            # Call the method
            manager.complete_task(task_id=task_id, result={"success": True})

            # Verify
            mock_finalize.assert_called_once_with(
                task_id=task_id, result={"success": True}
            )
            mock_log_completion.assert_called_once()
            assert mock_log_completion.call_args[0][0] is task

    def test_finalize_task_in_registry(self, manager):
        """Test finalize_task_in_registry method."""
        # Setup
        task_id = "test123"
        manager.task_registry = {task_id: {"status": "running"}}
        result = {"success": True}

        # Call the method
        with patch.object(manager, "get_timestamp") as mock_timestamp:
            mock_timestamp.return_value = "2023-01-01 12:00:00"

            manager.finalize_task_in_registry(task_id=task_id, result=result)

            # Verify
            registry_entry = manager.task_registry[task_id]
            assert registry_entry["status"] == "completed"
            assert registry_entry["result"] == result
            assert registry_entry["completed_at"] == "2023-01-01 12:00:00"
            assert "progress" in registry_entry
            assert registry_entry["progress"]["percent_complete"] == 100.0
