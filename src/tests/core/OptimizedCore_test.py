from pathlib import Path
from unittest.mock import AsyncMock, Mock, patch

import networkx as nx
import numpy as np
import pytest

from src.config.SystemConfig import SystemConfig
from src.core.OptimizedCore import (
    EnhancedProcessor,
    NLTKProcessor,
    OptimizedStructure,
    ProcessedContent,
    ProcessingResult,
    ReorganizationError,
    ReorganizationResult,
    StorageOrganizationError,
    StorageOrganizer,
    StructureOptimizationError,
    StructureOptimizer,
    SystemManager,
)
from src.FeatureCore import AnalysisResult as FCAnalysisResult


class TestOptimizedStructure:
    """Test suite for the OptimizedStructure class."""

    @pytest.fixture
    def config(self):
        """Create a mock SystemConfig instance."""
        return Mock(spec=SystemConfig)

    @pytest.fixture
    def optimized_structure(self, config):
        """Create an OptimizedStructure instance for testing."""
        return OptimizedStructure(config)

    def test_init(self, optimized_structure, config):
        """Test initialization of OptimizedStructure."""
        assert optimized_structure.config == config
        assert optimized_structure.graph is not None
        assert isinstance(optimized_structure.graph, nx.DiGraph)
        assert optimized_structure.summary == {}
        assert optimized_structure.statistics == {}
        assert optimized_structure.redundancies == []
        assert optimized_structure.structure == {}

    def test_add_node(self, optimized_structure):
        """Test adding a node to the graph."""
        # Setup
        node_id = "test_node"
        attributes = {"key": "value"}

        # Call the method
        optimized_structure.add_node(node_id, attributes)

        # Verify
        assert optimized_structure.graph.has_node(node_id)
        for key, value in attributes.items():
            assert optimized_structure.graph.nodes[node_id][key] == value

    def test_add_edge(self, optimized_structure):
        """Test adding an edge to the graph."""
        # Setup
        source = "source_node"
        target = "target_node"
        weight = 0.5

        # Add nodes first
        optimized_structure.add_node(source, {})
        optimized_structure.add_node(target, {})

        # Call the method
        optimized_structure.add_edge(source, target, weight)

        # Verify
        assert optimized_structure.graph.has_edge(source, target)
        assert optimized_structure.graph.edges[source, target]["weight"] == weight

    def test_get_node(self, optimized_structure):
        """Test getting a node from the graph."""
        # Setup
        node_id = "test_node"
        attributes = {"key": "value"}
        optimized_structure.add_node(node_id, attributes)

        # Call the method
        node = optimized_structure.get_node(node_id)

        # Verify
        assert node == attributes

    def test_get_nonexistent_node(self, optimized_structure):
        """Test getting a nonexistent node from the graph."""
        # Call the method with a nonexistent node
        node = optimized_structure.get_node("nonexistent")

        # Verify
        assert node is None

    def test_update_summary(self, optimized_structure):
        """Test updating the summary."""
        # Setup
        summary = {"key": "value"}

        # Call the method
        optimized_structure.update_summary(summary)

        # Verify
        assert optimized_structure.summary == summary

    def test_update_statistics(self, optimized_structure):
        """Test updating the statistics."""
        # Setup
        statistics = {"key": "value"}

        # Call the method
        optimized_structure.update_statistics(statistics)

        # Verify
        assert optimized_structure.statistics == statistics

    def test_add_redundancy(self, optimized_structure):
        """Test adding a redundancy."""
        # Setup
        redundancy = {"source": "source_node", "target": "target_node"}

        # Call the method
        optimized_structure.add_redundancy(redundancy)

        # Verify
        assert redundancy in optimized_structure.redundancies

    def test_set_structure(self, optimized_structure):
        """Test setting the structure."""
        # Setup
        structure = {"key": "value"}

        # Call the method
        optimized_structure.set_structure(structure)

        # Verify
        assert optimized_structure.structure == structure

    def test_get_structure(self, optimized_structure):
        """Test getting the structure."""
        # Setup
        structure = {"key": "value"}
        optimized_structure.set_structure(structure)

        # Call the method
        result = optimized_structure.get_structure()

        # Verify
        assert result == structure

    def test_to_dict(self, optimized_structure):
        """Test converting to a dictionary."""
        # Setup
        optimized_structure.summary = {"summary_key": "summary_value"}
        optimized_structure.statistics = {"stats_key": "stats_value"}
        optimized_structure.redundancies = [{"redundancy": "example"}]
        optimized_structure.structure = {"structure_key": "structure_value"}

        # Call the method
        result = optimized_structure.to_dict()

        # Verify
        assert result["summary"] == optimized_structure.summary
        assert result["statistics"] == optimized_structure.statistics
        assert result["redundancies"] == optimized_structure.redundancies
        assert result["structure"] == optimized_structure.structure


class TestStructureOptimizationError:
    """Test suite for the StructureOptimizationError class."""

    def test_init(self):
        """Test initialization of StructureOptimizationError."""
        # Setup
        message = "Test error message"

        # Create the error
        error = StructureOptimizationError(message)

        # Verify
        assert error.message == message
        assert error.type == "structure_optimization_error"
        assert error.error == message
        assert error.success is False
        assert error.timestamp is not None


class TestStructureOptimizer:
    """Test suite for the StructureOptimizer class."""

    @pytest.fixture
    def config(self):
        """Create a mock SystemConfig instance."""
        return Mock(spec=SystemConfig)

    @pytest.fixture
    def structure_optimizer(self, config):
        """Create a StructureOptimizer instance for testing."""
        with patch("src.core.OptimizedCore.pipeline") as mock_pipeline:
            mock_pipeline.return_value = Mock()
            with patch("src.core.OptimizedCore.status.Status") as mock_status:
                mock_status_instance = Mock()
                mock_status.return_value = mock_status_instance
                optimizer = StructureOptimizer(config)
                # Make the progress object accessible for verification
                optimizer.progress = mock_status_instance
                return optimizer

    @pytest.mark.asyncio
    async def test_optimize_structure(self, structure_optimizer):
        """Test optimize_structure method."""
        # Setup
        analysis_result = Mock(spec=FCAnalysisResult)
        docs = [{"id": "doc1", "content": "test content"}]

        # Mock the internal methods
        structure_optimizer._build_graph = AsyncMock(return_value=nx.DiGraph())
        structure_optimizer._generate_summaries = AsyncMock(
            return_value={"doc1": "summary"}
        )
        structure_optimizer._detect_redundancies = AsyncMock(return_value=[])

        # Call the method
        result = await structure_optimizer.optimize_structure(analysis_result, docs)

        # Verify
        assert isinstance(result, OptimizedStructure)
        structure_optimizer._build_graph.assert_called_once_with(docs, analysis_result)
        structure_optimizer._generate_summaries.assert_called_once_with(docs)
        structure_optimizer._detect_redundancies.assert_called_once_with(
            docs, analysis_result
        )

    @pytest.mark.asyncio
    async def test_build_graph(self, structure_optimizer):
        """Test _build_graph method."""
        # Setup
        docs = [{"id": "doc1", "content": "test content"}]
        analysis_result = Mock(spec=FCAnalysisResult)
        analysis_result.entities = {"doc1": ["entity1", "entity2"]}
        analysis_result.concepts = {"doc1": ["concept1"]}

        # Call the method
        graph = await structure_optimizer._build_graph(docs, analysis_result)

        # Verify
        assert isinstance(graph, nx.DiGraph)
        assert graph.has_node("doc1")
        assert graph.has_node("entity1")
        assert graph.has_node("entity2")
        assert graph.has_node("concept1")
        assert graph.has_edge("doc1", "entity1")
        assert graph.has_edge("doc1", "entity2")
        assert graph.has_edge("doc1", "concept1")

    @pytest.mark.asyncio
    async def test_generate_summaries(self, structure_optimizer):
        """Test _generate_summaries method."""
        # Setup
        docs = [
            {"id": "doc1", "content": "test content 1"},
            {"id": "doc2", "content": "test content 2"},
        ]

        # Mock summarizer
        structure_optimizer.summarizer = Mock()
        structure_optimizer.summarizer.return_value = [{"summary_text": "summary"}]

        # Call the method
        summaries = await structure_optimizer._generate_summaries(docs)

        # Verify
        assert isinstance(summaries, dict)
        assert len(summaries) == 2
        assert "doc1" in summaries
        assert "doc2" in summaries
        assert structure_optimizer.summarizer.call_count == 2

    @pytest.mark.asyncio
    async def test_detect_redundancies(self, structure_optimizer):
        """Test _detect_redundancies method."""
        # Setup
        docs = [
            {"id": "doc1", "content": "test content 1"},
            {"id": "doc2", "content": "similar content"},
            {"id": "doc3", "content": "very similar content"},
        ]
        analysis_result = Mock(spec=FCAnalysisResult)
        analysis_result.similarity_matrix = np.array(
            [[1.0, 0.8, 0.7], [0.8, 1.0, 0.9], [0.7, 0.9, 1.0]]
        )

        # Call the method
        redundancies = await structure_optimizer._detect_redundancies(
            docs, analysis_result
        )

        # Verify
        assert isinstance(redundancies, list)
        assert len(redundancies) > 0
        for redundancy in redundancies:
            assert "source" in redundancy
            assert "target" in redundancy
            assert "similarity" in redundancy


class TestReorganizationResult:
    """Test suite for the ReorganizationResult class."""

    def test_init(self):
        """Test initialization of ReorganizationResult."""
        # Setup
        status = "success"
        details = "Operation completed successfully"
        timestamp = "2023-01-01T12:00:00"

        # Create the result
        result = ReorganizationResult(status, details, timestamp)

        # Verify
        assert result.status == status
        assert result.details == details
        assert result.timestamp == timestamp

    def test_str_representation(self):
        """Test string representation of ReorganizationResult."""
        # Setup
        status = "success"
        details = "Operation completed successfully"
        timestamp = "2023-01-01T12:00:00"
        result = ReorganizationResult(status, details, timestamp)

        # Call __str__
        string_repr = str(result)

        # Verify
        assert status in string_repr
        assert details in string_repr
        assert timestamp in string_repr


class TestStorageOrganizationError:
    """Test suite for the StorageOrganizationError class."""

    def test_init(self):
        """Test initialization of StorageOrganizationError."""
        # Setup
        message = "Test error message"

        # Create the error
        error = StorageOrganizationError(message)

        # Verify
        assert error.message == message
        assert error.type == "storage_organization_error"
        assert error.error == message
        assert error.success is False
        assert error.timestamp is not None


class TestReorganizationError:
    """Test suite for the ReorganizationError class."""

    def test_init(self):
        """Test initialization of ReorganizationError."""
        # Setup
        message = "Test error message"

        # Create the error
        error = ReorganizationError(message)

        # Verify
        assert error.message == message
        assert error.type == "reorganization_error"
        assert error.error == message
        assert error.success is False
        assert error.timestamp is not None


class TestStorageOrganizer:
    """Test suite for the StorageOrganizer class."""

    @pytest.fixture
    def config(self):
        """Create a mock SystemConfig instance."""
        return Mock(spec=SystemConfig)

    @pytest.fixture
    def storage_organizer(self, config):
        """Create a StorageOrganizer instance for testing."""
        return StorageOrganizer(config)

    @pytest.fixture
    def optimized_structure(self, config):
        """Create an OptimizedStructure instance for testing."""
        structure = OptimizedStructure(config)
        structure.set_structure(
            {
                "folder1": {"file1.md": "content1", "file2.md": "content2"},
                "folder2": {"file3.md": "content3"},
            }
        )
        return structure

    @pytest.mark.asyncio
    async def test_reorganize_vault(self, storage_organizer, optimized_structure):
        """Test reorganize_vault method."""
        # Setup
        vault_path = Path("/test/vault")

        # Mock internal methods
        storage_organizer._create_backup = AsyncMock()
        storage_organizer._generate_structure = AsyncMock(
            return_value={"new_structure": True}
        )
        storage_organizer._plan_moves = AsyncMock(return_value=[("source", "dest")])
        storage_organizer._execute_moves = AsyncMock(return_value=True)

        # Call the method
        result = await storage_organizer.reorganize_vault(
            optimized_structure, vault_path
        )

        # Verify
        assert isinstance(result, ReorganizationResult)
        assert result.status == "success"
        storage_organizer._create_backup.assert_called_once_with(vault_path)
        storage_organizer._generate_structure.assert_called_once_with(
            optimized_structure
        )
        storage_organizer._plan_moves.assert_called_once()
        storage_organizer._execute_moves.assert_called_once()

    @pytest.mark.asyncio
    async def test_create_backup(self, storage_organizer):
        """Test _create_backup method."""
        # Setup
        vault_path = Path("/test/vault")

        # Mock file operations
        with patch("pathlib.Path.mkdir") as mock_mkdir, patch(
            "src.core.OptimizedCore.shutil.copytree"
        ) as mock_copytree:

            # Call the method
            await storage_organizer._create_backup(vault_path)

            # Verify
            mock_mkdir.assert_called_once()
            mock_copytree.assert_called_once()

    @pytest.mark.asyncio
    async def test_rollback(self, storage_organizer):
        """Test _rollback method."""
        # Setup
        backup_path = Path("/test/backup")

        # Mock file operations
        with patch("src.core.OptimizedCore.shutil.rmtree") as mock_rmtree, patch(
            "src.core.OptimizedCore.shutil.copytree"
        ) as mock_copytree:

            # Call the method
            await storage_organizer._rollback(backup_path)

            # Verify
            mock_rmtree.assert_called_once()
            mock_copytree.assert_called_once()

    def test_plan_moves(self, storage_organizer):
        """Test _plan_moves method."""
        # Setup
        new_structure = {
            "folder1": {"file1.md": "content1", "file2.md": "content2"},
            "folder2": {"file3.md": "content3"},
        }
        vault_path = Path("/test/vault")

        # Mock file operations
        with patch("pathlib.Path.exists") as mock_exists, patch(
            "pathlib.Path.is_file"
        ) as mock_is_file, patch("pathlib.Path.glob") as mock_glob:

            mock_exists.return_value = True
            mock_is_file.return_value = True
            mock_glob.return_value = [
                Path("/test/vault/old_folder/file1.md"),
                Path("/test/vault/old_folder/file2.md"),
                Path("/test/vault/old_folder/file3.md"),
            ]

            # Call the method
            moves = storage_organizer._plan_moves(new_structure, vault_path)

            # Verify
            assert isinstance(moves, list)
            assert len(moves) > 0
            for move in moves:
                assert isinstance(move, tuple)
                assert len(move) == 2


class TestProcessingResult:
    """Test suite for the ProcessingResult class."""

    def test_init_success(self):
        """Test initialization of ProcessingResult with success=True."""
        # Setup
        success = True
        metadata = {"key": "value"}
        stats = {"count": 5}

        # Create the result
        result = ProcessingResult(success, metadata, stats)

        # Verify
        assert result.success is True
        assert result.metadata == metadata
        assert result.stats == stats
        assert result.error is None
        assert result.timestamp is not None

    def test_init_failure(self):
        """Test initialization of ProcessingResult with success=False and error."""
        # Setup
        success = False
        error = "Test error message"

        # Create the result
        result = ProcessingResult(success, error=error)

        # Verify
        assert result.success is False
        assert result.metadata == {}
        assert result.stats == {}
        assert result.error == error
        assert result.timestamp is not None

    def test_init_default_values(self):
        """Test initialization of ProcessingResult with default values."""
        # Create the result with minimal arguments
        result = ProcessingResult(True)

        # Verify default values
        assert result.success is True
        assert result.metadata == {}
        assert result.stats == {}
        assert result.error is None
        assert result.timestamp is not None
        assert result.id is None
        assert result.type is None
        assert result.version is None
        assert result.source is None
        assert result.target is None


class TestSystemManager:
    """Test suite for the SystemManager class."""

    @pytest.fixture
    def config(self):
        """Create a mock SystemConfig instance."""
        return Mock(spec=SystemConfig)

    @pytest.fixture
    def system_manager(self, config):
        """Create a SystemManager instance for testing."""
        with patch(
            "src.core.OptimizedCore.MarkdownProcessor"
        ) as mock_md_processor, patch(
            "src.core.OptimizedCore.FeatureProcessor"
        ) as mock_feature_processor, patch(
            "src.core.OptimizedCore.AnalyticsEngine"
        ) as mock_analytics_engine, patch(
            "src.core.OptimizedCore.StructureOptimizer"
        ) as mock_structure_optimizer, patch(
            "src.core.OptimizedCore.StorageOrganizer"
        ) as mock_storage_organizer, patch(
            "src.core.OptimizedCore.status.Status"
        ) as mock_status:

            mock_status_instance = Mock()
            mock_status.return_value = mock_status_instance

            manager = SystemManager(config)
            # Make the components accessible for verification
            manager.di = mock_md_processor.return_value
            manager.fe = mock_feature_processor.return_value
            manager.cc = mock_analytics_engine.return_value
            manager.st = mock_structure_optimizer.return_value
            manager.so = mock_storage_organizer.return_value
            manager.progress = mock_status_instance

            return manager

    @pytest.mark.asyncio
    async def test_process_vault(self, system_manager):
        """Test process_vault method."""
        # Setup
        vault_path = Path("/test/vault")

        # Mock component methods
        system_manager.di.process_documents = AsyncMock(return_value=[{"id": "doc1"}])
        system_manager.fe.generate_features = AsyncMock(
            return_value=[{"feature": "value"}]
        )
        system_manager.cc.analyze_features = AsyncMock(
            return_value=Mock(spec=FCAnalysisResult)
        )
        system_manager.st.optimize_structure = AsyncMock(
            return_value=Mock(spec=OptimizedStructure)
        )
        system_manager.so.reorganize_vault = AsyncMock(
            return_value=Mock(spec=ReorganizationResult)
        )

        # Mock stats and metadata generation
        system_manager._generate_stats = Mock(return_value={"stats": "value"})
        system_manager._generate_metadata = Mock(return_value={"metadata": "value"})

        # Call the method
        result = await system_manager.process_vault(vault_path)

        # Verify
        assert isinstance(result, ProcessingResult)
        assert result.success is True
        system_manager.di.process_documents.assert_called_once_with(vault_path)
        system_manager.fe.generate_features.assert_called_once()
        system_manager.cc.analyze_features.assert_called_once()
        system_manager.st.optimize_structure.assert_called_once()
        system_manager.so.reorganize_vault.assert_called_once()
        system_manager._generate_stats.assert_called_once()
        system_manager._generate_metadata.assert_called_once()

    def test_generate_stats(self, system_manager):
        """Test _generate_stats method."""
        # Setup
        result = {
            "documents": [{"id": "doc1"}, {"id": "doc2"}],
            "entities": ["entity1", "entity2", "entity3"],
            "concepts": ["concept1", "concept2"],
        }

        # Call the method
        stats = system_manager._generate_stats(result)

        # Verify
        assert isinstance(stats, dict)
        assert "document_count" in stats
        assert stats["document_count"] == 2
        assert "entity_count" in stats
        assert stats["entity_count"] == 3
        assert "concept_count" in stats
        assert stats["concept_count"] == 2

    def test_generate_metadata(self, system_manager):
        """Test _generate_metadata method."""
        # Setup
        result = {
            "timestamp": "2023-01-01T12:00:00",
            "version": "1.0.0",
            "source": "test_source",
        }

        # Call the method
        metadata = system_manager._generate_metadata(result)

        # Verify
        assert isinstance(metadata, dict)
        assert "timestamp" in metadata
        assert metadata["timestamp"] == result["timestamp"]
        assert "version" in metadata
        assert metadata["version"] == result["version"]
        assert "source" in metadata
        assert metadata["source"] == result["source"]


class TestEnhancedProcessor:
    """Test suite for the EnhancedProcessor class."""

    @pytest.fixture
    def config(self):
        """Create a mock SystemConfig instance."""
        return Mock(spec=SystemConfig)

    @pytest.fixture
    def enhanced_processor(self, config):
        """Create an EnhancedProcessor instance for testing."""
        with patch("src.core.OptimizedCore.NLPCore") as mock_nlp_core, patch(
            "src.core.OptimizedCore.MLEngine"
        ) as mock_ml_engine, patch(
            "src.core.OptimizedCore.GraphAnalyzer"
        ) as mock_graph_analyzer, patch(
            "src.core.OptimizedCore.SummaryGenerator"
        ) as mock_summary_generator, patch(
            "src.core.OptimizedCore.DataHandler"
        ) as mock_data_handler, patch(
            "src.core.OptimizedCore.status.Status"
        ) as mock_status:

            mock_status_instance = Mock()
            mock_status.return_value = mock_status_instance

            processor = EnhancedProcessor(config)
            # Make the components accessible for verification
            processor.nlp_core = mock_nlp_core.return_value
            processor.ml_engine = mock_ml_engine.return_value
            processor.graph_analyzer = mock_graph_analyzer.return_value
            processor.summary_generator = mock_summary_generator.return_value
            processor.data_handler = mock_data_handler.return_value
            processor.progress = mock_status_instance

            return processor

    def test_init(self, enhanced_processor, config):
        """Test initialization of EnhancedProcessor."""
        assert enhanced_processor.config == config
        assert enhanced_processor.nlp_core is not None
        assert enhanced_processor.ml_engine is not None
        assert enhanced_processor.graph_analyzer is not None
        assert enhanced_processor.summary_generator is not None
        assert enhanced_processor.data_handler is not None
        assert enhanced_processor.progress is not None
        assert enhanced_processor.lock is not None


class TestNLTKProcessor:
    """Test suite for the NLTKProcessor class."""

    @pytest.fixture
    def nltk_processor(self):
        """Create an NLTKProcessor instance for testing."""
        with patch("nltk.tokenize.WhitespaceTokenizer") as mock_tokenizer, patch(
            "nltk.stem.PorterStemmer"
        ) as mock_stemmer, patch(
            "nltk.stem.WordNetLemmatizer"
        ) as mock_lemmatizer, patch(
            "nltk.corpus.stopwords.words"
        ) as mock_stopwords:

            mock_tokenizer.return_value = Mock()
            mock_stemmer.return_value = Mock()
            mock_lemmatizer.return_value = Mock()
            mock_stopwords.return_value = ["the", "a", "an"]

            processor = NLTKProcessor()
            # Make the components accessible for verification
            processor.tokenizer = mock_tokenizer.return_value
            processor.stemmer = mock_stemmer.return_value
            processor.lemmatizer = mock_lemmatizer.return_value
            processor.stop_words = set(mock_stopwords.return_value)
            processor.stop_words.update(
                [".", ",", "?", "!", ":", ";", "(", ")", "[", "]"]
            )

            return processor

    def test_init(self, nltk_processor):
        """Test initialization of NLTKProcessor."""
        assert nltk_processor.tokenizer is not None
        assert nltk_processor.stemmer is not None
        assert nltk_processor.lemmatizer is not None
        assert isinstance(nltk_processor.stop_words, set)
        assert "the" in nltk_processor.stop_words
        assert "." in nltk_processor.stop_words


class TestProcessedContent:
    """Test suite for the ProcessedContent class."""

    def test_init(self):
        """Test initialization of ProcessedContent."""
        # Setup
        tokens = ["token1", "token2"]
        entities = ["entity1", "entity2"]
        dependencies = ["dep1", "dep2"]
        embeddings = ["emb1", "emb2"]

        # Create the content
        content = ProcessedContent(tokens, entities, dependencies, embeddings)

        # Verify
        assert content.tokens == tokens
        assert content.entities == entities
        assert content.dependencies == dependencies
        assert content.embeddings == embeddings
