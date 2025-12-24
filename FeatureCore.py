import asyncio
import logging
import re
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List

import aiofiles
import numpy as np
import spacy
import torch
from hdbscan import HDBSCAN
from sklearn.decomposition import LatentDirichletAllocation
from transformers import AutoModel, AutoTokenizer


class VaultProcessingError(Exception):
    """Exception raised when there's an error processing the vault."""

    pass


class FeatureGenerationError(Exception):
    """Exception raised when there's an error in feature generation."""

    pass


class AnalysisError(Exception):
    """Exception raised when there's an error during analysis."""

    pass


@dataclass
class AnalysisResult:
    """Class for storing the results of feature analysis."""

    clusters: Any
    topics: Any
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SystemConfig:
    """
    A data class representing the system configuration.

    Args:
        max_threads (int): The maximum number of threads to use. Defaults to 8.
        batch_size (int): The size of each batch. Defaults to 1000.
        buffer_size (int): The size of the buffer. Defaults to 2MB (2048 * 1024).
        processing_mode (str): The processing mode to use. Defaults to "CUDA_ENABLED".
        error_tolerance (float): The error tolerance value. Defaults to 0.85.
    """

    max_threads: int = 8
    batch_size: int = 1000
    buffer_size: int = 2048 * 1024  # 2MB
    processing_mode: str = "CUDA_ENABLED"
    error_tolerance: float = 0.85


class MarkdownProcessor:
    """
    A class for processing Markdown files.

    Args:
        config (SystemConfig): The system configuration.

    Attributes:
        config (SystemConfig): The system configuration.
        executor (ThreadPoolExecutor): The executor for running tasks in parallel.
        logger (logging.Logger): The logger for logging messages.

    Methods:
        process_vault: Process a vault of Markdown files and return the aggregated results.

        _process_file: Process a single Markdown file and return the processed data.

    Returns:
        Dict[str, Any]: The aggregated results of processing the vault.

    Raises:
        VaultProcessingError: If there is an error processing the vault.

    Raises:
        VaultProcessingError: If there is an error processing the vault.
    """

    def __init__(self, config: SystemConfig):
        """
        Initialize the MarkdownProcessor.

        Args:
            self: The instance of the MarkdownProcessor.
            config (SystemConfig): The system configuration.

        Attributes:
            config (SystemConfig): The system configuration.
            executor (ThreadPoolExecutor): The executor for running tasks in parallel.
            logger (logging.Logger): The logger for logging messages.
        """

        self.config = config
        self.executor = ThreadPoolExecutor(max_workers=config.max_threads)
        self.logger = logging.getLogger("MarkdownProcessor")

    async def process_vault(self, vault_path: Path) -> Dict[str, Any]:
        """
        Process a vault of Markdown files and return the aggregated results.

        Args:
            self: The instance of the MarkdownProcessor.
            vault_path (Path): The path to the vault directory.

        Returns:
            Dict[str, Any]: The aggregated results of processing the vault.

        Raises:
            VaultProcessingError: If there is an error processing the vault.
        """

        try:
            files = list(vault_path.rglob("*.md"))
            batches = [
                files[i : i + self.config.batch_size]
                for i in range(0, len(files), self.config.batch_size)
            ]

            results = []
            for batch in batches:
                batch_results = await asyncio.gather(
                    *[self._process_file(file) for file in batch]
                )
                results.extend(batch_results)

            return self._aggregate_results(results)

        except Exception as e:
            self.logger.error(f"Vault processing error: {str(e)}")
            raise VaultProcessingError(f"Failed to process vault: {str(e)}") from e

    async def _process_file(self, file_path: Path) -> Dict[str, Any]:
        """
        Process a single Markdown file and return the processed data.

        Args:
            self: The instance of the MarkdownProcessor.
            file_path (Path): The path to the Markdown file.

        Returns:
            Dict[str, Any]: The processed data of the file, including path, metadata, content, and stats.
        """

        try:
            content = await self._read_file(file_path)
            metadata = self._extract_metadata(content)
            normalized = self._normalize_content(content)
            return {
                "path": str(file_path),
                "metadata": metadata,
                "content": normalized,
                "stats": self._compute_stats(normalized),
            }
        except Exception as e:
            self.logger.warning(f"File processing error: {str(e)}")
            return {"path": str(file_path), "error": str(e)}

    async def _read_file(self, file_path: Path) -> str:
        """Read file content asynchronously."""
        async with aiofiles.open(file_path, "r", encoding="utf-8") as f:
            return await f.read()

    def _extract_metadata(self, content: str) -> Dict[str, Any]:
        """Extract YAML frontmatter metadata from markdown content."""
        metadata: Dict[str, Any] = {}
        if content.startswith("---"):
            parts = content.split("---", 2)
            if len(parts) >= 3:
                frontmatter = parts[1].strip()
                for line in frontmatter.split("\n"):
                    if ":" in line:
                        key, value = line.split(":", 1)
                        metadata[key.strip()] = value.strip()
        return metadata

    def _normalize_content(self, content: str) -> str:
        """Normalize markdown content by removing frontmatter and extra whitespace."""
        # Remove YAML frontmatter
        if content.startswith("---"):
            parts = content.split("---", 2)
            if len(parts) >= 3:
                content = parts[2]
        # Normalize whitespace
        content = re.sub(r"\n{3,}", "\n\n", content)
        return content.strip()

    def _compute_stats(self, content: str) -> Dict[str, Any]:
        """Compute basic statistics for the content."""
        words = content.split()
        return {
            "word_count": len(words),
            "char_count": len(content),
            "line_count": content.count("\n") + 1,
        }

    def _aggregate_results(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Aggregate processing results from multiple files."""
        successful = [r for r in results if "error" not in r]
        failed = [r for r in results if "error" in r]
        return {
            "processed": successful,
            "errors": failed,
            "total": len(results),
            "success_count": len(successful),
            "error_count": len(failed),
        }


class FeatureProcessor:
    """
    A class for generating features from processed documents.

    Args:
        config (SystemConfig): The system configuration.

    Attributes:
        config (SystemConfig): The system configuration.
        model: The initialized BERT model.
        nlp: The loaded spaCy model for natural language processing.

    Methods:
        generate_features: Generate features from a list of processed documents.

        _generate_embedding: Generate BERT embeddings for a given content.

    Returns:
        np.ndarray: The generated features as a NumPy array.

    Raises:
        FeatureGenerationError: If there is an error generating the features.
    """

    def __init__(self, config: SystemConfig):
        """
        Initialize the FeatureProcessor.

        Args:
            self: The instance of the FeatureProcessor.
            config (SystemConfig): The system configuration.

        Attributes:
            config (SystemConfig): The system configuration.
            model: The initialized BERT model.
            tokenizer: The BERT tokenizer.
            nlp: The loaded spaCy model for natural language processing.
        """

        self.logger = logging.getLogger(__name__)
        self.config = config
        self.model, self.tokenizer = self._initialize_bert()
        self.nlp = spacy.load("en_core_web_trf")

    def _initialize_bert(self):
        """Initialize the BERT model and tokenizer."""
        model_name = "bert-base-uncased"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name)
        model.eval()
        return model, tokenizer

    async def _extract_nlp_features(self, content: str) -> np.ndarray:
        """Extract NLP features using spaCy."""
        doc = self.nlp(content[:10000])  # Limit content length for processing
        features = []
        # Entity counts by type
        entity_counts = {}
        for ent in doc.ents:
            entity_counts[ent.label_] = entity_counts.get(ent.label_, 0) + 1
        features.extend([entity_counts.get(label, 0) for label in ["PERSON", "ORG", "GPE", "DATE"]])
        # POS tag ratios
        pos_counts = {}
        for token in doc:
            pos_counts[token.pos_] = pos_counts.get(token.pos_, 0) + 1
        total_tokens = len(doc) or 1
        features.extend([pos_counts.get(pos, 0) / total_tokens for pos in ["NOUN", "VERB", "ADJ", "ADV"]])
        return np.array(features, dtype=np.float32)

    async def generate_features(
        self, processed_docs: List[Dict[str, Any]]
    ) -> np.ndarray:
        """
        Generate features from a list of processed documents.

        Args:
            self: The instance of the FeatureProcessor.
            processed_docs (List[Dict[str, Any]]): The processed documents.

        Returns:
            np.ndarray: The generated features as a NumPy array.

        Raises:
            FeatureGenerationError: If there is an error generating the features.
        """

        try:
            embeddings = []
            for doc in processed_docs:
                if "error" not in doc:
                    # Generate BERT embeddings
                    doc_embedding = await self._generate_embedding(doc["content"])
                    # Extract additional features
                    nlp_features = await self._extract_nlp_features(doc["content"])
                    # Combine features
                    combined = np.concatenate([doc_embedding, nlp_features])
                    embeddings.append(combined)

            return np.vstack(embeddings)

        except Exception as e:
            self.logger.error(f"Feature generation error: {str(e)}")
            raise FeatureGenerationError(str(e)) from e

    async def _generate_embedding(self, content: str) -> np.ndarray:
        """
        Generate BERT embeddings for a given content.

        Args:
            self: The instance of the FeatureProcessor.
            content (str): The content to generate embeddings for.

        Returns:
            np.ndarray: The generated embeddings as a NumPy array.
        """

        tokens = self.tokenizer(
            content, truncation=True, padding=True, return_tensors="pt"
        )
        with torch.no_grad():
            outputs = self.model(**tokens)
        return outputs.last_hidden_state.mean(dim=1).numpy()


class AnalyticsEngine:
    """
    A class for performing analytics on feature matrices.

    Args:
        config (SystemConfig): The system configuration.

    Attributes:
        config (SystemConfig): The system configuration.
        hdbscan: The initialized HDBSCAN model.
        lda: The initialized Latent Dirichlet Allocation model.

    Methods:
        analyze_features: Analyze a feature matrix and return the analysis results.
    """

    def __init__(self, config: SystemConfig):
        """
        Initialize the AnalyticsEngine.

        Args:
            self: The instance of the AnalyticsEngine.
            config (SystemConfig): The system configuration.

        Attributes:
            config (SystemConfig): The system configuration.
            logger: The logger for logging messages.
            hdbscan: The initialized HDBSCAN model.
            lda: The initialized Latent Dirichlet Allocation model.
        """

        self.config = config
        self.logger = logging.getLogger(__name__)
        self.hdbscan = HDBSCAN(
            min_cluster_size=5, min_samples=3, cluster_selection_epsilon=0.3
        )
        self.lda = LatentDirichletAllocation(
            n_components=20, random_state=42, n_jobs=config.max_threads
        )

    async def _generate_clusters(self, feature_matrix: np.ndarray) -> np.ndarray:
        """Generate clusters from the feature matrix using HDBSCAN."""
        return self.hdbscan.fit_predict(feature_matrix)

    async def _extract_topics(self, feature_matrix: np.ndarray) -> np.ndarray:
        """Extract topics from the feature matrix using LDA."""
        # Ensure non-negative values for LDA
        non_negative = np.abs(feature_matrix)
        return self.lda.fit_transform(non_negative)

    def _generate_metadata(self, clusters: np.ndarray, topics: np.ndarray) -> Dict[str, Any]:
        """Generate metadata from clustering and topic analysis results."""
        unique_clusters = np.unique(clusters)
        return {
            "num_clusters": len(unique_clusters[unique_clusters >= 0]),  # Exclude noise (-1)
            "num_topics": topics.shape[1] if len(topics.shape) > 1 else 0,
            "noise_points": int(np.sum(clusters == -1)),
            "cluster_sizes": {int(c): int(np.sum(clusters == c)) for c in unique_clusters if c >= 0},
        }

    async def analyze_features(self, feature_matrix: np.ndarray) -> AnalysisResult:
        """
        Analyze a feature matrix and return the analysis results.

        Args:
            self: The instance of the AnalyticsEngine.
            feature_matrix (np.ndarray): The feature matrix to analyze.

        Returns:
            AnalysisResult: The analysis results.
        """

        try:
            # Parallel processing of different analysis tasks
            cluster_task = asyncio.create_task(self._generate_clusters(feature_matrix))
            topic_task = asyncio.create_task(self._extract_topics(feature_matrix))

            # Wait for all analysis tasks to complete
            clusters, topics = await asyncio.gather(cluster_task, topic_task)

            return AnalysisResult(
                clusters=clusters,
                topics=topics,
                metadata=self._generate_metadata(clusters, topics),
            )

        except Exception as e:
            self.logger.error(f"Analysis error: {str(e)}")
            raise AnalysisError(str(e)) from e
