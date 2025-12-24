import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Dict, List

import spacy
import torch
from hdbscan import HDBSCAN
from numpy import np
from sklearn.decomposition import LatentDirichletAllocation


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
            nlp: The loaded spaCy model for natural language processing.
        """

        self.logger = logging.getLogger(__name__)
        self.config = config
        self.model = self._initialize_bert()
        self.nlp = spacy.load("en_core_web_trf")

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
            hdbscan: The initialized HDBSCAN model.
            lda: The initialized Latent Dirichlet Allocation model.
        """

        self.config = config
        self.hdbscan = HDBSCAN(
            min_cluster_size=5, min_samples=3, cluster_selection_epsilon=0.3
        )
        self.lda = LatentDirichletAllocation(
            n_components=20, random_state=42, n_jobs=config.max_threads
        )

    def parallel_process(self, items, process_func, max_workers=None):
        """
        Process items in parallel using ThreadPoolExecutor.

        Args:
            self: The instance of the AnalyticsEngine.
            items: Collection of items to process.
            process_func: Function to apply to each item.
            max_workers: Maximum number of worker threads (defaults to self.config.max_threads if None).

        Returns:
            List of results from processing each item.

        Raises:
            Exception: Any exception raised during parallel processing.
        """
        if max_workers is None:
            max_workers = self.config.max_threads

        try:
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                results = list(executor.map(process_func, items))
            return results
        except Exception as e:
            logging.error(f"Error in parallel processing: {str(e)}")
            raise

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
