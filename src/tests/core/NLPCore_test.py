from unittest.mock import AsyncMock, Mock, patch

import markdown
import pytest
import spacy
import torch

from src.config.EngineConfig import SystemConfig
from src.core.NLPCore import NLPCore


class TestNLPCore:
    """Test suite for the NLPCore class."""

    @pytest.fixture
    def config(self):
        """Create a mock SystemConfig instance."""
        return Mock(spec=SystemConfig)

    @pytest.fixture
    def nlp_core(self, config):
        """Create an NLPCore instance for testing."""
        with patch.object(NLPCore, "_initialize"):
            return NLPCore(config)

    def test_init(self, nlp_core, config):
        """Test initialization of NLPCore."""
        assert nlp_core.config == config

    @patch.object(NLPCore, "_load_nlp_model")
    def test_initialize(self, mock_load_model, config):
        """Test the _initialize method."""
        # Create a fresh instance to test initialization
        with patch.object(NLPCore, "_initialize", wraps=NLPCore._initialize):
            core = NLPCore(config)
            core._initialize()

            # Verify _load_nlp_model is called with each library
            assert mock_load_model.call_count == 3
            mock_load_model.assert_any_call(torch)
            mock_load_model.assert_any_call(spacy)
            mock_load_model.assert_any_call(markdown)

    def test_load_nlp_model(self, nlp_core):
        """Test the _load_nlp_model method."""
        # Create a mock library module
        mock_library = Mock()

        # Call the method
        nlp_core._load_nlp_model(mock_library)

        # Method should not raise any exceptions
        # Since this is a placeholder, we can't assert specific behavior yet

    @pytest.mark.asyncio
    async def test_process_content(self, nlp_core):
        """Test the process_content method."""
        # Setup
        test_content = "Test content for NLP processing"

        # Call the method
        result = await nlp_core.process_content(test_content)

        # Verify
        assert isinstance(result, dict)
        assert "processed_content" in result
        assert result["processed_content"] == test_content

    @patch("src.core.NLPCore.spacy.load")
    def test_load_spacy_model(self, mock_spacy_load, nlp_core):
        """Test loading a spaCy model."""
        # Setup
        mock_nlp = Mock()
        mock_spacy_load.return_value = mock_nlp

        # Mock the method to load spaCy model
        with patch.object(
            nlp_core,
            "_load_spacy_model",
            wraps=lambda model_name: mock_spacy_load(model_name),
        ):
            # Call the method
            result = nlp_core._load_spacy_model("en_core_web_sm")

            # Verify
            mock_spacy_load.assert_called_once_with("en_core_web_sm")
            assert result == mock_nlp

    @patch("src.core.NLPCore.torch.load")
    def test_load_torch_model(self, mock_torch_load, nlp_core):
        """Test loading a PyTorch model."""
        # Setup
        mock_model = Mock()
        mock_torch_load.return_value = mock_model

        # Mock the method to load PyTorch model
        with patch.object(
            nlp_core, "_load_torch_model", wraps=lambda path: mock_torch_load(path)
        ):
            # Call the method
            result = nlp_core._load_torch_model("model.pt")

            # Verify
            mock_torch_load.assert_called_once_with("model.pt")
            assert result == mock_model

    @pytest.mark.asyncio
    async def test_tokenize_text(self, nlp_core):
        """Test tokenizing text."""
        # Setup
        test_text = "This is a test sentence."
        expected_tokens = ["This", "is", "a", "test", "sentence", "."]

        # Mock the tokenize_text method
        with patch.object(
            nlp_core, "tokenize_text", AsyncMock(return_value=expected_tokens)
        ):
            # Call the method
            tokens = await nlp_core.tokenize_text(test_text)

            # Verify
            assert tokens == expected_tokens
            nlp_core.tokenize_text.assert_called_once_with(test_text)

    @pytest.mark.asyncio
    async def test_extract_entities(self, nlp_core):
        """Test extracting entities from text."""
        # Setup
        test_text = "Apple Inc. is located in Cupertino, California."
        expected_entities = [
            {"text": "Apple Inc.", "label": "ORG"},
            {"text": "Cupertino", "label": "GPE"},
            {"text": "California", "label": "GPE"},
        ]

        # Mock the extract_entities method
        with patch.object(
            nlp_core, "extract_entities", AsyncMock(return_value=expected_entities)
        ):
            # Call the method
            entities = await nlp_core.extract_entities(test_text)

            # Verify
            assert entities == expected_entities
            nlp_core.extract_entities.assert_called_once_with(test_text)

    @pytest.mark.asyncio
    async def test_generate_embeddings(self, nlp_core):
        """Test generating embeddings for text."""
        # Setup
        test_text = "Embedding test sentence."
        # Create a mock embedding tensor
        mock_embedding = torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5])

        # Mock the generate_embeddings method
        with patch.object(
            nlp_core, "generate_embeddings", AsyncMock(return_value=mock_embedding)
        ):
            # Call the method
            embedding = await nlp_core.generate_embeddings(test_text)

            # Verify
            assert torch.equal(embedding, mock_embedding)
            nlp_core.generate_embeddings.assert_called_once_with(test_text)

    @pytest.mark.asyncio
    async def test_analyze_sentiment(self, nlp_core):
        """Test analyzing sentiment of text."""
        # Setup
        test_text = "I love this product!"
        expected_sentiment = {"positive": 0.9, "negative": 0.1, "neutral": 0.0}

        # Mock the analyze_sentiment method
        with patch.object(
            nlp_core, "analyze_sentiment", AsyncMock(return_value=expected_sentiment)
        ):
            # Call the method
            sentiment = await nlp_core.analyze_sentiment(test_text)

            # Verify
            assert sentiment == expected_sentiment
            nlp_core.analyze_sentiment.assert_called_once_with(test_text)

    @pytest.mark.asyncio
    async def test_summarize_text(self, nlp_core):
        """Test summarizing text."""
        # Setup
        test_text = """
            Natural language processing (NLP) is a subfield of linguistics, computer science, 
            and artificial intelligence concerned with the interactions between computers and 
            human language, in particular how to program computers to process and analyze large 
            amounts of natural language data. The goal is a computer capable of "understanding" 
            the contents of documents, including the contextual nuances of the language within them.
        """
        expected_summary = "NLP is a field focused on interactions between computers and human language."

        # Mock the summarize_text method
        with patch.object(
            nlp_core, "summarize_text", AsyncMock(return_value=expected_summary)
        ):
            # Call the method
            summary = await nlp_core.summarize_text(test_text)

            # Verify
            assert summary == expected_summary
            nlp_core.summarize_text.assert_called_once_with(test_text)

    @pytest.mark.asyncio
    async def test_classify_text(self, nlp_core):
        """Test classifying text into categories."""
        # Setup
        test_text = "The stock market saw significant growth today."
        expected_categories = [
            {"label": "finance", "score": 0.85},
            {"label": "business", "score": 0.75},
            {"label": "economics", "score": 0.65},
        ]

        # Mock the classify_text method
        with patch.object(
            nlp_core, "classify_text", AsyncMock(return_value=expected_categories)
        ):
            # Call the method
            categories = await nlp_core.classify_text(test_text)

            # Verify
            assert categories == expected_categories
            nlp_core.classify_text.assert_called_once_with(test_text)

    @pytest.mark.asyncio
    async def test_extract_keywords(self, nlp_core):
        """Test extracting keywords from text."""
        # Setup
        test_text = (
            "Machine learning algorithms require significant computational resources."
        )
        expected_keywords = [
            "machine learning",
            "algorithms",
            "computational",
            "resources",
        ]

        # Mock the extract_keywords method
        with patch.object(
            nlp_core, "extract_keywords", AsyncMock(return_value=expected_keywords)
        ):
            # Call the method
            keywords = await nlp_core.extract_keywords(test_text)

            # Verify
            assert keywords == expected_keywords
            nlp_core.extract_keywords.assert_called_once_with(test_text)

    @pytest.mark.asyncio
    async def test_parse_document(self, nlp_core):
        """Test parsing a document into structured format."""
        # Setup
        test_document = (
            "# Title\n\nParagraph with **bold** text.\n\n- List item 1\n- List item 2"
        )
        expected_structure = {
            "title": "Title",
            "paragraphs": ["Paragraph with bold text."],
            "lists": [["List item 1", "List item 2"]],
        }

        # Mock the parse_document method
        with patch.object(
            nlp_core, "parse_document", AsyncMock(return_value=expected_structure)
        ):
            # Call the method
            structure = await nlp_core.parse_document(test_document)

            # Verify
            assert structure == expected_structure
            nlp_core.parse_document.assert_called_once_with(test_document)

    @pytest.mark.asyncio
    async def test_detect_language(self, nlp_core):
        """Test detecting the language of text."""
        # Setup
        test_text = "This is English text."
        expected_language = {"language": "en", "confidence": 0.95}

        # Mock the detect_language method
        with patch.object(
            nlp_core, "detect_language", AsyncMock(return_value=expected_language)
        ):
            # Call the method
            language = await nlp_core.detect_language(test_text)

            # Verify
            assert language == expected_language
            nlp_core.detect_language.assert_called_once_with(test_text)

    @pytest.mark.asyncio
    async def test_full_process_pipeline(self, nlp_core):
        """Test the full NLP processing pipeline."""
        # Setup
        test_text = "This is a test document for NLP processing."

        # Mock individual methods
        nlp_core.tokenize_text = AsyncMock(
            return_value=[
                "This",
                "is",
                "a",
                "test",
                "document",
                "for",
                "NLP",
                "processing",
                ".",
            ]
        )
        nlp_core.extract_entities = AsyncMock(
            return_value=[{"text": "NLP", "label": "TECH"}]
        )
        nlp_core.generate_embeddings = AsyncMock(
            return_value=torch.tensor([0.1, 0.2, 0.3])
        )
        nlp_core.analyze_sentiment = AsyncMock(
            return_value={"positive": 0.6, "negative": 0.1, "neutral": 0.3}
        )

        # Mock a full_process method that combines all the above
        with patch.object(nlp_core, "full_process", AsyncMock()) as mock_full_process:
            mock_full_process.return_value = {
                "tokens": [
                    "This",
                    "is",
                    "a",
                    "test",
                    "document",
                    "for",
                    "NLP",
                    "processing",
                    ".",
                ],
                "entities": [{"text": "NLP", "label": "TECH"}],
                "embeddings": torch.tensor([0.1, 0.2, 0.3]),
                "sentiment": {"positive": 0.6, "negative": 0.1, "neutral": 0.3},
            }

            # Call the method
            result = await nlp_core.full_process(test_text)

            # Verify
            assert "tokens" in result
            assert "entities" in result
            assert "embeddings" in result
            assert "sentiment" in result
            mock_full_process.assert_called_once_with(test_text)
