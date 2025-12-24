"""
Core natural language processing functionality.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

import markdown
import spacy
import torch

from src.config.EngineConfig import SystemConfig


@dataclass
class NLPCore:
    config: SystemConfig

    def __init__(self, config: SystemConfig):
        self.config = config

    async def process_content(self, content: str) -> Dict[str, Any]:
        # Placeholder for actual NLP processing
        return {"processed_content": content}

    def _initialize(self):
        """
        Initializes the NLP core.

        This method is responsible for initializing the NLP core,
        setting up the necessary components and resources required for
        natural language processing.
        """
        self._load_nlp_model(torch)
        self._load_nlp_model(spacy)
        self._load_nlp_model(markdown)

    def _load_nlp_model(self, library_module):
        """
        Loads the NLP model.

        This method is responsible for loading the NLP model,
        setting up the necessary components and resources required for
        natural language processing.

        Args:
            library_module: The library module to use for loading the model.

        Returns:
            None
        """
        if library_module == spacy:
            # Load spaCy model based on configuration
            model_name = (
                self.config.nlp_model_name
                if hasattr(self.config, "nlp_model_name")
                else "en_core_web_sm"
            )
            try:
                self.spacy_nlp = spacy.load(model_name)
                print(f"Loaded spaCy model: {model_name}")
            except OSError:
                # Fallback to smaller model if the specified one isn't available
                print(
                    f"Model {model_name} not found. Downloading en_core_web_sm instead."
                )
                spacy.cli.download("en_core_web_sm")
                self.spacy_nlp = spacy.load("en_core_web_sm")

        elif library_module == torch:
            # Initialize PyTorch-based models if specified in config
            self.device = torch.device(
                "cuda"
                if torch.cuda.is_available()
                and hasattr(self.config, "use_gpu")
                and self.config.use_gpu
                else "cpu"
            )
            print(f"PyTorch using device: {self.device}")

            # Initialize empty models dictionary to store any loaded models
            self.torch_models = {}

            # Load transformer models if specified in config
            if hasattr(self.config, "transformer_model_name"):
                from transformers import AutoModel, AutoTokenizer

                model_name = self.config.transformer_model_name
                try:
                    self.torch_models["tokenizer"] = AutoTokenizer.from_pretrained(
                        model_name
                    )
                    self.torch_models["model"] = AutoModel.from_pretrained(
                        model_name
                    ).to(self.device)
                    print(f"Loaded transformer model: {model_name}")
                except Exception as e:
                    print(f"Failed to load transformer model {model_name}: {str(e)}")

        elif library_module == markdown:
            # No actual model to load for markdown, just set it up
            self.markdown_extensions = (
                self.config.markdown_extensions
                if hasattr(self.config, "markdown_extensions")
                else ["extra", "smarty"]
            )

    def _read_file(self, file_path: Path) -> str:
        """
        Reads a file and returns its content.

        This method is responsible for reading a file and returning its content.

        Args:
            file_path: The path to the file to read.

        Returns:
            The content of the file as a string.

        Raises:
            FileNotFoundError: If the file does not exist.
            PermissionError: If the file cannot be read due to permission issues.
            UnicodeDecodeError: If the file cannot be decoded as UTF-8.
        """
        try:
            # Check if file exists
            if not file_path.exists():
                raise FileNotFoundError(f"File not found: {file_path}")

            # Try to read the file with UTF-8 encoding
            with open(file_path, "r", encoding="utf-8") as file:
                content = file.read()

            # Process markdown content if the file has a markdown extension
            if file_path.suffix.lower() in [".md", ".markdown"]:
                content = markdown.markdown(
                    content, extensions=getattr(self, "markdown_extensions", ["extra"])
                )

            return content
        except UnicodeDecodeError:
            # Fallback to read as binary if UTF-8 decoding fails
            try:
                with open(file_path, "rb") as file:
                    binary_content = file.read()

                # Try different encodings
                for encoding in ["latin-1", "cp1252", "iso-8859-1"]:
                    try:
                        return binary_content.decode(encoding)
                    except UnicodeDecodeError:
                        continue

                # Last resort: ignore errors
                return binary_content.decode("utf-8", errors="ignore")
            except Exception as e:
                print(f"Error reading file {file_path}: {str(e)}")
                raise

    def _extract_enhanced_metadata(self, content: str) -> Dict[str, Any]:
        """
        Extracts enhanced metadata from content.

        This method is responsible for extracting enhanced metadata from content.
        Analyzes the text to extract information about its structure, language,
        readability, and other metadata.

        Args:
            content: The text content to extract metadata from.

        Returns:
            A dictionary containing metadata about the content, including length,
            language statistics, readability metrics, and content structure.
        """
        if not hasattr(self, "spacy_nlp"):
            # Initialize spaCy if not already done
            self._load_nlp_model(spacy)

        # Process the content with spaCy
        doc = self.spacy_nlp(content)

        # Basic text metrics
        char_count = len(content)
        word_count = len([token for token in doc if not token.is_punct])
        sentence_count = len(list(doc.sents))
        paragraph_count = len([p for p in content.split("\n\n") if p.strip()])

        # Word length distribution
        word_lengths = [len(token.text) for token in doc if not token.is_punct]
        avg_word_length = sum(word_lengths) / max(1, len(word_lengths))

        # Sentence length distribution
        sentence_lengths = [
            len([token for token in sent if not token.is_punct]) for sent in doc.sents
        ]
        avg_sentence_length = sum(sentence_lengths) / max(1, len(sentence_lengths))

        # Parts of speech distribution
        pos_counts = {}
        for token in doc:
            pos = token.pos_
            pos_counts[pos] = pos_counts.get(pos, 0) + 1

        # Calculate readability metrics (Flesch Reading Ease)
        if sentence_count > 0 and word_count > 0:
            words_per_sentence = word_count / sentence_count
            syllables = sum(
                self._count_syllables(token.text) for token in doc if not token.is_punct
            )
            syllables_per_word = syllables / max(1, word_count)
            flesch_reading_ease = (
                206.835 - (1.015 * words_per_sentence) - (84.6 * syllables_per_word)
            )
        else:
            flesch_reading_ease = 0

        # Named entity recognition
        entity_counts = {}
        for ent in doc.ents:
            entity_counts[ent.label_] = entity_counts.get(ent.label_, 0) + 1

        # Language detection
        language = doc.lang_

        # Check for code snippets or structured data
        code_markers = [
            "def ",
            "class ",
            "function",
            "import ",
            "var ",
            "let ",
            "const ",
        ]
        potential_code = any(marker in content for marker in code_markers)

        # Check for URLs and emails
        urls = [token.text for token in doc if token.like_url]
        emails = [token.text for token in doc if token.like_email]

        # Use PyTorch for sentiment analysis if available
        sentiment_score = None
        if hasattr(self, "torch_models") and "model" in self.torch_models:
            try:
                from transformers import pipeline

                sentiment_analyzer = pipeline(
                    "sentiment-analysis", device=getattr(self, "device", -1)
                )
                # Process in chunks if content is too long
                max_length = 512
                if len(content) > max_length:
                    chunks = [
                        content[i : i + max_length]
                        for i in range(0, len(content), max_length)
                    ]
                    sentiments = [sentiment_analyzer(chunk)[0] for chunk in chunks]
                    # Average the sentiment scores
                    sentiment_score = sum(float(s["score"]) for s in sentiments) / len(
                        sentiments
                    )
                else:
                    result = sentiment_analyzer(content)[0]
                    sentiment_score = (
                        float(result["score"]) if "score" in result else None
                    )
            except Exception as e:
                print(f"Sentiment analysis failed: {str(e)}")

        return {
            "char_count": char_count,
            "word_count": word_count,
            "sentence_count": sentence_count,
            "paragraph_count": paragraph_count,
            "avg_word_length": avg_word_length,
            "avg_sentence_length": avg_sentence_length,
            "pos_distribution": pos_counts,
            "readability": {
                "flesch_reading_ease": flesch_reading_ease,
                "interpretation": self._interpret_flesch_score(flesch_reading_ease),
            },
            "entities": entity_counts,
            "language": language,
            "contains_code": potential_code,
            "urls": urls,
            "emails": emails,
            "sentiment_score": sentiment_score,
        }

    def _count_syllables(self, word: str) -> int:
        """
        Counts the number of syllables in a word using a simple heuristic.

        Args:
            word: The word to count syllables for.

        Returns:
            The estimated number of syllables.
        """
        word = word.lower()
        # Remove non-alpha characters
        word = "".join([c for c in word if c.isalpha()])
        if not word:
            return 0

        # Specific exceptions
        exception_dict = {
            "area": 3,
            "data": 2,
            "piano": 3,
            "video": 3,
            "idea": 3,
            "diary": 3,
            "ion": 2,
        }
        if word in exception_dict:
            return exception_dict[word]

        # Count vowel groups
        vowels = "aeiouy"
        count = 0
        prev_vowel = False
        for char in word:
            is_vowel = char in vowels
            if is_vowel and not prev_vowel:
                count += 1
            prev_vowel = is_vowel

        # Adjust for silent 'e' at the end
        if word.endswith("e") and len(word) > 2 and word[-2] not in vowels:
            count -= 1

        # Ensure at least one syllable
        return max(1, count)

    def _interpret_flesch_score(self, score: float) -> str:
        """
        Interprets a Flesch Reading Ease score.

        Args:
            score: The Flesch Reading Ease score.

        Returns:
            A string interpretation of the score.
        """
        if score >= 90:
            return "Very Easy - 5th grade"
        elif score >= 80:
            return "Easy - 6th grade"
        elif score >= 70:
            return "Fairly Easy - 7th grade"
        elif score >= 60:
            return "Standard - 8th-9th grade"
        elif score >= 50:
            return "Fairly Difficult - 10th-12th grade"
        elif score >= 30:
            return "Difficult - College"
        else:
            return "Very Difficult - College Graduate"

    def _process_content(self, content: str) -> Dict[str, Any]:
        """
        Processes content using the NLP model.

        This method is responsible for processing content using the NLP model.
        Applies various NLP techniques to analyze and extract information from text.

        Args:
            content: The text content to process.

        Returns:
            A dictionary containing the processed content, including extracted
            features, entities, sentiment analysis, and other NLP outputs.
        """
        # Initialize models if needed
        if not hasattr(self, "spacy_nlp"):
            self._load_nlp_model(spacy)

        # Process with spaCy for basic NLP tasks
        doc = self.spacy_nlp(content)

        # Extract entities
        entities = [
            {
                "text": ent.text,
                "label": ent.label_,
                "start": ent.start_char,
                "end": ent.end_char,
            }
            for ent in doc.ents
        ]

        # Extract key phrases (noun chunks)
        noun_chunks = [
            {
                "text": chunk.text,
                "root_text": chunk.root.text,
                "root_pos": chunk.root.pos_,
            }
            for chunk in doc.noun_chunks
        ]

        # Part-of-speech analysis
        pos_tags = [
            {
                "text": token.text,
                "pos": token.pos_,
                "tag": token.tag_,
                "dep": token.dep_,
            }
            for token in doc
        ]

        # Sentence segmentation
        sentences = [
            {
                "text": sent.text,
                "start": sent.start_char,
                "end": sent.end_char,
            }
            for sent in doc.sents
        ]

        # Use PyTorch for advanced NLP if available
        transformer_output = {}
        if hasattr(self, "torch_models") and "model" in self.torch_models:
            try:
                # Extract embeddings
                transformer_output = self._extract_enhanced_embeddings(content)
            except Exception as e:
                print(f"Transformer processing failed: {str(e)}")

        # Extract features
        features = self._extract_enhanced_features(content)

        # Extract metadata
        metadata = self._extract_enhanced_metadata(content)

        return {
            "entities": entities,
            "noun_chunks": noun_chunks,
            "pos_tags": pos_tags,
            "sentences": sentences,
            "features": features,
            "metadata": metadata,
            "transformer": transformer_output,
        }

    def _aggregate_enhanced_results(
        self, results: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Aggregates enhanced results.

        This method is responsible for aggregating enhanced results from multiple
        processing operations, combining them into a single comprehensive result.

        Args:
            results: A list of dictionaries containing the results to aggregate.

        Returns:
            A dictionary containing the aggregated results.
        """
        # Initialize aggregated results
        aggregated = {
            "entities": [],
            "keywords": [],
            "noun_chunks": [],
            "sentences": [],
            "features": {},
            "metadata": {
                "document_count": len(results),
                "char_count": 0,
                "word_count": 0,
                "sentence_count": 0,
            },
            "sentiment": {
                "scores": [],
                "average": None,
                "variance": None,
            },
        }

        # Initialize counters for weighted averaging
        total_words = 0
        feature_sums = {}
        entity_counts = {}
        keyword_scores = {}

        # Aggregate results
        for result in results:
            # Update document metadata
            metadata = result.get("metadata", {})
            aggregated["metadata"]["char_count"] += metadata.get("char_count", 0)
            word_count = metadata.get("word_count", 0)
            aggregated["metadata"]["word_count"] += word_count
            aggregated["metadata"]["sentence_count"] += metadata.get(
                "sentence_count", 0
            )
            total_words += word_count

            # Append entities with de-duplication
            for entity in result.get("entities", []):
                entity_text = entity.get("text", "").lower()
                if entity_text in entity_counts:
                    entity_counts[entity_text]["count"] += 1
                else:
                    entity_counts[entity_text] = {
                        "text": entity_text,
                        "label": entity.get("label"),
                        "count": 1,
                    }

            # Accumulate features for averaging
            for key, value in result.get("features", {}).items():
                if isinstance(value, (int, float)):
                    if key in feature_sums:
                        feature_sums[key] += (
                            value * word_count
                        )  # Weight by document size
                    else:
                        feature_sums[key] = value * word_count

            # Collect sentiment scores if available
            sentiment = result.get("sentiment_score")
            if sentiment is not None:
                aggregated["sentiment"]["scores"].append(sentiment)

            # Collect keywords with scores
            for keyword in result.get("keywords", []):
                keyword_text = keyword.get("text", "").lower()
                if keyword_text in keyword_scores:
                    keyword_scores[keyword_text]["score"] += keyword.get("count", 1)
                else:
                    keyword_scores[keyword_text] = {
                        "text": keyword_text,
                        "pos": keyword.get("pos"),
                        "score": keyword.get("count", 1),
                    }

        # Finalize aggregated entities (sort by frequency)
        aggregated["entities"] = sorted(
            entity_counts.values(), key=lambda x: x["count"], reverse=True
        )

        # Finalize aggregated keywords (sort by score)
        aggregated["keywords"] = sorted(
            keyword_scores.values(), key=lambda x: x["score"], reverse=True
        )

        # Calculate weighted averages for features
        if total_words > 0:
            aggregated["features"] = {
                key: value / total_words for key, value in feature_sums.items()
            }

        # Calculate sentiment statistics
        if aggregated["sentiment"]["scores"]:
            import numpy as np

            scores = aggregated["sentiment"]["scores"]
            aggregated["sentiment"]["average"] = np.mean(scores)
            aggregated["sentiment"]["variance"] = (
                np.var(scores) if len(scores) > 1 else 0
            )

        return aggregated

    def _extract_enhanced_features(self, content: str) -> Dict[str, Any]:
        """
        Extracts enhanced features from content.

        This method is responsible for extracting enhanced features from content,
        focusing on linguistic and semantic characteristics of the text.

        Args:
            content: The text content to extract features from.

        Returns:
            A dictionary containing extracted features, including lexical diversity,
            sentiment polarity, subjectivity, and readability metrics.
        """
        if not hasattr(self, "spacy_nlp"):
            self._load_nlp_model(spacy)

        # Process the content with spaCy
        doc = self.spacy_nlp(content)

        # Calculate lexical diversity (unique words / total words)
        tokens = [
            token.text.lower()
            for token in doc
            if not token.is_punct and not token.is_space
        ]
        unique_tokens = set(tokens)
        lexical_diversity = len(unique_tokens) / max(1, len(tokens))

        # Calculate lexical density (content words / total words)
        content_tokens = [
            token for token in doc if token.pos_ in ["NOUN", "VERB", "ADJ", "ADV"]
        ]
        lexical_density = len(content_tokens) / max(
            1, len([t for t in doc if not t.is_punct and not t.is_space])
        )

        # Calculate average word length
        avg_word_length = sum(
            len(token.text) for token in doc if not token.is_punct
        ) / max(1, len([t for t in doc if not t.is_punct]))

        # Calculate sentence complexity (average tokens per sentence)
        sentences = list(doc.sents)
        if sentences:
            avg_tokens_per_sentence = sum(
                len([t for t in sent if not t.is_punct]) for sent in sentences
            ) / len(sentences)
        else:
            avg_tokens_per_sentence = 0

        # Calculate stopword ratio
        stopwords = [token for token in doc if token.is_stop]
        stopword_ratio = len(stopwords) / max(1, len(tokens))

        # Calculate clause indicators (approximate using dependency parsing)
        clause_markers = [
            token
            for token in doc
            if token.dep_ in ["mark", "advcl", "ccomp", "xcomp", "acl"]
        ]
        clause_ratio = len(clause_markers) / max(1, len(sentences))

        # Check for formal vs. informal language markers
        formal_markers = [
            "therefore",
            "thus",
            "consequently",
            "furthermore",
            "nevertheless",
        ]
        informal_markers = ["like", "so", "pretty", "really", "gonna", "wanna"]

        formal_count = sum(token.text.lower() in formal_markers for token in doc)
        informal_count = sum(token.text.lower() in informal_markers for token in doc)

        formality_score = (formal_count - informal_count) / max(
            1, formal_count + informal_count + 1
        )

        # Check for first, second, and third person pronouns
        first_person = sum(
            token.text.lower() in ["i", "me", "my", "mine", "we", "us", "our", "ours"]
            for token in doc
        )
        second_person = sum(
            token.text.lower() in ["you", "your", "yours"] for token in doc
        )
        third_person = sum(
            token.text.lower()
            in [
                "he",
                "him",
                "his",
                "she",
                "her",
                "hers",
                "it",
                "its",
                "they",
                "them",
                "their",
                "theirs",
            ]
            for token in doc
        )

        # Extract sentiment using pattern analysis
        positive_words = [
            "good",
            "great",
            "excellent",
            "positive",
            "wonderful",
            "amazing",
            "superb",
            "fantastic",
        ]
        negative_words = [
            "bad",
            "poor",
            "terrible",
            "negative",
            "awful",
            "horrible",
            "disappointing",
        ]

        positive_count = sum(token.text.lower() in positive_words for token in doc)
        negative_count = sum(token.text.lower() in negative_words for token in doc)

        total_sentiment_words = positive_count + negative_count
        sentiment_polarity = 0
        if total_sentiment_words > 0:
            sentiment_polarity = (
                positive_count - negative_count
            ) / total_sentiment_words

        # Use PyTorch for more advanced features if available
        advanced_features = {}
        if hasattr(self, "torch_models") and "model" in self.torch_models:
            try:
                # Extract embeddings and use for clustering or classification
                import numpy as np
                from sklearn.decomposition import PCA

                embeddings = self._extract_enhanced_embeddings(content)
                if (
                    "embedding" in embeddings
                    and isinstance(embeddings["embedding"], np.ndarray)
                    and embeddings["embedding"].size > 2
                ):
                    pca = PCA(n_components=2)
                    embedding_2d = pca.fit_transform(
                        embeddings["embedding"].reshape(1, -1)
                    )
                    advanced_features["embedding_pca_1"] = float(embedding_2d[0, 0])
                    advanced_features["embedding_pca_2"] = float(embedding_2d[0, 1])
                    advanced_features["embedding_norm"] = float(
                        np.linalg.norm(embeddings["embedding"])
                    )
            except Exception as e:
                print(f"Advanced feature extraction failed: {str(e)}")

        return {
            "lexical_diversity": lexical_diversity,
            "lexical_density": lexical_density,
            "avg_word_length": avg_word_length,
            "avg_tokens_per_sentence": avg_tokens_per_sentence,
            "stopword_ratio": stopword_ratio,
            "clause_ratio": clause_ratio,
            "formality_score": formality_score,
            "pronoun_distribution": {
                "first_person": first_person,
                "second_person": second_person,
                "third_person": third_person,
            },
            "sentiment": {
                "polarity": sentiment_polarity,
                "positive_count": positive_count,
                "negative_count": negative_count,
            },
            **advanced_features,
        }

    def _extract_enhanced_embeddings(self, content: str) -> Dict[str, Any]:
        """
        Extracts enhanced embeddings from content.

        This method uses PyTorch transformer models to generate high-quality text
        embeddings that capture semantic meaning and context.

        Args:
            content: The text content to extract embeddings from.

        Returns:
            A dictionary containing the extracted embeddings and related metadata.
        """
        # Check if PyTorch and transformers are available
        if not hasattr(self, "torch_models") or "model" not in self.torch_models:
            # Initialize PyTorch if not already done
            self._load_nlp_model(torch)

        # If still no models, use a fallback approach
        if not hasattr(self, "torch_models") or "model" not in self.torch_models:
            # Fallback to simpler embedding if transformers not available
            return self._generate_fallback_embeddings(content)

        try:
            return self._thorough_extraction(content)
        except Exception as e:
            print(f"Error generating embeddings: {str(e)}")
            return self._generate_fallback_embeddings(content)

    def _thorough_extraction(self, content):
        """
        Extracts enhanced embeddings from the given content.

        Args:
            content (str): The input content from which to extract embeddings.

        Returns:
            dict: A dictionary containing the following keys:
                - "embedding" (ndarray): The final embedding.
                - "dimension" (int): The dimension of the embedding.
                - "model" (str): The name or path of the model used for generating embeddings.
                - "is_pooled" (bool): Indicates whether the model provides pooled output.

        Raises:
            None

        Examples:
            >>> nlp_core = NLPCore()
            >>> content = "This is an example sentence."
            >>> result = nlp_core._thorough_extraction(content)
            >>> print(result)
            {
                "embedding": array([...]),
                "dimension": 768,
                "model": "bert-base-uncased",
                "is_pooled": True
            }
        """

        tokenizer = self.torch_models.get("tokenizer")
        model = self.torch_models.get("model")
        device = getattr(self, "device", torch.device("cpu"))

        # Tokenize input
        # Handle long content by truncating or splitting into chunks
        max_length = getattr(tokenizer, "model_max_length", 512)

        # If content is too long, divide into chunks and average the embeddings
        if len(content) > max_length * 4:  # Approximate character to token ratio
            # Split into paragraphs or sentences
            chunks = [p for p in content.split("\n\n") if p.strip()]
            if not chunks or len(chunks) == 1:
                # Split by sentences if no paragraphs
                if hasattr(self, "spacy_nlp"):
                    doc = self.spacy_nlp(content)
                    chunks = [sent.text for sent in doc.sents]
                else:
                    # Simple sentence splitting
                    chunks = [s.strip() for s in content.split(".") if s.strip()]

            # Limit number of chunks to process
            max_chunks = 10
            if len(chunks) > max_chunks:
                # Select representative chunks (first, middle, last, and some in between)
                indices = [
                    0,  # First chunk
                    len(chunks) - 1,  # Last chunk
                    len(chunks) // 2,  # Middle chunk
                ]
                # Add some chunks in between
                step = len(chunks) // (max_chunks - 3)
                indices.extend(
                    list(range(step, len(chunks) - 1, step))[: max_chunks - 3]
                )
                indices = sorted(set(indices))  # Remove duplicates and sort
                chunks = [chunks[i] for i in indices]

            # Process each chunk
            all_embeddings = []
            for chunk in chunks:
                # Skip empty chunks
                if not chunk.strip():
                    continue

                # Tokenize with truncation
                inputs = tokenizer(
                    chunk,
                    return_tensors="pt",
                    truncation=True,
                    max_length=max_length,
                    padding=True,
                )
                inputs = {key: val.to(device) for key, val in inputs.items()}

                # Generate embeddings
                with torch.no_grad():
                    outputs = model(**inputs)

                # Get embeddings - usually use the [CLS] token embedding
                if hasattr(outputs, "last_hidden_state"):
                    # For BERT-like models, use [CLS] token
                    embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy()
                elif hasattr(outputs, "pooler_output"):
                    # Some models provide pooled output
                    embedding = outputs.pooler_output.cpu().numpy()
                else:
                    # Fallback: average all token embeddings
                    embedding = outputs[0].mean(dim=1).cpu().numpy()

                all_embeddings.append(embedding[0])

            # Average embeddings from all chunks
            import numpy as np

            if all_embeddings:
                final_embedding = np.mean(all_embeddings, axis=0)
            else:
                # Fallback if no valid chunks
                return self._generate_fallback_embeddings(content)
        else:
            # Process the entire content at once if it's not too long
            inputs = tokenizer(
                content,
                return_tensors="pt",
                truncation=True,
                max_length=max_length,
                padding=True,
            )
            inputs = {key: val.to(device) for key, val in inputs.items()}

            # Generate embeddings
            with torch.no_grad():
                outputs = model(**inputs)

            # Extract embeddings
            if hasattr(outputs, "last_hidden_state"):
                # For BERT-like models, use [CLS] token
                final_embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy()[0]
            elif hasattr(outputs, "pooler_output"):
                # Some models provide pooled output
                final_embedding = outputs.pooler_output.cpu().numpy()[0]
            else:
                # Fallback: average all token embeddings
                final_embedding = outputs[0].mean(dim=1).cpu().numpy()[0]

        # Get embedding dimension
        embedding_dim = (
            final_embedding.shape[0]
            if hasattr(final_embedding, "shape")
            else len(final_embedding)
        )

        return {
            "embedding": final_embedding,
            "dimension": embedding_dim,
            "model": getattr(model, "name_or_path", str(type(model))),
            "is_pooled": hasattr(outputs, "pooler_output"),
        }

    def _generate_fallback_embeddings(self, content: str) -> Dict[str, Any]:
        """
        Generates fallback embeddings when transformer models are not available.

        Args:
            content: The text content to embed.

        Returns:
            A dictionary containing simple bag-of-words or TF-IDF based embeddings.
        """
        try:
            import numpy as np
            from sklearn.feature_extraction.text import TfidfVectorizer

            # Generate TF-IDF embedding
            vectorizer = TfidfVectorizer(max_features=100, stop_words="english")
            tfidf_matrix = vectorizer.fit_transform([content])
            tfidf_embedding = tfidf_matrix.toarray()[0]

            # Get feature names for interpretability
            feature_names = vectorizer.get_feature_names_out()

            # Find top features
            top_indices = np.argsort(tfidf_embedding)[-10:][::-1]  # Top 10 terms
            top_features = [
                (feature_names[i], float(tfidf_embedding[i])) for i in top_indices
            ]

            return {
                "embedding": tfidf_embedding,
                "dimension": len(tfidf_embedding),
                "model": "tfidf_fallback",
                "is_pooled": False,
                "top_features": top_features,
            }
        except Exception as e:
            print(f"Fallback embedding generation failed: {str(e)}")
            # Return an empty embedding as last resort
            return {
                "embedding": np.zeros(50),  # Small default size
                "dimension": 50,
                "model": "zeros_fallback",
                "is_pooled": False,
                "error": str(e),
            }

    def _extract_enhanced_topic_features(self, content: str) -> Dict[str, Any]:
        """
        Extracts enhanced topic features from content.

        This method is responsible for extracting enhanced topic features from content.
        Uses NLP techniques to identify main topics and themes in the text.

        Args:
            content: The text content to extract topic features from.

        Returns:
            A dictionary containing topic features, including main topics,
            keywords, and topic distributions.
        """
        if not hasattr(self, "spacy_nlp"):
            # Initialize spaCy if not already done
            self._load_nlp_model(spacy)

        # Process the content with spaCy
        doc = self.spacy_nlp(content)

        keywords = [
            {"text": token.text, "pos": token.pos_, "count": 1}
            for token in doc
            if (
                token.pos_ in ["NOUN", "PROPN"]
                and not token.is_stop
                and len(token.text) > 1
            )
        ]
        # Aggregate and count keywords
        keyword_counts = {}
        for kw in keywords:
            text = kw["text"].lower()
            if text in keyword_counts:
                keyword_counts[text]["count"] += 1
            else:
                keyword_counts[text] = {"text": text, "pos": kw["pos"], "count": 1}

        # Sort keywords by frequency
        sorted_keywords = sorted(
            keyword_counts.values(), key=lambda x: x["count"], reverse=True
        )[
            :20
        ]  # Limit to top 20 keywords

        # Get named entities as potential topics
        entities = [{"text": ent.text, "label": ent.label_} for ent in doc.ents]

        # Extract noun phrases as potential topics
        noun_phrases = [
            chunk.text for chunk in doc.noun_chunks if len(chunk.text.split()) > 1
        ]

        # Use TF-IDF like approach for topic scoring if we have transformer models loaded
        topic_scores = {}
        if hasattr(self, "torch_models") and "model" in self.torch_models:
            from sklearn.feature_extraction.text import TfidfVectorizer

            # Use paragraphs as documents for TF-IDF
            paragraphs = [p.strip() for p in content.split("\n\n") if p.strip()]
            if len(paragraphs) > 1:
                try:
                    self._extracted_from__extract_enhanced_topic_features_(
                        TfidfVectorizer, paragraphs, topic_scores
                    )
                except Exception as e:
                    print(f"TF-IDF calculation failed: {str(e)}")

        return {
            "keywords": sorted_keywords,
            "entities": entities,
            "noun_phrases": noun_phrases[:20],  # Limit to top 20 phrases
            "topic_scores": topic_scores,
            "main_topics": sorted_keywords[:5],  # Top 5 keywords as main topics
        }

    # TODO Rename this here and in `_extract_enhanced_topic_features`
    def _extracted_from__extract_enhanced_topic_features_(
        self, TfidfVectorizer, paragraphs, topic_scores
    ):
        vectorizer = TfidfVectorizer(max_features=100, stop_words="english")
        tfidf_matrix = vectorizer.fit_transform(paragraphs)
        feature_names = vectorizer.get_feature_names_out()

        # Compute average TF-IDF scores across documents
        import numpy as np

        tfidf_means = np.mean(tfidf_matrix.toarray(), axis=0)

        # Create topic scores
        for i, feature in enumerate(feature_names):
            topic_scores[feature] = float(tfidf_means[i])

    def _extract_enhanced_graph_features(self, content: str) -> Dict[str, Any]:
        """
        Extracts enhanced graph features from content.

        This method is responsible for extracting enhanced graph features from content.
        Creates a graph representation of the text based on semantic relationships.

        Args:
            content: The text content to extract graph features from.

        Returns:
            A dictionary containing graph features, including nodes, edges,
            centrality measures, and graph statistics.
        """
        if not hasattr(self, "spacy_nlp"):
            # Initialize spaCy if not already done
            self._load_nlp_model(spacy)

        # Process the content with spaCy
        doc = self.spacy_nlp(content)

        # Create a graph representation using networkx
        import networkx as nx

        G = nx.Graph()

        # Add entities as nodes
        for ent in doc.ents:
            if not G.has_node(ent.text):
                G.add_node(ent.text, type="entity", label=ent.label_)

        # Add important tokens as nodes (nouns, verbs, adjectives)
        for token in doc:
            if (
                token.pos_ in ["NOUN", "PROPN", "VERB", "ADJ"]
                and not token.is_stop
                and not G.has_node(token.text)
            ):
                G.add_node(token.text, type="token", pos=token.pos_)

        # Add edges based on sentence co-occurrence
        for sent in doc.sents:
            sent_entities = list(sent.ents)
            sent_tokens = [
                token
                for token in sent
                if token.pos_ in ["NOUN", "PROPN", "VERB", "ADJ"] and not token.is_stop
            ]

            # Connect entities that appear in the same sentence
            for i, ent1 in enumerate(sent_entities):
                for ent2 in sent_entities[i + 1 :]:
                    if G.has_node(ent1.text) and G.has_node(ent2.text):
                        if G.has_edge(ent1.text, ent2.text):
                            G[ent1.text][ent2.text]["weight"] += 1
                        else:
                            G.add_edge(
                                ent1.text, ent2.text, weight=1, type="entity_entity"
                            )

            # Connect tokens that appear in the same sentence
            for i, token1 in enumerate(sent_tokens):
                for token2 in sent_tokens[i + 1 :]:
                    if G.has_node(token1.text) and G.has_node(token2.text):
                        if G.has_edge(token1.text, token2.text):
                            G[token1.text][token2.text]["weight"] += 1
                        else:
                            G.add_edge(
                                token1.text, token2.text, weight=1, type="token_token"
                            )

        # Compute centrality measures
        centrality = {}
        if len(G.nodes) > 0:
            try:
                centrality["degree"] = nx.degree_centrality(G)
                if len(G.nodes) > 1:  # These require connected graphs
                    try:
                        centrality["betweenness"] = nx.betweenness_centrality(G)
                        centrality["closeness"] = nx.closeness_centrality(G)
                    except Exception as e:
                        print(f"Some centrality measures failed: {str(e)}")
            except Exception as e:
                print(f"Centrality calculation failed: {str(e)}")

        # Prepare output
        nodes = [
            {
                "id": node,
                "type": data.get("type", "unknown"),
                "metadata": {k: v for k, v in data.items() if k != "type"},
            }
            for node, data in G.nodes(data=True)
        ]

        edges = [
            {
                "source": u,
                "target": v,
                "weight": data.get("weight", 1),
                "type": data.get("type", "unknown"),
            }
            for u, v, data in G.edges(data=True)
        ]

        # Graph statistics
        stats = {
            "node_count": G.number_of_nodes(),
            "edge_count": G.number_of_edges(),
            "density": nx.density(G) if G.number_of_nodes() > 1 else 0,
            "components": (
                nx.number_connected_components(G) if G.number_of_nodes() > 0 else 0
            ),
        }

        return {
            "nodes": nodes,
            "edges": edges,
            "centrality": centrality,
            "stats": stats,
            "graph_object": G,  # Return the graph object for further processing
        }

    def _extract_enhanced_metadata(self, content: str) -> Dict[str, Any]:
        """
        Extracts enhanced metadata from content.

        This method is responsible for extracting enhanced metadata from content.
        Analyzes the text to extract information about its structure, language,
        readability, and other metadata.

        Args:
            content: The text content to extract metadata from.

        Returns:
            A dictionary containing metadata about the content, including length,
            language statistics, readability metrics, and content structure.
        """
        if not hasattr(self, "spacy_nlp"):
            # Initialize spaCy if not already done
            self._load_nlp_model(spacy)

        # Process the content with spaCy
        doc = self.spacy_nlp(content)

        # Basic text statistics
        char_count = len(content)
        word_count = len(
            [token for token in doc if not token.is_punct and not token.is_space]
        )
        sentence_count = len(list(doc.sents))
        paragraph_count = len([p for p in content.split("\n\n") if p.strip()])

        # Average word length
        avg_word_length = sum(
            len(token.text)
            for token in doc
            if not token.is_punct and not token.is_space
        ) / max(1, word_count)

        # Average sentence length
        avg_sentence_length = word_count / max(1, sentence_count)

        # Language detection
        from langdetect import DetectorFactory, detect

        DetectorFactory.seed = 0  # For consistent results
        try:
            language = detect(content)
        except Exception:  # Ignoring specific error details
            language = "unknown"

        # Readability scores
        import textstat

        readability = {
            "flesch_reading_ease": textstat.flesch_reading_ease(content),
            "flesch_kincaid_grade": textstat.flesch_kincaid_grade(content),
            "gunning_fog": textstat.gunning_fog(content),
            "smog_index": textstat.smog_index(content),
            "coleman_liau_index": textstat.coleman_liau_index(content),
            "automated_readability_index": textstat.automated_readability_index(
                content
            ),
            "dale_chall_readability_score": textstat.dale_chall_readability_score(
                content
            ),
        }

        # Part-of-speech distribution
        pos_counts = {}
        for token in doc:
            pos = token.pos_
            pos_counts[pos] = pos_counts.get(pos, 0) + 1

        # Named entity counts
        entity_counts = {}
        for ent in doc.ents:
            entity_type = ent.label_
            entity_counts[entity_type] = entity_counts.get(entity_type, 0) + 1

        # Content structure analysis
        structure = {
            "paragraphs": paragraph_count,
            "sentences": sentence_count,
            "words": word_count,
            "characters": char_count,
            "avg_paragraph_length": word_count / max(1, paragraph_count),
            "avg_sentence_length": avg_sentence_length,
            "avg_word_length": avg_word_length,
        }

        # Sentiment analysis
        from textblob import TextBlob

        blob = TextBlob(content)
        sentiment = {
            "polarity": blob.sentiment.polarity,  # -1.0 to 1.0 (negative to positive)
            "subjectivity": blob.sentiment.subjectivity,  # 0.0 to 1.0 (objective to subjective)
        }

        return {
            "structure": structure,
            "language": language,
            "readability": readability,
            "pos_distribution": pos_counts,
            "entity_counts": entity_counts,
            "sentiment": sentiment,
            "creation_time": (
                torch.cuda.current_stream()
                .record_event()
                .elapsed_time(torch.cuda.Event(enable_timing=True))
                if torch.cuda.is_available()
                else 0
            ),
        }

    def _extract_enhanced_stats(self, content: str) -> Dict[str, Any]:
        """
        Extracts enhanced statistics from content.

        This method is responsible for extracting enhanced statistics from content.
        Performs statistical analysis on the text to identify patterns and insights.

        Args:
            content: The text content to extract statistics from.

        Returns:
            A dictionary containing statistical analysis of the content, including
            word frequency distributions, n-gram analysis, and other statistical measures.
        """
        if not hasattr(self, "spacy_nlp"):
            # Initialize spaCy if not already done
            self._load_nlp_model(spacy)

        # Process the content with spaCy
        doc = self.spacy_nlp(content)

        # Extract words (excluding punctuation, spaces, and stop words)
        words = [
            token.text.lower()
            for token in doc
            if not token.is_punct and not token.is_space and not token.is_stop
        ]

        # Word frequency distribution
        word_freq = {}
        for word in words:
            word_freq[word] = word_freq.get(word, 0) + 1

        # Sort by frequency
        sorted_word_freq = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
        top_words = sorted_word_freq[:50]  # Top 50 words

        bigrams = [(words[i], words[i + 1]) for i in range(len(words) - 1)]
        trigrams = [
            (words[i], words[i + 1], words[i + 2]) for i in range(len(words) - 2)
        ]
        # Calculate bigram frequencies
        bigram_freq = {}
        for bg in bigrams:
            bigram_text = f"{bg[0]} {bg[1]}"
            bigram_freq[bigram_text] = bigram_freq.get(bigram_text, 0) + 1

        # Calculate trigram frequencies
        trigram_freq = {}
        for tg in trigrams:
            trigram_text = f"{tg[0]} {tg[1]} {tg[2]}"
            trigram_freq[trigram_text] = trigram_freq.get(trigram_text, 0) + 1

        # Sort n-grams by frequency
        sorted_bigram_freq = sorted(
            bigram_freq.items(), key=lambda x: x[1], reverse=True
        )
        sorted_trigram_freq = sorted(
            trigram_freq.items(), key=lambda x: x[1], reverse=True
        )

        top_bigrams = sorted_bigram_freq[:30]  # Top 30 bigrams
        top_trigrams = sorted_trigram_freq[:20]  # Top 20 trigrams

        # Calculate lexical diversity (unique words / total words)
        lexical_diversity = len(word_freq) / max(1, len(words))

        # Basic statistical measures
        word_lengths = [len(word) for word in words]
        from statistics import mean, median, stdev

        word_length_stats = {
            "mean": mean(word_lengths) if word_lengths else 0,
            "median": median(word_lengths) if word_lengths else 0,
            "std_dev": stdev(word_lengths) if len(word_lengths) > 1 else 0,
            "min": min(word_lengths, default=0),
            "max": max(word_lengths, default=0),
        }

        # Sentence length statistics
        sentence_lengths = [
            len([token for token in sent if not token.is_punct and not token.is_space])
            for sent in doc.sents
        ]
        sentence_length_stats = {
            "mean": mean(sentence_lengths) if sentence_lengths else 0,
            "median": median(sentence_lengths) if sentence_lengths else 0,
            "std_dev": stdev(sentence_lengths) if len(sentence_lengths) > 1 else 0,
            "min": min(sentence_lengths, default=0),
            "max": max(sentence_lengths, default=0),
        }

        # Calculate TF-IDF if we're using PyTorch
        tfidf_analysis = {}
        if hasattr(self, "torch_models"):
            try:
                from sklearn.feature_extraction.text import TfidfVectorizer

                # Split content into paragraphs
                paragraphs = [p.strip() for p in content.split("\n\n") if p.strip()]

                if len(paragraphs) > 1:
                    self._extracted_from__extract_enhanced_stats_110(
                        TfidfVectorizer, paragraphs, tfidf_analysis
                    )
            except Exception as e:
                print(f"TF-IDF analysis failed: {str(e)}")

        return {
            "word_frequency": dict(top_words),
            "bigram_frequency": dict(top_bigrams),
            "trigram_frequency": dict(top_trigrams),
            "lexical_diversity": lexical_diversity,
            "word_length_stats": word_length_stats,
            "sentence_length_stats": sentence_length_stats,
            "total_unique_words": len(word_freq),
            "total_words": len(words),
            "tfidf_analysis": tfidf_analysis,
        }

    # TODO Rename this here and in `_extract_enhanced_stats`
    def _extracted_from__extract_enhanced_stats_110(
        self, TfidfVectorizer, paragraphs, tfidf_analysis
    ):
        # Calculate TF-IDF
        vectorizer = TfidfVectorizer(max_features=100, stop_words="english")
        tfidf_matrix = vectorizer.fit_transform(paragraphs)
        feature_names = vectorizer.get_feature_names_out()
        import numpy as np

        tfidf_means = np.mean(tfidf_matrix.toarray(), axis=0)

        # Get most significant terms by TF-IDF
        significant_terms = [
            (feature_names[i], float(tfidf_means[i]))
            for i in np.argsort(tfidf_means)[::-1][:50]
        ]
        tfidf_analysis["significant_terms"] = significant_terms
