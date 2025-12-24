"""
MarkdownProcessor class
"""

import asyncio
import logging
import re
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class MarkdownProcessingError(Exception):
    """Exception raised for errors in the Markdown processing."""

    message: str
    file_path: Optional[Path] = None

    def __str__(self) -> str:
        """
        Returns a string representation of the object.
        """
        if self.file_path:
            return f"{self.message} - File: {self.file_path}"
        return self.message

    def __repr__(self) -> str:
        """
        Returns a string representation of the object.
        """
        return f"MarkdownProcessingError(message={self.message}, file_path={self.file_path})"

    def __eq__(self, other) -> bool:
        if isinstance(other, MarkdownProcessingError):
            return self.message == other.message
        return False


class MarkdownProcessor:
    """
    MarkdownProcessor class
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the MarkdownProcessor.
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        self.logger.addHandler(logging.StreamHandler())
        self.logger.propagate = False
        self.max_threads = config.get("max_threads", 4)  # Default to 4 if not specified
        self.batch_size = config.get("batch_size", 10)  # Default to 10 if not specified
        self.executor = ThreadPoolExecutor(max_workers=self.max_threads)
        self.logger.info("MarkdownProcessor initialized.")
        self.logger.info(
            f"Config: max_threads={self.max_threads}, batch_size={self.batch_size}"
        )

    async def process_vault(self, vault_path: Path) -> Dict[str, Any]:
        """
        Process a vault of Markdown files and return the aggregated results.

        Args:
            vault_path (Path): The path to the vault directory.

        Returns:
            Dict[str, Any]: The aggregated results of processing the vault.

        Raises:
            MarkdownProcessingError: If there is an error processing the vault.
        """
        self.logger.info(f"Processing vault: {vault_path}")
        try:
            # Find all markdown files in the vault recursively
            files = list(vault_path.rglob("*.md"))
            self.logger.info(f"Found {len(files)} markdown files")

            # Process files in batches for better memory management
            batches = [
                files[i : i + self.batch_size]
                for i in range(0, len(files), self.batch_size)
            ]

            # Process each batch of files concurrently
            results = []
            for batch in batches:
                batch_results = await asyncio.gather(
                    *[self._process_file(file) for file in batch]
                )
                results.extend([r for r in batch_results if r is not None])

            # Aggregate the results
            aggregated = self._aggregate_results(results)
            self.logger.info(f"Successfully processed {len(results)} files")
            return aggregated

        except Exception as e:
            error_msg = f"Error processing vault: {str(e)}"
            self.logger.error(error_msg)
            raise MarkdownProcessingError(error_msg) from e

    async def _process_file(self, file_path: Path) -> Dict[str, Any]:
        """
        Process a single Markdown file and return the processed data.

        Args:
            file_path (Path): The path to the Markdown file.

        Returns:
            Dict[str, Any]: The processed data including metadata, content, and statistics.
        """
        try:
            self.logger.debug(f"Processing file: {file_path}")

            # Read the file content
            content = await self._read_file(file_path)

            # Extract metadata (YAML frontmatter) from content
            metadata = self._extract_metadata(content)

            # Normalize content (remove frontmatter, standardize line endings, etc.)
            normalized_content = self._normalize_content(content)

            # Compute statistics about the content
            stats = self._compute_stats(normalized_content)

            return {
                "path": str(file_path),
                "filename": file_path.name,
                "metadata": metadata,
                "content": normalized_content,
                "stats": stats,
                "last_modified": file_path.stat().st_mtime,
            }

        except Exception as e:
            self.logger.warning(f"Error processing file {file_path}: {str(e)}")
            return {"path": str(file_path), "filename": file_path.name, "error": str(e)}

    async def _read_file(self, file_path: Path) -> str:
        """
        Read the content of a file asynchronously.

        Args:
            file_path (Path): The path to the file.

        Returns:
            str: The content of the file.
        """
        try:
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(
                self.executor, self._read_file_sync, file_path
            )
        except Exception as e:
            raise MarkdownProcessingError(
                f"Failed to read file: {str(e)}", file_path
            ) from e

    def _read_file_sync(self, file_path: Path) -> str:
        """
        Synchronous method to read a file.

        Args:
            file_path (Path): The path to the file.

        Returns:
            str: The content of the file.
        """
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()

    def _extract_metadata(self, content: str) -> Dict[str, Any]:
        """
        Extract YAML frontmatter metadata from markdown content.

        Args:
            content (str): The markdown content.

        Returns:
            Dict[str, Any]: The extracted metadata.
        """
        # Simple YAML frontmatter extraction (between --- markers)
        metadata = {}

        # Use walrus operator to simplify assignment and conditional
        if frontmatter_match := re.match(
            r"^---\s*\n(.+?)\n---\s*\n", content, re.DOTALL
        ):
            try:
                # For proper YAML parsing, would use pyyaml package
                # But here we'll use a simple key-value extraction for demonstration
                frontmatter = frontmatter_match[1]
                for line in frontmatter.split("\n"):
                    if ":" in line:
                        key, value = line.split(":", 1)
                        metadata[key.strip()] = value.strip()
            except Exception as e:
                self.logger.warning(f"Failed to parse frontmatter: {str(e)}")

        return metadata

    def _normalize_content(self, content: str) -> str:
        """
        Normalize the markdown content by removing frontmatter and standardizing line endings.

        Args:
            content (str): The original markdown content.

        Returns:
            str: The normalized content.
        """
        # Remove frontmatter if present
        normalized = re.sub(r"^---\s*\n.+?\n---\s*\n", "", content, flags=re.DOTALL)

        # Standardize line endings to '\n'
        normalized = normalized.replace("\r\n", "\n").replace("\r", "\n")

        # Remove extra whitespace and ensure content starts without leading whitespace
        normalized = normalized.strip()

        return normalized

    def _compute_stats(self, content: str) -> Dict[str, Any]:
        """
        Compute statistics from the markdown content.

        Args:
            content (str): The normalized markdown content.

        Returns:
            Dict[str, Any]: Statistics about the content.
        """
        # Split into paragraphs, sections, etc.
        lines = content.split("\n")
        paragraphs = re.split(r"\n{2,}", content)

        # Count headings by level
        headings = {
            f"h{i}": len(re.findall(f"^{'#' * i}\s+.+$", content, re.MULTILINE))
            for i in range(1, 7)
        }

        # Get links and images
        links = re.findall(r"\[([^\]]+)\]\(([^)]+)\)", content)
        images = re.findall(r"!\[([^\]]+)\]\(([^)]+)\)", content)

        # Calculate word count (simple approach)
        words = re.findall(r"\w+", content)

        return {
            "char_count": len(content),
            "word_count": len(words),
            "line_count": len(lines),
            "paragraph_count": len(paragraphs),
            "headings": headings,
            "link_count": len(links),
            "image_count": len(images),
        }

    def _aggregate_results(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Aggregate the results from processing multiple files.

        Args:
            results (List[Dict[str, Any]]): The list of results from processing files.

        Returns:
            Dict[str, Any]: The aggregated results.
        """
        # Files with errors
        errors = [r for r in results if "error" in r]

        # Successfully processed files
        successful = [r for r in results if "error" not in r]

        # Total statistics
        total_stats = {
            "char_count": 0,
            "word_count": 0,
            "line_count": 0,
            "paragraph_count": 0,
            "headings": {f"h{i}": 0 for i in range(1, 7)},
            "link_count": 0,
            "image_count": 0,
        }

        # Aggregate statistics
        for result in successful:
            stats = result.get("stats", {})
            for key in [
                "char_count",
                "word_count",
                "line_count",
                "paragraph_count",
                "link_count",
                "image_count",
            ]:
                total_stats[key] += stats.get(key, 0)

            # Aggregate headings
            for h_level, count in stats.get("headings", {}).items():
                total_stats["headings"][h_level] += count

        return {
            "total_files": len(results),
            "successful_files": len(successful),
            "error_files": len(errors),
            "errors": errors,
            "files": successful,
            "total_stats": total_stats,
        }

    def _aggregate_enhanced_results(
        self, results: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Aggregate results from processed files.

        Args:
            results (List[Dict[str, Any]]): The results of processing files.

        Returns:
            Dict[str, Any]: The aggregated results.
        """
        try:
            pass
        except Exception as e:
            self.logger.error(f"Results aggregation error: {str(e)}")
            raise MarkdownProcessingError(str(e)) from e

    def _enhanced_file_processing(self, file_path: Path) -> Dict[str, Any]:
        """
        Process a single Markdown file with enhanced processing.

        Args:
            file_path (Path): The path to the Markdown file.

        Returns:
            Dict[str, Any]: The processed data of the file, including path, metadata, content, and stats.
        """
        try:
            pass
        except Exception as e:
            self.logger.error(f"Enhanced file processing error: {str(e)}")
            raise MarkdownProcessingError(str(e)) from e

    def _enhanced_vault_processing(self, vault_path: Path) -> Dict[str, Any]:
        """
        Process a vault of Markdown files with enhanced processing.

        Args:
            vault_path (Path): The path to the vault directory.

        Returns:
            Dict[str, Any]: The processed data of the vault, including path, metadata, content, and stats.
        """
        try:
            pass
        except Exception as e:
            self.logger.error(f"Enhanced vault processing error: {str(e)}")
            raise MarkdownProcessingError(str(e)) from e
