#!/usr/bin/env python3
import os
import shutil
from pathlib import Path

# Base directories
src_dir = Path("/Users/deadcoast/PyCharmMiscProject/robo_claud/src")
tests_dir = Path("/Users/deadcoast/PyCharmMiscProject/robo_claud/src/tests")

# Module directories
module_dirs = {
    "analyzers": [
        "AnalyzerCore",
        "CommunityAnalyst",
        "ResultAnalyzer",
        "DuplicationAnalyzer",
    ],
    "config": ["EngineConfig", "ResourceConfig", "SystemConfig", "TaskChainConfig"],
    "core": [
        "AdvancedCore",
        "ChainCore",
        "CompilationCore",
        "DataCore",
        "EarlyCore",
        "EnhancedCore",
        "ExecutionCore",
        "FeatureCore",
        "NLPCore",
        "OptimizedCore",
        "ProccessCore",
        "UpdateCore",
    ],
    "error_control": [
        "ExecutionError",
        "StateValidationError",
        "VaultProcessingError",
        "EnhancedProcessingError",
    ],
    "handlers": ["DataHandler", "ResourceAllocator"],
    "managers": [
        "ExecutionManager",
        "FileManager",
        "ResourceManager",
        "StateManager",
        "ModelPipelineManager",
    ],
    "monitors": ["ExecutionMonitor", "ProgressMonitor"],
    "processors": ["MarkdownProcessor", "ProcessingPool"],
    "Results": ["ChainResult", "ContinuationResult", "ExecutionResult"],
}

# Additional categorization for files that don't match the standard module pattern
special_cases = {
    "Normalizer": "utils",
    "MDParser": "utils",
    "MetaExtractor": "utils",
    "BERTEmbedding": "ml",
    "MetaFeatureExtractor": "ml",
    "BatchData": "data",
    "FeatureMatrix": "data",
    "ClusteringEngine": "ml",
    "TopicModeling": "ml",
    "HierarchicalClassifier": "ml",
    "BacklinkGraph": "graph",
    "SummaryEngine": "ml",
    "NoteGraph": "graph",
    "Continuation": "workflow",
    "DocumentProcessor": "processors",
    "ContentSearchEngine": "ml",
    "ContextManager": "managers",
    "KnowledgeGraph": "graph",
    "RelationExtractor": "ml",
    "VectorStore": "ml",
    "CacheManager": "managers",
    "EmbeddingEngine": "ml",
}

# Create directories if they don't exist
for module in list(module_dirs.keys()) + list(set(special_cases.values())):
    os.makedirs(tests_dir / module, exist_ok=True)

# Get all test files
test_files = [f for f in os.listdir(tests_dir) if f.endswith("_test.py")]

# Count moved files for reporting
moved_files = 0

# Move test files to appropriate directories
for test_file in test_files:
    module_name = test_file.replace("_test.py", "")

    target_dir = next(
        (module for module, classes in module_dirs.items() if module_name in classes),
        None,
    )
    # If not found, check special cases
    if target_dir is None and module_name in special_cases:
        target_dir = special_cases[module_name]

    # If we found a target directory, move the file
    if target_dir:
        source = tests_dir / test_file
        destination = tests_dir / target_dir / test_file

        # Only move if source exists and destination doesn't
        if source.exists() and not destination.exists():
            shutil.move(str(source), str(destination))
            moved_files += 1
            print(f"Moved {test_file} to {target_dir}/")

print(
    f"\nOrganization complete! Moved {moved_files} test files into module-specific directories."
)
