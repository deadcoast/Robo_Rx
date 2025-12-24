# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Robo Rx is a modular document processing and task management framework for high-performance data analysis and workflow orchestration. It processes Markdown documents through feature extraction, NLP analysis, and task chain execution.

## Commands

### Setup
```bash
uv venv && source .venv/bin/activate
uv sync
python -m spacy download en_core_web_trf
```

### Running Tests
```bash
pytest src/tests/                           # Run all tests
pytest src/tests/core/ChainCore_test.py     # Run single test file
pytest -k "test_init"                       # Run tests matching pattern
```

### Linting & Formatting
```bash
ruff check src/                             # Lint
ruff check --fix src/                       # Auto-fix
black src/                                  # Format
mypy src/                                   # Type check
```

### Running the Application
```bash
python main.py
```

## Architecture

### Processing Pipeline

The system follows a pipeline architecture where data flows through distinct stages:

1. **Document Ingestion** → `MarkdownProcessor` reads and normalizes markdown files
2. **Feature Extraction** → `FeatureProcessor` generates embeddings via `BERTEmbedding` and NLP features via `NLPCore`
3. **Analytics** → `AnalyticsEngine` performs clustering (`ClusteringEngine`) and topic modeling (`TopicModeling`)
4. **Execution** → `ExecutionCore`/`ExecutionController` manages task execution with `ChainExecutor` handling task chains

### Core Module Relationships

```
src/
├── core/           # Processing engines and orchestration
│   ├── ChainCore.py       # ChainManager, ChainExecutor - task chain orchestration
│   ├── ExecutionCore.py   # ExecutionCore, ExecutionController - execution flow
│   ├── FeatureCore.py     # FeatureProcessor, AnalyticsEngine - feature/analytics
│   └── NLPCore.py         # NLP processing pipeline
├── managers/       # State, resource, and execution management
│   ├── ResourceManager.py # Manages numpy/torch/pandas resources with type safety
│   └── StateManager.py    # Application state
├── processors/     # Data transformation
│   ├── MarkdownProcessor.py   # Vault processing, metadata extraction
│   └── ProcessingPool.py      # Parallel processing
├── config/         # Configuration dataclasses
│   ├── SystemConfig.py        # Core system configuration
│   ├── TaskChainConfig.py     # Chain execution config
│   └── EngineConfig.py        # Engine-specific config
├── Results/        # Result dataclasses
│   ├── ExecutionResult.py
│   ├── ChainResult.py
│   └── ContinuationResult.py
└── validators/     # Validation logic
    └── ValidatorCore.py
```

### Key Patterns

- **Async Processing**: Core processors use `async/await` for non-blocking operations
- **Dataclasses**: Configuration and result types use `@dataclass` for immutability
- **Metrics Collection**: Each processing stage tracks metrics (timing, counts, errors)
- **Resource Management**: `ResourceManager` handles typed resources (numpy arrays, torch tensors, pandas DataFrames) with context manager support

### Task Chain System

Tasks are organized into chains with dependencies and priorities:
- `TaskChainConfig` defines chain configuration (id, priority, dependencies, resource requirements)
- `ChainManager` orchestrates chains via priority queue
- `ChainExecutor` executes individual chains, computing `ChainMetrics`

### Test Structure

Tests mirror the source structure in `src/tests/`:
- Use pytest fixtures for test setup
- Tests are named `{Module}_test.py`
- Mock external dependencies (spacy, torch models)
