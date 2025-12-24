# Robo Claud

A modular, extensible document processing and task management system designed for high-performance data analysis and workflow orchestration.

## Overview

Robo Claud is a sophisticated Python framework that provides comprehensive document processing, feature extraction, task management, and system analysis capabilities. The system is built with a focus on modularity, allowing for flexible configuration and extension of its core components.

## System Architecture

Robo Claud follows a modular architecture with specialized core components that work together to form a complete processing pipeline:

```tree
Robo Claud
├── src/                      # Source code directory
│   ├── main.py               # Main entry point and core classes
│   ├── AdvancedCore.py       # Enhanced feature processing
│   ├── ChainCore.py          # Task chain management
│   ├── CompilationCore.py    # System analysis and report compilation
│   ├── DataCore.py           # Data frame and dataset processing
│   ├── EarlyCore.py          # Initial processing components
│   ├── EnhancedCore.py       # Improved processing capabilities
│   ├── ExecutionCore.py      # Task execution management
│   ├── FeatureCore.py        # Feature extraction and analytics
│   ├── OptimizedCore.py      # Performance-optimized components
│   ├── ProccessCore.py       # Core processing functionality
│   ├── UpdateCore.py         # System update and tracking
│   └── ValidatorCore.py      # Data and process validation
```

## Core Components

### Document Processing

- **MarkdownProcessor**: Processes Markdown documents, extracts metadata, and normalizes content.
- **FeatureProcessor**: Generates feature vectors from processed documents using embeddings and NLP.

### Task Management

- **TaskRegistryManager**: Manages task registration and execution.
- **ChainManager**: Orchestrates task chains with dependencies and priorities.
- **ExecutionCore**: Controls the execution flow of tasks and monitors performance.

### Analytics

- **AnalyticsEngine**: Performs analytics on feature matrices using clustering and topic modeling.
- **AdvancedFeatureProcessor**: Extends basic feature processing with topic modeling and graph features.

### System Components

- **CompilationCore**: Coordinates system analysis, report generation, and validation.
- **SystemScanner**: Performs deep scanning of system parameters and diagnostics.
- **UpdateTracker**: Monitors the progress of system updates.

---

## Requirements

Robo Claud relies on several Python packages:

- Python 3.8+
- asyncio
- numpy
- networkx
- torch
- pyarrow
- safetensors
- spaCy (with en_core_web_trf model)

## Installation

---
### 01

Clone the repo:

```sh
git clone https://github.com/deadcoast/robo_claud.git
cd robo_claud
```

> [!NOTE]
> The repo is managed by uv package manager
> To install `uv`

---
### 02
#### macOS

```sh
# [A] Install uv using homebrew
brew install uv
```

```sh
# [B] Install uv using curl 
curl -LsSf https://astral.sh/uv/install.sh | sh
```

```sh
# [C] install using wget
wget -qO- https://astral.sh/uv/install.sh | sh
```

### 03
#### Windows

```sh
# [A] install via pip
pip install uv
```

```ps1
# [B] powershell installation
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

> The -ExecutionPolicy ByPass flag allows running the installation script from the internet.

## Setup .venv

---
### 04
#### macOS .venv

```sh
uv venv
source .venv/bin/activate 
```

#### Windows .venv

```ps1
.venv\Scripts\activate
```

---
### 05
#### Initiate a uv project

```sh
uv init
```

Install the required dependencies:

```sh
pip install -r requirements.txt
```

Install the spaCy model:

```sh
python -m spacy download en_core_web_trf
```

---

### Usage

#### Basic Usage

```python
from src.main import ProgressTracker, MarkdownProcessor
from src.FeatureCore import SystemConfig

# Initialize configuration
config = SystemConfig(
    max_threads=8,
    batch_size=1000,
    processing_mode="CUDA_ENABLED"
)

# Create a processor
processor = MarkdownProcessor(config)

# Process documents
async def process_documents(vault_path):
    results = await processor.process_vault(vault_path)
    return results
```

### Advanced Features

```python
from src.AdvancedCore import AdvancedFeatureProcessor
from src.ChainCore import ChainManager, TaskChainConfig
from pathlib import Path

# Initialize advanced processor
advanced_processor = AdvancedFeatureProcessor(config)

# Process documents with enhanced features
async def generate_advanced_features(docs):
    feature_set = await advanced_processor.generate_enhanced_features(docs)
    return feature_set

# Set up a task chain
chain_config = TaskChainConfig(
    chain_id="process_and_analyze",
    priority=1,
    dependencies=["data_load"],
    resource_requirements={"memory": 4.0, "cpu": 2.0}
)

chain_manager = ChainManager(config)
# Add chain to execution queue
```

### System Analysis

```python
from src.CompilationCore import CompilationEngine
from src.CompilationCore import SystemData

# Create a compilation engine
engine = CompilationEngine()

# Prepare system data
system_data = SystemData()

# Execute compilation process
result = engine.execute_compilation(system_data)
```

## Architecture Design

Robo Claud is built with the following architectural principles:

1. **Modularity**: Each core component focuses on a specific aspect of functionality
2. **Asynchronous Processing**: Extensive use of async/await for non-blocking operations
3. **Pipeline Architecture**: Data flows through distinct processing stages
4. **Error Handling**: Comprehensive error tracking and reporting throughout the system
5. **Metrics Collection**: Performance and execution metrics are gathered at each stage

## License

[Specify your license here]

## Contributing

[Your contribution guidelines]
