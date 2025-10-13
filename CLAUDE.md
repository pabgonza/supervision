# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Supervision is a Python computer vision library that provides reusable utilities for object detection, tracking, annotation, and dataset management. It's designed to be model-agnostic and work with popular ML frameworks like Ultralytics YOLO, Transformers, and MMDetection.

## Development Commands

**IMPORTANT:** Always use the `supervision_dev` conda environment for all commands and tests:
```bash
conda run -n supervision_dev <command>
```

### Environment Setup
```bash
# Install in development mode with all dependencies
pip install -e ".[dev,docs,build]"

# Install pre-commit hooks
pre-commit install
```

### Code Quality and Testing
```bash
# Run linting and formatting (uses Ruff)
ruff check --fix supervision test examples
ruff format supervision test examples

# Run pre-commit checks on all files
pre-commit run --all-files

# Run tests
pytest

# Run tests with coverage
pytest --cov=supervision

# Run specific test module
pytest test/detection/test_core.py

# Run tests in parallel
pytest -n auto
```

### Documentation
```bash
# Serve documentation locally
mkdocs serve

# Build documentation
mkdocs build

# Deploy docs to gh-pages
mkdocs gh-deploy
```

### Build and Release
```bash
# Build package
python -m build

# Check package before release
twine check dist/*
```

## Code Architecture

### Core Structure
- **`supervision/`**: Main library package
  - **`detection/`**: Object detection utilities and `Detections` class
  - **`annotators/`**: Visual annotation tools (boxes, labels, masks, etc.)
  - **`dataset/`**: Dataset loading/saving for COCO, YOLO, Pascal VOC formats
  - **`tracker/`**: Object tracking implementations (ByteTracker, etc.)
  - **`geometry/`**: Geometric primitives and utilities
  - **`metrics/`**: Evaluation metrics for detection and tracking
  - **`utils/`**: Utility functions and helpers
  - **`validators/`**: Input validation functions

### Key Components
- **`Detections`**: Core class representing object detection results with bounding boxes, confidence scores, class IDs, and optional masks/tracking data
- **Annotators**: Modular visualization components that can be chained together
- **LineZone/PolygonZone**: Geometric zones for counting and tracking objects
- **Dataset classes**: Unified interface for different annotation formats

### Design Patterns
- Model-agnostic design with format converters for popular frameworks
- Composable annotators that can be chained
- Numpy-based operations for performance
- Consistent API patterns across components

## Testing

Tests are organized in the `test/` directory mirroring the main package structure. Use pytest for running tests:

```bash
# Run all tests
pytest

# Run specific test categories
pytest test/detection/
pytest test/annotators/
pytest test/dataset/
```

## Code Style

- **Formatting**: Uses Ruff for linting and formatting (configured in `pyproject.toml`)
- **Docstrings**: Google-style docstrings
- **Type hints**: Full type annotations required
- **Line length**: 88 characters (Black-compatible)
- **Import sorting**: Handled by Ruff

## Pre-commit Hooks

The repository uses pre-commit hooks for code quality:
- Trailing whitespace removal
- YAML/TOML validation  
- Ruff linting and formatting
- Bandit security checks
- Codespell for typo detection

## Development Workflow

1. Create feature branch with conventional naming (`feat/`, `fix/`, `docs/`, etc.)
2. Make changes following code style guidelines
3. Add tests for new functionality
4. Ensure all pre-commit hooks pass
5. Create PR with conventional commit messages
6. PR targets `develop` branch (not `main`)

## Commit Guidelines

- Use conventional commit format: `type: description`
- **NEVER** include references to Claude, Claude Code, or AI tools in commit messages
- Keep commits focused and descriptive
- No emojis in commit messages
- Follow existing commit message style in the repository

## Common Tasks

### Adding New Annotator
1. Implement in `supervision/annotators/core.py`
2. Add to `__init__.py` imports
3. Add tests in `test/annotators/`
4. Update documentation

### Adding Model Connector
1. Add converter method to `supervision/detection/core.py` in `Detections` class
2. Follow pattern: `@classmethod from_<framework_name>`
3. Add comprehensive tests
4. Update README examples

### Dataset Format Support
1. Implement loader/saver in `supervision/dataset/formats/`
2. Add methods to `DetectionDataset` class
3. Add format-specific tests
4. Update dataset documentation