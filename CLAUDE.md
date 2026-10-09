# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Supervision is a Python computer vision library that provides reusable utilities for object detection, tracking, annotation, and dataset management. It's designed to be model-agnostic and work with popular ML frameworks like Ultralytics YOLO, Transformers, and MMDetection.

**This is a fork**: `origin` is `pabgonza/supervision`, `upstream` is `roboflow/supervision`.
Work here targets PRs to upstream, which is why commits follow their conventions and PRs go
to `develop`. Sync with `git fetch upstream && git merge upstream/develop` on a
`feat/upstream-sync-<version>` branch (last: 0.27.0).

## Change workflow (OpenSpec)

Non-trivial work goes through `openspec/changes/<change-id>/` in THIS repo (initialized 2026-09-15; planning artifacts in Spanish, code and commits in English following upstream). Run `openspec` commands from this directory, not from a consumer's checkout. `openspec list` shows the active changes; the first one was `serialize-stream-capture-reconnect` (`StreamCapture` reconnects from two threads over the same `cv2.VideoCapture`, SIGSEGV on a silent RTSP stream). Consumers install the fork with pip from `develop` HEAD: a fix here reaches Ball Counter v2 only after rebuilding its detector image.

## Development Commands

Run commands and tests with the `supervision_dev` env's Python directly (conda is not on PATH; numpy there has no MKL):
```powershell
& "C:\Users\pabgo\miniforge3\envs\supervision_dev\python.exe" -m <module> ...   # e.g. -m pytest, -m ruff
```

### Environment Setup
```bash
# Install in development mode with all dependencies
pip install -e . --group dev --group docs --group build   # pip >= 25.1

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

# Run specific test module
pytest test/detection/test_core.py
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