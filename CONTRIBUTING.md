# Contributing to Uni-Layer

Thank you for your interest in contributing to Uni-Layer! This document provides guidelines and instructions for contributing.

## Table of Contents

1. [Code of Conduct](#code-of-conduct)
2. [Getting Started](#getting-started)
3. [Development Setup](#development-setup)
4. [Contributing Guidelines](#contributing-guidelines)
5. [Adding New Metrics](#adding-new-metrics)
6. [Adding Model Support](#adding-model-support)
7. [Testing](#testing)
8. [Documentation](#documentation)
9. [Submitting Changes](#submitting-changes)

## Code of Conduct

We are committed to providing a welcoming and inclusive environment. Please be respectful and constructive in all interactions.

## Getting Started

1. Fork the repository on GitHub
2. Clone your fork locally:
   ```bash
   git clone https://github.com/YOUR_USERNAME/Uni-Layer.git
   cd Uni-Layer
   ```

3. Add the upstream repository:
   ```bash
   git remote add upstream https://github.com/GeoffreyWang1117/Uni-Layer.git
   ```

## Development Setup

### Install Development Dependencies

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install package in editable mode with dev dependencies
pip install -e ".[dev]"
```

### Install Pre-commit Hooks (Optional but Recommended)

```bash
pip install pre-commit
pre-commit install
```

## Contributing Guidelines

### Code Style

We follow PEP 8 style guidelines with some modifications:

- Line length: 100 characters
- Use Black for code formatting
- Use isort for import sorting

Format your code before committing:

```bash
black uni_layer/
isort uni_layer/
```

### Commit Messages

Write clear, descriptive commit messages:

```
Add Fisher Information metric for layer importance

- Implement empirical Fisher Information calculation
- Add support for both classification and regression
- Include comprehensive docstrings and examples
```

### Branch Naming

Use descriptive branch names:

- `feature/metric-name` - for new metrics
- `feature/model-support` - for new model architectures
- `bugfix/issue-description` - for bug fixes
- `docs/topic` - for documentation updates

## Adding New Metrics

### Metric Structure

All metrics should inherit from `LayerMetric` base class:

```python
from uni_layer.core.base_metric import LayerMetric
from typing import Dict, Any, Optional
import torch.nn as nn

class YourMetric(LayerMetric):
    """
    Brief description of what your metric measures.

    Detailed explanation of:
    - What the metric computes
    - How to interpret the results
    - When to use this metric

    Args:
        param1: Description of parameter 1
        param2: Description of parameter 2

    Returns:
        Dictionary with metric values

    Example:
        >>> metric = YourMetric(param1=value1)
        >>> result = metric.compute(model, layer, ...)
    """

    def __init__(self, param1=default1, param2=default2, **kwargs):
        super().__init__(
            name="your_metric_name",
            category="category",  # e.g., "optimization", "spectral", etc.
            requires_gradient=False,  # True if needs gradients
            requires_data=True,  # True if needs data samples
            **kwargs
        )
        self.param1 = param1
        self.param2 = param2

    def compute(
        self,
        model: nn.Module,
        layer: nn.Module,
        layer_name: str,
        layer_idx: int,
        data_loader: Optional[Any] = None,
        device: str = "cuda",
        **kwargs
    ) -> Dict[str, float]:
        """
        Compute the metric.

        Args:
            model: Full model
            layer: Specific layer to analyze
            layer_name: Name of the layer
            layer_idx: Index of the layer
            data_loader: Data loader (if requires_data=True)
            device: Device to run on

        Returns:
            Dictionary with metric values
        """
        # Your metric computation logic here

        return {
            "your_metric": value,
            "your_metric_std": std_value,  # Optional
        }
```

### Metric Checklist

- [ ] Inherits from `LayerMetric`
- [ ] Has comprehensive docstring
- [ ] Implements `compute()` method
- [ ] Returns dictionary with metric values
- [ ] Handles errors gracefully
- [ ] Includes type hints
- [ ] Works with different model architectures
- [ ] Has unit tests
- [ ] Documented in `docs/METRICS.md`
- [ ] Added to `uni_layer/metrics/__init__.py`

### Adding to Framework

1. Create metric file:
   ```
   uni_layer/metrics/category/your_metric.py
   ```

2. Update category `__init__.py`:
   ```python
   from uni_layer.metrics.category.your_metric import YourMetric
   __all__ = [..., "YourMetric"]
   ```

3. Update main metrics `__init__.py`:
   ```python
   from uni_layer.metrics.category.your_metric import YourMetric
   __all__ = [..., "YourMetric"]
   ```

4. Add documentation to `docs/METRICS.md`

5. Add example to `examples/`

6. Add tests to `tests/metrics/test_your_metric.py`

## Adding Model Support

To improve layer extraction for a specific architecture:

1. Update `uni_layer/utils/layer_utils.py`:
   - Add architecture detection in `get_architecture_family()`
   - Add layer extraction logic in `get_model_layers()`
   - Add layer type identification in `identify_layer_type()`

2. Create example in `examples/`:
   ```
   examples/your_architecture_analysis.py
   ```

3. Document in README.md

## Testing

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=uni_layer --cov-report=html

# Run specific test file
pytest tests/metrics/test_gradient_norm.py

# Run specific test
pytest tests/metrics/test_gradient_norm.py::test_gradient_norm_basic
```

### Writing Tests

Create test files in `tests/` directory:

```python
import pytest
import torch
import torch.nn as nn
from uni_layer.metrics import YourMetric

def test_your_metric_basic():
    """Test basic functionality"""
    # Setup
    model = nn.Linear(10, 5)
    metric = YourMetric()

    # Test
    result = metric.compute(...)

    # Assert
    assert "your_metric" in result
    assert result["your_metric"] >= 0

def test_your_metric_with_different_architectures():
    """Test with different model types"""
    # Test with CNN, Transformer, etc.
    pass
```

## Documentation

### Docstring Format

Use Google-style docstrings:

```python
def function(arg1: int, arg2: str) -> bool:
    """
    Brief description.

    Detailed description if needed.

    Args:
        arg1: Description of arg1
        arg2: Description of arg2

    Returns:
        Description of return value

    Raises:
        ValueError: When input is invalid

    Example:
        >>> result = function(1, "test")
        >>> print(result)
        True
    """
```

### README Updates

If adding major features:
- Update main README.md
- Add to feature list
- Update examples
- Update installation instructions if needed

## Submitting Changes

### Pull Request Process

1. Update your fork:
   ```bash
   git fetch upstream
   git checkout main
   git merge upstream/main
   ```

2. Create feature branch:
   ```bash
   git checkout -b feature/your-feature-name
   ```

3. Make changes and commit:
   ```bash
   git add .
   git commit -m "Description of changes"
   ```

4. Push to your fork:
   ```bash
   git push origin feature/your-feature-name
   ```

5. Create Pull Request on GitHub

### PR Checklist

- [ ] Code follows style guidelines
- [ ] All tests pass
- [ ] New tests added for new features
- [ ] Documentation updated
- [ ] CHANGELOG.md updated (if applicable)
- [ ] No merge conflicts
- [ ] Descriptive PR title and description

### PR Description Template

```markdown
## Description
Brief description of changes

## Type of Change
- [ ] Bug fix
- [ ] New feature (metric, model support, etc.)
- [ ] Documentation update
- [ ] Performance improvement
- [ ] Code refactoring

## Testing
- [ ] Unit tests added/updated
- [ ] Manual testing performed
- [ ] All tests pass

## Checklist
- [ ] Code follows style guidelines
- [ ] Documentation updated
- [ ] No breaking changes (or documented)
```

## Questions?

- Open an issue for bugs or feature requests
- Start a discussion for questions
- Contact maintainers for sensitive issues

## Recognition

Contributors will be acknowledged in:
- README.md Contributors section
- Release notes
- Paper acknowledgments (for significant contributions)

Thank you for contributing to Uni-Layer! 🎉
