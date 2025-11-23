# Contributing to FAORU

Thank you for your interest in contributing to FAORU! This document provides guidelines for contributing to the project.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Process](#development-process)
- [Coding Standards](#coding-standards)
- [Testing](#testing)
- [Pull Request Process](#pull-request-process)

## Code of Conduct

We expect all contributors to follow our code of conduct:

- Be respectful and inclusive
- Focus on constructive feedback
- Collaborate openly and transparently
- Help others learn and grow

> **Note**: The associated research paper is currently under peer review at TPAMI. This codebase represents the implementation described in the submission.

## Getting Started

### Development Environment

1. **Fork and clone the repository:**

```bash
git clone https://github.com/YOUR_USERNAME/faoru.git
cd faoru
```

2. **Create a virtual environment:**

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install development dependencies:**

```bash
pip install -e ".[dev]"
```

4. **Install pre-commit hooks:**

```bash
pre-commit install
```

### Project Structure

```
faoru/
├── faoru/              # Core implementation
│   ├── residual.py     # FAORU residual module
│   ├── variants.py     # Lambda parameterizations
│   └── transforms.py   # Frequency transforms
├── models/             # Model implementations
│   ├── vit.py         # Vision Transformers
│   └── resnet.py      # ResNets
├── tools/              # Analysis tools
├── configs/            # Training configurations
├── scripts/            # Reproduction scripts
└── tests/              # Unit tests
```

## Development Process

### Branching Strategy

- `main`: Stable release branch
- `develop`: Development branch for integration
- `feature/*`: New features
- `bugfix/*`: Bug fixes
- `hotfix/*`: Critical fixes for production

### Workflow

1. **Create a new branch:**

```bash
git checkout -b feature/your-feature-name
```

2. **Make your changes** with frequent commits

3. **Write tests** for new functionality

4. **Run tests and linting:**

```bash
pytest tests/
flake8 faoru/ models/ tools/
black faoru/ models/ tools/
```

5. **Commit your changes:**

```bash
git commit -m "feat: Add new frequency transform"
```

Follow [Conventional Commits](https://www.conventionalcommits.org/):
- `feat:` New feature
- `fix:` Bug fix
- `docs:` Documentation changes
- `test:` Test additions/modifications
- `refactor:` Code refactoring
- `perf:` Performance improvements
- `chore:` Maintenance tasks

## Coding Standards

### Python Style Guide

We follow [PEP 8](https://pep8.org/) with these additional guidelines:

**Code Formatting:**
- Use [Black](https://github.com/psf/black) for code formatting (88 character line length)
- Use [isort](https://pycqa.github.io/isort/) for import sorting
- Use type hints for all function signatures

**Example:**

```python
from typing import Tuple, Optional
import torch
import torch.nn as nn

def compute_alpha_k(
    F: torch.Tensor,
    X: torch.Tensor,
    epsilon: float = 1e-8
) -> torch.Tensor:
    """
    Compute per-frequency projection coefficients.
    
    Args:
        F: [B, S, D//2+1] frequency-domain features (complex)
        X: [B, S, D//2+1] frequency-domain skip connection (complex)
        epsilon: Numerical stability constant
    
    Returns:
        [B, S, D//2+1] projection coefficients α_k
    """
    # Real part of inner product
    inner_prod = F.real * X.real + F.imag * X.imag  # [B, S, D//2+1]
    
    # Squared norm of X
    x_norm_sq = X.real**2 + X.imag**2  # [B, S, D//2+1]
    
    # α_k = Re(<F, X>) / (||X||² + ε)
    alpha_k = inner_prod / (x_norm_sq + epsilon)
    
    return alpha_k
```

**Documentation:**
- All modules, classes, and functions must have docstrings
- Use [Google-style docstrings](https://google.github.io/styleguide/pyguide.html#38-comments-and-docstrings)
- Include type hints in docstrings for clarity

**Naming Conventions:**
- Classes: `PascalCase` (e.g., `FAORUResidual`)
- Functions/variables: `snake_case` (e.g., `compute_alpha_k`)
- Constants: `UPPER_SNAKE_CASE` (e.g., `DEFAULT_EPSILON`)
- Private members: prefix with `_` (e.g., `_internal_helper`)

## Testing

### Writing Tests

All new features must include tests. Use pytest for testing:

```python
# tests/test_residual.py
import torch
import pytest
from faoru.residual import FAORUResidual

def test_faoru_residual_shape():
    """Test FAORU residual output shape"""
    residual = FAORUResidual(dim=768, variant='learnable')
    x = torch.randn(2, 196, 768)
    f_x = torch.randn(2, 196, 768)
    
    output = residual(x, f_x)
    
    assert output.shape == (2, 196, 768)

def test_faoru_residual_identity():
    """Test FAORU with zero residual returns identity"""
    residual = FAORUResidual(dim=768, variant='learnable')
    x = torch.randn(2, 196, 768)
    f_x = torch.zeros(2, 196, 768)
    
    output = residual(x, f_x)
    
    assert torch.allclose(output, x, atol=1e-5)

@pytest.mark.parametrize('variant', ['learnable', 'piecewise', 'smooth'])
def test_faoru_variants(variant):
    """Test all FAORU variants"""
    residual = FAORUResidual(dim=768, variant=variant)
    x = torch.randn(2, 196, 768)
    f_x = torch.randn(2, 196, 768)
    
    output = residual(x, f_x)
    
    assert output.shape == (2, 196, 768)
    assert not torch.isnan(output).any()
```

### Running Tests

```bash
# Run all tests
pytest tests/

# Run with coverage
pytest tests/ --cov=faoru --cov-report=html

# Run specific test file
pytest tests/test_residual.py

# Run specific test
pytest tests/test_residual.py::test_faoru_residual_shape
```

## Pull Request Process

### Before Submitting

1. **Ensure all tests pass:**

```bash
pytest tests/
```

2. **Check code quality:**

```bash
flake8 faoru/ models/ tools/
black --check faoru/ models/ tools/
isort --check faoru/ models/ tools/
```

3. **Update documentation** if needed

4. **Add entry to CHANGELOG.md**

### Submitting PR

1. **Push your branch:**

```bash
git push origin feature/your-feature-name
```

2. **Create pull request** on GitHub with:
   - Clear title following Conventional Commits
   - Detailed description of changes
   - Link to related issues
   - Screenshots/examples if applicable

3. **PR Template:**

```markdown
## Description
Brief description of the changes

## Motivation
Why are these changes needed?

## Changes
- Change 1
- Change 2

## Testing
How was this tested?

## Checklist
- [ ] Tests pass locally
- [ ] Code follows style guidelines
- [ ] Documentation updated
- [ ] CHANGELOG.md updated
```

### Review Process

1. **Automated checks** must pass (CI/CD)
2. **Code review** by at least one maintainer
3. **Address feedback** and update PR
4. **Squash and merge** once approved

## Types of Contributions

### Bug Reports

Submit bug reports as GitHub issues with:
- Clear title
- Steps to reproduce
- Expected vs actual behavior
- Environment details (OS, Python version, PyTorch version)
- Minimal reproducible example

### Feature Requests

Submit feature requests as GitHub issues with:
- Clear use case
- Proposed implementation (if any)
- Expected benefits
- Potential drawbacks

### Code Contributions

Welcome contributions:
- **New frequency transforms** (e.g., Wavelet, Walsh)
- **Model integrations** (e.g., ConvNeXt, Swin Transformer)
- **Analysis tools** (e.g., feature visualization, ablation studies)
- **Documentation improvements**
- **Performance optimizations**

## Questions?

- Open a [GitHub Discussion](https://github.com/YOUR_ORG/faoru/discussions)
- Email the maintainers at [email@example.com]

Thank you for contributing! 🎉
