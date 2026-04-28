# Contributing to QML-MedImage

Thank you for your interest in contributing to QML-MedImage! This document provides guidelines and instructions for contributing.

## Getting Started

### Development Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/sebasmos/quantumVE.git
   cd QML-MedImage
   ```

2. **Create a development environment**
   ```bash
   conda create -n qml-medimage python=3.11 -y
   conda activate qml-medimage
   pip install -e ".[dev]"
   ```

3. **Verify installation**
   ```bash
   python -c "import sklearn; import qiskit; print('OK')"
   pytest tests/ -v
   ```

## Development Workflow

### Branch Naming

Use descriptive branch names:
- `feature/add-new-kernel` - for new features
- `fix/memory-leak` - for bug fixes
- `docs/update-readme` - for documentation updates
- `refactor/cleanup-utils` - for code refactoring

### Making Changes

1. Create a new branch from `main`
2. Make your changes
3. Run tests to ensure nothing is broken
4. Commit with clear, descriptive messages
5. Push and open a pull request

## Code Style

### Python Guidelines

- Follow PEP 8 style guidelines
- Use type hints where appropriate
- Write docstrings for public functions and classes
- Keep functions focused and modular

### Example Function

```python
def compute_kernel_matrix(
    X: np.ndarray,
    Y: np.ndarray,
    n_qubits: int = 2
) -> np.ndarray:
    """
    Compute quantum kernel matrix between two datasets.

    Args:
        X: First dataset of shape (n_samples_X, n_features)
        Y: Second dataset of shape (n_samples_Y, n_features)
        n_qubits: Number of qubits to use for the quantum circuit

    Returns:
        Kernel matrix of shape (n_samples_X, n_samples_Y)
    """
    # Implementation
    pass
```

## Testing

### Test Structure

The test suite is organized into categories:

| Test File | GPU Required | Description |
|-----------|--------------|-------------|
| `test_basic.py` | No | Core imports, qve module, sklearn integration |
| `test_imports.py` | Yes | cuQuantum/cupy imports, GPU functionality |
| `test_script_imports.py` | Yes | Full script import chain, circuit conversion |
| `test_qsvm_quick.py` | Yes | End-to-end QSVM with test data |

### Running Tests

```bash
# Run all tests (GPU tests auto-skip if unavailable)
pytest tests/ -v

# Run basic tests only (no GPU required, good for CI)
pytest tests/test_basic.py -v

# Run specific test file
pytest tests/test_imports.py -v

# Run with coverage
pytest tests/ -v --cov=qve

# Run GPU tests on compute node
srun --gres=gpu:1 pytest tests/test_imports.py -v
```

### Writing Tests

- Place tests in the `tests/` directory
- Name test files `test_*.py`
- Name test functions `test_*`
- Include docstrings describing what is being tested
- Use `pytest.mark.skipif` for tests requiring GPU

Example for GPU-dependent test:

```python
import pytest

def gpu_available():
    try:
        import cupy as cp
        a = cp.array([1, 2, 3])
        return True
    except Exception:
        return False

@pytest.mark.skipif(not gpu_available(), reason="GPU not available")
def test_gpu_computation():
    """Test GPU-accelerated computation."""
    import cupy as cp
    a = cp.array([1, 2, 3])
    assert int(cp.sum(a)) == 6
```

Example basic test:

```python
def test_kernel_matrix_shape():
    """Test that kernel matrix has correct shape."""
    X = np.random.randn(10, 4)
    Y = np.random.randn(5, 4)
    K = compute_kernel_matrix(X, Y)
    assert K.shape == (10, 5)
```

### Test Requirements

Before submitting a PR:
- All existing tests must pass
- New features should include tests
- Bug fixes should include regression tests

## Pull Request Guidelines

### Before Submitting

1. **Run tests**: `pytest tests/ -v`
2. **Check code style**: Ensure code follows project conventions
3. **Update documentation**: If your changes affect usage, update relevant docs

### PR Description

Include in your PR description:
- Summary of changes
- Motivation and context
- How to test the changes
- Any breaking changes

### Review Process

1. PRs require at least one approval before merging
2. Address review feedback promptly
3. Keep PRs focused - one feature/fix per PR

## Project Structure

```
QML-MedImage/
├── scripts/              # Training and evaluation scripts
│   ├── svm_insurance*.py    # Classical SVM scripts
│   └── qsvm_cuda_*.py       # Quantum SVM scripts
├── slurm/                # SLURM job submission scripts
├── qve/                  # Core quantum kernel module
├── tests/                # Test suite
├── docs/                 # Documentation
├── pre-processing/       # Data preparation notebooks
└── Results/              # Output directory for results
```

## Reporting Issues

### Bug Reports

When reporting bugs, include:
- Python version and OS
- Steps to reproduce
- Expected vs actual behavior
- Error messages and stack traces
- Relevant environment info (CUDA version, etc.)

### Feature Requests

For feature requests, describe:
- The problem you're trying to solve
- Your proposed solution
- Any alternatives considered

## Questions?

- Open an issue on GitHub for questions
- Check existing issues and documentation first

## License

By contributing, you agree that your contributions will be licensed under the MIT License.
