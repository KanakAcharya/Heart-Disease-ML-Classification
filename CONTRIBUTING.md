# Contributing to Heart Disease ML Classification

Thank you for your interest in contributing! This document provides guidelines and instructions for contributing to the project.

## Code of Conduct

Please be respectful and inclusive in all interactions. We are committed to providing a welcoming and inspiring community.

## Getting Started

### Prerequisites
- Python 3.8+
- Git
- Docker (optional, for containerized testing)

### Setup Development Environment

1. Fork the repository
2. Clone your fork:
   ```bash
   git clone https://github.com/YOUR_USERNAME/Heart-Disease-ML-Classification.git
   cd Heart-Disease-ML-Classification
   ```

3. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

4. Install dependencies:
   ```bash
   pip install -r requirements.txt
   pip install pytest pytest-cov pylint
   ```

## Development Workflow

### Creating a Feature Branch

```bash
git checkout -b feature/your-feature-name
```

### Code Standards

- Follow PEP 8 style guide
- Use type hints where applicable
- Write docstrings for all functions and classes
- Keep functions small and focused
- Use meaningful variable names

### Testing

Run tests before committing:

```bash
python -m pytest tests/ -v
```

Run tests with coverage:

```bash
python -m pytest tests/ --cov=src --cov-report=html
```

### Linting

```bash
pylint src/
```

## Submitting Changes

1. Commit with clear, descriptive messages:
   ```bash
   git commit -m "Add feature: description of what was added"
   ```

2. Push to your fork:
   ```bash
   git push origin feature/your-feature-name
   ```

3. Submit a Pull Request with:
   - Clear description of changes
   - Reference to related issues (if any)
   - Confirmation that tests pass

## Types of Contributions

- **Bug Fixes**: Report issues and submit PRs with fixes
- **Features**: Propose new features via issues first
- **Documentation**: Improve README, docstrings, and guides
- **Tests**: Add tests for new features or edge cases
- **Performance**: Suggest optimizations with benchmarks

## Areas for Contribution

- Model improvements (hyperparameter tuning, new algorithms)
- Data preprocessing enhancements
- Feature engineering techniques
- API improvements
- Documentation and examples
- CI/CD enhancements

## Reporting Issues

When reporting bugs, please include:
- Python version
- Operating system
- Detailed steps to reproduce
- Expected vs. actual behavior
- Error messages or stack traces

## Questions?

Feel free to open an issue with the "question" label or reach out via GitHub discussions.

Thank you for contributing!
