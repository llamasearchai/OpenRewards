# Contributing to OpenRewards

Thank you for your interest in contributing to OpenRewards! This document provides guidelines for contributing to our reward modeling platform.

## Table of Contents
- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [Contributing Process](#contributing-process)
- [Code Standards](#code-standards)
- [Testing](#testing)
- [Documentation](#documentation)
- [Submitting Changes](#submitting-changes)

## Code of Conduct

We are committed to providing a welcoming and inclusive environment for all contributors. Please read and follow our [Code of Conduct](CODE_OF_CONDUCT.md).

## Getting Started

### Prerequisites
- Python 3.11+
- Node.js 18+
- Rust 1.70+
- Git
- Docker (optional but recommended)

### Development Setup

1. **Fork and Clone the Repository**
   ```bash
   git clone https://github.com/llamasearchai/OpenRewards.git
   cd OpenRewards
   ```

2. **Set Up Development Environment**
   ```bash
   # Create and activate virtual environment
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   
   # Install development dependencies
   cd RewardModeling/python
   pip install -e ".[dev,all]"
   
   # Install pre-commit hooks
   pre-commit install
   ```

3. **Verify Installation**
   ```bash
   # Run tests to ensure everything works
   make test
   
   # Check code formatting
   make lint
   ```

## Contributing Process

### 1. Choose an Issue
- Browse our [issues](https://github.com/llamasearchai/OpenRewards/issues)
- Look for issues labeled `good first issue` for beginners
- Comment on the issue to let others know you're working on it

### 2. Create a Branch
```bash
git checkout -b feature/your-feature-name
# or
git checkout -b fix/your-bug-fix
```

### 3. Make Changes
- Follow our [code standards](#code-standards)
- Write or update tests as needed
- Update documentation if necessary

### 4. Test Your Changes
```bash
# Run full test suite
make test

# Run specific tests
pytest tests/test_specific_module.py

# Check code coverage
make test-coverage
```

### 5. Submit Pull Request
- Push your branch to your fork
- Create a pull request with a clear description
- Reference any related issues

## Code Standards

### Python Code Style
We use strict code formatting and linting:

```bash
# Format code
black reward_modeling tests
isort reward_modeling tests

# Check linting
flake8 reward_modeling tests
mypy reward_modeling
```

**Key Standards:**
- Follow PEP 8 style guide
- Use type hints for all function signatures
- Maximum line length: 88 characters (Black default)
- Use descriptive variable and function names
- Add docstrings for all public functions and classes

### TypeScript/JavaScript
- Use ESLint and Prettier for formatting
- Follow React best practices
- Use TypeScript for type safety

### Rust
- Use `rustfmt` for formatting
- Use `clippy` for linting
- Follow Rust naming conventions

### Documentation Style
- Use clear, concise language
- Include code examples where helpful
- Update API documentation for any interface changes
- Use proper markdown formatting

## Testing

### Writing Tests
- Write unit tests for new functionality
- Include integration tests for complex features
- Test edge cases and error conditions
- Aim for high test coverage (>90%)

### Test Categories
```bash
# Unit tests
pytest tests/unit/

# Integration tests
pytest tests/integration/

# Fast tests only (for development)
pytest -m "not slow"

# All tests including slow ones
pytest tests/
```

### Test Structure
```python
def test_function_name():
    """Test description explaining what is being tested."""
    # Arrange
    setup_data = create_test_data()
    
    # Act
    result = function_to_test(setup_data)
    
    # Assert
    assert result.is_valid()
    assert result.value == expected_value
```

## Documentation

### Types of Documentation
1. **Code Documentation**: Docstrings and inline comments
2. **API Documentation**: Auto-generated from code
3. **User Guides**: Step-by-step tutorials
4. **Developer Documentation**: Architecture and design decisions

### Documentation Standards
- Use Google-style docstrings for Python
- Include examples in docstrings
- Keep README files up to date
- Document breaking changes

### Building Documentation
```bash
# Build documentation locally
cd docs
make html

# Serve documentation
make docs-serve
```

## Submitting Changes

### Pull Request Guidelines
1. **Clear Title**: Use descriptive title summarizing the change
2. **Detailed Description**: Explain what and why, not just how
3. **Reference Issues**: Link to related issues using `Fixes #123`
4. **Small Changes**: Keep PRs focused and reasonably sized
5. **Tests Included**: Ensure new features have tests
6. **Documentation Updated**: Update docs for user-facing changes

### Pull Request Template
```markdown
## Description
Brief description of changes

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Documentation update
- [ ] Performance improvement
- [ ] Other (please describe)

## Testing
- [ ] Tests pass locally
- [ ] New tests added for functionality
- [ ] Integration tests updated

## Documentation
- [ ] Code documentation updated
- [ ] User documentation updated
- [ ] API documentation updated

## Checklist
- [ ] Code follows style guidelines
- [ ] Self-review completed
- [ ] No breaking changes (or clearly documented)
```

### Review Process
1. **Automated Checks**: CI/CD pipeline runs tests and linting
2. **Code Review**: Maintainers review code for quality and correctness
3. **Discussion**: Address feedback and make requested changes
4. **Approval**: Maintainer approves and merges the PR

## Development Workflows

### Common Tasks
```bash
# Start development environment
make dev-setup

# Run development servers
make run-api        # Start FastAPI server
make run-frontend   # Start React development server

# Code quality checks
make lint          # Run all linting
make format        # Format all code
make test          # Run test suite

# Build and deployment
make build         # Build all components
make docker-build  # Build Docker images
```

### Debugging
- Use logging instead of print statements
- Set up proper IDE debugging configuration
- Use pytest fixtures for test data
- Leverage Docker for consistent environments

## Getting Help

### Resources
- **Documentation**: Check our [docs](docs/) directory
- **Issues**: Search existing [GitHub issues](https://github.com/llamasearchai/OpenRewards/issues)
- **Discussions**: Use [GitHub Discussions](https://github.com/llamasearchai/OpenRewards/discussions) for questions
- **Discord**: Join our community Discord server

### Asking Questions
When asking for help:
1. Search existing issues and discussions first
2. Provide minimal reproducible example
3. Include relevant system information
4. Describe expected vs actual behavior
5. Share error messages and logs

## Recognition

Contributors are recognized in several ways:
- Listed in our contributors list
- Mentioned in release notes for significant contributions
- Invited to join our contributor Discord channel
- Opportunity to become a maintainer for consistent contributors

## License

By contributing to OpenRewards, you agree that your contributions will be licensed under the MIT License.

---

Thank you for contributing to OpenRewards! Your contributions help advance AI alignment research and make the platform better for everyone. 