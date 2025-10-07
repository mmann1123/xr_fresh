# Contributing to xr_fresh

Thank you for your interest in contributing to xr_fresh! We welcome contributions from the community.

## Ways to Contribute

- **Bug Reports**: Open an issue describing the bug, including steps to reproduce
- **Feature Requests**: Suggest new features or enhancements via issues
- **Code Contributions**: Submit pull requests with bug fixes or new features
- **Documentation**: Improve or expand documentation
- **Examples**: Share notebooks or examples of using xr_fresh

## Getting Started

### Development Setup

1. Fork the repository on GitHub
2. Clone your fork locally:
   ```bash
   git clone https://github.com/mmann1123/xr_fresh
   cd xr_fresh
   ```

3. Create a development environment:
   ```bash
   conda create -n xr_fresh_dev geowombat -c conda-forge
   conda activate xr_fresh_dev
   ```

4. Install xr_fresh in development mode:
   ```bash
   pip install -e .
   ```

5. If the C++ extension fails to build:
   ```bash
   python setup.py build_ext --inplace
   ```

### Running Tests

Run the test suite to ensure everything works:
```bash
python -m unittest discover -s tests -p 'test_*.py'
```

## Contribution Workflow

1. **Create a branch** for your work:
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make your changes**:
   - Write clear, documented code
   - Follow existing code style and conventions
   - Add tests for new functionality
   - Update documentation as needed

3. **Test your changes**:
   ```bash
   python -m unittest discover -s tests -p 'test_*.py'
   ```

4. **Commit your changes**:
   ```bash
   git add .
   git commit -m "Brief description of your changes"
   ```

5. **Push to your fork**:
   ```bash
   git push origin feature/your-feature-name
   ```

6. **Open a Pull Request** on GitHub with:
   - Clear description of changes
   - Reference to related issues (if applicable)
   - Screenshots or examples (if relevant)

## Code Guidelines

### Python Style
- Follow PEP 8 style guidelines
- Use meaningful variable and function names
- Add docstrings to functions and classes
- Keep functions focused and modular

### Feature Functions
When adding new feature calculation functions:
- Inherit from `gw.TimeModule`
- Use JAX arrays (`jnp`) for GPU compatibility when possible
- Avoid Python control flow (if/for loops) in JAX-traced code
- Include comprehensive docstrings with mathematical formulas
- Add examples in the docstring

Example:
```python
class new_feature(gw.TimeModule):
    """
    Brief description of the feature.

    .. math::

        Feature = \\sum_{i=1}^{n} x_i

    Args:
        parameter (type): Description of parameter

    Returns:
        result (numpy.ndarray): Description of output

    Example:
        >>> with gw.series(files) as src:
        >>>     src.apply(new_feature(parameter=value), bands=1, outfile="output.tif")
    """

    def __init__(self, parameter=default):
        super(new_feature, self).__init__()
        self.parameter = parameter

    def calculate(self, array):
        # Implementation using jnp operations
        return jnp.some_operation(array, axis=0).squeeze()
```

### Testing
- Add tests for new features in the `tests/` directory
- Ensure tests pass before submitting PR
- Test with different data types and edge cases
- Include tests for JAX compatibility if applicable

### Documentation
- Update relevant documentation files
- Add examples to notebooks if appropriate
- Keep CLAUDE.md updated with architectural changes
- Update README.md if adding major features

## JAX Compatibility Notes

Some features may not be compatible with JAX tracing due to Python control flow. If adding features with loops or conditionals:
- Document the limitation in the docstring
- Consider NumPy-based alternatives
- Update documentation to note JAX incompatibility

Features with known JAX issues:
- `longest_strike_above_mean`
- `longest_strike_below_mean`

## Reporting Issues

When reporting bugs, include:
- Python version and environment details
- xr_fresh version
- Steps to reproduce the issue
- Expected vs. actual behavior
- Error messages and stack traces
- Sample data or minimal reproducible example

## Questions?

- Open a GitHub issue for questions
- Check existing issues and documentation first
- Be respectful and constructive in discussions

## Code of Conduct

This project follows the [Contributor Covenant Code of Conduct](CODE_OF_CONDUCT.md). By participating, you agree to uphold this code.

## License

By contributing to xr_fresh, you agree that your contributions will be licensed under the MIT License.

---

Thank you for contributing to xr_fresh!
