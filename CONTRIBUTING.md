# Contributing to Heart Disease CNN Prediction

Thank you for your interest in contributing! Here are some guidelines:

## Getting Started

1. **Fork the repository** on GitHub
2. **Clone your fork locally**:
   ```bash
   git clone https://github.com/yourusername/heart_disease_cnn.git
   cd heart_disease_cnn
   ```

3. **Create a virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

4. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   pip install -e ".[dev]"
   ```

## Development Workflow

1. **Create a branch** for your feature:
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make your changes** and test thoroughly

3. **Run tests**:
   ```bash
   pytest tests/
   ```

4. **Commit with clear messages**:
   ```bash
   git commit -m "feat: describe your change"
   ```

5. **Push to your fork** and **create a Pull Request**

## Code Standards

- Follow PEP 8 style guidelines
- Add docstrings to functions and classes
- Update README if adding new features
- Ensure code passes through `flake8`

## Reporting Issues

When reporting bugs, please include:
- Python version
- Operating system
- Steps to reproduce
- Expected vs actual behavior
- Full error traceback

## License

By contributing, you agree that your contributions will be licensed under the MIT License.
