# Contributing to DeafBench

Thank you for your interest in contributing to DeafBench!

## Development Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/488315/DeafBench.git
   cd DeafBench
   ```

2. Create a virtual environment & install in editable mode with test dependencies:
   ```bash
   python -m venv .venv
   # Windows:
   .venv\Scripts\Activate.ps1
   # Linux/macOS:
   source .venv/bin/activate

   pip install -e ".[test]"
   ```

3. Run tests:
   ```bash
   pytest
   ```

## Code Guidelines

- Keep initial scope focused on accessibility-specific evaluation metrics.
- Maintain test coverage for metrics parsing and evaluation logic.
- Follow PEP 8 and Python formatting conventions.
