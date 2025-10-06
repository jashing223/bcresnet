## Agent Instructions

This document outlines the conventions and commands for working with this codebase.

### Build and Test

- **Dependencies**: Install dependencies with `pip install -r requirements.txt`.
- **Training**: Run `python main.py` to train the model.
- **Evaluation**: Run `python main.py --eval --ckpt <path_to_checkpoint>` to evaluate a model.
- **Testing**: There is no formal test suite. When making changes, manually verify them by running the training and evaluation scripts.

### Code Style

- **Formatting**: Adhere to PEP 8. Use a linter like `flake8` or `black` if you have one configured.
- **Imports**: Group imports as follows: standard library, third-party libraries, and then local application imports.
- **Naming**: Use `snake_case` for variables and functions, and `PascalCase` for classes.
- **Docstrings**: Use Google-style docstrings for all modules, classes, and functions.
- **Error Handling**: Use specific exception types where possible. Avoid broad `except` clauses.
- **Typing**: This project does not use type hints.
