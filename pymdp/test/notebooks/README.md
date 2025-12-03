# Developer Guide: Writing Testable Notebooks

This guide explains how to write Jupyter notebooks that work well with the automated testing infrastructure.

## Overview

All notebooks in the `examples/` directory are automatically tested using `nbval`, a pytest plugin that executes notebooks and validates their outputs.

## Tips for writing testable (and useful!) notebooks
1. Keep your notebooks small and tightly defined
2. Avoid printing massive outputs

## Running Notebook Tests Locally

### Basic Commands

:warning: `nbval` does not play nicely with `pytest-xdist` running tests in parallel. `nbval` tests must be called with a single worker (-n 1).

```bash
# Test all non-legacy notebooks (execution only, no output validation, will only fail if an exception is raised)
uv run pytest --nbval-lax examples/

# Test specific directory
uv run pytest --nbval-lax examples/api/

# Test with full output cell validation
uv run pytest --nbval examples/api/

# Test specific notebook
uv run pytest --nbval examples/api/model_construction_tutorial.ipynb -v
```

:warning: For now, notebook tests are opt-in, i.e. running `uv run pytest test` will *not* check notebooks.
This tool is currently intended to help test notebooks locally, and eventually we will move them into CI if we can keep execution times low.

The legacy notebooks are not tested.