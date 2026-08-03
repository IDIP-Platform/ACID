# Installation

ACID is a Python package for microscopy image processing, quality control,
segmentation, and feature extraction workflows.

## Requirements

- Python `>=3.11,<3.13`
- [`uv`](https://docs.astral.sh/uv/) for environment and dependency management

Check your Python version:

```bash
python --version
```

## Clone the Repository

```bash
git clone https://github.com/IDIP-Platform/ACID.git
cd ACID
```

## One-Time Setup

Run these steps once after cloning the repository.

## Create the Environment

Create a local virtual environment:

```bash
uv venv
```

Activate it:

=== "Linux/macOS"

    ```bash
    source .venv/bin/activate
    ```

=== "Windows PowerShell"

    ```powershell
    .venv\Scripts\Activate.ps1
    ```

## Install ACID

Install the package and its core dependencies:

```bash
uv sync
```

For editable development, install the local package in editable mode:

```bash
uv pip install -e .
```

## VS Code Setup

When working in VS Code:

1. Open the ACID repository folder.
2. Open a terminal with **Terminal > New Terminal**.
3. Create and activate the virtual environment.
4. Run `uv sync`.
5. Run `uv pip install -e .` for editable package development.
6. Select the interpreter with **Python: Select Interpreter** and choose the
   `.venv` Python executable.

## Install Documentation Dependencies

The documentation tools are kept in the optional `docs` dependency group.

```bash
uv sync --extra docs
```

To install all optional dependency groups:

```bash
uv sync --all-extras
```

## Run the Documentation Site Locally

After installing the `docs` group, serve the documentation site with:

```bash
uv run zensical serve
```

## Use ACID in Notebooks

If you want to use the environment from Jupyter or VS Code notebooks, install an
IPython kernel:

```bash
uv pip install ipykernel
python -m ipykernel install --user --name acid-env --display-name "ACID"
```

Then select the `ACID` kernel in your notebook editor.

In VS Code notebooks, use the kernel selector in the top-right corner and choose
the `ACID` kernel.

## Daily Restart

When returning to the project:

=== "Linux/macOS"

    ```bash
    cd path/to/ACID
    source .venv/bin/activate
    uv sync
    uv pip install -e .
    ```

=== "Windows PowerShell"

    ```powershell
    cd path\to\ACID
    .venv\Scripts\Activate.ps1
    uv sync
    uv pip install -e .
    ```

Run `uv sync` whenever `pyproject.toml` or `uv.lock` changes.

## Update Dependencies

When `pyproject.toml` or `uv.lock` changes, synchronize the environment again:

```bash
uv sync
```

For documentation work:

```bash
uv sync --extra docs
```
