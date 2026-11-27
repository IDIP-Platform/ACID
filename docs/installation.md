# Installation

Install ACID with `uv` from a local checkout.

## Requirements

- Python `>=3.11,<3.13`
- [`uv`](https://docs.astral.sh/uv/)

## Clone the Repository

=== "Linux"

    ```bash
    git clone https://github.com/IDIP-Platform/ACID.git
    cd ACID
    ```

=== "macOS"

    ```bash
    git clone https://github.com/IDIP-Platform/ACID.git
    cd ACID
    ```

=== "Windows PowerShell"

    ```powershell
    git clone https://github.com/IDIP-Platform/ACID.git
    cd ACID
    ```

## Install

=== "Linux"

    ```bash
    uv venv
    source .venv/bin/activate
    uv sync
    uv pip install -e .
    ```

=== "macOS"

    ```bash
    uv venv
    source .venv/bin/activate
    uv sync
    uv pip install -e .
    ```

=== "Windows PowerShell"

    ```powershell
    uv venv
    .venv\Scripts\Activate.ps1
    uv sync
    uv pip install -e .
    ```

To run the whole pipeline with Snakemake, install the `pipeline` project optional dependencies (Snakemake and its SLURM executor plugin):

```bash
uv sync --locked --extra pipeline
```

See [Snakemake pipeline](workflows/snakemake.md).

## VS Code

Open the repository folder, then select the project interpreter:

=== "Linux"

    `.venv/bin/python`

=== "macOS"

    `.venv/bin/python`

=== "Windows PowerShell"

    `.venv\Scripts\python.exe`

## Notebooks

Install JupyterLab, widgets and a Jupyter kernel from the `notebooks` project optional dependencies:

=== "Linux"

    ```bash
    uv sync --locked --extra notebooks
    python -m ipykernel install --user --name acid-env --display-name "ACID"
    ```

=== "macOS"

    ```bash
    uv sync --locked --extra notebooks
    python -m ipykernel install --user --name acid-env --display-name "ACID"
    ```

=== "Windows PowerShell"

    ```powershell
    uv sync --locked --extra notebooks
    python -m ipykernel install --user --name acid-env --display-name "ACID"
    ```

Select the `ACID` kernel in VS Code or Jupyter.

## Documentation

=== "Linux"

    ```bash
    uv sync --locked --group docs
    uv run --locked --group docs zensical serve
    ```

=== "macOS"

    ```bash
    uv sync --locked --group docs
    uv run --locked --group docs zensical serve
    ```

=== "Windows PowerShell"

    ```powershell
    uv sync --locked --group docs
    uv run --locked --group docs zensical serve
    ```

Run `uv sync` again whenever `pyproject.toml` or `uv.lock` changes.
