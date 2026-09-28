# Quickstart

This guide walks through the one-time setup for running ACID locally.

## Prerequisites

- Python `>=3.11,<3.13`
- `uv` installed and available from your terminal
- [VS Code](https://code.visualstudio.com/) if you plan to work from the editor

## Open the Project

Open a terminal and navigate to the project root:

=== "Linux"

    ```bash
    cd path/to/ACID
    ```

=== "macOS"

    ```bash
    cd path/to/ACID
    ```

=== "Windows PowerShell"

    ```powershell
    cd path\to\ACID
    ```

## Create the Environment

Create a local virtual environment:

=== "Linux"

    ```bash
    uv venv
    ```

=== "macOS"

    ```bash
    uv venv
    ```

=== "Windows PowerShell"

    ```powershell
    uv venv
    ```

Activate the virtual environment:

=== "Linux"

    ```bash
    source .venv/bin/activate
    ```

=== "macOS"

    ```bash
    source .venv/bin/activate
    ```

=== "Windows PowerShell"

    ```powershell
    .venv\Scripts\Activate.ps1
    ```

## Install ACID

Install the project in editable mode:

=== "Linux"

    ```bash
    uv pip install -e .
    ```

=== "macOS"

    ```bash
    uv pip install -e .
    ```

=== "Windows PowerShell"

    ```powershell
    uv pip install -e .
    ```

## Add a Notebook Kernel

Install Jupyter kernel support:

=== "Linux"

    ```bash
    uv pip install ipykernel
    python -m ipykernel install --user --name acid-env --display-name "ACID"
    ```

=== "macOS"

    ```bash
    uv pip install ipykernel
    python -m ipykernel install --user --name acid-env --display-name "ACID"
    ```

=== "Windows PowerShell"

    ```powershell
    uv pip install ipykernel
    python -m ipykernel install --user --name acid-env --display-name "ACID"
    ```

## Set Up VS Code

1. Open the ACID repository folder in VS Code.
2. Open a terminal with **Terminal > New Terminal**.
3. Create and activate the virtual environment with the commands for your
   operating system.
4. Install ACID in editable mode.
5. Select the project interpreter:
   - Press **Ctrl + Shift + P**.
   - Run **Python: Select Interpreter**.

=== "Linux"

    Choose `.venv/bin/python`.

=== "macOS"

    Choose `.venv/bin/python`.

=== "Windows PowerShell"

    Choose `.venv\Scripts\python.exe`.

For notebooks, open the kernel selector in the top-right corner and choose the
`ACID` kernel.

## Verify the Environment

After activation, confirm Python is available from the virtual environment:

=== "Linux"

    ```bash
    python --version
    ```

=== "macOS"

    ```bash
    python --version
    ```

=== "Windows PowerShell"

    ```powershell
    python --version
    ```

You are ready to run scripts, notebooks, and project commands from the activated
environment.
