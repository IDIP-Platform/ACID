# Dependencies

ACID uses `pyproject.toml` and `uv.lock` for dependency management.

## Source of Truth

Use these files as the current dependency source of truth:

- `pyproject.toml`: declared package dependencies, project optional dependencies and dependency groups
- `uv.lock`: resolved dependency versions

Install the core environment with:

```bash
uv sync
```

Install documentation dependencies with:

```bash
uv sync --locked --group docs
```

Install all project optional dependencies with:

```bash
uv sync --locked --all-extras
```

## Project Optional Dependencies and Dependency Groups

Project optional dependencies are installable choices for users of ACID:

- `notebooks`: JupyterLab, the Jupyter kernel, widgets and `nbstripout` for the repository notebooks; install with `uv sync --locked --extra notebooks`.
- `pipeline`: Snakemake and its SLURM executor plugin for the repository workflow; install with `uv sync --locked --extra pipeline`.

These optional dependencies provide the software environment; the notebook and Snakemake workflow files are in this repository, not in the installed wheel.

Dependency groups are for contributors and documentation builders:

- `dev`: tests, linting and pre-commit; installed by `uv sync` by default.
- `docs`: Zensical and mkdocstrings support; install with `uv sync --locked --group docs`.

## Legacy Environment Files

The README mentions historical environment files such as `acid_develop.yml`,
`acid_neo.yml`, and `20260202_acid_develop.txt`.

These files are useful as development history or workstation references, but the
preferred installation path for the repository is now `uv`.

!!! warning "`acid_neo` status"
    The README states that the `acid_neo` environment had not been tested as of
    2026-02-02. Do not treat it as a validated installation path without
    retesting.

## Dependency Placement

When adding dependencies, keep them scoped to where they are needed:

- runtime imports used by `src/acid`: core dependencies
- documentation tooling: `docs` dependency group
- test and lint tools: `dev` dependency group
- Jupyter tools used to run the repository notebooks: `notebooks` project optional dependencies
- Snakemake and its executor plugin: `pipeline` project optional dependencies

This keeps the core ACID installation smaller and makes optional workflows
clearer.
