# Dependencies

ACID uses `pyproject.toml` and `uv.lock` for dependency management.

## Source of Truth

Use these files as the current dependency source of truth:

- `pyproject.toml`: declared package dependencies and optional dependency groups
- `uv.lock`: resolved dependency versions

Install the core environment with:

```bash
uv sync
```

Install documentation dependencies with:

```bash
uv sync --extra docs
```

Install every optional dependency group with:

```bash
uv sync --all-extras
```

## Optional Groups

The current documentation tooling belongs to the `docs` optional dependency
group. This group includes tools such as Zensical and mkdocstrings support.

Documentation-only dependencies should stay out of the core package dependency
list unless package code imports them at runtime.

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
- documentation tooling: `docs` optional group
- test and lint tools: development optional group
- notebook-only or visualization-only tools: optional group unless imported by
  package code at runtime

This keeps the core ACID installation smaller and makes optional workflows
clearer.
