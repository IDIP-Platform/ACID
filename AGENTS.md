# ACID repository guidance

ACID is a Python pipeline for microscopy image processing, quality control,
segmentation, and cellular feature extraction in a dengue study. The current
workflow focuses on fixed samples. Future acquisitions may add a time (`T`)
axis: read named image axes and preserve their metadata; never silently treat
time points as fields of view or channels. If an acquisition has multiple time
points that a workflow does not support, reject it with a clear error before
writing derived results.

For each change, report what changed, the verification commands and results,
and any checks skipped or blocked. Never claim a check passed unless it ran.

## Testing

- Use `pytest` exclusively. Do not import `unittest`, use `unittest.TestCase` or
  `unittest.mock`, or run `python -m unittest`. Use pytest fixtures and
  `monkeypatch` for test setup and substitution.
- Add or update focused tests when behavior changes. Test observable results and
  relevant edge cases, especially image axes, channel identity, metadata,
  segmentation labels, and numerical outputs. Use small synthetic arrays and
  `@pytest.mark.parametrize` where useful; avoid tests that merely repeat the
  implementation.
- Keep automated tests independent of private or ignored ND2 data, network
  access, and GPU availability. Use `tmp_path` for generated files and do not
  modify files under `data/raw`.
- Run affected tests first with `uv run --locked pytest <test_path>`. For code
  changes, then run `uv run --locked pytest tests`; if it cannot run, report why
  and which tests did run. Run Snakemake workflow tests with
  `uv run --locked --group pipeline pytest tests/integration/workflow` so they
  do not silently skip for lack of Snakemake. Report skipped tests.
- When a change can alter scientific measurements or segmentation, validate
  representative real images when available and document the data, expected
  behavior, observed result, and any manual image review.

## Packaging/Dependencies Management

- Treat `pyproject.toml` as the source of truth for package metadata and direct
  dependencies, and keep matching `uv.lock` changes in the same change set.
  Preserve the supported Python range (`>=3.11,<3.13`) unless changing support
  deliberately.
- Use `uv add` and `uv remove` to change dependencies. Place a dependency
  according to how it is used: installed package runtime requirements in
  `[project].dependencies`, optional installed features in extras, and tools
  used only for development, pipelines, or notebooks in their existing groups.
  Keep documentation build dependencies available through the `docs` extra
  used by `.github/workflows/docs.yml`. If consolidating the duplicate `docs`
  group, update its callers and `uv.lock` in the same change.
- Use `uv sync --locked` for reproducible setup and `uv run --locked` for
  project commands. Do not use ad hoc `pip` or `uv pip` installs to record a
  project dependency, and do not edit `uv.lock` by hand. Check lock consistency
  with `uv lock --check` after dependency changes.
- For changes to packaging or build metadata, run `uv build` and inspect the
  resulting wheel and source distribution. Do not include raw microscopy data,
  local outputs, or secrets in distributions.

## Linting/Formatting

- For new or edited Markdown prose, keep each paragraph on one source line and use editor soft wrap for reading. Preserve intentional line breaks in lists, tables, and code blocks; do not hard-wrap prose at 80 characters.
- Use Ruff as the Python linter and formatter. Check changed Python files with
  `uv run --locked ruff check <paths>` and
  `uv run --locked ruff format --check <paths>`.
- Apply `uv run --locked ruff format <paths>` only to files in the current
  change. Do not reformat unrelated modules as part of a functional change.
- Fix reported issues instead of adding broad ignores. If a suppression is
  necessary, scope it narrowly and explain why. Follow the repository's
  Google-style docstring convention for public APIs.

## Pre-commit

- This repository already has `.pre-commit-config.yaml`. Its active hooks
  check common file issues, format Python with Ruff, and strip notebook output.
  The `ruff-check` hook is currently disabled, so run Ruff lint explicitly as
  described above.
- Before committing, run `uv run --locked pre-commit run --files <changed_files>`
  and inspect any changes made by hooks. Use `--all-files` only for a deliberate
  repository-wide check or cleanup; it may rewrite unrelated files.
- Hook installation with `uv run --locked pre-commit install` is a local
  contributor setup step. If hook environments cannot be installed or run,
  report the limitation instead of claiming the checks passed.
