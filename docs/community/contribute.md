# Contributing

Thank you for contributing to ACID. This project develops Python tools and
notebook workflows for microscopy image processing, quality control,
segmentation, and feature extraction.

## Ways to Contribute

You can contribute by:

- reporting bugs or edge cases in the image-processing workflows
- proposing new features or pipeline improvements
- improving notebooks, documentation, or examples
- adding or improving tests
- cleaning package dependencies or development tooling
- improving code readability, typing, or maintainability

## Before You Start

For larger changes, please open or comment on a GitHub issue first. This helps
align the proposed work with the current project priorities and avoids
duplicated effort.

Use the repository issue templates when possible:

- bug reports for broken behavior
- feature requests for new capabilities or workflow changes
- documentation improvements for unclear or missing documentation

## Development Setup

ACID uses Python `>=3.11,<3.13` and `uv` for dependency management.

Clone the repository:

```bash
git clone https://github.com/IDIP-Platform/ACID.git
cd ACID
```

Create and activate a virtual environment:

```bash
uv venv
source .venv/bin/activate
```

On Windows PowerShell:

```powershell
uv venv
.venv\Scripts\Activate.ps1
```

Install dependencies:

```bash
uv sync
```

Install the package in editable mode:

```bash
uv pip install -e .
```

For documentation work, install the documentation dependencies:

```bash
uv sync --extra docs
```

## Branches and Commits

Create a branch for your change:

```bash
git checkout -b short-description-of-change
```

Keep pull requests focused. A PR should usually address one issue, workflow
step, documentation area, or dependency cleanup topic.

## Code Changes

When changing package code, prefer small, explicit changes that follow the
existing module structure under `src/acid`.

If your change affects a scientific workflow, describe:

- the affected ACID step
- the expected input data or metadata
- the expected output
- any assumptions about image dimensions, channels, labels, or file formats
- how the result was validated

## Code Style and Docstrings

Public functions, classes, and modules should use Google-style docstrings.
The API reference is generated from docstrings with `mkdocstrings`, so write
docstrings for users of the documentation as well as maintainers of the code.

See the [Docstring Guide](docstrings.md) for ACID conventions and examples.

## Notebook Changes

Notebook updates should be kept intentional and reviewable.

Before opening a pull request:

- clear unrelated exploratory output when possible
- keep notebook changes focused on the workflow being modified
- mention which notebook cells or workflow steps were checked
- include screenshots, plots, or output examples when they help reviewers

## Documentation Changes

Documentation is built with Zensical.

When editing documentation, follow the [Markdown Guide](markdown.md) for common
Markdown patterns used in this site.

Run the docs locally with:

```bash
uv run zensical serve
```

Build the docs with:

```bash
uv run zensical build --clean
```

For documentation-only PRs, runtime data validation is usually not required.
Instead, explain what was reviewed and what should be clearer after the change.

## Dependencies

Keep dependencies scoped to where they are needed.

- Core package dependencies belong in `[project].dependencies`.
- Documentation tools belong in the optional `docs` group.
- Development-only tools should go in an optional development group.
- Notebook, visualization, or segmentation-only dependencies should be optional
  unless the core package imports them at runtime.

When adding or moving a dependency, explain whether it is required by package
code, notebooks, documentation, visualization, or development tooling.

## Validation

Functionality changes require validation. Automated tests are encouraged, but
manual validation is also important for image-processing workflows.

In your pull request, include:

- commands you ran
- data used for validation, if relevant
- steps a reviewer can follow
- expected result
- screenshots, plots, or output examples when useful

If private or local microscopy data was used, describe it without exposing
sensitive information. Include file type, approximate size, relevant channels,
and metadata assumptions.

## Pull Requests

When opening a pull request, fill in the PR template. In particular, describe:

- what changed and why
- user-visible impact
- verification performed
- validation steps
- data used, if relevant
- documentation or dependency changes
- risk level and rollback notes

Use `N/A` for sections that do not apply.

## Code of Conduct

All contributors are expected to follow the project
[Code of Conduct](https://github.com/IDIP-Platform/ACID/blob/main/CODE_OF_CONDUCT.md).
