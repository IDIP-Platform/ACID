# Contributing to ACID

Thank you for contributing to ACID. This repository contains Python tools and
notebook workflows for microscopy image processing, quality control,
segmentation, and feature extraction.

For the full contributor guide, see
[docs/community/contribute.md](docs/community/contribute.md).

## Before You Start

For larger changes, open or comment on a GitHub issue first. Use the available
issue templates for bug reports, feature requests, and documentation
improvements.

Keep pull requests focused on one clear topic, such as a bug fix, workflow
change, notebook update, dependency cleanup, or documentation improvement.

## Development Setup

ACID uses Python `>=3.11,<3.13` and `uv`.

```bash
git clone https://github.com/IDIP-Platform/ACID.git
cd ACID
uv venv
source .venv/bin/activate
uv sync
uv pip install -e .
```

On Windows PowerShell, activate the environment with:

```powershell
.venv\Scripts\Activate.ps1
```

For documentation work:

```bash
uv sync --extra docs
uv run zensical serve
```

## Validation Expectations

Functionality changes should be validated before opening a pull request.
Automated tests are encouraged, but manual validation is also important for
image-processing workflows.

In your pull request, describe:

- commands you ran
- data used for validation, if relevant
- manual validation steps
- expected result
- screenshots, plots, output examples, or notebook cells when useful

If private or local microscopy data was used, describe it without exposing
sensitive information.

## Dependencies

Keep dependencies scoped to where they are needed:

- core package dependencies in `[project].dependencies`
- documentation tools in the optional `docs` group
- development-only tools in an optional development group
- notebook, visualization, or segmentation-only tools as optional dependencies
  unless they are required by package imports at runtime

## Pull Requests

Please fill in the pull request template. Use `N/A` for sections that do not
apply.

Before submitting, check that:

- the change is focused and reviewable
- tests or validation are described
- documentation or notebooks are updated when needed
- dependency changes are justified
- related issues are linked

## Code of Conduct

All contributors are expected to follow the project
[Code of Conduct](CODE_OF_CONDUCT.md).
