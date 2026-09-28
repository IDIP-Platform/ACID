"""Load packaged defaults and one experiment override, without changing data."""

import math
import re
from importlib.resources import files
from pathlib import Path

from omegaconf import DictConfig, OmegaConf


def _root(project_root):
    if project_root is not None:
        return Path(project_root).expanduser().resolve()
    for candidate in (Path.cwd(), *Path.cwd().parents):
        if (candidate / "pyproject.toml").is_file():
            return candidate
    raise ValueError("Cannot find project root; pass project_root explicitly.")


def _check_types(value, reference, path=""):
    """Check fixed default types; null defaults and kwargs remain flexible."""
    if path.endswith(".axes") and (
        value is None
        or type(value) is int
        or (isinstance(value, list) and all(type(v) is int for v in value))
    ):
        return
    if isinstance(reference, dict) and isinstance(value, dict):
        for key, item in value.items():
            if key in reference:
                _check_types(item, reference[key], f"{path}.{key}".strip("."))
    elif reference is not None and value is not None:
        expected = type(reference)
        valid = isinstance(value, expected)
        if expected in (int, float):
            valid = isinstance(value, (int, float)) and not isinstance(value, bool)
        if not valid:
            raise ValueError(
                f"{path}: expected {expected.__name__}, got {type(value).__name__}"
            )
    elif value is None and reference is not None:
        raise ValueError(f"{path}: null is not supported for this setting")


def _validate(config):
    if not isinstance(config.experiment.id, str) or not re.fullmatch(
        r"[A-Za-z0-9][A-Za-z0-9_-]*", config.experiment.id
    ):
        raise ValueError(
            "experiment.id must contain only letters, digits, underscores or hyphens"
        )
    for key in ("title", "description", "author", "notes"):
        if not isinstance(config.experiment[key], str):
            raise TypeError(f"experiment.{key} must be a string")
    if config.experiment.created is not None:
        from datetime import date

        date.fromisoformat(str(config.experiment.created))
    proc = config.object_segmentation.processing
    for key in ("diameter", "downsampling_factor"):
        value = proc[key]
        if (
            not isinstance(value, (float, int))
            or isinstance(value, bool)
            or not math.isfinite(value)
            or value <= 0
        ):
            raise ValueError(f"object_segmentation.processing.{key} must be positive")
    for key in ("nucleus_position", "concanavalin_position", "actin_position"):
        if type(proc[key]) is not int or proc[key] < 0:
            raise ValueError(
                f"object_segmentation.processing.{key} must be a nonnegative integer"
            )
    if config.background_correction.processing.method not in (
        "division",
        "subtraction",
    ):
        raise ValueError(
            "background_correction.processing.method must be division or subtraction"
        )


def _normalize_paths(node, root):
    for key in node:
        value = node[key]
        if isinstance(value, DictConfig):
            _normalize_paths(value, root)
        elif key.endswith(("_dir", "_directory")) or key in (
            "directory",
            "output_root",
        ):
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{key} must be a nonempty path string")
            node[key] = str((root / Path(value).expanduser()).resolve())


def load_config(config_file=None, *, project_root=None, preview=True):
    """Merge one override over defaults. Relative paths use the project root.

    Loading does not create outputs or require inputs to exist. Stage readers
    check input existence when processing. Validation covers keys, fixed default
    types and selected stage constraints, not every scientific parameter.
    """
    root = _root(project_root)
    default = OmegaConf.create(
        files("acid.config").joinpath("default.yaml").read_text()
    )
    baseline = OmegaConf.to_container(default, resolve=True)
    OmegaConf.set_struct(default, True)
    override = OmegaConf.create({})
    if config_file is not None:
        path = root / Path(config_file).expanduser()
        if not path.is_file():
            raise FileNotFoundError(f"Configuration file not found: {path}")
        override = OmegaConf.load(path)
        if not isinstance(override, DictConfig):
            raise ValueError("Configuration must be a YAML mapping")
        if OmegaConf.select(override, "experiment.id") != path.stem:
            raise ValueError(
                f"experiment.id must match configuration filename: {path.stem}"
            )
    config = OmegaConf.merge(default, override)
    OmegaConf.resolve(config)
    _check_types(OmegaConf.to_container(config), baseline)
    _validate(config)
    _normalize_paths(config, root)
    if preview:
        print(f"Experiment: {config.experiment.id} — {config.experiment.title}")
        print("Overrides:\n" + OmegaConf.to_yaml(override))
        print("Resolved directories:\n" + OmegaConf.to_yaml(config.shared.paths))
    OmegaConf.set_readonly(config, True)
    return config


def prepare_output(config):
    """Create a missing settings record, or check an existing one.

    A recreated record describes the current configuration; it cannot establish
    which settings produced files already present in the output folder.
    """
    output = Path(config.shared.output_root)
    snapshot = output / "config.resolved.yaml"
    text = OmegaConf.to_yaml(config, resolve=True, sort_keys=True)
    if snapshot.exists():
        previous = OmegaConf.to_yaml(
            OmegaConf.load(snapshot), resolve=True, sort_keys=True
        )
        if previous != text:
            raise ValueError(
                f"Configuration differs from {snapshot}. Choose a new experiment ID "
                "or output root to retain separate results."
            )
        return snapshot
    output.mkdir(parents=True, exist_ok=True)
    with snapshot.open("x") as stream:
        stream.write(text)
    return snapshot
