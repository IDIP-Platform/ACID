"""Tests for configuration loading and output snapshots."""

from pathlib import Path

import pytest
from omegaconf.errors import ConfigKeyError

from acid.config import load_config, loader, prepare_output

ROOT = Path(__file__).resolve().parents[3]


def override(tmp_path, text):
    """Write an experiment override in the temporary directory."""
    path = tmp_path / "exp_003.yaml"
    path.write_text("experiment:\n  id: exp_003\n" + text)
    return path


def test_defaults_and_experiment_isolation(tmp_path):
    """Keep defaults independent from experiment-specific overrides."""
    default = load_config(project_root=ROOT, preview=False)
    first = load_config("configs/exp_001.yaml", project_root=ROOT, preview=False)
    second_file = tmp_path / "exp_002.yaml"
    second_file.write_text("experiment:\n  id: exp_002\n")
    second = load_config(second_file, project_root=ROOT, preview=False)
    assert default.object_segmentation.processing.diameter == 75
    assert first.object_segmentation.processing.diameter == 100
    assert second.object_segmentation.processing.diameter == 75
    assert first.shared.paths.extracted_fov_dir != second.shared.paths.extracted_fov_dir
    assert first.shared.paths.corrected_fov_dir != second.shared.paths.corrected_fov_dir
    assert (
        first.background_correction.metadata.directory
        == first.shared.paths.metadata_dir
    )
    assert (
        first.object_segmentation.metadata.directory
        == first.background_correction.metadata.directory
    )


def test_paths_from_notebook_directory(monkeypatch):
    """Resolve project paths when loading from a notebook directory."""
    monkeypatch.chdir(ROOT / "notebooks_refactored")
    config = load_config("configs/exp_001.yaml", preview=False)
    assert config.shared.output_root == str(ROOT / "data/processed/exp_001")


@pytest.mark.parametrize(
    "text,error",
    [
        ("object_segmentation:\n  processing:\n    diamter: 80\n", ConfigKeyError),
        ("object_segmentation:\n  processing:\n    diameter: wrong\n", ValueError),
        ("object_segmentation:\n  processing:\n    diameter: -1\n", ValueError),
    ],
)
def test_bad_overrides(tmp_path, text, error):
    """Reject unknown keys and invalid processing values."""
    path = override(tmp_path, text)
    with pytest.raises(error):
        load_config(path, project_root=tmp_path, preview=False)


def test_missing_and_mismatched_file(tmp_path):
    """Reject missing files and experiment IDs that differ from filenames."""
    with pytest.raises(FileNotFoundError):
        load_config("missing.yaml", project_root=tmp_path)
    path = tmp_path / "exp_001.yaml"
    path.write_text("experiment:\n  id: exp_002\n")
    with pytest.raises(ValueError, match="filename"):
        load_config(path, project_root=tmp_path)


def test_packaged_default_must_be_a_mapping(tmp_path, monkeypatch):
    """Report a clear error when the packaged YAML has the wrong root type."""
    (tmp_path / "default.yaml").write_text("- invalid\n")
    monkeypatch.setattr(loader, "files", lambda _package: tmp_path)

    with pytest.raises(
        ValueError, match="Packaged default configuration must be a YAML mapping"
    ):
        load_config(project_root=tmp_path, preview=False)


def test_snapshot_guard(tmp_path):
    """Reject changed settings after an output snapshot exists."""
    path = override(tmp_path, "")
    config = load_config(path, project_root=tmp_path, preview=False)
    snapshot = prepare_output(config)
    assert prepare_output(config) == snapshot
    override(tmp_path, "object_segmentation:\n  processing:\n    diameter: 90\n")
    changed = load_config(path, project_root=tmp_path, preview=False)
    with pytest.raises(ValueError, match="differs"):
        prepare_output(changed)


def test_list_replacement_and_nullable_default(tmp_path):
    """Accept an axis list and a value for a nullable default."""
    path = override(
        tmp_path,
        "feature_extraction:\n  processing:\n    axes: [0]\nbackground_correction:\n  processing:\n    max_clip_value: 100\n",
    )
    config = load_config(path, project_root=tmp_path, preview=False)
    assert list(config.feature_extraction.processing.axes) == [0]
    assert config.background_correction.processing.max_clip_value == 100


@pytest.mark.parametrize(
    "axes, expected",
    [("null", None), ("-1", -1), ("[0, 2]", [0, 2])],
)
def test_valid_axis_overrides(tmp_path, axes, expected):
    """Accept null, integer, and integer-list axis settings."""
    path = override(tmp_path, f"feature_extraction:\n  processing:\n    axes: {axes}\n")
    config = load_config(path, project_root=tmp_path, preview=False)
    actual = config.feature_extraction.processing.axes
    assert (list(actual) if isinstance(expected, list) else actual) == expected


@pytest.mark.parametrize(
    "setting, message",
    [
        (
            "feature_extraction:\n  processing:\n    axes: true\n",
            "axes: expected int, got bool",
        ),
        (
            "feature_extraction:\n  processing:\n    axes: [0, true]\n",
            "axes: expected int, got list",
        ),
        (
            "shared:\n  dataset_split:\n    labels:\n      train: true\n",
            "train: expected int, got bool",
        ),
        (
            "project_identity:\n  project_name: null\n",
            "project_name: null is not supported",
        ),
    ],
)
def test_invalid_fixed_default_types(tmp_path, setting, message):
    """Reject invalid axis, fixed numeric, and null types."""
    path = override(tmp_path, setting)
    with pytest.raises(ValueError, match=message):
        load_config(path, project_root=tmp_path, preview=False)


@pytest.mark.parametrize("created", ["20260101", '"20260101"', "2026-01-01"])
def test_created_accepts_existing_date_forms(tmp_path, created):
    """Accept integer, compact string, and dashed date forms."""
    path = override(tmp_path, f"  created: {created}\n")
    config = load_config(path, project_root=tmp_path, preview=False)
    assert str(config.experiment.created).replace("-", "") == "20260101"


def test_invalid_created_names_setting(tmp_path):
    """Identify the metadata field when a date is invalid."""
    path = override(tmp_path, "  created: not-a-date\n")
    with pytest.raises(
        ValueError, match=r"experiment\.created must be a valid ISO date"
    ):
        load_config(path, project_root=tmp_path, preview=False)


def test_all_stages_share_one_metadata_directory():
    """Point every stage at the shared metadata directory."""
    config = load_config(project_root=ROOT, preview=False)
    for stage in (
        "field_of_view_extraction",
        "dataset_splitting",
        "quality_control",
        "background_function_calculation",
        "background_correction",
        "object_segmentation",
        "feature_extraction",
    ):
        assert config[stage].metadata.directory == str(
            Path(config.shared.output_root) / "metadata"
        )
        assert "output_directory" not in config[stage].metadata


def test_background_functions_belong_to_selected_experiment():
    """Isolate background functions by experiment."""
    for name in (None, "configs/exp_001.yaml"):
        config = load_config(name, project_root=ROOT, preview=False)
        expected = str(Path(config.shared.output_root) / "background_functions")
        assert config.shared.paths.background_functions_dir == expected
        assert config.background_function_calculation.image_saving.directory == expected
        assert (
            config.background_correction.background_function_selection.directory
            == expected
        )


def test_extracted_fov_belongs_to_selected_experiment():
    """Isolate extracted fields of view by experiment."""
    for name in (None, "configs/exp_001.yaml"):
        config = load_config(name, project_root=ROOT, preview=False)
        expected = str(Path(config.shared.output_root) / "extracted_fov")
        assert config.shared.paths.extracted_fov_dir == expected
        assert config.field_of_view_extraction.image_saving.directory == expected
        assert config.quality_control.processing.fov_directory == expected
        assert (
            config.background_function_calculation.processing.fov_directory == expected
        )


def test_metadata_is_isolated_between_experiments():
    """Resolve metadata paths separately for each experiment."""
    default = load_config(project_root=ROOT, preview=False)
    experiment = load_config("configs/exp_001.yaml", project_root=ROOT, preview=False)
    assert default.shared.paths.metadata_dir != experiment.shared.paths.metadata_dir
    assert experiment.shared.paths.metadata_dir == str(
        ROOT / "data/processed/exp_001/metadata"
    )


def test_recreate_deleted_snapshot_preserves_existing_files(tmp_path):
    """Recreate a missing snapshot without disturbing output files."""
    config = load_config(project_root=tmp_path, preview=False)
    snapshot = prepare_output(config)
    result = snapshot.parent / "existing_result.csv"
    result.write_text("value\n42\n")
    original = snapshot.read_text()
    snapshot.unlink()
    assert prepare_output(config) == snapshot
    assert snapshot.read_text() == original
    assert result.read_text() == "value\n42\n"


def test_create_snapshot_in_existing_input_folder(tmp_path):
    """Create the snapshot when the metadata directory already exists."""
    config = load_config(project_root=tmp_path, preview=False)
    metadata = Path(config.shared.paths.metadata_dir)
    metadata.mkdir(parents=True)
    source = metadata / "part_4a.csv"
    source.write_text("value\n1\n")
    assert prepare_output(config).is_file()
    assert source.read_text() == "value\n1\n"


def test_snapshot_writes_unicode_metadata(tmp_path):
    """Write non-ASCII metadata as UTF-8 in the configuration snapshot."""
    path = override(tmp_path, "  title: Mikroskopie – Zellen\n")
    config = load_config(path, project_root=tmp_path, preview=False)
    snapshot = prepare_output(config)
    assert "Mikroskopie – Zellen" in snapshot.read_text(encoding="utf-8")
