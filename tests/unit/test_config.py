import json
from pathlib import Path

import pytest
from omegaconf.errors import ConfigKeyError

from acid.config import load_config, prepare_output

ROOT = Path(__file__).resolve().parents[2]


def override(tmp_path, text):
    path = tmp_path / "exp_003.yaml"
    path.write_text("experiment:\n  id: exp_003\n" + text)
    return path


def test_defaults_and_experiment_isolation(tmp_path):
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
    path = override(tmp_path, text)
    with pytest.raises(error):
        load_config(path, project_root=tmp_path, preview=False)


def test_missing_and_mismatched_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_config("missing.yaml", project_root=tmp_path)
    path = tmp_path / "exp_001.yaml"
    path.write_text("experiment:\n  id: exp_002\n")
    with pytest.raises(ValueError, match="filename"):
        load_config(path, project_root=tmp_path)


def test_snapshot_guard(tmp_path):
    path = override(tmp_path, "")
    config = load_config(path, project_root=tmp_path, preview=False)
    snapshot = prepare_output(config)
    assert prepare_output(config) == snapshot
    override(tmp_path, "object_segmentation:\n  processing:\n    diameter: 90\n")
    changed = load_config(path, project_root=tmp_path, preview=False)
    with pytest.raises(ValueError, match="differs"):
        prepare_output(changed)


def test_list_replacement_and_nullable_default(tmp_path):
    path = override(
        tmp_path,
        "feature_extraction:\n  processing:\n    axes: [0]\nbackground_correction:\n  processing:\n    max_clip_value: 100\n",
    )
    config = load_config(path, project_root=tmp_path, preview=False)
    assert list(config.feature_extraction.processing.axes) == [0]
    assert config.background_correction.processing.max_clip_value == 100


def test_notebooks_use_shared_loader_and_output_metadata():
    for path in (ROOT / "notebooks_refactored").glob("*.ipynb"):
        cells = json.loads(path.read_text())["cells"]
        code = "\n".join(
            "".join(c["source"]) for c in cells if c["cell_type"] == "code"
        )
        assert "config = load_config(CONFIG_FILE)" in code
        assert "prepare_output(config)" in code
        assert (
            "metadata_directory = metadata_config.directory" in code
            or "Path(metadata_config.directory)" in code
            or "from acid.utils.metadata.saving import save_metadata_dataframe" in code
        )
        assert "OmegaConf.load(CONFIG_PATH_FILE)" not in code
        for cell in cells:
            if cell["cell_type"] == "code":
                source = "".join(cell["source"])
                if not any(
                    line.lstrip().startswith(("%", "!")) for line in source.splitlines()
                ):
                    compile(source, str(path), "exec")


def test_background_input_check_uses_selected_metadata_directory():
    path = ROOT / "notebooks_refactored/part4_background_correction_pipeline.ipynb"
    cells = json.loads(path.read_text())["cells"]
    source = next(
        "".join(c["source"])
        for c in cells
        if "for input_directory in" in "".join(c["source"])
    )
    assert "metadata_cfg.directory," in source
    assert "paths_cfg.output_metadata_dir," not in source


def test_all_stages_share_one_metadata_directory():
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
    default = load_config(project_root=ROOT, preview=False)
    experiment = load_config("configs/exp_001.yaml", project_root=ROOT, preview=False)
    assert default.shared.paths.metadata_dir != experiment.shared.paths.metadata_dir
    assert experiment.shared.paths.metadata_dir == str(
        ROOT / "data/processed/exp_001/metadata"
    )


def test_recreate_deleted_snapshot_preserves_existing_files(tmp_path):
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
    config = load_config(project_root=tmp_path, preview=False)
    metadata = Path(config.shared.paths.metadata_dir)
    metadata.mkdir(parents=True)
    source = metadata / "part_4a.csv"
    source.write_text("value\n1\n")
    assert prepare_output(config).is_file()
    assert source.read_text() == "value\n1\n"
