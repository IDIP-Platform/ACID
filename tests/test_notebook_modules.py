import ast
import importlib
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
NEW_MODULES = [
    "acid.io.image_loading",
    "acid.utils.metadata.rows",
    "acid.utils.metadata.saving",
    "acid.utils.row_processing",
    "acid.image_processing.background.apply_background_correction",
    "acid.image_processing.background.compute_background_function",
    "acid.image_processing.segmentation_preprocessing",
    "acid.image_processing.fov_extraction",
    "acid.segmentation.pipeline",
    "acid.feature_extraction.pipeline",
    "acid.data_preparation.map_category",
    "acid.data_preparation.train_test_split",
    "acid.image_quality_control.fov_quality",
    "acid.image_quality_control.display_qc",
]


def code_cells(notebook):
    cells = json.loads(notebook.read_text())["cells"]
    for cell in cells:
        if cell["cell_type"] == "code":
            source = "".join(cell["source"])
            yield "\n".join(
                "" if line.lstrip().startswith(("%", "!")) else line
                for line in source.splitlines()
            )


@pytest.mark.parametrize(
    "notebook",
    sorted((ROOT / "notebooks_refactored").glob("*.ipynb")),
    ids=lambda path: path.name,
)
def test_refactored_notebooks_define_no_functions(notebook):
    defined = [
        node.name
        for source in code_cells(notebook)
        for node in ast.walk(ast.parse(source))
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
    ]

    assert defined == []


@pytest.mark.parametrize("module", NEW_MODULES)
def test_new_modules_import(module):
    importlib.import_module(module)


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
