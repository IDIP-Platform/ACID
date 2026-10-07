import ast
import importlib
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
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
