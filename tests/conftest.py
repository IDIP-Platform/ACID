from pathlib import Path
import pytest
import shutil

PROJECT_ROOT = Path(__file__).resolve().parent.parent
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"


@pytest.fixture(scope="session", autouse=True)
def setup_test_data():
    raw_dir = Path("data/raw")
    raw_dir.mkdir(parents=True, exist_ok=True)

    source_root = Path("tests/reference")
    for dirname in ("experiment_A07.2", "experiment_A07.3", "experiment_A07.4"):
        shutil.copytree(source_root / dirname, raw_dir / dirname, dirs_exist_ok=True)

    yield

    shutil.rmtree(raw_dir)
