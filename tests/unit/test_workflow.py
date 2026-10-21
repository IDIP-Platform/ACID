"""Dry runs of the Snakemake workflow (`workflow/Snakefile`)."""

import re
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest
from conftest import write_pipeline_dataset

pytest.importorskip("snakemake")

SNAKEFILE = Path(__file__).resolve().parents[2] / "workflow" / "Snakefile"


def planned_jobs(project_root, experiment_id="pipeline_test"):
    """Dry-run the workflow and return {rule: number of planned jobs}."""
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "snakemake",
            "--snakefile",
            str(SNAKEFILE),
            "--directory",
            str(project_root),
            "--dry-run",
            "--config",
            f"acid_config=configs/{experiment_id}.yaml",
            f"project_root={project_root}",
        ],
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr + completed.stdout
    stats = completed.stdout.split("Job stats:")[1].split("total")[0]
    return {
        name: int(count)
        for name, count in re.findall(r"^(\w+)\s+(\d+)$", stats, flags=re.MULTILINE)
    }


def test_dry_run_plans_one_extraction_job_per_acquisition(pipeline_config, tmp_path):
    experiment = tmp_path / "raw" / "experiment_A07.2"
    experiment.mkdir(parents=True)
    for well in (1, 2, 3):
        (experiment / f"H7_DENV2_MOI1_30h_fixed_stained_well{well}.nd2").write_text("")

    jobs = planned_jobs(tmp_path)

    # FOV jobs are added only after the `split` checkpoint has run.
    assert jobs["extract_acquisition"] == 3
    assert jobs["merge_extraction"] == jobs["split"] == jobs["merge_features"] == 1
    assert "measure_quality" not in jobs


def test_dry_run_after_split_plans_one_job_per_training_fov(pipeline_config, tmp_path):
    (tmp_path / "raw").mkdir()
    part_2 = write_pipeline_dataset(pipeline_config)
    metadata = Path(pipeline_config.shared.paths.metadata_dir)
    # Written in stage order, so that part 2 is newer than part 1.
    split_df = pd.read_csv(part_2)
    split_df.to_csv(metadata / "ACID_metadata_part_1.csv", index=False)
    split_df.to_csv(metadata / "ACID_metadata_part_2.csv", index=False)

    jobs = planned_jobs(tmp_path)

    assert "extract_acquisition" not in jobs and "split" not in jobs
    for rule in (
        "measure_quality",
        "correct_background",
        "segment",
        "extract_features",
    ):
        assert jobs[rule] == 2, rule
    assert jobs["background_function"] == jobs["merge_features"] == 1
