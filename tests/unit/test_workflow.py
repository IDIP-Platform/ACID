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


def dry_run(project_root, experiment_id="pipeline_test", extra_args=()):
    """Dry-run the workflow and return the finished process."""
    return subprocess.run(
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
            *extra_args,
        ],
        capture_output=True,
        text=True,
    )


def planned_jobs(project_root, experiment_id="pipeline_test", extra_args=()):
    """Dry-run the workflow and return {rule: number of planned jobs}."""
    completed = dry_run(project_root, experiment_id, extra_args)
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


def write_split_metadata(pipeline_config, tmp_path):
    """Write part 1 and part 2 so that the split checkpoint is up to date."""
    (tmp_path / "raw").mkdir()
    part_2 = write_pipeline_dataset(pipeline_config)
    metadata = Path(pipeline_config.shared.paths.metadata_dir)
    # Written in stage order, so that part 2 is newer than part 1.
    split_df = pd.read_csv(part_2)
    split_df.to_csv(metadata / "ACID_metadata_part_1.csv", index=False)
    split_df.to_csv(metadata / "ACID_metadata_part_2.csv", index=False)


BATCH_RULES = ("measure_quality", "correct_background", "segment", "extract_features")


def test_dry_run_after_split_plans_one_job_per_batch(pipeline_config, tmp_path):
    write_split_metadata(pipeline_config, tmp_path)

    jobs = planned_jobs(tmp_path)

    # Two training FOVs fit into one batch of the default batch sizes.
    assert "extract_acquisition" not in jobs and "split" not in jobs
    for rule in BATCH_RULES:
        assert jobs[rule] == 1, rule
    assert jobs["background_function"] == jobs["merge_features"] == 1


def test_dry_run_uses_batch_sizes_from_configfile(pipeline_config, tmp_path):
    write_split_metadata(pipeline_config, tmp_path)
    configfile = tmp_path / "batches.yaml"
    configfile.write_text(
        "batch_sizes:\n"
        + "".join(
            f"  {rule}: {1 if rule == 'segment' else 50}\n" for rule in BATCH_RULES
        )
    )

    jobs = planned_jobs(tmp_path, extra_args=("--configfile", str(configfile)))

    assert jobs["segment"] == 2
    assert jobs["measure_quality"] == jobs["extract_features"] == 1


def test_dry_run_ignores_raw_folder_once_part_1_exists(pipeline_config, tmp_path):
    write_split_metadata(pipeline_config, tmp_path)
    experiment = tmp_path / "raw" / "experiment_A07.9"
    experiment.mkdir()
    (experiment / "H7_DENV2_MOI1_30h_fixed_stained_well9.nd2").write_text("")

    jobs = planned_jobs(tmp_path)

    assert "extract_acquisition" not in jobs and "merge_extraction" not in jobs
    assert "split" not in jobs


def test_dry_run_without_part_1_and_raw_files_fails_clearly(pipeline_config, tmp_path):
    (tmp_path / "raw").mkdir()

    completed = dry_run(tmp_path)

    assert completed.returncode != 0
    assert "No matching acquisition files" in completed.stdout + completed.stderr


def test_dry_run_quotes_paths_with_spaces(tmp_path):
    root = tmp_path / "my project"
    (root / "configs").mkdir(parents=True)
    (root / "configs" / "spaced.yaml").write_text(
        'experiment:\n  id: "spaced"\nshared:\n  paths:\n    raw_acquisitions_dir: "raw"\n'
    )
    experiment = root / "raw" / "experiment_A07.2"
    experiment.mkdir(parents=True)
    (experiment / "H7_DENV2_MOI1_30h_fixed_stained_well1.nd2").write_text("")

    completed = dry_run(root, "spaced", extra_args=("--printshellcmds",))

    assert completed.returncode == 0, completed.stderr + completed.stdout
    assert f"--project-root '{root}'" in completed.stdout
