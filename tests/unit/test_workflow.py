"""Dry runs of the Snakemake workflow (`workflow/Snakefile`)."""

import re
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest
import yaml
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


def test_batch_wildcard_accepts_four_or_more_digits():
    constraint = re.search(r'batch=r"([^"]+)"', SNAKEFILE.read_text()).group(1)

    assert re.fullmatch(constraint, "0000")
    assert re.fullmatch(constraint, "10000")
    assert not re.fullmatch(constraint, "123")


WORKFLOW = SNAKEFILE.parent
PROFILES = sorted((WORKFLOW / "profiles").glob("*/config.yaml"))


def workflow_rules():
    """Return (all rules, local rules) defined by the Snakefile and its rule files."""
    sources = [SNAKEFILE, *sorted((WORKFLOW / "rules").glob("*.smk"))]
    text = "\n".join(path.read_text() for path in sources)
    rules = set(re.findall(r"^\s*(?:rule|checkpoint) (\w+):", text, flags=re.MULTILINE))
    local = set()
    for block in re.findall(r"^\s*localrules:\n((?:\s+\w+,?\n)+)", text, re.MULTILINE):
        local.update(re.findall(r"\w+", block))
    return rules, local


@pytest.mark.parametrize("profile", PROFILES, ids=lambda path: path.parent.name)
def test_profile_sets_threads_of_every_rule(profile):
    rules, _ = workflow_rules()
    settings = yaml.safe_load(profile.read_text())

    assert set(settings["set-threads"]) == rules


@pytest.mark.parametrize("profile", PROFILES, ids=lambda path: path.parent.name)
def test_slurm_profile_sets_resources_of_every_submitted_rule(profile):
    settings = yaml.safe_load(profile.read_text())
    if settings.get("executor") != "slurm":
        pytest.skip("not a SLURM profile")
    rules, local = workflow_rules()

    assert set(settings["set-resources"]) == rules - local
    for rule, resources in settings["set-resources"].items():
        assert resources["cpus_per_task"] == settings["set-threads"][rule], rule


@pytest.mark.parametrize("profile", PROFILES, ids=lambda path: path.parent.name)
def test_dry_run_with_profile(profile, pipeline_config, tmp_path):
    write_split_metadata(pipeline_config, tmp_path)

    jobs = planned_jobs(tmp_path, extra_args=("--profile", str(profile.parent)))

    assert jobs["segment"] == jobs["merge_features"] == 1
