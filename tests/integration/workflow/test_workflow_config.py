"""Static checks of Snakemake rules and profiles."""

import re
from pathlib import Path

import pytest
import yaml

SNAKEFILE = Path(__file__).resolve().parents[3] / "workflow" / "Snakefile"
WORKFLOW = SNAKEFILE.parent
PROFILES = sorted((WORKFLOW / "profiles").glob("*/config.yaml"))


def test_batch_wildcard_accepts_four_or_more_digits():
    constraint = re.search(r'batch=r"([^"]+)"', SNAKEFILE.read_text()).group(1)

    assert re.fullmatch(constraint, "0000")
    assert re.fullmatch(constraint, "10000")
    assert not re.fullmatch(constraint, "123")


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
