"""Command line for the Snakemake workflow jobs.

Run as `python -m acid.pipeline <command> --config <yaml> ...`. Every command
loads the configuration with `load_config` and runs one function of
`acid.pipeline.fov_jobs` or `acid.pipeline.merge_jobs`, e.g.

    python -m acid.pipeline segment --config configs/exp_001.yaml \\
        --input data/processed/exp_001/metadata/ACID_metadata_part_4b.csv \\
        --fov H7_DENV2_MOI1_40h_fixed_stained_well6_A07p4_G6 \\
        --output data/processed/exp_001/work/segmentation/<fov>.json

The commands can also be run by hand, e.g. to reprocess one field of view.
"""

import argparse
import logging

from acid.config import load_config
from acid.pipeline import fov_jobs, merge_jobs

# ---- Setting built-in logging
logger = logging.getLogger(__name__)

# Per-FOV commands: config, --input metadata, --fov name, --output record.
FOV_COMMANDS = {
    "measure-quality": fov_jobs.measure_quality_job,
    "correct-background": fov_jobs.correct_background_job,
    "segment": fov_jobs.segment_job,
    "extract-features": fov_jobs.extract_features_job,
}


# ----------------------------------------------------------
# ---------------  PUBLIC INTERFACE  -----------------------
# ----------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser with one subcommand per job.

    Returns:
        argparse.ArgumentParser: The parser.
    """
    parser = argparse.ArgumentParser(prog="python -m acid.pipeline")
    parser.add_argument("--log-level", default="INFO")
    parser.add_argument(
        "--project-root",
        help="Base of relative paths; default: nearest directory with pyproject.toml",
    )
    commands = parser.add_subparsers(dest="command", required=True)

    def command(name, help_text):
        sub = commands.add_parser(name, help=help_text)
        sub.add_argument("--config", required=True, help="Experiment override YAML")
        sub.add_argument("--output", required=True)
        return sub

    command("extract-acquisition", "Extract the FOVs of one raw file").add_argument(
        "--acquisition", required=True
    )
    command("merge-extraction", "Combine per-acquisition metadata").add_argument(
        "--inputs", nargs="*", default=[]
    )
    command("split", "Map treatments and split train/test").add_argument(
        "--input", required=True
    )
    for name in FOV_COMMANDS:
        sub = command(name, f"{name} for one FOV")
        sub.add_argument("--input", required=True)
        sub.add_argument("--fov", required=True)
    sub = command("flag-quality", "Combine QC measurements and flag")
    sub.add_argument("--input", required=True)
    sub.add_argument("--records", nargs="+", required=True)
    command("background-function", "Calculate background functions").add_argument(
        "--input", required=True
    )
    sub = command("merge", "Write per-FOV results into the metadata")
    sub.add_argument(
        "--stage", required=True, choices=sorted(merge_jobs.RESULT_UPDATERS)
    )
    sub.add_argument("--input", required=True)
    sub.add_argument("--records", nargs="+", required=True)
    return parser


def main(argv: list[str] | None = None) -> None:
    """Parse the command line and run one job.

    Args:
        argv (list[str] | None): Arguments without the program name; `None`
            reads `sys.argv`.
    """
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=args.log_level.upper(),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    config = load_config(args.config, project_root=args.project_root, preview=False)

    if args.command == "extract-acquisition":
        fov_jobs.extract_acquisition_job(config, args.acquisition, args.output)
    elif args.command == "merge-extraction":
        merge_jobs.merge_fov_extraction(config, args.inputs, args.output)
    elif args.command == "split":
        merge_jobs.split_dataset_job(config, args.input, args.output)
    elif args.command in FOV_COMMANDS:
        FOV_COMMANDS[args.command](config, args.input, args.fov, args.output)
    elif args.command == "flag-quality":
        merge_jobs.flag_quality_job(config, args.input, args.records, args.output)
    elif args.command == "background-function":
        merge_jobs.background_function_job(config, args.input, args.output)
    elif args.command == "merge":
        merge_jobs.merge_results(
            config, args.stage, args.input, args.records, args.output
        )
    logger.info("%s finished: %s", args.command, args.output)
