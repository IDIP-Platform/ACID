# Snakemake Pipeline

`workflow/Snakefile` runs the whole pipeline, from raw ND2 files to feature
tables, with a single command. It does the same processing as the notebooks in
`notebooks_refactored/` and reads the same configuration, but runs one job per
raw file and one job per batch of fields of view (FOVs), so jobs can run in
parallel. After an interruption it only repeats what is missing.

## Running

Install the `pipeline` dependency group once:

```bash
uv sync --frozen --group pipeline
```

Run on the local machine:

```bash
uv run --frozen --group pipeline snakemake -s workflow/Snakefile \
    --profile workflow/profiles/local --config acid_config=configs/exp_001.yaml
```

Run on the Chimera cluster from a login node, inside `tmux` or `screen` so the
main process keeps running while it submits SLURM jobs:

```bash
uv run --frozen --group pipeline snakemake -s workflow/Snakefile \
    --profile workflow/profiles/chimera --config acid_config=configs/exp_001.yaml
```

Add `-n` (dry run) to see the planned jobs without running them. Run the same
command again to continue after a failure or interruption.

## Configuration

The only Snakemake setting is `acid_config`, the experiment file used by the
notebooks (`configs/<experiment id>.yaml`, merged over the package defaults by
`load_config`). All paths and scientific parameters come from this file. As in
the notebooks, outputs go to `shared.output_root`, and a changed configuration
needs a new experiment id: the first job stops if `config.resolved.yaml` in the
output folder records different settings.

Two more files hold settings that do not change results, so they are not part
of the ACID configuration:

- `workflow/config.yaml`: `batch_sizes`, the number of FOVs one job processes
  per rule (default QC 5, correction 5, segmentation 50, features 2, chosen
  from times measured on Chimera). Segmentation takes ~0.3 s per FOV on a warm
  GPU node, but the first job on a node spends 1-2 min reading the Python
  environment (torch/CUDA) from `/home`, so it uses few large batches. A job
  has a fixed start-up cost (imports, reading the metadata, loading the
  segmentation model, about 15 s on Chimera) that is large compared with the
  work per FOV, so light steps are batched more. Override them with
  `--configfile <file>`.
- `workflow/profiles/<name>/config.yaml`: compute resources of every rule
  (threads, CPUs, memory, runtime, partition, GPU), each rule listed
  explicitly. A batch rule's runtime must cover a whole batch.

## Jobs

| Stage | Jobs | Writes |
| --- | --- | --- |
| part0 extraction | one per raw file, then one merge | `ACID_metadata_part_1.csv`, `extracted_fov/` |
| part1 split | one (checkpoint) | `ACID_metadata_part_2.csv` |
| part2 quality control | one measurement job per batch, then flagging and graphs | `ACID_metadata_part_3.csv`, QC graphs |
| part3 background functions | one | `ACID_metadata_part_4a.csv`, `background_functions/` |
| part4 background correction | one per batch, then one merge | `ACID_metadata_part_4b.csv`, `corrected_fov/` |
| part5 segmentation | one per batch (GPU), then one merge | `ACID_metadata_part_5.csv`, `segmentation_masks/` |
| part6 feature extraction | one per batch, then one merge | `ACID_metadata_part_6.csv`, `feature_tables/` |

The batch jobs cover the training split, as the notebooks do. The list of FOVs
is only known after extraction, so the split is a Snakemake checkpoint: once it
has run, the training FOVs are cut, in row order, into batches and the batch
jobs are added.

The metadata files have no date prefix, unlike the files saved by the notebooks
(`<YYYYMMDD>_ACID_metadata_part_3.csv`), because Snakemake needs output names
that do not change from day to day. They still match the `file_selection`
patterns, so a notebook started after the pipeline finds them.

Each batch job writes the result records of its FOVs to
`work/<stage>/batch_<i>.json` under the output folder. Logs are in
`logs/<rule>/`.

## Failed fields of view

A FOV that cannot be processed (for example an unreadable file) does not stop
the pipeline or its batch. The batch job writes a failure record for it, the
merge leaves the stage's columns empty for that row, as the notebooks do, and
lists it in `reports/<stage>_failures.csv`. A merge fails only if every FOV
failed. A batch job that crashes (out of memory, time limit) is retried once on
Chimera, as a whole; after fixing the cause, run the command again.

## Running one job by hand

Every rule calls `python -m acid.pipeline <command>`, which can also be run
directly, for example to reprocess one FOV (`--fov` takes one or more names):

```bash
uv run --frozen python -m acid.pipeline segment --config configs/exp_001.yaml \
    --input data/processed/exp_001/metadata/ACID_metadata_part_4b.csv \
    --fov <fov name> --output /tmp/<fov name>.json
```

`python -m acid.pipeline --help` lists the commands.

## Chimera profile

`workflow/profiles/chimera/config.yaml` submits with
`snakemake-executor-plugin-slurm` under the `rse` account: light jobs to
`ffa-short`, the background-function calculation to `ffa` and segmentation to
`gpu-ffa` with one whole GPU per batch (`--gres=gpu:1`; Chimera accepts GPUs only
through `--gres`, and only two gpu-ffa nodes offer shared MPS GPUs).
Every raw file and every FOV batch is its own SLURM job; jobs that use several
threads (background functions: 8 CPUs, extraction: 16, segmentation: 4) get
them through `set-threads` and `cpus_per_task`. Runtimes and memory are first estimates; adjust
them in the profile after the first run.
