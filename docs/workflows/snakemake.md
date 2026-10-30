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

Run on a SLURM cluster from a login node, inside `tmux` or `screen` so the
main process keeps running while it submits SLURM jobs, with a profile made
from the template `workflow/profiles/slurm` (see
[SLURM clusters](#slurm-clusters)):

```bash
uv run --frozen --group pipeline snakemake -s workflow/Snakefile \
    --profile workflow/profiles/<cluster> --config acid_config=configs/exp_001.yaml
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
  from times measured on a SLURM cluster). Segmentation takes ~0.3 s per FOV on
  a warm GPU node, but the first job on a node can spend 1-2 min reading the
  Python environment (torch/CUDA) from a network file system, so it uses few
  large batches. A job has a fixed start-up cost (imports, reading the
  metadata, loading the segmentation model, about 15 s) that is large compared with the
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

Once `ACID_metadata_part_1.csv` exists, the raw folder is no longer read:
the extraction rules are not defined, so an unmounted, archived or changed
raw folder does not rerun the extraction or change the train/test split. New
raw data belong to a new experiment id (or delete `part_1` to extract again).

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
failed. A batch job that crashes (out of memory, time limit) is retried once by
the SLURM profiles, as a whole; after fixing the cause, run the command again.

## Running one job by hand

Every rule calls `python -m acid.pipeline <command>`, which can also be run
directly, for example to reprocess one FOV (`--fov` takes one or more names):

```bash
uv run --frozen python -m acid.pipeline segment --config configs/exp_001.yaml \
    --input data/processed/exp_001/metadata/ACID_metadata_part_4b.csv \
    --fov <fov name> --output /tmp/<fov name>.json
```

`python -m acid.pipeline --help` lists the commands.

## SLURM clusters

`workflow/profiles/slurm/config.yaml` is a template that submits with
`snakemake-executor-plugin-slurm`. Copy the folder, replace the `YOUR_...`
values (account and the partitions for short CPU jobs, the background function
and GPU jobs) and check the partitions' limits; until the placeholders are
replaced, `sbatch` rejects the first job. `workflow/profiles/README.md` lists
the steps.

Every raw file and every FOV batch is its own SLURM job. Jobs that use several
threads (background function and extraction: 16, segmentation: 4) get them
through `set-threads` and `cpus_per_task`; segmentation asks for one whole GPU
per batch with `gres: "gpu:1"`. The background function's rolling-ball fit is
single-threaded, so its runtime depends on the CPU's single-core speed; pick
fast nodes with `constraint` if the cluster has node features. Runtimes and
memory are first estimates from one run of 205 FOVs; adjust them in the
profile after the first run.
