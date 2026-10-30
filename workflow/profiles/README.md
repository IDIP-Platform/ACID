# Snakemake profiles

A profile is a folder with a `config.yaml` of Snakemake command-line options,
selected with `--profile workflow/profiles/<name>`. The profiles here set only
**where the jobs run and with which resources** (threads, CPUs, memory,
runtime, partition, GPU). They do not change results: scientific parameters
are in the ACID config (`--config acid_config=configs/<id>.yaml`), and the
number of FOVs per job in `workflow/config.yaml` (`batch_sizes`).

| Profile | Runs jobs | Use it for |
| --- | --- | --- |
| `local` | as processes on this machine, at most `cores: 4` side by side | small data sets, trying the workflow, dry runs |
| `chimera` | as SLURM jobs on the Chimera cluster (account `rse`, partitions `ffa-short`, `ffa`, `gpu-ffa`) | full runs on Chimera |
| `slurm` | as SLURM jobs; a template with `YOUR_...` placeholders | starting a profile for another SLURM cluster |

```bash
uv sync --frozen --group pipeline
uv run --frozen --group pipeline snakemake -s workflow/Snakefile \
    --profile workflow/profiles/local --config acid_config=configs/<id>.yaml
```

Add `-n` for a dry run. Options given on the command line override the
profile, e.g. `--cores 8` with `local` or `--jobs 50` with a SLURM profile.

## A profile for another SLURM cluster

1. Copy the template: `cp -r workflow/profiles/slurm workflow/profiles/<cluster>`.
2. Replace every `YOUR_...` value: the account (`sacctmgr show associations user=$USER`)
   and the partitions for short CPU jobs, the one long CPU job
   (`background_function`, up to 2 h) and GPU jobs (`segment`). Until they are
   replaced, `sbatch` rejects the first job.
3. Check the partitions' time and memory limits (`sinfo -o "%P %l %m %G"`) and
   how the cluster takes GPUs. The template asks for `gres: "gpu:1"`; use
   `slurm_extra` for anything else your cluster needs.
4. Optionally pick fast nodes for `background_function` with
   `constraint: "<node feature>"`: its rolling-ball fit is single-threaded, so
   it depends on the CPU's single-core speed.
5. Dry-run (`-n`), then run, and adjust `mem_mb` and `runtime` with
   `sacct`/`seff`. A batch rule's runtime must cover a whole batch of
   `batch_sizes` FOVs.

Run the Snakemake main process on a login node inside `tmux` or `screen`: it
submits the jobs and must keep running until they finish.

## Adding a rule

Every profile lists every rule in `set-threads`, and every SLURM profile every
rule that is submitted as a job (not in `localrules`) in `set-resources`, with
`cpus_per_task` equal to its threads. `tests/unit/test_workflow.py` checks
this and dry-runs each profile, so a new rule needs an entry in all profiles.

See `docs/workflows/snakemake.md` for the whole workflow.
