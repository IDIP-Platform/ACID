# Part 3: background-function calculation (one job) and
# part 4: background correction (one job per batch of training FOVs).
#
# Config sections: background_function_calculation, background_correction.


# Python threads of the background fit: one per channel, up to n_workers.
BACKGROUND_WORKERS = max(1, CFG.background_function_calculation.processing.n_workers)


# Average the unflagged training FOVs (per dataset, well or grid position,
# depending on processing.background_function_strategy), fit the background
# function and save it to background_functions/ (file name with the date).
# Needs all FOVs, so one job. CPU only (no GPU). Reading the FOVs and the
# median use the rule's threads (--threads); the fit runs one Python thread
# per channel (processing.n_workers). The rolling ball inside is
# single-threaded, so the fit takes as long as one channel and depends on the
# CPU's single-core speed (see workflow/profiles/slurm). OMP_NUM_THREADS splits
# the job's CPUs between the channel threads. The CPU count comes from the
# profile (set-threads); the default here is n_workers. Calls
# acid.pipeline.merge_jobs.background_function_job.
rule background_function:
    input:
        PART_3,
    output:
        PART_4A,
    log:
        LOGS / "background_function.log",
    threads: BACKGROUND_WORKERS
    shell:
        "OMP_NUM_THREADS=$(( {threads} / {BACKGROUND_WORKERS} > 0 ? "
        "{threads} / {BACKGROUND_WORKERS} : 1 )) "
        "{ACID} background-function --config {ACID_CONFIG_ARG} --threads {threads} "
        "--input {input:q} --output {output:q} > {log:q} 2>&1"


# Correct a batch of extracted FOVs with their background function (loaded
# once per batch) and save them to corrected_fov/; the result records go to
# work/background_correction/batch_<i>.json. A FOV that cannot be corrected
# gives a failure record, not a failed job. Batch size:
# batch_sizes.correct_background. Calls
# acid.pipeline.fov_jobs.correct_background_job.
rule correct_background:
    input:
        metadata=PART_4A,
        # The batch's FOV list (params) is read from the split checkpoint's
        # output, which Snakemake requires as an input inside the SLURM job.
        split=PART_2,
    output:
        WORK / "background_correction" / "batch_{batch}.json",
    log:
        LOGS / "correct_background" / "batch_{batch}.log",
    params:
        fovs=batch_fovs("correct_background"),
    shell:
        "{ACID} correct-background --config {ACID_CONFIG_ARG} --input {input.metadata:q} "
        "--fov {params.fovs:q} --output {output:q} > {log:q} 2>&1"


# Write the correction results of all FOVs into part_4b and list failed FOVs
# in reports/background_correction_failures.csv. Calls
# acid.pipeline.merge_jobs.merge_results (stage background_correction).
rule merge_background_correction:
    input:
        metadata=PART_4A,
        records=batch_records("correct_background", "background_correction"),
    output:
        PART_4B,
    log:
        LOGS / "merge_background_correction.log",
    shell:
        "{ACID} merge --stage background_correction --config {ACID_CONFIG_ARG} "
        "--input {input.metadata:q} --records {input.records:q} "
        "--output {output:q} > {log:q} 2>&1"
