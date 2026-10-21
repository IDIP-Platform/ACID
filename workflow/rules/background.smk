# Part 3: background-function calculation (one job) and
# part 4: background correction (one job per training FOV).
#
# Config sections: background_function_calculation, background_correction.


# Python threads of the background fit: one per channel, up to n_workers.
BACKGROUND_WORKERS = max(1, CFG.background_function_calculation.processing.n_workers)


# Average the unflagged training FOVs (per dataset, well or grid position,
# depending on processing.background_function_strategy), fit the background
# function and save it to background_functions/ (file name with the date).
# Needs all FOVs, so one job. CPU only (no GPU). The fit runs one Python
# thread per channel (processing.n_workers); the rolling ball inside each
# uses OpenMP threads, so the job's CPUs are split between the channels via
# OMP_NUM_THREADS. The CPU count comes from the profile (set-threads); the
# default here is n_workers. Calls
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
        "{ACID} background-function --config {ACID_CONFIG} "
        "--input {input:q} --output {output:q} > {log:q} 2>&1"


# Correct one extracted FOV with its background function and save it to
# corrected_fov/; the result record goes to
# work/background_correction/<fov>.json. A FOV that cannot be corrected gives
# a failure record, not a failed job. Calls
# acid.pipeline.fov_jobs.correct_background_job.
rule correct_background:
    input:
        metadata=PART_4A,
    output:
        WORK / "background_correction" / "{fov}.json",
    log:
        LOGS / "correct_background" / "{fov}.log",
    group:
        "background_correction"
    shell:
        "{ACID} correct-background --config {ACID_CONFIG} --input {input.metadata:q} "
        "--fov {wildcards.fov:q} --output {output:q} > {log:q} 2>&1"


# Write the correction results of all FOVs into part_4b and list failed FOVs
# in reports/background_correction_failures.csv. Calls
# acid.pipeline.merge_jobs.merge_results (stage background_correction).
rule merge_background_correction:
    input:
        metadata=PART_4A,
        records=fov_records("background_correction"),
    output:
        PART_4B,
    log:
        LOGS / "merge_background_correction.log",
    shell:
        "{ACID} merge --stage background_correction --config {ACID_CONFIG} "
        "--input {input.metadata:q} --records {input.records:q} "
        "--output {output:q} > {log:q} 2>&1"
