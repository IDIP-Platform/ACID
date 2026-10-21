# Part 2: image quality control.
#
# Config section: quality_control. Only training FOVs are measured, as in the
# part2 notebook.


# Measure the per-channel QC statistics (PLLS, normalized intensity skewness,
# mean over std) of one extracted FOV and write them to
# work/quality_control/<fov>.json. Calls
# acid.pipeline.fov_jobs.measure_quality_job.
rule measure_quality:
    input:
        metadata=PART_2,
    output:
        WORK / "quality_control" / "{fov}.json",
    log:
        LOGS / "measure_quality" / "{fov}.log",
    shell:
        "{ACID} measure-quality --config {ACID_CONFIG} --input {input.metadata:q} "
        "--fov {wildcards.fov:q} --output {output:q} > {log:q} 2>&1"


# Collect the QC values of all training FOVs, flag the FOVs outside the
# mean-over-std thresholds and save the QC graphs to reports/quality_control/
# (if graphs_saving.save_graphs). Needs all FOVs, so one job. Calls
# acid.pipeline.merge_jobs.flag_quality_job.
rule flag_quality:
    input:
        metadata=PART_2,
        records=fov_records("quality_control"),
    output:
        PART_3,
    log:
        LOGS / "flag_quality.log",
    shell:
        "{ACID} flag-quality --config {ACID_CONFIG} --input {input.metadata:q} "
        "--records {input.records:q} --output {output:q} > {log:q} 2>&1"
