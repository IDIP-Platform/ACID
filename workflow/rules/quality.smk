# Part 2: image quality control.
#
# Config section: quality_control. Only training FOVs are measured, as in the
# part2 notebook; batch size: batch_sizes.measure_quality in workflow/config.yaml.


# Measure the per-channel QC statistics (PLLS, normalized intensity skewness,
# mean over std) of a batch of extracted FOVs and write them to
# work/quality_control/batch_<i>.json. Calls
# acid.pipeline.fov_jobs.measure_quality_job.
rule measure_quality:
    input:
        # Also the split checkpoint's output, from which the batch's FOV list
        # (params) is read.
        metadata=PART_2,
    output:
        WORK / "quality_control" / "batch_{batch}.json",
    log:
        LOGS / "measure_quality" / "batch_{batch}.log",
    params:
        fovs=batch_fovs("measure_quality"),
    shell:
        "{ACID} measure-quality --config {ACID_CONFIG_ARG} --input {input.metadata:q} "
        "--fov {params.fovs:q} --output {output:q} > {log:q} 2>&1"


# Collect the QC values of all training FOVs, flag the FOVs outside the
# mean-over-std thresholds and save the QC graphs to reports/quality_control/
# (if graphs_saving.save_graphs). Needs all FOVs, so one job. Calls
# acid.pipeline.merge_jobs.flag_quality_job.
rule flag_quality:
    input:
        metadata=PART_2,
        records=batch_records("measure_quality", "quality_control"),
    output:
        PART_3,
    log:
        LOGS / "flag_quality.log",
    shell:
        "{ACID} flag-quality --config {ACID_CONFIG_ARG} --input {input.metadata:q} "
        "--records {input.records:q} --output {output:q} > {log:q} 2>&1"
