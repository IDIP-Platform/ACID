# Part 6: feature extraction (one job per batch of the FOVs of part_5).
#
# Config section: feature_extraction. Like the part6 notebook, it processes
# every row of part_5 (the training FOVs) without re-splitting; batch size:
# batch_sizes.extract_features.


# For each FOV of the batch: smooth the corrected image, drop objects touching
# the image edge in its mask, measure region properties, Hessian and
# structure-tensor features per object and save the table to
# feature_tables/<fov>_bg.csv; the result records go to
# work/feature_extraction/batch_<i>.json. Calls
# acid.pipeline.fov_jobs.extract_features_job.
rule extract_features:
    input:
        metadata=PART_5,
        # The batch's FOV list (params) is read from the split checkpoint's
        # output, which Snakemake requires as an input inside the SLURM job.
        split=PART_2,
    output:
        WORK / "feature_extraction" / "batch_{batch}.json",
    log:
        LOGS / "extract_features" / "batch_{batch}.log",
    params:
        fovs=batch_fovs("extract_features"),
    shell:
        "{ACID} extract-features --config {ACID_CONFIG_ARG} --input {input.metadata:q} "
        "--fov {params.fovs:q} --output {output:q} > {log:q} 2>&1"


# Write the feature-extraction results of all FOVs into part_6 (the final
# metadata) and list failed FOVs in reports/feature_extraction_failures.csv.
# Calls acid.pipeline.merge_jobs.merge_results (stage feature_extraction).
rule merge_features:
    input:
        metadata=PART_5,
        records=batch_records("extract_features", "feature_extraction"),
    output:
        PART_6,
    log:
        LOGS / "merge_features.log",
    shell:
        "{ACID} merge --stage feature_extraction --config {ACID_CONFIG_ARG} "
        "--input {input.metadata:q} --records {input.records:q} "
        "--output {output:q} > {log:q} 2>&1"
