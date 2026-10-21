# Part 6: feature extraction (one job per FOV of part_5).
#
# Config section: feature_extraction. Like the part6 notebook, it processes
# every row of part_5 (the training FOVs) without re-splitting.


# Smooth one corrected FOV, drop objects touching the image edge in its mask,
# measure region properties, Hessian and structure-tensor features per object
# and save the table to feature_tables/<fov>.csv; the result record goes to
# work/feature_extraction/<fov>.json. Calls
# acid.pipeline.fov_jobs.extract_features_job.
rule extract_features:
    input:
        metadata=PART_5,
    output:
        WORK / "feature_extraction" / "{fov}.json",
    log:
        LOGS / "extract_features" / "{fov}.log",
    group:
        "feature_extraction"
    shell:
        "{ACID} extract-features --config {ACID_CONFIG} --input {input.metadata:q} "
        "--fov {wildcards.fov:q} --output {output:q} > {log:q} 2>&1"


# Write the feature-extraction results of all FOVs into part_6 (the final
# metadata) and list failed FOVs in reports/feature_extraction_failures.csv.
# Calls acid.pipeline.merge_jobs.merge_results (stage feature_extraction).
rule merge_features:
    input:
        metadata=PART_5,
        records=fov_records("feature_extraction"),
    output:
        PART_6,
    log:
        LOGS / "merge_features.log",
    shell:
        "{ACID} merge --stage feature_extraction --config {ACID_CONFIG} "
        "--input {input.metadata:q} --records {input.records:q} "
        "--output {output:q} > {log:q} 2>&1"
