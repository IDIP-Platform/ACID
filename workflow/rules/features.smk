# Part 6: feature extraction per FOV.


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
