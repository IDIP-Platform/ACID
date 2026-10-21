# Part 5: object segmentation per FOV (Cellpose; GPU resources in the profile).


rule segment:
    input:
        metadata=PART_4B,
    output:
        WORK / "object_segmentation" / "{fov}.json",
    log:
        LOGS / "segment" / "{fov}.log",
    group:
        "object_segmentation"
    shell:
        "{ACID} segment --config {ACID_CONFIG} --input {input.metadata:q} "
        "--fov {wildcards.fov:q} --output {output:q} > {log:q} 2>&1"


rule merge_segmentation:
    input:
        metadata=PART_4B,
        records=fov_records("object_segmentation"),
    output:
        PART_5,
    log:
        LOGS / "merge_segmentation.log",
    shell:
        "{ACID} merge --stage object_segmentation --config {ACID_CONFIG} "
        "--input {input.metadata:q} --records {input.records:q} "
        "--output {output:q} > {log:q} 2>&1"
