# Part 2: QC measurement per FOV, then flagging and graphs for all FOVs.


rule measure_quality:
    input:
        metadata=PART_2,
    output:
        WORK / "quality_control" / "{fov}.json",
    log:
        LOGS / "measure_quality" / "{fov}.log",
    group:
        "quality_control"
    shell:
        "{ACID} measure-quality --config {ACID_CONFIG} --input {input.metadata:q} "
        "--fov {wildcards.fov:q} --output {output:q} > {log:q} 2>&1"


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
