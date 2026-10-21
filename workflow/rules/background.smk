# Part 3 (background functions from all unflagged FOVs) and part 4
# (background correction per FOV).


rule background_function:
    input:
        PART_3,
    output:
        PART_4A,
    log:
        LOGS / "background_function.log",
    threads: CFG.background_function_calculation.processing.n_workers
    shell:
        "{ACID} background-function --config {ACID_CONFIG} "
        "--input {input:q} --output {output:q} > {log:q} 2>&1"


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
