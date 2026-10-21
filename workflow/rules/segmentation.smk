# Part 5: object segmentation (one job per training FOV).
#
# Config section: object_segmentation. Cellpose uses the GPU when one is
# available; the Chimera profile sends each FOV job to gpu-ffa with one GPU.


# Preprocess one corrected FOV (nucleus channel, merged concanavalin/actin),
# segment it with the model from object_segmentation.processing.model_backend,
# resize the mask to full size and save it to segmentation_masks/; the result
# record goes to work/object_segmentation/<fov>.json. Calls
# acid.pipeline.fov_jobs.segment_job.
rule segment:
    input:
        metadata=PART_4B,
    output:
        WORK / "object_segmentation" / "{fov}.json",
    log:
        LOGS / "segment" / "{fov}.log",
    shell:
        "{ACID} segment --config {ACID_CONFIG} --input {input.metadata:q} "
        "--fov {wildcards.fov:q} --output {output:q} > {log:q} 2>&1"


# Write the segmentation results of all FOVs into part_5 and list failed FOVs
# in reports/object_segmentation_failures.csv. Calls
# acid.pipeline.merge_jobs.merge_results (stage object_segmentation).
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
