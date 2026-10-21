# Part 5: object segmentation (one job per batch of training FOVs).
#
# Config section: object_segmentation. Cellpose uses the GPU when one is
# available; the Chimera profile sends each batch job to gpu-ffa with one GPU.
# The model is loaded once per batch (batch_sizes.segment).


# For each corrected FOV of the batch: preprocess it (nucleus channel, merged
# concanavalin/actin), segment it with the model from
# object_segmentation.processing.model_backend, resize the mask to full size
# and save it to segmentation_masks/; the result records go to
# work/object_segmentation/batch_<i>.json. Calls
# acid.pipeline.fov_jobs.segment_job.
rule segment:
    input:
        metadata=PART_4B,
        # The batch's FOV list (params) is read from the split checkpoint's
        # output, which Snakemake requires as an input inside the SLURM job.
        split=PART_2,
    output:
        WORK / "object_segmentation" / "batch_{batch}.json",
    log:
        LOGS / "segment" / "batch_{batch}.log",
    params:
        fovs=batch_fovs("segment"),
    shell:
        "{ACID} segment --config {ACID_CONFIG} --input {input.metadata:q} "
        "--fov {params.fovs:q} --output {output:q} > {log:q} 2>&1"


# Write the segmentation results of all FOVs into part_5 and list failed FOVs
# in reports/object_segmentation_failures.csv. Calls
# acid.pipeline.merge_jobs.merge_results (stage object_segmentation).
rule merge_segmentation:
    input:
        metadata=PART_4B,
        records=batch_records("segment", "object_segmentation"),
    output:
        PART_5,
    log:
        LOGS / "merge_segmentation.log",
    shell:
        "{ACID} merge --stage object_segmentation --config {ACID_CONFIG} "
        "--input {input.metadata:q} --records {input.records:q} "
        "--output {output:q} > {log:q} 2>&1"
