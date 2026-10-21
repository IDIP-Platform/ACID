# Part 0: field-of-view extraction (one job per raw acquisition) and
# part 1: dataset splitting (one job, a checkpoint).
#
# Config sections: field_of_view_extraction, dataset_splitting.


def find_acquisitions():
    """Map `<experiment dir>/<file stem>` to each raw acquisition file.

    Uses the same discovery as the part0 notebook (`discover_acquisitions`):
    experiment directories inside `field_of_view_extraction.acquisitions.directory`
    and the matching files (e.g. .nd2) inside them, sorted. The key, e.g.
    `experiment_A07.2/H7_DENV2_MOI1_30h_fixed_stained_well1`, is the
    {acquisition} wildcard. Raises if no raw file is found.
    """
    cfg = CFG.field_of_view_extraction
    root = Path(cfg.acquisitions.directory)
    return {
        str(path.relative_to(root).with_suffix("")): path
        for path in discover_acquisitions(cfg)
    }


# Once part_1 exists it is the source of all later stages and the extraction
# rules are not defined: otherwise a raw folder that is unmounted, archived or
# changed later would change merge_extraction's inputs, make Snakemake rerun
# the extraction and rewrite part_1 (and with it the train/test split). New
# raw data go into a new experiment id, as with the notebooks.
if not Path(PART_1).exists():
    ACQUISITIONS = find_acquisitions()

    # Extract one raw file: save one OME-TIFF per scene into extracted_fov/
    # and the scenes' metadata rows into work/extraction/<acquisition>.csv.
    # Calls acid.pipeline.fov_jobs.extract_acquisition_job.
    rule extract_acquisition:
        input:
            lambda wildcards: ACQUISITIONS[wildcards.acquisition],
        output:
            WORK / "extraction" / "{acquisition}.csv",
        log:
            LOGS / "extract_acquisition" / "{acquisition}.log",
        shell:
            "{ACID} extract-acquisition --config {ACID_CONFIG_ARG} "
            "--acquisition {input:q} --output {output:q} > {log:q} 2>&1"

    # Concatenate the per-acquisition CSVs into part_1, in the order the part0
    # notebook extracts them, so the rows (and therefore the random train/test
    # split) are the same. Calls acid.pipeline.merge_jobs.merge_fov_extraction.
    rule merge_extraction:
        input:
            [WORK / "extraction" / f"{key}.csv" for key in ACQUISITIONS],
        output:
            PART_1,
        log:
            LOGS / "merge_extraction.log",
        shell:
            "{ACID} merge-extraction --config {ACID_CONFIG_ARG} "
            "--inputs {input:q} --output {output:q} > {log:q} 2>&1"

    # A light job in the Snakemake main process.
    localrules:
        merge_extraction,


# Map treatments from the plate layout (found in the raw folder by
# dataset_splitting.plate_layout) and add the train/test split column.
# A checkpoint: after it runs, `fov_names()` in the Snakefile reads part_2
# and Snakemake adds one job per training FOV for the next stages.
# Calls acid.pipeline.merge_jobs.split_dataset_job.
checkpoint split:
    input:
        PART_1,
    output:
        PART_2,
    log:
        LOGS / "split.log",
    shell:
        "{ACID} split --config {ACID_CONFIG_ARG} "
        "--input {input:q} --output {output:q} > {log:q} 2>&1"
