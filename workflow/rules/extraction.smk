# Part 0 (one job per raw acquisition) and part 1 (dataset splitting).


def find_acquisitions():
    """Map `<experiment dir>/<file stem>` to each raw acquisition file."""
    cfg = CFG.field_of_view_extraction
    root = Path(cfg.acquisitions.directory)
    try:
        acquisitions = discover_acquisitions(cfg)
    except (FileNotFoundError, ValueError) as error:
        # Fine once part 1 exists; merge_extraction fails otherwise.
        logger.warning(f"No raw acquisitions found in {root}: {error}")
        return {}
    return {str(path.relative_to(root).with_suffix("")): path for path in acquisitions}


ACQUISITIONS = find_acquisitions()


rule extract_acquisition:
    input:
        lambda wildcards: ACQUISITIONS[wildcards.acquisition],
    output:
        WORK / "extraction" / "{acquisition}.csv",
    log:
        LOGS / "extract_acquisition" / "{acquisition}.log",
    group:
        "extraction"
    shell:
        "{ACID} extract-acquisition --config {ACID_CONFIG} "
        "--acquisition {input:q} --output {output:q} > {log:q} 2>&1"


rule merge_extraction:
    input:
        # In discover_acquisitions order, as the part0 notebook extracts them.
        [WORK / "extraction" / f"{key}.csv" for key in ACQUISITIONS],
    output:
        PART_1,
    log:
        LOGS / "merge_extraction.log",
    shell:
        "{ACID} merge-extraction --config {ACID_CONFIG} "
        "--inputs {input:q} --output {output:q} > {log:q} 2>&1"


checkpoint split:
    input:
        PART_1,
    output:
        PART_2,
    log:
        LOGS / "split.log",
    shell:
        "{ACID} split --config {ACID_CONFIG} "
        "--input {input:q} --output {output:q} > {log:q} 2>&1"
