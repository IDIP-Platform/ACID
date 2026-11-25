workspace "ACID" "Current execution paths and batch artifact handoffs" {
    model {
        operator = person "Pipeline operator" "Runs notebook cells or starts Snakemake."
        analyst = person "Data analyst" "Reads metadata and per-cell features."

        acid = softwareSystem "ACID" "Processes fixed-sample microscopy acquisitions into cellular features." {
            !docs docs

            baseline = container "Baseline notebook kernel" "Runs notebooks/ cells with mostly inline logic and selected acid helpers. Checked-in paths are machine-specific." "Jupyter / Python" "Entry Point,Legacy"
            notebooks = container "Refactored notebook kernel" "Runs notebooks_refactored/ cells, importing acid stage functions. Checked-in CONFIG_FILE is None." "Jupyter / Python" "Entry Point"
            workflow = container "Snakemake orchestrator" "Runs workflow/Snakefile rules locally or with a configured executor profile." "Snakemake" "Entry Point" {
                group "Batch rule groups" {
                    wExtract = component "Extraction rules" "Extract scenes and merge acquisition results." "workflow/rules/extraction.smk"
                    wSplit = component "Split checkpoint" "Assign treatments and train/test rows." "workflow/rules/extraction.smk"
                    wQuality = component "Quality rules" "Measure and flag FOV quality." "workflow/rules/quality.smk"
                    wBackground = component "Background-function rule" "Fit and save shared backgrounds." "workflow/rules/background.smk"
                    wCorrection = component "Correction rules" "Correct FOV backgrounds and merge results." "workflow/rules/background.smk"
                    wSegmentation = component "Segmentation rules" "Create masks and merge results." "workflow/rules/segmentation.smk"
                    wFeatures = component "Feature rules" "Extract per-object features and merge results." "workflow/rules/features.smk"
                }
            }
            worker = container "Python batch worker" "An ephemeral process running python -m acid.pipeline; imports the shared acid package for one job." "Python" "Entry Point"

            raw = container "Experiment inputs" "Raw ND2 acquisitions and a selected plate-layout CSV." "Filesystem: data/raw by default" "Input Data"
            settings = container "Experiment configuration" "YAML override required by Snakemake and optional for refactored notebooks; packaged defaults are in src/acid/config." "Filesystem: configs/<id>.yaml" "Configuration"
            processed = container "Pipeline outputs" "Derived metadata, OME-TIFFs, backgrounds, masks, features, reports, logs, and batch work files." "Filesystem: data/processed/<id> by default" "Output Data"
            baselineInputs = container "Baseline experiment inputs" "Notebook-configured raw ND2 files and plate-layout CSV; checked-in paths include Windows drives." "Filesystem: notebook-specific input paths" "Input Data"
            baselineOutputs = container "Baseline pipeline outputs" "Derived stage files and notebook metadata at configured paths, including data/proc and secondary_output." "Filesystem: notebook-specific output paths" "Output Data"
        }

        operator -> acid "Runs a pipeline path" "Jupyter / CLI"
        analyst -> acid "Uses results" "CSV"
        operator -> baseline "Runs baseline cells after configuring paths" "Jupyter"
        operator -> notebooks "Runs refactored stage cells" "Jupyter"
        operator -> workflow "Starts a batch run with experiment YAML" "CLI"
        baseline -> baselineInputs "Reads raw acquisitions and plate layout" "ND2 / CSV"
        baseline -> baselineOutputs "Reads prior-stage results and writes derived files" "OME-TIFF / CSV"
        notebooks -> raw "Reads acquisitions and plate layout through acid code" "ND2 / CSV"
        notebooks -> processed "Reads and writes stage artifacts through acid code" "OME-TIFF / CSV / YAML"
        workflow -> settings "Requires and loads experiment YAML" "YAML"
        workflow -> raw "Discovers raw acquisition paths when part_1 does not exist" "Filesystem"
        workflow -> worker "Launches a CLI process for each processing rule" "Python CLI"
        workflow -> processed "Tracks stage CSVs and batch work files" "Filesystem"
        worker -> raw "Reads acquisitions and plate layout" "ND2 / CSV"
        worker -> settings "Loads experiment override" "YAML"
        worker -> processed "Reads and writes stage artifacts and result records" "OME-TIFF / CSV / JSON / YAML"
        analyst -> processed "Reads per-cell features and metadata" "CSV"

        wExtract -> wSplit "Writes part_1 CSV; split reads it" "CSV via filesystem"
        wSplit -> wQuality "Writes part_2 CSV; QC reads training rows" "CSV via filesystem"
        wQuality -> wBackground "Writes part_3 CSV; fit reads unflagged training rows" "CSV via filesystem"
        wBackground -> wCorrection "Writes part_4a CSV and background OME-TIFFs; correction reads both" "CSV / OME-TIFF via filesystem"
        wCorrection -> wSegmentation "Writes part_4b CSV and corrected OME-TIFFs; segmentation reads both" "CSV / OME-TIFF via filesystem"
        wSegmentation -> wFeatures "Writes part_5 CSV and masks; features read both" "CSV / OME-TIFF via filesystem"
        wFeatures -> processed "Writes part_6 CSV and feature tables" "CSV via filesystem"
        wExtract -> wQuality "Writes extracted OME-TIFFs; QC reads them" "OME-TIFF via filesystem"
        wExtract -> wBackground "Writes extracted OME-TIFFs; background fit reads them" "OME-TIFF via filesystem"
        wCorrection -> wFeatures "Writes corrected OME-TIFFs; features read them" "OME-TIFF via filesystem"
    }

    views {
        container acid "ExecutionPaths" "Actual processes, file boundaries, and three runner paths" {
            include *
            autoLayout lr
        }

        component workflow "BatchArtifacts" "Snakemake rule-group dependencies; arrows are file handoffs, never function calls" {
            include wExtract wSplit wQuality wBackground wCorrection wSegmentation wFeatures processed
            autoLayout lr
        }

        styles {
            element "Person" {
                shape Person
                background #134e4a
                color #ffffff
            }
            element "Software System" {
                background #1d4ed8
                color #ffffff
            }
            element "Container" {
                background #2563eb
                color #ffffff
            }
            element "Component" {
                background #60a5fa
                color #111827
            }
            element "Entry Point" {
                background #0f766e
                color #ffffff
            }
            element "Input Data" {
                shape Cylinder
                background #166534
                color #ffffff
            }
            element "Configuration" {
                shape Folder
                background #6d28d9
                color #ffffff
            }
            element "Output Data" {
                shape Cylinder
                background #a16207
                color #ffffff
            }
            element "Legacy" {
                background #64748b
                color #ffffff
            }
        }
    }

    configuration {
        scope softwaresystem
    }
}
