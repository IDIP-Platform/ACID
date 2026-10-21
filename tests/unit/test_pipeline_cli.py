import pandas as pd
import pytest

from acid.pipeline.cli import build_parser, main
from acid.pipeline.records import read_records


def run(tmp_path, *args):
    main(
        [
            "--project-root",
            str(tmp_path),
            args[0],
            "--config",
            "configs/pipeline_test.yaml",
            *args[1:],
        ]
    )


def test_measure_quality_command_writes_records_of_all_fovs(
    pipeline_config, pipeline_dataset, tmp_path
):
    output = tmp_path / "r.json"

    run(
        tmp_path,
        "measure-quality",
        "--input",
        str(pipeline_dataset),
        "--fov",
        "p_well1_A07p2_A1",
        "p_well2_A07p2_A1",
        "--output",
        str(output),
    )

    assert [r["row_index"] for r in read_records([output])] == [0, 2]


def test_merge_extraction_command_concatenates_inputs(pipeline_config, tmp_path):
    inputs = []
    for name in ("a", "b"):
        path = tmp_path / f"{name}.csv"
        path.write_text(f"ome_tif_file_name\n{name}.ome.tif\n")
        inputs.append(str(path))
    output = tmp_path / "part_1.csv"

    run(tmp_path, "merge-extraction", "--inputs", *inputs, "--output", str(output))

    assert pd.read_csv(output)["ome_tif_file_name"].tolist() == [
        "a.ome.tif",
        "b.ome.tif",
    ]


def test_merge_command_rejects_unknown_stage():
    with pytest.raises(SystemExit):
        build_parser().parse_args(
            [
                "merge",
                "--config",
                "c.yaml",
                "--output",
                "o",
                "--stage",
                "x",
                "--input",
                "i",
                "--records",
                "r",
            ]
        )
