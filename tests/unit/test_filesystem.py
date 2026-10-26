from pathlib import Path

import pytest

from acid.utils.filesystem.filesystem import list_directory_entries


@pytest.fixture
def directory_with_entries(tmp_path: Path) -> Path:
    """Create a directory with visible files, hidden files, and a subdirectory."""
    for file_name in [
        "alpha.csv",
        "beta.csv",
        "gamma.ome.tif",
        "alpha_bad.csv",
        ".hidden.csv",
    ]:
        (tmp_path / file_name).write_text("content")

    (tmp_path / "nested").mkdir()
    return tmp_path


def test_list_directory_entries_returns_sorted_visible_file_names(
    directory_with_entries: Path,
) -> None:
    result = list_directory_entries(directory_with_entries)

    assert result == [
        "alpha.csv",
        "alpha_bad.csv",
        "beta.csv",
        "gamma.ome.tif",
    ]


def test_list_directory_entries_filters_by_include_tokens(
    directory_with_entries: Path,
) -> None:
    result = list_directory_entries(
        directory_with_entries,
        include=["alpha", ".ome.tif"],
    )

    assert result == ["alpha.csv", "alpha_bad.csv", "gamma.ome.tif"]


def test_list_directory_entries_applies_exclude_after_include(
    directory_with_entries: Path,
) -> None:
    result = list_directory_entries(
        directory_with_entries,
        include=".csv",
        exclude="bad",
    )

    assert result == ["alpha.csv", "beta.csv"]


def test_list_directory_entries_can_include_directories(
    directory_with_entries: Path,
) -> None:
    result = list_directory_entries(directory_with_entries, files_only=False)

    assert result == [
        "alpha.csv",
        "alpha_bad.csv",
        "beta.csv",
        "gamma.ome.tif",
        "nested",
    ]


def test_list_directory_entries_can_return_paths(directory_with_entries: Path) -> None:
    result = list_directory_entries(
        directory_with_entries,
        include="alpha",
        return_paths=True,
    )

    assert result == [
        directory_with_entries / "alpha.csv",
        directory_with_entries / "alpha_bad.csv",
    ]


def test_list_directory_entries_can_disable_hidden_filter(
    directory_with_entries: Path,
) -> None:
    result = list_directory_entries(
        directory_with_entries,
        include=".hidden",
        hidden_prefix=None,
    )

    assert result == [".hidden.csv"]


def test_list_directory_entries_raises_for_missing_directory(tmp_path: Path) -> None:
    missing_directory = tmp_path / "missing"

    with pytest.raises(NotADirectoryError, match="Not a directory"):
        list_directory_entries(missing_directory)


def test_list_directory_entries_raises_for_invalid_include(
    directory_with_entries: Path,
) -> None:
    with pytest.raises(TypeError, match="include must be None"):
        list_directory_entries(directory_with_entries, include=10)  # type: ignore[arg-type]


def test_list_directory_entries_raises_for_invalid_exclude_item(
    directory_with_entries: Path,
) -> None:
    with pytest.raises(TypeError, match="All exclude values must be strings"):
        list_directory_entries(
            directory_with_entries,
            exclude=[".csv", 10],  # type: ignore[list-item]
        )
