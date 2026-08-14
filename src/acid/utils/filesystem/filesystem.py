"""Filesystem utilities for working with files, directories, and paths.

This module contains reusable helpers for common filesystem operations.
"""
import logging
from collections.abc import Iterable
from os import PathLike
from pathlib import Path

# ---- Setting built-in logging
logger = logging.getLogger(__name__)

# ----------------------------------------------------------
# ---------------  PUBLIC INTERFACE  -----------------------
# ----------------------------------------------------------


def list_directory_entries(
    directory: str | PathLike,
    include: str | Iterable[str] | None = None,
    exclude: str | Iterable[str] | None = None,
    hidden_prefix: str | None = ".",
    files_only: bool = True,
    return_paths: bool = False,
) -> list[str] | list[Path]:
    """List visible directory entries with optional substring filters.

    Args:
        directory: Directory whose entries should be listed.
        include: Optional substring or substrings that entry names must contain.
            If a string is provided, entries containing that string are kept. If
            an iterable of strings is provided, entries containing any of those
            strings are kept. If None, no inclusion filtering is applied.
        exclude: Optional substring or substrings that entry names must not
            contain. If a string is provided, entries containing that string are
            removed. If an iterable of strings is provided, entries containing
            any of those strings are removed. If None, no exclusion filtering is
            applied. Exclusion has priority over inclusion.
        hidden_prefix: Prefix used to identify hidden entries. Entries whose
            names start with this prefix are skipped. If None, hidden entries are
            not filtered.
        files_only: If True, only regular files are returned. If False, files and
            directories are returned.
        return_paths: If True, return Path objects. If False, return entry names
            as strings.

    Returns:
        A sorted list of entry names or Path objects, depending on return_paths.

    Raises:
        NotADirectoryError: If directory does not exist or is not a directory.
        TypeError: If include or exclude is not None, a string, or an iterable of
            strings.
    """
    directory = Path(directory)

    if not directory.is_dir():
        raise NotADirectoryError(f"Not a directory: {directory}")

    include_tokens = _normalize_tokens(include, "include")
    exclude_tokens = _normalize_tokens(exclude, "exclude")

    entries = [
        entry if return_paths else entry.name
        for entry in directory.iterdir()
        if _should_keep_entry(
            entry,
            include_tokens=include_tokens,
            exclude_tokens=exclude_tokens,
            hidden_prefix=hidden_prefix,
            files_only=files_only,
        )
    ]

    if not entries:
        logger.info("Directory is empty")
    return sorted(entries, key=_sort_key)


def create_output_directories(
    output_directory: str | PathLike,
    secondary_output_directory: str | PathLike | None = "secondary_output",
    enable_secondary_output: bool = True,
) -> tuple[Path, Path | None]:
    """Create primary and optional secondary output directories.

    Args:
        output_directory: Path to the primary output directory.
        secondary_output_directory: Path to the secondary output directory. This
            directory is created only when enable_secondary_output is True and
            secondary_output_directory is not None.
        enable_secondary_output: Whether to create the secondary output
            directory.

    Returns:
        A tuple containing the primary output directory path and the secondary
        output directory path. The secondary path is None when secondary output
        creation is disabled.

    Raises:
        FileExistsError: If a requested path exists but is not a directory.
        OSError: If one of the directories cannot be created.
    """
    enable_secondary_output = _as_bool(
        enable_secondary_output,
        "enable_secondary_output",
    )

    output_path = _ensure_directory(output_directory, label="primary output")

    secondary_output_path = None
    if enable_secondary_output and secondary_output_directory is not None:
        secondary_output_path = _ensure_directory(
            secondary_output_directory,
            label="secondary output",
        )
    else:
        logger.info("Secondary output directory creation is disabled.")

    return output_path, secondary_output_path


# ----------------------------------------------------------
# ---------------  HELPER FUNCTIONS  -----------------------
# ----------------------------------------------------------


def _as_bool(value: bool | str, argument_name: str) -> bool:
    """Convert a boolean or boolean-like string to bool.

    Args:
        value: Boolean value or string representation of a boolean.
        argument_name: Name of the argument being converted.

    Returns:
        Converted boolean value.

    Raises:
        TypeError: If value is not a bool or string.
        ValueError: If value is a string but not a supported boolean value.
    """
    if isinstance(value, bool):
        return value

    if not isinstance(value, str):
        raise TypeError(f"{argument_name} must be a bool or string.")

    normalized_value = value.strip().lower()

    if normalized_value == "true":
        return True

    if normalized_value == "false":
        return False

    raise ValueError(
        f"{argument_name} must be one of: true, false. Got {value!r}."
    )


def _ensure_directory(
    directory: str | PathLike,
    label: str = "output",
) -> Path:
    """Create a directory if needed and return it as a Path.

    Args:
        directory: Directory path to create.
        label: Human-readable directory role used in log messages.

    Returns:
        Created or existing directory path.

    Raises:
        FileExistsError: If directory exists but is not a directory.
        OSError: If the directory cannot be created.
    """
    directory_path = Path(directory)
    already_exists = directory_path.is_dir()

    logger.info("Ensuring %s directory exists: %s", label, directory_path)

    try:
        directory_path.mkdir(parents=True, exist_ok=True)
    except OSError:
        logger.exception("Failed to create %s directory: %s", label, directory_path)
        raise

    if already_exists:
        logger.info("%s directory already exists: %s", label.capitalize(), directory_path)
    else:
        logger.info("%s directory created: %s", label.capitalize(), directory_path)

    return directory_path


def _should_keep_entry(
    entry: Path,
    include_tokens: list[str],
    exclude_tokens: list[str],
    hidden_prefix: str | None,
    files_only: bool,
) -> bool:
    """Check whether a directory entry matches all filters.

    Args:
        entry: Directory entry to evaluate.
        include_tokens: Substrings that entry names may contain. If empty, no
            inclusion filtering is applied.
        exclude_tokens: Substrings that entry names must not contain. If empty,
            no exclusion filtering is applied.
        hidden_prefix: Prefix used to identify hidden entries. If None, hidden
            entries are not filtered.
        files_only: If True, only regular files are accepted.

    Returns:
        True if the entry should be kept, False otherwise.
    """
    name = entry.name

    return (
        _is_visible(name, hidden_prefix)
        and _is_allowed_entry_type(entry, files_only)
        and _matches_include_filter(name, include_tokens)
        and _matches_exclude_filter(name, exclude_tokens)
    )


def _is_visible(name: str, hidden_prefix: str | None) -> bool:
    """Check whether an entry name should be treated as visible.

    Args:
        name: Directory entry name.
        hidden_prefix: Prefix used to identify hidden entries. If None, no entry
            is treated as hidden.

    Returns:
        True if the entry should be kept, False otherwise.
    """
    return hidden_prefix is None or not name.startswith(hidden_prefix)


def _is_allowed_entry_type(entry: Path, files_only: bool) -> bool:
    """Check whether a directory entry has an allowed type.

    Args:
        entry: Directory entry to evaluate.
        files_only: If True, only regular files are accepted. If False, any entry
            type accepted by directory iteration is allowed.

    Returns:
        True if the entry type is allowed, False otherwise.
    """
    return not files_only or entry.is_file()


def _matches_include_filter(name: str, include_tokens: list[str]) -> bool:
    """Check whether an entry name matches the inclusion filter.

    Args:
        name: Directory entry name.
        include_tokens: Substrings that may appear in the entry name. If empty,
            no inclusion filtering is applied.

    Returns:
        True if the name matches the inclusion filter, False otherwise.
    """
    return not include_tokens or any(token in name for token in include_tokens)


def _matches_exclude_filter(name: str, exclude_tokens: list[str]) -> bool:
    """Check whether an entry name passes the exclusion filter.

    Args:
        name: Directory entry name.
        exclude_tokens: Substrings that must not appear in the entry name. If
            empty, no exclusion filtering is applied.

    Returns:
        True if the name passes the exclusion filter, False otherwise.
    """
    return not exclude_tokens or not any(token in name for token in exclude_tokens)


def _sort_key(entry: str | Path) -> str:
    """Return a stable sorting key for a directory entry result.

    Args:
        entry: Directory entry represented as either a string name or a Path
            object.

    Returns:
        Entry name used for sorting.
    """
    return entry.name if isinstance(entry, Path) else entry


def _normalize_tokens(
    tokens: str | Iterable[str] | None,
    argument_name: str,
) -> list[str]:
    """Normalize substring filters into a list of strings.

    Args:
        tokens: A substring, an iterable of substrings, or None.
        argument_name: Name of the argument being normalized. Used to create
            clearer error messages.

    Returns:
        A list of strings. Returns an empty list when tokens is None.

    Raises:
        TypeError: If tokens is not None, a string, or an iterable of strings.
        TypeError: If any value in tokens is not a string.
    """
    if tokens is None:
        return []

    if isinstance(tokens, str):
        return [tokens]

    try:
        normalized = list(tokens)
    except TypeError as exc:
        raise TypeError(
            f"{argument_name} must be None, a string, or an iterable of strings."
        ) from exc

    if not all(isinstance(token, str) for token in normalized):
        raise TypeError(f"All {argument_name} values must be strings.")

    return normalized
