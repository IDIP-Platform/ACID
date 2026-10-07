"""Extract fields of view and their metadata from raw microscope acquisitions.

This module is the field-of-view extraction stage (`part0` notebook). It finds
the raw acquisition files (one directory per experiment), splits every
multi-scene file into one OME-TIFF per scene (field of view) with
ImageJ-compatible metadata, saves the original OME-XML metadata next to them
and collects one metadata row per field of view.

Functions take the `field_of_view_extraction` configuration section (`cfg`);
each docstring names the keys it reads.
"""

import logging
from pathlib import Path

from omegaconf import DictConfig, OmegaConf

from acid.utils.listdirNHF import listdirNHF

# ---- Setting built-in logging
logger = logging.getLogger(__name__)


# ----------------------------------------------------------
# ---------------  PUBLIC INTERFACE  -----------------------
# ----------------------------------------------------------


def discover_acquisitions(cfg: DictConfig) -> list[Path]:
    """Find the raw acquisition files to extract.

    Lists the experiment directories directly inside the acquisitions
    directory, then the matching files inside each of them. Files lying
    directly in the acquisitions directory (e.g. the plate layout) are
    ignored. Hidden entries are skipped.

    Args:
        cfg (DictConfig): The `field_of_view_extraction` section. Reads
            `acquisitions.directory`,
            `acquisitions.experiment_directory_selection.include`/`exclude`
            (substrings an experiment directory name must contain / must not
            contain; `null` means no filter) and
            `acquisitions.file_selection.include`/`exclude` (the same for
            file names, e.g. `".nd2"`).

    Returns:
        list[Path]: Full paths of the acquisition files, sorted by experiment
        directory and then by file name.

    Raises:
        ValueError: If no matching file is found.
    """
    root = Path(cfg.acquisitions.directory)
    selection = OmegaConf.to_container(
        cfg.acquisitions.experiment_directory_selection, resolve=True
    )
    file_selection = OmegaConf.to_container(
        cfg.acquisitions.file_selection, resolve=True
    )
    acquisitions = []
    for name in sorted(
        listdirNHF(root, target=selection["include"], exclude=selection["exclude"])
    ):
        directory = root / name
        if not directory.is_dir():
            continue
        for filename in sorted(
            listdirNHF(
                directory,
                target=file_selection["include"],
                exclude=file_selection["exclude"],
            )
        ):
            path = directory / filename
            if path.is_file():
                acquisitions.append(path)
    if not acquisitions:
        raise ValueError(f"No matching acquisition files in {root}")
    return acquisitions

