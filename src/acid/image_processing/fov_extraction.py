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
import os
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from ome_types import to_xml
from omegaconf import DictConfig, OmegaConf

from acid.image_processing.extract_metadata import extract_bioio_scene_metadata
from acid.image_processing.make_imagej_metadata import imagej_compatible_metadata_dict
from acid.image_processing.name_metadata import extract_name_metadata
from acid.image_processing.save_metadata import save_xml_string
from acid.utils.listdirNHF import listdirNHF
from acid.utils.open_image import bioio_open_image
from acid.utils.save_image import tifffile_save_ometiff

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



def extract_acquisition(acquisition: Path, cfg: DictConfig) -> list[pd.Series]:
    """Split one raw acquisition into one OME-TIFF per scene (field of view).

    Saves the acquisition's OME-XML metadata as `<stem>_<xml_suffix>` and, for
    every scene, the squeezed image data as
    `<stem>_<experiment>_<scene><ome_suffix>` (e.g.
    `H7_DENV2_MOI1_40h_fixed_stained_well6_A07p4_A1.ome.tif`) with
    ImageJ-compatible metadata. The experiment name is taken from the parent
    directory name (e.g. `experiment_A07.4` gives `A07.4`, written as `A07p4`
    in file names); condition, organism, imaging hours and well are parsed
    from the file name.

    Args:
        acquisition (Path): Full path of the raw acquisition file (e.g. ND2),
            inside its experiment directory.
        cfg (DictConfig): The `field_of_view_extraction` section. Reads
            `image_saving.directory`, `image_saving.xml_suffix`,
            `image_saving.ome_suffix`, `image_saving.save_file_name_separator`,
            `image_saving.save_imagej_compatible`, `image_saving.photometric`;
            `naming.*` (separators, `experiment_index` and the `*_bitinfo`
            positions of each name component); and `image_metadata.*`
            (`processing_date_format`, `dimensions_order_metadata_name`,
            `location`, `microscope`, `objective`, the five `channels`,
            `physical_size_unit`).

    Returns:
        list[pd.Series]: One metadata series per scene, as returned by
        `extract_bioio_scene_metadata`.

    Raises:
        IndexError: If fewer than five channel names are configured.
    """
    input_file = acquisition.name
    experiment_subdir = acquisition.parent.name
    scenes_metadata_collection = []
    logger.info("Working on %s", input_file)
    bioio_input_image, input_image_metadata = bioio_open_image(
        str(acquisition), return_metadata=True
    )
    xml_input_image_metadata = to_xml(input_image_metadata)
    save_xml_string(
        xml_input_image_metadata,
        os.path.join(
            cfg.image_saving.directory,
            f"{acquisition.stem}{cfg.image_saving.save_file_name_separator}{cfg.image_saving.xml_suffix}",
        ),
    )
    for scene_n, scene in enumerate(bioio_input_image.scenes):
        logger.info("Working on scene %s", scene)
        bioio_input_image.set_scene(scene)
        experiment = experiment_subdir.split(cfg.naming.experiment_separator)[
            cfg.naming.experiment_index
        ]
        name_metadata_dict = extract_name_metadata(
            acquisition.stem,
            separator=cfg.naming.acquisition_filename_separator,
            infobits={
                "processing_date": datetime.now()
                .astimezone()
                .strftime(cfg.image_metadata.processing_date_format),
                "experiment": experiment,
                "condition_1": tuple(cfg.naming.condition_1_bitinfo),
                "infectious_organism": tuple(cfg.naming.infectious_organism_bitinfo),
                "condition_2": tuple(cfg.naming.condition_2_bitinfo),
                "imaging_hours": tuple(cfg.naming.imaging_hours_bitinfo),
                "well": tuple(cfg.naming.well_bitinfo),
            },
        )
        save_file_name = f"{acquisition.stem}{cfg.naming.acquisition_filename_separator}{experiment.replace(cfg.naming.sub_experiment_separator, cfg.naming.sub_experiment_separator_replacement)}{cfg.naming.acquisition_filename_separator}{scene}{cfg.image_saving.ome_suffix}"
        scene_metadata_series, scene_metadata_dict = extract_bioio_scene_metadata(
            bioio_scene=bioio_input_image,
            dims_order_name=cfg.image_metadata.dimensions_order_metadata_name,
            raw_file_name=input_file,
            scene_name=scene,
            processing_date_yymmdd=name_metadata_dict["processing_date"],
            ome_tif_file_name=save_file_name,
            location=cfg.image_metadata.location,
            microscope=cfg.image_metadata.microscope,
            objective=cfg.image_metadata.objective,
            experiment=experiment,
            condition_1=name_metadata_dict["condition_1"],
            infectious_organism=name_metadata_dict["infectious_organism"],
            condition_2=name_metadata_dict["condition_2"],
            imaging_hours=name_metadata_dict["imaging_hours"],
            well=name_metadata_dict["well"],
            channel_0=cfg.image_metadata.channels[0],
            channel_1=cfg.image_metadata.channels[1],
            channel_2=cfg.image_metadata.channels[2],
            channel_3=cfg.image_metadata.channels[3],
            channel_4=cfg.image_metadata.channels[4],
            physical_size_unit_x=cfg.image_metadata.physical_size_unit,
            physical_size_unit_y=cfg.image_metadata.physical_size_unit,
        )
        scenes_metadata_collection.append(scene_metadata_series)
        input_scene = bioio_input_image.data
        input_scene = np.squeeze(input_scene)
        imagej_scene_metadata_dict = imagej_compatible_metadata_dict(
            scene_metadata_dict
        )
        tifffile_save_ometiff(
            os.path.join(cfg.image_saving.directory, save_file_name),
            data=input_scene,
            imagej=cfg.image_saving.save_imagej_compatible,
            photometric=cfg.image_saving.photometric,
            metadata=imagej_scene_metadata_dict,
        )
    return scenes_metadata_collection



def extract_fields_of_view(acquisitions: list[Path], cfg: DictConfig) -> pd.DataFrame:
    """Extract every scene of every acquisition and collect their metadata.

    Args:
        acquisitions (list[Path]): Acquisition files, usually from
            `discover_acquisitions`.
        cfg (DictConfig): The `field_of_view_extraction` section (see
            `extract_acquisition`). `image_metadata.channels` must list
            exactly five channel names.

    Returns:
        pd.DataFrame: One row per extracted field of view, in acquisition and
        scene order.

    Raises:
        ValueError: If not exactly five channels are configured, or if no
            field of view was extracted.
    """
    if len(cfg.image_metadata.channels) != 5:
        raise ValueError("This extraction workflow requires five configured channels")
    scenes = []
    for acquisition in acquisitions:
        logger.info("Extracting %s", acquisition)
        scenes.extend(extract_acquisition(acquisition, cfg))
    if not scenes:
        raise ValueError("No fields of view were extracted")
    return pd.concat(scenes, axis=1).T

