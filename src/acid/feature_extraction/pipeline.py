"""Extract per-object features from every segmented field of view.

This module is the feature-extraction stage (`part6` notebook). For each
metadata row it loads the background-corrected field of view and its
segmentation mask, smooths the image, drops objects touching the image edge,
measures region properties plus Hessian and structure-tensor eigenvalue
features per object, saves one CSV per field of view and returns one result
record (see `acid.utils.row_processing`).

Most functions take the `feature_extraction` configuration section; each
docstring names the keys it reads.
"""

import logging

from omegaconf import DictConfig

# ---- Setting built-in logging
logger = logging.getLogger(__name__)

#: Keys of `feature_extraction.metadata.dataframe_columns` whose values are the
#: metadata columns written by this stage, in output order.
FEATURE_METADATA_COLUMN_CONFIG_KEYS = (
    "metadata_df_date_clm_name",
    "metadata_df_file_name_clm_name",
    "metadata_df_method_clm_name",
)


# ----------------------------------------------------------
# ---------------  PUBLIC INTERFACE  -----------------------
# ----------------------------------------------------------


def get_feature_metadata_columns(dataframe_columns: DictConfig) -> list[str]:
    """Return the metadata column names written by feature extraction.

    Args:
        dataframe_columns (DictConfig): The
            `feature_extraction.metadata.dataframe_columns` section. Reads the
            keys listed in `FEATURE_METADATA_COLUMN_CONFIG_KEYS`.

    Returns:
        list[str]: Column names for the extraction date, the feature table
        file name and the preprocessing description.
    """
    return [dataframe_columns[key] for key in FEATURE_METADATA_COLUMN_CONFIG_KEYS]
