import logging

from acid.segmentation.models.cellpose import CellposeSegmentationModel

# ---- Setting built-in logging
logger = logging.getLogger(__name__)


def create_segmentation_model(name: str, **kwargs):

    if name == "cellpose":
        return CellposeSegmentationModel(**kwargs)

    raise ValueError(f"Unknown segmentation model: {name}")
