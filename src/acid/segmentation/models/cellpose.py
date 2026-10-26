import logging

import cellpose
from cellpose import models
from torch.cuda import is_available


from acid.segmentation.base import SegmentationModel

# ---- Setting built-in logging
logger = logging.getLogger(__name__)


class CellposeSegmentationModel(SegmentationModel):
    def __init__(self, gpu: bool | None = None):
        if gpu is None:
            gpu = is_available()

        self.name = "cellpose"
        self.version = cellpose.version
        self.gpu = gpu

        self.model = models.CellposeModel(gpu=gpu)

    def eval(self, image, **kwargs):
        return self.model.eval(image, **kwargs)
