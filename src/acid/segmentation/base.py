import logging

from abc import ABC, abstractmethod

# ---- Setting built-in logging
logger = logging.getLogger(__name__)


class SegmentationModel(ABC):
    name: str
    version: str

    @abstractmethod
    def eval(self, image, **kwargs):
        """Run model inference and return the model-native segmentation outputs."""
        pass
