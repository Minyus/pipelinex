from ..utils import DictToDict
from skimage import segmentation


class SkimageSegmentationDictToDict(DictToDict):
    module = segmentation


class SkimageSegmentationSlic(SkimageSegmentationDictToDict):
    fn = "slic"


class SkimageMarkBoundaries(SkimageSegmentationDictToDict):
    fn = "mark_boundaries"
