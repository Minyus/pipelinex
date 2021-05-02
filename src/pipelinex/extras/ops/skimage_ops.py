from skimage import segmentation

from pipelinex.utils import DictToDict


class SkimageSegmentationDictToDict(DictToDict):
    module = segmentation


class SkimageSegmentationFelzenszwalb(SkimageSegmentationDictToDict):
    fn = "felzenszwalb"


class SkimageSegmentationSlic(SkimageSegmentationDictToDict):
    fn = "slic"


class SkimageSegmentationQuickshift(SkimageSegmentationDictToDict):
    fn = "quickshift"


class SkimageSegmentationWatershed(SkimageSegmentationDictToDict):
    fn = "watershed"


class SkimageMarkBoundaries(SkimageSegmentationDictToDict):
    fn = "mark_boundaries"
