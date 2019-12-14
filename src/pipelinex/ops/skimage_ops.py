from ..utils import DictToDict, list_of_dict_to_dict_of_list
from skimage import segmentation


class SkimageSegmentationDictToDict(DictToDict):
    module = segmentation


class SkimageSegmentationSlic(SkimageSegmentationDictToDict):
    fn = "slic"
