import cv2
import numpy as np
from ..utils import list_of_dict_to_dict_of_list


class DictToDict:
    module = None
    fn = None

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        assert isinstance(self.fn, str)

    def __call__(self, d):
        kwargs = self.kwargs
        if self.module is None:
            fn = eval(self.fn)
        else:
            fn = getattr(self.module, self.fn)
        assert callable(fn)
        assert isinstance(d, dict)
        out = {k: fn(e, **kwargs) for k, e in d.items()}
        return out


class NpDictToDict(DictToDict):
    module = np


class CvDictToDict(DictToDict):
    module = cv2


class NpConcat(NpDictToDict):
    fn = "concatenate"


class NpStack(NpDictToDict):
    fn = "stack"


class NpSum(NpDictToDict):
    fn = "sum"


class NpMean(NpDictToDict):
    fn = "mean"


class NpSquare(NpDictToDict):
    fn = "square"


class NpSqrt(NpDictToDict):
    fn = "sqrt"


class NpAbs(NpDictToDict):
    fn = "abs"


def fit_to_uint8(a):
    min = a.min()
    max = a.max()
    return (255 * (a - min) / (max - min)).astype(np.uint8)


class CvFitToUint8(DictToDict):
    fn = "fit_to_uint8"


class CvModuleListMerge:
    def __init__(self, *modules):
        for m in modules:
            assert callable(m)
        self.modules = modules

    def __call__(self, d):
        assert isinstance(d, dict)
        list_d = [m(d) for m in self.modules]
        d_list = list_of_dict_to_dict_of_list(list_d)
        return d_list


class CvModuleConcat(CvModuleListMerge):
    def __call__(self, d):
        d_list = super().__call__(d)
        d_out = NpConcat(axis=0)(d_list)
        return d_out


class CvModuleStack(CvModuleListMerge):
    def __call__(self, d):
        d_list = super().__call__(d)
        np_stack = NpStack(axis=0)
        d_stacked = np_stack(d_list)
        return d_stacked


class CvModuleSum(CvModuleStack):
    def __call__(self, d):
        d_stacked = super().__call__(d)
        d_out = NpSum(axis=0)(d_stacked)
        return d_out


class CvModuleMean(CvModuleStack):
    def __call__(self, d):
        d_stacked = super().__call__(d)
        d_out = NpMean(axis=0)(d_stacked)
        return d_out


class CvModuleL1(CvModuleStack):
    def __call__(self, d):
        d_stacked = super().__call__(d)
        d_out = NpSum(axis=0)(NpAbs()(d_stacked))
        return d_out


class CvModuleL2(CvModuleStack):
    def __call__(self, d):
        d_stacked = super().__call__(d)
        d_out = NpSqrt()(NpSum(axis=0)(NpSquare()(d_stacked)))
        return d_out


class CvScale:
    def __init__(self, width, height):
        self.width = width
        self.height = height

    def __call__(self, img):
        width = self.width
        height = self.height
        if isinstance(width, float):
            width = int(img.shape[1] * width)
        if isinstance(height, float):
            height = int(img.shape[0] * height)
        return cv2.resize(img, (width, height))


class CvResize(CvDictToDict):
    fn = "resize"


class CvCvtColor(CvDictToDict):
    fn = "cvtColor"


class CvDilate(CvDictToDict):
    fn = "dilate"


class CvErode(CvDictToDict):
    fn = "erode"


class CvFilter2d(CvDictToDict):
    fn = "filter2D"


class CvBlur(CvDictToDict):
    fn = "blur"


class CvBoxFilter(CvDictToDict):
    fn = "boxFilter"


class CvGaussianBlur(CvDictToDict):
    fn = "GaussianBlur"


class CvMedianBlur(CvDictToDict):
    fn = "medianBlur"


class CvBilateralFilter(CvDictToDict):
    fn = "bilateralFilter"


class CvCanny(CvDictToDict):
    fn = "Canny"


class CvHoughLinesP(CvDictToDict):
    fn = "HoughLinesP"


class CvLine(CvDictToDict):
    fn = "line"


class CvEqualizeHist(CvDictToDict):
    fn = "equalizeHist"


class CvBGR2Gray(CvDictToDict):
    fn = "cvtColor"

    def __init__(self, *args, **kwargs):
        kwargs["code"] = cv2.COLOR_BGR2GRAY
        super().__init__(*args, **kwargs)


class CvBGR2HSV(CvDictToDict):
    fn = "cvtColor"

    def __init__(self, *args, **kwargs):
        kwargs["code"] = cv2.COLOR_BGR2HSV
        super().__init__(*args, **kwargs)


diagonal_edge_kernel_dict = {
    1: [
        np.array([[1, 0, 0], [0, 0, 0], [0, 0, -1]]),
        np.array([[0, 0, 1], [0, 0, 0], [-1, 0, 0]]),
    ],
    2: [
        np.array([[2, 1, 0], [1, 0, -1], [0, -1, -2]]),
        np.array([[0, 1, 2], [-1, 0, 1], [-2, -1, 0]]),
    ],
}


class CvDiagonalEdgeFilter2d(CvModuleL2):
    def __init__(self, kernel_type=2, **kwargs):
        kwargs.setdefault("ddepth", -1)
        self.modules = [
            CvFilter2d(kernel=k, **kwargs)
            for k in diagonal_edge_kernel_dict[kernel_type]
        ]
