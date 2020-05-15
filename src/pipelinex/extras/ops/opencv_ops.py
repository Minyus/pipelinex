from pipelinex.utils import DictToDict, dict_io, list_of_dict_to_dict_of_list
import cv2
import numpy as np


class NpDictToDict(DictToDict):
    module = np


class CvDictToDict(DictToDict):
    module = cv2


class NpZerosLike(NpDictToDict):
    fn = "zeros_like"


class NpOnesLike(NpDictToDict):
    fn = "ones_like"


class NpFullLike(NpDictToDict):
    fn = "full_like"


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


def _fit_to_1(a):
    assert isinstance(a, np.ndarray)
    min = a.min()
    max = a.max()
    return (a - min) / (max - min)


@dict_io
def fit_to_1(a):
    return _fit_to_1(a)


def _fit_to_uint8(a):
    return (_fit_to_1(a) * 255).astype(np.uint8)


@dict_io
def fit_to_uint8(a):
    return _fit_to_uint8(a)


def _expand_repeat(a, repeats=1, axis=None):
    return np.repeat(np.expand_dims(a, axis=axis), repeats=repeats, axis=axis)


@dict_io
def expand_repeat(a, repeats=1, axis=None):
    return _expand_repeat(a, repeats=repeats, axis=axis)


def _sum_up(*imgs):
    imgs = [
        (_expand_repeat(img, repeats=3, axis=2) if img.ndim == 2 else img)
        for img in imgs
    ]
    imgs = [_fit_to_1(img) for img in imgs]
    out_img = np.sum(np.stack(imgs), axis=0)
    if out_img.shape[2] == 1:
        out_img = np.squeeze(out_img, axis=2)
    return out_img


@dict_io
def sum_up(*imgs):
    return _sum_up(*imgs)


def _mix_up(*imgs):
    mean_img = _sum_up(*imgs) / len(imgs)
    return (mean_img * 255).astype(np.uint8)


@dict_io
def mix_up(*imgs):
    return _mix_up(*imgs)


def _overlay(*imgs):
    clipped_img = np.clip(_sum_up(*imgs), 0, 1)
    return (clipped_img * 255).astype(np.uint8)


@dict_io
def overlay(*imgs):
    return _overlay(*imgs)


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


class CvSobel(CvDictToDict):
    fn = "Sobel"

    def __init__(self, ddepth="CV_64F", **kwargs):
        if isinstance(ddepth, str):
            kwargs["ddepth"] = getattr(cv2, ddepth)
        else:
            kwargs["ddepth"] = ddepth
        super().__init__(**kwargs)


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


class CvThreshold(CvDictToDict):
    fn = "threshold"

    def __init__(self, type="THRESH_BINARY", **kwargs):
        if isinstance(type, str):
            kwargs["type"] = getattr(cv2, type)
        else:
            kwargs["type"] = type
        super().__init__(**kwargs)

    def __call__(self, *args, **kwargs):
        out = super().__call__(*args, **kwargs)
        return out


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
