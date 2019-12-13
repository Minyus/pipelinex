import cv2


class CvBaseMethod:
    method = None

    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs

    def __call__(self, img):
        args = self.args
        kwargs = self.kwargs

        if self.method is None:
            img = img
        else:
            if isinstance(img, list):
                return [getattr(cv2, self.method)(e, *args, **kwargs) for e in img]
            if isinstance(img, dict):
                return {
                    k: getattr(cv2, self.method)(e, *args, **kwargs)
                    for k, e in img.items()
                }
            else:
                return getattr(cv2, self.method)(img, *args, **kwargs)


class CvScale(CvBaseMethod):
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


class CvResize(CvBaseMethod):
    method = "resize"


class CvCvtColor(CvBaseMethod):
    method = "cvtColor"


class CvDilate(CvBaseMethod):
    method = "dilate"


class CvErode(CvBaseMethod):
    method = "erode"


class CvFilter2d(CvBaseMethod):
    method = "filter2D"


class CvBlur(CvBaseMethod):
    method = "blur"


class CvBoxFilter(CvBaseMethod):
    method = "boxFilter"


class CvGaussianBlur(CvBaseMethod):
    method = "GaussianBlur"


class CvMedianBlur(CvBaseMethod):
    method = "medianBlur"


class CvBilateralFilter(CvBaseMethod):
    method = "bilateralFilter"


class CvAdaptiveBilateralFilter(CvBaseMethod):
    method = "adaptiveBilateralFilter"


class CvCanny(CvBaseMethod):
    method = "Canny"


class CvHoughLinesP(CvBaseMethod):
    method = "HoughLinesP"


class CvLine(CvBaseMethod):
    method = "line"


class CvEqualizeHist(CvBaseMethod):
    method = "equalizeHist"


class CvBGR2Gray(CvBaseMethod):
    method = "cvtColor"

    def __init__(self, *args, **kwargs):
        self.args = args
        kwargs["code"] = cv2.COLOR_BGR2GRAY
        self.kwargs = kwargs


class CvBGR2HSV(CvBaseMethod):
    method = "cvtColor"

    def __init__(self, *args, **kwargs):
        self.args = args
        kwargs["code"] = cv2.COLOR_BGR2HSV
        self.kwargs = kwargs
