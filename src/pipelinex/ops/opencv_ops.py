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
            img = getattr(cv2, self.method)(img, *args, **kwargs)
        return img


class CvResize(CvBaseMethod):
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


class CvCvtColor(CvBaseMethod):
    method = "cvtColor"


class CvGaussianBlur(CvBaseMethod):
    method = "GaussianBlur"


class CvCanny(CvBaseMethod):
    method = "Canny"


class CvHoughLinesP(CvBaseMethod):
    method = "HoughLinesP"


class CvLine(CvBaseMethod):
    method = "line"


class CvBGR2Gray:
    def __call__(self, img):
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
