import numpy as np

_to_channel_last_dict = {3: (-2, -1, -3), 4: (0, -2, -1, -3)}


def to_channel_last_arr(a):
    if a.ndim in {3, 4}:
        return np.transpose(a, axes=_to_channel_last_dict.get(a.ndim))
    else:
        return a


_to_channel_first_dict = {3: (-1, -3, -2), 4: (0, -1, -3, -2)}


def to_channel_first_arr(a):
    if a.ndim in {3, 4}:
        return np.transpose(a, axes=_to_channel_first_dict.get(a.ndim))
    else:
        return a
