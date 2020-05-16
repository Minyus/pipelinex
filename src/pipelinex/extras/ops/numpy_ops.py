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


def reverse_channel(a, channel_first=False):
    if a.ndim == 3:
        if channel_first:
            return a[::-1, :, :]
        else:
            return a[:, :, ::-1]
    if a.ndim == 4:
        if channel_first:
            return a[:, ::-1, :, :]
        else:
            return a[:, :, :, ::-1]
    return a


class ReverseChannel:
    def __init__(self, channel_first=False):
        self.channel_first = channel_first

    def __call__(self, a):
        return reverse_channel(a, channel_first=self.channel_first)
